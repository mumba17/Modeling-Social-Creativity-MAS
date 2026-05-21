import torch
import numpy as np
import torchvision
import os
from typing import List, Dict
from framework import Scheduler, Agent, Artifact, ArtifactGenerator, Logger
import genart
from features import FeatureExtractor
from knn import kNN
from wundtcurve import WundtCurve
import random
from collections import deque
from contextlib import nullcontext
from timing_utils import time_it
from torch.nn.parallel import scatter, replicate, parallel_apply, gather

IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)

class _FunctionWrapper(torch.nn.Module):
    """
    Wraps a stateless utility function inside a PyTorch nn.Module container.
    
    This encapsulation allows functional execution blocks (such as batched kNN computations)
    to interface natively with PyTorch's internal multi-GPU parallelization infrastructure.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class StatsTracker:
    """
    Tracks and maintains rolling statistical distributions of agent behaviors.
    """
    def __init__(self, window_size=10000, threshold_window_size=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Circular buffers for metrics
        self.novelty_max_size = window_size
        self.novelty_buffer = torch.zeros(window_size, device=self.device, dtype=torch.float32)
        self.novelty_ptr = 0
        self.novelty_size = 0
        
        self.self_interest_window = deque(maxlen=threshold_window_size)
        self.other_interest_window = deque(maxlen=threshold_window_size)
        self.cumulative_interest_window = deque(maxlen=threshold_window_size)
        
        self.p1 = 0.0
        self.p99 = 1.0
        self.self_thresh = 0.1      
        self.domain_thresh = 0.1    
        self.boredom_thresh = 0.2   

    def record_self_interest(self, interest_value: float):
        """Appends a finite self-evaluation score to the rolling queue."""
        if np.isfinite(interest_value):
            self.self_interest_window.append(interest_value)
    
    def record_other_interest(self, interest_value: float):
        """Appends a finite social-evaluation score to the rolling queue."""
        if np.isfinite(interest_value):
            self.other_interest_window.append(interest_value)

    def update_novelty_stats(self, new_novelty_tensor: torch.Tensor, step_count: int, recalc_interval=3):
        """
        Updates the global 1st and 99th percentile bounds using raw novelty scores.
        """
        valid_mask = torch.isfinite(new_novelty_tensor)
        valid_values = new_novelty_tensor[valid_mask].to(self.device, dtype=torch.float32).flatten()
        num_new = valid_values.numel()
        
        if num_new == 0:
            return
            
        if num_new > self.novelty_max_size:
            valid_values = valid_values[-self.novelty_max_size:]
            num_new = self.novelty_max_size
            
        end_idx = self.novelty_ptr + num_new
        if end_idx <= self.novelty_max_size:
            self.novelty_buffer[self.novelty_ptr:end_idx] = valid_values
        else:
            overflow = end_idx - self.novelty_max_size
            self.novelty_buffer[self.novelty_ptr:] = valid_values[:-overflow]
            self.novelty_buffer[:overflow] = valid_values[-overflow:]
            
        self.novelty_ptr = (self.novelty_ptr + num_new) % self.novelty_max_size
        self.novelty_size = min(self.novelty_size + num_new, self.novelty_max_size)
        
        if step_count >= 5 and step_count % recalc_interval == 0 and self.novelty_size > 100:
            active_buffer = self.novelty_buffer[:self.novelty_size]
            
            quantiles = torch.quantile(active_buffer, torch.tensor([0.01, 0.99], device=self.device))
            
            self.p1 = quantiles[0].item()
            self.p99 = quantiles[1].item()
            
            if self.p99 == self.p1:
                self.p99 += 1e-6

    def get_normalized_novelty(self, raw_score: float) -> float:
        """
        Maps a raw continuous scalar distance into a closed [0, 1] range using 
        the tracked historical percentile bounds.
        """
        if not np.isfinite(raw_score):
            return np.nan
        numerator = raw_score - self.p1
        denominator = self.p99 - self.p1
        if not np.isfinite(denominator) or denominator == 0:
            return np.nan
        return np.clip(numerator / denominator, 0.0, 1.0)

    def update_thresholds(self, all_agents: List[Agent]):
        """
        Computes system-wide performance filters based on active agent distributions.
        """
        if len(self.self_interest_window) > 10:
            self.self_thresh = np.percentile(list(self.self_interest_window), 80)
        if len(self.other_interest_window) > 10:
            self.domain_thresh = np.percentile(list(self.other_interest_window), 80)
        
        cumulative_interests = [a.average_interest for a in all_agents if np.isfinite(a.average_interest)]
        if cumulative_interests:
            self.cumulative_interest_window.extend(cumulative_interests)
            self.boredom_thresh = np.percentile(list(self.cumulative_interest_window), 10)
        self.boredom_thresh = max(self.boredom_thresh, -0.99999)

class ParallelScheduler(Scheduler):
    """
    A batch-vectorized orchestration pipeline for multi-agent simulations.
    
    Consolidates agent generation, rendering, visual feature extraction, and historical 
    memory cross-evaluation into highly batched GPU workloads.
    """
    def __init__(self, num_agents: int, artifact_generator: ArtifactGenerator, logger: Logger,
                 share_count: int = 5, uniform_novelty_pref: bool = False,
                 use_static_noise: bool = False, pca_dims: int = 128,
                 pca_calibration_samples: int = 500, distance_metric: str = 'cosine',
                 boredom_mode: str = 'classic', strict_integrity_mode: bool = True,
                 save_images: bool = False, image_output_dir: str = None,
                 use_personal_threshold: bool = False):
        self.num_agents = num_agents
        self.artifact_generator = artifact_generator
        self.logger = logger
        self.step_count = 0
        self.share_count = share_count
        self.uniform_novelty_pref = uniform_novelty_pref
        self.distance_metric = distance_metric
        self.boredom_mode = boredom_mode
        self.save_images = save_images
        self.image_output_dir = image_output_dir
        self.use_personal_threshold = use_personal_threshold

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
        self.use_amp = True
        self.use_static_noise = use_static_noise
        self.strict_integrity_mode = strict_integrity_mode

        self.multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.device_ids = list(range(torch.cuda.device_count())) if self.multi_gpu else None

        if self.save_images and self.image_output_dir:
            os.makedirs(self.image_output_dir, exist_ok=True)

        self.image_generator = genart.VectorizedImageGenerator(32, 32, device=self.device, use_static_noise=self.use_static_noise)
        self.imagenet_mean = torch.tensor(IMAGENET_MEAN_RGB, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor(IMAGENET_STD_RGB, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        
        _output_dims = pca_dims if pca_dims and pca_dims > 0 else None
        _feature_extractor = FeatureExtractor(
            output_dims=_output_dims, use_amp=self.use_amp,
            image_generator=self.image_generator if _output_dims else None, n_calibration=pca_calibration_samples
        )
        self.feature_extractor = torch.nn.DataParallel(_feature_extractor, device_ids=self.device_ids) if self.multi_gpu else _feature_extractor.to(self.device)
        
        self.feature_dim = _feature_extractor.output_dims 
        self.max_memory_size = 2500
        
        global_dtype = torch.float32
        
        self.global_memory_buffer = torch.zeros(
            (self.num_agents, self.max_memory_size, self.feature_dim),
            device=self.device,
            dtype=global_dtype
        )
        self.agents: List[Agent] = self._initialize_agents()
        self.domain = deque(maxlen=250_000)
        self.stats = StatsTracker()
        
        self.agent_reward_means = torch.tensor([a.wundt.reward_mean for a in self.agents], device=self.device, dtype=torch.float32)
        self.agent_reward_stds = torch.tensor([a.wundt.reward_std for a in self.agents], device=self.device, dtype=torch.float32)
        self.agent_punish_means = torch.tensor([a.wundt.punish_mean for a in self.agents], device=self.device, dtype=torch.float32)
        self.agent_punish_stds = torch.tensor([a.wundt.punish_std for a in self.agents], device=self.device, dtype=torch.float32)
        self.agent_alphas = torch.tensor([a.wundt.alpha for a in self.agents], device=self.device, dtype=torch.float32)

        self.self_threshold = self.stats.self_thresh
        self.domain_threshold = self.stats.domain_thresh
        self.boredom_threshold = self.stats.boredom_thresh

    @time_it
    def _sanitize_tensor(self, tensor: torch.Tensor, source: str) -> torch.Tensor:
        """Handles numerical singularities in tensors asynchronously."""
        if tensor is None: return tensor
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    @time_it
    def _sanitize_scalar(self, value: float, fallback: float) -> float:
        """Sanitizes scalar float values."""
        return float(value) if np.isfinite(value) else (float(fallback) if np.isfinite(fallback) else 0.0)

    @time_it
    def _initialize_agents(self) -> List[Agent]:
        """Instantiates simulation agents with specific preference curves and tracking buffers."""
        agents = []
        for i in range(self.num_agents):
            preferred_novelty = 0.5 if self.uniform_novelty_pref else np.clip(np.random.normal(0.5, 0.155), 0, 1)
            wundt = WundtCurve(
                reward_mean=max(0.1, preferred_novelty - 0.2), reward_std=0.15,
                punish_mean=min(0.9, preferred_novelty + 0.2), punish_std=0.15, alpha=1.2
            )
            agent = Agent(
                unique_id=i, knn=kNN(agent_id=i, max_size=self.max_memory_size), wundt=wundt,
                gen_depth=np.random.randint(4, 6), preferred_novelty=preferred_novelty
            )
            
            # Inject a slice of the global matrix directly into the agent's kNN.
            agent.knn.memory_buffer = self.global_memory_buffer[i]
            agent.knn.dtype = self.global_memory_buffer.dtype
            agent.knn._empty_feature_vectors = torch.empty((0, self.feature_dim), device=self.device, dtype=self.global_memory_buffer.dtype)

            agent.num_self_evals, agent.num_other_evals, agent.num_shares, agent.num_domain_adoptions = 0, 0, 0, 0
            agent.total_novelty_generated, agent.total_interest_generated = 0.0, 0.0
            agent.self_interest_window = deque(maxlen=100)
            agent.self_threshold = 0.1
            
            agent.recent_expr_strs = deque(maxlen=5)
            
            agents.append(agent)
            
            self.logger.log_event('agent_init', {
                'agent_id': agent.unique_id, 'preferred_novelty': agent.preferred_novelty,
                'reward_mean': agent.wundt.reward_mean, 'punishment_mean': agent.wundt.punish_mean
            })
        return agents

    @time_it
    def _parallel_apply_custom(self, function, *args):
        """
        Distributes tensors and metadata across multiple available GPU devices.
        """
        primary_scatter_tensor = args[0]
        scattered_primary = scatter(primary_scatter_tensor, self.device_ids)
        active_count = len(scattered_primary)

        if active_count == 0: return function(*args)
        scattered_args = [[scattered_primary[i]] for i in range(active_count)]

        for arg in args[1:]:
            if isinstance(arg, torch.Tensor) and arg.ndim > 0 and arg.shape[0] == primary_scatter_tensor.shape[0]:
                scattered_tensor = scatter(arg, self.device_ids[:active_count])
                for i in range(active_count): scattered_args[i].append(scattered_tensor[i])
            else:
                for i in range(active_count):
                    scattered_args[i].append(arg.to(scattered_primary[i].device) if isinstance(arg, torch.Tensor) else arg)
        
        module_replicas = replicate(_FunctionWrapper(function), self.device_ids[:active_count])
        outputs = parallel_apply(module_replicas, [tuple(arg_list) for arg_list in scattered_args], devices=self.device_ids[:active_count])
        return gather(outputs, self.device)

    @time_it
    def _gpu_stream_context(self):
        """Provides an execution context for a non-blocking dedicated CUDA background stream."""
        return torch.cuda.stream(self.gpu_stream) if (self.device.type == 'cuda' and self.gpu_stream is not None) else nullcontext()

    @time_it
    def _prepare_knn_batch_state(self, agents: List[Agent], feature_dim: int, device: torch.device):
        """
        Prepares memory views and state tracking tensors for batched operations.
        """
        num_agents = len(agents)
        if num_agents == 0:
             return torch.empty(0, 0, feature_dim, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

            # Return a view mapping active agents directly, matching query length N.
            # This keeps the expected global_buffer size N to match queries.
        agent_ids = torch.tensor([a.unique_id for a in agents], dtype=torch.long, device=device)
        current_sizes = torch.tensor([a.knn.current_size for a in agents], dtype=torch.long, device=device)
        agent_ks = torch.tensor([a.knn.k for a in agents], dtype=torch.long, device=device)

        return self.global_memory_buffer[agent_ids], current_sizes, agent_ks

    @time_it
    def refresh_current_interest_phase(self):
        """Recalculates baseline interest scores for active agent configurations."""
        active = [a for a in self.agents if a.current_features is not None]
        if not active: return

        with self._gpu_stream_context():
            query_batch = self._sanitize_tensor(torch.stack([a.current_features for a in active]), 'refresh.query_batch')
            global_buffer, current_sizes, agent_ks = self._prepare_knn_batch_state(active, query_batch.shape[1], self.device)
            global_buffer = self._sanitize_tensor(global_buffer, 'refresh.global_buffer')
            novelty_scores_tensor = self._sanitize_tensor(kNN.batch_evaluate_novelty_for_agents(query_batch, global_buffer, current_sizes, agent_ks), 'refresh.novelty_scores')

            p1, p99 = self.stats.p1, self.stats.p99
            denom = p99 - p1 if (p99 - p1) != 0 else 1e-6
            normalized_novelty_tensor = torch.clamp((novelty_scores_tensor - p1) / denom, 0.0, 1.0)
            
            active_ids = torch.tensor([a.unique_id for a in active], device=self.device, dtype=torch.long)
            interest_scores_tensor = WundtCurve.batch_hedonic_value(
                normalized_novelty_tensor,
                self.agent_reward_means[active_ids], self.agent_reward_stds[active_ids],
                self.agent_punish_means[active_ids], self.agent_punish_stds[active_ids],
                self.agent_alphas[active_ids]
            )

        if self.device.type == 'cuda': torch.cuda.synchronize()
        novelty_scores = normalized_novelty_tensor.cpu().numpy()
        interest_scores = interest_scores_tensor.cpu().numpy()

        for i, agent in enumerate(active):
            agent.current_interest = float(interest_scores[i])
            agent.current_novelty = float(novelty_scores[i])

    @time_it
    def step(self):
        """Executes a full simulation cycle across all agents."""
        self.refresh_current_interest_phase()
        generated_artifacts = self.generation_phase()
        evaluated_artifacts = self.evaluation_phase(generated_artifacts)
        self.individual_evaluation_phase(evaluated_artifacts)
        messages = self.sharing_phase(evaluated_artifacts)
        interaction_results = self.interaction_phase(messages)
        if self.step_count != 0: self.boredom_phase()
        self.update_system_thresholds()
        self._log_step_metrics(interaction_results)
        self.step_count += 1

    @time_it
    def generation_phase(self) -> List[Artifact]:
        """Generates raw structural expressions for all agents."""
        return self.artifact_generator.generate(self.agents)

    @time_it
    def evaluation_phase(self, artifacts: List[Artifact]) -> List[Artifact]:
        """
        Renders expressions into images and extracts high-dimensional visual feature vectors.
        """
        if not artifacts: return []
        image_tensor_batch = self._sanitize_tensor(self.image_generator.generate_batch([a.content for a in artifacts], use_amp=self.use_amp), 'evaluation.image_generator.output')
        if image_tensor_batch.shape[0] == 0: return []

        if self.save_images and self.image_output_dir:
            image_batch_cpu = image_tensor_batch.detach().to('cpu')
            for i, artifact in enumerate(artifacts):
                torchvision.utils.save_image(image_batch_cpu[i], os.path.join(self.image_output_dir, f"{artifact.id}.png"))
        
        normalized_batch = self._sanitize_tensor((image_tensor_batch - self.imagenet_mean) / self.imagenet_std, 'evaluation.normalized_batch')
        with self._gpu_stream_context(), torch.no_grad():
            features_batch = self._sanitize_tensor(self.feature_extractor(normalized_batch).detach(), 'evaluation.features_batch')
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        for i, artifact in enumerate(artifacts):
            artifact.features = features_batch[i].clone()
        return artifacts

    @time_it
    def sharing_phase(self, evaluated_artifacts: List[Artifact]) -> List[Dict]:
        """Identifies highly interesting expressions and generates messages for randomly selected peers."""
        messages = []
        agent_to_artifact = {art.producer_id: art for art in evaluated_artifacts}
        
        for agent in self.agents:
            just_generated = agent_to_artifact.get(agent.unique_id)
            threshold = agent.self_threshold if self.use_personal_threshold else self.self_threshold
            if just_generated and just_generated.interest > threshold:
                agent.num_shares += 1
                num_recipients = min(self.share_count, self.num_agents - 1)
                if num_recipients <= 0: continue
                recipients = random.sample([a.unique_id for a in self.agents if a.unique_id != agent.unique_id], k=num_recipients)
                for recipient_id in recipients:
                    messages.append({'artifact': just_generated, 'sender_id': agent.unique_id, 'recipient_id': recipient_id})
        return messages

    @time_it
    def interaction_phase(self, messages: List[Dict]) -> List[Dict]:
        """Processes peer sharing networks and evaluates artifacts against recipient histories."""
        if not messages: return []

        with self._gpu_stream_context():
            query_batch = self._sanitize_tensor(torch.stack([msg['artifact'].features for msg in messages]), 'interaction.query_batch')
            global_buffer, current_sizes, agent_ks = self._prepare_knn_batch_state(self.agents, query_batch.shape[1], self.device)
            global_buffer = self._sanitize_tensor(global_buffer, 'interaction.global_buffer')
            message_to_agent_map = torch.tensor([msg['recipient_id'] for msg in messages], device=self.device, dtype=torch.long)

            if self.multi_gpu and query_batch.shape[0] >= len(self.device_ids):
                novelty_scores_tensor = self._parallel_apply_custom(kNN.batch_evaluate_novelty_for_messages, query_batch, message_to_agent_map, global_buffer, current_sizes, agent_ks)
            else:
                novelty_scores_tensor = kNN.batch_evaluate_novelty_for_messages(query_batch, message_to_agent_map, global_buffer, current_sizes, agent_ks)
            novelty_scores_tensor = self._sanitize_tensor(novelty_scores_tensor, 'interaction.novelty_scores_tensor.after_knn')

            p1, p99 = self.stats.p1, self.stats.p99
            denom = p99 - p1 if (p99 - p1) != 0 else 1e-6
            normalized_novelty_tensor = torch.clamp((novelty_scores_tensor - p1) / denom, 0.0, 1.0)
            
            batch_r_means = self.agent_reward_means[message_to_agent_map]
            batch_r_stds = self.agent_reward_stds[message_to_agent_map]
            batch_p_means = self.agent_punish_means[message_to_agent_map]
            batch_p_stds = self.agent_punish_stds[message_to_agent_map]
            batch_alphas = self.agent_alphas[message_to_agent_map]
            
            interest_scores_tensor = WundtCurve.batch_hedonic_value(
                normalized_novelty_tensor,
                batch_r_means, batch_r_stds,
                batch_p_means, batch_p_stds,
                batch_alphas
            )

        if self.device.type == 'cuda': torch.cuda.synchronize()
        novelty_scores = normalized_novelty_tensor.cpu().numpy()
        interest_scores = interest_scores_tensor.cpu().numpy()
        interaction_results = []
        
        interaction_agent_ids = []
        interaction_ptrs = []
        interaction_features = []
        staged_share_logs = []
        
        for i, message in enumerate(messages):
            recipient = self.agents[message['recipient_id']]
            artifact = message['artifact']
            normalized_novelty = float(novelty_scores[i])
            interest = float(interest_scores[i])
            
            self.stats.record_other_interest(interest)
            recipient.num_other_evals += 1
            accepted = interest > self.domain_threshold
            
            if accepted:
                self.domain.append(artifact)
                recipient.num_domain_adoptions += 1

            adopted_received = interest > recipient.current_interest
            if adopted_received:
                recipient.current_expression = artifact.content._copy()
                recipient.current_interest = interest
                recipient.current_features = artifact.features.clone()
                recipient.current_artifact_id = artifact.id
                recipient.current_creator_id = artifact.creator_id
                recipient.current_expr_str = artifact.expr_str

            artifact_expr_str = artifact.expr_str
            if artifact_expr_str not in recipient.recent_expr_strs:
                interaction_agent_ids.append(recipient.unique_id)
                interaction_ptrs.append(recipient.knn.ptr)
                interaction_features.append(artifact.features)
                
                recipient.artifact_memory.append({
                    'id': artifact.id, 'expression': artifact.content, 'expr_str': artifact_expr_str, 'creator_id': artifact.creator_id
                })
                recipient.recent_expr_strs.append(artifact_expr_str)
                
                recipient.knn.ptr = (recipient.knn.ptr + 1) % self.max_memory_size
                recipient.knn.current_size = min(recipient.knn.current_size + 1, self.max_memory_size)
            
            interaction_results.append({'accepted': accepted, 'interest': interest, 'novelty': normalized_novelty})
            
            staged_share_logs.append({
                'agent_id': message['sender_id'], 'step': self.step_count, 'sender_id': message['sender_id'], 'recipient_id': recipient.unique_id,
                'artifact_id': artifact.id, 'expression': artifact_expr_str, 'evaluated_novelty': normalized_novelty, 'evaluated_interest': interest,
                'accepted': accepted, 'adopted': adopted_received, 'creator_id': artifact.creator_id, 'evaluator_id': recipient.unique_id, 'domain_size': len(self.domain)
            })
            
        if interaction_agent_ids:
            with torch.no_grad():
                stacked_inter_features = torch.stack(interaction_features)
                normalized_inter_features = torch.nn.functional.normalize(stacked_inter_features, p=2, dim=1)
                
                gpu_inter_agent_ids = torch.tensor(interaction_agent_ids, dtype=torch.long, device=self.device)
                gpu_inter_ptrs = torch.tensor(interaction_ptrs, dtype=torch.long, device=self.device)
                
                self.global_memory_buffer[gpu_inter_agent_ids, gpu_inter_ptrs, :] = normalized_inter_features
                
        if staged_share_logs:
            self.logger.log_events_batch('share', staged_share_logs)
                
        return interaction_results

    @time_it
    def individual_evaluation_phase(self, evaluated_artifacts: List[Artifact]):
        """Evaluates newly generated artifacts and updates the producing agent's state."""
        if not evaluated_artifacts: return

        with self._gpu_stream_context():
            query_batch = self._sanitize_tensor(torch.stack([art.features for art in evaluated_artifacts]), 'individual.query_batch')
            global_buffer, current_sizes, agent_ks = self._prepare_knn_batch_state(self.agents, query_batch.shape[1], self.device)
            global_buffer = self._sanitize_tensor(global_buffer, 'individual.global_buffer')

            if self.multi_gpu and query_batch.shape[0] >= len(self.device_ids):
                novelty_scores_tensor = self._parallel_apply_custom(kNN.batch_evaluate_novelty_for_agents, query_batch, global_buffer, current_sizes, agent_ks)
            else:
                novelty_scores_tensor = kNN.batch_evaluate_novelty_for_agents(query_batch, global_buffer, current_sizes, agent_ks)
            
            novelty_scores_tensor = self._sanitize_tensor(novelty_scores_tensor, 'individual.novelty_scores_tensor.after_knn')
            self.stats.update_novelty_stats(novelty_scores_tensor, self.step_count)

            p1, p99 = self.stats.p1, self.stats.p99
            denom = p99 - p1 if (p99 - p1) != 0 else 1e-6
            normalized_novelty_tensor = torch.clamp((novelty_scores_tensor - p1) / denom, 0.0, 1.0)
            
            producer_ids = torch.tensor([art.producer_id for art in evaluated_artifacts], device=self.device, dtype=torch.long)
            interest_scores_tensor = WundtCurve.batch_hedonic_value(
                normalized_novelty_tensor,
                self.agent_reward_means[producer_ids], self.agent_reward_stds[producer_ids],
                self.agent_punish_means[producer_ids], self.agent_punish_stds[producer_ids],
                self.agent_alphas[producer_ids]
            )

        if self.device.type == 'cuda': torch.cuda.synchronize()
        novelty_scores = normalized_novelty_tensor.cpu().numpy()
        interest_scores = interest_scores_tensor.cpu().numpy()

        individual_agent_ids = []
        individual_ptrs = []
        individual_features = []
        staged_gen_logs = []

        for i, artifact in enumerate(evaluated_artifacts):
            agent = self.agents[artifact.producer_id]
            normalized_novelty = float(novelty_scores[i])
            interest = float(interest_scores[i])

            self.stats.record_self_interest(interest)
            if np.isfinite(interest):
                agent.self_interest_window.append(interest)
                
            agent.update_hall_of_fame(artifact.content, interest, creator_id=artifact.creator_id)
            previous_interest = agent.current_interest
            agent.current_novelty = normalized_novelty

            agent.num_self_evals += 1
            agent.total_novelty_generated += normalized_novelty
            agent.total_interest_generated += interest
            
            artifact_expr_str = artifact.expr_str
            if artifact_expr_str not in agent.recent_expr_strs:
                individual_agent_ids.append(agent.unique_id)
                individual_ptrs.append(agent.knn.ptr)
                individual_features.append(query_batch[i])

                agent.artifact_memory.append({
                    'id': artifact.id, 'expression': artifact.content, 'expr_str': artifact_expr_str, 'creator_id': artifact.creator_id
                })
                agent.recent_expr_strs.append(artifact_expr_str)
                
                agent.knn.ptr = (agent.knn.ptr + 1) % self.max_memory_size
                agent.knn.current_size = min(agent.knn.current_size + 1, self.max_memory_size)

            artifact.novelty, artifact.interest = normalized_novelty, interest
            adopted = False
            if agent.current_expression is None or interest > previous_interest:
                agent.current_expression = artifact.content
                agent.current_features = artifact.features.clone()
                agent.current_interest = interest
                agent.current_artifact_id = artifact.id
                agent.current_creator_id = artifact.creator_id
                agent.current_expr_str = artifact_expr_str
                adopted = True

            agent.average_interest = agent.alpha * agent.average_interest + (1 - agent.alpha) * agent.current_interest
            
            staged_gen_logs.append({
                'step': self.step_count, 'agent_id': agent.unique_id, 'artifact_id': artifact.id, 'expression': artifact_expr_str,
                'novelty': normalized_novelty, 'interest': interest, 'adopted': adopted, 'parent1_id': artifact.parent1_id, 'parent2_id': artifact.parent2_id,
                'creator_id': artifact.creator_id, 'evaluator_id': agent.unique_id, 'domain_size': len(self.domain)
            })

        if individual_agent_ids:
            with torch.no_grad():
                stacked_features = torch.stack(individual_features)
                normalized_features = torch.nn.functional.normalize(stacked_features, p=2, dim=1)
                gpu_agent_ids = torch.tensor(individual_agent_ids, dtype=torch.long, device=self.device)
                gpu_ptrs = torch.tensor(individual_ptrs, dtype=torch.long, device=self.device)
                self.global_memory_buffer[gpu_agent_ids, gpu_ptrs, :] = normalized_features

        if staged_gen_logs:
            self.logger.log_events_batch('generation', staged_gen_logs)

    @time_it
    def boredom_phase(self):
        """Triggers exploratory actions for agents experiencing decreasing baseline interest levels."""
        classic_agents, chosen_artifacts = [], []
        for agent in self.agents:
            if agent.average_interest >= self.boredom_threshold: continue
            if not self.domain: continue
            classic_agents.append(agent)
            chosen_artifacts.append(random.choice(self.domain))

        if not classic_agents: return

        cached_features, uncached_indices, uncached_artifacts = [], [] ,[]
        for idx, art in enumerate(chosen_artifacts):
            if art.features is not None: cached_features.append((idx, art.features))
            else:
                uncached_indices.append(idx)
                uncached_artifacts.append(art)

        if cached_features: feature_dim = cached_features[0][1].shape[0]
        elif uncached_artifacts:
            _probe_batch = self.image_generator.generate_batch([uncached_artifacts[0].content], use_amp=self.use_amp)
            if _probe_batch.shape[0] == 0: return
            with self._gpu_stream_context(), torch.no_grad():
                _f = self.feature_extractor((_probe_batch - self.imagenet_mean) / self.imagenet_std).detach()
            if self.device.type == 'cuda': torch.cuda.synchronize()
            feature_dim = _f.shape[1]
            uncached_artifacts[0].features = _f[0].clone()
            cached_features.append((uncached_indices[0], _f[0].clone()))
            uncached_indices, uncached_artifacts = uncached_indices[1:], uncached_artifacts[1:]
        else: return

        uncached_feature_map = {}
        if uncached_artifacts:
            image_tensor_batch = self.image_generator.generate_batch([art.content for art in uncached_artifacts], use_amp=self.use_amp)
            if image_tensor_batch.shape[0] > 0:
                normalized_batch = (image_tensor_batch - self.imagenet_mean) / self.imagenet_std
                with self._gpu_stream_context(), torch.no_grad():
                    rendered_features = self.feature_extractor(normalized_batch).detach()
                if self.device.type == 'cuda': torch.cuda.synchronize()
                for j, (orig_idx, art) in enumerate(zip(uncached_indices, uncached_artifacts)):
                    art.features = rendered_features[j].clone()
                    uncached_feature_map[orig_idx] = rendered_features[j].clone()

        features_list = [None] * len(classic_agents)
        for orig_idx, feat in cached_features: features_list[orig_idx] = feat
        for orig_idx, feat in uncached_feature_map.items(): features_list[orig_idx] = feat

        valid = [(i, f) for i, f in enumerate(features_list) if f is not None]
        if not valid: return
        if len(valid) < len(classic_agents):
            keep_idx = [i for i, _ in valid]
            classic_agents, chosen_artifacts, features_list = [classic_agents[i] for i in keep_idx], [chosen_artifacts[i] for i in keep_idx], [f for _, f in valid]

        features_batch = torch.stack(features_list)
        consolidated_memories, memory_indices, agent_ks = self._prepare_knn_batch_state(classic_agents, features_batch.shape[1], self.device)
        novelty_scores = kNN.batch_evaluate_novelty_for_agents(features_batch, consolidated_memories, memory_indices, agent_ks)

        p1, p99 = self.stats.p1, self.stats.p99
        denom = p99 - p1 if (p99 - p1) != 0 else 1e-6
        normalized_novelty_tensor = torch.clamp((novelty_scores - p1) / denom, 0.0, 1.0)
        
        bored_agent_ids_tensor = torch.tensor([a.unique_id for a in classic_agents], device=self.device, dtype=torch.long)
        interest_scores_tensor = WundtCurve.batch_hedonic_value(
            normalized_novelty_tensor,
            self.agent_reward_means[bored_agent_ids_tensor], self.agent_reward_stds[bored_agent_ids_tensor],
            self.agent_punish_means[bored_agent_ids_tensor], self.agent_punish_stds[bored_agent_ids_tensor],
            self.agent_alphas[bored_agent_ids_tensor]
        )

        normalized_novelties = normalized_novelty_tensor.cpu().numpy()
        interest_scores = interest_scores_tensor.cpu().numpy()

        boredom_agent_ids = []
        boredom_ptrs = []
        boredom_features = []
        staged_boredom_logs = []

        for i, (agent, domain_artifact) in enumerate(zip(classic_agents, chosen_artifacts)):
            features = features_batch[i]
            normalized_novelty = float(normalized_novelties[i])
            interest = float(interest_scores[i])

            artifact_expr_str = domain_artifact.expr_str
            if artifact_expr_str not in [mem['expr_str'] for mem in list(agent.artifact_memory)[-5:]]:
                boredom_agent_ids.append(agent.unique_id)
                boredom_ptrs.append(agent.knn.ptr)
                boredom_features.append(features)

                agent.artifact_memory.append({
                    'id': domain_artifact.id, 'expression': domain_artifact.content, 'expr_str': artifact_expr_str, 'creator_id': domain_artifact.creator_id
                })
                agent.knn.ptr = (agent.knn.ptr + 1) % self.max_memory_size
                agent.knn.current_size = min(agent.knn.current_size + 1, self.max_memory_size)

            adopted = interest > agent.current_interest
            if adopted:
                agent.current_expression = domain_artifact.content._copy()
                agent.current_features, agent.current_interest, agent.current_artifact_id, agent.current_creator_id = features.clone(), interest, domain_artifact.id, domain_artifact.creator_id
                agent.current_expr_str = artifact_expr_str

            staged_boredom_logs.append({
                'step': self.step_count, 'agent_id': agent.unique_id, 'artifact_id': domain_artifact.id, 'expression': artifact_expr_str,
                'novelty': normalized_novelty, 'interest': interest, 'adopted': adopted, 'source': 'domain_classic', 'trigger_novelty': getattr(agent, 'current_novelty', 0.5),
                'creator_id': domain_artifact.creator_id, 'evaluator_id': agent.unique_id, 'domain_size': len(self.domain)
            })

        if boredom_agent_ids:
            with torch.no_grad():
                stacked_boredom_feats = torch.stack(boredom_features)
                normalized_boredom_feats = torch.nn.functional.normalize(stacked_boredom_feats, p=2, dim=1)
                gpu_boredom_agents = torch.tensor(boredom_agent_ids, dtype=torch.long, device=self.device)
                gpu_boredom_ptrs = torch.tensor(boredom_ptrs, dtype=torch.long, device=self.device)
                self.global_memory_buffer[gpu_boredom_agents, gpu_boredom_ptrs, :] = normalized_boredom_feats

        if staged_boredom_logs:
            self.logger.log_events_batch('boredom_adoption', staged_boredom_logs)

    @time_it
    def _normalize_novelty(self, raw_novelty: float) -> float:
        """Normalizes a raw novelty score using the current tracking stats."""
        return self.stats.get_normalized_novelty(raw_novelty)

    def update_system_thresholds(self):
        """Recalculates global filtering boundaries from rolling percentile distributions."""
        self.stats.update_thresholds(self.agents)
        self.boredom_threshold = self.stats.boredom_thresh
        self.self_threshold = self.stats.self_thresh
        self.domain_threshold = self.stats.domain_thresh
        
        if self.use_personal_threshold:
            for agent in self.agents:
                if len(agent.self_interest_window) > 10:
                    agent.self_threshold = np.percentile(list(agent.self_interest_window), 80)

    def _log_step_metrics(self, interaction_results: List[Dict]):
        """Collects step analytics across the agent population and logs the values."""
        avg_accepted_interest, avg_rejected_interest, accepted_count, rejected_count = 0, 0, 0, 0
        if interaction_results:
            accepted_interests = [r['interest'] for r in interaction_results if r['accepted']]
            rejected_interests = [r['interest'] for r in interaction_results if not r['accepted']]
            accepted_count, rejected_count = len(accepted_interests), len(rejected_interests)
            if accepted_interests: avg_accepted_interest = sum(accepted_interests) / accepted_count
            if rejected_interests: avg_rejected_interest = sum(rejected_interests) / rejected_count

        avg_knn_size = sum([a.knn.feature_vectors.shape[0] for a in self.agents]) / self.num_agents
        avg_current_interest = sum(a.current_interest for a in self.agents) / self.num_agents
        avg_average_interest = sum(a.average_interest for a in self.agents) / self.num_agents
        avg_current_novelty = sum(getattr(a, 'current_novelty', 0.0) for a in self.agents) / self.num_agents

        self.logger.log_event('step_end', {
            'step': self.step_count, 'domain_size': len(self.domain), 'self_threshold': self.self_threshold,
            'domain_threshold': self.domain_threshold, 'boredom_threshold': self.boredom_threshold,
            'avg_accepted_interest': avg_accepted_interest, 'avg_rejected_interest': avg_rejected_interest,
            'accepted_count': accepted_count, 'rejected_count': rejected_count, 'avg_knn_size': avg_knn_size,
            'avg_current_interest': avg_current_interest, 'avg_average_interest': avg_average_interest, 'avg_current_novelty': avg_current_novelty,
            'total_self_evals': sum(a.num_self_evals for a in self.agents), 'total_other_evals': sum(a.num_other_evals for a in self.agents),
            'total_shares': sum(a.num_shares for a in self.agents), 'total_domain_adoptions': sum(a.num_domain_adoptions for a in self.agents)
        })
        
        if self.step_count % 10 == 0:
            for agent in self.agents:
                    self.logger.log_event('agent_state', {
                    'step': self.step_count, 'agent_id': agent.unique_id, 'cumulative_interest': agent.average_interest,
                    'repository_size': agent.knn.feature_vectors.shape[0], 'k_value': agent.knn.k, 'boredom_triggered': False,
                    'num_self_evals': agent.num_self_evals, 'num_other_evals': agent.num_other_evals, 'num_shares': agent.num_shares,
                    'num_domain_adoptions': agent.num_domain_adoptions, 'avg_novelty_generated': agent.total_novelty_generated / max(1, agent.num_self_evals),
                    'avg_interest_generated': agent.total_interest_generated / max(1, agent.num_self_evals)
                })

    def close(self):
        """Clears active processing tracks and flushes all log pipelines."""
        if hasattr(self.logger, 'close'): self.logger.close()