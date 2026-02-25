"""
Simulation Scheduler (Algorithm 1)
===================================

This module contains the `ParallelScheduler`, which is the heart of the
simulation. It implements Algorithm 1 from the thesis:

    Per step, for all agents:
    1. generation_phase()           → Algorithm 1, step 1 (Generate)
    2. evaluation_phase()           → Image rendering + feature extraction
    3. individual_evaluation_phase()→ Algorithm 1, step 2 (Self-evaluate)
    4. sharing_phase()              → Algorithm 1, step 3 (Share decision)
    5. interaction_phase()          → Algorithm 1, step 4 (Receive & evaluate)
    6. boredom_phase()              → Algorithm 1, step 5 (Boredom check)
    7. update_system_thresholds()   → Algorithm 1, step 6 (Update thresholds)

DEVIATION(paper 3.5): All phases use batched GPU pipelines
rather than sequential per-agent processing. This is a pure
performance optimization that does not change algorithmic
semantics; all agents still see the same memory snapshot
within each phase.
"""

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
import itertools
from collections import deque
from contextlib import nullcontext
from timing_utils import time_it
from torch.nn.parallel import scatter, replicate, parallel_apply, gather

class _FunctionWrapper(torch.nn.Module):
    """
    A helper class to wrap a stateless function in an nn.Module.

    This allows stateless functions, such as those in the kNN class, to be
    used with PyTorch's `DataParallel` utilities, which expect an `nn.Module`.
    """
    def __init__(self, func):
        """
        Initializes the wrapper.

        Args:
            func (callable): The stateless function to wrap.
        """
        super().__init__()
        self.func = func
    def forward(self, *args, **kwargs):
        """Executes the wrapped function."""
        return self.func(*args, **kwargs)

class StatsTracker:
    """
    Tracks rolling statistics for novelty normalization (3.3.3)
    and dynamic threshold computation (3.4).

    Novelty normalization (3.3.3):
        P1/P99 percentile bounds from a rolling window of 10,000
        raw novelty scores. Recalculated every 3 steps, skipping
        the first 5 steps for initialization.

    Threshold formulas (3.4):
        self_thresh    = percentile(self_interest_window, 80)
        domain_thresh  = percentile(other_interest_window, 80)
        boredom_thresh = percentile(cumulative_interests, 10)

    DEVIATION(paper 3.3.3): Novelty stats are updated only from
    self-generated artifacts, not from received shared artifacts.
    This is a design choice: production events define the novelty
    landscape, consumption events do not.
    """
    def __init__(self, window_size=10000, threshold_window_size=100):
        # Novelty normalization window (3.3.3: 10,000 scores)
        self.novelty_window = deque(maxlen=window_size)
        
        # Threshold windows (3.4: rolling window size 100)
        self.self_interest_window = deque(maxlen=threshold_window_size)
        self.other_interest_window = deque(maxlen=threshold_window_size)
        self.cumulative_interest_window = deque(maxlen=threshold_window_size)
        
        # Paper default thresholds (3.4)
        self.p1 = 0.0
        self.p99 = 1.0
        self.self_thresh = 0.1      # τ_C initial value
        self.domain_thresh = 0.1    # τ_D initial value
        self.boredom_thresh = 0.2   # τ_B initial value

    def record_self_interest(self, interest_value):
        """Record interest from an agent evaluating their own artifact."""
        self.self_interest_window.append(interest_value)
    
    def record_other_interest(self, interest_value):
        """Record interest from an agent evaluating a received artifact."""
        self.other_interest_window.append(interest_value)

    def update_novelty_stats(self, new_novelty_tensor, step_count, recalc_interval=3):
        """
        Ingests raw novelty scores and updates P1/P99 bounds (3.3.3).

        Recalculation schedule: every 3 steps, skip first 5.
        """
        novelty_list = new_novelty_tensor.detach().cpu().numpy().flatten().tolist()
        self.novelty_window.extend(novelty_list)
        
        # 3.3.3: recalculate bounds every recalc_interval steps,
        # skip first 5 steps to accumulate a meaningful baseline.
        if step_count >= 5 and step_count % recalc_interval == 0 and len(self.novelty_window) > 100:
            novelty_array = np.array(self.novelty_window)
            self.p1 = np.percentile(novelty_array, 1)
            self.p99 = np.percentile(novelty_array, 99)
            
            if self.p99 == self.p1:
                self.p99 += 1e-6

    def get_normalized_novelty(self, raw_score):
        """
        P1/P99 percentile normalization (3.3.3).
        Maps raw kNN distance onto [0,1] for Wundt curve input.
        Formula: clip((x - P1) / (P99 - P1), 0, 1)
        """
        numerator = raw_score - self.p1
        denominator = self.p99 - self.p1
        normalized = numerator / denominator
        return np.clip(normalized, 0.0, 1.0)

    def update_thresholds(self, all_agents):
        """
        Dynamic threshold update (3.4, Algorithm 1 step 6).

        Paper formulas:
          τ_C = percentile(self_interest_window, 80)
          τ_D = percentile(other_interest_window, 80)
          τ_B = percentile(cumulative_interests_all_agents, 10)
        """
        # Self threshold (τ_C): share if interest exceeds this
        if len(self.self_interest_window) > 10:
            self.self_thresh = np.percentile(list(self.self_interest_window), 80)
        
        # Domain threshold (τ_D): accept shared artifact if above
        if len(self.other_interest_window) > 10:
            self.domain_thresh = np.percentile(list(self.other_interest_window), 80)
        
        # Boredom threshold (τ_B): trigger boredom if S_i below
        cumulative_interests = [a.average_interest for a in all_agents]
        if cumulative_interests:
            self.cumulative_interest_window.extend(cumulative_interests)
            self.boredom_thresh = np.percentile(list(self.cumulative_interest_window), 10)
        
        # Safety floor to prevent division-by-zero edge cases
        self.boredom_thresh = max(self.boredom_thresh, 0.001)


class ParallelScheduler(Scheduler):
    """
    Orchestrates the simulation using parallel, batched steps.
    """
    def __init__(self, num_agents: int, artifact_generator: ArtifactGenerator, logger: Logger,
                 share_count: int = 5, uniform_novelty_pref: bool = False,
                 use_static_noise: bool = False, feature_dims: int = 0,
                 pca_calibration_samples: int = 500, distance_metric: str = 'cosine',
                 boredom_mode: str = 'extended', adopt_shared_expression: bool = False,
                 save_images: bool = False,
                 image_output_dir: str = None):
        """
        Initializes the ParallelScheduler.

        Key parameter notes:
          share_count: N in Algorithm 1 step 3 (paper: configurable)
          distance_metric: 'cosine' (default) or 'euclidean'
          boredom_mode: 'classic' (paper) or 'extended' (deviation)
        """
        self.num_agents = num_agents
        self.artifact_generator = artifact_generator
        self.logger = logger
        self.step_count = 0

        self.share_count = share_count
        self.uniform_novelty_pref = uniform_novelty_pref
        self.distance_metric = distance_metric
        self.boredom_mode = boredom_mode
        self.adopt_shared_expression = adopt_shared_expression
        self.save_images = save_images
        self.image_output_dir = image_output_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
        
        self.use_amp = True
        self.use_static_noise = use_static_noise

        self.multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.device_ids = list(range(torch.cuda.device_count())) if self.multi_gpu else None

        if self.multi_gpu:
            print(f"Activating multi-GPU mode on {len(self.device_ids)} devices: {self.device_ids}")
        else:
            print(f"Running on single device: {self.device}")

        if self.save_images and self.image_output_dir:
            os.makedirs(self.image_output_dir, exist_ok=True)

        self.image_generator = genart.VectorizedImageGenerator(32, 32, device=self.device, use_static_noise=self.use_static_noise)
        
        # Feature extractor setup (3.3.2)
        # feature_dims=0 → raw 128d; >0 → PCA reduction
        _output_dims = feature_dims if feature_dims and feature_dims > 0 else None
        _feature_extractor = FeatureExtractor(
            output_dims=_output_dims,
            use_amp=self.use_amp,
            image_generator=self.image_generator if _output_dims else None,
            n_calibration=pca_calibration_samples
        )
        if self.multi_gpu:
            self.feature_extractor = torch.nn.DataParallel(_feature_extractor, device_ids=self.device_ids)
        else:
            self.feature_extractor = _feature_extractor.to(self.device)

        self.agents: List[Agent] = self._initialize_agents()

        # Domain: shared artifact repository (2.2 DIFI model)
        # Paper: no curation, artifacts remain indefinitely.
        self.domain: List[Artifact] = []
        self.max_domain_size = 1000000000

        # Initialize Stats Tracker
        self.stats = StatsTracker()

        # Initialize thresholds using the tracker's defaults
        self.self_threshold = self.stats.self_thresh
        self.domain_threshold = self.stats.domain_thresh
        self.boredom_threshold = self.stats.boredom_thresh

    @time_it
    def _initialize_agents(self) -> List[Agent]:
        """
        Creates and initializes agents with paper-specified
        parameters (3.3.4, 3.4).
        """
        agents = []
        for i in range(self.num_agents):
            # Preferred novelty ~ N(0.5, 0.155) clipped [0,1] (3.3.4)
            if self.uniform_novelty_pref:
                preferred_novelty = 0.5
            else:
                preferred_novelty = np.clip(np.random.normal(0.5, 0.155), 0, 1)

            # Wundt curve params (3.3.4): all match paper
            # reward_mean = max(0.1, p-0.2), punish_mean = min(0.9, p+0.2)
            # σ_r = σ_p = 0.15, α = 1.2
            wundt = WundtCurve(
                reward_mean=max(0.1, preferred_novelty - 0.2),
                reward_std=0.15,
                punish_mean=min(0.9, preferred_novelty + 0.2),
                punish_std=0.15,
                alpha=1.2
            )
            # DEVIATION(paper 3.2): gen_depth 4-6 vs paper 6-10.
            # Testing has shown that at depth 4-6 we already get a rich variety of artifacts while keeping GPU rendering times manageable. 
            # Depths above 6 lead to significantly longer render times and larger expression trees (which ramps up the feature extractor workload), 
            # which can bottleneck the simulation without providing proportional benefits in artifact diversity for our purposes.
            # Paper: Random tree depth sampled from [6, 10].
            # Code: Sampled from [4, 6) for faster rendering and
            #       smaller expression trees.
            agent = Agent(
                unique_id=i,
                knn=kNN(agent_id=i, max_size=1000),
                wundt=wundt,
                gen_depth=np.random.randint(4, 6),
                preferred_novelty=preferred_novelty
            )

            # Initialize new attributes for tracking stats
            agent.num_self_evals = 0
            agent.num_other_evals = 0
            agent.num_shares = 0
            agent.num_domain_adoptions = 0
            agent.total_novelty_generated = 0.0
            agent.total_interest_generated = 0.0

            agents.append(agent)
            
            # Log agent initialization details
            self.logger.log_event('agent_init', {
                'agent_id': agent.unique_id,
                'preferred_novelty': agent.preferred_novelty,
                'reward_mean': agent.wundt.reward_mean,
                'punishment_mean': agent.wundt.punish_mean
            })
            
        return agents

    @time_it
    def _parallel_apply_custom(self, function, *args):
        """
        Parallelizes a function across available GPUs with custom data handling.
        """
        primary_scatter_tensor = args[0]
        
        scattered_args = [[] for _ in self.device_ids]
        
        scattered_primary = scatter(primary_scatter_tensor, self.device_ids)
        for i in range(len(self.device_ids)):
            scattered_args[i].append(scattered_primary[i])

        for arg in args[1:]:
            if isinstance(arg, torch.Tensor) and arg.shape[0] == primary_scatter_tensor.shape[0]:
                scattered_tensor = scatter(arg, self.device_ids)
                for i in range(len(self.device_ids)):
                    scattered_args[i].append(scattered_tensor[i])
            else:
                for i in range(len(self.device_ids)):
                    replicated_arg = arg.to(f'cuda:{self.device_ids[i]}') if isinstance(arg, torch.Tensor) else arg
                    scattered_args[i].append(replicated_arg)
        
        wrapped_module = _FunctionWrapper(function)
        module_replicas = replicate(wrapped_module, self.device_ids)
        
        inputs = [tuple(arg_list) for arg_list in scattered_args]

        outputs = parallel_apply(module_replicas, inputs, devices=self.device_ids)

        return gather(outputs, self.device)

    def _gpu_stream_context(self):
        """Return a safe stream context manager for CUDA and CPU execution."""
        if self.device.type == 'cuda' and self.gpu_stream is not None:
            return torch.cuda.stream(self.gpu_stream)
        return nullcontext()

    @time_it
    def step(self):
        """
        Executes one full step of Algorithm 1.

        Phase ordering:
          1. Generate     → generation_phase()
          2. Render + FE  → evaluation_phase()
          3. Self-eval    → individual_evaluation_phase()
          4. Share decide → sharing_phase()
          5. Receive eval → interaction_phase()
          6. Boredom      → boredom_phase()
          7. Thresholds   → update_system_thresholds()
        """
        generated_artifacts = self.generation_phase()
        evaluated_artifacts = self.evaluation_phase(generated_artifacts)
        self.individual_evaluation_phase(evaluated_artifacts)
        messages = self.sharing_phase(evaluated_artifacts)
        interaction_results = self.interaction_phase(messages)
        
        if self.step_count != 0:
            self.boredom_phase()
            
        self.update_system_thresholds()
        
        self._log_step_metrics(interaction_results)
        self.step_count += 1

    @time_it
    def generation_phase(self) -> List[Artifact]:
        """
        Algorithm 1, step 1: Generate new artifacts for all agents.
        Each agent breeds its current expression with one from
        memory, or creates a random tree if no prior expression.
        """
        return self.artifact_generator.generate(self.agents)

    @time_it
    def evaluation_phase(self, artifacts: List[Artifact]) -> List[Artifact]:
        """
        Renders expression trees to images and extracts feature
        vectors (3.3.2). This is a helper phase not explicitly
        in Algorithm 1 as it prepares the data needed by
        individual_evaluation_phase and interaction_phase.

        DEVIATION(paper 3.5): All artifacts are rendered and
        feature-extracted in one batched GPU pass rather than
        sequentially per agent.
        """
        if not artifacts:
            return []
        
        expressions = [a.content for a in artifacts]
        
        # 1. Generate directly to GPU Tensor (N, 3, H, W) range [0, 1]
        image_tensor_batch = self.image_generator.generate_batch(expressions, use_amp=self.use_amp)

        if image_tensor_batch.shape[0] == 0:
             return []

        if self.save_images and self.image_output_dir:
            image_batch_cpu = image_tensor_batch.detach().to('cpu')
            for i, artifact in enumerate(artifacts):
                image_path = os.path.join(self.image_output_dir, f"{artifact.id}.png")
                torchvision.utils.save_image(image_batch_cpu[i], image_path)
        
        # 2. ImageNet normalization on GPU -- required because
        # ResNet-18 was pre-trained on ImageNet statistics.
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        normalized_batch = (image_tensor_batch - mean) / std

        # 3. Extract Features
        with self._gpu_stream_context():
            with torch.no_grad():
                features_batch = self.feature_extractor(normalized_batch).detach()
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        for i, artifact in enumerate(artifacts):
            artifact.features = features_batch[i]
        return artifacts

    @time_it
    def sharing_phase(self, evaluated_artifacts: List[Artifact]) -> List[Dict]:
        """
        Algorithm 1, step 3: Share decision.
        If interest > τ_C (self_threshold), share with N randomly
        selected agents (3.4).
        """
        messages = []
        
        # Create a map from producer_id to their just-generated artifact
        agent_to_artifact = {art.producer_id: art for art in evaluated_artifacts}
        
        for agent in self.agents:
            # Get the artifact this agent JUST generated
            just_generated = agent_to_artifact.get(agent.unique_id)
            
            if just_generated and just_generated.interest > self.self_threshold:
                agent.num_shares += 1
                num_recipients = min(self.share_count, self.num_agents - 1)
                if num_recipients <= 0: continue
                
                recipients = random.sample([a.unique_id for a in self.agents if a.unique_id != agent.unique_id], k=num_recipients)
                
                for recipient_id in recipients:
                    messages.append({
                        'artifact': just_generated,
                        'sender_id': agent.unique_id,
                        'recipient_id': recipient_id
                    })
        return messages


    @time_it
    def interaction_phase(self, messages: List[Dict]) -> List[Dict]:
        """
        Algorithm 1, step 4: Receive & evaluate shared artifacts.
        Each recipient evaluates via their own kNN + Wundt curve.
        If interest > τ_D (domain_threshold), artifact enters the
        domain and recipient adopts the expression (3.4).
        """
        if not messages:
            return []

        # Batch evaluate novelty for all messages
        with self._gpu_stream_context():
            query_features_list = [msg['artifact'].features for msg in messages]
            query_batch = torch.stack(query_features_list)
            
            # Prepare memory tensors (same as individual evaluation)
            memory_tensors = [agent.knn.feature_vectors for agent in self.agents]
            feature_dim = query_batch.shape[1]
            memory_tensors = [mem if mem.numel() > 0 else torch.empty(0, feature_dim, device=self.device) for mem in memory_tensors]
            consolidated_memories = torch.cat(memory_tensors, dim=0)
            
            memory_indices = torch.zeros(self.num_agents, 2, dtype=torch.long, device=self.device)
            agent_ks = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
            current_index = 0
            for i, agent in enumerate(self.agents):
                mem_size = agent.knn.feature_vectors.shape[0]
                if mem_size > 0:
                    memory_indices[i] = torch.tensor([current_index, current_index + mem_size], device=self.device)
                    current_index += mem_size
                else:
                    memory_indices[i] = torch.tensor([-1, -1], device=self.device)
                agent_ks[i] = agent.knn.k

            # Map messages to recipient agent IDs for batch processing
            message_to_agent_map = torch.tensor([msg['recipient_id'] for msg in messages], device=self.device, dtype=torch.long)

            if self.multi_gpu and query_batch.shape[0] > 1:
                novelty_scores_tensor = self._parallel_apply_custom(
                    kNN.batch_evaluate_novelty_for_messages,
                    query_batch, message_to_agent_map, consolidated_memories, memory_indices, agent_ks, self.distance_metric
                )
            else:
                novelty_scores_tensor = kNN.batch_evaluate_novelty_for_messages(
                    query_batch, message_to_agent_map, consolidated_memories, memory_indices, agent_ks, self.distance_metric
                )
            
            # DEVIATION(paper 3.3.3): Only self-generated
            # artifacts define the novelty landscape.
            # Receiving is consumption, not production.

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        novelty_scores = novelty_scores_tensor.cpu().numpy()

        interaction_results = []
        
        for i, message in enumerate(messages):
            recipient = self.agents[message['recipient_id']]
            artifact = message['artifact']
            
            raw_novelty = novelty_scores[i]
            # Use same normalization as generation phase
            normalized_novelty = self._normalize_novelty(raw_novelty)
            interest = recipient.wundt.hedonic_value(normalized_novelty, experience=recipient.knn.current_size)
            
            self.stats.record_other_interest(interest)
            
            recipient.num_other_evals += 1
            
            accepted = False
            if interest > self.domain_threshold:
                accepted = True
                
                # Artifact enters the domain (3.4)
                self.domain.append(artifact)
                # Keep domain from growing infinitely
                if len(self.domain) > self.max_domain_size:
                    self.domain.pop(0) # Remove oldest
                
                # DEVIATION(paper 3.4): Recipient adoption of the
                # shared expression as current state is optional.
                if self.adopt_shared_expression:
                    recipient.current_expression = artifact.content._copy()
                    recipient.current_interest = interest
                    recipient.current_artifact_id = artifact.id
                recipient.current_creator_id = artifact.creator_id

            # Receiving implies exposure: update recipient memory
            # even when the artifact is not accepted into domain.
            # DEVIATION(paper 3.3.1): Uniqueness check only add
            # to kNN if expression string differs from last 5.
            expr_str = artifact.content.to_string()
            recent_exprs = [mem['expression'].to_string() for mem in recipient.artifact_memory[-5:]]
            if expr_str not in recent_exprs:
                recipient.knn.add_feature_vectors(artifact.features.unsqueeze(0), self.step_count)
                recipient.artifact_memory.append({
                    'id': artifact.id,
                    'expression': artifact.content,
                    'features': artifact.features,
                    'creator_id': artifact.creator_id
                })
            
            interaction_results.append({
                'accepted': accepted,
                'interest': interest,
                'novelty': normalized_novelty
            })

            self.logger.log_event('share', {
                'step': self.step_count, 'sender_id': message['sender_id'], 'recipient_id': recipient.unique_id,
                'artifact_id': artifact.id, 'expression': artifact.content.to_string(),
                'evaluated_novelty': normalized_novelty, 'evaluated_interest': interest,
                'accepted': accepted,
                'creator_id': artifact.creator_id,
                'evaluator_id': recipient.unique_id,
                'domain_size': len(self.domain)
            })
            
        return interaction_results

    @time_it
    def individual_evaluation_phase(self, evaluated_artifacts: List[Artifact]):
        """
        Algorithm 1, step 2: Self-evaluate own artifact.
        ResNet-18 features → kNN novelty → normalize → Wundt
        curve hedonic value → update cumulative interest (3.3).

        DEVIATION(paper 3.3.1): Uniqueness check which compares
        expression string against last 5 in memory before adding
        to kNN. Prevents duplicate features from diluting novelty.
        Not described in the paper.
        """
        if not evaluated_artifacts:
            return

        with self._gpu_stream_context():
            query_features_list = [art.features for art in evaluated_artifacts]
            query_batch = torch.stack(query_features_list)

            memory_tensors = [agent.knn.feature_vectors for agent in self.agents]
            
            feature_dim = query_batch.shape[1]
            memory_tensors = [mem if mem.numel() > 0 else torch.empty(0, feature_dim, device=self.device) for mem in memory_tensors]

            consolidated_memories = torch.cat(memory_tensors, dim=0)
            
            memory_indices = torch.zeros(self.num_agents, 2, dtype=torch.long, device=self.device)
            agent_ks = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
            current_index = 0
            for i, agent in enumerate(self.agents):
                mem_size = agent.knn.feature_vectors.shape[0]
                if mem_size > 0:
                    memory_indices[i] = torch.tensor([current_index, current_index + mem_size], device=self.device)
                    current_index += mem_size
                else:
                    memory_indices[i] = torch.tensor([-1, -1], device=self.device)
                agent_ks[i] = agent.knn.k

            if self.multi_gpu and query_batch.shape[0] > 1:
                novelty_scores_tensor = self._parallel_apply_custom(
                    kNN.batch_evaluate_novelty_for_agents,
                    query_batch, consolidated_memories, memory_indices, agent_ks, self.distance_metric
                )
            else:
                novelty_scores_tensor = kNN.batch_evaluate_novelty_for_agents(
                    query_batch, consolidated_memories, memory_indices, agent_ks, self.distance_metric
                )
            
            self.stats.update_novelty_stats(novelty_scores_tensor, self.step_count)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        novelty_scores = novelty_scores_tensor.cpu().numpy()

        for i, artifact in enumerate(evaluated_artifacts):
            agent = self.agents[artifact.producer_id]
            features = query_batch[i].unsqueeze(0)

            raw_novelty = novelty_scores[i]
            normalized_novelty = self._normalize_novelty(raw_novelty)
            interest = agent.wundt.hedonic_value(normalized_novelty, experience=agent.knn.current_size)
            
            self.stats.record_self_interest(interest)
            
            # --- Update Agent State ---
            # DEVIATION(paper 3.4): Hall of fame update not in paper.
            agent.update_hall_of_fame(artifact.content, interest)
            # Cumulative interest EMA (3.4): S_i = α·S_i + (1-α)·h_i
            agent.average_interest = agent.alpha * agent.average_interest + (1 - agent.alpha) * interest
            agent.self_eval_history.append(interest)
            
            # Store previous interest for adoption comparison
            previous_interest = agent.current_interest

            # Needed by extended boredom mode for overwhelm detection
            agent.current_novelty = normalized_novelty

            agent.num_self_evals += 1
            agent.total_novelty_generated += normalized_novelty
            agent.total_interest_generated += interest
            
            # DEVIATION(paper 3.3.1): Uniqueness check
            # last 5 expressions compared by string to prevent
            # duplicate features entering kNN memory.
            expr_str = artifact.content.to_string()
            recent_exprs = [mem['expression'].to_string() for mem in agent.artifact_memory[-5:]]
            is_unique = expr_str not in recent_exprs
            
            if is_unique:
                agent.knn.add_feature_vectors(features, self.step_count)
                agent.artifact_memory.append({
                    'id': artifact.id, 
                    'expression': artifact.content, 
                    'features': artifact.features,
                    'creator_id': artifact.creator_id
                })

            artifact.novelty = normalized_novelty
            artifact.interest = interest

            # Adoption check: is my new creation better than what I had?
            adopted = False
            if agent.current_expression is None or interest > previous_interest:
                agent.current_expression = artifact.content
                agent.current_interest = interest
                agent.current_artifact_id = artifact.id
                agent.current_creator_id = artifact.creator_id
                adopted = True

            self.logger.log_event('generation', {
                'step': self.step_count, 'agent_id': agent.unique_id, 'artifact_id': artifact.id,
                'expression': artifact.content.to_string(), 'novelty': normalized_novelty, 'interest': interest,
                'adopted': adopted,
                'parent1_id': artifact.parent1_id, 'parent2_id': artifact.parent2_id,
                'creator_id': artifact.creator_id,
                'evaluator_id': agent.unique_id,
                'domain_size': len(self.domain)
            })

    @time_it
    def boredom_phase(self):
        """
        Algorithm 1, step 5: Boredom check (3.4).
        Triggers when cumulative interest S_i < τ_B.

        Two modes are available:

        CLASSIC MODE (paper-compliant, Algorithm 1):
            Retrieve a random artifact from the domain, evaluate
            it through the agent's kNN + Wundt curve, adopt if
            hedonic value exceeds current interest.

            DEVIATION(paper 3.5): All classic bored agents are
            processed in a single batched GPU pass (image render
            → feature extraction → kNN novelty) rather than one
            GPU dispatch per agent.

        EXTENDED MODE:
            DEVIATION(paper 3.4): Not described in paper.
            Paper: One boredom mechanism (random domain retrieval).
            Code: Differentiates two causes of low interest:
              - "True Boredom" (low novelty): explore domain
                or restart with random expression.
              - "Overwhelm" (high novelty): retreat to a known
                high-interest artifact from hall of fame.
        """
        # ------------------------------------------------------------------
        # Pass 1: classify bored agents
        # Extended-mode agents need no GPU work; process them immediately.
        # Classic-mode agents are collected for a batched GPU pipeline.
        # ------------------------------------------------------------------
        classic_agents   = []
        chosen_artifacts = []

        for agent in self.agents:
            if agent.average_interest >= self.boredom_threshold:
                continue

            if self.boredom_mode != 'classic':
                self._boredom_extended(agent)
                continue

            # Classic mode: sample a domain artifact now so that the memory
            # snapshot used for novelty is consistent with the pre-batch state.
            if not self.domain:
                continue

            domain_artifact = random.choice(self.domain)
            classic_agents.append(agent)
            chosen_artifacts.append(domain_artifact)

        if not classic_agents:
            return

        # ------------------------------------------------------------------
        # Feature acquisition: reuse cached features when available.
        # evaluation_phase already stores artifact.features for every artifact
        # that passes through the domain, so re-rendering from the GPU is
        # unnecessary in the common case.  Only artifacts that somehow lack a
        # cached feature vector (e.g. injected without going through
        # evaluation_phase) fall back to the GPU pipeline.
        # ------------------------------------------------------------------
        cached_features:   list = []   # (agent_idx, feature_tensor) for artifacts with .features
        uncached_indices:  list = []   # positions in classic_agents that need rendering
        uncached_artifacts: list = []  # corresponding chosen_artifacts

        for idx, art in enumerate(chosen_artifacts):
            if art.features is not None:
                cached_features.append((idx, art.features))
            else:
                uncached_indices.append(idx)
                uncached_artifacts.append(art)

        # Determine feature dimensionality from first available source.
        if cached_features:
            feature_dim = cached_features[0][1].shape[0]
        elif uncached_artifacts:
            # Need at least one GPU render to learn feature_dim.
            _probe_batch = self.image_generator.generate_batch(
                [uncached_artifacts[0].content], use_amp=self.use_amp
            )
            if _probe_batch.shape[0] == 0:
                return
            _mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            _std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            with self._gpu_stream_context():
                with torch.no_grad():
                    _f = self.feature_extractor((_probe_batch - _mean) / _std).detach()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            feature_dim = _f.shape[1]
            # Store back so this artifact is also cached going forward.
            uncached_artifacts[0].features = _f[0]
            cached_features.append((uncached_indices[0], _f[0]))
            uncached_indices = uncached_indices[1:]
            uncached_artifacts = uncached_artifacts[1:]
        else:
            return  # nothing to process

        # Render any remaining uncached artifacts in one batch.
        uncached_feature_map: dict = {}
        if uncached_artifacts:
            expressions = [art.content for art in uncached_artifacts]
            image_tensor_batch = self.image_generator.generate_batch(expressions, use_amp=self.use_amp)
            if image_tensor_batch.shape[0] > 0:
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                normalized_batch = (image_tensor_batch - mean) / std
                with self._gpu_stream_context():
                    with torch.no_grad():
                        rendered_features = self.feature_extractor(normalized_batch).detach()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                for j, (orig_idx, art) in enumerate(zip(uncached_indices, uncached_artifacts)):
                    art.features = rendered_features[j]          # cache for future reuse
                    uncached_feature_map[orig_idx] = rendered_features[j]

        # Assemble the ordered features_batch tensor (n_classic, feat_dim).
        features_list = [None] * len(classic_agents)
        for orig_idx, feat in cached_features:
            features_list[orig_idx] = feat
        for orig_idx, feat in uncached_feature_map.items():
            features_list[orig_idx] = feat

        valid = [(i, f) for i, f in enumerate(features_list) if f is not None]
        if not valid:
            return

        # Drop any agents whose artifact had an empty render (edge case).
        if len(valid) < len(classic_agents):
            keep_idx    = [i for i, _ in valid]
            classic_agents   = [classic_agents[i]   for i in keep_idx]
            chosen_artifacts = [chosen_artifacts[i] for i in keep_idx]
            features_list    = [f for _, f in valid]

        features_batch = torch.stack(features_list)   # (n_classic, feat_dim)

        # ------------------------------------------------------------------
        # Build kNN batch tensors (mirrors interaction_phase)
        # ------------------------------------------------------------------
        n_classic   = len(classic_agents)
        feature_dim = features_batch.shape[1]

        memory_tensors = [
            agent.knn.feature_vectors if agent.knn.feature_vectors.numel() > 0
            else torch.empty(0, feature_dim, device=self.device)
            for agent in classic_agents
        ]
        consolidated_memories = torch.cat(memory_tensors, dim=0)  # (total_mem, feat_dim)

        memory_indices = torch.zeros(n_classic, 2, dtype=torch.long, device=self.device)
        agent_ks       = torch.zeros(n_classic,    dtype=torch.long, device=self.device)
        current_index  = 0

        for i, agent in enumerate(classic_agents):
            mem_size = agent.knn.feature_vectors.shape[0]
            if mem_size > 0:
                memory_indices[i] = torch.tensor(
                    [current_index, current_index + mem_size], device=self.device
                )
                current_index += mem_size
            else:
                memory_indices[i] = torch.tensor([-1, -1], device=self.device)
            agent_ks[i] = agent.knn.k

        # ------------------------------------------------------------------
        # Batch novelty computation
        # ------------------------------------------------------------------
        novelty_scores = kNN.batch_evaluate_novelty_for_agents(
            features_batch, consolidated_memories, memory_indices, agent_ks, self.distance_metric
        )  # shape: (n_classic,)

        # ------------------------------------------------------------------
        # Pass 2: distribute results
        # ------------------------------------------------------------------
        for i, (agent, domain_artifact) in enumerate(zip(classic_agents, chosen_artifacts)):
            features           = features_batch[i]           # (feat_dim,)
            raw_novelty        = novelty_scores[i].item()
            normalized_novelty = self._normalize_novelty(raw_novelty)
            interest           = agent.wundt.hedonic_value(normalized_novelty)

            # Add to memory regardless
            agent.knn.add_feature_vectors(features.unsqueeze(0), self.step_count)
            agent.artifact_memory.append({
                'id':         domain_artifact.id,
                'expression': domain_artifact.content,
                'features':   features,
                'creator_id': domain_artifact.creator_id,
            })

            # Adopt only if better than current
            if interest > agent.current_interest:
                agent.current_expression  = domain_artifact.content._copy()
                agent.current_interest    = interest
                agent.current_artifact_id = domain_artifact.id
                agent.current_creator_id  = domain_artifact.creator_id

            self.logger.log_event('boredom_adoption', {
                'step':            self.step_count,
                'agent_id':        agent.unique_id,
                'source':          'domain_classic',
                'trigger_novelty': getattr(agent, 'current_novelty', 0.5),
                'creator_id':      domain_artifact.creator_id,
                'evaluator_id':    agent.unique_id,
                'domain_size':     len(self.domain),
            })

    def _boredom_extended(self, agent):
        """
        Extended boredom mechanism (Saunders, 2025).

        DEVIATION(paper 3.4): Entire method is not in the paper.
        Paper: Single boredom mode (random domain retrieval).
        Code: Differentiates overwhelm (→ hedonic retreat to
              hall of fame) from true boredom (→ domain explore
              or random restart).

        Distinguishes between overwhelm (retreat) and true
        boredom (explore).
        """
        current_nov = getattr(agent, 'current_novelty', 0.5)
        parent_expr = None
        source_creator_id = agent.unique_id
        source_type = "random"
        
        # Overwhelm detection: novelty exceeds preferred + 0.2
        # means the agent is seeing things too far from its taste.
        is_overwhelmed = current_nov > (agent.preferred_novelty + 0.2)
        
        if is_overwhelmed and agent.hall_of_fame:
            # Hedonic retreat: return to best-known artifact
            entry = agent.hall_of_fame[0]
            parent_expr = entry[1]
            source_creator_id = agent.current_creator_id if agent.current_creator_id is not None else agent.unique_id
            source_type = "hedonic_retreat"
        elif self.domain and random.random() < 0.7:
            # True boredom: sample domain for fresh inspiration
            parent_artifact = random.choice(self.domain)
            parent_expr = parent_artifact.content
            source_creator_id = parent_artifact.creator_id
            source_type = "domain_exploration"
        else:
            # Creative restart: generate entirely new expression
            parent_expr = genart.ExpressionNode.create_random(depth=agent.gen_depth)
            source_creator_id = agent.unique_id
            source_type = "random_restart"
        
        new_expr = parent_expr._copy()
        new_expr.mutate(rate=0.1, max_depth=agent.gen_depth)
        
        agent.current_expression = new_expr
        agent.current_interest = 0.0
        agent.current_creator_id = source_creator_id
        
        self.logger.log_event('boredom_adoption', {
            'step': self.step_count,
            'agent_id': agent.unique_id,
            'source': source_type,
            'trigger_novelty': current_nov,
            'creator_id': source_creator_id,
            'evaluator_id': agent.unique_id,
            'domain_size': len(self.domain)
        })

    def _normalize_novelty(self, raw_novelty):
        """
        Normalizes a raw novelty score to the [0, 1] range using the StatsTracker.
        """
        return self.stats.get_normalized_novelty(raw_novelty)

    def update_system_thresholds(self):
        """
        Algorithm 1, step 6: Update thresholds (3.4).
        Delegates to StatsTracker, then syncs values back
        to scheduler attributes for use in next step's phases.
        """
        self.stats.update_thresholds(self.agents)
        
        # Sync values to scheduler so they are available for next step's sharing phase
        self.boredom_threshold = self.stats.boredom_thresh
        self.self_threshold = self.stats.self_thresh
        self.domain_threshold = self.stats.domain_thresh

    def _log_step_metrics(self, interaction_results):
        """
        Logs aggregate metrics for the current step.
        """
        avg_accepted_interest = 0
        avg_rejected_interest = 0
        if interaction_results:
            accepted_interests = [r['interest'] for r in interaction_results if r['accepted']]
            rejected_interests = [r['interest'] for r in interaction_results if not r['accepted']]
            if accepted_interests:
                avg_accepted_interest = sum(accepted_interests) / len(accepted_interests)
            if rejected_interests:
                avg_rejected_interest = sum(rejected_interests) / len(rejected_interests)

        avg_knn_size = sum([a.knn.feature_vectors.shape[0] for a in self.agents]) / self.num_agents

        self.logger.log_event('step_end', {
            'step': self.step_count,
            'domain_size': len(self.domain),
            'self_threshold': self.self_threshold,
            'domain_threshold': self.domain_threshold,
            'boredom_threshold': self.boredom_threshold,
            'avg_accepted_interest': avg_accepted_interest,
            'avg_rejected_interest': avg_rejected_interest,
            'avg_knn_size': avg_knn_size
        })
        
        # Log agent states periodically
        if self.step_count % 10 == 0:
            for agent in self.agents:
                 self.logger.log_event('agent_state', {
                    'step': self.step_count,
                    'agent_id': agent.unique_id,
                    'cumulative_interest': agent.average_interest,
                    'repository_size': agent.knn.feature_vectors.shape[0],
                    'k_value': agent.knn.k,
                    'boredom_triggered': False,
                    'num_self_evals': agent.num_self_evals,
                    'num_other_evals': agent.num_other_evals,
                    'num_shares': agent.num_shares,
                    'num_domain_adoptions': agent.num_domain_adoptions,
                    'avg_novelty_generated': agent.total_novelty_generated / max(1, agent.num_self_evals),
                    'avg_interest_generated': agent.total_interest_generated / max(1, agent.num_self_evals)
                })

    def close(self):
        """
        Clean up resources.
        """
        if hasattr(self.logger, 'close'):
            self.logger.close()