"""
k-Nearest Neighbors (kNN) Memory
================================

This module implements a GPU-accelerated k-Nearest Neighbors memory system
for novelty estimation. Each agent maintains its own kNN instance to track
the artifacts it has observed and calculate how novel a new artifact is
relative to its recorded experience.

Novelty score calculation:
    novelty = mean_distance(k-NN) / std_dev(k-NN distances)

Dividing by the standard deviation normalizes the score across agents 
with different historical memory distributions.
"""

import torch
import numpy as np
from timing_utils import time_it

class kNN:
    """
    A GPU-accelerated k-Nearest Neighbors implementation for novelty search.
    """
    def __init__(self, agent_id=None, max_size=1000, dtype=torch.float32):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == 'cuda' else torch.float32
        self.max_size = max_size
        
        self.memory_buffer = None 
        self.k = 15 
        self.ptr = 0 
        self.current_size = 0

        self._empty_feature_vectors = torch.empty((0,), device=self.device, dtype=self.dtype)

    @property
    def feature_vectors(self):
        """
        Returns the valid populated slice of the memory buffer.
        Excludes unallocated padding and returns an empty tensor if no features are stored.
        """
        if self.memory_buffer is None or self.current_size == 0:
            return self._empty_feature_vectors
        
        return self.memory_buffer[:self.current_size]

    @time_it
    @staticmethod
    @torch.no_grad()
    def batch_evaluate_novelty_for_agents(
        queries: torch.Tensor,
        global_buffer: torch.Tensor,
        current_sizes: torch.Tensor,
        agent_ks: torch.Tensor,
        chunk_size: int = 1000
    ) -> torch.Tensor:
        """
        Calculates novelty scores for a batch of agents against their unique histories
        using batched matrix multiplications with cosine similarity.
        """
        num_queries = queries.shape[0]
        if num_queries == 0:
            return torch.empty(0, device=queries.device, dtype=queries.dtype)
            
        M = global_buffer.shape[1]
        device = queries.device
        dtype = queries.dtype

        if M == 0:
            return torch.ones(num_queries, device=device, dtype=dtype)
            
        novelties = []
        # Process the population queries in chunks to control VRAM footprint
        for start_idx in range(0, num_queries, chunk_size):
            end_idx = min(start_idx + chunk_size, num_queries)
            
            # Extract current slice of query vectors, masking out invalid numerical artifacts
            q_chunk = queries[start_idx:end_idx]
            q_chunk = torch.nan_to_num(q_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Extract tracking metadata matching the current chunk indices
            sizes_chunk = current_sizes[start_idx:end_idx]
            ks_chunk = agent_ks[start_idx:end_idx]
            rec_buf = global_buffer[start_idx:end_idx]
            
            # Normalize vectors to calculate cosine similarity via matrix dot products
            q_chunk = torch.nn.functional.normalize(q_chunk, p=2, dim=1)
            sims = torch.bmm(q_chunk.unsqueeze(1), rec_buf.transpose(1, 2)).squeeze(1)
                
            # Create a coordinate grid to identify filled slots versus unallocated padding
            positions = torch.arange(M, device=device).unsqueeze(0)
            valid = positions < sizes_chunk.unsqueeze(1)
            
            # Force similarities of empty tracking slots to -1e9 so topk ignores them
            sims.masked_fill_(~valid, -1e9)
            
            # Calculate the widest neighborhood parameter required across this batch
            max_k = int(torch.minimum(torch.tensor(M, device=device), ks_chunk.max()).item())
            if max_k <= 0:
                novelties.append(torch.ones(end_idx - start_idx, device=device, dtype=dtype))
                continue
                
            # Pull the closest neighbors based on highest similarity scores
            top_sims, _ = torch.topk(sims, k=max_k, dim=1, largest=True)
            
            # Limit neighborhood size if the agent has fewer total entries than its k parameter
            eff_ks = torch.clamp(torch.minimum(ks_chunk, sizes_chunk), min=1)
            
            # Build index tensor to map across neighborhood rows sequentially
            k_idx = torch.arange(max_k, device=device).unsqueeze(0)
            
            # Isolate entries falling strictly inside each agent's effective k boundary
            k_mask = k_idx < eff_ks.unsqueeze(1)
            
            # Zero out neighbor entries falling outside the active target mask range
            masked_sims = top_sims.masked_fill(~k_mask, 0.0)
            sum_sims = masked_sims.sum(dim=1)
            eff_k_float = eff_ks.float()
            
            # Translate similarity scores to standard statistical distance metrics
            mean = sum_sims / eff_k_float
            masked_sq = (top_sims ** 2).masked_fill_(~k_mask, 0.0)
            mean_sq = masked_sq.sum(dim=1) / eff_k_float
            variance = torch.clamp(mean_sq - mean ** 2, min=1e-9)
            std = torch.sqrt(variance)
            
            # Normalize distance by standard deviation to scale scores uniformly across agents
            novelty = (1.0 - mean) / torch.clamp(std, min=1e-6)
                
            # Default to maximum novelty if the agent has insufficient history
            novelty.masked_fill_(sizes_chunk <= 1, 1.0)
            novelty = torch.nan_to_num(novelty, nan=0.0, posinf=1.0, neginf=-1.0)
            novelties.append(novelty)
            
        return torch.cat(novelties) if novelties else torch.empty(0, device=queries.device, dtype=queries.dtype)

    @time_it
    @staticmethod
    @torch.no_grad()
    def batch_evaluate_novelty_for_messages(
        queries: torch.Tensor,
        recipient_ids: torch.Tensor,
        global_buffer: torch.Tensor,
        current_sizes: torch.Tensor,
        agent_ks: torch.Tensor,
        chunk_size: int = 4096
    ) -> torch.Tensor:
        """
        Calculates novelty scores for shared artifacts against recipient memories
        using vectorized matrix operations with cosine similarity.
        """
        num_queries = queries.shape[0]
        if num_queries == 0:
            return torch.empty(0, device=queries.device, dtype=queries.dtype)
            
        M = global_buffer.shape[1]
        device = queries.device
        dtype = queries.dtype

        if M == 0:
            return torch.ones(num_queries, device=device, dtype=dtype)
            
        novelties = []
        # Step through message batches to compute peer evaluations efficiently
        for start_idx in range(0, num_queries, chunk_size):
            end_idx = min(start_idx + chunk_size, num_queries)
            q_chunk = queries[start_idx:end_idx]
            ids_chunk = recipient_ids[start_idx:end_idx]
            
            q_chunk = torch.nan_to_num(q_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Map tracking dimensions matching the specific message recipient targets
            sizes_chunk = current_sizes[ids_chunk]
            ks_chunk = agent_ks[ids_chunk]
            
            # Pull discrete history slices corresponding to target recipient spaces
            rec_buf = global_buffer[ids_chunk]
            
            # Execute batch matrix multiplication to find cross-network cosine similarity
            q_chunk = torch.nn.functional.normalize(q_chunk, p=2, dim=1)
            sims = torch.bmm(q_chunk.unsqueeze(1), rec_buf.transpose(1, 2)).squeeze(1)
                
            # Generate mask tracking array to separate valid slots from padding indices
            positions = torch.arange(M, device=device).unsqueeze(0)
            valid = positions < sizes_chunk.unsqueeze(1)
            sims.masked_fill_(~valid, -1e9)
            
            # Determine maximum neighborhood parameters required across recipients
            max_k = int(torch.minimum(torch.tensor(M, device=device), ks_chunk.max()).item())
            if max_k <= 0:
                novelties.append(torch.ones(end_idx - start_idx, device=device, dtype=dtype))
                continue
                
            # Extract closest neighbor match elements
            top_sims, _ = torch.topk(sims, k=max_k, dim=1, largest=True)
            
            # Bound evaluation thresholds based on current recipient memory allocation depth
            eff_ks = torch.clamp(torch.minimum(ks_chunk, sizes_chunk), min=1)
            k_idx = torch.arange(max_k, device=device).unsqueeze(0)
            k_mask = k_idx < eff_ks.unsqueeze(1)
            
            # Mask neighbor components residing beyond current effective limits
            masked_sims = top_sims.masked_fill(~k_mask, 0.0)
            sum_sims = masked_sims.sum(dim=1)
            eff_k_float = eff_ks.float()
            
            # Evaluate statistical distribution averages and variances
            mean = sum_sims / eff_k_float
            masked_sq = (top_sims ** 2).masked_fill_(~k_mask, 0.0)
            mean_sq = masked_sq.sum(dim=1) / eff_k_float
            variance = torch.clamp(mean_sq - mean ** 2, min=1e-9)
            std = torch.sqrt(variance)
            novelty = (1.0 - mean) / torch.clamp(std, min=1e-6)
                
            # Isolate entries lacking historical context to provide default values
            novelty.masked_fill_(sizes_chunk <= 1, 1.0)
            novelty = torch.nan_to_num(novelty, nan=0.0, posinf=1.0, neginf=-1.0)
            novelties.append(novelty)
            
        return torch.cat(novelties) if novelties else torch.empty(0, device=queries.device, dtype=queries.dtype)

    @time_it
    def add_feature_vectors(self, new_feature_vectors, step=0, pre_normalize: bool = True):
        """
        Appends new feature vectors to the memory buffer using a circular structure.
        """
        try:
            new_feature_vectors = new_feature_vectors.to(self.device, dtype=self.dtype)
            
            if pre_normalize:
                new_feature_vectors = torch.nn.functional.normalize(
                    new_feature_vectors, p=2, dim=1
                )
                
            # Lazily initialize storage block if first allocation pass
            if self.memory_buffer is None:
                feature_dim = new_feature_vectors.shape[1]
                self.memory_buffer = torch.zeros(
                    (self.max_size, feature_dim), 
                    device=self.device, 
                    dtype=self.dtype
                )
                self._empty_feature_vectors = torch.empty(
                    (0, feature_dim), device=self.device, dtype=self.dtype
                )
            
            num_new = new_feature_vectors.shape[0]
            
            # Cap incoming arrays if update footprint exceeds buffer limits
            if num_new > self.max_size:
                new_feature_vectors = new_feature_vectors[-self.max_size:]
                num_new = self.max_size

            start_idx = self.ptr
            end_idx = start_idx + num_new
            
            # Insert updates linearly or wrap circular buffer bounds across boundaries
            if end_idx <= self.max_size:
                self.memory_buffer[start_idx:end_idx] = new_feature_vectors
            else:
                overflow = end_idx - self.max_size
                self.memory_buffer[start_idx:] = new_feature_vectors[:-overflow]
                self.memory_buffer[:overflow] = new_feature_vectors[-overflow:]
            
            # Increment internal write cursor trackers
            self.ptr = (self.ptr + num_new) % self.max_size
            self.current_size = min(self.current_size + num_new, self.max_size)
            
        except Exception as e:
            print(f"Agent {self.agent_id} failed to add features: {e}")