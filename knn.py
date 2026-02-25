"""
k-Nearest Neighbors (kNN) Memory (3.3.1)
==========================================

This module implements a GPU-accelerated k-Nearest Neighbors memory system
for novelty estimation. Each agent maintains its own kNN instance to track
the artifacts it has seen and calculate how "novel" a new artifact is
relative to that experience.

Novelty formula (3.3.1):
    novelty = mean_distance(k-NN) / std_dev(k-NN distances)

The division by standard deviation normalizes for varying
scale across agents with different memory distributions.
"""

import torch
import numpy as np
import random
from timing_utils import time_it

class kNN:
    """
    A GPU-accelerated k-Nearest Neighbors implementation for novelty search.

    DEVIATION(paper 3.3.1): Uses a pre-allocated circular buffer
    with max_size instead of unbounded memory growth.
    Paper: Implies memory grows without bound.
    Code: Circular buffer caps at max_size (default 5000),
          overwriting oldest entries. This prevents GPU OOM
          and keeps novelty estimation focused on recent
          experience rather than ancient history.
    """
    def __init__(self, agent_id=None, max_size=5000, dtype=torch.float32):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == 'cuda' else torch.float32
        self.max_size = max_size
        
        # Lazy-initialized circular buffer; allocated on first
        # add_feature_vectors() call when feature_dim is known.
        self.memory_buffer = None 
        
        # DEVIATION(paper 3.3.1): Fixed k instead of elbow method.
        # Paper: Automated elbow method to dynamically determine
        #        optimal k per agent.
        # Code: Fixed k=15 for consistent scientific baseline.
        self.k = 15 
        
        # Circular buffer write pointer and occupancy counter
        self.ptr = 0 
        self.current_size = 0

    @property
    def feature_vectors(self):
        """
        Returns the valid (populated) portion of the buffer.
        Hides zero-padding from consumers; returns empty tensor
        when no features have been stored yet.
        """
        if self.memory_buffer is None or self.current_size == 0:
            # Empty tensor so .numel() / .shape[0] work safely
            return torch.tensor([], device=self.device, dtype=self.dtype)
        
        # kNN is order-independent, so slicing [0:current_size]
        # is valid even after the buffer wraps around.
        return self.memory_buffer[:self.current_size]

    @staticmethod
    @torch.no_grad()
    def batch_evaluate_novelty_for_agents(queries, all_memories, memory_indices, agent_ks, metric='cosine'):
        """
        Calculates novelty for a batch of agents against their
        unique memories in a single GPU operation (3.3.1).

        DEVIATION(paper 3.5): Batched GPU computation.
        Paper: Describes sequential per-agent novelty evaluation.
        Code: All agents' novelty is computed in one batched
              matrix multiply for throughput on GPU.

        DEVIATION(paper 3.3.1): Default metric is cosine.
        Paper: Describes Euclidean distance for kNN.
        Code: Defaults to cosine; Euclidean available via flag.
        """
        num_agents, _ = queries.shape
        total_memory_size = all_memories.shape[0]
        device = queries.device

        if total_memory_size == 0:
            return torch.ones(num_agents, device=device)

        if metric == 'cosine':
            # Cosine similarity: higher = more similar = less novel
            queries_norm = torch.nn.functional.normalize(queries, p=2, dim=1)
            all_memories_norm = torch.nn.functional.normalize(all_memories, p=2, dim=1)
            similarity_matrix = torch.matmul(queries_norm, all_memories_norm.T)
            # largest=True selects most-similar (nearest) neighbors
            use_largest = True
        else:  # euclidean - paper-described metric (3.3.1)
            # Pairwise L2: ||a-b||² = ||a||² + ||b||² - 2a·b
            q_sq = (queries ** 2).sum(dim=1, keepdim=True)
            m_sq = (all_memories ** 2).sum(dim=1, keepdim=True)
            # Store as negative distance so largest=True
            # still selects nearest neighbors
            similarity_matrix = -(q_sq + m_sq.T - 2 * torch.matmul(queries, all_memories.T))
            use_largest = True

        agent_indices = torch.arange(num_agents, device=device).unsqueeze(1)
        memory_indices_broadcast = torch.arange(total_memory_size, device=device).unsqueeze(0)
        
        start_indices = memory_indices[:, 0].unsqueeze(1)
        end_indices = memory_indices[:, 1].unsqueeze(1)

        valid_memory_mask = (memory_indices_broadcast >= start_indices) & (memory_indices_broadcast < end_indices)
        
        masked_similarity = torch.where(valid_memory_mask, similarity_matrix, -1e9)

        max_k = agent_ks.max().item()
        
        valid_mem_sizes = memory_indices[:, 1] - memory_indices[:, 0]
        
        if not (valid_mem_sizes > 0).any():
            return torch.ones(num_agents, device=device)
        
        min_memory_size = valid_mem_sizes[valid_mem_sizes > 0].min().item() if (valid_mem_sizes > 0).any() else 1

        actual_k = min(max_k, min_memory_size)
        if actual_k <= 0:
             return torch.zeros(num_agents, device=device)

        top_k_similarities, _ = torch.topk(masked_similarity, k=actual_k, dim=1, largest=use_largest)

        k_indices = torch.arange(actual_k, device=device).unsqueeze(0)
        agent_k_mask = k_indices < agent_ks.unsqueeze(1)

        masked_top_k_similarities = torch.where(agent_k_mask, top_k_similarities, 0.0)
        sum_of_similarities = masked_top_k_similarities.sum(dim=1)
        
        # Mean of top-k similarities
        effective_k = torch.clamp(agent_ks, min=1).float()
        
        if metric == 'cosine':
            mean_similarities = sum_of_similarities / effective_k
            
            # Std dev of k-NN similarities (3.3.1: novelty uses
            # mean/std ratio for scale-invariance across agents)
            masked_top_k_sq = torch.where(agent_k_mask, top_k_similarities ** 2, 0.0)
            sum_sq = masked_top_k_sq.sum(dim=1)
            mean_sq = sum_sq / effective_k
            variance = mean_sq - mean_similarities ** 2
            std_dev = torch.sqrt(torch.clamp(variance, min=1e-8))
            
            # Cosine novelty: (1 - mean_sim) / std_dev
            # Higher when query is far from all neighbors
            raw_novelty = (1.0 - mean_similarities) / torch.clamp(std_dev, min=1e-6)
            novelty_scores = raw_novelty
        else:
            # Euclidean novelty (3.3.1):
            # mean_distance / std_dev of k-NN distances
            # Negate back to positive distances
            mean_distance = -sum_of_similarities / effective_k
            
            masked_top_k_sq = torch.where(agent_k_mask, (-top_k_similarities) ** 2, 0.0)
            sum_sq = masked_top_k_sq.sum(dim=1)
            mean_sq = sum_sq / effective_k
            variance = mean_sq - mean_distance ** 2
            std_dev = torch.sqrt(torch.clamp(variance, min=1e-8))
            
            raw_novelty = mean_distance / torch.clamp(std_dev, min=1e-6)
            novelty_scores = raw_novelty
        
        # Agents with ≤1 memory item can't compute meaningful
        # novelty - default to 1.0 (maximally novel) so they
        # accept early artifacts and bootstrap their memory.
        novelty_scores = torch.where(valid_mem_sizes <= 1, torch.ones_like(novelty_scores), novelty_scores)
        
        return novelty_scores

    @staticmethod
    @torch.no_grad()
    def batch_evaluate_novelty_for_messages(queries, message_to_agent_map, all_memories, memory_indices, agent_ks, metric='cosine'):
        """
        Calculates novelty for shared artifacts against recipient
        memories (3.4, Algorithm 1 step 4).

        Same GPU-batched approach as batch_evaluate_novelty_for_agents
        but maps each query to its recipient's memory via
        message_to_agent_map.
        """
        num_queries, _ = queries.shape
        total_memory_size = all_memories.shape[0]
        device = queries.device

        if total_memory_size == 0 or num_queries == 0:
            return torch.ones(num_queries, device=device)

        if metric == 'cosine':
            queries_norm = torch.nn.functional.normalize(queries, p=2, dim=1)
            all_memories_norm = torch.nn.functional.normalize(all_memories, p=2, dim=1)
            similarity_matrix = torch.matmul(queries_norm, all_memories_norm.T)
            use_largest = True
        else:  # euclidean
            q_sq = (queries ** 2).sum(dim=1, keepdim=True)
            m_sq = (all_memories ** 2).sum(dim=1, keepdim=True)
            similarity_matrix = -(q_sq + m_sq.T - 2 * torch.matmul(queries, all_memories.T))
            use_largest = True

        recipient_mem_indices = memory_indices[message_to_agent_map]
        start_indices = recipient_mem_indices[:, 0].unsqueeze(1)
        end_indices = recipient_mem_indices[:, 1].unsqueeze(1)

        memory_broadcast = torch.arange(total_memory_size, device=device).unsqueeze(0)
        valid_memory_mask = (memory_broadcast >= start_indices) & (memory_broadcast < end_indices)

        masked_similarity = torch.where(valid_memory_mask, similarity_matrix, -1e9)
        
        query_ks = agent_ks[message_to_agent_map]
        max_k_in_batch = query_ks.max().item()

        recipient_mem_sizes = end_indices.squeeze(1) - start_indices.squeeze(1)
        
        if not (recipient_mem_sizes > 0).any():
            return torch.ones(num_queries, device=device)
        
        min_memory_size_in_batch = recipient_mem_sizes[recipient_mem_sizes > 0].min().item() if (recipient_mem_sizes > 0).any() else 1
        
        actual_k = min(max_k_in_batch, min_memory_size_in_batch)
        if actual_k <= 0:
            return torch.zeros(num_queries, device=device)

        top_k_similarities, _ = torch.topk(masked_similarity, k=actual_k, dim=1, largest=use_largest)

        k_indices = torch.arange(actual_k, device=device).unsqueeze(0)
        agent_k_mask = k_indices < query_ks.unsqueeze(1)

        masked_top_k = torch.where(agent_k_mask, top_k_similarities, 0.0)
        sum_of_similarities = masked_top_k.sum(dim=1)

        effective_k = torch.clamp(query_ks, min=1).float()
        
        if metric == 'cosine':
            mean_similarities = sum_of_similarities / effective_k
            
            masked_top_k_sq = torch.where(agent_k_mask, top_k_similarities ** 2, 0.0)
            sum_sq = masked_top_k_sq.sum(dim=1)
            mean_sq = sum_sq / effective_k
            variance = mean_sq - mean_similarities ** 2
            std_dev = torch.sqrt(torch.clamp(variance, min=1e-8))
            
            raw_novelty = (1.0 - mean_similarities) / torch.clamp(std_dev, min=1e-6)
            novelty_scores = raw_novelty
        else:
            mean_distance = -sum_of_similarities / effective_k
            
            masked_top_k_sq = torch.where(agent_k_mask, (-top_k_similarities) ** 2, 0.0)
            sum_sq = masked_top_k_sq.sum(dim=1)
            mean_sq = sum_sq / effective_k
            variance = mean_sq - mean_distance ** 2
            std_dev = torch.sqrt(torch.clamp(variance, min=1e-8))
            
            raw_novelty = mean_distance / torch.clamp(std_dev, min=1e-6)
            novelty_scores = raw_novelty
            
        novelty_scores = torch.where(recipient_mem_sizes <= 1, torch.ones_like(novelty_scores), novelty_scores)

        return novelty_scores

    @time_it
    def add_feature_vectors(self, new_feature_vectors, step=0):
        """
        Add new feature vectors using the circular buffer.

        Overwrites oldest entries when buffer is full, keeping
        memory bounded at max_size. This is the write path
        complementing the read-only feature_vectors property.
        """
        try:
            new_feature_vectors = new_feature_vectors.to(self.device, dtype=self.dtype)
                
            # Lazy initialization of the main buffer
            if self.memory_buffer is None:
                feature_dim = new_feature_vectors.shape[1]
                self.memory_buffer = torch.zeros(
                    (self.max_size, feature_dim), 
                    device=self.device, 
                    dtype=self.dtype
                )
            
            num_new = new_feature_vectors.shape[0]
            
            # Clip if batch is larger than entire memory (edge case)
            if num_new > self.max_size:
                new_feature_vectors = new_feature_vectors[-self.max_size:]
                num_new = self.max_size

            # Calculate indices for circular buffer insertion
            start_idx = self.ptr
            end_idx = start_idx + num_new
            
            if end_idx <= self.max_size:
                # No wrapping
                self.memory_buffer[start_idx:end_idx] = new_feature_vectors
            else:
                # Wrap around
                overflow = end_idx - self.max_size
                self.memory_buffer[start_idx:] = new_feature_vectors[:-overflow]
                self.memory_buffer[:overflow] = new_feature_vectors[-overflow:]
            
            # Update pointers
            self.ptr = (self.ptr + num_new) % self.max_size
            self.current_size = min(self.current_size + num_new, self.max_size)
            
        except Exception as e:
            print(f"Agent {self.agent_id} failed to add features: {e}")