"""
Wundt Curve Modeling
====================

Implements the psychological Wundt Curve modeling engine utilizing PyTorch
to parallelize multi-agent evaluation passes directly on the GPU.
"""

import torch
import numpy as np
from timing_utils import time_it

class WundtCurve:
    """
    Models the Wundt Curve for calculating interest from novelty.
    Supports individual scalar evaluations and batched tensor operations.
    """
    def __init__(self, reward_mean=0.3, reward_std=0.15,
                 punish_mean=0.7, punish_std=0.15, alpha=1.2):
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.punish_mean = punish_mean
        self.punish_std = punish_std
        self.alpha = alpha
    
    @time_it
    def hedonic_value(self, x: float) -> float:
        """
        Computes the hedonic value for a single scalar novelty score.
        """
        sqrt_2 = 1.4142135623730951
        with torch.no_grad():
            t_x = torch.tensor(x, dtype=torch.float32)
            r = 0.5 * (1 + torch.erf((t_x - self.reward_mean) / (self.reward_std * sqrt_2)))
            p = 0.5 * (1 + torch.erf((t_x - self.punish_mean) / (self.punish_std * sqrt_2)))
            h = r - self.alpha * p
            return float(torch.clamp(h, -1.0, 1.0).item())

    @staticmethod
    @torch.no_grad()
    @time_it
    def batch_hedonic_value(
        x: torch.Tensor,
        reward_means: torch.Tensor,
        reward_stds: torch.Tensor,
        punish_means: torch.Tensor,
        punish_stds: torch.Tensor,
        alphas: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes net hedonic values for a batch of novelty scores.
        All input tensors must share matching dimensions and device targets.
        """
        sqrt_2 = 1.4142135623730951
        
        r = 0.5 * (1.0 + torch.erf((x - reward_means) / (reward_stds * sqrt_2)))
        p = 0.5 * (1.0 + torch.erf((x - punish_means) / (punish_stds * sqrt_2)))
        
        h = r - alphas * p
        return torch.clamp(h, -1.0, 1.0)
    
    @time_it
    def find_peak_novelty(self):
        """
        Finds the novelty value that maximizes interest.
        """
        x = np.linspace(0, 1, 1000)
        h = [self.hedonic_value(xi) for xi in x]
        return x[np.argmax(h)]