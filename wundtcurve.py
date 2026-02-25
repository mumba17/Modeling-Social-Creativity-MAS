"""
Wundt Curve Modeling (3.3.4)
=============================

This module implements the Wundt Curve, a psychological model describing the
relationship between a stimulus's intensity (novelty) and its hedonic value
(interest). It models interest as the difference between a reward process
and a punishment process, both modeled as cumulative Gaussians.

Paper formula (3.3.4):
    H(x) = R(x) - α·P(x)
    where R, P are cumulative Gaussian CDFs.
"""

import numpy as np
from scipy.special import erf
from timing_utils import time_it

class WundtCurve:
    """
    Models the Wundt Curve for calculating interest from novelty.

    The curve is defined by two cumulative Gaussian functions: one for reward
    (increasing with novelty) and one for punishment (increasing with novelty,
    usually later/slower). The net interest is Reward - alpha * Punishment.

    Paper defaults (3.3.4):
        reward_mean  = max(0.1, p - 0.2)  where p = preferred_novelty
        punish_mean  = min(0.9, p + 0.2)
        reward_std   = punish_std = 0.15
        alpha        = 1.2

    Attributes:
        reward_mean (float): Mean of the reward Gaussian.
        reward_std (float): Standard deviation of the reward Gaussian.
        punish_mean (float): Mean of the punishment Gaussian.
        punish_std (float): Standard deviation of the punishment Gaussian.
        alpha (float): Weighting factor for the punishment component.
    """
    # DEVIATION(paper 3.3.4): Constructor defaults differ from paper.
    # Paper: reward_std=0.15, punish_std=0.15, alpha=1.2.
    # Code: defaults here are 0.1/0.1/1.0, but the scheduler
    #       overrides with paper-correct values during agent init.
    def __init__(self, reward_mean=0.3, reward_std=0.1, 
                 punish_mean=0.7, punish_std=0.1, alpha=1):
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.punish_mean = punish_mean
        self.punish_std = punish_std
        self.alpha = alpha

    @time_it
    def cumulative_gaussian(self, x, mean, std):
        """
        Computes the cumulative Gaussian function (3.3.4).
        Formula: F(x|mean,std) = 1/2[1 + erf((x-mean)/(std√2))]
        """
        return 0.5 * (1 + erf((x - mean)/(std * np.sqrt(2))))

    def reward(self, x):
        """Computes the reward component R(x)."""
        return self.cumulative_gaussian(x, self.reward_mean, self.reward_std)

    def punishment(self, x):
        """Computes the punishment component P(x)."""
        return self.cumulative_gaussian(x, self.punish_mean, self.punish_std)

    @time_it
    def hedonic_value(self, x, experience=None, maturity_threshold=15):
        """
        Computes the net hedonic value (interest) (3.3.4).
        H(x) = R(x) - αP(x), clipped to [0, 1].

        If `experience` is provided (e.g., agent's memory size), the punishment 
        weight (alpha) is annealed. When experience is 0, alpha is 0 (high tolerance 
        for extreme novelty). As experience reaches maturity_threshold, alpha 
        returns to its full baseline value.

        DEVIATION(paper 3.3.4): Experience-based alpha annealing.
        Paper: alpha is a fixed constant (1.2) at all times.
        Code: alpha scales from 0 → baseline as agent memory
              grows, so new agents tolerate extreme novelty.
        """
        r = self.reward(x)
        p = self.punishment(x)
        
        # DEVIATION(paper 3.3.4): Annealing not in paper.
        # Scales punishment weight by agent maturity so fresh
        # agents explore freely before settling into preferences.
        current_alpha = self.alpha
        if experience is not None:
            maturity = min(1.0, experience / maturity_threshold)
            current_alpha = self.alpha * maturity
            
        h = r - current_alpha * p
        return float(np.clip(h, 0.0, 1.0))

    def find_peak_novelty(self):
        """
        Finds the novelty value that maximizes interest.
        """
        x = np.linspace(0, 1, 1000)
        h = [self.hedonic_value(xi) for xi in x]
        return x[np.argmax(h)]