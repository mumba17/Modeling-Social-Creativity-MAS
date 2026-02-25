"""
Feature Extraction Module (3.3.2)
==================================

Extracts visual features from generated artifacts using a pre-trained ResNet-18.
Layer 2 is used as the feature extraction point because deeper layers
(3, 4) suffer from feature homogenization on 32x32 inputs - the spatial
resolution collapses and distinct images produce near-identical vectors,
destroying novelty signal (3.3.2).

Supports two dimensionality modes:
- Raw (128d): Layer 2 → AdaptiveAvgPool → L2 normalize. No reduction.
- PCA (e.g. 64d): Same backbone, then PCA projection fitted on a
  calibration batch of random artifacts at startup.

DEVIATION(paper 3.3.2): PCA reduction is an optional extension.
Paper: Experiments used 64d PCA; code defaults to raw 128d.
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.decomposition import PCA
import numpy as np

from timing_utils import time_it

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """
    ResNet-18 Layer 2 feature extractor with optional PCA
    dimensionality reduction (3.3.2).
    
    Args:
        output_dims: Target feature dimensions. None or 0 for raw
                     Layer 2 (128d). Any positive int triggers PCA.
        use_amp: Enable automatic mixed precision on CUDA.
        image_generator: Required if output_dims > 0. Used to
                         generate calibration artifacts for PCA.
        n_calibration: Number of random artifacts for PCA fitting.
    """
    def __init__(self, output_dims=None, use_amp=True, 
                 image_generator=None, n_calibration=500):
        super().__init__()
        self.use_amp = use_amp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"FeatureExtractor using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # --- ResNet-18 backbone up to Layer 2 (3.3.2) ---
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Paper (3.3.2): 3x3 kernel, stride=1 replaces default
        # 7x7 conv1 to preserve spatial detail at 32x32 input.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.layer1,
            # Paper (3.3.2): Layer 2 is optimal - deeper layers
            # homogenize features on small images.
            model.layer2
        )
        
        # Freeze all backbone parameters - we never train this
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determine raw backbone output dimensionality
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32)
            dummy_out = self.pooling(self.features(dummy))
            self.backbone_dims = torch.flatten(dummy_out, 1).shape[1]
        
        logger.info(f"Backbone output: {self.backbone_dims}d")
        
        # --- Optional PCA reduction ---
        # DEVIATION(paper 3.3.2): PCA is an optional code feature.
        # Paper experiments used 64d PCA; code defaults to raw
        # 128d (output_dims=0) unless explicitly configured.
        self.pca_projection = None
        self.output_dims = self.backbone_dims  # Default: raw
        
        if output_dims and output_dims > 0 and output_dims < self.backbone_dims:
            if image_generator is None:
                raise ValueError(
                    f"image_generator is required for PCA reduction "
                    f"(feature_dims={output_dims}). Pass the VectorizedImageGenerator "
                    f"instance, or set feature_dims=0 for raw features."
                )
            self.output_dims = output_dims
            self.to(self.device)
            self.eval()
            self._fit_pca(image_generator, n_calibration, output_dims)
        else:
            if output_dims and output_dims > 0:
                logger.warning(
                    f"Requested {output_dims}d but backbone only produces "
                    f"{self.backbone_dims}d. Using raw features."
                )
            self.to(self.device)
            self.eval()
        
        logger.info(
            f"Final feature dimensionality: {self.output_dims}d"
            f"{' (PCA)' if self.pca_projection is not None else ' (raw)'}"
        )
    
    @torch.no_grad()
    def _fit_pca(self, image_generator, n_samples, target_dims):
        """
        Fit PCA on random artifacts to derive a projection matrix.
        
        Generates n_samples random expression trees, renders them,
        extracts raw backbone features, then fits sklearn PCA.
        The resulting principal components are stored as a frozen
        nn.Linear layer for GPU-accelerated projection in forward().
        
        Deterministic given the random seed set at simulation start.
        """
        from genart import ExpressionNode
        import random as _random
        
        logger.info(f"Fitting PCA: generating {n_samples} calibration artifacts...")
        
        # Generate diverse random expressions across the full depth range
        expressions = [
            ExpressionNode.create_random(depth=_random.randint(6, 10)) 
            for _ in range(n_samples)
        ]
        
        # Render to images
        image_batch = image_generator.generate_batch(
            expressions, use_amp=self.use_amp
        )
        
        # Apply ImageNet normalization (same as scheduler.evaluation_phase)
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        ).view(1, 3, 1, 1)
        normalized_batch = (image_batch - mean) / std
        
        # Extract raw backbone features (no PCA yet)
        raw_features = self._extract_raw(normalized_batch).cpu().numpy()
        
        # Fit PCA
        pca = PCA(n_components=target_dims)
        pca.fit(raw_features)
        
        # Store as a frozen linear layer: output = input @ components.T
        self.pca_projection = nn.Linear(
            raw_features.shape[1], target_dims, bias=False
        )
        self.pca_projection.weight.data = torch.tensor(
            pca.components_, dtype=torch.float32
        )
        self.pca_projection.weight.requires_grad = False
        self.pca_projection.to(self.device)
        
        variance_retained = pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA fitted: {raw_features.shape[1]}d → {target_dims}d, "
            f"variance retained: {variance_retained:.1%}"
        )
    
    def _extract_raw(self, x):
        """Extract raw pooled features from backbone, no projection or norm."""
        x = x.to(self.device)
        x = self.features(x)
        x = self.pooling(x)
        return torch.flatten(x, 1)
    
    @time_it
    def forward(self, x):
        """
        Extract features from a batch of normalized image tensors.
        
        Pipeline (3.3.2):
            ResNet Layer 2 → AdaptiveAvgPool → [PCA] → L2 norm
        
        DEVIATION(paper 3.3.2): L2 normalization here combined
        with cosine distance in kNN causes double normalization.
        Paper: Mentions L2 norm for feature vectors.
        Code: L2 norm applied here AND cosine metric re-normalizes
              in kNN.batch_evaluate_novelty_*. Harmless with cosine
              (result is identical) but worth noting.
        
        Args:
            x: Tensor of shape (B, 3, 32, 32), ImageNet-normalized.
        
        Returns:
            Tensor of shape (B, output_dims), L2-normalized.
        """
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16,
            enabled=self.use_amp and self.device.type == 'cuda'
        ):
            x = self._extract_raw(x)
            
            if self.pca_projection is not None:
                x = self.pca_projection(x)
            
            # L2 normalization (paper: consistent distance calculations)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x.to(torch.float32)
    
    def get_memory_usage(self):
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(0) // 1024 ** 2,
                "reserved": torch.cuda.memory_reserved(0) // 1024 ** 2,
            }
        return {"allocated": 0, "reserved": 0}