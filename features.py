"""
Feature Extraction Module
=========================

Extracts visual features from generated artifacts using a pre-trained ResNet-18.
Layer 2 is used as the feature extraction point because deeper layers (layers 3 and 4)
suffer from feature homogenization on small 32x32 inputs, where spatial resolution
collapses and distinct images produce near-identical vectors.

Supports two dimensionality modes:
- Raw (128d): Layer 2 → AdaptiveAvgPool → L2 normalize. No reduction.
- PCA (configurable): Layer 2 → AdaptiveAvgPool → PCA projection → L2 normalize.

We recommend to use PCA-16, as it retains nearly identical variance to the full 128d space
while reducing computational overhead.
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.decomposition import PCA
import numpy as np

from timing_utils import time_it

logger = logging.getLogger(__name__)

IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)


class FeatureExtractor(nn.Module):
    """
    ResNet-18 Layer 2 feature extractor with optional PCA dimensionality reduction.
    
    Args:
        output_dims: Target feature dimensions. None or 0 for raw Layer 2 (128d). 
                     Any positive int triggers PCA projection.
        use_amp: Enable automatic mixed precision on CUDA execution pathways.
        image_generator: Required if output_dims > 0. Used to generate calibration
                         artifacts for fitting the PCA projection layer.
        n_calibration: Number of random artifacts generated for PCA fitting.
    """
    def __init__(self, output_dims=None, use_amp=True, 
                 image_generator=None, n_calibration=500):
        super().__init__()
        self.use_amp = use_amp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Register ImageNet normalization constants as persistent buffers
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN_RGB, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD_RGB, dtype=torch.float32).view(1, 3, 1, 1)
        )
        
        logger.info(f"FeatureExtractor using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load pre-trained ResNet-18 model weights
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the first convolutional layer from 7x7 to 3x3 to retain low-level 
        # spatial details that would otherwise be lost on small 32x32 pixel inputs.
        pretrained_conv1 = model.conv1.weight.data.clone()  # Shape: (64, 3, 7, 7)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Interpolate pre-trained weights into the downscaled 3x3 kernel structure
        with torch.no_grad():
            model.conv1.weight.copy_(
                torch.nn.functional.interpolate(
                    pretrained_conv1, size=(3, 3),
                    mode='bilinear', align_corners=False
                )
            )
        
        # Slice backbone up to Layer 2; deeper configurations homogenize feature signals on small canvases
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.layer1,
            model.layer2
        )
        
        # Freeze network parameters to avoid unintended gradient calculation or optimization adjustments
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Global pooling to downsample spatial grids to 1x1 dimensions
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Pass a structural dummy tensor to verify and compute raw backbone output dimensionality
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32)
            dummy_out = self.pooling(self.features(dummy))
            self.backbone_dims = torch.flatten(dummy_out, 1).shape[1]
        
        logger.info(f"Backbone output: {self.backbone_dims}d")
        
        self.pca_projection = None
        self.output_dims = self.backbone_dims  
        
        # Configure PCA projection if explicit low-dimensional targets are supplied
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
        
        # Construct synthetically diverse expression nodes to populate variance space
        expressions = [
            ExpressionNode.create_random(depth=_random.randint(6, 10)) 
            for _ in range(n_samples)
        ]
        
        # Materialize visual expression structures into raw pixel buffers
        image_batch = image_generator.generate_batch(
            expressions, use_amp=self.use_amp
        )
        
        # Normalize image batches against expected ImageNet standard distributions
        normalized_batch = (image_batch - self.imagenet_mean) / self.imagenet_std
        
        # Extract features and convert back to CPU NumPy arrays for Scikit-Learn compliance
        raw_features = self._extract_raw(normalized_batch).cpu().numpy()
        
        # Execute principal component calculations across extracted samples
        pca = PCA(n_components=target_dims)
        pca.fit(raw_features)
        
        # Convert eigenvectors into frozen weight parameters inside a standard Linear block
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
        
        # Release heavy multi-dimensional graphics arrays and initialization memory tracks
        image_generator.stack_buffer = None
        image_generator.sp_buffer = None
        
        if hasattr(image_generator, '_clear_caches'):
            image_generator._clear_caches()
        
        # Explicit clean up of unused array resources
        del expressions
        del image_batch
        del normalized_batch
        del raw_features
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("PCA calibration complete, calibration buffers released")    

    def _extract_raw(self, x):
        """Extract raw pooled features from backbone, no projection or norm."""
        # Query persistent buffers to safely discover device mappings across parallel loops
        model_device = self.imagenet_mean.device
        if x.device != model_device:
            x = x.to(model_device, non_blocking=True)
        x = self.features(x)
        x = self.pooling(x)
        return torch.flatten(x, 1)
    
    @time_it
    def forward(self, x):
        """
        Extract features from a batch of normalized image tensors.
        
        Pipeline:
            ResNet Layer 2 → AdaptiveAvgPool → [PCA Projection] → L2 Normalization
        
        Args:
            x: Tensor of shape (B, 3, 32, 32), ImageNet-normalized.
        
        Returns:
            Tensor of shape (B, output_dims), L2-normalized.
        """
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16,
            enabled=self.use_amp and self.device.type == 'cuda'
        ):
            # Pass image tensor down the backbone network layers
            x = self._extract_raw(x)
            
            # Route vectors through projection weights if downscaling is enabled
            if self.pca_projection is not None:
                x = self.pca_projection(x)
            
            # Apply standard L2 normalization across feature dimensions
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x.to(torch.float32)
    
    def get_memory_usage(self):
        """Returns VRAM footprint allocations if running on CUDA."""
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(0) // 1024 ** 2,
                "reserved": torch.cuda.memory_reserved(0) // 1024 ** 2,
            }
        return {"allocated": 0, "reserved": 0}