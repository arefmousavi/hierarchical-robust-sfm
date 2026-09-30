"""Shared library for hierarchical robustification of speech foundation models."""

from .utils import load_config, set_seed, resolve_device, l2_snr_radii, project_l2_snr
from .models import SpeechFoundationBackbone, build_backbone, transformer_states
from .modules import ConvexLayerFusion, NormalizedLinearClassifier, FullTaskModel
from .metrics import (
    representation_distance_per_layer,
    lse_beta,
    preservation_loss,
    estimate_sigma,
    pool_frames,
    normalize_pooled,
    pairwise_margins,
    soft_margin,
    normalized_margin,
)
from .attacks import FoundationHierarchicalPGD
from .datasets import build_classification_loaders, build_foundation_loader

__all__ = [
    "load_config",
    "set_seed",
    "resolve_device",
    "l2_snr_radii",
    "project_l2_snr",
    "SpeechFoundationBackbone",
    "build_backbone",
    "transformer_states",
    "ConvexLayerFusion",
    "NormalizedLinearClassifier",
    "FullTaskModel",
    "representation_distance_per_layer",
    "lse_beta",
    "preservation_loss",
    "estimate_sigma",
    "pool_frames",
    "normalize_pooled",
    "pairwise_margins",
    "soft_margin",
    "normalized_margin",
    "FoundationHierarchicalPGD",
    "build_classification_loaders",
    "build_foundation_loader",
]
