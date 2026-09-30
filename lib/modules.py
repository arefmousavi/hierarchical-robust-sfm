"""Convex layer fusion, linear classifier, and the frozen-backbone task model.

    z_alpha(x) = sum_l alpha_l q_l(x),   alpha = softmax(w^alpha)
    g(z)       = W z + b

Clean adaptation trains ``(alpha, W, b)``. Margin refinement freezes ``alpha``
and moves only the linear head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import normalize_pooled, pool_frames
from .models import SpeechFoundationBackbone


class ConvexLayerFusion(nn.Module):
    """``sum_l softmax(logits)_l * q_l``. Softmax enforces ``alpha >= 0``, ``sum alpha = 1``."""

    def __init__(self, num_layers: int, init: torch.Tensor | None = None):
        super().__init__()
        self.num_layers = int(num_layers)
        start = torch.zeros(self.num_layers) if init is None else init.clone().float()
        if start.numel() != self.num_layers:
            raise ValueError(f"fusion init has {start.numel()} entries for {self.num_layers} layers")
        self.logits = nn.Parameter(start)

    def weights(self) -> torch.Tensor:
        """Convex coefficients ``alpha_l = softmax(w^alpha)_l``.

        Softmax over finite logits gives ``alpha_l > 0`` and ``sum_l alpha_l = 1``
        exactly, which is the layer-fusion constraint in the paper.
        """
        return F.softmax(self.logits, dim=0)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.size(0) != self.num_layers:
            raise ValueError(f"fusion expected {self.num_layers} layers, got {q.size(0)}")
        alpha = self.weights().to(dtype=q.dtype)
        if q.ndim == 3:  # [L, B, D]
            return (q * alpha[:, None, None]).sum(dim=0)
        if q.ndim == 4:  # [L, B, T, D]
            return (q * alpha[:, None, None, None]).sum(dim=0)
        raise ValueError(f"fusion expected 3D or 4D q, got {tuple(q.shape)}")

    @torch.no_grad()
    def report(self) -> list[float]:
        return [float(v) for v in self.weights().detach().cpu()]


class NormalizedLinearClassifier(nn.Module):
    """Linear readout ``W z + b`` on the already-normalised fused coordinates ``z_alpha``."""

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(int(hidden_size), int(num_classes))
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter:
        return self.linear.bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


@dataclass
class HeadState:
    fusion_logits: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor


class FullTaskModel(nn.Module):
    """Frozen SFM -> Eq. (3) coordinates -> convex fusion -> linear classifier.

    Used for the four utterance-level downstream tasks.
    """

    def __init__(
        self,
        backbone: SpeechFoundationBackbone,
        num_classes: int,
        sigma: torch.Tensor,
        mu: torch.Tensor | None = None,
        mode: str = "pooled",
    ):
        super().__init__()
        if mode != "pooled":
            raise ValueError(f"Only pooled utterance classification is supported, got mode={mode!r}")
        self.backbone = backbone
        self.mode = mode
        self.num_classes = int(num_classes)
        self.register_buffer("sigma", sigma.clone())
        if mu is None:
            self.register_buffer("mu", torch.zeros(backbone.num_layers, backbone.hidden_size))
            self.has_mu = False
        else:
            self.register_buffer("mu", mu.clone())
            self.has_mu = True
        self.fusion = ConvexLayerFusion(backbone.num_layers)
        self.head = NormalizedLinearClassifier(backbone.hidden_size, num_classes)
        self.backbone.freeze()

    @property
    def hidden_size(self) -> int:
        return self.backbone.hidden_size

    @property
    def num_layers(self) -> int:
        return self.backbone.num_layers

    def pooled_states(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        states, frame_mask = self.backbone.transformer_states(waveforms, lengths)
        return pool_frames(states, frame_mask)

    def layer_features(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Normalised pooled per-layer coordinates ``q_l(x)`` with shape ``[L, B, D]``."""
        states, frame_mask = self.backbone.transformer_states(waveforms, lengths)
        mu = self.mu if self.has_mu else None
        pooled = pool_frames(states, frame_mask)
        return normalize_pooled(pooled, self.sigma, mu, self.hidden_size)

    def fused_features(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.layer_features(waveforms, lengths))

    def logits_from_layers(self, q: torch.Tensor) -> torch.Tensor:
        return self.head(self.fusion(q))

    def logits_from_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.fused_features(waveforms, lengths))

    def freeze_fusion(self) -> "FullTaskModel":
        self.fusion.logits.requires_grad_(False)
        return self

    def trainable_head_params(self) -> list[nn.Parameter]:
        return [self.head.weight, self.head.bias]

    def state(self) -> HeadState:
        return HeadState(
            self.fusion.logits.detach().clone().cpu(),
            self.head.weight.detach().clone().cpu(),
            self.head.bias.detach().clone().cpu(),
        )

    @torch.no_grad()
    def install(self, state: HeadState) -> None:
        device = self.head.weight.device
        self.fusion.logits.copy_(state.fusion_logits.to(device))
        self.head.weight.copy_(state.weight.to(device))
        self.head.bias.copy_(state.bias.to(device))

    def export_head(self) -> dict:
        return {
            "fusion_logits": self.fusion.logits.detach().cpu(),
            "weight": self.head.weight.detach().cpu(),
            "bias": self.head.bias.detach().cpu(),
            "sigma": self.sigma.detach().cpu(),
            "mu": self.mu.detach().cpu() if self.has_mu else None,
            "has_mu": self.has_mu,
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "mode": self.mode,
            "alpha": self.fusion.report(),
        }
