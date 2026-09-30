"""Config loading, reproducibility, and L2 SNR threat-set helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse as a mapping")
    repo_root = Path(path).resolve().parent.parent
    path_keys = {
        "root", "local_path", "train_manifest", "val_manifest",
        "test_manifest", "sigma_manifest", "sigma_path", "mu_path",
        "output_dir", "foundation_checkpoint",
    }
    def resolve(value, key=""):
        if isinstance(value, dict):
            return {k: resolve(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v, key) for v in value]
        if not isinstance(value, str):
            return value
        if key in path_keys and value:
            if "${" in value:
                raise ValueError(
                    f"Config {path} contains an environment-variable placeholder in {key!r}. "
                    "Set dataset and checkpoint paths directly in the YAML file."
                )
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            return str(candidate.resolve())
        return value

    cfg = resolve(cfg)
    checkpoint = cfg.get("foundation_checkpoint")
    if isinstance(checkpoint, str) and "{model}" in checkpoint:
        selected = str(cfg.get("foundation_model", "")).lower()
        if selected not in {"wav2vec2", "hubert", "wavlm"}:
            raise ValueError(
                "A foundation_checkpoint with {model} requires foundation_model "
                "to be wav2vec2, hubert, or wavlm"
            )
        cfg["foundation_model"] = selected
        cfg["foundation_checkpoint"] = checkpoint.replace("{model}", selected)
    return cfg


def tensor_fingerprint(value: torch.Tensor) -> str:
    """Stable SHA-256 identity for a saved numeric calibration tensor."""
    data = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def resolve_dtype(name: str | None = "float32") -> torch.dtype:
    value = (name or "float32").lower()
    if value in {"float32", "fp32"}:
        return torch.float32
    if value in {"float64", "double", "fp64"}:
        return torch.float64
    raise ValueError(f"Unsupported training dtype: {name!r}; use float32 or float64")


def resolve_device(name: str | None = "cuda") -> torch.device:
    requested = (name or "cuda").lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested in the config, but no CUDA device is available")
        return torch.device(requested)
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {name!r}")


def save_json(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def waveform_mask(waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Boolean mask over valid (unpadded) waveform samples. ``[B, T]``."""
    return torch.arange(waveforms.size(1), device=waveforms.device).unsqueeze(0) < lengths.unsqueeze(1)


def l2_snr_radii(waveforms: torch.Tensor, lengths: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Per-sample L2 radius ``||x||_2 * 10^(-S/20)``, using unpadded samples only."""
    valid = waveform_mask(waveforms, lengths)
    signal_norm = torch.sqrt(
        (waveforms.square() * valid).sum(dim=1).clamp_min(1e-12)
    )
    return signal_norm * (10.0 ** (-float(snr_db) / 20.0))


def project_l2_snr(
    adversarial: torch.Tensor,
    clean: torch.Tensor,
    lengths: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Project onto the intersection of the L2 SNR ball and ``[-1, 1]``."""
    valid = waveform_mask(clean, lengths)
    delta = torch.where(valid, adversarial - clean, torch.zeros_like(clean))
    delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(delta_norm), radii.view(-1, 1) / delta_norm)
    return torch.where(valid, (clean + delta * scale).clamp(-1.0, 1.0), clean)


def linear_warmup_decay(step: int, total: int, warmup_frac: float = 0.1) -> float:
    total = max(1, int(total))
    warmup = max(1, int(round(warmup_frac * total)))
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))
