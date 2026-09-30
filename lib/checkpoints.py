"""Load the bundled Stage-1 checkpoint used by downstream adaptation."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .utils import resolve_dtype, tensor_fingerprint


def resolve_foundation_checkpoint(
    backbone_checkpoint: str | Path,
    cfg: dict,
) -> tuple[dict, dict, torch.Tensor, torch.dtype]:
    checkpoint_dir = Path(backbone_checkpoint).resolve()
    meta_path = checkpoint_dir / "foundation_meta.json"
    sigma_path = checkpoint_dir / "sigma.pt"
    if not meta_path.is_file() or not sigma_path.is_file():
        raise FileNotFoundError(
            f"Stage-1 checkpoint must contain foundation_meta.json and sigma.pt: {checkpoint_dir}"
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    sigma = torch.load(sigma_path, map_location="cpu", weights_only=False)
    if tensor_fingerprint(sigma) != metadata.get("sigma_sha256"):
        raise ValueError(f"Stage-1 sigma checksum mismatch: {sigma_path}")

    encoder = {
        "type": metadata["encoder_type"],
        "pretrained_name": metadata["pretrained_name"],
        "local_path": metadata.get("local_path"),
        "padding_mode": metadata.get("padding_mode", "reference"),
    }
    dtype = resolve_dtype(cfg.get("dtype", metadata.get("dtype", "float32")))
    return metadata, encoder, sigma, dtype
