"""Stage 1 / Foundation-AFT: hierarchical backbone robustification.

    min_theta  E_x [ max_{delta in Delta_S(x)} LSE_beta(d)  +  lambda_1 L_pres(x) ]

``sigma_l`` is estimated on the calibration split if ``sigma_path`` is missing.
The inner attack is PGD-K on ``LSE_beta`` inside the L2 SNR ball; ``L_pres``
is evaluated outside that loop against a frozen pretrained copy ``theta_b``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.attacks import FoundationHierarchicalPGD
from lib.datasets import build_foundation_loader
from lib.metrics import estimate_sigma, lse_beta, preservation_loss, representation_distance_per_layer
from lib.models import build_backbone, transformer_states
from lib.utils import linear_warmup_decay, load_config, resolve_device, resolve_dtype, save_json, set_seed, tensor_fingerprint


def select_encoder(cfg: dict) -> str:
    """Resolve the selected backbone from config/foundation.yaml into cfg["encoder"]."""
    model_cfg = cfg.get("model") or {}
    selected = str(model_cfg.get("selected", "")).lower()
    choices = model_cfg.get("choices") or {}
    if selected not in choices:
        raise ValueError(
            f"model.selected must be one of {sorted(choices)}, got {selected!r}"
        )
    cfg["encoder"] = dict(choices[selected])
    sigma_path = str(cfg["foundation"]["sigma_path"]).replace("{model}", selected)
    cfg["foundation"]["sigma_path"] = sigma_path
    cfg["train"]["output_dir"] = str(cfg["train"]["output_dir"]).replace("{model}", selected)
    cfg["model"]["selected"] = selected
    return selected


def maybe_estimate_sigma(cfg: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    path = Path(cfg["foundation"]["sigma_path"])
    metadata_path = path.with_suffix(".json")
    if path.is_file():
        if not metadata_path.is_file():
            raise RuntimeError(
                f"Calibration metadata missing for {path}. Move the legacy sigma file "
                "aside and rerun Stage 1 to calibrate it from the configured split."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "encoder_type": cfg["encoder"]["type"].lower(),
            "pretrained_name": cfg["encoder"]["pretrained_name"],
            "local_path": cfg["encoder"].get("local_path"),
            "padding_mode": cfg["encoder"].get("padding_mode", "reference"),
            "sigma_manifest": cfg["data"]["sigma_manifest"],
            "data_root": cfg["data"]["root"],
            "sample_rate": int(cfg["data"].get("sample_rate", 16000)),
            "max_len_sec": float(cfg["data"].get("max_len_sec", 10.0)),
            "sigma_samples": cfg["foundation"].get("sigma_samples"),
            "dtype": str(dtype).split(".")[-1],
        }
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Sigma metadata mismatch at {path}: expected {expected}, got {metadata}")
        sigma = torch.load(path, map_location=device, weights_only=False)
        if tensor_fingerprint(sigma) != metadata.get("sigma_sha256"):
            raise ValueError(f"Sigma checksum mismatch at {path}")
        print(f"Reusing sigma_l from {path}  shape={tuple(sigma.shape)}")
        return sigma.to(device=device, dtype=dtype)
    print(f"sigma_path missing ({path}); estimating on the calibration split")
    data = cfg["data"]
    foundation = cfg["foundation"]
    loader = build_foundation_loader(cfg, "sigma_manifest", "sigma_samples", False)
    encoder_cfg = cfg["encoder"]
    probe = build_backbone(
        encoder_cfg["type"],
        encoder_cfg["pretrained_name"],
        local_path=encoder_cfg.get("local_path"),
        padding_mode=encoder_cfg.get("padding_mode", "reference"),
        freeze=True,
    ).to(device=device, dtype=dtype)
    sigma = estimate_sigma(probe, loader, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sigma.cpu(), path)
    save_json({
        "encoder_type": encoder_cfg["type"].lower(),
        "pretrained_name": encoder_cfg["pretrained_name"],
        "local_path": encoder_cfg.get("local_path"),
        "padding_mode": encoder_cfg.get("padding_mode", "reference"),
        "sigma_manifest": data["sigma_manifest"],
        "data_root": data["root"],
        "sample_rate": int(data.get("sample_rate", 16000)),
        "max_len_sec": float(data.get("max_len_sec", 10.0)),
        "sigma_samples": foundation.get("sigma_samples"),
        "sigma_sha256": tensor_fingerprint(sigma),
        "dtype": str(dtype).split(".")[-1],
    }, metadata_path)
    print(f"Saved {sigma.numel()} sigma_l values to {path.resolve()}")
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sigma.to(device=device, dtype=dtype)


def hierarchical_losses(adv_states, clean_states, base_states, sigma, mask, beta):
    distances = representation_distance_per_layer(adv_states, clean_states, sigma, mask)
    adv_loss = lse_beta(distances, beta).mean()
    pres_loss = preservation_loss(clean_states, base_states, sigma, mask)
    return adv_loss, pres_loss, distances


@torch.no_grad()
def evaluate_preservation(robust, base, loader, sigma, device, dtype):
    """Clean validation preservation loss."""
    total, count = 0.0, 0
    was_training = robust.training
    robust.eval()
    try:
        for batch in tqdm(loader, desc="foundation clean val", leave=False):
            x = batch["waveforms"].to(device=device, dtype=dtype)
            lengths = batch["lengths"].to(device)
            clean, mask = transformer_states(robust, x, lengths)
            base_clean, _ = transformer_states(base, x, lengths)
            loss = preservation_loss(clean, base_clean, sigma, mask)
            total += float(loss.cpu()) * x.size(0)
            count += x.size(0)
    finally:
        robust.train(was_training)
    return total / max(1, count)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    selected_model = select_encoder(cfg)
    print(f"Selected backbone: {selected_model} -> {cfg['encoder']['pretrained_name']}")
    set_seed(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision", "high"))
    device = resolve_device(cfg.get("device", "cuda"))
    dtype = resolve_dtype(cfg.get("dtype", "float32"))
    print(f"Device: {device}  dtype: {dtype}")

    sigma = maybe_estimate_sigma(cfg, device, dtype)
    encoder_cfg = cfg["encoder"]
    robust = build_backbone(
        encoder_cfg["type"],
        encoder_cfg["pretrained_name"],
        local_path=encoder_cfg.get("local_path"),
        padding_mode=encoder_cfg.get("padding_mode", "reference"),
        checkpoint=args.resume,
        freeze=False,
    ).to(device=device, dtype=dtype)
    base = build_backbone(
        encoder_cfg["type"],
        encoder_cfg["pretrained_name"],
        local_path=encoder_cfg.get("local_path"),
        padding_mode=encoder_cfg.get("padding_mode", "reference"),
        freeze=True,
    ).to(device=device, dtype=dtype)
    if sigma.ndim != 1 or sigma.numel() != robust.num_layers:
        raise ValueError(f"Expected {robust.num_layers} sigma values, got {tuple(sigma.shape)}")
    foundation_meta = {
        "selected_model": selected_model,
        "encoder_type": encoder_cfg["type"].lower(),
        "pretrained_name": encoder_cfg["pretrained_name"],
        "local_path": encoder_cfg.get("local_path"),
        "padding_mode": encoder_cfg.get("padding_mode", "reference"),
        "sigma_sha256": tensor_fingerprint(sigma),
        "dtype": str(dtype).split(".")[-1],
        "sigma_manifest": cfg["data"]["sigma_manifest"],
        "train_manifest": cfg["data"]["train_manifest"],
        "data_root": cfg["data"]["root"],
        "sample_rate": int(cfg["data"].get("sample_rate", 16000)),
        "max_len_sec": float(cfg["data"].get("max_len_sec", 10.0)),
    }
    if args.resume:
        resume_meta_path = Path(args.resume) / "foundation_meta.json"
        if not resume_meta_path.is_file():
            raise FileNotFoundError(f"Resume metadata missing: {resume_meta_path}")
        resume_meta = json.loads(resume_meta_path.read_text(encoding="utf-8"))
        if resume_meta != foundation_meta:
            raise ValueError(
                f"Resume checkpoint does not match this backbone, data, and sigma: {args.resume}"
            )
    if cfg["train"].get("gradient_checkpointing", False):
        robust.enable_gradient_checkpointing()

    train_loader = build_foundation_loader(cfg, "train_manifest", "train_samples", True)
    val_loader = build_foundation_loader(cfg, "val_manifest", "val_samples", False)

    train_cfg = cfg["train"]
    attack_cfg = train_cfg["attack"]
    beta = float(cfg["foundation"].get("beta", 30.0))
    lambda_pres = float(cfg["foundation"].get("lambda_pres", 0.2))
    print(
        f"STAGE 1 / Eq. (6): L={robust.num_layers}  beta={beta:g}  lambda_1={lambda_pres:g}  "
        f"S={float(attack_cfg.get('target_snr', 30)):g} dB  "
        f"PGD steps={int(attack_cfg.get('steps', 10))}  "
        f"(L_pres is outside the inner max)"
    )
    attacker = FoundationHierarchicalPGD(
        steps=int(attack_cfg.get("steps", 10)),
        alpha=float(attack_cfg.get("alpha", 0.25)),
        target_snr=float(attack_cfg.get("target_snr", 30)),
        beta=beta,
        momentum=float(attack_cfg.get("momentum", 0.0)),
    )
    optimizer = torch.optim.Adam(
        robust.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    accumulation = int(train_cfg.get("grad_accumulation_steps", 1))
    updates = math.ceil(len(train_loader) / accumulation) * int(train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: linear_warmup_decay(
            step, updates, float(train_cfg.get("warmup_ratio", 0.1))
        ),
    )

    output = Path(train_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    global_step = 0
    start_epoch = 0
    if args.resume:
        state_path = Path(args.resume) / "training_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"])
            global_step = int(state["global_step"])
            history_path = output / "history.json"
            if history_path.is_file():
                history = json.loads(history_path.read_text())
            print(f"Resuming Foundation-AFT after epoch {start_epoch}")

    save_json(cfg, output / "run_config.json")

    for epoch in range(start_epoch + 1, int(train_cfg["epochs"]) + 1):
        robust.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"adv": 0.0, "pres": 0.0, "count": 0}
        progress = tqdm(train_loader, desc=f"foundation epoch {epoch}")
        for batch_index, batch in enumerate(progress, 1):
            x = batch["waveforms"].to(device=device, dtype=dtype)
            lengths = batch["lengths"].to(device)
            x_adv = attacker.attack(robust, x, lengths, sigma)
            clean, mask = transformer_states(robust, x, lengths)
            adv, adv_mask = transformer_states(robust, x_adv, lengths)
            with torch.no_grad():
                base_clean, _ = transformer_states(base, x, lengths)
            if not torch.equal(mask, adv_mask):
                raise RuntimeError("Clean/adversarial frame masks differ")
            adv_loss, pres_loss, _ = hierarchical_losses(
                adv, clean, base_clean, sigma, mask, beta
            )
            loss = adv_loss + lambda_pres * pres_loss
            group_start = ((batch_index - 1) // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            (loss / group_size).backward()
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    robust.parameters(), float(train_cfg.get("max_grad_norm", 1.0))
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            count = x.size(0)
            totals["adv"] += float(adv_loss.detach().cpu()) * count
            totals["pres"] += float(pres_loss.detach().cpu()) * count
            totals["count"] += count
            progress.set_postfix(
                L_adv=f"{float(adv_loss.detach()):.4f}",
                L_pres=f"{float(pres_loss.detach()):.4f}",
            )

        n = max(1, totals["count"])
        val_pres = evaluate_preservation(robust, base, val_loader, sigma, device, dtype)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_L_adv": totals["adv"] / n,
            "train_L_pres": totals["pres"] / n,
            "val_L_pres": val_pres,
        }
        history.append(record)
        print(record)

        checkpoint = output / f"checkpoint-epoch-{epoch}"
        robust.save_pretrained(checkpoint)
        save_json(foundation_meta, checkpoint / "foundation_meta.json")
        torch.save(sigma.detach().cpu(), checkpoint / "sigma.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint / "training_state.pt",
        )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    final_dir = output / "theta_r"
    robust.save_pretrained(final_dir)
    save_json(foundation_meta, final_dir / "foundation_meta.json")
    torch.save(sigma.detach().cpu(), final_dir / "sigma.pt")
    print(f"Saved robust backbone theta_r to {final_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Stage-1 robustification")
    parser.add_argument("--config", "-c", required=True, help="config/foundation.yaml")
    parser.add_argument("--resume", help="Foundation checkpoint directory to resume")
    main(parser.parse_args())
