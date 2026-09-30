"""Downstream adaptation: clean task fit of (alpha, W, b), then head-only margin refinement.

Phase 1 (Eq. 8), backbone frozen:

    min_{alpha, W, b}  E_{(x,y)}  CE( W z_alpha(x) + b, y )

Phase 2 (Eq. 11), backbone and alpha_0 frozen; no downstream adversarial examples:

    min_{W, b}  E_{(x,y)}  [ CE - lambda_2 m_tau(x) ]

Representations ``q_l(x)`` and fused ``z_{alpha_0}(x)`` are cached so Phase 2
runs as a linear-head problem. The same runner optionally performs clean test
evaluation and the paper's four-attack adversarial evaluation from the task YAML.
For ER, it automatically executes all five IEMOCAP LOSO folds and reports the
unweighted mean of the five fold-level metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.checkpoints import resolve_foundation_checkpoint
from lib.attacks import (
    AutoPGDCE,
    FusedDisplacementPGD,
    NormalizedMarginPGD,
    PsychoacousticCEPGD,
    achieved_snr_db,
)
from lib.datasets import build_classification_loaders, classification_collate
from lib.metrics import (
    estimate_mu,
    margin_statistics,
    normalized_margin,
    soft_margin,
)
from lib.models import build_backbone
from lib.modules import FullTaskModel, HeadState
from lib.utils import (
    l2_snr_radii,
    linear_warmup_decay,
    load_config,
    resolve_device,
    save_json,
    set_seed,
    waveform_mask,
)


def _base_dataset(dataset):
    ds = dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def _dataset_num_classes(dataset) -> int:
    return int(_base_dataset(dataset).num_classes)


def _dataset_classes(dataset):
    return list(_base_dataset(dataset).classes)


def _maybe_subset(dataset, max_items: int | None, seed: int):
    if not max_items or int(max_items) >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randperm(len(dataset), generator=generator)[: int(max_items)]
    return Subset(dataset, index.tolist())


@torch.no_grad()
def cache_layer_features(
    model: FullTaskModel,
    loader: DataLoader,
    device: torch.device,
    desc: str,
):
    """Cache pooled layer features on CPU without a second full-size copy."""
    model.eval()
    total = len(loader.dataset)
    if total == 0:
        raise RuntimeError(f"{desc}: empty loader")

    features = None
    labels = torch.empty(total, dtype=torch.long)
    offset = 0
    for batch in tqdm(loader, desc=desc, leave=False):
        waveforms = batch["waveforms"].to(
            device=device, dtype=model.head.weight.dtype
        )
        lengths = batch["lengths"].to(device)
        q = model.layer_features(waveforms, lengths).cpu()
        if features is None:
            features = torch.empty(
                q.size(0), total, q.size(2), dtype=q.dtype
            )
        end = offset + q.size(1)
        features[:, offset:end].copy_(q)
        labels[offset:end].copy_(batch["labels"])
        offset = end

    if features is None or offset != total:
        raise RuntimeError(f"{desc}: cached {offset} examples, expected {total}")
    return features, labels



@torch.no_grad()
def evaluate_clean_classification(model: FullTaskModel, loader: DataLoader, device: torch.device) -> float:
    """Full test-set accuracy; called only after checkpoint selection."""
    model.eval()
    correct, count = 0, 0
    for batch in tqdm(loader, desc="clean test", leave=False):
        logits = model(batch["waveforms"].to(device=device, dtype=model.head.weight.dtype), batch["lengths"].to(device))
        y = batch["labels"].to(device)
        correct += int((logits.argmax(dim=-1) == y).sum())
        count += y.numel()
    return 100.0 * correct / max(1, count)


def accuracy_from_layers(model: FullTaskModel, q: torch.Tensor, y: torch.Tensor, batch: int = 4096) -> float:
    if q.size(1) == 0:
        return float("nan")
    correct = 0
    with torch.no_grad():
        for start in range(0, q.size(1), batch):
            chunk = q[:, start:start + batch].to(model.head.weight.device)
            pred = model.logits_from_layers(chunk).argmax(dim=-1).cpu()
            correct += int((pred == y[start:start + batch]).sum())
    return 100.0 * correct / float(q.size(1))


@torch.no_grad()
def _fuse_in_chunks(model: FullTaskModel, q: torch.Tensor, device: torch.device, chunk: int = 1024) -> torch.Tensor:
    """Fuse cached ``q [L, N, D]`` without moving the whole tensor to GPU at once."""
    parts = []
    for start in range(0, q.size(1), chunk):
        parts.append(model.fusion(q[:, start:start + chunk].to(device)).cpu())
    return torch.cat(parts, dim=0)


def evaluate_head(model: FullTaskModel, q: torch.Tensor, y: torch.Tensor, tau: float) -> dict[str, float]:
    # Keep the feature cache on CPU; only bounded chunks enter the device.
    z = _fuse_in_chunks(model, q, model.head.weight.device, chunk=64)
    return _eval_from_z(model, z, y, tau)


def clean_adaptation(
    model: FullTaskModel,
    q_train: torch.Tensor,
    y_train: torch.Tensor,
    q_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> tuple[HeadState, list[dict]]:
    adapt = cfg["adapt"]
    torch.nn.init.normal_(model.head.weight, std=0.01)
    torch.nn.init.zeros_(model.head.bias)
    with torch.no_grad():
        model.fusion.logits.zero_()
    model.fusion.logits.requires_grad_(True)
    model.head.weight.requires_grad_(True)
    model.head.bias.requires_grad_(True)

    n = q_train.size(1)
    batch = int(adapt.get("batch", 256)) or n
    epochs = int(adapt.get("epochs", 100))
    steps_per_epoch = max(1, math.ceil(n / batch))
    total_steps = steps_per_epoch * epochs
    groups = [
        {"params": [model.fusion.logits], "lr": float(adapt["lr"]) * float(adapt.get("fusion_lr_scale", 1.0))},
        {"params": [model.head.weight, model.head.bias], "lr": float(adapt["lr"])},
    ]
    optimiser = torch.optim.Adam(groups, lr=float(adapt["lr"]), weight_decay=float(adapt.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: linear_warmup_decay(s, total_steps, adapt.get("warmup_frac", 0.1))
    )
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)) + 991)
    history = []
    best = None
    log_every = int(adapt.get("log_every", 10))

    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        running = 0.0
        for start in range(0, n, batch):
            picked = order[start:start + batch]
            qb = q_train[:, picked].to(device)
            yb = y_train[picked].to(device)
            logits = model.logits_from_layers(qb)
            loss = F.cross_entropy(logits, yb, label_smoothing=float(adapt.get("label_smoothing", 0.0)))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            scheduler.step()
            running += float(loss.detach()) * picked.numel()
        val_acc = accuracy_from_layers(model, q_val, y_val)
        record = {"epoch": epoch + 1, "loss": running / max(1, n), "val_accuracy": val_acc}
        history.append(record)
        if best is None or val_acc >= best[0]:
            best = (val_acc, model.state())
        if log_every and (epoch + 1) % log_every == 0:
            print(f"    adapt epoch {epoch + 1:>3}  loss {record['loss']:.4f}  val {val_acc:.2f}%")

    assert best is not None
    model.install(best[1])
    train_acc = accuracy_from_layers(model, q_train, y_train)
    counts = torch.bincount(y_train, minlength=int(model.num_classes))
    majority = 100.0 * float(counts.max()) / max(1, int(counts.sum()))
    if train_acc <= majority + 1.0:
        print(
            f"WARNING: clean adaptation finished at {train_acc:.2f}% train vs majority "
            f"{majority:.2f}%. Raise adapt.epochs or adapt.lr ({total_steps} steps)."
        )
    print(f"Clean adaptation: train {train_acc:.2f}%  val {best[0]:.2f}%  alpha={model.fusion.report()}")
    return best[1], history


def _margin_term(z, W, b, y, objective: str, tau: float) -> torch.Tensor:
    """Margin term used by the Stage-2 objective."""
    if objective == "smooth":
        return soft_margin(z, W, b, y, tau)
    if objective == "hard":
        return normalized_margin(z, W, b, y)
    if objective == "ce":
        return torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
    raise ValueError(f"unknown stage-2 objective: {objective!r}")


def _refine_once(
    model: FullTaskModel,
    parent: HeadState,
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    device: torch.device,
    lambda2: float,
    tau: float,
    objective: str,
) -> tuple[HeadState, list[dict], dict[str, float], bool]:
    """One STAGE 2 fit at a fixed (lambda2, tau), starting from (alpha_0, W_0, b_0)."""
    stage2 = cfg["stage2"]
    model.install(parent)
    model.freeze_fusion()
    model.head.weight.requires_grad_(True)
    model.head.bias.requires_grad_(True)
    n = z_train.size(0)
    batch = int(stage2.get("batch", 0)) or n
    epochs = int(stage2.get("epochs", 50))
    steps_per_epoch = max(1, math.ceil(n / batch))
    total_steps = steps_per_epoch * epochs
    optimiser = torch.optim.Adam(
        model.trainable_head_params(),
        lr=float(stage2.get("lr", 1e-3)),
        weight_decay=float(stage2.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: linear_warmup_decay(s, total_steps, stage2.get("warmup_frac", 0.1))
    )
    parent_metrics = _eval_from_z(model, z_val, y_val, tau)
    floor = parent_metrics["clean_accuracy"] - float(stage2.get("max_clean_drop_pp", 0.5))
    generator = torch.Generator().manual_seed(int(stage2.get("seed", 7)))
    history = []
    log_every = int(stage2.get("log_every", 5))
    select_metric = stage2.get("select_metric", "m10")
    if select_metric not in parent_metrics:
        raise ValueError(f"Unknown Stage-2 select_metric: {select_metric!r}")
    if float(stage2.get("max_clean_drop_pp", 0.5)) < 0:
        raise ValueError("max_clean_drop_pp must be non-negative")
    # Parent is eligible from epoch zero; strict improvement retains it on ties.
    best = (parent_metrics[select_metric], parent, parent_metrics)

    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        running = 0.0
        for start in range(0, n, batch):
            picked = order[start:start + batch]
            zb = z_train[picked].to(device)
            yb = y_train[picked].to(device)
            W, b = model.head.weight, model.head.bias
            logits = zb @ W.t() + b
            loss = F.cross_entropy(logits, yb)
            if lambda2 > 0 and objective != "ce":
                loss = loss - lambda2 * _margin_term(zb, W, b, yb, objective, tau).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            scheduler.step()
            running += float(loss.detach()) * picked.numel()
        metrics = _eval_from_z(model, z_val, y_val, tau)
        record = {
            "epoch": epoch + 1,
            "lambda2": lambda2,
            "tau": tau,
            "loss": running / max(1, n),
            **metrics,
        }
        history.append(record)
        admissible = metrics["clean_accuracy"] >= floor
        if select_metric not in metrics:
            raise ValueError(f"Unknown Stage-2 select_metric: {select_metric!r}")
        score = metrics[select_metric]
        if admissible and score > best[0]:
            best = (score, model.state(), metrics)
        if log_every and (epoch + 1) % log_every == 0:
            print(
                f"    refine lambda2={lambda2:g} tau={tau:g} epoch {epoch + 1:>3}  "
                f"loss {record['loss']:.4f}  val {metrics['clean_accuracy']:.2f}%  "
                f"m10 {metrics['m10']:.4f}"
            )

    model.install(best[1])
    return best[1], history, best[2], best[1] is not parent


def margin_refinement(
    model: FullTaskModel,
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> tuple[HeadState, list[dict], dict[str, float]]:
    """Stage 2 with clean-only checkpoint selection."""
    stage2 = cfg["stage2"]
    objective = str(stage2.get("objective", "smooth"))
    parent = model.state()
    parent_metrics = _eval_from_z(model, z_val, y_val, float(stage2.get("tau", 0.1)))
    grid_lambda = [float(v) for v in (stage2.get("lambda2_grid") or [stage2.get("lambda2", 2.0)])]
    grid_tau = [float(v) for v in (stage2.get("tau_grid") or [stage2.get("tau", 0.1)])]
    if any(v < 0 for v in grid_lambda):
        raise ValueError("stage2 lambda2 values must be non-negative")
    if any(v <= 0 for v in grid_tau):
        raise ValueError("stage2 tau values must be positive")
    if objective != "smooth":
        grid_tau = [float(stage2.get("tau", 0.1))]
    if objective == "ce":
        grid_lambda = [0.0]

    print(
        f"STAGE 2 / Eq. (11): objective={objective}  "
        f"lambda2 grid={grid_lambda}  tau grid={grid_tau}  "
        f"select={stage2.get('select_metric', 'm10')}  "
        f"max clean drop={stage2.get('max_clean_drop_pp', 0.5)} pp  "
        f"(alpha_0 frozen)"
    )
    print(f"  fusion alpha_0 = {model.fusion.report()}")

    select_metric = stage2.get("select_metric", "m10")
    if select_metric not in parent_metrics:
        raise ValueError(f"Unknown Stage-2 select_metric: {select_metric!r}")
    # Compare all runs at the configured reporting tau, including tau-dependent
    # metrics such as soft_margin_mean. The parent comes first to win all ties.
    reporting_tau = float(stage2.get("tau", 0.1))
    floor = parent_metrics["clean_accuracy"] - float(stage2.get("max_clean_drop_pp", 0.5))
    candidates = [{"state": parent, "metrics": parent_metrics, "constraint_met": True,
                   "lambda2": None, "tau": reporting_tau}]
    all_history: list[dict] = []
    for lam in grid_lambda:
        for tau in grid_tau:
            state, history, metrics, ok = _refine_once(
                model, parent, z_train, y_train, z_val, y_val, cfg, device,
                lambda2=float(lam), tau=float(tau), objective=objective,
            )
            all_history.extend(history)
            metrics = _eval_from_z(model, z_val, y_val, reporting_tau)
            candidates.append({
                "lambda2": float(lam),
                "tau": float(tau),
                "state": state,
                "metrics": metrics,
                "constraint_met": metrics["clean_accuracy"] >= floor,
            })
            print(
                f"  candidate lambda2={float(lam):g} tau={float(tau):g}: "
                f"val {metrics['clean_accuracy']:.2f}%  m10 {metrics['m10']:.4f}  "
                f"{'improved' if ok else 'parent retained (no admissible improvement)'}"
            )

    admissible = [c for c in candidates if c["constraint_met"]]
    select_metric = stage2.get("select_metric", "m10")
    winner = max(admissible, key=lambda c: c["metrics"][select_metric])
    if winner["state"] is parent:
        print("STAGE 2: no admissible refinement improved the selected metric; retaining clean-adapted parent.")
        model.install(parent)
        return parent, all_history, parent_metrics

    if not torch.equal(winner["state"].fusion_logits, parent.fusion_logits):
        raise RuntimeError("STAGE 2 moved alpha_0; fusion must stay frozen (paper Sec. 2.4)")
    model.install(winner["state"])
    metrics = winner["metrics"]
    print(
        f"STAGE 2 selected lambda2={winner['lambda2']:g} tau={winner['tau']:g}: "
        f"val {metrics['clean_accuracy']:.2f}%  m10 {metrics['m10']:.4f}  "
        f"(parent {parent_metrics['clean_accuracy']:.2f}%)"
    )
    return winner["state"], all_history, metrics


@torch.no_grad()
def _eval_from_z(model: FullTaskModel, z: torch.Tensor, y: torch.Tensor, tau: float,
                 batch: int = 64) -> dict[str, float]:
    if batch <= 0:
        raise ValueError("validation batch must be positive")
    device = model.head.weight.device
    W, b = model.head.weight.detach(), model.head.bias.detach()
    margins, soft_margins, correct = [], [], 0
    for start in range(0, len(y), batch):
        zz = z[start:start + batch].to(device)
        yy = y[start:start + batch].to(device)
        correct += int(((zz @ W.t() + b).argmax(dim=-1) == yy).sum())
        margins.append(normalized_margin(zz, W, b, yy).cpu())
        soft_margins.append(soft_margin(zz, W, b, yy, tau).cpu())
    report = margin_statistics(torch.cat(margins), torch.cat(soft_margins), tau, W.size(0))
    report["clean_accuracy"] = 100.0 * correct / len(y)
    return report



def _build_attacks(eval_cfg: dict, sample_rate: int):
    steps = int(eval_cfg.get("steps", 50))
    restarts = int(eval_cfg.get("restarts", 2))
    snr = float(eval_cfg.get("target_snr", 30.0))
    seed = int(eval_cfg.get("seed", 2027))
    attacks = []

    apgd = eval_cfg.get("apgd_ce", {})
    if apgd.get("enabled", True):
        attacks.append((
            "AutoPGD-CE",
            AutoPGDCE(
                steps=steps,
                restarts=restarts,
                target_snr=snr,
                seed=seed + 11,
                rho=float(apgd.get("rho", 0.75)),
            ),
        ))

    margin = eval_cfg.get("normalized_margin", {})
    if margin.get("enabled", True):
        attacks.append((
            "Normalized-margin",
            NormalizedMarginPGD(
                steps=steps,
                restarts=restarts,
                alpha=float(margin.get("alpha", 0.25)),
                target_snr=snr,
                seed=seed + 23,
            ),
        ))

    displacement = eval_cfg.get("fused_displacement", {})
    if displacement.get("enabled", True):
        attacks.append((
            "Fused-displacement",
            FusedDisplacementPGD(
                steps=steps,
                restarts=restarts,
                alpha=float(displacement.get("alpha", 0.25)),
                target_snr=snr,
                seed=seed + 37,
            ),
        ))

    psycho = eval_cfg.get("psychoacoustic", {})
    if psycho.get("enabled", True):
        attacks.append((
            "Psychoacoustic",
            PsychoacousticCEPGD(
                steps=steps,
                restarts=restarts,
                alpha=float(psycho.get("alpha", 0.25)),
                target_snr=snr,
                seed=seed + 53,
                sample_rate=sample_rate,
                n_fft=int(psycho.get("n_fft", 2048)),
                hop_length=int(psycho.get("hop_length", 512)),
                masking_weight=float(psycho.get("masking_weight", 0.05)),
            ),
        ))
    if not attacks:
        raise ValueError("evaluation.adversarial.enabled=true but no attacks are enabled")
    return attacks


def evaluate_adversarial_classification(
    model: FullTaskModel,
    test_set,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    phase: str,
) -> dict:
    """Run the four evaluation attacks and compute per-example union RA."""
    eval_cfg = cfg.get("evaluation", {}).get("adversarial", {})
    max_test = int(eval_cfg.get("max_test_utterances", 0) or 0)
    selected_test = _maybe_subset(test_set, max_test, int(eval_cfg.get("seed", 2027)))
    loader = DataLoader(
        selected_test,
        batch_size=int(eval_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 2)),
        collate_fn=classification_collate,
    )
    attacks = _build_attacks(eval_cfg, int(cfg["data"].get("sample_rate", 16000)))
    attack_names = [name for name, _ in attacks]
    print(
        f"Adversarial evaluation [{phase}]: N={len(selected_test)}, "
        f"S={float(eval_cfg.get('target_snr', 30.0)):g} dB, "
        f"steps={int(eval_cfg.get('steps', 50))}, restarts={int(eval_cfg.get('restarts', 2))}, "
        f"attacks={attack_names}"
    )

    model.eval()
    model.freeze_fusion()
    model.head.weight.requires_grad_(False)
    model.head.bias.requires_grad_(False)

    all_clean = []
    all_union = []
    per_attack_masks = {name: [] for name in attack_names}
    snr_values = {name: [] for name in attack_names}
    radius_ratios = {name: [] for name in attack_names}

    for batch in tqdm(loader, desc=f"{phase} adversarial eval", leave=False):
        x = batch["waveforms"].to(device=device, dtype=model.head.weight.dtype)
        lengths = batch["lengths"].to(device)
        y = batch["labels"].to(device)
        with torch.no_grad():
            clean_correct = model(x, lengths).argmax(dim=-1).eq(y)
        union_robust = clean_correct.clone()
        all_clean.append(clean_correct.cpu())
        radii = l2_snr_radii(x, lengths, float(eval_cfg.get("target_snr", 30.0)))
        valid = waveform_mask(x, lengths)

        for name, attack in attacks:
            adv = attack.attack(model, x, lengths, y)
            if adv.shape != x.shape or not torch.isfinite(adv).all():
                raise RuntimeError(f"{name} returned an invalid adversarial batch")

            snr = achieved_snr_db(x, adv, lengths)
            delta = torch.where(valid, adv - x, torch.zeros_like(x))
            ratio = delta.norm(dim=1) / radii.clamp_min(1e-12)
            if float(ratio.max()) > 1.00001:
                raise RuntimeError(
                    f"{name} violated the L2/SNR constraint: max radius ratio "
                    f"{float(ratio.max()):.6f}"
                )

            with torch.no_grad():
                adv_correct = model(adv, lengths).argmax(dim=-1).eq(y)
            robust = clean_correct & adv_correct
            per_attack_masks[name].append(robust.cpu())
            union_robust &= adv_correct
            snr_values[name].append(snr.detach().cpu())
            radius_ratios[name].append(ratio.detach().cpu())
        all_union.append(union_robust.cpu())

    clean_mask = torch.cat(all_clean)
    union_mask = torch.cat(all_union)
    attack_masks = {name: torch.cat(parts) for name, parts in per_attack_masks.items()}
    n = max(1, clean_mask.numel())
    results = {
        "num_examples": int(clean_mask.numel()),
        "clean_accuracy": 100.0 * float(clean_mask.sum()) / n,
        "per_attack_robust_accuracy": {
            name: 100.0 * float(mask.sum()) / n for name, mask in attack_masks.items()
        },
        "per_example_union_robust_accuracy": 100.0 * float(union_mask.sum()) / n,
        "threat_model": {
            "target_snr_db": float(eval_cfg.get("target_snr", 30.0)),
            "steps": int(eval_cfg.get("steps", 50)),
            "restarts": int(eval_cfg.get("restarts", 2)),
        },
        "snr_audit": {},
    }
    for name in attack_names:
        snr = torch.cat(snr_values[name]).float()
        ratio = torch.cat(radius_ratios[name]).float()
        results["snr_audit"][name] = {
            "min_snr_db": float(snr.min()),
            "mean_snr_db": float(snr.mean()),
            "max_radius_ratio": float(ratio.max()),
            "mean_radius_ratio": float(ratio.mean()),
        }

    save_json(results, output_dir / f"{phase}_adversarial_metrics.json")
    torch.save(
        {
            "clean_correct": clean_mask,
            "attack_robust": attack_masks,
            "union_robust": union_mask,
        },
        output_dir / f"{phase}_adversarial_masks.pt",
    )
    return results


def _arithmetic_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return float(sum(float(v) for v in values) / len(values))


def _aggregate_adversarial(records: list[dict]) -> dict:
    if not records:
        return {}
    attacks = list(records[0]["per_attack_robust_accuracy"].keys())
    for record in records[1:]:
        if list(record["per_attack_robust_accuracy"].keys()) != attacks:
            raise ValueError("LOSO folds used different attack sets")
    return {
        "clean_accuracy": _arithmetic_mean([r["clean_accuracy"] for r in records]),
        "per_attack_robust_accuracy": {
            name: _arithmetic_mean([r["per_attack_robust_accuracy"][name] for r in records])
            for name in attacks
        },
        "per_example_union_robust_accuracy": _arithmetic_mean(
            [r["per_example_union_robust_accuracy"] for r in records]
        ),
    }


def _run_single_task(cfg: dict, fold: int | None = None) -> dict:
    task = str(cfg["task"]).lower()
    if task not in {"ks", "ic", "sid", "er"}:
        raise ValueError(f"Only ks, ic, sid, and er are supported, got {task!r}")
    if task == "er":
        if fold is None:
            raise ValueError("ER requires an explicit LOSO fold inside the unified runner")
        cfg["data"]["fold"] = int(fold)

    set_seed(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision", "high"))
    device = resolve_device(cfg.get("device", "cuda"))
    checkpoint_dir = Path(cfg["foundation_checkpoint"]).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint_dir}")

    foundation_meta, encoder_cfg, sigma, dtype = resolve_foundation_checkpoint(checkpoint_dir, cfg)
    selected = cfg.get("foundation_model")
    if selected and foundation_meta.get("selected_model") != selected:
        raise ValueError(
            f"Configured foundation_model={selected!r}, but {checkpoint_dir} "
            f"contains {foundation_meta.get('selected_model')!r}"
        )
    checkpoint_id = hashlib.sha256(str(checkpoint_dir).encode("utf-8")).hexdigest()[:10]
    output_root = Path(cfg["train"]["output_dir"])
    if task == "er":
        output_root = output_root / f"fold{int(fold)}"
    output = output_root / f"{encoder_cfg['type']}-{checkpoint_id}-seed{int(cfg.get('seed', 42))}"
    output.mkdir(parents=True, exist_ok=True)
    save_json({**cfg, "encoder": encoder_cfg, "foundation_meta": foundation_meta}, output / "run_config.json")
    print(
        f"Device: {device}  dtype: {dtype}  task: {task}  "
        f"backbone: {encoder_cfg['pretrained_name']}"
    )

    backbone = build_backbone(
        encoder_cfg["type"],
        encoder_cfg["pretrained_name"],
        local_path=encoder_cfg.get("local_path"),
        padding_mode=encoder_cfg.get("padding_mode", "reference"),
        checkpoint=str(checkpoint_dir),
        freeze=True,
    ).to(device=device, dtype=dtype)
    if sigma.ndim != 1 or sigma.numel() != backbone.num_layers:
        raise ValueError(f"Expected {backbone.num_layers} sigma values, got {tuple(sigma.shape)}")

    train_set, val_set, test_set, _, _, _ = build_classification_loaders(cfg)
    feature_bs = int(cfg.get("adapt", {}).get("feature_batch_size", cfg["data"].get("batch_size", 8)))
    workers = int(cfg["data"].get("num_workers", 4))
    train_loader = DataLoader(
        train_set, batch_size=feature_bs, shuffle=False, num_workers=workers,
        collate_fn=classification_collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=feature_bs, shuffle=False, num_workers=workers,
        collate_fn=classification_collate,
    )
    test_loader = DataLoader(
        test_set, batch_size=feature_bs, shuffle=False, num_workers=workers,
        collate_fn=classification_collate,
    )
    num_classes = _dataset_num_classes(train_set)
    print(f"Splits: train={len(train_set)}  val={len(val_set)}  test={len(test_set)}  classes={num_classes}")

    probe = FullTaskModel(backbone, num_classes, sigma=sigma, mu=None, mode="pooled").to(device=device, dtype=dtype)
    mu = estimate_mu(probe, val_loader, device)
    torch.save(mu.cpu(), output / "mu.pt")
    model = FullTaskModel(backbone, num_classes, sigma=sigma, mu=mu, mode="pooled").to(device=device, dtype=dtype)

    # Phase 1: clean adaptation of alpha, W, b.
    q_train, y_train = cache_layer_features(model, train_loader, device, "cache q_l train")
    q_val, y_val = cache_layer_features(model, val_loader, device, "cache q_l val")
    clean_state, adapt_history = clean_adaptation(model, q_train, y_train, q_val, y_val, cfg, device)
    clean_val = evaluate_head(model, q_val, y_val, float(cfg["stage2"].get("tau", 0.1)))

    clean_checkpoint = output / "clean_adapted_checkpoint.pt"
    torch.save(
        {
            "format": "icassp-hierarchical-sfm-v1",
            "phase": "clean_adapted",
            "backbone_checkpoint": str(checkpoint_dir),
            "fold": int(fold) if task == "er" else None,
            "head": model.export_head(),
            "history": adapt_history,
            "validation_metrics": clean_val,
            "classes": _dataset_classes(train_set),
        },
        clean_checkpoint,
    )

    # Phase 2: freeze alpha_0 and refine W,b with the clean soft-margin loss.
    model.install(clean_state)
    model.freeze_fusion()
    with torch.no_grad():
        z_train = _fuse_in_chunks(model, q_train, device)
        z_val = _fuse_in_chunks(model, q_val, device)
    refined_state, refine_history, refined_val = margin_refinement(
        model, z_train, y_train, z_val, y_val, cfg, device
    )

    final_checkpoint = output / "final_robust_downstream_model.pt"
    torch.save(
        {
            "format": "icassp-hierarchical-sfm-v1",
            "phase": "margin_refined",
            "backbone_checkpoint": str(checkpoint_dir),
            "fold": int(fold) if task == "er" else None,
            "head": model.export_head(),
            "adapt_history": adapt_history,
            "refine_history": refine_history,
            "validation_metrics": refined_val,
            "task": task,
            "classes": _dataset_classes(train_set),
        },
        final_checkpoint,
    )

    results = {
        "task": task,
        "fold": int(fold) if task == "er" else None,
        "run_dir": str(output),
        "clean_adapted": {"validation": clean_val},
        "margin_refined": {"validation": refined_val},
        "alpha": model.fusion.report(),
    }

    evaluation = cfg.get("evaluation", {})
    if bool(evaluation.get("clean", False)):
        model.install(clean_state)
        results["clean_adapted"]["test_clean_accuracy"] = evaluate_clean_classification(model, test_loader, device)
        model.install(refined_state)
        results["margin_refined"]["test_clean_accuracy"] = evaluate_clean_classification(model, test_loader, device)

    adv_cfg = evaluation.get("adversarial", {})
    if bool(adv_cfg.get("enabled", False)):
        phases = list(adv_cfg.get("phases", ["clean_adapted", "margin_refined"]))
        allowed = {"clean_adapted", "margin_refined"}
        unknown = [phase for phase in phases if phase not in allowed]
        if unknown:
            raise ValueError(f"Unknown adversarial evaluation phase(s): {unknown}")
        if "clean_adapted" in phases:
            model.install(clean_state)
            results["clean_adapted"]["adversarial"] = evaluate_adversarial_classification(
                model, test_set, cfg, device, output, "clean_adapted"
            )
        if "margin_refined" in phases:
            model.install(refined_state)
            results["margin_refined"]["adversarial"] = evaluate_adversarial_classification(
                model, test_set, cfg, device, output, "margin_refined"
            )

    save_json(results, output / "metrics.json")
    print(json.dumps(results, indent=2, default=str))

    # Release one LOSO fold before the next one starts.
    del model, probe, backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _run_iemocap_loso(cfg: dict) -> dict:
    folds = [int(v) for v in cfg["data"].get("loso_folds", [1, 2, 3, 4, 5])]
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"ER requires LOSO folds [1,2,3,4,5], got {folds}")
    if cfg["data"].get("loso_aggregation") != "unweighted_mean_of_fold_accuracies":
        raise ValueError("ER must use unweighted_mean_of_fold_accuracies")

    fold_results = []
    for fold in folds:
        print(f"\n========== IEMOCAP LOSO fold {fold}/5: Session {fold} held out ==========")
        fold_cfg = copy.deepcopy(cfg)
        fold_results.append(_run_single_task(fold_cfg, fold=fold))

    summary = {
        "task": "er",
        "protocol": "IEMOCAP five-session leave-one-session-out",
        "folds": folds,
        "aggregation": "unweighted arithmetic mean of fold-level metrics (no pooled examples)",
        "fold_results": fold_results,
        "mean": {},
    }

    if all("test_clean_accuracy" in r["clean_adapted"] for r in fold_results):
        summary["mean"]["clean_adapted_test_accuracy"] = _arithmetic_mean(
            [r["clean_adapted"]["test_clean_accuracy"] for r in fold_results]
        )
    if all("test_clean_accuracy" in r["margin_refined"] for r in fold_results):
        summary["mean"]["margin_refined_test_accuracy"] = _arithmetic_mean(
            [r["margin_refined"]["test_clean_accuracy"] for r in fold_results]
        )

    for phase in ("clean_adapted", "margin_refined"):
        robust = [r[phase].get("adversarial") for r in fold_results]
        if all(item is not None for item in robust):
            summary["mean"][f"{phase}_adversarial"] = _aggregate_adversarial(robust)

    checkpoint_id = hashlib.sha256(str(Path(cfg["foundation_checkpoint"]).resolve()).encode("utf-8")).hexdigest()[:10]
    summary_dir = Path(cfg["train"]["output_dir"]) / f"loso-{checkpoint_id}-seed{int(cfg.get('seed', 42))}"
    summary_path = save_json(summary, summary_dir / "loso_summary.json")
    print("\n========== IEMOCAP LOSO summary ==========")
    print(json.dumps(summary["mean"], indent=2))
    print(f"Saved LOSO summary to {summary_path}")
    return summary


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    task = str(cfg.get("task", "")).lower()
    if task == "er":
        _run_iemocap_loso(cfg)
    elif task in {"ks", "ic", "sid"}:
        _run_single_task(cfg)
    else:
        raise ValueError(f"Config task must be one of ks/ic/sid/er, got {task!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean adaptation, margin refinement, and evaluation")
    parser.add_argument("--config", "-c", required=True, help="config/{ks,ic,sid,er}.yaml")
    main(parser.parse_args())
