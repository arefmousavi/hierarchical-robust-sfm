"""Paper geometry: layer scales, rho^2, LSE_beta, preservation, and classifier margins.

Equations (ICASSP draft):

  sigma_l^2 = E_x[ (T_x D)^{-1} || h^l_{theta_b}(x) ||_F^2 ]

  rho_l^2(a, b) = (T_x D sigma_l^2)^{-1} sum_t || a_t - b_t ||_2^2

  LSE_beta(d) = beta^{-1} log[ L^{-1} sum_l exp(beta d_l) ]

  L_pres(x) = L^{-1} sum_l rho_l^2(h_theta^l(x), h_{theta_b}^l(x))

  q_l(x) = ( Pi(h^l(x)) - mu_l ) / ( sigma_l sqrt(D) )

  m_yj(x) = [ (w_y - w_j)^T z + (b_y - b_j) ] / || w_y - w_j ||_2

  m_tau(x) = -tau log sum_{j != y} exp( -m_yj(x) / tau )
"""

from __future__ import annotations

import math

import torch
from tqdm.auto import tqdm

PAIRWISE_EPS = 1e-12


def representation_distance_per_layer(
    V: torch.Tensor,
    U: torch.Tensor,
    scales: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Eq. (2) for every Transformer layer. ``V, U: [L, B, T, D]`` -> ``[B, L]``."""
    if V.shape != U.shape or V.ndim != 4:
        raise ValueError(f"Expected matching [L,B,T,D] tensors, got {V.shape}, {U.shape}")
    # Compute in at least float32, and preserve float64 for numerical audits.
    promoted = torch.promote_types(V.dtype, U.dtype)
    dtype = torch.float64 if promoted == torch.float64 else torch.float32
    scales = torch.as_tensor(scales, device=V.device, dtype=dtype)
    if scales.ndim != 1 or scales.numel() != V.size(0):
        raise ValueError(f"Expected {V.size(0)} positive scales, got {tuple(scales.shape)}")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Every representation scale must be finite and positive")
    mask = frame_mask.to(dtype=dtype).unsqueeze(0).unsqueeze(-1)
    valid = frame_mask.sum(dim=1).clamp_min(1).to(dtype=dtype)
    squared = ((V.to(dtype) - U.to(dtype)) * mask).square().sum(dim=(2, 3))
    normalized = squared / (valid.unsqueeze(0) * V.size(-1))
    return (normalized / scales.square().unsqueeze(1)).transpose(0, 1)


def lse_beta(distances: torch.Tensor, beta: float) -> torch.Tensor:
    """Stable ``(1/beta) log mean_l exp(beta d_l)``. ``distances`` is ``[B, L]``."""
    beta = float(beta)
    if beta <= 0:
        raise ValueError("LSE beta must be positive")
    if distances.ndim != 2:
        raise ValueError(f"Expected [B, L] layer distances, got {tuple(distances.shape)}")
    peak = distances.amax(dim=-1, keepdim=True)
    return peak.squeeze(-1) + torch.log(
        torch.exp(beta * (distances - peak)).mean(dim=-1).clamp_min(1e-12)
    ) / beta


def preservation_loss(
    clean_states: torch.Tensor,
    base_states: torch.Tensor,
    scales: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Eq. (5): ``L_pres = mean_l rho^2_l(h_theta(x), h_{theta_b}(x))``, then batched mean."""
    return representation_distance_per_layer(
        clean_states, base_states, scales, frame_mask
    ).mean(dim=1).mean()


def pool_frames(states: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool valid frames. ``[L, B, T, D] -> [L, B, D]``."""
    mask = frame_mask.float().unsqueeze(0).unsqueeze(-1)
    counts = frame_mask.sum(dim=1).clamp_min(1).float()
    return (states * mask).sum(dim=2) / counts.unsqueeze(0).unsqueeze(-1)


def normalize_pooled(
    pooled: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor | None,
    hidden_size: int,
) -> torch.Tensor:
    """Eq. (3): ``q_l = (Pi(h^l) - mu_l) / (sigma_l sqrt(D))``. ``[L, B, D]``."""
    scale = (sigma.to(device=pooled.device, dtype=pooled.dtype) * float(hidden_size) ** 0.5)
    centred = pooled if mu is None else pooled - mu.to(device=pooled.device, dtype=pooled.dtype)[:, None, :]
    return centred / scale.clamp_min(1e-12)[:, None, None]


@torch.no_grad()
def estimate_sigma(encoder, dataloader, device: torch.device) -> torch.Tensor:
    """Utterance-equal estimate of ``sigma_l`` from a frozen clean backbone."""
    encoder.eval()
    totals = None
    examples = 0
    for batch in tqdm(dataloader, desc="estimate sigma_l", leave=False):
        waveforms = batch["waveforms"].to(
            device=device, dtype=next(encoder.parameters()).dtype
        )
        lengths = batch["lengths"].to(device)
        states, frame_mask = encoder.transformer_states(waveforms, lengths)
        mask = frame_mask.to(dtype=torch.float64).unsqueeze(0).unsqueeze(-1)
        valid = frame_mask.sum(dim=1).clamp_min(1).to(dtype=torch.float64)
        values = (states.double() * mask).square().sum(dim=(2, 3))
        values = values / (valid.unsqueeze(0) * states.size(-1))
        contribution = values.double().sum(dim=1)
        totals = contribution if totals is None else totals + contribution
        examples += waveforms.size(0)
    if not examples:
        raise ValueError("Cannot estimate sigma on an empty dataset")
    sigma = torch.sqrt(totals / examples).to(dtype=states.dtype).cpu()
    if not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise RuntimeError("Estimated sigma is not finite and positive")
    return sigma


@torch.no_grad()
def estimate_mu(model, dataloader, device: torch.device) -> torch.Tensor:
    """Task-specific ``mu_l``: mean pooled representation per layer on held-out data."""
    model.eval()
    total = None
    count = 0
    for batch in tqdm(dataloader, desc="estimate mu_l", leave=False):
        waveforms = batch["waveforms"].to(
            device=device, dtype=model.head.weight.dtype
        )
        lengths = batch["lengths"].to(device)
        pooled = model.pooled_states(waveforms, lengths)
        contribution = pooled.sum(dim=1).double()
        total = contribution if total is None else total + contribution
        count += pooled.size(1)
    if count == 0:
        raise RuntimeError("mu estimation saw no utterances")
    return (total / float(count)).to(dtype=pooled.dtype).cpu()


def pairwise_weight_norms(W: torch.Tensor) -> torch.Tensor:
    """``||w_c - w_k||_2`` for every class pair. ``[K, K]``."""
    return _weight_distances(W, W)


def _weight_distances(rows: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # Direct cdist avoids explicit [N,K,D] differences and cancellation in
    # ||a||^2 + ||b||^2 - 2<a,b> for equal or nearly equal classifier rows.
    distances = torch.cdist(rows, W, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    return torch.sqrt(distances.square() + PAIRWISE_EPS)


def pairwise_margins(
    features: torch.Tensor, W: torch.Tensor, b: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """``m_yj(x)`` for every competitor ``j``. ``[N, K]``, true-class slot at ``+inf``."""
    logits = features @ W.t() + b
    numerator = logits.gather(1, y[:, None]) - logits
    denominator = _weight_distances(W[y], W)
    return (numerator / denominator).scatter(1, y[:, None], float("inf"))


def normalized_margin(
    features: torch.Tensor, W: torch.Tensor, b: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """``m(x) = min_{j != y} m_yj(x)``. Positive iff the linear head predicts ``y``."""
    return pairwise_margins(features, W, b, y).min(dim=1).values


def soft_margin(
    features: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Eq. (10): ``m_tau(x)``, a lower bound on ``m(x)`` within ``tau log(K-1)``."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    ratios = -pairwise_margins(features, W, b, y) / float(tau)
    return -float(tau) * torch.logsumexp(ratios, dim=1)


def soft_margin_gap(tau: float, num_classes: int) -> float:
    if num_classes < 2:
        return 0.0
    return float(tau) * math.log(float(num_classes - 1))


@torch.no_grad()
def margin_report(
    features: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    tau: float | None = None,
) -> dict[str, float]:
    m = normalized_margin(features, W, b, y)
    soft = None if tau is None else soft_margin(features, W, b, y, tau)
    return margin_statistics(m, soft, tau, int(W.size(0)))


def margin_statistics(
    m: torch.Tensor,
    soft: torch.Tensor | None = None,
    tau: float | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Reduce combined per-example margins, never minibatch percentiles."""
    report = {
        "m10": float(torch.quantile(m, 0.10)),
        "m25": float(torch.quantile(m, 0.25)),
        "median_margin": float(m.median()),
        "mean_margin": float(m.mean()),
        "min_margin": float(m.min()),
        "frac_positive": float((m > 0).float().mean()),
    }
    if tau is not None:
        report["soft_margin_mean"] = float(soft.mean())
        report["soft_margin_gap"] = soft_margin_gap(tau, num_classes)
    return report


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    return float(100.0 * (logits.argmax(dim=-1) == y).float().mean())
