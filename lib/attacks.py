"""Stage-1 PGD and downstream white-box evaluation attacks."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .metrics import lse_beta, normalized_margin, representation_distance_per_layer
from .models import transformer_states
from .utils import l2_snr_radii, project_l2_snr, waveform_mask


@contextmanager
def _attack_mode(model):
    """Temporarily enable waveform gradients through a frozen task backbone."""
    was_training = model.training
    model.eval()
    backbone = getattr(model, "backbone", model)
    enable = getattr(backbone, "enable_attack_mode", None)
    disable = getattr(backbone, "disable_attack_mode", None)
    if enable is not None:
        enable()
    try:
        yield
    finally:
        if disable is not None:
            disable()
        model.train(was_training)


def _random_l2_start(
    clean: torch.Tensor,
    lengths: torch.Tensor,
    radii: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    valid = waveform_mask(clean, lengths)
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    noise = torch.where(valid, noise, torch.zeros_like(noise))
    direction = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # Random start inside the per-example L2 ball.
    radius_fraction = torch.rand(
        clean.size(0), 1, device=clean.device, dtype=clean.dtype, generator=generator
    )
    return project_l2_snr(
        clean + direction * radii.view(-1, 1) * radius_fraction,
        clean,
        lengths,
        radii,
    )


def achieved_snr_db(
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Per-example realised SNR in dB over valid (unpadded) samples."""
    valid = waveform_mask(clean, lengths)
    signal = torch.sqrt((clean.square() * valid).sum(dim=1).clamp_min(1e-24))
    delta = torch.where(valid, adversarial - clean, torch.zeros_like(clean))
    noise = delta.norm(dim=1).clamp_min(1e-24)
    return 20.0 * torch.log10(signal / noise)


class FoundationHierarchicalPGD:
    """Projected gradient ascent on ``LSE_beta(d)`` inside ``Delta_S(x)``.

    The threat set is ``||delta||_2 <= ||x||_2 * 10^(-S/20)`` with ``x+delta``
    clipped to ``[-1, 1]``. The preservation term is handled by the outer loss.
    """

    def __init__(
        self,
        steps: int = 10,
        alpha: float = 0.25,
        target_snr: float = 30.0,
        beta: float = 30.0,
        momentum: float = 0.0,
    ):
        self.steps = int(steps)
        self.alpha = float(alpha)
        self.target_snr = float(target_snr)
        self.beta = float(beta)
        self.momentum = float(momentum)
        self.last_trajectory: list[float] = []
        if self.steps <= 0 or self.alpha <= 0:
            raise ValueError("PGD steps and alpha must be positive")
        if self.beta <= 0:
            raise ValueError("LSE beta must be positive")

    @torch.enable_grad()
    def attack(self, encoder, waveforms: torch.Tensor, lengths: torch.Tensor, scales: torch.Tensor):
        was_training = encoder.training
        encoder.eval()
        try:
            clean = waveforms.detach()
            valid = waveform_mask(clean, lengths)
            radii = l2_snr_radii(clean, lengths, self.target_snr)
            step = (self.alpha * radii).view(-1, 1)
            with torch.no_grad():
                anchor, _ = transformer_states(encoder, clean, lengths)
                anchor = anchor.detach()

            noise = torch.randn_like(clean)
            noise = torch.where(valid, noise, torch.zeros_like(noise))
            noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-12)
            adversarial = project_l2_snr(
                clean + 0.1 * radii.view(-1, 1) * noise, clean, lengths, radii
            )
            velocity = torch.zeros_like(clean)
            self.last_trajectory = []
            encoder.enable_attack_mode()

            for _ in range(self.steps):
                adversarial = adversarial.detach().requires_grad_(True)
                states, frame_mask = transformer_states(encoder, adversarial, lengths)
                distances = representation_distance_per_layer(
                    states, anchor, scales, frame_mask
                )
                objective = lse_beta(distances, self.beta).mean()
                gradient = torch.autograd.grad(objective, adversarial)[0] * valid
                self.last_trajectory.append(float(objective.detach().cpu()))
                with torch.no_grad():
                    direction = gradient / gradient.norm(
                        dim=1, keepdim=True
                    ).clamp_min(1e-12)
                    if self.momentum > 0:
                        velocity = self.momentum * velocity + direction
                        update = velocity / velocity.norm(
                            dim=1, keepdim=True
                        ).clamp_min(1e-12)
                    else:
                        update = direction
                    adversarial = project_l2_snr(
                        adversarial + step * update, clean, lengths, radii
                    )
            return adversarial.detach()
        finally:
            encoder.disable_attack_mode()
            encoder.train(was_training)


class _ObjectivePGD:
    """Multi-restart L2 PGD maximising a per-example white-box objective."""

    def __init__(
        self,
        *,
        steps: int = 50,
        restarts: int = 2,
        alpha: float = 0.25,
        target_snr: float = 30.0,
        seed: int = 0,
    ):
        self.steps = int(steps)
        self.restarts = int(restarts)
        self.alpha = float(alpha)
        self.target_snr = float(target_snr)
        self.seed = int(seed)
        self._calls = 0
        if self.steps <= 0 or self.restarts <= 0 or self.alpha <= 0:
            raise ValueError("steps, restarts, and alpha must be positive")

    def _prepare(self, model, clean, lengths, labels):
        return None

    def _objective(self, model, adversarial, lengths, labels, context) -> torch.Tensor:
        raise NotImplementedError

    @torch.enable_grad()
    def attack(self, model, waveforms: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor):
        clean = waveforms.detach()
        labels = labels.detach()
        valid = waveform_mask(clean, lengths)
        radii = l2_snr_radii(clean, lengths, self.target_snr)
        step = self.alpha * radii.view(-1, 1)
        best_adv = clean.clone()
        best_obj = torch.full(
            (clean.size(0),), -torch.inf, device=clean.device, dtype=clean.dtype
        )

        call_seed = self.seed + 1_000_003 * self._calls
        self._calls += 1
        with _attack_mode(model):
            context = self._prepare(model, clean, lengths, labels)
            for restart in range(self.restarts):
                generator = torch.Generator(device=clean.device)
                generator.manual_seed(call_seed + 104729 * restart)
                adversarial = _random_l2_start(clean, lengths, radii, generator)
                for _ in range(self.steps):
                    adversarial = adversarial.detach().requires_grad_(True)
                    objective = self._objective(model, adversarial, lengths, labels, context)
                    if objective.ndim != 1 or objective.numel() != clean.size(0):
                        raise RuntimeError("attack objective must return one scalar per example")
                    gradient = torch.autograd.grad(objective.sum(), adversarial)[0]
                    gradient = torch.where(valid, gradient, torch.zeros_like(gradient))
                    direction = gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    with torch.no_grad():
                        current = objective.detach()
                        improve = current > best_obj
                        best_obj = torch.where(improve, current, best_obj)
                        best_adv[improve] = adversarial.detach()[improve]
                        adversarial = project_l2_snr(
                            adversarial + step * direction, clean, lengths, radii
                        )
                with torch.no_grad():
                    final_obj = self._objective(model, adversarial, lengths, labels, context)
                    improve = final_obj > best_obj
                    best_obj = torch.where(improve, final_obj, best_obj)
                    best_adv[improve] = adversarial.detach()[improve]
        return best_adv.detach()


class NormalizedMarginPGD(_ObjectivePGD):
    """Defense-aware attack that minimizes the true-class normalized margin."""

    def _objective(self, model, adversarial, lengths, labels, context):
        z = model.fused_features(adversarial, lengths)
        return -normalized_margin(z, model.head.weight, model.head.bias, labels)


class FusedDisplacementPGD(_ObjectivePGD):
    """Defense-aware attack that maximizes ``||z_alpha(x+delta)-z_alpha(x)||_2``."""

    def _prepare(self, model, clean, lengths, labels):
        with torch.no_grad():
            return model.fused_features(clean, lengths).detach()

    def _objective(self, model, adversarial, lengths, labels, context):
        z = model.fused_features(adversarial, lengths)
        return (z - context).norm(dim=-1)


class AutoPGDCE:
    """L2 AutoPGD-CE with per-example SNR radii and adaptive step sizes."""

    def __init__(
        self,
        *,
        steps: int = 50,
        restarts: int = 2,
        target_snr: float = 30.0,
        seed: int = 0,
        rho: float = 0.75,
    ):
        self.steps = int(steps)
        self.restarts = int(restarts)
        self.target_snr = float(target_snr)
        self.seed = int(seed)
        self.rho = float(rho)
        self._calls = 0
        if self.steps <= 0 or self.restarts <= 0:
            raise ValueError("steps and restarts must be positive")
        if not 0 < self.rho <= 1:
            raise ValueError("rho must be in (0, 1]")

    @staticmethod
    def _ce(model, adversarial, lengths, labels):
        return F.cross_entropy(model(adversarial, lengths), labels, reduction="none")

    @torch.enable_grad()
    def attack(self, model, waveforms: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor):
        clean = waveforms.detach()
        labels = labels.detach()
        valid = waveform_mask(clean, lengths)
        radii = l2_snr_radii(clean, lengths, self.target_snr)
        global_best_adv = clean.clone()
        global_best_loss = torch.full(
            (clean.size(0),), -torch.inf, device=clean.device, dtype=clean.dtype
        )

        call_seed = self.seed + 1_000_003 * self._calls
        self._calls += 1
        with _attack_mode(model):
            for restart in range(self.restarts):
                generator = torch.Generator(device=clean.device)
                generator.manual_seed(call_seed + 130363 * restart)
                x = _random_l2_start(clean, lengths, radii, generator)
                x_old = x.clone()
                step_size = 2.0 * radii.view(-1, 1)
                best_x = x.clone()
                best_loss = torch.full_like(global_best_loss, -torch.inf)
                best_loss_at_check = best_loss.clone()
                losses: list[torch.Tensor] = []
                k = max(1, int(round(0.22 * self.steps)))
                k_min = max(1, int(round(0.06 * self.steps)))
                size_decr = max(1, int(round(0.03 * self.steps)))
                since_check = 0

                for iteration in range(self.steps):
                    x = x.detach().requires_grad_(True)
                    loss = self._ce(model, x, lengths, labels)
                    grad = torch.autograd.grad(loss.sum(), x)[0]
                    grad = torch.where(valid, grad, torch.zeros_like(grad))
                    direction = grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    with torch.no_grad():
                        improved = loss > best_loss
                        best_loss = torch.where(improved, loss, best_loss)
                        best_x[improved] = x.detach()[improved]
                        losses.append(loss.detach())

                        candidate = project_l2_snr(
                            x + step_size * direction, clean, lengths, radii
                        )
                        if iteration == 0:
                            x_new = candidate
                        else:
                            # APGD momentum: mostly the current projected step,
                            # with a smaller contribution from the last motion.
                            x_new = project_l2_snr(
                                x + 0.75 * (candidate - x) + 0.25 * (x - x_old),
                                clean,
                                lengths,
                                radii,
                            )
                        x_old = x.detach()
                        x = x_new
                        since_check += 1

                        if since_check >= k and len(losses) >= 2:
                            recent = torch.stack(losses[-k:], dim=0)
                            increases = (recent[1:] > recent[:-1]).float().sum(dim=0)
                            oscillating = increases <= self.rho * max(1, k - 1)
                            no_improve = best_loss <= best_loss_at_check + 1e-12
                            reduce = oscillating | no_improve
                            if reduce.any():
                                step_size[reduce] *= 0.5
                                x[reduce] = best_x[reduce]
                                x_old[reduce] = best_x[reduce]
                            best_loss_at_check = best_loss.clone()
                            since_check = 0
                            k = max(k - size_decr, k_min)

                with torch.no_grad():
                    final_loss = self._ce(model, x, lengths, labels)
                    improved = final_loss > best_loss
                    best_loss = torch.where(improved, final_loss, best_loss)
                    best_x[improved] = x.detach()[improved]
                    improve_global = best_loss > global_best_loss
                    global_best_loss = torch.where(improve_global, best_loss, global_best_loss)
                    global_best_adv[improve_global] = best_x[improve_global]
        return global_best_adv.detach()


# ---------------------------------------------------------------------------
# Psychoacoustic masking attack (Qin et al.-style frequency masking)
# ---------------------------------------------------------------------------


def _bark_scale(frequency_hz: torch.Tensor) -> torch.Tensor:
    return 13.0 * torch.atan(0.76 * frequency_hz / 1000.0) + 3.5 * torch.atan(
        (frequency_hz / 7500.0).square()
    )


def _absolute_threshold_hearing(frequency_hz: torch.Tensor) -> torch.Tensor:
    f = (frequency_hz.clamp_min(20.0) / 1000.0)
    return 3.64 * f.pow(-0.8) - 6.5 * torch.exp(-0.6 * (f - 3.3).square()) + 1e-3 * f.pow(4)


@torch.no_grad()
def psychoacoustic_masking_threshold(
    clean: torch.Tensor,
    *,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a Qin-et-al.-style global frequency masking threshold.

    Returns ``(theta_db, clean_psd_max_db, valid_frequency_mask)`` where
    ``theta_db`` has shape ``[B, F, frames]``.  The implementation follows the
    tonal-masker, Bark-spreading, and absolute-threshold equations described in
    the cited psychoacoustic attack family.  It is used as a *shaping penalty*
    while the outer threat set remains the paper's 30-dB L2 ball.
    """
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")
    device, dtype = clean.device, clean.dtype
    if clean.size(1) < n_fft:
        clean = F.pad(clean, (0, n_fft - clean.size(1)))
    window = torch.hann_window(n_fft, device=device, dtype=dtype)
    spectrum = torch.stft(
        clean,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = (spectrum.abs() / float(n_fft)).square().clamp_min(1e-20)
    psd_db = 10.0 * torch.log10(power)
    psd_peak = psd_db.amax(dim=1, keepdim=True)
    normalized = 96.0 - psd_peak + psd_db

    frequencies = torch.linspace(0.0, sample_rate / 2.0, spectrum.size(1), device=device, dtype=dtype)
    bark = _bark_scale(frequencies)
    ath = _absolute_threshold_hearing(frequencies)
    valid_freq = frequencies >= 20.0
    # Values outside the modeled hearing range are excluded from the penalty.
    ath = torch.where(valid_freq, ath, torch.full_like(ath, -120.0))

    # Bark-neighborhood bounds for the "largest within 0.5 Bark" criterion.
    bark_cpu = bark.detach().cpu()
    lo = torch.searchsorted(bark_cpu, bark_cpu - 0.5, right=False).tolist()
    hi = torch.searchsorted(bark_cpu, bark_cpu + 0.5, right=True).tolist()

    theta = torch.empty_like(normalized)
    for b in range(clean.size(0)):
        for frame in range(normalized.size(2)):
            p = normalized[b, :, frame]
            local = torch.zeros_like(p, dtype=torch.bool)
            if p.numel() > 2:
                local[1:-1] = (p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:])
            candidates = torch.nonzero(local & (p >= ath) & valid_freq, as_tuple=False).flatten()
            selected: list[int] = []
            for idx in candidates.tolist():
                if p[idx] >= p[lo[idx]:hi[idx]].max():
                    selected.append(idx)

            # Start with the threshold in quiet in linear power units.
            total_linear = torch.pow(10.0, ath / 10.0)
            if selected:
                idx = torch.tensor(selected, device=device, dtype=torch.long)
                # Eq. (22): smooth tonal-masker PSD with immediate neighbors.
                left = (idx - 1).clamp_min(0)
                right = (idx + 1).clamp_max(p.numel() - 1)
                masker_linear = (
                    torch.pow(10.0, p[left] / 10.0)
                    + torch.pow(10.0, p[idx] / 10.0)
                    + torch.pow(10.0, p[right] / 10.0)
                )
                masker_db = 10.0 * torch.log10(masker_linear.clamp_min(1e-20))
                b_i = bark[idx]
                delta_b = bark.unsqueeze(0) - b_i.unsqueeze(1)
                slope_right = -27.0 + 0.37 * torch.clamp(masker_db - 40.0, min=0.0)
                spread = torch.where(
                    delta_b <= 0,
                    27.0 * delta_b,
                    slope_right.unsqueeze(1) * delta_b,
                )
                delta_m = -6.025 - 0.275 * b_i
                individual = masker_db.unsqueeze(1) + delta_m.unsqueeze(1) + spread
                total_linear = total_linear + torch.pow(10.0, individual / 10.0).sum(dim=0)
            theta[b, :, frame] = 10.0 * torch.log10(total_linear.clamp_min(1e-20))
    return theta, psd_peak, valid_freq


def _psychoacoustic_violation(
    delta: torch.Tensor,
    theta_db: torch.Tensor,
    clean_psd_peak_db: torch.Tensor,
    valid_frequency_mask: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    if delta.size(1) < n_fft:
        delta = F.pad(delta, (0, n_fft - delta.size(1)))
    window = torch.hann_window(n_fft, device=delta.device, dtype=delta.dtype)
    spectrum = torch.stft(
        delta,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = (spectrum.abs() / float(n_fft)).square().clamp_min(1e-20)
    p_delta = 10.0 * torch.log10(power)
    normalized_delta = 96.0 - clean_psd_peak_db + p_delta
    violation = F.relu(normalized_delta - theta_db)
    mask = valid_frequency_mask[None, :, None].to(dtype=violation.dtype)
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0) * violation.size(2)
    return (violation * mask).sum(dim=(1, 2)) / denom


class PsychoacousticCEPGD(_ObjectivePGD):
    """CE attack shaped by a differentiable frequency-masking penalty.

    The attack still projects every iterate onto the same SNR-bounded L2 threat
    set as the other evaluation attacks.  ``masking_weight`` controls how much
    the optimizer prefers perturbation energy below the clean speech's global
    masking threshold.
    """

    def __init__(
        self,
        *,
        steps: int = 50,
        restarts: int = 2,
        alpha: float = 0.25,
        target_snr: float = 30.0,
        seed: int = 0,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        masking_weight: float = 0.05,
    ):
        super().__init__(
            steps=steps,
            restarts=restarts,
            alpha=alpha,
            target_snr=target_snr,
            seed=seed,
        )
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.masking_weight = float(masking_weight)
        if self.masking_weight < 0:
            raise ValueError("masking_weight must be non-negative")

    def _prepare(self, model, clean, lengths, labels):
        # Compute thresholds only on valid waveform samples. This matters for KS,
        # whose dataset pads clips to one second while retaining the true length.
        per_example = []
        for i, length in enumerate(lengths.tolist()):
            one = clean[i:i + 1, : int(length)]
            theta, peak, valid_freq = psychoacoustic_masking_threshold(
                one,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            per_example.append((theta, peak, valid_freq, int(length)))
        return per_example, clean.detach()

    def _objective(self, model, adversarial, lengths, labels, context):
        per_example, clean = context
        ce = F.cross_entropy(model(adversarial, lengths), labels, reduction="none")
        violations = []
        for i, (theta, peak, valid_freq, length) in enumerate(per_example):
            delta = adversarial[i:i + 1, :length] - clean[i:i + 1, :length]
            violations.append(
                _psychoacoustic_violation(
                    delta,
                    theta,
                    peak,
                    valid_freq,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                )[0]
            )
        violation = torch.stack(violations)
        return ce - self.masking_weight * violation
