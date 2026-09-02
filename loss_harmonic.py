"""Harmonic-aware extension of the verified RhythmMamba HybridLoss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import HybridLoss, _normalized_frequency_power
from metrics import calculate_hr


MINIMUM_BPM = 45.0
MAXIMUM_BPM = 149.0
HARMONIC_RATIOS = (0.5, 1.5, 2.0)
HARMONIC_BAND_HALF_WIDTH_BPM = 2.0
HARMONIC_WEIGHT = 0.1
EPSILON = 1e-8


def _band_power(
    spectrum: torch.Tensor,
    bpm_range: torch.Tensor,
    center_bpm: float,
) -> torch.Tensor:
    mask = torch.abs(bpm_range - float(center_bpm)) <= HARMONIC_BAND_HALF_WIDTH_BPM
    if not bool(mask.any()):
        return spectrum.new_tensor(0.0)
    return spectrum[0, mask].sum()


def harmonic_competition_loss(
    prediction: torch.Tensor,
    label: torch.Tensor,
    fs: int = 30,
    diff_flag: bool = False,
) -> torch.Tensor:
    """Penalize harmonic power relative to power near the ground-truth HR."""
    _, label_hr = calculate_hr(
        prediction.detach().cpu().numpy(),
        label.detach().cpu().numpy(),
        fs=fs,
        diff_flag=diff_flag,
    )
    bpm_range = torch.arange(
        int(MINIMUM_BPM),
        int(MAXIMUM_BPM) + 1,
        device=prediction.device,
        dtype=torch.float32,
    )
    spectrum = _normalized_frequency_power(prediction.squeeze(-1), fs, bpm_range)
    fundamental_power = _band_power(spectrum, bpm_range, float(label_hr))

    harmonic_power = spectrum.new_tensor(0.0)
    valid_harmonics = 0
    for ratio in HARMONIC_RATIOS:
        harmonic_bpm = float(label_hr) * ratio
        if MINIMUM_BPM <= harmonic_bpm <= MAXIMUM_BPM:
            harmonic_power = harmonic_power + _band_power(
                spectrum, bpm_range, harmonic_bpm
            )
            valid_harmonics += 1

    if valid_harmonics == 0:
        return spectrum.new_tensor(0.0)
    ratio = (fundamental_power + EPSILON) / (
        fundamental_power + harmonic_power + EPSILON
    )
    return -torch.log(torch.clamp(ratio, min=EPSILON, max=1.0))


def _shared_base_and_harmonic(
    prediction: torch.Tensor,
    label: torch.Tensor,
    base_loss: HybridLoss,
    fs: int,
    diff_flag: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute L0 and the harmonic term using one HR and spectrum calculation."""
    time_loss = base_loss.pearson(prediction, label)
    if torch.isnan(time_loss):
        time_loss = prediction.new_tensor(0.0)
    _, label_hr = calculate_hr(
        prediction.detach().cpu().numpy(),
        label.detach().cpu().numpy(),
        fs=fs,
        diff_flag=diff_flag,
    )
    bpm_range = torch.arange(
        int(MINIMUM_BPM),
        int(MAXIMUM_BPM) + 1,
        device=prediction.device,
        dtype=torch.float32,
    )
    spectrum = _normalized_frequency_power(prediction.squeeze(-1), fs, bpm_range)
    target_index = torch.tensor(
        [label_hr - MINIMUM_BPM], device=prediction.device, dtype=torch.long
    )
    frequency_loss = F.cross_entropy(spectrum, target_index)
    verified_base = 0.2 * time_loss + frequency_loss

    fundamental_power = _band_power(spectrum, bpm_range, float(label_hr))
    harmonic_power = spectrum.new_tensor(0.0)
    valid_harmonics = 0
    for ratio_value in HARMONIC_RATIOS:
        harmonic_bpm = float(label_hr) * ratio_value
        if MINIMUM_BPM <= harmonic_bpm <= MAXIMUM_BPM:
            harmonic_power = harmonic_power + _band_power(
                spectrum, bpm_range, harmonic_bpm
            )
            valid_harmonics += 1
    if valid_harmonics == 0:
        harmonic_loss = spectrum.new_tensor(0.0)
    else:
        competition_ratio = (fundamental_power + EPSILON) / (
            fundamental_power + harmonic_power + EPSILON
        )
        harmonic_loss = -torch.log(
            torch.clamp(competition_ratio, min=EPSILON, max=1.0)
        )
    return verified_base, harmonic_loss


class HarmonicAwareHybridLoss(nn.Module):
    """L0 plus a fixed weighted fundamental-versus-harmonics competition."""

    def __init__(self, harmonic_weight: float = HARMONIC_WEIGHT):
        super().__init__()
        if harmonic_weight < 0:
            raise ValueError("harmonic_weight must be non-negative")
        self.base = HybridLoss()
        self.harmonic_weight = float(harmonic_weight)

    def components(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
        epoch: int,
        fs: int = 30,
        diff_flag: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_loss, harmonic_loss = _shared_base_and_harmonic(
            prediction, label, self.base, fs, diff_flag
        )
        total_loss = base_loss + self.harmonic_weight * harmonic_loss
        return total_loss, base_loss, harmonic_loss

    def forward(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
        epoch: int,
        fs: int = 30,
        diff_flag: bool = False,
    ) -> torch.Tensor:
        return self.components(prediction, label, epoch, fs, diff_flag)[0]
