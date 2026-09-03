"""Controlled RhythmMamba loss ablations for the Stage-2 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import HybridLoss, NegativePearsonLoss, _normalized_frequency_power
from metrics import calculate_hr


MINIMUM_BPM = 45.0
MAXIMUM_BPM = 149.0
CONCENTRATION_HALF_WIDTH_BPM = 3.0
HARMONIC_BAND_HALF_WIDTH_BPM = 2.0
HARMONIC_RATIOS = (0.5, 1.5, 2.0)
EPSILON = 1e-8


@dataclass(frozen=True)
class LossVariant:
    code: str
    description: str
    pearson_weight: float
    ce_weight: float
    concentration_weight: float
    harmonic_weight: float


LOSS_VARIANTS = {
    "L2_HARMONIC_REPLACE": LossVariant(
        code="L2_HARMONIC_REPLACE",
        description="0.2 Pearson + 0.1 harmonic; CE removed",
        pearson_weight=0.2,
        ce_weight=0.0,
        concentration_weight=0.0,
        harmonic_weight=0.1,
    ),
    "L3_CONCENTRATION": LossVariant(
        code="L3_CONCENTRATION",
        description="0.2 Pearson + 1.0 concentration; CE removed",
        pearson_weight=0.2,
        ce_weight=0.0,
        concentration_weight=1.0,
        harmonic_weight=0.0,
    ),
    "L4_CONCENTRATION_HARMONIC": LossVariant(
        code="L4_CONCENTRATION_HARMONIC",
        description="0.2 Pearson + 1.0 concentration + 0.1 harmonic; CE removed",
        pearson_weight=0.2,
        ce_weight=0.0,
        concentration_weight=1.0,
        harmonic_weight=0.1,
    ),
    "L5_CE_CONCENTRATION": LossVariant(
        code="L5_CE_CONCENTRATION",
        description="0.2 Pearson + 1.0 CE + 1.0 concentration",
        pearson_weight=0.2,
        ce_weight=1.0,
        concentration_weight=1.0,
        harmonic_weight=0.0,
    ),
}


def _band_power(
    spectrum: torch.Tensor,
    bpm_range: torch.Tensor,
    center_bpm: float,
    half_width_bpm: float,
) -> torch.Tensor:
    mask = torch.abs(bpm_range - float(center_bpm)) <= float(half_width_bpm)
    if not bool(mask.any()):
        return spectrum.new_tensor(0.0)
    return spectrum[0, mask].sum()


def _label_hr(
    prediction: torch.Tensor,
    label: torch.Tensor,
    fs: int,
    diff_flag: bool,
) -> float:
    prediction_1d = prediction.reshape(-1)
    label_1d = label.reshape(-1)
    _, label_hr = calculate_hr(
        prediction_1d.detach().cpu().numpy(),
        label_1d.detach().cpu().numpy(),
        fs=fs,
        diff_flag=diff_flag,
    )
    return float(label_hr)


def spectral_components(
    prediction: torch.Tensor,
    label: torch.Tensor,
    fs: int = 30,
    diff_flag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return official CE, concentration, and harmonic competition losses."""
    label_hr = _label_hr(prediction, label, fs, diff_flag)
    bpm_range = torch.arange(
        int(MINIMUM_BPM),
        int(MAXIMUM_BPM) + 1,
        device=prediction.device,
        dtype=torch.float32,
    )
    spectrum = _normalized_frequency_power(prediction.reshape(-1), fs, bpm_range)

    target_index = torch.tensor(
        [label_hr - MINIMUM_BPM],
        device=prediction.device,
        dtype=torch.long,
    )
    target_index = target_index.clamp(0, bpm_range.numel() - 1)
    ce_loss = F.cross_entropy(spectrum, target_index)

    correct_power = _band_power(
        spectrum,
        bpm_range,
        label_hr,
        CONCENTRATION_HALF_WIDTH_BPM,
    )
    concentration_loss = -torch.log(
        torch.clamp(correct_power, min=EPSILON, max=1.0)
    )

    fundamental_power = _band_power(
        spectrum,
        bpm_range,
        label_hr,
        HARMONIC_BAND_HALF_WIDTH_BPM,
    )
    harmonic_power = spectrum.new_tensor(0.0)
    valid_harmonics = 0
    for ratio in HARMONIC_RATIOS:
        harmonic_bpm = label_hr * ratio
        if MINIMUM_BPM <= harmonic_bpm <= MAXIMUM_BPM:
            harmonic_power = harmonic_power + _band_power(
                spectrum,
                bpm_range,
                harmonic_bpm,
                HARMONIC_BAND_HALF_WIDTH_BPM,
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
    return ce_loss, concentration_loss, harmonic_loss


class LossSuiteCriterion(nn.Module):
    """Compose Pearson, CE, concentration, and harmonic terms by variant."""

    def __init__(self, variant_code: str):
        super().__init__()
        if variant_code not in LOSS_VARIANTS:
            raise ValueError(f"Unknown loss variant: {variant_code}")
        self.variant = LOSS_VARIANTS[variant_code]
        self.pearson = NegativePearsonLoss()

    def components(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
        epoch: int,
        fs: int = 30,
        diff_flag: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del epoch
        pearson_loss = self.pearson(prediction, label)
        if torch.isnan(pearson_loss):
            pearson_loss = prediction.new_tensor(0.0)
        ce_loss, concentration_loss, harmonic_loss = spectral_components(
            prediction, label, fs, diff_flag
        )
        total_loss = (
            self.variant.pearson_weight * pearson_loss
            + self.variant.ce_weight * ce_loss
            + self.variant.concentration_weight * concentration_loss
            + self.variant.harmonic_weight * harmonic_loss
        )
        return (
            total_loss,
            pearson_loss,
            ce_loss,
            concentration_loss,
            harmonic_loss,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
        epoch: int,
        fs: int = 30,
        diff_flag: bool = False,
    ) -> torch.Tensor:
        return self.components(prediction, label, epoch, fs, diff_flag)[0]


def official_l0_parity_loss(
    prediction: torch.Tensor,
    label: torch.Tensor,
    epoch: int,
    fs: int = 30,
    diff_flag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose exact official-versus-shared-component parity for verification."""
    official = HybridLoss()(prediction, label, epoch, fs, diff_flag)
    pearson = NegativePearsonLoss()(prediction, label)
    if torch.isnan(pearson):
        pearson = prediction.new_tensor(0.0)
    ce, _, _ = spectral_components(prediction, label, fs, diff_flag)
    reconstructed = 0.2 * pearson + ce
    return official, reconstructed
