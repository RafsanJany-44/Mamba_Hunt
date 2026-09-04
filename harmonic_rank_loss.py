"""Conditional harmonic-ranking extension of the official RhythmMamba loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import HybridLoss, _normalized_frequency_power
from metrics import calculate_hr


MINIMUM_BPM = 45.0
MAXIMUM_BPM = 149.0
HARMONIC_RATIOS = (0.5, 1.5, 2.0)
HALF_WIDTH_BPM = 2.0
MARGIN = 0.05
RANK_WEIGHT = 0.02
EPSILON = 1e-8


def _band_power(spectrum, bpm_range, center_bpm, half_width_bpm):
    mask = torch.abs(bpm_range - float(center_bpm)) <= float(half_width_bpm)
    if not bool(mask.any()):
        return spectrum.new_tensor(0.0)
    return spectrum[0, mask].sum()


def conditional_harmonic_rank_loss(
    prediction,
    label,
    fs=30,
    diff_flag=False,
    margin=MARGIN,
):
    """Require GT-fundamental power to exceed valid 0.5x/1.5x/2x peaks.

    Each hinge term is zero once P(f_gt) >= P(f_wrong) + margin. Terms are
    averaged over valid harmonic locations so samples have comparable scale.
    """
    prediction = prediction.reshape(-1)
    label = label.reshape(-1)
    _, label_hr = calculate_hr(
        prediction.detach().cpu().numpy(),
        label.detach().cpu().numpy(),
        fs=fs,
        diff_flag=diff_flag,
    )
    label_hr = float(label_hr)
    bpm_range = torch.arange(
        int(MINIMUM_BPM),
        int(MAXIMUM_BPM) + 1,
        device=prediction.device,
        dtype=torch.float32,
    )
    spectrum = _normalized_frequency_power(prediction, fs, bpm_range)
    fundamental = _band_power(
        spectrum, bpm_range, label_hr, HALF_WIDTH_BPM
    )

    terms = []
    for ratio in HARMONIC_RATIOS:
        wrong_bpm = label_hr * ratio
        if MINIMUM_BPM <= wrong_bpm <= MAXIMUM_BPM:
            wrong = _band_power(
                spectrum, bpm_range, wrong_bpm, HALF_WIDTH_BPM
            )
            terms.append(F.relu(float(margin) + wrong - fundamental))
    if not terms:
        return spectrum.sum() * 0.0
    return torch.stack(terms).mean()


class OfficialWithConditionalHarmonicRank(nn.Module):
    """L = L_official + 0.02 * L_conditional_harmonic_rank."""

    def __init__(self, rank_weight=RANK_WEIGHT, margin=MARGIN):
        super().__init__()
        self.official = HybridLoss()
        self.rank_weight = float(rank_weight)
        self.margin = float(margin)

    def components(self, prediction, label, epoch, fs=30, diff_flag=False):
        official = self.official(prediction, label, epoch, fs, diff_flag)
        rank = conditional_harmonic_rank_loss(
            prediction, label, fs, diff_flag, self.margin
        )
        total = official + self.rank_weight * rank
        return total, official, rank

    def forward(self, prediction, label, epoch, fs=30, diff_flag=False):
        return self.components(prediction, label, epoch, fs, diff_flag)[0]
