"""The loss terms used by the official RhythmMamba trainer."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import calculate_hr


class NegativePearsonLoss(nn.Module):
    def forward(self, prediction, label):
        prediction = prediction.view(1, -1)
        label = label.view(1, -1)

        sum_x = torch.sum(prediction)
        sum_y = torch.sum(label)
        sum_xy = torch.sum(prediction * label)
        sum_x2 = torch.sum(prediction**2)
        sum_y2 = torch.sum(label**2)
        length = prediction.shape[1]

        correlation = (length * sum_xy - sum_x * sum_y) / torch.sqrt(
            (length * sum_x2 - sum_x**2) * (length * sum_y2 - sum_y**2)
        )
        return 1 - correlation


def _normalized_frequency_power(output, fs, bpm_range):
    """Official sinusoidal spectral projection over the requested BPM bins."""
    output = output.view(1, -1)
    length = output.shape[1]
    device = output.device

    unit_per_hz = fs / length
    frequencies_hz = bpm_range.to(device=device, dtype=torch.float32) / 60.0
    k = frequencies_hz / unit_per_hz

    two_pi_n_over_n = (
        2
        * math.pi
        * torch.arange(length, device=device, dtype=torch.float32)
        / length
    )
    window = torch.from_numpy(np.hanning(length)).to(
        device=device, dtype=torch.float32
    )

    windowed = output * window.view(1, -1)
    k = k.view(1, -1, 1)
    phase = two_pi_n_over_n.view(1, 1, -1)
    windowed = windowed.view(1, 1, -1)

    power = torch.sum(windowed * torch.sin(k * phase), dim=-1) ** 2
    power = power + torch.sum(windowed * torch.cos(k * phase), dim=-1) ** 2
    return power / power.sum()


def frequency_cross_entropy(prediction, label, fs=30, diff_flag=False):
    _, label_hr = calculate_hr(
        prediction.detach().cpu().numpy(),
        label.detach().cpu().numpy(),
        fs=fs,
        diff_flag=diff_flag,
    )

    bpm_range = torch.arange(45, 150, device=prediction.device, dtype=torch.float32)
    spectrum = _normalized_frequency_power(prediction, fs, bpm_range)
    target_index = torch.tensor(
        [label_hr - 45], device=prediction.device, dtype=torch.long
    )
    return F.cross_entropy(spectrum, target_index)


class HybridLoss(nn.Module):
    """Official objective: 0.2 x negative Pearson + frequency CE."""

    def __init__(self):
        super().__init__()
        self.pearson = NegativePearsonLoss()

    def forward(self, prediction, label, epoch, fs=30, diff_flag=False):
        time_loss = self.pearson(prediction, label)
        if torch.isnan(time_loss):
            time_loss = prediction.new_tensor(0.0)
        frequency_loss = frequency_cross_entropy(
            prediction.squeeze(-1), label.squeeze(-1), fs, diff_flag
        )
        return 0.2 * time_loss + frequency_loss
