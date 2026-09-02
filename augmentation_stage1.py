"""Additional Stage-1 online augmentation for RhythmMamba."""

from __future__ import annotations

import torch


class OnlineRGBChannelGain:
    """Clip-consistent RGB gains that preserve temporal frequency.

    One gain is drawn per channel and per selected sample. Gains are normalized
    to mean one so the operation changes channel balance rather than overall
    brightness. The same gains are applied to every frame and pixel in a clip.
    """

    def __init__(self, probability: float = 0.5, low: float = 0.85, high: float = 1.15):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if not 0.0 < low <= high:
            raise ValueError("gain range must be positive and ordered")
        self.probability = float(probability)
        self.low = float(low)
        self.high = float(high)

    def __call__(self, data: torch.Tensor) -> tuple[torch.Tensor, int]:
        if data.ndim != 5 or data.shape[2] != 3:
            raise ValueError(f"Expected [B,T,3,H,W], got {tuple(data.shape)}")
        batch_size = data.shape[0]
        selected = torch.rand(batch_size, device=data.device) < self.probability
        gains = torch.empty(
            (batch_size, 3), device=data.device, dtype=data.dtype
        ).uniform_(self.low, self.high)
        gains = gains / gains.mean(dim=1, keepdim=True)
        gains[~selected] = 1.0
        return data * gains[:, None, :, None, None], int(selected.sum().item())

