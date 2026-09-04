"""Synthetic checks for the Stage-3 conditional harmonic-ranking loss."""

import math

import torch

from harmonic_rank_loss import (
    OfficialWithConditionalHarmonicRank,
    conditional_harmonic_rank_loss,
)
from loss import HybridLoss


def wave(bpm, length=160, fs=30.0):
    t = torch.arange(length, dtype=torch.float32) / fs
    return torch.sin(2.0 * math.pi * (bpm / 60.0) * t)


def main():
    label = wave(72.0)
    examples = {
        "fundamental": wave(72.0),
        "1.5x": wave(108.0),
        "2x": wave(144.0),
    }
    values = {
        name: float(conditional_harmonic_rank_loss(pred, label).item())
        for name, pred in examples.items()
    }
    if values["fundamental"] > 1e-4:
        raise RuntimeError(f"Fundamental rank loss is not near zero: {values}")
    if values["1.5x"] <= values["fundamental"]:
        raise RuntimeError(f"1.5x ranking check failed: {values}")
    if values["2x"] <= values["fundamental"]:
        raise RuntimeError(f"2x ranking check failed: {values}")

    pred = (wave(108.0) + 0.02 * torch.randn(160)).requires_grad_(True)
    criterion = OfficialWithConditionalHarmonicRank()
    total, official, rank = criterion.components(pred, label, 0)
    total.backward()
    if pred.grad is None or not torch.isfinite(pred.grad).all():
        raise RuntimeError("Finite-gradient check failed")
    if float(pred.grad.abs().sum()) <= 0.0:
        raise RuntimeError("Nonzero-gradient check failed")

    zero = OfficialWithConditionalHarmonicRank(rank_weight=0.0)(pred, label, 0)
    reference = HybridLoss()(pred, label, 0)
    parity = float(torch.abs(zero - reference).item())
    if parity != 0.0:
        raise RuntimeError(f"Zero-weight official parity failed: {parity}")

    print("=" * 78)
    print("CONDITIONAL HARMONIC-RANK LOSS VERIFICATION: PASSED")
    print("=" * 78)
    print(f"Fundamental rank loss : {values['fundamental']:.8f}")
    print(f"1.5x rank loss        : {values['1.5x']:.8f}")
    print(f"2x rank loss          : {values['2x']:.8f}")
    print(f"Example official loss : {float(official.item()):.8f}")
    print(f"Example rank loss     : {float(rank.item()):.8f}")
    print(f"Example total loss    : {float(total.item()):.8f}")
    print(f"Zero-weight parity    : {parity:.8g}")
    print(f"Gradient L1 norm      : {float(pred.grad.abs().sum()):.8f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
