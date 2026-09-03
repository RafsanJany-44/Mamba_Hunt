"""Numerical verification for the Stage-2 RhythmMamba loss suite."""

from __future__ import annotations

import math

import numpy as np
import torch

from loss_suite_stage2 import (
    LOSS_VARIANTS,
    LossSuiteCriterion,
    official_l0_parity_loss,
    spectral_components,
)


FS = 30
LENGTH = 160
GT_BPM = 72.0


def signal_at_bpm(bpm: float) -> torch.Tensor:
    time = torch.arange(LENGTH, dtype=torch.float32) / FS
    signal = torch.sin(2.0 * math.pi * (bpm / 60.0) * time)
    return signal


def main() -> None:
    torch.manual_seed(20260902)
    np.random.seed(20260902)
    label = signal_at_bpm(GT_BPM)
    fundamental = signal_at_bpm(GT_BPM)
    harmonic_1p5 = signal_at_bpm(GT_BPM * 1.5)
    harmonic_2 = signal_at_bpm(GT_BPM * 2.0)

    official, reconstructed = official_l0_parity_loss(
        fundamental, label, epoch=0, fs=FS
    )
    parity_error = float(torch.abs(official - reconstructed).item())
    if parity_error > 1e-7:
        raise RuntimeError(f"Official L0 parity failed: {parity_error}")

    values = {}
    for name, prediction in (
        ("fundamental", fundamental),
        ("1.5x", harmonic_1p5),
        ("2x", harmonic_2),
    ):
        ce, concentration, harmonic = spectral_components(prediction, label, FS)
        values[name] = {
            "ce": float(ce.item()),
            "concentration": float(concentration.item()),
            "harmonic": float(harmonic.item()),
        }
        if not all(math.isfinite(value) for value in values[name].values()):
            raise RuntimeError(f"Non-finite spectral component for {name}")

    if not values["fundamental"]["concentration"] < values["1.5x"]["concentration"]:
        raise RuntimeError("Concentration loss did not penalize the 1.5x signal")
    if not values["fundamental"]["concentration"] < values["2x"]["concentration"]:
        raise RuntimeError("Concentration loss did not penalize the 2x signal")
    if not values["fundamental"]["harmonic"] < values["1.5x"]["harmonic"]:
        raise RuntimeError("Harmonic loss did not penalize the 1.5x signal")
    if not values["fundamental"]["harmonic"] < values["2x"]["harmonic"]:
        raise RuntimeError("Harmonic loss did not penalize the 2x signal")

    gradient_norms = {}
    for variant_code in LOSS_VARIANTS:
        prediction = (
            fundamental + 0.15 * torch.randn_like(fundamental)
        ).detach().requires_grad_(True)
        criterion = LossSuiteCriterion(variant_code)
        total = criterion(prediction, label, epoch=0, fs=FS)
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite total loss for {variant_code}")
        total.backward()
        if prediction.grad is None or not torch.isfinite(prediction.grad).all():
            raise RuntimeError(f"Invalid gradient for {variant_code}")
        norm = float(prediction.grad.norm().item())
        if norm <= 0.0:
            raise RuntimeError(f"Zero gradient for {variant_code}")
        gradient_norms[variant_code] = norm

    print("=" * 78)
    print("STAGE-2 LOSS-SUITE VERIFICATION: PASSED")
    print("=" * 78)
    print(f"Official L0 parity error : {parity_error:.10g}")
    for name in ("fundamental", "1.5x", "2x"):
        item = values[name]
        print(
            f"{name:11s} | CE={item['ce']:.8f} | "
            f"C={item['concentration']:.8f} | H={item['harmonic']:.8f}"
        )
    print("Finite nonzero gradients:")
    for variant_code, norm in gradient_norms.items():
        print(f"  {variant_code:30s}: {norm:.8f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
