"""Mandatory pre-training checks for the harmonic-aware loss."""

import math

import torch

from loss import HybridLoss
from loss_harmonic import HarmonicAwareHybridLoss, harmonic_competition_loss


FS = 30
LENGTH = 160
TRUE_BPM = 60.0


def sine(bpm: float) -> torch.Tensor:
    time = torch.arange(LENGTH, dtype=torch.float32) / FS
    return torch.sin(2.0 * math.pi * (bpm / 60.0) * time)


def main() -> None:
    label = sine(TRUE_BPM)
    fundamental = sine(TRUE_BPM)
    harmonic_1p5 = sine(TRUE_BPM * 1.5)
    harmonic_2x = sine(TRUE_BPM * 2.0)

    fundamental_loss = harmonic_competition_loss(fundamental, label, FS)
    harmonic_1p5_loss = harmonic_competition_loss(harmonic_1p5, label, FS)
    harmonic_2x_loss = harmonic_competition_loss(harmonic_2x, label, FS)
    if not fundamental_loss < harmonic_1p5_loss:
        raise RuntimeError("1.5x ordering check failed")
    if not fundamental_loss < harmonic_2x_loss:
        raise RuntimeError("2x ordering check failed")

    prediction = (fundamental + 0.25 * harmonic_1p5).requires_grad_(True)
    criterion = HarmonicAwareHybridLoss()
    total, base, harmonic = criterion.components(prediction, label, 0, FS, False)
    total.backward()
    if not torch.isfinite(total):
        raise RuntimeError("Total loss is non-finite")
    if prediction.grad is None or not torch.isfinite(prediction.grad).all():
        raise RuntimeError("Harmonic-aware gradient is missing or non-finite")

    parity_prediction = prediction.detach().clone()
    verified_base = HybridLoss()(parity_prediction, label, 0, FS, False)
    zero_weight = HarmonicAwareHybridLoss(harmonic_weight=0.0)(
        parity_prediction, label, 0, FS, False
    )
    if not torch.equal(verified_base, zero_weight):
        raise RuntimeError("Zero-weight L0 parity check failed")

    print("=" * 78)
    print("HARMONIC-AWARE LOSS VERIFICATION: PASSED")
    print("=" * 78)
    print(f"Fundamental harmonic loss : {fundamental_loss.item():.8f}")
    print(f"1.5x harmonic loss        : {harmonic_1p5_loss.item():.8f}")
    print(f"2x harmonic loss          : {harmonic_2x_loss.item():.8f}")
    print(f"Example base loss         : {base.item():.8f}")
    print(f"Example harmonic loss     : {harmonic.item():.8f}")
    print(f"Example total loss        : {total.item():.8f}")
    print("Zero-weight L0 parity     : PASSED")
    print("Finite-gradient check     : PASSED")


if __name__ == "__main__":
    main()

