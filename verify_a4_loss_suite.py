"""Preflight verification for the controlled A4 + L1-L4 experiment."""

from __future__ import annotations

import torch

from cross_settings import PURE_CROSS_MATCHED, UBFC_CROSS_MATCHED
from dataset_pure_a4 import create_pure_a4_loaders
from dataset_ubfc_a4 import create_ubfc_a4_loaders
from loss_suite_stage2 import (
    LossSuiteCriterion,
    build_loss_variants,
    official_l0_parity_loss,
)
from trainer_a4_loss_suite import LOSS_WEIGHT_OVERRIDES, VARIANT_ORDER


EXPECTED = {
    "L1_OFFICIAL_HARMONIC": (0.2, 1.0, 0.0, 1.0),
    "L2_HARMONIC_REPLACE": (0.2, 0.0, 0.0, 1.0),
    "L3_CONCENTRATION": (0.2, 0.0, 1.0, 0.0),
    "L4_CONCENTRATION_HARMONIC": (0.2, 0.0, 1.0, 1.0),
}


def verify_weights() -> None:
    variants = build_loss_variants(LOSS_WEIGHT_OVERRIDES)
    if tuple(VARIANT_ORDER) != tuple(EXPECTED):
        raise RuntimeError("The planned L1-L4 order is incorrect")
    for code, expected in EXPECTED.items():
        variant = variants[code]
        actual = (
            variant.pearson_weight,
            variant.ce_weight,
            variant.concentration_weight,
            variant.harmonic_weight,
        )
        if actual != expected:
            raise RuntimeError(f"Weight mismatch for {code}: {actual} != {expected}")


def verify_losses() -> None:
    time = torch.arange(160, dtype=torch.float32) / 30.0
    label = torch.sin(2.0 * torch.pi * 1.2 * time)
    base_prediction = label + 0.05 * torch.sin(2.0 * torch.pi * 2.4 * time)

    official, reconstructed = official_l0_parity_loss(
        base_prediction.clone(), label, epoch=0, fs=30, diff_flag=False
    )
    parity_error = float(torch.abs(official - reconstructed).item())
    if parity_error != 0.0:
        raise RuntimeError(f"Official L0 parity failed: {parity_error}")

    print(f"Official L0 parity error: {parity_error:g}")
    for code in VARIANT_ORDER:
        prediction = base_prediction.clone().requires_grad_(True)
        criterion = LossSuiteCriterion(code, LOSS_WEIGHT_OVERRIDES)
        total = criterion(prediction, label, epoch=0, fs=30, diff_flag=False)
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite loss for {code}")
        total.backward()
        if prediction.grad is None or not torch.isfinite(prediction.grad).all():
            raise RuntimeError(f"Invalid gradient for {code}")
        gradient = float(prediction.grad.abs().mean().item())
        if gradient <= 0.0:
            raise RuntimeError(f"Zero gradient for {code}")
        print(f"{code:34s} | loss={total.item():.8f} | mean|grad|={gradient:.8f}")


def verify_a4_data() -> None:
    pure_train, pure_valid = create_pure_a4_loaders(PURE_CROSS_MATCHED)
    ubfc_train, ubfc_valid = create_ubfc_a4_loaders(UBFC_CROSS_MATCHED)
    expected_counts = {
        "PURE": (len(pure_train.dataset), len(pure_valid.dataset), 596, 154),
        "UBFC": (len(ubfc_train.dataset), len(ubfc_valid.dataset), 378, 105),
    }
    for source, (train_count, valid_count, expected_train, expected_valid) in expected_counts.items():
        if (train_count, valid_count) != (expected_train, expected_valid):
            raise RuntimeError(
                f"{source} count mismatch: {(train_count, valid_count)} != "
                f"{(expected_train, expected_valid)}"
            )
        print(f"{source:4s} A4 clips: train={train_count}, clean validation={valid_count}")


def main() -> None:
    print("=" * 88)
    print("A4 + L1-L4 LOSS-SUITE VERIFICATION")
    print("=" * 88)
    verify_weights()
    verify_losses()
    verify_a4_data()
    print("=" * 88)
    print("A4 + L1-L4 LOSS-SUITE VERIFICATION: PASSED")
    print("=" * 88)


if __name__ == "__main__":
    main()
