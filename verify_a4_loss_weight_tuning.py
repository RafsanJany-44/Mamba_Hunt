"""Preflight verification for A4 L3/L4 loss-weight tuning."""

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
from trainer_a4_loss_weight_tuning import WEIGHT_TUNING_JOBS, job_weights


def verify_weights() -> None:
    if len(WEIGHT_TUNING_JOBS) != 8:
        raise RuntimeError("Exactly eight tuning jobs are required")
    experiment_codes = [job[0] for job in WEIGHT_TUNING_JOBS]
    if len(set(experiment_codes)) != len(experiment_codes):
        raise RuntimeError("Duplicate tuning experiment code")
    for experiment_code, code, concentration, harmonic in WEIGHT_TUNING_JOBS:
        weights = job_weights(concentration, harmonic)
        variant = build_loss_variants(weights)[code]
        expected = (0.2, 0.0, concentration, harmonic)
        actual = (
            variant.pearson_weight,
            variant.ce_weight,
            variant.concentration_weight,
            variant.harmonic_weight,
        )
        if actual != expected:
            raise RuntimeError(
                f"Weight mismatch for {experiment_code}: {actual} != {expected}"
            )


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
    for experiment_code, code, concentration, harmonic in WEIGHT_TUNING_JOBS:
        weights = job_weights(concentration, harmonic)
        prediction = base_prediction.clone().requires_grad_(True)
        criterion = LossSuiteCriterion(code, weights)
        total = criterion(prediction, label, epoch=0, fs=30, diff_flag=False)
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite loss for {experiment_code}")
        total.backward()
        if prediction.grad is None or not torch.isfinite(prediction.grad).all():
            raise RuntimeError(f"Invalid gradient for {experiment_code}")
        gradient = float(prediction.grad.abs().mean().item())
        if gradient <= 0.0:
            raise RuntimeError(f"Zero gradient for {experiment_code}")
        print(
            f"{experiment_code:18s} | loss={total.item():.8f} | "
            f"mean|grad|={gradient:.8f}"
        )


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
    print("A4 L3/L4 LOSS-WEIGHT TUNING VERIFICATION")
    print("=" * 88)
    verify_weights()
    verify_losses()
    verify_a4_data()
    print("=" * 88)
    print("A4 L3/L4 LOSS-WEIGHT TUNING VERIFICATION: PASSED")
    print("=" * 88)


if __name__ == "__main__":
    main()
