"""Fail-fast verification for the final A4 multi-seed training suite."""

from __future__ import annotations

import torch

import dataset_pure_a4
import dataset_ubfc_a4
from cross_settings import PURE_CROSS_MATCHED, UBFC_CROSS_MATCHED
from loss_suite_stage2 import LossSuiteCriterion, build_loss_variants
from trainer_a4_final_multiseed import FINAL_CONFIGS, SEEDS, job_weights


EXPECTED = {
    "PURE": {
        "experiment": "L4_C100_H100",
        "variant": "L4_CONCENTRATION_HARMONIC",
        "concentration": 1.0,
        "harmonic": 1.0,
        "training_clips": 596,
        "validation_clips": 154,
    },
    "UBFC": {
        "experiment": "L3_C050",
        "variant": "L3_CONCENTRATION",
        "concentration": 0.5,
        "harmonic": 0.0,
        "training_clips": 378,
        "validation_clips": 105,
    },
}


def check_loss(source_name: str) -> None:
    expected = EXPECTED[source_name]
    actual = FINAL_CONFIGS[source_name]
    wanted = (
        expected["experiment"],
        expected["variant"],
        expected["concentration"],
        expected["harmonic"],
    )
    if actual != wanted:
        raise RuntimeError(f"{source_name} configuration mismatch: {actual} != {wanted}")

    overrides = job_weights(expected["concentration"], expected["harmonic"])
    variant = build_loss_variants(overrides)[expected["variant"]]
    weights = (
        variant.pearson_weight,
        variant.ce_weight,
        variant.concentration_weight,
        variant.harmonic_weight,
    )
    expected_weights = (
        0.2,
        0.0,
        expected["concentration"],
        expected["harmonic"],
    )
    if weights != expected_weights:
        raise RuntimeError(
            f"{source_name} effective weights mismatch: {weights} != {expected_weights}"
        )

    prediction = torch.randn(2, 160, requires_grad=True)
    label = torch.randn(2, 160)
    criterion = LossSuiteCriterion(expected["variant"], overrides)
    loss = torch.stack(
        [criterion(prediction[i], label[i], 0, 30, False) for i in range(2)]
    ).mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"{source_name} loss is not finite")
    loss.backward()
    if prediction.grad is None or not torch.isfinite(prediction.grad).all():
        raise RuntimeError(f"{source_name} gradients are not finite")

    print(
        f"{source_name}: {expected['experiment']} | "
        f"P/CE/C/H={weights[0]:.1f}/{weights[1]:.1f}/{weights[2]:.1f}/{weights[3]:.1f} "
        f"| finite gradient=PASSED"
    )


def check_loaders() -> None:
    dataset_pure_a4.SEED = SEEDS[0]
    pure_train, pure_valid = dataset_pure_a4.create_pure_a4_loaders(
        PURE_CROSS_MATCHED
    )
    dataset_ubfc_a4.SEED = SEEDS[0]
    ubfc_train, ubfc_valid = dataset_ubfc_a4.create_ubfc_a4_loaders(
        UBFC_CROSS_MATCHED
    )
    counts = {
        "PURE": (len(pure_train.dataset), len(pure_valid.dataset)),
        "UBFC": (len(ubfc_train.dataset), len(ubfc_valid.dataset)),
    }
    for source_name, (training, validation) in counts.items():
        expected = EXPECTED[source_name]
        wanted = (expected["training_clips"], expected["validation_clips"])
        if (training, validation) != wanted:
            raise RuntimeError(
                f"{source_name} loader counts mismatch: "
                f"{(training, validation)} != {wanted}"
            )
        print(
            f"{source_name}: training/validation clips={training}/{validation} | PASSED"
        )


def main() -> None:
    if SEEDS != (101, 102, 103, 104):
        raise RuntimeError(f"Unexpected seeds: {SEEDS}")
    if 100 in SEEDS:
        raise RuntimeError("Seed 100 must not be retrained")

    check_loss("PURE")
    check_loss("UBFC")
    check_loaders()

    names = {
        f"{source}_A4_{EXPECTED[source]['experiment']}_SEED{seed}"
        for source in EXPECTED
        for seed in SEEDS
    }
    if len(names) != 8:
        raise RuntimeError("Experiment names are not unique")

    print("=" * 88)
    print("A4 FINAL MULTI-SEED VERIFICATION: PASSED")
    print("=" * 88)
    print("Seeds              : 101, 102, 103, 104 (seed 100 reused)")
    print("Models to train    : 8 total (4 PURE + 4 UBFC)")
    print("Checkpoint policy  : best validation checkpoint only")
    print("Maximum epochs     : 100")
    print("Minimum epochs     : 30")
    print("Early-stop patience: 10")


if __name__ == "__main__":
    main()
