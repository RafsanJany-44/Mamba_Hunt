"""Preflight verification for the paired UBFC A2 five-seed experiment."""

import torch

import dataset_stage1
from cross_settings import UBFC_CROSS_MATCHED
from loss import HybridLoss
from loss_suite_stage2 import LossSuiteCriterion
from trainer_ubfc_a2_multiseed import SEEDS, VARIANTS


def main():
    if SEEDS != (100, 101, 102, 103, 104):
        raise RuntimeError(f"Unexpected seed set: {SEEDS}")
    if set(VARIANTS) != {"L0", "L5"}:
        raise RuntimeError(f"Unexpected variants: {tuple(VARIANTS)}")

    generator_seeds = []
    loader_sizes = []
    for seed in SEEDS:
        dataset_stage1.SEED = seed
        train_loader, validation_loader = dataset_stage1.create_stage1_loaders(
            UBFC_CROSS_MATCHED, "UBFC", True
        )
        generator_seeds.append(train_loader.generator.initial_seed())
        loader_sizes.append((len(train_loader.dataset), len(validation_loader.dataset)))
    if tuple(generator_seeds) != SEEDS:
        raise RuntimeError(f"Loader seeds do not match: {generator_seeds}")
    if len(set(loader_sizes)) != 1 or loader_sizes[0] != (378, 105):
        raise RuntimeError(f"Unexpected loader sizes: {loader_sizes}")

    prediction = torch.randn(160, requires_grad=True)
    label = torch.randn(160)
    l0 = HybridLoss()(prediction, label, 0, 30, False)
    l5 = LossSuiteCriterion("L5_CE_CONCENTRATION")(
        prediction, label, 0, 30, False
    )
    if not torch.isfinite(l0) or not torch.isfinite(l5):
        raise RuntimeError("Non-finite loss in preflight")
    (l0 + l5).backward()
    if prediction.grad is None or not torch.isfinite(prediction.grad).all():
        raise RuntimeError("Finite-gradient check failed")

    print("=" * 82)
    print("UBFC A2 PAIRED FIVE-SEED PREFLIGHT: PASSED")
    print("=" * 82)
    print(f"Seeds              : {SEEDS}")
    print(f"Loader seeds       : {tuple(generator_seeds)}")
    print(f"Train/valid clips  : {loader_sizes[0][0]}/{loader_sizes[0][1]}")
    print("L0                 : official 0.2 Pearson + CE")
    print("L5                 : official 0.2 Pearson + CE + concentration")
    print("A2                 : 50% original / 50% offline sampling")
    print("Finite gradients    : PASSED")
    print("=" * 82)


if __name__ == "__main__":
    main()
