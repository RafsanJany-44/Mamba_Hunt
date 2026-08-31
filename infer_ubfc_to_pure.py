"""Inference on PURE using the standalone UBFC-trained cross checkpoint."""

from dataclasses import replace

from settings import PURE, UBFC_CROSS_CHECKPOINT
from trainer import evaluate_experiment


if __name__ == "__main__":
    print("Cross-dataset inference: UBFC -> PURE")
    full_pure = replace(PURE, test_begin=0.0, test_end=1.0)
    evaluate_experiment(full_pure, UBFC_CROSS_CHECKPOINT)
