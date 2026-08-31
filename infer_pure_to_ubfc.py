"""Inference on UBFC using the standalone PURE-trained cross checkpoint."""

from dataclasses import replace

from settings import PURE_CROSS_CHECKPOINT, UBFC
from trainer import evaluate_experiment


if __name__ == "__main__":
    print("Cross-dataset inference: PURE -> UBFC")
    full_ubfc = replace(UBFC, test_begin=0.0, test_end=1.0)
    evaluate_experiment(full_ubfc, PURE_CROSS_CHECKPOINT)
