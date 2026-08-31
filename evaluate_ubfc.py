"""Evaluate the final simplified UBFC checkpoint."""

from settings import UBFC
from trainer import evaluate_experiment


if __name__ == "__main__":
    evaluate_experiment(UBFC)
