"""Evaluate the final simplified PURE checkpoint."""

from settings import PURE
from trainer import evaluate_experiment


if __name__ == "__main__":
    evaluate_experiment(PURE)
