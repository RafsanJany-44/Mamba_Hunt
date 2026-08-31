"""Train PURE exactly as the official cross-dataset configuration."""

from cross_settings import PURE_CROSS_MATCHED
from cross_trainer import train_cross_experiment


if __name__ == "__main__":
    train_cross_experiment(PURE_CROSS_MATCHED)
