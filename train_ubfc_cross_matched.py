"""Train UBFC exactly as the official cross-dataset configuration."""

from cross_settings import UBFC_CROSS_MATCHED
from cross_trainer import train_cross_experiment


if __name__ == "__main__":
    train_cross_experiment(UBFC_CROSS_MATCHED)
