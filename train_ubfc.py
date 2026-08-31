"""Train and test the simplified baseline on the official UBFC split."""

from settings import UBFC
from trainer import train_experiment


if __name__ == "__main__":
    train_experiment(UBFC)
