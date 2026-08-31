"""Train and test the simplified baseline on the official PURE split."""

from settings import PURE
from trainer import train_experiment


if __name__ == "__main__":
    train_experiment(PURE)
