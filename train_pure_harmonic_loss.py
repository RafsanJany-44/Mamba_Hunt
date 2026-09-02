"""Train PURE A0-Harmonic and A2-Harmonic sequentially on one GPU."""

from trainer_harmonic_loss import train_source


if __name__ == "__main__":
    train_source("PURE")

