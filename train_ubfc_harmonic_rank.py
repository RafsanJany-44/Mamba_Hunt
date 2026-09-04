"""Train UBFC A2 with official loss plus conditional harmonic ranking."""

from trainer_harmonic_rank import train_source


if __name__ == "__main__":
    train_source("UBFC")
