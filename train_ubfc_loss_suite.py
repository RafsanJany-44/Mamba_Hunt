"""Train the eight new UBFC Stage-2 loss/augmentation models on one GPU."""

from trainer_loss_suite import train_source


if __name__ == "__main__":
    train_source("UBFC")
