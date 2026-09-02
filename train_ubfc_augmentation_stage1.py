"""Train UBFC A1, A2, and A3 sequentially on the selected visible GPU."""

from cross_settings import UBFC_CROSS_MATCHED
from trainer_augmentation_stage1 import train_source


if __name__ == "__main__":
    train_source(UBFC_CROSS_MATCHED, "UBFC")

