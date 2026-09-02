"""Train PURE A1, A2, and A3 sequentially on the selected visible GPU."""

from cross_settings import PURE_CROSS_MATCHED
from trainer_augmentation_stage1 import train_source


if __name__ == "__main__":
    train_source(PURE_CROSS_MATCHED, "PURE")

