"""Run the PURE A4 seed-100 pilot through the verified generic A4 trainer."""

import dataset_pure_a4
import train_ubfc_a4_l0_seed100 as trainer
from cross_settings import PURE_CROSS_MATCHED


if __name__ == "__main__":
    trainer.NAME = "PURE_A4_L0_SEED100"
    trainer.SOURCE_DATASET = "PURE"
    trainer.MODEL_FAMILY_DIRECTORY = "pure_a4_pilot"
    trainer.SEED = 100
    trainer.UBFC_CROSS_MATCHED = PURE_CROSS_MATCHED
    trainer.dataset_ubfc_a4 = dataset_pure_a4
    trainer.main()
