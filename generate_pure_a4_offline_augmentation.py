"""Generate all four A4 variants for every PURE 0.0-0.8 training clip."""

import generate_ubfc_a4_offline_augmentation as generator
from settings import PURE


if __name__ == "__main__":
    generator.DATASET_NAME = "PURE"
    generator.EXPERIMENT = PURE
    generator.EXPECTED_RECORDINGS = 47
    generator.EXPECTED_CLIPS = 596
    generator.main()
