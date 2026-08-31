"""Fixed experiment settings for the simplified RhythmMamba baseline.

Edit this file directly when an experiment setting must change.  The
simplified project intentionally uses neither YAML nor command-line options.
"""

from dataclasses import dataclass
from pathlib import Path


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# This follows the Mamba_Hunt folder wherever it is moved.
MAMBA_HUNT_ROOT = Path(__file__).resolve().parent
# PREPROCESSED_ROOT = Path(
#     "/media/data/rPPG/rPPG_Data/Mamba_Hunt/RhythmMamba_Preprocessed"
# )

PREPROCESSED_ROOT = Path(
    "/home/rafsan/Documents/Data/Mamba_Hunt_Data/"
    "RhythmMamba_Preprocessed_Independent"
)


OUTPUT_ROOT = MAMBA_HUNT_ROOT / "results"
CHECKPOINT_ROOT = MAMBA_HUNT_ROOT / "Official_Checkpoints/"

# Standalone pretrained weights used for official cross-dataset inference.
PURE_CROSS_CHECKPOINT = CHECKPOINT_ROOT / "PURE_cross_RhythmMamba.pth"
UBFC_CROSS_CHECKPOINT = CHECKPOINT_ROOT / "UBFC_cross_RhythmMamba.pth"


# -----------------------------------------------------------------------------
# Settings shared by PURE and UBFC-rPPG
# -----------------------------------------------------------------------------

DEVICE = "cuda:0"
NUM_GPUS = 1
SEED = 100
FS = 30
CHUNK_LENGTH = 160
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 3e-4
USE_AUGMENTATION = True
TRAIN_WORKERS = 16
TEST_WORKERS = 4
SAVE_EVERY_EPOCH = True


@dataclass(frozen=True)
class Experiment:
    """Dataset-specific values that were previously stored in YAML."""

    name: str
    cache_parent: Path
    train_begin: float
    train_end: float
    test_begin: float
    test_end: float
    inference_batch_size: int

    @property
    def model_dir(self) -> Path:
        return OUTPUT_ROOT / "models" / self.name

    @property
    def log_dir(self) -> Path:
        return OUTPUT_ROOT / "logs"

    @property
    def final_checkpoint(self) -> Path:
        return self.model_dir / f"{self.name}_RhythmMamba_Epoch{EPOCHS - 1}.pth"


PURE = Experiment(
    name="PURE",
    cache_parent=PREPROCESSED_ROOT / "PURE",
    train_begin=0.0,
    train_end=0.6,
    test_begin=0.6,
    test_end=1.0,
    inference_batch_size=16,
)

UBFC = Experiment(
    name="UBFC",
    cache_parent=PREPROCESSED_ROOT / "UBFC",
    train_begin=0.0,
    train_end=0.72,
    test_begin=0.72,
    test_end=1.0,
    inference_batch_size=2,
)

# Additional datasets are currently configured for complete-dataset external
# evaluation.  Their 70/30 subject-wise manifests are also available for later
# training experiments.
BH = Experiment(
    name="BH",
    cache_parent=PREPROCESSED_ROOT / "BH",
    train_begin=0.0,
    train_end=0.7,
    test_begin=0.0,
    test_end=1.0,
    inference_batch_size=4,
)

UBFC_PHYS = Experiment(
    name="UBFC_PHYS",
    cache_parent=PREPROCESSED_ROOT / "UBFC_PHYS",
    train_begin=0.0,
    train_end=0.7,
    test_begin=0.0,
    test_end=1.0,
    inference_batch_size=2,
)

COHFACE = Experiment(
    name="COHFACE",
    cache_parent=PREPROCESSED_ROOT / "COHFACE",
    train_begin=0.0,
    train_end=0.7,
    test_begin=0.0,
    test_end=1.0,
    inference_batch_size=4,
)

TOKYOTECH = Experiment(
    name="TOKYOTECH",
    cache_parent=PREPROCESSED_ROOT / "TOKYOTECH",
    train_begin=0.0,
    train_end=0.7,
    test_begin=0.0,
    test_end=1.0,
    inference_batch_size=4,
)
