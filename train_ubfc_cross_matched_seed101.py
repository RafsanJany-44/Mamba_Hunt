"""Train the official-matched UBFC experiment with seed 101."""

from settings import Experiment, UBFC
import cross_trainer
from trainer import set_reproducible


SEED = 101

UBFC_CROSS_MATCHED_SEED101 = Experiment(
    name="UBFC_CROSS_MATCHED_SEED101",
    cache_parent=UBFC.cache_parent,
    train_begin=0.0,
    train_end=0.8,
    test_begin=0.8,
    test_end=1.0,
    inference_batch_size=UBFC.inference_batch_size,
)


def set_seed_101():
    set_reproducible(SEED)


if __name__ == "__main__":
    cross_trainer.set_reproducible = set_seed_101
    print("Training seed:", SEED)
    cross_trainer.train_cross_experiment(UBFC_CROSS_MATCHED_SEED101)

