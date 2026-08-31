"""Separate official-matched cross-training experiment definitions."""

from settings import Experiment, PURE, UBFC


PURE_CROSS_MATCHED = Experiment(
    name="PURE_CROSS_MATCHED",
    cache_parent=PURE.cache_parent,
    train_begin=0.0,
    train_end=0.8,
    test_begin=0.8,
    test_end=1.0,
    inference_batch_size=PURE.inference_batch_size,
)


UBFC_CROSS_MATCHED = Experiment(
    name="UBFC_CROSS_MATCHED",
    cache_parent=UBFC.cache_parent,
    train_begin=0.0,
    train_end=0.8,
    test_begin=0.8,
    test_end=1.0,
    inference_batch_size=UBFC.inference_batch_size,
)
