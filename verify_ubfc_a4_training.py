"""Read-only verification of the generated UBFC A4 dataset and loader."""

from collections import Counter

import dataset_ubfc_a4
from cross_settings import UBFC_CROSS_MATCHED


def main():
    training, validation = dataset_ubfc_a4.create_ubfc_a4_loaders(UBFC_CROSS_MATCHED)
    dataset = training.dataset
    if len(dataset) != 378 or len(validation.dataset) != 105:
        raise RuntimeError(
            f"Unexpected sizes: training={len(dataset)}, validation={len(validation.dataset)}"
        )
    if any(len(paths) != 4 for paths in dataset.variants):
        raise RuntimeError("Every original clip must have exactly four variants")
    counts = Counter()
    for _ in range(20):
        counts[dataset[0][4]] += 1
    print("=" * 78)
    print("UBFC A4 TRAINING INPUT VERIFICATION: PASSED")
    print("=" * 78)
    print(f"Original training clips : {len(dataset)}")
    print(f"A4 variants per clip    : {len(dataset.variants[0])}")
    print(f"Clean validation clips  : {len(validation.dataset)}")
    print(f"Example sampling counts : {dict(counts)}")
    print("Training policy         : 50% original; 50% one uniformly chosen variant")
    print("Validation policy       : clean original clips only")


if __name__ == "__main__":
    main()
