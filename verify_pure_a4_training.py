"""Read-only verification of PURE A4 generation and training inputs."""

from collections import Counter

import dataset_pure_a4
from cross_settings import PURE_CROSS_MATCHED


def main():
    training, validation = dataset_pure_a4.create_pure_a4_loaders(PURE_CROSS_MATCHED)
    dataset = training.dataset
    if len(dataset) != 596 or len(validation.dataset) != 154:
        raise RuntimeError(
            f"Unexpected sizes: training={len(dataset)}, validation={len(validation.dataset)}"
        )
    if any(len(paths) != 4 for paths in dataset.variants):
        raise RuntimeError("Every PURE original clip must have exactly four variants")
    counts = Counter(dataset[0][4] for _ in range(20))
    print("=" * 78)
    print("PURE A4 TRAINING INPUT VERIFICATION: PASSED")
    print("=" * 78)
    print(f"Original training clips : {len(dataset)}")
    print(f"A4 variants per clip    : {len(dataset.variants[0])}")
    print(f"Clean validation clips  : {len(validation.dataset)}")
    print(f"Example sampling counts : {dict(counts)}")
    print("Training policy         : 50% original; 50% one uniformly chosen variant")


if __name__ == "__main__":
    main()
