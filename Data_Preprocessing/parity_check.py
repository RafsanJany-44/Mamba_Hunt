"""Compare independent smoke clips with the verified official-generated cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .dataset_registry import get_adapter
    from .settings import (
        DATASET_SETTINGS,
        DATASETS_TO_PROCESS,
        REFERENCE_CACHE_ROOT,
        SMOKE_CACHE_ROOT,
    )
except ImportError:
    from dataset_registry import get_adapter
    from settings import (
        DATASET_SETTINGS,
        DATASETS_TO_PROCESS,
        REFERENCE_CACHE_ROOT,
        SMOKE_CACHE_ROOT,
    )


def _matching_files(root: Path, dataset_name: str, saved_id: str, kind: str) -> list[Path]:
    files = sorted((root / dataset_name).rglob(f"{saved_id}_{kind}*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No {saved_id}_{kind}*.npy below {root / dataset_name}"
        )
    return files


def _compare_pair(reference: Path, candidate: Path) -> tuple[bool, float]:
    reference_array = np.load(reference, mmap_mode="r")
    candidate_array = np.load(candidate, mmap_mode="r")
    if reference_array.shape != candidate_array.shape:
        print(f"Shape mismatch: {reference_array.shape} != {candidate_array.shape}")
        return False, float("inf")
    if reference_array.dtype != candidate_array.dtype:
        print(f"Dtype mismatch: {reference_array.dtype} != {candidate_array.dtype}")
        return False, float("inf")
    maximum_difference = float(
        np.max(np.abs(np.asarray(reference_array) - np.asarray(candidate_array)))
    )
    return bool(np.array_equal(reference_array, candidate_array)), maximum_difference


def main() -> None:
    selected = [name for name in DATASETS_TO_PROCESS if name in {"PURE", "UBFC"}]
    if not selected:
        raise SystemExit(
            "Exact official-cache parity applies only to PURE and UBFC. "
            "Use validate_cache.py for newly integrated datasets."
        )
    all_passed = True
    for dataset_name in selected:
        adapter = get_adapter(dataset_name)
        recordings = adapter.discover(DATASET_SETTINGS[dataset_name].raw_root)
        if not recordings:
            raise RuntimeError(f"No raw {dataset_name} recordings were found")
        saved_id = recordings[0].saved_id

        print("=" * 70)
        print(f"{dataset_name} OFFICIAL vs INDEPENDENT PREPROCESSING PARITY")
        print("=" * 70)
        dataset_passed = True
        maximum_difference = 0.0

        for kind in ("input", "label"):
            reference_files = _matching_files(
                REFERENCE_CACHE_ROOT, dataset_name, saved_id, kind
            )
            candidate_files = _matching_files(
                SMOKE_CACHE_ROOT, dataset_name, saved_id, kind
            )
            if len(reference_files) != len(candidate_files):
                print(
                    f"{kind.title()} clip-count mismatch: "
                    f"{len(reference_files)} != {len(candidate_files)}"
                )
                dataset_passed = False
                continue

            for reference, candidate in zip(reference_files, candidate_files):
                equal, difference = _compare_pair(reference, candidate)
                dataset_passed &= equal
                maximum_difference = max(maximum_difference, difference)

        print(f"Recording                 : {recordings[0].source_id}")
        print(f"Maximum absolute difference: {maximum_difference:.12g}")
        print(f"Parity result             : {'PASSED' if dataset_passed else 'FAILED'}")
        all_passed &= dataset_passed

    if not all_passed:
        raise SystemExit(
            "Preprocessing parity failed. Do not replace the verified cache."
        )
    print("=" * 70)
    print("Independent preprocessing parity: PASSED")


if __name__ == "__main__":
    main()
