"""Bitwise parity check for complete independent PURE and UBFC caches.

This script compares every input clip, every label clip, and every split
manifest against the verified cache generated through the official repository.
Arrays are memory-mapped and scanned in bounded blocks to limit RAM usage.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from .settings import (
        DATASET_SETTINGS,
        DATASETS_TO_PROCESS,
        FULL_CACHE_ROOT,
        REFERENCE_CACHE_ROOT,
        cache_name,
    )
except ImportError:
    from settings import (
        DATASET_SETTINGS,
        DATASETS_TO_PROCESS,
        FULL_CACHE_ROOT,
        REFERENCE_CACHE_ROOT,
        cache_name,
    )


# About 8 MB per float64 block. Two arrays and their difference remain small.
VALUES_PER_BLOCK = 1_000_000


def _files_by_name(directory: Path, pattern: str) -> dict[str, Path]:
    files = list(directory.glob(pattern))
    mapping = {path.name: path for path in files}
    if len(mapping) != len(files):
        raise RuntimeError(f"Duplicate filenames found in {directory}")
    return mapping


def _compare_array_files(reference: Path, candidate: Path) -> tuple[bool, float]:
    reference_array = np.load(reference, mmap_mode="r")
    candidate_array = np.load(candidate, mmap_mode="r")

    if reference_array.shape != candidate_array.shape:
        print(
            f"Shape mismatch for {reference.name}: "
            f"{reference_array.shape} != {candidate_array.shape}"
        )
        return False, float("inf")
    if reference_array.dtype != candidate_array.dtype:
        print(
            f"Dtype mismatch for {reference.name}: "
            f"{reference_array.dtype} != {candidate_array.dtype}"
        )
        return False, float("inf")

    reference_flat = reference_array.reshape(-1)
    candidate_flat = candidate_array.reshape(-1)
    maximum_difference = 0.0
    equal = True

    for start in range(0, reference_flat.size, VALUES_PER_BLOCK):
        end = min(start + VALUES_PER_BLOCK, reference_flat.size)
        reference_block = reference_flat[start:end]
        candidate_block = candidate_flat[start:end]
        if not np.array_equal(reference_block, candidate_block, equal_nan=True):
            equal = False
            difference = np.abs(reference_block - candidate_block)
            finite_difference = difference[np.isfinite(difference)]
            if finite_difference.size:
                maximum_difference = max(
                    maximum_difference, float(np.max(finite_difference))
                )
            elif difference.size:
                maximum_difference = float("inf")
    return equal, maximum_difference


def _manifest_membership(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", newline="") as handle:
        rows = csv.DictReader(handle)
        return {Path(row["input_files"]).name for row in rows}


def _compare_manifests(
    dataset_name: str,
    reference_parent: Path,
    candidate_parent: Path,
) -> bool:
    descriptor = cache_name(dataset_name)
    passed = True
    for begin, end in DATASET_SETTINGS[dataset_name].splits:
        filename = f"{descriptor}_{float(begin)}_{float(end)}.csv"
        reference = reference_parent / "DataFileLists" / filename
        candidate = candidate_parent / "DataFileLists" / filename
        reference_membership = _manifest_membership(reference)
        candidate_membership = _manifest_membership(candidate)
        equal = reference_membership == candidate_membership
        print(
            f"Split {float(begin)}-{float(end)}: "
            f"official={len(reference_membership)}, "
            f"independent={len(candidate_membership)}, "
            f"{'MATCH' if equal else 'MISMATCH'}"
        )
        passed &= equal
    return passed


def _compare_dataset(dataset_name: str) -> bool:
    descriptor = cache_name(dataset_name)
    reference_parent = REFERENCE_CACHE_ROOT / dataset_name
    candidate_parent = FULL_CACHE_ROOT / dataset_name
    reference_directory = reference_parent / descriptor
    candidate_directory = candidate_parent / descriptor

    if not reference_directory.is_dir():
        raise FileNotFoundError(f"Official reference cache missing: {reference_directory}")
    if not candidate_directory.is_dir():
        raise FileNotFoundError(f"Independent full cache missing: {candidate_directory}")

    print("=" * 78)
    print(f"{dataset_name} COMPLETE OFFICIAL vs INDEPENDENT PREPROCESSING PARITY")
    print("=" * 78)

    all_equal = True
    total_files = 0
    matched_files = 0
    maximum_difference = 0.0

    for kind in ("input", "label"):
        reference_files = _files_by_name(reference_directory, f"*_{kind}*.npy")
        candidate_files = _files_by_name(candidate_directory, f"*_{kind}*.npy")
        reference_names = set(reference_files)
        candidate_names = set(candidate_files)

        missing = sorted(reference_names - candidate_names)
        unexpected = sorted(candidate_names - reference_names)
        if missing or unexpected:
            print(
                f"{kind.title()} filename mismatch: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if missing:
                print(f"First missing: {missing[0]}")
            if unexpected:
                print(f"First unexpected: {unexpected[0]}")
            all_equal = False

        common_names = sorted(reference_names & candidate_names)
        total_files += len(reference_names | candidate_names)
        for name in tqdm(common_names, desc=f"Comparing {dataset_name} {kind}s"):
            equal, difference = _compare_array_files(
                reference_files[name], candidate_files[name]
            )
            matched_files += int(equal)
            all_equal &= equal
            maximum_difference = max(maximum_difference, difference)
            if not equal:
                print(f"Array mismatch: {name}; max difference={difference}")

    manifests_equal = _compare_manifests(
        dataset_name, reference_parent, candidate_parent
    )
    all_equal &= manifests_equal

    print(f"Total array files          : {total_files}")
    print(f"Bit-identical array files  : {matched_files}")
    print(f"Maximum absolute difference: {maximum_difference:.12g}")
    print(f"Manifest membership        : {'MATCH' if manifests_equal else 'MISMATCH'}")
    print(f"Dataset parity             : {'PASSED' if all_equal else 'FAILED'}")
    return all_equal


def main() -> None:
    print(f"Official reference: {REFERENCE_CACHE_ROOT}")
    print(f"Independent cache : {FULL_CACHE_ROOT}")
    selected = [name for name in DATASETS_TO_PROCESS if name in {"PURE", "UBFC"}]
    if not selected:
        raise SystemExit(
            "Exact official-cache parity applies only to PURE and UBFC. "
            "Use validate_cache.py for newly integrated datasets."
        )
    results = [_compare_dataset(name) for name in selected]
    if not all(results):
        raise SystemExit(
            "Full preprocessing parity failed. Keep using the verified reference cache."
        )
    print("=" * 78)
    print("COMPLETE PURE AND UBFC PREPROCESSING PARITY: PASSED")


if __name__ == "__main__":
    main()
