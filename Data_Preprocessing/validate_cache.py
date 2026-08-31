"""Validate independent full-cache structure without loading all video data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from .dataset_registry import get_adapter
    from .settings import (
        DATASET_SETTINGS,
        DATASETS_TO_PROCESS,
        FULL_CACHE_ROOT,
        TRANSFORM,
        cache_name,
    )
except ImportError:
    from dataset_registry import get_adapter
    from settings import (
        DATASET_SETTINGS,
        DATASETS_TO_PROCESS,
        FULL_CACHE_ROOT,
        TRANSFORM,
        cache_name,
    )


def _chunk_index(path: Path, marker: str) -> int:
    return int(path.stem.rsplit(marker, 1)[1])


def _validate_recording(cache_directory: Path, saved_id: str, expected_dtype: str) -> int:
    inputs = sorted(
        cache_directory.glob(f"{saved_id}_input*.npy"),
        key=lambda path: _chunk_index(path, "input"),
    )
    labels = sorted(
        cache_directory.glob(f"{saved_id}_label*.npy"),
        key=lambda path: _chunk_index(path, "label"),
    )
    if not inputs or len(inputs) != len(labels):
        raise RuntimeError(
            f"{saved_id}: input/label clips are missing or unequal "
            f"({len(inputs)} inputs, {len(labels)} labels)."
        )

    for expected, (input_path, label_path) in enumerate(zip(inputs, labels)):
        if _chunk_index(input_path, "input") != expected:
            raise RuntimeError(f"{saved_id}: non-contiguous input chunks")
        if _chunk_index(label_path, "label") != expected:
            raise RuntimeError(f"{saved_id}: non-contiguous label chunks")
        video = np.load(input_path, mmap_mode="r")
        label = np.load(label_path, mmap_mode="r")
        expected_video_shape = (
            TRANSFORM.chunk_length,
            TRANSFORM.height,
            TRANSFORM.width,
            3,
        )
        if video.shape != expected_video_shape or label.shape != (TRANSFORM.chunk_length,):
            raise RuntimeError(
                f"{saved_id} chunk {expected}: unexpected shapes "
                f"{video.shape} and {label.shape}"
            )
        target_dtype = np.dtype(expected_dtype)
        if video.dtype != target_dtype or label.dtype != target_dtype:
            raise RuntimeError(
                f"{saved_id} chunk {expected}: expected {target_dtype} cache, got "
                f"{video.dtype} and {label.dtype}"
            )

    # Check finiteness for one complete pair per recording without rereading
    # every pixel in the 60-90 GB cache.
    if not np.isfinite(np.asarray(np.load(inputs[0], mmap_mode="r"))).all():
        raise RuntimeError(f"{saved_id}: non-finite values in {inputs[0]}")
    if not np.isfinite(np.asarray(np.load(labels[0], mmap_mode="r"))).all():
        raise RuntimeError(f"{saved_id}: non-finite values in {labels[0]}")
    return len(inputs)


def _validate_manifests(dataset_name: str, cache_parent: Path) -> None:
    manifest_directory = cache_parent / "DataFileLists"
    descriptor = cache_name(dataset_name)
    for begin, end in DATASET_SETTINGS[dataset_name].splits:
        manifest = manifest_directory / f"{descriptor}_{float(begin)}_{float(end)}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest}")
        with manifest.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"Empty manifest: {manifest}")
        missing = [
            row["input_files"]
            for row in rows
            if not Path(row["input_files"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{manifest} contains {len(missing)} missing files; first: {missing[0]}"
            )


def main() -> None:
    for dataset_name in DATASETS_TO_PROCESS:
        adapter = get_adapter(dataset_name)
        recordings = adapter.discover(DATASET_SETTINGS[dataset_name].raw_root)
        cache_parent = FULL_CACHE_ROOT / dataset_name
        cache_directory = cache_parent / cache_name(dataset_name)
        if not cache_directory.is_dir():
            raise FileNotFoundError(f"Full cache does not exist: {cache_directory}")

        total_clips = 0
        for recording in recordings:
            total_clips += _validate_recording(
                cache_directory,
                recording.saved_id,
                DATASET_SETTINGS[dataset_name].cache_dtype,
            )
        _validate_manifests(dataset_name, cache_parent)

        print("=" * 70)
        print(f"{dataset_name} INDEPENDENT FULL-CACHE VALIDATION")
        print("=" * 70)
        print(f"Recordings: {len(recordings)}")
        print(f"Clips     : {total_clips}")
        print(f"Cache     : {cache_directory}")
        print(f"{dataset_name} full-cache validation: PASSED")


if __name__ == "__main__":
    main()
