"""Generate deterministic one-of-four offline augmentation caches.

The source video is read through the verified PURE/UBFC adapters, cropped and
resized exactly as in the baseline, augmented before standardization, and then
globally standardized per recording.  Labels are never rewritten: metadata
points to the verified baseline label file for every augmented input.

This module has no command-line arguments.  Use either
``generate_pure_offline_augmentation.py`` or
``generate_ubfc_offline_augmentation.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from Data_Preprocessing.common import crop_face_resize, standardized_data
from Data_Preprocessing.dataset_registry import get_adapter
from Data_Preprocessing.settings import DATASET_SETTINGS, TRANSFORM
from dataset import find_file_list
from settings import PURE, UBFC


# -----------------------------------------------------------------------------
# Fixed Stage-1 policy (edit here only if defining a new experiment)
# -----------------------------------------------------------------------------

SEED = 20260901
SPLIT_BEGIN = 0.0
SPLIT_END = 0.8
OUTPUT_ROOT = Path(
    "/home/rafsan/Documents/Data/Mamba_Hunt_Data/"
    "RhythmMamba_Offline_Augmentation"
)

JPEG_QUALITY = (70, 90)
BLUR_SIGMA = (0.3, 0.8)
GAMMA = (0.85, 1.15)
CONTRAST = (0.90, 1.10)
TRANSFORMS = ("jpeg", "blur", "gamma", "contrast")

EXPECTED = {
    "PURE": {"recordings": 47, "clips": 596},
    "UBFC": {"recordings": 33, "clips": 378},
}


def _recording_id(path: str | Path) -> str:
    stem = Path(path).stem
    if "_input" not in stem:
        raise ValueError(f"Cannot determine recording ID from {path}")
    return stem.rsplit("_input", 1)[0]


def _chunk_id(path: str | Path) -> int:
    stem = Path(path).stem
    return int(stem.rsplit("_input", 1)[1])


def _read_training_manifest(experiment) -> tuple[Path, list[str]]:
    manifest = find_file_list(experiment, SPLIT_BEGIN, SPLIT_END)
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        files = sorted(row["input_files"] for row in csv.DictReader(handle))
    if not files:
        raise RuntimeError(f"Training manifest is empty: {manifest}")
    missing = [path for path in files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Training manifest contains {len(missing)} missing inputs. "
            f"First: {missing[0]}"
        )
    return manifest, files


def _stable_rng(dataset: str, input_path: str) -> random.Random:
    text = f"{SEED}|{dataset}|{Path(input_path).name}".encode("utf-8")
    number = int.from_bytes(hashlib.sha256(text).digest()[:8], "big")
    return random.Random(number)


def _assignment(dataset: str, input_path: str) -> tuple[str, float]:
    rng = _stable_rng(dataset, input_path)
    transform = rng.choice(TRANSFORMS)
    if transform == "jpeg":
        parameter = float(rng.randint(*JPEG_QUALITY))
    elif transform == "blur":
        parameter = rng.uniform(*BLUR_SIGMA)
    elif transform == "gamma":
        parameter = rng.uniform(*GAMMA)
    else:
        parameter = rng.uniform(*CONTRAST)
    return transform, parameter


def _jpeg_frame(rgb: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not success:
        raise RuntimeError("OpenCV JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV JPEG decoding failed")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _augment_clip(clip: np.ndarray, transform: str, parameter: float) -> np.ndarray:
    """Apply one clip-consistent transform to raw RGB uint8 frames."""
    clip = np.asarray(np.clip(clip, 0, 255), dtype=np.uint8)
    if transform == "jpeg":
        return np.stack(
            [_jpeg_frame(frame, int(round(parameter))) for frame in clip], axis=0
        )
    if transform == "blur":
        return np.stack(
            [cv2.GaussianBlur(frame, (0, 0), sigmaX=parameter, sigmaY=parameter)
             for frame in clip],
            axis=0,
        )
    values = clip.astype(np.float32) / 255.0
    if transform == "gamma":
        values = np.power(values, parameter)
    elif transform == "contrast":
        values = (values - 0.5) * parameter + 0.5
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            np.save(handle, array)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write empty metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _validate_existing(path: Path, source: Path) -> None:
    augmented = np.load(path, mmap_mode="r")
    original = np.load(source, mmap_mode="r")
    if augmented.shape != original.shape:
        raise RuntimeError(
            f"Existing augmented shape {augmented.shape} != {original.shape}: {path}"
        )
    if augmented.dtype != np.float32:
        raise RuntimeError(f"Expected float32 augmented cache, got {augmented.dtype}: {path}")
    if not np.isfinite(augmented).all():
        raise RuntimeError(f"Existing augmented cache contains NaN/Inf: {path}")


def _output_directory(dataset: str, source_inputs: list[str]) -> Path:
    source_cache_name = Path(source_inputs[0]).parent.name
    return OUTPUT_ROOT / dataset / f"{source_cache_name}_OfflineOneOfFour"


def generate(dataset: str) -> None:
    dataset = dataset.upper()
    if dataset not in ("PURE", "UBFC"):
        raise ValueError("Offline Stage-1 generation supports PURE or UBFC only")
    experiment = PURE if dataset == "PURE" else UBFC
    manifest, inputs = _read_training_manifest(experiment)
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in inputs:
        grouped[_recording_id(path)].append(path)
    for paths in grouped.values():
        paths.sort(key=_chunk_id)

    expected = EXPECTED[dataset]
    if len(grouped) != expected["recordings"] or len(inputs) != expected["clips"]:
        raise RuntimeError(
            f"{dataset} split mismatch. Expected {expected}, observed "
            f"recordings={len(grouped)}, clips={len(inputs)}"
        )

    adapter = get_adapter(dataset)
    raw_root = DATASET_SETTINGS[dataset].raw_root
    discovered = {recording.saved_id: recording for recording in adapter.discover(raw_root)}
    missing_recordings = sorted(set(grouped) - set(discovered))
    if missing_recordings:
        raise RuntimeError(f"Raw recordings not found: {missing_recordings}")

    output_dir = _output_directory(dataset, inputs)
    metadata_path = output_dir.parent / f"{dataset}_offline_augmentation_metadata.csv"
    summary_path = output_dir.parent / f"{dataset}_offline_augmentation_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    transform_counts: Counter[str] = Counter()
    reused = 0
    generated = 0

    print("=" * 78)
    print(f"{dataset} OFFLINE AUGMENTATION — ONE OF FOUR")
    print("=" * 78)
    print(f"Raw root          : {raw_root}")
    print(f"Training manifest : {manifest}")
    print(f"Output directory  : {output_dir}")
    print(f"Seed              : {SEED}")
    print(f"Recordings        : {len(grouped)}")
    print(f"Clips             : {len(inputs)}")

    for recording_id in tqdm(sorted(grouped), desc=f"{dataset} recordings"):
        source_paths = grouped[recording_id]
        output_paths = [output_dir / Path(path).name for path in source_paths]
        all_existing = all(path.is_file() for path in output_paths)
        any_existing = any(path.is_file() for path in output_paths)
        if any_existing and not all_existing:
            raise RuntimeError(
                f"Partial generated recording found for {recording_id}. "
                "Remove only that recording's generated files, then rerun."
            )

        assignments = [_assignment(dataset, path) for path in source_paths]
        if all_existing:
            for output_path, source_path in zip(output_paths, source_paths):
                _validate_existing(output_path, Path(source_path))
            reused += len(source_paths)
        else:
            recording = discovered[recording_id]
            raw_frames = adapter.read_frames(recording)
            cropped = crop_face_resize(raw_frames).astype(np.uint8)
            del raw_frames

            required_frames = len(source_paths) * TRANSFORM.chunk_length
            if cropped.shape[0] < required_frames:
                raise RuntimeError(
                    f"{recording_id}: cropped video has {cropped.shape[0]} frames, "
                    f"but {required_frames} are required by the verified manifest"
                )
            augmented_video = cropped
            for index, (transform, parameter) in enumerate(assignments):
                start = index * TRANSFORM.chunk_length
                end = start + TRANSFORM.chunk_length
                augmented_video[start:end] = _augment_clip(
                    augmented_video[start:end], transform, parameter
                )
            standardized = standardized_data(augmented_video).astype(np.float32)
            del augmented_video

            for index, (output_path, source_path) in enumerate(
                zip(output_paths, source_paths)
            ):
                start = index * TRANSFORM.chunk_length
                end = start + TRANSFORM.chunk_length
                value = standardized[start:end]
                original = np.load(source_path, mmap_mode="r")
                if value.shape != original.shape:
                    raise RuntimeError(
                        f"Generated shape {value.shape} != original {original.shape}: "
                        f"{source_path}"
                    )
                if not np.isfinite(value).all():
                    raise RuntimeError(f"Generated NaN/Inf for {source_path}")
                _atomic_save(output_path, value)
                generated += 1
            del standardized, cropped

        for source_path, output_path, (transform, parameter) in zip(
            source_paths, output_paths, assignments
        ):
            source = Path(source_path)
            label = Path(str(source).replace("input", "label"))
            if not label.is_file():
                raise FileNotFoundError(f"Verified label is missing: {label}")
            transform_counts[transform] += 1
            rows.append(
                {
                    "dataset": dataset,
                    "recording_id": recording_id,
                    "chunk_id": _chunk_id(source),
                    "original_input": str(source.resolve()),
                    "augmented_input": str(output_path.resolve()),
                    "original_label": str(label.resolve()),
                    "transform": transform,
                    "parameter": f"{parameter:.8f}",
                    "seed": SEED,
                    "frames": TRANSFORM.chunk_length,
                    "height": TRANSFORM.height,
                    "width": TRANSFORM.width,
                    "channels": 3,
                    "dtype": "float32",
                    "finite": True,
                }
            )

    rows.sort(key=lambda row: (str(row["recording_id"]), int(row["chunk_id"])))
    _atomic_csv(metadata_path, rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "seed": SEED,
        "policy": "exactly_one_of_jpeg_blur_gamma_contrast_per_training_clip",
        "augmentation_order": "crop_resize -> offline_augmentation -> recording_standardization",
        "training_manifest": str(manifest.resolve()),
        "output_directory": str(output_dir.resolve()),
        "metadata_csv": str(metadata_path.resolve()),
        "recordings": len(grouped),
        "clips": len(rows),
        "newly_generated": generated,
        "reused_after_validation": reused,
        "transform_counts": dict(sorted(transform_counts.items())),
        "label_policy": "verified_original_label_paths_reused_without_modification",
        "validation": "PASSED",
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)

    print("=" * 78)
    print(f"{dataset} OFFLINE AUGMENTATION COMPLETED")
    print("=" * 78)
    print(f"Generated          : {generated}")
    print(f"Reused/validated   : {reused}")
    print(f"Transform counts   : {dict(sorted(transform_counts.items()))}")
    print(f"Metadata           : {metadata_path}")
    print(f"Summary            : {summary_path}")
    print("Shape/finite/labels: PASSED")

