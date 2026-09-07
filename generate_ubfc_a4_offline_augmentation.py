"""Generate four deterministic offline variants for every UBFC training clip."""

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


SEED = 20260906
DATASET_NAME = "UBFC"
EXPERIMENT = UBFC
SPLIT_BEGIN = 0.0
SPLIT_END = 0.8
OUTPUT_ROOT = Path(
    "/home/rafsan/Documents/Data/Mamba_Hunt_Data/"
    "RhythmMamba_Offline_Augmentation_A4"
)

JPEG_QUALITY = (70, 90)
BLUR_SIGMA = (0.3, 0.8)
GAMMA = (0.85, 1.15)
CONTRAST = (0.90, 1.10)
TRANSFORMS = ("jpeg", "blur", "gamma", "contrast")
EXPECTED_RECORDINGS = 33
EXPECTED_CLIPS = 378


def recording_id(path):
    stem = Path(path).stem
    if "_input" not in stem:
        raise ValueError(f"Cannot determine recording ID: {path}")
    return stem.rsplit("_input", 1)[0]


def chunk_id(path):
    return int(Path(path).stem.rsplit("_input", 1)[1])


def stable_parameter(input_path, transform):
    key = f"{SEED}|{DATASET_NAME}|{Path(input_path).name}|{transform}".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    if transform == "jpeg":
        return float(rng.randint(*JPEG_QUALITY))
    if transform == "blur":
        return rng.uniform(*BLUR_SIGMA)
    if transform == "gamma":
        return rng.uniform(*GAMMA)
    if transform == "contrast":
        return rng.uniform(*CONTRAST)
    raise ValueError(f"Unknown transform: {transform}")


def jpeg_frame(rgb, quality):
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


def augment_clip(clip, transform, parameter):
    """Apply one parameter consistently to all 160 raw RGB frames."""
    clip = np.asarray(np.clip(clip, 0, 255), dtype=np.uint8)
    if transform == "jpeg":
        return np.stack(
            [jpeg_frame(frame, int(round(parameter))) for frame in clip], axis=0
        )
    if transform == "blur":
        return np.stack(
            [
                cv2.GaussianBlur(
                    frame, (0, 0), sigmaX=parameter, sigmaY=parameter
                )
                for frame in clip
            ],
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


def atomic_npy(path, array):
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


def atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_array(path, original_path):
    value = np.load(path, mmap_mode="r")
    original = np.load(original_path, mmap_mode="r")
    if value.shape != original.shape:
        raise RuntimeError(f"Shape mismatch: {path}")
    if value.dtype != np.float32:
        raise RuntimeError(f"Expected float32: {path}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"NaN/Inf found: {path}")


def main():
    manifest = find_file_list(EXPERIMENT, SPLIT_BEGIN, SPLIT_END)
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        inputs = sorted(row["input_files"] for row in csv.DictReader(handle))
    if len(inputs) != EXPECTED_CLIPS:
        raise RuntimeError(f"Expected {EXPECTED_CLIPS} inputs, found {len(inputs)}")
    grouped = defaultdict(list)
    for path in inputs:
        grouped[recording_id(path)].append(path)
    for paths in grouped.values():
        paths.sort(key=chunk_id)
    if len(grouped) != EXPECTED_RECORDINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDINGS} recordings, found {len(grouped)}"
        )

    adapter = get_adapter(DATASET_NAME)
    raw_root = DATASET_SETTINGS[DATASET_NAME].raw_root
    discovered = {
        recording.saved_id: recording for recording in adapter.discover(raw_root)
    }
    missing = sorted(set(grouped) - set(discovered))
    if missing:
        raise RuntimeError(f"Missing raw recordings: {missing}")

    output_dir = OUTPUT_ROOT / DATASET_NAME / "OfflineAllFour"
    metadata_path = output_dir.parent / f"{DATASET_NAME}_A4_offline_augmentation_metadata.csv"
    summary_path = output_dir.parent / f"{DATASET_NAME}_A4_offline_augmentation_summary.json"
    rows = []
    counts = Counter()
    generated = reused = 0

    print("=" * 82)
    print(f"{DATASET_NAME} A4 OFFLINE AUGMENTATION — ALL FOUR VARIANTS PER CLIP")
    print("=" * 82)
    print(f"Raw root          : {raw_root}")
    print(f"Training manifest : {manifest}")
    print(f"Output directory  : {output_dir}")
    print(f"Seed              : {SEED}")
    print(f"Recordings        : {len(grouped)}")
    print(f"Original clips    : {len(inputs)}")
    print(f"Expected variants : {len(inputs) * len(TRANSFORMS)}")

    for rid in tqdm(sorted(grouped), desc=f"{DATASET_NAME} A4 recordings"):
        source_paths = grouped[rid]
        expected_paths = [
            output_dir / transform / Path(source).name
            for transform in TRANSFORMS
            for source in source_paths
        ]
        exists = [path.is_file() for path in expected_paths]
        if any(exists) and not all(exists):
            raise RuntimeError(
                f"Partial A4 recording found for {rid}. Preserve it for diagnosis "
                "or remove only that recording's incomplete A4 outputs before rerun."
            )

        if all(exists):
            for output_path in expected_paths:
                original_path = next(
                    path for path in source_paths if Path(path).name == output_path.name
                )
                validate_array(output_path, original_path)
            reused += len(expected_paths)
        else:
            raw_frames = adapter.read_frames(discovered[rid])
            cropped = crop_face_resize(raw_frames).astype(np.uint8)
            del raw_frames
            required = len(source_paths) * TRANSFORM.chunk_length
            if cropped.shape[0] < required:
                raise RuntimeError(
                    f"{rid}: {cropped.shape[0]} frames available, {required} required"
                )

            for transform in TRANSFORMS:
                augmented_recording = cropped.copy()
                for index, source in enumerate(source_paths):
                    start = index * TRANSFORM.chunk_length
                    end = start + TRANSFORM.chunk_length
                    parameter = stable_parameter(source, transform)
                    augmented_recording[start:end] = augment_clip(
                        augmented_recording[start:end], transform, parameter
                    )
                standardized = standardized_data(augmented_recording).astype(np.float32)
                del augmented_recording
                for index, source in enumerate(source_paths):
                    start = index * TRANSFORM.chunk_length
                    end = start + TRANSFORM.chunk_length
                    output_path = output_dir / transform / Path(source).name
                    value = standardized[start:end]
                    original = np.load(source, mmap_mode="r")
                    if value.shape != original.shape or not np.isfinite(value).all():
                        raise RuntimeError(f"Invalid generated clip: {output_path}")
                    atomic_npy(output_path, value)
                    generated += 1
                del standardized
            del cropped

        for source in source_paths:
            label = Path(str(source).replace("input", "label"))
            if not label.is_file():
                raise FileNotFoundError(f"Missing verified label: {label}")
            for transform in TRANSFORMS:
                parameter = stable_parameter(source, transform)
                output_path = output_dir / transform / Path(source).name
                counts[transform] += 1
                rows.append({
                    "dataset": DATASET_NAME,
                    "recording_id": rid,
                    "chunk_id": chunk_id(source),
                    "original_input": str(Path(source).resolve()),
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
                })

    rows.sort(key=lambda row: (row["recording_id"], row["chunk_id"], row["transform"]))
    if len(rows) != EXPECTED_CLIPS * len(TRANSFORMS):
        raise RuntimeError(f"Metadata row mismatch: {len(rows)}")
    if any(counts[name] != EXPECTED_CLIPS for name in TRANSFORMS):
        raise RuntimeError(f"Transform count mismatch: {dict(counts)}")
    if len({row["augmented_input"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate augmented paths in metadata")
    atomic_csv(metadata_path, rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "policy": "all_four_variants_per_original_clip",
        "seed": SEED,
        "training_split": [SPLIT_BEGIN, SPLIT_END],
        "recordings": len(grouped),
        "original_clips": len(inputs),
        "variants_per_clip": len(TRANSFORMS),
        "total_augmented_clips": len(rows),
        "generated": generated,
        "reused_validated": reused,
        "transform_counts": dict(counts),
        "jpeg_quality": list(JPEG_QUALITY),
        "blur_sigma": list(BLUR_SIGMA),
        "gamma": list(GAMMA),
        "contrast": list(CONTRAST),
        "temporal_policy": "one fixed parameter across all 160 frames of a clip",
        "standardization": "global per full transformed recording variant",
        "label_policy": "reuse verified original label without modification",
        "metadata": str(metadata_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "status": "PASSED",
    }
    atomic_json(summary_path, summary)
    print("=" * 82)
    print(f"{DATASET_NAME} A4 OFFLINE AUGMENTATION COMPLETED")
    print("=" * 82)
    print(f"Generated          : {generated}")
    print(f"Reused/validated   : {reused}")
    print(f"Original clips     : {len(inputs)}")
    print(f"Augmented clips    : {len(rows)}")
    print(f"Transform counts   : {dict(counts)}")
    print(f"Metadata           : {metadata_path}")
    print(f"Summary            : {summary_path}")
    print("Shape/finite/labels: PASSED")


if __name__ == "__main__":
    main()
