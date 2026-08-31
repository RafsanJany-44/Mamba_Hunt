"""Shared preprocessing operations extracted from the official data loaders.

PURE and UBFC intentionally retain the verified official behavior.  New
datasets use the same first-frame face crop, global per-recording
standardization, 160-frame non-overlapping chunks, and a documented conversion
to the model's 30 Hz time base.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

try:
    from .settings import (
        DATASET_SETTINGS,
        MAX_WORKERS,
        OVERWRITE_EXISTING,
        RUN_MODE,
        SMOKE_RECORDINGS_PER_DATASET,
        TRANSFORM,
        cache_name,
        selected_cache_root,
    )
except ImportError:
    from settings import (
        DATASET_SETTINGS,
        MAX_WORKERS,
        OVERWRITE_EXISTING,
        RUN_MODE,
        SMOKE_RECORDINGS_PER_DATASET,
        TRANSFORM,
        cache_name,
        selected_cache_root,
    )


PACKAGE_ROOT = Path(__file__).resolve().parent
FACE_CASCADE_PATH = PACKAGE_ROOT / "assets" / "haarcascade_frontalface_default.xml"


@dataclass(frozen=True)
class Recording:
    """One raw video/label pair in a dataset-specific native layout."""

    dataset: str
    source_id: str
    saved_id: str
    frame_source: str
    label_source: str
    subject_id: int | None = None
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProcessResult:
    dataset: str
    source_id: str
    saved_id: str
    input_files: tuple[str, ...]
    frame_count: int
    label_count: int
    clip_count: int
    reused_existing: bool
    quality: tuple[tuple[str, str], ...] = ()


def resample_ppg(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Match the official PURE PPG interpolation."""
    return np.interp(
        np.linspace(1, signal.shape[0], target_length),
        np.linspace(1, signal.shape[0], signal.shape[0]),
        signal,
    )


def standardized_data(data: np.ndarray) -> np.ndarray:
    """Global video z-score used by the official BaseLoader."""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


def standardized_label(label: np.ndarray) -> np.ndarray:
    """Global label z-score used by the official BaseLoader."""
    label = label - np.mean(label)
    label = label / np.std(label)
    label[np.isnan(label)] = 0
    return label


def detect_face(
    frame: np.ndarray,
    use_larger_box: bool,
    larger_box_coefficient: float,
) -> np.ndarray:
    """Run the same Haar-cascade selection used by the official loader."""
    if not FACE_CASCADE_PATH.is_file():
        raise FileNotFoundError(f"Face cascade is missing: {FACE_CASCADE_PATH}")

    detector = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"OpenCV could not load {FACE_CASCADE_PATH}")

    face_zone = detector.detectMultiScale(frame)
    if len(face_zone) < 1:
        print("Warning: no face detected; using the official fallback crop.")
        # Deliberately preserves the official coordinate ordering.
        face_box = np.asarray([0, 0, frame.shape[0], frame.shape[1]], dtype=np.int64)
    elif len(face_zone) >= 2:
        # The official implementation chooses the detection with largest width.
        coordinate_argmax = np.argmax(face_zone, axis=0)
        face_box = np.asarray(face_zone[coordinate_argmax[2]]).copy()
        print("Warning: multiple faces detected; cropping the largest-width face.")
    else:
        face_box = np.asarray(face_zone[0]).copy()

    if use_larger_box:
        face_box[0] = max(
            0,
            face_box[0]
            - (larger_box_coefficient - 1.0) / 2 * face_box[2],
        )
        face_box[1] = max(
            0,
            face_box[1]
            - (larger_box_coefficient - 1.0) / 2 * face_box[3],
        )
        face_box[2] = larger_box_coefficient * face_box[2]
        face_box[3] = larger_box_coefficient * face_box[3]
    return face_box


def crop_face_resize(frames: np.ndarray) -> np.ndarray:
    """Crop and resize a complete video using the verified official settings."""
    t = TRANSFORM
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError(f"Expected non-empty [T,H,W,3] frames, got {frames.shape}")

    if t.dynamic_detection:
        number_of_detections = ceil(frames.shape[0] / t.detection_frequency)
    else:
        number_of_detections = 1

    face_regions = []
    for index in range(number_of_detections):
        if t.crop_face:
            face_regions.append(
                detect_face(
                    frames[t.detection_frequency * index],
                    t.use_large_face_box,
                    t.large_box_coefficient,
                )
            )
        else:
            face_regions.append([0, 0, frames.shape[1], frames.shape[2]])
    face_regions = np.asarray(face_regions, dtype=int)

    median_region = None
    if t.use_median_face_box:
        median_region = np.median(face_regions, axis=0).astype(int)

    # np.zeros without dtype is intentional: the official cache uses float64.
    resized = np.zeros((frames.shape[0], t.height, t.width, 3))
    for index, frame in enumerate(frames):
        reference_index = index // t.detection_frequency if t.dynamic_detection else 0
        if t.crop_face:
            region = median_region if median_region is not None else face_regions[reference_index]
            x, y, width, height = region
            frame = frame[
                max(y, 0) : min(y + height, frame.shape[0]),
                max(x, 0) : min(x + width, frame.shape[1]),
            ]
            if frame.size == 0:
                raise RuntimeError(f"Face crop became empty at frame {index}: {region}")
        resized[index] = cv2.resize(
            frame,
            (t.width, t.height),
            interpolation=cv2.INTER_AREA,
        )
    return resized


def crop_resize_frame_sequence(frames: Any) -> np.ndarray:
    """Crop a frame iterator with one first-frame box and store compact uint8.

    This has the same fixed-box behavior as ``crop_face_resize`` but avoids
    keeping full-resolution long videos in memory.
    """
    iterator = iter(frames)
    try:
        first = np.asarray(next(iterator))
    except StopIteration as error:
        raise RuntimeError("The frame source is empty") from error
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"Expected RGB [H,W,3] frame, got {first.shape}")

    if TRANSFORM.crop_face:
        region = np.asarray(
            detect_face(
                first,
                TRANSFORM.use_large_face_box,
                TRANSFORM.large_box_coefficient,
            ),
            dtype=int,
        )
    else:
        region = np.asarray([0, 0, first.shape[1], first.shape[0]], dtype=int)

    def resize_one(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if TRANSFORM.crop_face:
            x, y, width, height = region
            frame = frame[
                max(y, 0) : min(y + height, frame.shape[0]),
                max(x, 0) : min(x + width, frame.shape[1]),
            ]
            if frame.size == 0:
                raise RuntimeError(f"Face crop became empty: {region}")
        return cv2.resize(
            frame,
            (TRANSFORM.width, TRANSFORM.height),
            interpolation=cv2.INTER_AREA,
        )

    resized = [resize_one(first)]
    resized.extend(resize_one(frame) for frame in iterator)
    return np.asarray(resized, dtype=np.uint8)


def video_frame_iterator(video_path: str | Path):
    """Yield RGB frames from an AVI/video file and always release OpenCV."""
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 0)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            yield cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def temporal_resample(
    frames: np.ndarray,
    frame_times: np.ndarray,
    label: np.ndarray,
    label_times: np.ndarray,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly align video and waveform to a shared uniform time grid."""
    frames = np.asarray(frames)
    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    label_times = np.asarray(label_times, dtype=np.float64).reshape(-1)
    if frames.shape[0] != frame_times.size or label.size != label_times.size:
        raise ValueError("Sample arrays and timestamp arrays have unequal lengths")
    if frames.shape[0] < 2 or label.size < 2:
        raise ValueError("At least two video and label samples are required")
    if np.any(np.diff(frame_times) <= 0) or np.any(np.diff(label_times) <= 0):
        raise ValueError("Video and label timestamps must be strictly increasing")

    frame_times = frame_times - frame_times[0]
    label_times = label_times - label_times[0]
    frame_step = float(np.median(np.diff(frame_times)))
    label_step = float(np.median(np.diff(label_times)))
    duration = min(frame_times[-1] + frame_step, label_times[-1] + label_step)
    target_count = int(np.floor(duration * target_fps + 1e-7))
    if target_count < TRANSFORM.chunk_length:
        raise RuntimeError(
            f"Only {target_count} aligned samples; need {TRANSFORM.chunk_length}"
        )
    target_times = np.arange(target_count, dtype=np.float64) / target_fps

    right = np.searchsorted(frame_times, target_times, side="right")
    right = np.clip(right, 1, frames.shape[0] - 1)
    left = right - 1
    span = frame_times[right] - frame_times[left]
    weight = np.clip(
        ((target_times - frame_times[left]) / span).astype(np.float32),
        0.0,
        1.0,
    )

    # Work in small blocks to cap temporary memory for 1024x1024 source videos.
    output = np.empty((target_count, *frames.shape[1:]), dtype=np.float32)
    for start in range(0, target_count, 64):
        end = min(start + 64, target_count)
        block_weight = weight[start:end, None, None, None]
        output[start:end] = (
            frames[left[start:end]].astype(np.float32) * (1.0 - block_weight)
            + frames[right[start:end]].astype(np.float32) * block_weight
        )
    aligned_label = np.interp(target_times, label_times, label)
    return output, aligned_label


def preprocess_arrays(
    frames: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the official crop/resize and standardization transformations."""
    if TRANSFORM.data_type != "Standardized":
        raise ValueError("This independent baseline currently preserves Standardized data only.")
    if TRANSFORM.label_type != "Standardized":
        raise ValueError("This independent baseline currently preserves Standardized labels only.")

    processed_frames = standardized_data(crop_face_resize(frames))
    processed_labels = standardized_label(np.asarray(labels))
    return processed_frames, processed_labels


def standardize_aligned_arrays(
    frames: np.ndarray,
    labels: np.ndarray,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize already cropped/time-aligned arrays for new datasets."""
    frames = np.asarray(frames, dtype=np.float32)
    frame_mean = float(np.mean(frames, dtype=np.float64))
    frame_std = float(np.std(frames, dtype=np.float64))
    if frame_std == 0 or not np.isfinite(frame_std):
        frames.fill(0)
    else:
        frames -= frame_mean
        frames /= frame_std
    labels = standardized_label(np.asarray(labels, dtype=np.float64))
    target_dtype = np.dtype(dtype)
    return frames.astype(target_dtype, copy=False), labels.astype(target_dtype, copy=False)


def image_quality_metrics(frames: np.ndarray) -> tuple[tuple[str, str], ...]:
    """Compact raw-crop quality indicators used for later failure analysis."""
    frames = np.asarray(frames)
    step = max(1, frames.shape[0] // 300)
    indices = np.arange(0, frames.shape[0], step)
    sample = np.asarray(frames[indices], dtype=np.float32)
    gray = np.mean(sample, axis=-1)
    sharpness = []
    for frame in sample:
        uint8_frame = np.clip(frame, 0, 255).astype(np.uint8)
        sharpness.append(float(cv2.Laplacian(uint8_frame, cv2.CV_64F).var()))
    if frames.shape[0] > 1:
        next_indices = np.minimum(indices + 1, frames.shape[0] - 1)
        next_gray = np.mean(np.asarray(frames[next_indices], dtype=np.float32), axis=-1)
        motion = float(np.mean(np.abs(next_gray - gray)))
    else:
        motion = float("nan")
    values = {
        "image_brightness_mean": float(np.mean(gray)),
        "image_contrast_std": float(np.std(gray)),
        "dark_pixel_percent": float(np.mean(gray <= 10.0) * 100.0),
        "bright_pixel_percent": float(np.mean(gray >= 245.0) * 100.0),
        "sharpness_laplacian_variance": float(np.mean(sharpness)),
        "motion_mean_absolute_difference": motion,
    }
    return tuple((key, f"{value:.12g}") for key, value in values.items())


def _atomic_numpy_save(path: Path, value: np.ndarray) -> None:
    """Write one npy file atomically so interrupted runs leave no partial cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary_name = handle.name
            np.save(handle, value)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _existing_recording_files(cache_directory: Path, saved_id: str) -> tuple[Path, ...]:
    inputs = tuple(sorted(cache_directory.glob(f"{saved_id}_input*.npy")))
    labels = tuple(sorted(cache_directory.glob(f"{saved_id}_label*.npy")))
    if bool(inputs) != bool(labels) or len(inputs) != len(labels):
        raise RuntimeError(
            f"Incomplete existing cache for {saved_id} in {cache_directory}. "
            "Remove only that recording's input/label files or set OVERWRITE_EXISTING=True."
        )
    expected_inputs = {
        cache_directory / f"{saved_id}_input{index}.npy" for index in range(len(inputs))
    }
    expected_labels = {
        cache_directory / f"{saved_id}_label{index}.npy" for index in range(len(labels))
    }
    if inputs and (set(inputs) != expected_inputs or set(labels) != expected_labels):
        raise RuntimeError(
            f"Non-contiguous existing chunks were found for {saved_id} in {cache_directory}."
        )
    return inputs


def _process_one_recording(
    dataset_name: str,
    recording: Recording,
    cache_directory_text: str,
    overwrite_existing: bool,
) -> ProcessResult:
    """Worker entry point; kept top-level for Windows/macOS multiprocessing."""
    try:
        from .dataset_registry import get_adapter
    except ImportError:
        from dataset_registry import get_adapter

    cache_directory = Path(cache_directory_text)
    existing = _existing_recording_files(cache_directory, recording.saved_id)
    if existing and not overwrite_existing:
        return ProcessResult(
            dataset=dataset_name,
            source_id=recording.source_id,
            saved_id=recording.saved_id,
            input_files=tuple(str(path.resolve()) for path in existing),
            frame_count=-1,
            label_count=-1,
            clip_count=len(existing),
            reused_existing=True,
        )

    adapter = get_adapter(dataset_name)
    dataset_settings = DATASET_SETTINGS[dataset_name]
    if hasattr(adapter, "read_aligned"):
        frames, labels = adapter.read_aligned(recording)
        quality = image_quality_metrics(frames)
        processed_frames, processed_labels = standardize_aligned_arrays(
            frames,
            labels,
            dataset_settings.cache_dtype,
        )
    else:
        # The verified PURE/UBFC route is deliberately unchanged.
        frames = adapter.read_frames(recording)
        labels = adapter.read_label(recording)
        labels = adapter.align_label(labels, frames.shape[0])
        processed_frames, processed_labels = preprocess_arrays(frames, labels)
        quality = ()

    frame_clip_count = processed_frames.shape[0] // TRANSFORM.chunk_length
    label_clip_count = processed_labels.shape[0] // TRANSFORM.chunk_length
    if label_clip_count > frame_clip_count:
        raise RuntimeError(
            f"{recording.source_id}: labels create {label_clip_count} clips but "
            f"frames create only {frame_clip_count}."
        )
    if label_clip_count == 0:
        raise RuntimeError(
            f"{recording.source_id}: recording is shorter than one "
            f"{TRANSFORM.chunk_length}-frame clip."
        )

    if overwrite_existing:
        for old_path in cache_directory.glob(f"{recording.saved_id}_input*.npy"):
            old_path.unlink()
        for old_path in cache_directory.glob(f"{recording.saved_id}_label*.npy"):
            old_path.unlink()

    input_files = []
    for chunk_id in range(label_clip_count):
        start = chunk_id * TRANSFORM.chunk_length
        end = start + TRANSFORM.chunk_length
        input_path = cache_directory / f"{recording.saved_id}_input{chunk_id}.npy"
        label_path = cache_directory / f"{recording.saved_id}_label{chunk_id}.npy"
        _atomic_numpy_save(input_path, processed_frames[start:end])
        _atomic_numpy_save(label_path, processed_labels[start:end])
        input_files.append(str(input_path.resolve()))

    return ProcessResult(
        dataset=dataset_name,
        source_id=recording.source_id,
        saved_id=recording.saved_id,
        input_files=tuple(input_files),
        frame_count=int(frames.shape[0]),
        label_count=int(labels.shape[0]),
        clip_count=label_clip_count,
        reused_existing=False,
        quality=quality,
    )


def _write_manifest(path: Path, input_files: list[str]) -> None:
    if not input_files:
        raise RuntimeError(f"Refusing to create an empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_files"])
        writer.writeheader()
        for input_file in sorted(input_files):
            writer.writerow({"input_files": input_file})
    os.replace(temporary, path)


def _write_recording_metadata(
    dataset_parent: Path,
    recordings: list[Recording],
    results: dict[str, ProcessResult],
) -> Path:
    """Persist condition/task/illumination fields for stratified evaluation."""
    path = dataset_parent / "recording_metadata.csv"
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = {
                row["recording_id"]: row
                for row in csv.DictReader(handle)
                if row.get("recording_id")
            }
    rows: list[dict[str, object]] = []
    for recording in recordings:
        row: dict[str, object] = {
            "dataset": recording.dataset,
            "source_id": recording.source_id,
            "recording_id": recording.saved_id,
            "subject_id": "" if recording.subject_id is None else recording.subject_id,
        }
        row.update(dict(recording.metadata))
        result = results.get(recording.saved_id)
        if result is not None and result.quality:
            row.update(dict(result.quality))
        elif recording.saved_id in existing:
            for key, value in existing[recording.saved_id].items():
                row.setdefault(key, value)
        rows.append(row)
    keys = ["dataset", "source_id", "recording_id", "subject_id"]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def _build_manifests(
    dataset_name: str,
    adapter: Any,
    recordings: list[Recording],
    results: dict[str, ProcessResult],
    dataset_parent: Path,
) -> list[Path]:
    descriptor = cache_name(dataset_name)
    manifest_directory = dataset_parent / "DataFileLists"
    manifests = []

    if RUN_MODE.lower().strip() == "smoke":
        splits = ((0.0, 1.0),)
    else:
        splits = DATASET_SETTINGS[dataset_name].splits

    for begin, end in splits:
        split_recordings = adapter.split(recordings, begin, end)
        files = []
        for recording in split_recordings:
            result = results.get(recording.saved_id)
            if result is None:
                raise RuntimeError(f"No preprocessing result for {recording.source_id}")
            files.extend(result.input_files)
        manifest = manifest_directory / f"{descriptor}_{float(begin)}_{float(end)}.csv"
        _write_manifest(manifest, files)
        manifests.append(manifest)
    return manifests


def run_dataset(dataset_name: str) -> dict[str, Any]:
    """Preprocess one registered dataset and create compatible file lists."""
    dataset_name = dataset_name.upper()
    if dataset_name not in DATASET_SETTINGS:
        raise KeyError(f"No settings registered for {dataset_name}")

    try:
        from .dataset_registry import get_adapter
    except ImportError:
        from dataset_registry import get_adapter

    adapter = get_adapter(dataset_name)
    dataset_settings = DATASET_SETTINGS[dataset_name]
    recordings = adapter.discover(dataset_settings.raw_root)
    if not recordings:
        raise RuntimeError(f"No {dataset_name} recordings found in {dataset_settings.raw_root}")

    if RUN_MODE.lower().strip() == "smoke":
        recordings = recordings[:SMOKE_RECORDINGS_PER_DATASET]

    dataset_parent = selected_cache_root() / dataset_name
    cache_directory = dataset_parent / cache_name(dataset_name)
    cache_directory.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"INDEPENDENT {dataset_name} PREPROCESSING ({RUN_MODE.upper()})")
    print("=" * 70)
    print(f"Raw root       : {dataset_settings.raw_root}")
    print(f"Recordings     : {len(recordings)}")
    print(f"Cache directory: {cache_directory}")

    results_list = []
    worker_count = max(1, min(MAX_WORKERS, len(recordings)))
    if worker_count == 1:
        iterator = recordings
        for recording in tqdm(iterator, desc=f"Preprocessing {dataset_name}"):
            results_list.append(
                _process_one_recording(
                    dataset_name,
                    recording,
                    str(cache_directory),
                    OVERWRITE_EXISTING,
                )
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = {
                executor.submit(
                    _process_one_recording,
                    dataset_name,
                    recording,
                    str(cache_directory),
                    OVERWRITE_EXISTING,
                ): recording
                for recording in recordings
            }
            with tqdm(total=len(futures), desc=f"Preprocessing {dataset_name}") as progress:
                for future in as_completed(futures):
                    recording = futures[future]
                    try:
                        results_list.append(future.result())
                    except Exception as error:
                        raise RuntimeError(
                            f"Preprocessing failed for {dataset_name}/{recording.source_id}"
                        ) from error
                    progress.update(1)

    results = {result.saved_id: result for result in results_list}
    manifests = _build_manifests(
        dataset_name,
        adapter,
        recordings,
        results,
        dataset_parent,
    )
    metadata_file = _write_recording_metadata(dataset_parent, recordings, results)
    total_clips = sum(result.clip_count for result in results_list)
    reused = sum(result.reused_existing for result in results_list)

    print(f"Generated clips : {total_clips}")
    print(f"Reused records  : {reused}")
    print(f"Manifest files  : {len(manifests)}")
    print(f"Independent {dataset_name} preprocessing: PASSED")
    return {
        "dataset": dataset_name,
        "recordings": len(recordings),
        "clips": total_clips,
        "cache_directory": cache_directory,
        "manifests": manifests,
        "metadata_file": metadata_file,
    }
