"""Mega evaluation for saved RhythmMamba intra and cross checkpoints.

Three measurement protocols are evaluated without retraining:

1. ``official_mamba``: exact verified RhythmMamba/rPPG-Toolbox evaluation.
2. ``old``: 8-second windows, 1-second stride, 240-point FFT at 30 FPS.
3. ``prism``: 10-second non-overlapping windows, 16,384-point FFT.

No YAML files or command-line arguments are used. Edit the configuration
section and run:

    CUDA_VISIBLE_DEVICES=1 python Mamba_Hunt/eval_protocols.py

Checkpoints and preprocessed caches are read only.
"""

from __future__ import annotations

import csv
import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, filtfilt, periodogram, welch
from scipy.sparse import eye, spdiags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Plotly is required. Install it with: python -m pip install plotly"
    ) from error

from dataset import find_file_list
from model import RhythmMamba
from settings import EPOCHS, FS, MAMBA_HUNT_ROOT, PURE, UBFC


# =============================================================================
# EDITABLE CONFIGURATION
# =============================================================================

DEVICE = "cuda:0"
WINDOW_BATCH_SIZE = 2

RUNS_TO_EVALUATE = (
    "PURE_INTRA",
    "UBFC_INTRA",
    "PURE_TO_UBFC_CROSS",
    "UBFC_TO_PURE_CROSS",
)

PROTOCOLS_TO_RUN = ("official_mamba", "old", "prism")

PURE_INTRA_CHECKPOINT = (
    MAMBA_HUNT_ROOT
    / "results"
    / "models"
    / "PURE"
    / f"PURE_RhythmMamba_Epoch{EPOCHS - 1}.pth"
)
UBFC_INTRA_CHECKPOINT = (
    MAMBA_HUNT_ROOT
    / "results"
    / "models"
    / "UBFC"
    / f"UBFC_RhythmMamba_Epoch{EPOCHS - 1}.pth"
)
PURE_CROSS_CHECKPOINT = (
    MAMBA_HUNT_ROOT / "Official_Checkpoints" / "PURE_cross_RhythmMamba.pth"
)
UBFC_CROSS_CHECKPOINT = (
    MAMBA_HUNT_ROOT / "Official_Checkpoints" / "UBFC_cross_RhythmMamba.pth"
)

OUTPUT_ROOT = MAMBA_HUNT_ROOT / "results" / "evaluation_protocols"

GENERATE_SIGNAL_PLOTS = True
GENERATE_PSD_DIAGNOSTICS = True
GENERATE_SUMMARY_PLOTS = True
SAVE_SIGNAL_SAMPLE_TABLES = True

# "cdn" creates small HTML files. Change to True to embed Plotly JavaScript
# into every HTML file for completely offline viewing.
PLOTLY_JS_MODE: str | bool = "cdn"


# =============================================================================
# PROTOCOL AND RUN DEFINITIONS
# =============================================================================


OFFICIAL_CHUNK_LENGTH = 160
OFFICIAL_WELCH_NFFT = int(1e5 / FS)


@dataclass(frozen=True)
class Protocol:
    name: str
    display_name: str
    mode: str
    aggregation: str
    window_seconds: float | None
    stride_seconds: float | None
    nfft: int | None
    bandpass_low_hz: float
    bandpass_high_hz: float
    bandpass_order: int
    prediction_bpm_min: float
    prediction_bpm_max: float
    gt_bpm_min: float
    gt_bpm_max: float

    @property
    def bin_spacing_bpm(self) -> float:
        if self.mode == "official":
            nfft = OFFICIAL_WELCH_NFFT
        elif self.nfft is not None:
            nfft = self.nfft
        else:
            nfft = int(round(float(self.window_seconds) * FS))
        return FS / nfft * 60.0


PROTOCOLS = {
    "official_mamba": Protocol(
        name="official_mamba",
        display_name="Official RhythmMamba Evaluation",
        mode="official",
        aggregation="recording",
        window_seconds=None,
        stride_seconds=None,
        nfft=OFFICIAL_WELCH_NFFT,
        bandpass_low_hz=0.75,
        bandpass_high_hz=2.5,
        bandpass_order=1,
        prediction_bpm_min=45.0,
        prediction_bpm_max=150.0,
        gt_bpm_min=45.0,
        gt_bpm_max=150.0,
    ),
    "old": Protocol(
        name="old",
        display_name="Old Protocol",
        mode="window_fft",
        aggregation="window_to_recording",
        window_seconds=8.0,
        stride_seconds=1.0,
        #nfft=None,
        nfft=1800, #for 1 bpm resolution
        bandpass_low_hz=0.67,
        bandpass_high_hz=3.0,
        bandpass_order=3,
        prediction_bpm_min=40.0,
        prediction_bpm_max=180.0,
        gt_bpm_min=40.0,
        gt_bpm_max=150.0,
    ),
    "prism": Protocol(
        name="prism",
        display_name="PRISM Evaluation Protocol",
        mode="window_fft",
        aggregation="window_to_recording",
        window_seconds=10.0,
        stride_seconds=10.0,
        nfft=16384,
        bandpass_low_hz=0.75,
        bandpass_high_hz=2.5,
        bandpass_order=2,
        prediction_bpm_min=45.0,
        prediction_bpm_max=150.0,
        gt_bpm_min=45.0,
        gt_bpm_max=150.0,
    ),
}


@dataclass(frozen=True)
class EvaluationRun:
    name: str
    checkpoint: Path
    experiment: Any
    split_begin: float
    split_end: float
    description: str


EVALUATION_RUNS = {
    "PURE_INTRA": EvaluationRun(
        name="PURE_INTRA",
        checkpoint=PURE_INTRA_CHECKPOINT,
        experiment=PURE,
        split_begin=PURE.test_begin,
        split_end=PURE.test_end,
        description="PURE intra checkpoint on the PURE test split",
    ),
    "UBFC_INTRA": EvaluationRun(
        name="UBFC_INTRA",
        checkpoint=UBFC_INTRA_CHECKPOINT,
        experiment=UBFC,
        split_begin=UBFC.test_begin,
        split_end=UBFC.test_end,
        description="UBFC intra checkpoint on the UBFC test split",
    ),
    "PURE_TO_UBFC_CROSS": EvaluationRun(
        name="PURE_TO_UBFC_CROSS",
        checkpoint=PURE_CROSS_CHECKPOINT,
        experiment=UBFC,
        split_begin=0.0,
        split_end=1.0,
        description="PURE cross checkpoint on the complete UBFC dataset",
    ),
    "UBFC_TO_PURE_CROSS": EvaluationRun(
        name="UBFC_TO_PURE_CROSS",
        checkpoint=UBFC_CROSS_CHECKPOINT,
        experiment=PURE,
        split_begin=0.0,
        split_end=1.0,
        description="UBFC cross checkpoint on the complete PURE dataset",
    ),
}


@dataclass
class Measurement:
    recording_id: str
    measurement_id: int
    start_frame: int
    end_frame: int
    prediction: np.ndarray
    label: np.ndarray
    predicted_hr: float
    gt_hr: float
    waveform_pearson: float
    snr_db: float

    @property
    def absolute_error(self) -> float:
        return abs(self.predicted_hr - self.gt_hr)


# =============================================================================
# GENERAL HELPERS
# =============================================================================


CLIP_PATTERN = re.compile(r"^(?P<recording>.+)_input(?P<chunk>\d+)\.npy$")


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:180]


def finite_mean(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def finite_std(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.std(values)) if values.size else float("nan")


def finite_standard_error(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return (
        float(np.std(values) / np.sqrt(values.size))
        if values.size
        else float("nan")
    )


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    length = min(first.size, second.size)
    first = first[:length]
    second = second[:length]
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    standard_deviation = np.std(signal)
    if standard_deviation == 0 or not np.isfinite(standard_deviation):
        return np.zeros_like(signal)
    return (signal - np.mean(signal)) / standard_deviation


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_output_directories(run: EvaluationRun, protocol: Protocol) -> dict[str, Path]:
    base = OUTPUT_ROOT / run.name / protocol.name
    directories = {
        "base": base,
        "tables": base / "tables",
        "plots": base / "plots",
        "signals": base / "signal_comparisons",
        "signal_tables": base / "signal_tables",
        "diagnostics": base / "diagnostics",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


# =============================================================================
# CACHE AND CHECKPOINT LOADING
# =============================================================================


def read_manifest(file_list: Path) -> dict[str, list[tuple[int, Path]]]:
    recordings: dict[str, list[tuple[int, Path]]] = {}
    with file_list.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            input_path = Path(row["input_files"])
            match = CLIP_PATTERN.fullmatch(input_path.name)
            if match is None:
                raise ValueError(f"Unexpected cached filename: {input_path.name}")
            recording_id = match.group("recording")
            chunk_id = int(match.group("chunk"))
            recordings.setdefault(recording_id, []).append((chunk_id, input_path))
    if not recordings:
        raise RuntimeError(f"No recordings were found in {file_list}")
    for recording_id in recordings:
        recordings[recording_id].sort(key=lambda item: item[0])
    return recordings


def read_recording_metadata(file_list: Path) -> dict[str, dict[str, str]]:
    """Read optional preprocessing metadata for condition-wise analysis."""
    path = file_list.parent.parent / "recording_metadata.csv"
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["recording_id"]: row for row in rows if row.get("recording_id")}


def grouped_recording_rows(
    rows: list[dict[str, object]], metadata_fields: list[str]
) -> list[dict[str, object]]:
    """Summarize failure/quality metrics by task, condition, or illumination."""
    output = []
    for field in metadata_fields:
        values = sorted({str(row.get(field, "")) for row in rows if row.get(field, "") != ""})
        for value in values:
            selected = [row for row in rows if str(row.get(field, "")) == value]
            output.append({
                "group_field": field,
                "group_value": value,
                "recordings": len(selected),
                "mean_recording_mae_bpm": finite_mean([row["mae_bpm"] for row in selected]),
                "mean_recording_rmse_bpm": finite_mean([row["rmse_bpm"] for row in selected]),
                "mean_waveform_pearson": finite_mean([row["mean_waveform_pearson"] for row in selected]),
                "mean_snr_db": finite_mean([row["mean_snr_db"] for row in selected]),
            })
    return output


def quality_error_correlation_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Relate preprocessing image-quality indicators to recording MAE."""
    fields = (
        "image_brightness_mean",
        "image_contrast_std",
        "dark_pixel_percent",
        "bright_pixel_percent",
        "sharpness_laplacian_variance",
        "motion_mean_absolute_difference",
    )
    output = []
    for field in fields:
        pairs = []
        for row in rows:
            try:
                quality = float(row[field])
                error = float(row["mae_bpm"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(quality) and np.isfinite(error):
                pairs.append((quality, error))
        if pairs:
            quality, error = map(np.asarray, zip(*pairs))
            output.append({
                "quality_metric": field,
                "recordings": len(pairs),
                "quality_mean": finite_mean(quality),
                "quality_std": finite_std(quality),
                "pearson_with_recording_mae": pearson(quality, error),
            })
    return output


def load_recording(
    clip_paths: list[tuple[int, Path]],
) -> tuple[np.ndarray, np.ndarray]:
    frame_clips = []
    label_clips = []
    expected_chunk = 0

    for chunk_id, input_path in clip_paths:
        if chunk_id != expected_chunk:
            raise RuntimeError(
                f"Non-contiguous chunk sequence: expected {expected_chunk}, got {chunk_id}"
            )
        expected_chunk += 1
        label_path = input_path.with_name(
            input_path.name.replace("_input", "_label", 1)
        )
        if not input_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing cache pair: {input_path}, {label_path}")

        frames = np.load(input_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected [T,H,W,3], got {frames.shape}: {input_path}")
        if labels.ndim != 1 or labels.shape[0] != frames.shape[0]:
            raise ValueError(f"Input/label mismatch: {input_path}, {label_path}")

        frame_clips.append(np.asarray(frames, dtype=np.float32))
        label_clips.append(np.asarray(labels, dtype=np.float64))

    return np.concatenate(frame_clips), np.concatenate(label_clips)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a dictionary: {path}")
    if "model_state" in payload:
        payload = payload["model_state"]
    elif "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"Checkpoint does not contain a model state dictionary: {path}")
    return {key.removeprefix("module."): value for key, value in payload.items()}


def load_model(path: Path) -> RhythmMamba:
    model = RhythmMamba().to(DEVICE)
    model.load_state_dict(checkpoint_state(path), strict=True)
    model.eval()
    return model


# =============================================================================
# MODEL INFERENCE
# =============================================================================


@torch.no_grad()
def predict_windows(
    model: RhythmMamba,
    frames: np.ndarray,
    starts: list[int],
    window_length: int,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for first in range(0, len(starts), WINDOW_BATCH_SIZE):
        batch_starts = starts[first : first + WINDOW_BATCH_SIZE]
        windows = np.stack(
            [frames[start : start + window_length] for start in batch_starts]
        )
        windows = np.transpose(windows, (0, 1, 4, 2, 3))
        tensor = torch.from_numpy(np.ascontiguousarray(windows)).to(DEVICE)
        prediction = model(tensor)
        prediction = (
            prediction - torch.mean(prediction, dim=-1, keepdim=True)
        ) / torch.std(prediction, dim=-1, keepdim=True)
        outputs.extend(
            np.asarray(item, dtype=np.float64)
            for item in prediction.detach().cpu().numpy()
        )
    return outputs


# =============================================================================
# OFFICIAL RHYTHMMAMBA EVALUATION
# =============================================================================


def smoothness_detrend(signal: np.ndarray, regularization: float = 100.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    length = signal.shape[0]
    diagonals = np.asarray(
        [np.ones(length), -2 * np.ones(length), np.ones(length)]
    )
    difference = spdiags(
        diagonals,
        np.asarray([0, 1, 2]),
        length - 2,
        length,
    ).tocsc()
    system = eye(length, format="csc") + regularization**2 * (
        difference.T @ difference
    )
    trend = spsolve(system, signal)
    return signal - trend


def bandpass(signal: np.ndarray, protocol: Protocol) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    nyquist = FS / 2.0
    b, a = butter(
        protocol.bandpass_order,
        [
            protocol.bandpass_low_hz / nyquist,
            protocol.bandpass_high_hz / nyquist,
        ],
        btype="bandpass",
    )
    return filtfilt(b, a, signal)


def official_postprocess(signal: np.ndarray, protocol: Protocol) -> np.ndarray:
    return bandpass(smoothness_detrend(signal), protocol)


def official_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies, power = welch(
        np.asarray(signal, dtype=np.float64),
        FS,
        nfft=1e5 / FS,
        nperseg=np.min((len(signal) - 1, 256)),
    )
    return frequencies * 60.0, power


def official_hr(signal: np.ndarray, bpm_min: float, bpm_max: float) -> float:
    bpm, power = official_spectrum(signal)
    valid = (bpm > bpm_min) & (bpm < bpm_max)
    return float(bpm[valid][np.argmax(power[valid])])


def next_power_of_two(value: int) -> int:
    return 1 if value == 0 else 2 ** (value - 1).bit_length()


def official_snr(prediction: np.ndarray, gt_hr: float) -> float:
    first_harmonic = gt_hr / 60.0
    second_harmonic = 2.0 * first_harmonic
    deviation = 6.0 / 60.0
    nfft = next_power_of_two(len(prediction))
    frequencies, power = periodogram(
        np.expand_dims(prediction, 0),
        fs=FS,
        nfft=nfft,
        detrend=False,
    )
    power = np.squeeze(power)
    harmonic_1 = (frequencies >= first_harmonic - deviation) & (
        frequencies <= first_harmonic + deviation
    )
    harmonic_2 = (frequencies >= second_harmonic - deviation) & (
        frequencies <= second_harmonic + deviation
    )
    remainder = (
        (frequencies >= 0.75)
        & (frequencies <= 2.5)
        & ~harmonic_1
        & ~harmonic_2
    )
    signal_power = power[harmonic_1].sum() + power[harmonic_2].sum()
    noise_power = power[remainder].sum()
    return 0.0 if noise_power == 0 else float(20 * np.log10(signal_power / noise_power))


def evaluate_official_recording(
    model: RhythmMamba,
    recording_id: str,
    frames: np.ndarray,
    labels: np.ndarray,
    protocol: Protocol,
) -> list[Measurement]:
    if len(frames) != len(labels):
        raise ValueError(
            f"Official evaluation requires equal signal lengths, got "
            f"frames={len(frames)}, labels={len(labels)}"
        )
    if len(frames) % OFFICIAL_CHUNK_LENGTH != 0:
        raise ValueError(
            f"Official cache length {len(frames)} is not divisible by "
            f"the {OFFICIAL_CHUNK_LENGTH}-frame chunk length"
        )
    starts = list(
        range(0, len(frames) - OFFICIAL_CHUNK_LENGTH + 1, OFFICIAL_CHUNK_LENGTH)
    )
    predictions = predict_windows(model, frames, starts, OFFICIAL_CHUNK_LENGTH)
    prediction = np.concatenate(predictions)
    label = labels[: len(prediction)]
    prediction = official_postprocess(prediction, protocol)
    label = official_postprocess(label, protocol)
    predicted_hr = official_hr(
        prediction, protocol.prediction_bpm_min, protocol.prediction_bpm_max
    )
    gt_hr = official_hr(label, protocol.gt_bpm_min, protocol.gt_bpm_max)
    return [
        Measurement(
            recording_id=recording_id,
            measurement_id=0,
            start_frame=0,
            end_frame=len(prediction),
            prediction=prediction,
            label=label,
            predicted_hr=predicted_hr,
            gt_hr=gt_hr,
            waveform_pearson=pearson(prediction, label),
            snr_db=official_snr(prediction, gt_hr),
        )
    ]


# =============================================================================
# OLD AND PRISM EVALUATION
# =============================================================================


def direct_fft_spectrum(
    signal: np.ndarray,
    nfft: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    signal = signal - np.mean(signal)
    transform_length = len(signal) if nfft is None else int(nfft)
    frequencies = np.fft.rfftfreq(transform_length, d=1.0 / FS)
    power = np.abs(np.fft.rfft(signal, n=transform_length)) ** 2
    return frequencies * 60.0, power


def direct_fft_hr(
    signal: np.ndarray,
    bpm_min: float,
    bpm_max: float,
    nfft: int | None,
) -> float:
    bpm, power = direct_fft_spectrum(signal, nfft)
    valid = (bpm >= bpm_min) & (bpm <= bpm_max)
    return float(bpm[valid][np.argmax(power[valid])])


def evaluate_window_recording(
    model: RhythmMamba,
    recording_id: str,
    frames: np.ndarray,
    labels: np.ndarray,
    protocol: Protocol,
) -> list[Measurement]:
    window_length = int(round(float(protocol.window_seconds) * FS))
    stride_length = int(round(float(protocol.stride_seconds) * FS))
    usable_length = min(len(frames), len(labels))
    if usable_length < window_length:
        return []
    starts = list(range(0, usable_length - window_length + 1, stride_length))
    predictions = predict_windows(model, frames, starts, window_length)
    measurements = []
    for measurement_id, (start, prediction) in enumerate(zip(starts, predictions)):
        end = start + window_length
        label = labels[start:end]
        # Matches the provided old/PRISM file: filter the model waveform, while
        # GT HR is selected directly from the configured FFT search band.
        prediction = bandpass(prediction, protocol)
        predicted_hr = direct_fft_hr(
            prediction,
            protocol.prediction_bpm_min,
            protocol.prediction_bpm_max,
            protocol.nfft,
        )
        gt_hr = direct_fft_hr(
            label,
            protocol.gt_bpm_min,
            protocol.gt_bpm_max,
            protocol.nfft,
        )
        measurements.append(
            Measurement(
                recording_id=recording_id,
                measurement_id=measurement_id,
                start_frame=start,
                end_frame=end,
                prediction=prediction,
                label=np.asarray(label, dtype=np.float64),
                predicted_hr=predicted_hr,
                gt_hr=gt_hr,
                waveform_pearson=pearson(prediction, label),
                snr_db=float("nan"),
            )
        )
    return measurements


def evaluate_recording(
    model: RhythmMamba,
    recording_id: str,
    frames: np.ndarray,
    labels: np.ndarray,
    protocol: Protocol,
) -> list[Measurement]:
    if protocol.mode == "official":
        return evaluate_official_recording(
            model, recording_id, frames, labels, protocol
        )
    return evaluate_window_recording(model, recording_id, frames, labels, protocol)


# =============================================================================
# MEASUREMENT TABLES AND DIAGNOSTICS
# =============================================================================


def measurement_spectrum(
    signal: np.ndarray,
    protocol: Protocol,
) -> tuple[np.ndarray, np.ndarray]:
    if protocol.mode == "official":
        return official_spectrum(signal)
    return direct_fft_spectrum(signal, protocol.nfft)


def failure_type(predicted_hr: float, gt_hr: float, bin_bpm: float) -> str:
    if not np.isfinite(predicted_hr) or not np.isfinite(gt_hr):
        return "nan"
    error = abs(predicted_hr - gt_hr)
    ratio = gt_hr / predicted_hr if predicted_hr > 0 else 0.0
    tolerance = max(4.0, 0.5 * bin_bpm)
    if error < tolerance:
        return "correct"
    if 1.7 < ratio < 2.3:
        return "sub_harmonic_half"
    if 2.5 < ratio < 3.5:
        return "sub_harmonic_third"
    if 0.3 < ratio < 0.6:
        return "super_harmonic_2x"
    if 0.6 < ratio < 0.8:
        return "super_harmonic_1p5x"
    if error <= bin_bpm:
        return "one_bin"
    if error <= 2 * bin_bpm:
        return "two_bins"
    if error <= 3 * bin_bpm:
        return "three_bins"
    return "large_error"


def top_spectrum_peaks(
    signal: np.ndarray,
    protocol: Protocol,
    bpm_min: float,
    bpm_max: float,
    prefix: str,
) -> dict[str, float]:
    bpm, power = measurement_spectrum(signal, protocol)
    valid = (bpm >= bpm_min) & (bpm <= bpm_max)
    bpm = bpm[valid]
    power = power[valid]
    if power.size == 0:
        return {}
    normalized = power / (np.max(power) + 1e-12)
    indices = np.argsort(normalized)[::-1][:3]
    result = {}
    for rank, index in enumerate(indices, start=1):
        result[f"{prefix}_top{rank}_bpm"] = float(bpm[index])
        result[f"{prefix}_top{rank}_power"] = float(normalized[index])
    return result


def measurement_row(
    run: EvaluationRun,
    protocol: Protocol,
    measurement: Measurement,
) -> dict[str, object]:
    prediction = np.asarray(measurement.prediction, dtype=np.float64)
    label = np.asarray(measurement.label, dtype=np.float64)
    signed_error = measurement.predicted_hr - measurement.gt_hr
    return {
        "run_name": run.name,
        "dataset": run.experiment.name,
        "protocol": protocol.name,
        "recording_id": measurement.recording_id,
        "measurement_id": measurement.measurement_id,
        "aggregation_unit": (
            "complete_recording" if protocol.mode == "official" else "window"
        ),
        "start_frame": measurement.start_frame,
        "end_frame": measurement.end_frame,
        "start_second": measurement.start_frame / FS,
        "end_second": measurement.end_frame / FS,
        "duration_seconds": (measurement.end_frame - measurement.start_frame) / FS,
        "predicted_hr_bpm": measurement.predicted_hr,
        "gt_hr_bpm": measurement.gt_hr,
        "signed_error_bpm": signed_error,
        "absolute_error_bpm": measurement.absolute_error,
        "absolute_percentage_error": (
            abs(signed_error / measurement.gt_hr) * 100
            if measurement.gt_hr != 0
            else float("nan")
        ),
        "waveform_pearson": measurement.waveform_pearson,
        "snr_db": measurement.snr_db,
        "signal_samples": min(prediction.size, label.size),
        "gt_signal_mean": finite_mean(label),
        "gt_signal_std": finite_std(label),
        "prediction_signal_mean": finite_mean(prediction),
        "prediction_signal_std": finite_std(prediction),
        "fft_bin_spacing_bpm": protocol.bin_spacing_bpm,
        "failure_type": failure_type(
            measurement.predicted_hr,
            measurement.gt_hr,
            protocol.bin_spacing_bpm,
        ),
    }


def peak_row(
    run: EvaluationRun,
    protocol: Protocol,
    measurement: Measurement,
) -> dict[str, object]:
    row = {
        "run_name": run.name,
        "dataset": run.experiment.name,
        "protocol": protocol.name,
        "recording_id": measurement.recording_id,
        "measurement_id": measurement.measurement_id,
        "gt_hr_bpm": measurement.gt_hr,
        "predicted_hr_bpm": measurement.predicted_hr,
        "failure_type": failure_type(
            measurement.predicted_hr,
            measurement.gt_hr,
            protocol.bin_spacing_bpm,
        ),
    }
    row.update(
        top_spectrum_peaks(
            measurement.label,
            protocol,
            protocol.gt_bpm_min,
            protocol.gt_bpm_max,
            "gt",
        )
    )
    row.update(
        top_spectrum_peaks(
            measurement.prediction,
            protocol,
            protocol.prediction_bpm_min,
            protocol.prediction_bpm_max,
            "prediction",
        )
    )
    return row


def recording_row(
    run: EvaluationRun,
    protocol: Protocol,
    recording_id: str,
    measurements: list[Measurement],
) -> dict[str, object]:
    errors = np.asarray([item.absolute_error for item in measurements])
    predicted_hr = np.asarray([item.predicted_hr for item in measurements])
    gt_hr = np.asarray([item.gt_hr for item in measurements])
    percentage_errors = np.abs((predicted_hr - gt_hr) / gt_hr) * 100
    signed_errors = predicted_hr - gt_hr
    return {
        "run_name": run.name,
        "dataset": run.experiment.name,
        "protocol": protocol.name,
        "recording_id": recording_id,
        "number_of_measurements": len(measurements),
        "mae_bpm": finite_mean(errors),
        "rmse_bpm": float(np.sqrt(finite_mean(errors**2))),
        "bias_bpm": finite_mean(signed_errors),
        "error_sd_bpm": finite_std(signed_errors),
        "mape_percent": finite_mean(percentage_errors),
        "hr_pearson": pearson(predicted_hr, gt_hr),
        "mean_waveform_pearson": finite_mean(
            [item.waveform_pearson for item in measurements]
        ),
        "mean_snr_db": finite_mean([item.snr_db for item in measurements]),
    }


def write_signal_sample_table(
    path: Path,
    measurements: list[Measurement],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "recording_id",
            "measurement_id",
            "sample_index",
            "time_seconds",
            "gt_signal",
            "predicted_signal",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for measurement in measurements:
            length = min(len(measurement.label), len(measurement.prediction))
            for sample_index in range(length):
                writer.writerow(
                    {
                        "recording_id": measurement.recording_id,
                        "measurement_id": measurement.measurement_id,
                        "sample_index": sample_index,
                        "time_seconds": (
                            measurement.start_frame + sample_index
                        )
                        / FS,
                        "gt_signal": measurement.label[sample_index],
                        "predicted_signal": measurement.prediction[sample_index],
                    }
                )


# =============================================================================
# INTERACTIVE SIGNAL AND PSD PLOTS
# =============================================================================


def make_signal_comparison_plot(
    measurements: list[Measurement],
    protocol: Protocol,
    output_path: Path,
    title: str,
) -> None:
    if not measurements:
        return
    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        subplot_titles=("GT and RhythmMamba Waveforms", "Frequency Spectra"),
        vertical_spacing=0.12,
    )
    traces_per_measurement = 4
    slider_steps = []

    for index, measurement in enumerate(measurements):
        visible = index == 0
        length = min(len(measurement.label), len(measurement.prediction))
        time = (
            np.arange(length, dtype=np.float64) + measurement.start_frame
        ) / FS
        gt_bpm, gt_power = measurement_spectrum(measurement.label[:length], protocol)
        pred_bpm, pred_power = measurement_spectrum(
            measurement.prediction[:length], protocol
        )
        gt_valid = (gt_bpm >= protocol.gt_bpm_min) & (gt_bpm <= protocol.gt_bpm_max)
        pred_valid = (pred_bpm >= protocol.prediction_bpm_min) & (
            pred_bpm <= protocol.prediction_bpm_max
        )

        figure.add_trace(
            go.Scatter(
                x=time,
                y=zscore(measurement.label[:length]),
                mode="lines",
                name=f"GT | {measurement.gt_hr:.2f} BPM",
                line=dict(color="#2E86AB", width=2),
                visible=visible,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=time,
                y=zscore(measurement.prediction[:length]),
                mode="lines",
                name=(
                    f"RhythmMamba | {measurement.predicted_hr:.2f} BPM | "
                    f"error={measurement.absolute_error:.2f}"
                ),
                line=dict(color="#F18F01", width=2),
                visible=visible,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=gt_bpm[gt_valid],
                y=gt_power[gt_valid] / (np.max(gt_power[gt_valid]) + 1e-12),
                mode="lines",
                name="GT spectrum",
                line=dict(color="#2E86AB", width=2),
                visible=visible,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=pred_bpm[pred_valid],
                y=pred_power[pred_valid] / (np.max(pred_power[pred_valid]) + 1e-12),
                mode="lines",
                name="RhythmMamba spectrum",
                line=dict(color="#F18F01", width=2),
                visible=visible,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        visibility = [False] * (traces_per_measurement * len(measurements))
        for offset in range(traces_per_measurement):
            visibility[traces_per_measurement * index + offset] = True
        slider_steps.append(
            dict(
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": (
                            f"{title}<br>Measurement {measurement.measurement_id} | "
                            f"GT={measurement.gt_hr:.2f}, "
                            f"Prediction={measurement.predicted_hr:.2f}, "
                            f"MAE={measurement.absolute_error:.2f} BPM"
                        )
                    },
                ],
                label=str(measurement.measurement_id),
            )
        )

    figure.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    figure.update_yaxes(title_text="Standardized amplitude", row=1, col=1)
    figure.update_xaxes(title_text="Heart rate (BPM)", row=2, col=1)
    figure.update_yaxes(title_text="Normalized power", row=2, col=1)
    figure.update_layout(
        title=title,
        template="plotly_white",
        width=1500,
        height=900,
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Measurement: "},
                pad={"t": 45},
                steps=slider_steps,
            )
        ],
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02),
        margin=dict(l=80, r=40, t=125, b=100),
    )
    figure.write_html(str(output_path), include_plotlyjs=PLOTLY_JS_MODE)


def make_psd_diagnostic_plot(
    measurements: list[Measurement],
    protocol: Protocol,
    output_path: Path,
    title: str,
) -> None:
    if not measurements:
        return
    figure = go.Figure()
    traces_per_measurement = 5
    slider_steps = []

    for index, measurement in enumerate(measurements):
        visible = index == 0
        gt_bpm, gt_power = measurement_spectrum(measurement.label, protocol)
        pred_bpm, pred_power = measurement_spectrum(measurement.prediction, protocol)
        gt_valid = (gt_bpm >= protocol.gt_bpm_min) & (gt_bpm <= protocol.gt_bpm_max)
        pred_valid = (pred_bpm >= protocol.prediction_bpm_min) & (
            pred_bpm <= protocol.prediction_bpm_max
        )
        gt_normalized = gt_power[gt_valid] / (np.max(gt_power[gt_valid]) + 1e-12)
        pred_normalized = pred_power[pred_valid] / (
            np.max(pred_power[pred_valid]) + 1e-12
        )

        figure.add_trace(
            go.Scatter(
                x=gt_bpm[gt_valid],
                y=gt_normalized,
                mode="lines",
                name="GT spectrum",
                line=dict(color="#2E86AB", width=2),
                visible=visible,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=pred_bpm[pred_valid],
                y=pred_normalized,
                mode="lines",
                name="RhythmMamba spectrum",
                line=dict(color="#F18F01", width=2),
                visible=visible,
            )
        )
        marker_values = [measurement.gt_hr, measurement.gt_hr / 2, measurement.gt_hr * 2]
        marker_names = ["GT", "GT/2", "GT×2"]
        marker_colors = ["#111111", "#D62728", "#9467BD"]
        marker_dashes = ["dash", "dashdot", "dot"]
        for value, name, color, dash in zip(
            marker_values, marker_names, marker_colors, marker_dashes
        ):
            figure.add_trace(
                go.Scatter(
                    x=[value, value],
                    y=[0, 1.05],
                    mode="lines",
                    name=f"{name}={value:.2f}",
                    line=dict(color=color, dash=dash),
                    visible=visible,
                )
            )

        visibility = [False] * (traces_per_measurement * len(measurements))
        for offset in range(traces_per_measurement):
            visibility[traces_per_measurement * index + offset] = True
        slider_steps.append(
            dict(
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": (
                            f"{title}<br>Measurement {measurement.measurement_id} | "
                            f"GT={measurement.gt_hr:.2f}, "
                            f"Prediction={measurement.predicted_hr:.2f}, "
                            f"{failure_type(measurement.predicted_hr, measurement.gt_hr, protocol.bin_spacing_bpm)}"
                        )
                    },
                ],
                label=str(measurement.measurement_id),
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="Heart rate (BPM)",
        yaxis_title="Normalized power",
        template="plotly_white",
        width=1450,
        height=750,
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Measurement: "},
                pad={"t": 45},
                steps=slider_steps,
            )
        ],
        legend=dict(orientation="v", x=1.02, y=1),
        margin=dict(l=80, r=220, t=125, b=100),
    )
    figure.write_html(str(output_path), include_plotlyjs=PLOTLY_JS_MODE)


# =============================================================================
# SUMMARY TABLES AND PLOTS
# =============================================================================


def summarize_protocol(
    run: EvaluationRun,
    protocol: Protocol,
    file_list: Path,
    measurement_rows: list[dict[str, object]],
    recording_rows: list[dict[str, object]],
) -> dict[str, object]:
    predicted_hr = np.asarray(
        [row["predicted_hr_bpm"] for row in measurement_rows], dtype=np.float64
    )
    gt_hr = np.asarray(
        [row["gt_hr_bpm"] for row in measurement_rows], dtype=np.float64
    )
    errors = np.abs(predicted_hr - gt_hr)
    recording_mae = np.asarray(
        [row["mae_bpm"] for row in recording_rows], dtype=np.float64
    )
    recording_rmse = np.asarray(
        [row["rmse_bpm"] for row in recording_rows], dtype=np.float64
    )
    valid = np.isfinite(predicted_hr) & np.isfinite(gt_hr) & np.isfinite(errors)
    predicted_hr = predicted_hr[valid]
    gt_hr = gt_hr[valid]
    errors = errors[valid]

    squared_errors = errors**2
    percentage_errors = np.abs((predicted_hr - gt_hr) / gt_hr) * 100
    hr_correlation = pearson(predicted_hr, gt_hr)
    snr_values = np.asarray(
        [row["snr_db"] for row in measurement_rows], dtype=np.float64
    )
    pearson_standard_error = (
        float(np.sqrt((1 - hr_correlation**2) / (len(errors) - 2)))
        if len(errors) > 2 and np.isfinite(hr_correlation)
        else float("nan")
    )

    return {
        "run_name": run.name,
        "description": run.description,
        "checkpoint": str(run.checkpoint.resolve()),
        "dataset": run.experiment.name,
        "split_begin": run.split_begin,
        "split_end": run.split_end,
        "file_list": str(file_list.resolve()),
        "protocol": protocol.name,
        "protocol_display_name": protocol.display_name,
        "aggregation": protocol.aggregation,
        "window_seconds": protocol.window_seconds,
        "stride_seconds": protocol.stride_seconds,
        "nfft": protocol.nfft,
        "fft_bin_spacing_bpm": protocol.bin_spacing_bpm,
        "bandpass_low_hz": protocol.bandpass_low_hz,
        "bandpass_high_hz": protocol.bandpass_high_hz,
        "bandpass_order": protocol.bandpass_order,
        "number_of_recordings": len(recording_rows),
        "number_of_measurements": int(errors.size),
        "primary_recording_macro_mae_bpm": finite_mean(recording_mae),
        "recording_macro_mae_standard_error_bpm": finite_standard_error(
            recording_mae
        ),
        "recording_macro_rmse_bpm": finite_mean(recording_rmse),
        "pooled_measurement_mae_bpm": finite_mean(errors),
        "pooled_measurement_rmse_bpm": float(np.sqrt(finite_mean(squared_errors))),
        "pooled_measurement_mape_percent": finite_mean(percentage_errors),
        "pooled_measurement_pearson": hr_correlation,
        "absolute_error_sd_bpm": finite_std(errors),
        "accuracy_within_5_bpm_percent": float(np.mean(errors <= 5) * 100),
        "pooled_mae_standard_error_bpm": finite_standard_error(errors),
        "reported_rmse_plus_minus_official_formula": finite_standard_error(
            squared_errors
        ),
        "pooled_mape_standard_error_percent": finite_standard_error(
            percentage_errors
        ),
        "pooled_pearson_standard_error": pearson_standard_error,
        "mean_waveform_pearson": finite_mean(
            [row["waveform_pearson"] for row in measurement_rows]
        ),
        "mean_snr_db": finite_mean(snr_values),
        "snr_standard_error_db": finite_standard_error(snr_values),
    }


def failure_summary_rows(
    measurement_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for row in measurement_rows:
        name = str(row["failure_type"])
        counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    return [
        {
            "failure_type": name,
            "count": count,
            "percentage": count / total * 100 if total else float("nan"),
        }
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def make_recording_mae_plot(
    recording_rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    rows = sorted(recording_rows, key=lambda row: str(row["recording_id"]))
    figure = go.Figure(
        go.Bar(
            x=[row["recording_id"] for row in rows],
            y=[row["mae_bpm"] for row in rows],
            marker_color="#386CB0",
            text=[f"{row['mae_bpm']:.2f}" for row in rows],
            textposition="outside",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Recording",
        yaxis_title="HR MAE (BPM)",
        template="plotly_white",
        width=max(1100, len(rows) * 38),
        height=600,
        xaxis_tickangle=-45,
    )
    figure.write_html(str(output_path), include_plotlyjs=PLOTLY_JS_MODE)


def make_hr_scatter_plot(
    measurement_rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    gt = np.asarray([row["gt_hr_bpm"] for row in measurement_rows])
    prediction = np.asarray([row["predicted_hr_bpm"] for row in measurement_rows])
    minimum = float(np.nanmin(np.concatenate([gt, prediction])))
    maximum = float(np.nanmax(np.concatenate([gt, prediction])))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=gt,
            y=prediction,
            mode="markers",
            marker=dict(color="#F18F01", size=7, opacity=0.7),
            text=[
                f"{row['recording_id']} / {row['measurement_id']}"
                for row in measurement_rows
            ],
            hovertemplate="%{text}<br>GT=%{x:.2f}<br>Prediction=%{y:.2f}<extra></extra>",
            name="Measurements",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Ideal",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="GT HR (BPM)",
        yaxis_title="Predicted HR (BPM)",
        template="plotly_white",
        width=750,
        height=700,
    )
    figure.write_html(str(output_path), include_plotlyjs=PLOTLY_JS_MODE)


def make_error_histogram(
    measurement_rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    errors = [row["absolute_error_bpm"] for row in measurement_rows]
    figure = go.Figure(
        go.Histogram(x=errors, nbinsx=40, marker_color="#7A5195")
    )
    figure.update_layout(
        title=title,
        xaxis_title="Absolute HR error (BPM)",
        yaxis_title="Count",
        template="plotly_white",
        width=850,
        height=600,
    )
    figure.write_html(str(output_path), include_plotlyjs=PLOTLY_JS_MODE)


def make_bland_altman_plot(
    measurement_rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    gt = np.asarray([row["gt_hr_bpm"] for row in measurement_rows], dtype=float)
    prediction = np.asarray(
        [row["predicted_hr_bpm"] for row in measurement_rows], dtype=float
    )
    averages = (prediction + gt) / 2
    differences = prediction - gt
    bias = float(np.mean(differences))
    standard_deviation = float(np.std(differences))
    upper = bias + 1.96 * standard_deviation
    lower = bias - 1.96 * standard_deviation
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=averages,
            y=differences,
            mode="markers",
            marker=dict(color="#EF5675", size=7, opacity=0.7),
            name="Measurements",
        )
    )
    for value, name, color, dash in [
        (bias, f"Bias={bias:.2f}", "black", "solid"),
        (upper, f"Upper={upper:.2f}", "red", "dash"),
        (lower, f"Lower={lower:.2f}", "red", "dash"),
    ]:
        figure.add_hline(y=value, line_color=color, line_dash=dash, annotation_text=name)
    figure.update_layout(
        title=title,
        xaxis_title="Mean of predicted and GT HR (BPM)",
        yaxis_title="Predicted − GT HR (BPM)",
        template="plotly_white",
        width=850,
        height=650,
    )
    figure.write_html(str(output_path), include_plotlyjs=PLOTLY_JS_MODE)


def make_failure_plot(
    failure_rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    figure = go.Figure(
        go.Bar(
            x=[row["failure_type"] for row in failure_rows],
            y=[row["percentage"] for row in failure_rows],
            marker_color="#FFA600",
            text=[f"{row['percentage']:.1f}%" for row in failure_rows],
            textposition="outside",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Failure category",
        yaxis_title="Measurements (%)",
        template="plotly_white",
        width=1050,
        height=600,
        xaxis_tickangle=-25,
    )
    figure.write_html(str(output_path), include_plotlyjs=PLOTLY_JS_MODE)


def write_summary_text(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "=" * 84,
        f"{summary['run_name']} — {summary['protocol_display_name']}",
        "=" * 84,
        f"Description                : {summary['description']}",
        f"Checkpoint                 : {summary['checkpoint']}",
        f"Dataset / split            : {summary['dataset']} / {summary['split_begin']}–{summary['split_end']}",
        f"Aggregation                : {summary['aggregation']}",
        f"Recordings                 : {summary['number_of_recordings']}",
        f"Expected recordings        : {summary.get('number_of_expected_recordings', summary['number_of_recordings'])}",
        f"Failed recordings          : {summary.get('number_of_failed_recordings', 0)}",
        f"Completion status          : {summary.get('completion_status', 'PASSED')}",
        f"Measurements               : {summary['number_of_measurements']}",
        f"Window / stride            : {summary['window_seconds']} / {summary['stride_seconds']} seconds",
        f"FFT bin spacing            : {summary['fft_bin_spacing_bpm']:.6f} BPM",
        f"Band-pass                  : {summary['bandpass_low_hz']}–{summary['bandpass_high_hz']} Hz, order {summary['bandpass_order']}",
        "-" * 84,
        f"PRIMARY recording-macro MAE: {summary['primary_recording_macro_mae_bpm']:.6f} BPM",
        f"Pooled measurement MAE     : {summary['pooled_measurement_mae_bpm']:.6f} BPM",
        f"Pooled measurement RMSE    : {summary['pooled_measurement_rmse_bpm']:.6f} BPM",
        f"Pooled measurement MAPE    : {summary['pooled_measurement_mape_percent']:.6f} %",
        f"Pooled measurement Pearson : {summary['pooled_measurement_pearson']:.6f}",
        f"Absolute-error SD           : {summary['absolute_error_sd_bpm']:.6f} BPM",
        f"Accuracy within 5 BPM      : {summary['accuracy_within_5_bpm_percent']:.3f} %",
        f"Recording-macro MAE SE     : {summary['recording_macro_mae_standard_error_bpm']:.6f} BPM",
        f"Pooled MAE standard error  : {summary['pooled_mae_standard_error_bpm']:.6f} BPM",
        f"Official-formula RMSE +/-  : {summary['reported_rmse_plus_minus_official_formula']:.6f}",
        f"Pooled MAPE standard error : {summary['pooled_mape_standard_error_percent']:.6f} %",
        f"Pooled Pearson std. error  : {summary['pooled_pearson_standard_error']:.6f}",
        f"Mean waveform Pearson      : {summary['mean_waveform_pearson']:.6f}",
        f"Mean SNR                    : {summary['mean_snr_db']:.6f} dB",
        f"SNR standard error         : {summary['snr_standard_error_db']:.6f} dB",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# ONE PROTOCOL AND ONE CHECKPOINT RUN
# =============================================================================


def run_protocol(
    model: RhythmMamba,
    run: EvaluationRun,
    protocol: Protocol,
    file_list: Path,
    recordings: dict[str, list[tuple[int, Path]]],
) -> dict[str, object]:
    directories = make_output_directories(run, protocol)
    measurement_rows: list[dict[str, object]] = []
    recording_rows: list[dict[str, object]] = []
    peak_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    metadata = read_recording_metadata(file_list)

    print()
    print("#" * 84)
    print(f"{run.name} — {protocol.display_name}")
    print("#" * 84)

    for recording_id, clip_paths in tqdm(
        recordings.items(),
        desc=f"{run.name}/{protocol.name}",
        ncols=100,
    ):
        try:
            frames, labels = load_recording(clip_paths)
            measurements = evaluate_recording(
                model,
                recording_id,
                frames,
                labels,
                protocol,
            )
            if not measurements:
                raise RuntimeError("Recording is shorter than the protocol window")

            new_measurements = [measurement_row(run, protocol, item) for item in measurements]
            new_peaks = [peak_row(run, protocol, item) for item in measurements]
            new_recording = recording_row(run, protocol, recording_id, measurements)
            extra = metadata.get(recording_id, {})
            for row in [*new_measurements, *new_peaks, new_recording]:
                for key, value in extra.items():
                    if key not in {"dataset", "recording_id"}:
                        row[key] = value
            measurement_rows.extend(new_measurements)
            peak_rows.extend(new_peaks)
            recording_rows.append(new_recording)

            file_stem = safe_name(recording_id)
            title = (
                f"{run.name} | {protocol.display_name} | Recording {recording_id}"
            )
            if SAVE_SIGNAL_SAMPLE_TABLES:
                write_signal_sample_table(
                    directories["signal_tables"] / f"{file_stem}_signals.csv",
                    measurements,
                )
            if GENERATE_SIGNAL_PLOTS:
                make_signal_comparison_plot(
                    measurements,
                    protocol,
                    directories["signals"] / f"{file_stem}_signal_comparison.html",
                    title,
                )
            if GENERATE_PSD_DIAGNOSTICS:
                make_psd_diagnostic_plot(
                    measurements,
                    protocol,
                    directories["diagnostics"] / f"{file_stem}_psd_diagnostic.html",
                    title,
                )
        except Exception as error:
            traceback.print_exc()
            error_rows.append(
                {
                    "run_name": run.name,
                    "protocol": protocol.name,
                    "recording_id": recording_id,
                    "error": str(error),
                }
            )

    if not measurement_rows or not recording_rows:
        raise RuntimeError(f"No valid results for {run.name}/{protocol.name}")

    failure_rows = failure_summary_rows(measurement_rows)
    summary = summarize_protocol(
        run,
        protocol,
        file_list,
        measurement_rows,
        recording_rows,
    )
    summary["number_of_expected_recordings"] = len(recordings)
    summary["number_of_failed_recordings"] = len(error_rows)
    summary["completion_status"] = "PASSED" if not error_rows else "PARTIAL_WITH_ERRORS"

    write_csv(directories["tables"] / "ALL_MEASUREMENT_RESULTS.csv", measurement_rows)
    write_csv(directories["tables"] / "ALL_RECORDING_RESULTS.csv", recording_rows)
    write_csv(
        directories["tables"] / "WORST_100_MEASUREMENTS.csv",
        sorted(
            measurement_rows,
            key=lambda row: float(row["absolute_error_bpm"]),
            reverse=True,
        )[:100],
    )
    write_csv(
        directories["tables"] / "NON_CORRECT_MEASUREMENTS.csv",
        [row for row in measurement_rows if row["failure_type"] != "correct"],
    )
    write_csv(directories["tables"] / "PSD_TOP_PEAKS_SUMMARY.csv", peak_rows)
    write_csv(directories["tables"] / "FAILURE_TYPE_SUMMARY.csv", failure_rows)
    group_candidates = {
        "condition",
        "condition_id",
        "task",
        "scenario",
        "session",
        "illumination",
        "sync_method",
    }
    metadata_fields = sorted(
        {key for item in metadata.values() for key in item if key in group_candidates}
    )
    write_csv(
        directories["tables"] / "GROUPED_RECORDING_SUMMARY.csv",
        grouped_recording_rows(recording_rows, metadata_fields),
    )
    write_csv(
        directories["tables"] / "IMAGE_QUALITY_ERROR_CORRELATION.csv",
        quality_error_correlation_rows(recording_rows),
    )
    write_csv(directories["tables"] / "ERRORS.csv", error_rows)
    write_csv(directories["tables"] / "EXTENDED_METRICS.csv", [summary])
    (directories["tables"] / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_text(directories["tables"] / "summary.txt", summary)

    if GENERATE_SUMMARY_PLOTS:
        plot_title = f"{run.name} — {protocol.display_name}"
        make_recording_mae_plot(
            recording_rows,
            directories["plots"] / "recording_mae.html",
            f"Recording-level MAE — {plot_title}",
        )
        make_hr_scatter_plot(
            measurement_rows,
            directories["plots"] / "predicted_vs_gt_hr.html",
            f"Predicted versus GT HR — {plot_title}",
        )
        make_error_histogram(
            measurement_rows,
            directories["plots"] / "absolute_error_distribution.html",
            f"Absolute-error distribution — {plot_title}",
        )
        make_bland_altman_plot(
            measurement_rows,
            directories["plots"] / "bland_altman.html",
            f"Bland–Altman analysis — {plot_title}",
        )
        make_failure_plot(
            failure_rows,
            directories["plots"] / "failure_categories.html",
            f"Failure categories — {plot_title}",
        )

    print(
        f"{run.name} | {protocol.name} | "
        f"MAE={summary['primary_recording_macro_mae_bpm']:.6f} BPM"
    )
    if error_rows:
        print(f"WARNING: {len(error_rows)} recordings failed; inspect tables/ERRORS.csv")
    print(f"Saved: {directories['base']}")
    return summary


# =============================================================================
# CROSS-PROTOCOL COMPARISON
# =============================================================================


def make_protocol_comparison_plot(
    summaries: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    protocols = [row["protocol"] for row in summaries]
    mae = [row["primary_recording_macro_mae_bpm"] for row in summaries]
    rmse = [row["pooled_measurement_rmse_bpm"] for row in summaries]
    accuracy = [row["accuracy_within_5_bpm_percent"] for row in summaries]
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Recording-macro MAE", "Pooled RMSE", "Accuracy within 5 BPM"),
    )
    figure.add_trace(go.Bar(x=protocols, y=mae, marker_color="#386CB0"), row=1, col=1)
    figure.add_trace(go.Bar(x=protocols, y=rmse, marker_color="#F18F01"), row=1, col=2)
    figure.add_trace(go.Bar(x=protocols, y=accuracy, marker_color="#7A5195"), row=1, col=3)
    figure.update_yaxes(title_text="BPM", row=1, col=1)
    figure.update_yaxes(title_text="BPM", row=1, col=2)
    figure.update_yaxes(title_text="Percent", row=1, col=3)
    figure.update_layout(
        title=title,
        template="plotly_white",
        width=1450,
        height=600,
        showlegend=False,
    )
    figure.write_html(str(output_path), include_plotlyjs=PLOTLY_JS_MODE)


def write_protocol_comparison_note(path: Path) -> None:
    text = """Protocol comparison interpretation
==================================

official_mamba
    One HR estimate per reconstructed recording using the exact verified
    RhythmMamba/rPPG-Toolbox detrending, first-order band-pass and Welch method.

old
    One HR estimate per overlapping 8-second window. Errors are first averaged
    within each recording and then averaged across recordings.

prism
    One HR estimate per non-overlapping 10-second window using the PRISM
    evaluation protocol only. This does not implement the full PRISM algorithm.

Because the aggregation units differ, the protocol values measure different
evaluation questions. They should be presented as a protocol-sensitivity study,
not as three interchangeable estimates of one identical quantity.
"""
    path.write_text(text, encoding="utf-8")


def write_output_guide(path: Path) -> None:
    text = """RhythmMamba multi-protocol evaluation outputs
================================================

Each experiment directory contains three protocol directories:

official_mamba
    Exact verified RhythmMamba evaluation: reconstruct the complete recording,
    smoothness-prior detrending, first-order 0.75-2.5 Hz Butterworth filter,
    and Welch dominant-frequency HR in 45-150 BPM.

old
    8-second windows with 1-second stride, third-order 0.67-3.0 Hz filter,
    and a direct 240-point FFT at 30 FPS.

prism
    10-second non-overlapping windows, second-order 0.75-2.5 Hz filter, and
    a 16,384-point zero-padded FFT. This is the PRISM evaluation protocol only,
    not the complete PRISM prediction algorithm.

Inside each protocol directory:

tables/
    Measurement-level, recording-level, spectral-peak, failure, error, and
    extended-summary CSV files plus JSON/text summaries.

plots/
    Recording MAE, predicted-vs-GT HR, error histogram, Bland-Altman, and
    failure-category interactive HTML plots.

signal_comparisons/
    Interactive GT-vs-RhythmMamba waveform and spectrum plots per recording.

signal_tables/
    Sample-level GT and predicted waveform values per recording.

diagnostics/
    Interactive PSD views with GT, half-GT, and double-GT markers.

The top-level all_results_summary.csv contains one row for every
experiment/protocol combination. Checkpoints and caches are never modified.
"""
    path.write_text(text, encoding="utf-8")


# =============================================================================
# VALIDATION AND MAIN
# =============================================================================


def validate_configuration() -> None:
    unknown_runs = set(RUNS_TO_EVALUATE) - set(EVALUATION_RUNS)
    unknown_protocols = set(PROTOCOLS_TO_RUN) - set(PROTOCOLS)
    if unknown_runs:
        raise KeyError(f"Unknown evaluation runs: {sorted(unknown_runs)}")
    if unknown_protocols:
        raise KeyError(f"Unknown protocols: {sorted(unknown_protocols)}")
    if WINDOW_BATCH_SIZE < 1:
        raise ValueError("WINDOW_BATCH_SIZE must be at least one")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RhythmMamba evaluation")

    missing = [
        EVALUATION_RUNS[name].checkpoint
        for name in RUNS_TO_EVALUATE
        if not EVALUATION_RUNS[name].checkpoint.is_file()
    ]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Configured checkpoints are missing:\n{paths}\n"
            "Edit the paths at the top of eval_protocols.py."
        )

    for name in RUNS_TO_EVALUATE:
        run = EVALUATION_RUNS[name]
        find_file_list(run.experiment, run.split_begin, run.split_end)


def main() -> None:
    validate_configuration()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_output_guide(OUTPUT_ROOT / "README.txt")

    print("=" * 84)
    print("RHYTHMMAMBA THREE-PROTOCOL MEGA EVALUATION")
    print("=" * 84)
    print(f"Runs      : {RUNS_TO_EVALUATE}")
    print(f"Protocols : {PROTOCOLS_TO_RUN}")
    print(f"Output    : {OUTPUT_ROOT}")

    all_summaries: list[dict[str, object]] = []

    for run_name in RUNS_TO_EVALUATE:
        run = EVALUATION_RUNS[run_name]
        file_list = find_file_list(run.experiment, run.split_begin, run.split_end)
        recordings = read_manifest(file_list)
        model = load_model(run.checkpoint)
        run_summaries = []

        for protocol_name in PROTOCOLS_TO_RUN:
            summary = run_protocol(
                model,
                run,
                PROTOCOLS[protocol_name],
                file_list,
                recordings,
            )
            run_summaries.append(summary)
            all_summaries.append(summary)

        comparison_directory = OUTPUT_ROOT / run.name / "protocol_comparison"
        comparison_directory.mkdir(parents=True, exist_ok=True)
        write_csv(
            comparison_directory / "PROTOCOL_COMPARISON.csv",
            run_summaries,
        )
        make_protocol_comparison_plot(
            run_summaries,
            comparison_directory / "protocol_comparison.html",
            f"Protocol comparison — {run.name}",
        )
        write_protocol_comparison_note(
            comparison_directory / "INTERPRETATION.txt"
        )

        del model
        torch.cuda.empty_cache()

    write_csv(OUTPUT_ROOT / "all_results_summary.csv", all_summaries)
    print()
    print("=" * 84)
    print("MEGA EVALUATION COMPLETED")
    print("=" * 84)
    print(f"Combined summary: {OUTPUT_ROOT / 'all_results_summary.csv'}")


if __name__ == "__main__":
    main()
