"""Compare three defensible TokyoTech video/contact-PPG alignments.

This is an audit, not preprocessing.  It extracts a lightweight face RGB trace
from all nine 30-fps videos for each subject, compares first-180s, last-180s,
and full-duration-rescale PPG candidates, and writes transparent per-window
scores.  The user must inspect the report before enabling TokyoTech in
settings.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, periodogram

try:
    from .datasets.tokyotech import aligned_subject_ppg
    from .settings import TOKYOTECH_RAW_ROOT, TOKYOTECH_SYNC_REPORT
except ImportError:
    from datasets.tokyotech import aligned_subject_ppg
    from settings import TOKYOTECH_RAW_ROOT, TOKYOTECH_SYNC_REPORT


METHODS = ("first_180_seconds", "last_180_seconds", "rescale_full_duration")


def bandpass(signal: np.ndarray, fs: float) -> np.ndarray:
    b, a = butter(2, [0.7 / (fs / 2), 3.0 / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, np.asarray(signal, dtype=np.float64))


def hr(signal: np.ndarray, fs: float) -> float:
    frequencies, power = periodogram(bandpass(signal, fs), fs=fs, nfft=16384)
    keep = (frequencies >= 0.7) & (frequencies <= 3.0)
    return float(frequencies[keep][np.argmax(power[keep])] * 60.0)


def face_rgb_trace(videos: list[Path]) -> np.ndarray:
    traces = []
    box = None
    cascade = cv2.CascadeClassifier(
        str(Path(__file__).resolve().parent / "assets" / "haarcascade_frontalface_default.xml")
    )
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if box is None:
                faces = cascade.detectMultiScale(rgb)
                if len(faces):
                    box = max(faces, key=lambda item: item[2])
                else:
                    h, w = rgb.shape[:2]
                    box = (w // 4, h // 5, w // 2, 3 * h // 5)
            x, y, width, height = map(int, box)
            roi = rgb[max(y, 0):min(y + height, rgb.shape[0]), max(x, 0):min(x + width, rgb.shape[1])]
            traces.append(np.mean(roi, axis=(0, 1)))
        capture.release()
    return np.asarray(traces, dtype=np.float64)


def pos_signal(rgb: np.ndarray) -> np.ndarray:
    normalized = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-12) - 1.0
    x = 3.0 * normalized[:, 0] - 2.0 * normalized[:, 1]
    y = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]
    return x + (np.std(x) / (np.std(y) + 1e-12)) * y


def best_lag_correlation(first: np.ndarray, second: np.ndarray, max_lag: int = 150) -> float:
    first = (first - np.mean(first)) / (np.std(first) + 1e-12)
    second = (second - np.mean(second)) / (np.std(second) + 1e-12)
    values = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            values.append(np.corrcoef(first[-lag:], second[:lag])[0, 1])
        elif lag > 0:
            values.append(np.corrcoef(first[:-lag], second[lag:])[0, 1])
        else:
            values.append(np.corrcoef(first, second)[0, 1])
    return float(np.nanmax(np.abs(values)))


def main() -> None:
    root = Path(TOKYOTECH_RAW_ROOT)
    window_rows = []
    subject_rows = []
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit()):
        videos = sorted((subject_dir / "30fps").glob("*.avi"))
        video_signal = bandpass(pos_signal(face_rgb_trace(videos)), 30.0)
        method_scores = []
        for method in METHODS:
            ppg = aligned_subject_ppg(subject_dir / "contactPPG.mat", method)
            ppg30 = np.interp(np.arange(5400) / 30.0, np.arange(ppg.size) / 2048.0, ppg)
            ppg30 = bandpass(ppg30, 30.0)
            errors = []
            correlations = []
            for segment in range(9):
                start, end = segment * 600, (segment + 1) * 600
                video_part, ppg_part = video_signal[start:end], ppg30[start:end]
                error = abs(hr(video_part, 30.0) - hr(ppg_part, 30.0))
                correlation = best_lag_correlation(video_part, ppg_part)
                errors.append(error)
                correlations.append(correlation)
                window_rows.append({
                    "subject": subject_dir.name,
                    "segment": segment + 1,
                    "method": method,
                    "hr_absolute_error_bpm": error,
                    "best_lag_absolute_correlation": correlation,
                })
            method_scores.append((float(np.mean(errors)), -float(np.mean(correlations)), method))
        diagnostic_best = min(method_scores)[2]
        subject_rows.append({
            "subject": subject_dir.name,
            "contact_ppg_duration_seconds": np.asarray(
                __import__("scipy.io", fromlist=["loadmat"]).loadmat(subject_dir / "contactPPG.mat")["dataA"]
            ).size / 2048.0,
            "recommended_method": "first_180_seconds",
            "diagnostic_best_method": diagnostic_best,
            "decision_rule": "Use the protocol-defined first 180 seconds; discard extra PPG tail without time warping.",
        })

    output = Path(TOKYOTECH_SYNC_REPORT)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)
    output.write_text(json.dumps({"subjects": subject_rows}, indent=2) + "\n", encoding="utf-8")
    print(f"Detailed audit : {output.with_suffix('.csv')}")
    print(f"Recommendations: {output}")
    print("Inspect both files before enabling TokyoTech preprocessing in settings.py.")


if __name__ == "__main__":
    main()
