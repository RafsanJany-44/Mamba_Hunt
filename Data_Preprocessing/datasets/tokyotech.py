"""TokyoTech 30-fps adapter with an explicit synchronization safety gate."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from ..common import Recording, crop_resize_frame_sequence, temporal_resample, video_frame_iterator
    from ..settings import (
        TARGET_FPS,
        TOKYOTECH_ACCEPT_AUDIT_RECOMMENDATIONS,
        TOKYOTECH_SYNC_REPORT,
    )
    from .shared import load_small_video, subject_split, video_probe
except ImportError:
    from common import Recording, crop_resize_frame_sequence, temporal_resample, video_frame_iterator
    from settings import TARGET_FPS, TOKYOTECH_ACCEPT_AUDIT_RECOMMENDATIONS, TOKYOTECH_SYNC_REPORT
    from datasets.shared import load_small_video, subject_split, video_probe


SUBJECT = re.compile(r"^\d{2}$")
VIDEO = re.compile(r"^(\d{4})to(\d{4})\.avi$")


@lru_cache(maxsize=1)
def _accepted_methods() -> dict[str, str]:
    if not TOKYOTECH_ACCEPT_AUDIT_RECOMMENDATIONS:
        raise RuntimeError(
            "TokyoTech preprocessing is intentionally blocked. Run "
            "Data_Preprocessing/audit_tokyotech_sync.py, inspect the report, "
            "then set TOKYOTECH_ACCEPT_AUDIT_RECOMMENDATIONS=True in settings.py."
        )
    path = Path(TOKYOTECH_SYNC_REPORT)
    if not path.is_file():
        raise FileNotFoundError(f"TokyoTech synchronization report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = {
        str(row["subject"]): str(row["recommended_method"])
        for row in payload.get("subjects", [])
    }
    required = {f"{index:02d}" for index in range(1, 10)}
    if set(methods) != required:
        raise RuntimeError(f"TokyoTech audit must contain subjects 01-09, found {sorted(methods)}")
    return methods


def aligned_subject_ppg(mat_path: Path, method: str) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("TokyoTech requires SciPy: python -m pip install scipy") from error
    signal = np.asarray(loadmat(mat_path)["dataA"], dtype=np.float64).reshape(-1)
    target_samples = int(180 * 2048)
    if method == "first_180_seconds":
        selected = signal[:target_samples]
    elif method == "last_180_seconds":
        selected = signal[-target_samples:]
    elif method == "rescale_full_duration":
        selected = np.interp(
            np.linspace(0, signal.size - 1, target_samples),
            np.arange(signal.size),
            signal,
        )
    else:
        raise ValueError(f"Unknown TokyoTech alignment method: {method}")
    if selected.size != target_samples:
        raise RuntimeError(f"TokyoTech PPG is too short in {mat_path}: {selected.size}")
    return np.asarray(selected, dtype=np.float64)


class TokyoTechAdapter:
    name = "TOKYOTECH"

    def discover(self, raw_root: Path) -> list[Recording]:
        recordings = []
        root = Path(raw_root).expanduser().resolve()
        accepted = _accepted_methods() if TOKYOTECH_ACCEPT_AUDIT_RECOMMENDATIONS else None
        for subject_directory in root.iterdir():
            if not subject_directory.is_dir() or SUBJECT.fullmatch(subject_directory.name) is None:
                continue
            subject = int(subject_directory.name)
            mat_path = subject_directory / "contactPPG.mat"
            videos = sorted((subject_directory / "30fps").glob("*.avi"))
            if len(videos) != 9 or not mat_path.is_file():
                raise FileNotFoundError(
                    f"TokyoTech subject {subject_directory.name} needs nine 30fps videos and contactPPG.mat"
                )
            for video in videos:
                match = VIDEO.fullmatch(video.name)
                if match is None:
                    raise ValueError(f"Unexpected TokyoTech video name: {video}")
            # The nine files are consecutive 20-second pieces of one 180-second
            # acquisition.  Treating them as one recording avoids discarding a
            # 120-frame remainder nine separate times during 160-frame chunking.
            recordings.append(
                Recording(
                    dataset=self.name,
                    source_id=subject_directory.name,
                    saved_id=f"TOKYOTECH_s{subject:02d}",
                    frame_source=str(subject_directory / "30fps"),
                    label_source=str(mat_path),
                    subject_id=subject,
                    metadata=(
                        ("segments", "9"),
                        ("duration_seconds", "180"),
                        ("sync_method", accepted[subject_directory.name] if accepted else "pending_audit"),
                    ),
                )
            )
        return sorted(recordings, key=lambda item: item.subject_id)

    split = staticmethod(subject_split)

    @staticmethod
    def read_aligned(recording: Recording) -> tuple[np.ndarray, np.ndarray]:
        subject = f"{int(recording.subject_id):02d}"
        method = _accepted_methods()[subject]
        label = aligned_subject_ppg(Path(recording.label_source), method)
        videos = sorted(Path(recording.frame_source).glob("*.avi"))

        def all_frames():
            for video in videos:
                yield from video_frame_iterator(video)

        frames = crop_resize_frame_sequence(all_frames())
        frame_times = np.arange(frames.shape[0], dtype=np.float64) / 30.0
        label_times = np.arange(label.size, dtype=np.float64) / 2048.0
        return temporal_resample(frames, frame_times, label, label_times, TARGET_FPS)

    def probe(self, recording: Recording) -> dict[str, object]:
        videos = sorted(Path(recording.frame_source).glob("*.avi"))
        result = video_probe(videos[0])
        try:
            from scipy.io import loadmat
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("TokyoTech requires SciPy") from error
        ppg = np.asarray(loadmat(recording.label_source)["dataA"]).reshape(-1)
        result.update(
            source_id=recording.source_id,
            video_segments=len(videos),
            total_video_frames=sum(int(video_probe(path)["frame_count"]) for path in videos),
            contact_ppg_samples=int(ppg.size),
            contact_ppg_duration_seconds=float(ppg.size / 2048.0),
            synchronization_status=(
                "audit accepted" if TOKYOTECH_ACCEPT_AUDIT_RECOMMENDATIONS else "audit required"
            ),
        )
        return result
