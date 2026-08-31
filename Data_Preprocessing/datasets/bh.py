"""Native BH-rPPG adapter (35 subjects x three illumination conditions)."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

try:
    from ..common import Recording, crop_resize_frame_sequence, temporal_resample
    from ..settings import TARGET_FPS
    from .shared import subject_split
except ImportError:
    from common import Recording, crop_resize_frame_sequence, temporal_resample
    from settings import TARGET_FPS
    from datasets.shared import subject_split


BH_ID = re.compile(r"^(\d+)_(0|1|2)$")
CONDITION_NAMES = {0: "low", 1: "medium", 2: "high"}


class BhAdapter:
    name = "BH"

    def discover(self, raw_root: Path) -> list[Recording]:
        recordings = []
        for directory in Path(raw_root).expanduser().resolve().iterdir():
            if not directory.is_dir():
                continue
            match = BH_ID.fullmatch(directory.name)
            if match is None:
                continue
            subject, condition = map(int, match.groups())
            frame_directory = directory / directory.name
            timestamp_path = directory / "timestamps.csv"
            wave_path = directory / "wave.csv"
            if not frame_directory.is_dir() or not timestamp_path.is_file() or not wave_path.is_file():
                raise FileNotFoundError(f"Incomplete BH-rPPG recording: {directory}")
            recordings.append(
                Recording(
                    dataset=self.name,
                    source_id=directory.name,
                    saved_id=f"BH_s{subject:02d}_{CONDITION_NAMES[condition]}",
                    frame_source=str(frame_directory),
                    label_source=str(wave_path),
                    subject_id=subject,
                    metadata=(("condition_id", str(condition)), ("condition", CONDITION_NAMES[condition])),
                )
            )
        return sorted(recordings, key=lambda item: (item.subject_id, item.source_id))

    split = staticmethod(subject_split)

    @staticmethod
    def _paths(recording: Recording) -> list[Path]:
        paths = sorted(Path(recording.frame_source).glob("Frame_*.png"))
        if not paths:
            raise FileNotFoundError(f"No BH-rPPG PNG frames in {recording.frame_source}")
        return paths

    def read_aligned(self, recording: Recording) -> tuple[np.ndarray, np.ndarray]:
        paths = self._paths(recording)

        def rgb_frames():
            for path in paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    raise RuntimeError(f"OpenCV could not read {path}")
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames = crop_resize_frame_sequence(rgb_frames())
        recording_root = Path(recording.frame_source).parent
        frame_times = np.loadtxt(recording_root / "timestamps.csv", dtype=np.float64) / 1000.0
        label = np.genfromtxt(recording.label_source, delimiter=",", names=True)["Wave"]
        duration = frame_times[-1] + float(np.median(np.diff(frame_times)))
        label_times = np.linspace(0.0, duration, label.size, endpoint=False)
        return temporal_resample(frames, frame_times, label, label_times, TARGET_FPS)

    def probe(self, recording: Recording) -> dict[str, object]:
        paths = self._paths(recording)
        first = cv2.imread(str(paths[0]))
        times = np.loadtxt(Path(recording.frame_source).parent / "timestamps.csv") / 1000.0
        label = np.genfromtxt(recording.label_source, delimiter=",", names=True)["Wave"]
        return {
            "source_id": recording.source_id,
            "frame_count": len(paths),
            "duration_seconds": float(times[-1]),
            "derived_fps": float((len(times) - 1) / (times[-1] - times[0])),
            "first_frame_shape": tuple(first.shape),
            "label_count": int(label.size),
            "condition": dict(recording.metadata)["condition"],
        }
