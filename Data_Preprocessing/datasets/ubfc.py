"""Native UBFC-rPPG dataset adapter."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

try:
    from ..common import Recording
except ImportError:
    from common import Recording


VID_ID = re.compile(r"^vid_(\d+)$")
SUBJECT_ID = re.compile(r"^subject(\d+)$")


class UbfcAdapter:
    name = "UBFC"

    @staticmethod
    def _directory_number(directory: Path) -> int | None:
        match = VID_ID.fullmatch(directory.name) or SUBJECT_ID.fullmatch(directory.name)
        return int(match.group(1)) if match else None

    def discover(self, raw_root: Path) -> list[Recording]:
        """Accept native vid_N names or official-compatible subjectN names."""
        raw_root = Path(raw_root).expanduser().resolve()
        native = sorted(path for path in raw_root.glob("vid_*") if path.is_dir())
        compatible = sorted(path for path in raw_root.glob("subject*") if path.is_dir())
        directories = native if native else compatible

        recordings = []
        for directory in directories:
            number = self._directory_number(directory)
            if number is None:
                continue
            video_candidates = (directory / f"vid_{number}.avi", directory / "vid.avi")
            label_candidates = (
                directory / f"ground_truth_{number}.txt",
                directory / "ground_truth.txt",
            )
            video_path = next((path for path in video_candidates if path.is_file()), None)
            label_path = next((path for path in label_candidates if path.is_file()), None)
            if video_path is None or label_path is None:
                raise FileNotFoundError(
                    f"Incomplete UBFC recording in {directory}: "
                    f"video={video_path}, label={label_path}"
                )
            recordings.append(
                Recording(
                    dataset=self.name,
                    source_id=directory.name,
                    saved_id=f"subject{number}",
                    frame_source=str(video_path),
                    label_source=str(label_path),
                    subject_id=number,
                )
            )
        # This retains the official lexicographic directory split order.
        return recordings

    @staticmethod
    def split(recordings: list[Recording], begin: float, end: float) -> list[Recording]:
        if begin == 0 and end == 1:
            return list(recordings)
        return recordings[int(begin * len(recordings)) : int(end * len(recordings))]

    @staticmethod
    def read_frames(recording: Recording) -> np.ndarray:
        capture = cv2.VideoCapture(recording.frame_source)
        capture.set(cv2.CAP_PROP_POS_MSEC, 0)
        frames = []
        success, frame = capture.read()
        while success:
            frames.append(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB))
            success, frame = capture.read()
        capture.release()
        if not frames:
            raise RuntimeError(f"OpenCV could not read frames from {recording.frame_source}")
        return np.asarray(frames)

    @staticmethod
    def read_label(recording: Recording) -> np.ndarray:
        with Path(recording.label_source).open("r") as handle:
            lines = handle.read().split("\n")
        waveform = [float(value) for value in lines[0].split()]
        if not waveform:
            raise RuntimeError(f"UBFC waveform is empty: {recording.label_source}")
        return np.asarray(waveform)

    @staticmethod
    def align_label(label: np.ndarray, frame_count: int) -> np.ndarray:
        # Deliberately preserves the official UBFC loader: no resampling.
        return label

    def probe(self, recording: Recording) -> dict[str, object]:
        capture = cv2.VideoCapture(recording.frame_source)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        success, frame = capture.read()
        capture.release()
        if not success:
            raise RuntimeError(f"OpenCV could not read {recording.frame_source}")
        return {
            "source_id": recording.source_id,
            "frame_count": frame_count,
            "fps": fps,
            "resolution": (width, height),
            "first_frame_shape": tuple(frame.shape),
            "label_count": int(self.read_label(recording).shape[0]),
            "frame_source": recording.frame_source,
            "label_source": recording.label_source,
        }

