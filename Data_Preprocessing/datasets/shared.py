"""Utilities shared by the additional native-layout dataset adapters."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from ..common import Recording, crop_resize_frame_sequence, video_frame_iterator
except ImportError:
    from common import Recording, crop_resize_frame_sequence, video_frame_iterator


def subject_split(
    recordings: list[Recording], begin: float, end: float
) -> list[Recording]:
    """Create deterministic subject-disjoint fractional splits."""
    if begin == 0 and end == 1:
        return list(recordings)
    subjects = sorted({item.subject_id for item in recordings})
    if None in subjects:
        raise RuntimeError("Subject-wise split requested with a missing subject id")
    chosen = set(subjects[int(begin * len(subjects)) : int(end * len(subjects))])
    return [item for item in recordings if item.subject_id in chosen]


def video_probe(path: str | Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    result = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "resolution": (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    }
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"OpenCV could not read {path}")
    result["first_frame_shape"] = tuple(frame.shape)
    return result


def load_small_video(path: str | Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid FPS metadata in {path}: {fps}")
    return crop_resize_frame_sequence(video_frame_iterator(path)), fps
