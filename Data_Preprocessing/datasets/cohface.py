"""Native COHFACE video/HDF5 adapter."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

try:
    from ..common import Recording, temporal_resample
    from ..settings import TARGET_FPS
    from .shared import load_small_video, subject_split, video_probe
except ImportError:
    from common import Recording, temporal_resample
    from settings import TARGET_FPS
    from datasets.shared import load_small_video, subject_split, video_probe


COHFACE_ID = re.compile(r"^Subj_(\d+)_(\d+)$")


class CohfaceAdapter:
    name = "COHFACE"

    def discover(self, raw_root: Path) -> list[Recording]:
        recordings = []
        for directory in Path(raw_root).expanduser().resolve().glob("Subj_*_*"):
            match = COHFACE_ID.fullmatch(directory.name)
            if not directory.is_dir() or match is None:
                continue
            subject, session = map(int, match.groups())
            video = directory / f"data_{subject}_{session}.avi"
            label = directory / f"data_{subject}_{session}.hdf5"
            if not video.is_file() or not label.is_file():
                raise FileNotFoundError(f"Incomplete COHFACE pair in {directory}")
            illumination = "lamp" if session in (0, 1) else "natural"
            recordings.append(
                Recording(
                    dataset=self.name,
                    source_id=f"{subject}_{session}",
                    saved_id=f"COHFACE_s{subject:02d}_session{session}_{illumination}",
                    frame_source=str(video),
                    label_source=str(label),
                    subject_id=subject,
                    metadata=(("session", str(session)), ("illumination", illumination)),
                )
            )
        return sorted(recordings, key=lambda item: (item.subject_id, item.source_id))

    split = staticmethod(subject_split)

    @staticmethod
    def _label(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
        try:
            import h5py
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("COHFACE requires h5py: python -m pip install h5py") from error
        with h5py.File(path, "r") as handle:
            pulse = np.asarray(handle["pulse"], dtype=np.float64)
            times = np.asarray(handle["time"], dtype=np.float64)
            rate = float(np.asarray(handle.attrs["sample-rate-hz"]).reshape(-1)[0])
        return pulse, times, rate

    def read_aligned(self, recording: Recording) -> tuple[np.ndarray, np.ndarray]:
        frames, fps = load_small_video(recording.frame_source)
        label, label_times, _ = self._label(recording.label_source)
        frame_times = np.arange(frames.shape[0], dtype=np.float64) / fps
        return temporal_resample(frames, frame_times, label, label_times, TARGET_FPS)

    def probe(self, recording: Recording) -> dict[str, object]:
        result = video_probe(recording.frame_source)
        label, times, rate = self._label(recording.label_source)
        result.update(
            source_id=recording.source_id,
            label_count=int(label.size),
            bvp_rate_hz=rate,
            bvp_duration_seconds=float(times[-1]),
            **dict(recording.metadata),
        )
        return result
