"""Native UBFC-PHYS adapter for the locally available s1-s20 subset."""

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


SUBJECT = re.compile(r"^s(\d+)$")


class UbfcPhysAdapter:
    name = "UBFC_PHYS"

    def discover(self, raw_root: Path) -> list[Recording]:
        recordings = []
        for directory in Path(raw_root).expanduser().resolve().glob("s*"):
            match = SUBJECT.fullmatch(directory.name)
            if not directory.is_dir() or match is None:
                continue
            subject = int(match.group(1))
            info_path = directory / f"info_s{subject}.txt"
            lines = info_path.read_text(encoding="utf-8").splitlines()
            scenario = lines[2].strip().lower() if len(lines) >= 3 else "unknown"
            for task in ("T1", "T2", "T3"):
                video = directory / f"vid_s{subject}_{task}.avi"
                bvp = directory / f"bvp_s{subject}_{task}.csv"
                if not video.is_file() or not bvp.is_file():
                    raise FileNotFoundError(f"Incomplete UBFC-PHYS pair: {video}, {bvp}")
                recordings.append(
                    Recording(
                        dataset=self.name,
                        source_id=f"s{subject}_{task}",
                        saved_id=f"UBFC_PHYS_s{subject:02d}_{task}_{scenario}",
                        frame_source=str(video),
                        label_source=str(bvp),
                        subject_id=subject,
                        metadata=(("task", task), ("scenario", scenario)),
                    )
                )
        return sorted(recordings, key=lambda item: (item.subject_id, item.source_id))

    split = staticmethod(subject_split)

    @staticmethod
    def read_aligned(recording: Recording) -> tuple[np.ndarray, np.ndarray]:
        frames, fps = load_small_video(recording.frame_source)
        label = np.loadtxt(recording.label_source, delimiter=",", dtype=np.float64)
        frame_times = np.arange(frames.shape[0], dtype=np.float64) / fps
        label_times = np.arange(label.size, dtype=np.float64) / 64.0
        return temporal_resample(frames, frame_times, label, label_times, TARGET_FPS)

    def probe(self, recording: Recording) -> dict[str, object]:
        result = video_probe(recording.frame_source)
        label = np.loadtxt(recording.label_source, delimiter=",")
        result.update(
            source_id=recording.source_id,
            label_count=int(label.size),
            bvp_rate_hz=64.0,
            **dict(recording.metadata),
        )
        return result
