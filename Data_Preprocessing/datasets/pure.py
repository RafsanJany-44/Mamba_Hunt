"""Native PURE dataset adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

try:
    from ..common import Recording, resample_ppg
except ImportError:
    from common import Recording, resample_ppg


PURE_ID = re.compile(r"^\d{2}-\d{2}$")


class PureAdapter:
    name = "PURE"

    @staticmethod
    def _pairs_in(directory: Path) -> list[tuple[Path, Path]]:
        pairs = []
        if not directory.is_dir():
            return pairs
        for json_path in sorted(directory.glob("*.json")):
            if not PURE_ID.fullmatch(json_path.stem):
                continue
            image_directory = directory / json_path.stem
            if image_directory.is_dir():
                pairs.append((image_directory, json_path))
        return pairs

    def discover(self, raw_root: Path) -> list[Recording]:
        """Accept either PURE itself or the user's PURE/ALL/ALL directory."""
        raw_root = Path(raw_root).expanduser().resolve()
        preferred_roots = (raw_root, raw_root / "ALL" / "ALL")
        candidates = [(root, self._pairs_in(root)) for root in preferred_roots]
        root, pairs = max(candidates, key=lambda item: len(item[1]))

        if not pairs:
            # Fallback for another PURE extraction layout.  Duplicate stems are
            # rejected so an ambiguous root cannot silently mix dataset copies.
            recursive_pairs = []
            for json_path in raw_root.rglob("*.json"):
                if PURE_ID.fullmatch(json_path.stem):
                    image_directory = json_path.parent / json_path.stem
                    if image_directory.is_dir():
                        recursive_pairs.append((image_directory, json_path))
            pairs = sorted(recursive_pairs, key=lambda pair: str(pair[1]))
            stems = [json_path.stem for _, json_path in pairs]
            if len(stems) != len(set(stems)):
                raise RuntimeError(
                    f"Multiple PURE copies were found below {raw_root}. "
                    "Set PURE_RAW_ROOT directly to the directory containing 01-01.json."
                )

        recordings = []
        for image_directory, json_path in pairs:
            source_id = json_path.stem
            compact_id = source_id.replace("-", "")
            recordings.append(
                Recording(
                    dataset=self.name,
                    source_id=source_id,
                    saved_id=str(int(compact_id)),
                    frame_source=str(image_directory),
                    label_source=str(json_path),
                    subject_id=int(compact_id[:2]),
                )
            )
        return sorted(recordings, key=lambda recording: recording.source_id)

    @staticmethod
    def split(recordings: list[Recording], begin: float, end: float) -> list[Recording]:
        if begin == 0 and end == 1:
            return list(recordings)
        by_subject: dict[int, list[Recording]] = {}
        for recording in recordings:
            if recording.subject_id is None:
                raise RuntimeError(f"PURE subject is missing for {recording.source_id}")
            by_subject.setdefault(recording.subject_id, []).append(recording)
        subjects = sorted(by_subject)
        selected = subjects[int(begin * len(subjects)) : int(end * len(subjects))]
        return [recording for subject in selected for recording in by_subject[subject]]

    @staticmethod
    def read_frames(recording: Recording) -> np.ndarray:
        image_paths = sorted(Path(recording.frame_source).glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No PNG frames in {recording.frame_source}")
        frames = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"OpenCV could not read {image_path}")
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return np.asarray(frames)

    @staticmethod
    def read_label(recording: Recording) -> np.ndarray:
        with Path(recording.label_source).open("r") as handle:
            payload = json.load(handle)
        waveform = [item["Value"]["waveform"] for item in payload["/FullPackage"]]
        if not waveform:
            raise RuntimeError(f"PURE waveform is empty: {recording.label_source}")
        return np.asarray(waveform)

    @staticmethod
    def align_label(label: np.ndarray, frame_count: int) -> np.ndarray:
        return resample_ppg(label, frame_count)

    def probe(self, recording: Recording) -> dict[str, object]:
        image_paths = sorted(Path(recording.frame_source).glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No PNG frames in {recording.frame_source}")
        first = cv2.imread(str(image_paths[0]))
        if first is None:
            raise RuntimeError(f"OpenCV could not read {image_paths[0]}")
        return {
            "source_id": recording.source_id,
            "frame_count": len(image_paths),
            "first_frame_shape": tuple(first.shape),
            "label_count": int(self.read_label(recording).shape[0]),
            "frame_source": recording.frame_source,
            "label_source": recording.label_source,
        }

