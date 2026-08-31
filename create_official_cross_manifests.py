"""Create official 80/20 PURE and UBFC manifests from existing full caches.

No videos or labels are preprocessed again.  Only two CSV file lists per
dataset are created: 0.0-0.8 for training and 0.8-1.0 for validation.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from dataset import find_file_list
from settings import PURE, UBFC


SPLIT_POINT = 0.8
UBFC_ID = re.compile(r"^subject(\d+)$")


def read_manifest(path: Path) -> list[str]:
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    files = sorted(row["input_files"] for row in rows)
    if not files:
        raise RuntimeError(f"Empty full manifest: {path}")
    missing = [path for path in files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Full manifest contains {len(missing)} missing files. "
            f"First missing file: {missing[0]}"
        )
    return files


def recording_id(input_file: str) -> str:
    stem = Path(input_file).stem
    if "_input" not in stem:
        raise ValueError(f"Cannot determine recording ID from {input_file}")
    return stem.rsplit("_input", 1)[0]


def pure_subject(recording: str) -> int:
    if "-" in recording:
        return int(recording.split("-", 1)[0])
    if not recording.isdigit():
        raise ValueError(f"Unexpected PURE recording ID: {recording}")
    return int(recording.zfill(4)[:2])


def split_pure(files: list[str]) -> tuple[list[str], list[str]]:
    subjects = sorted({pure_subject(recording_id(path)) for path in files})
    boundary = int(SPLIT_POINT * len(subjects))
    train_subjects = set(subjects[:boundary])
    train = [
        path
        for path in files
        if pure_subject(recording_id(path)) in train_subjects
    ]
    validation = [
        path
        for path in files
        if pure_subject(recording_id(path)) not in train_subjects
    ]
    return train, validation


def split_ubfc(files: list[str]) -> tuple[list[str], list[str]]:
    recordings = sorted({recording_id(path) for path in files})
    for recording in recordings:
        if UBFC_ID.fullmatch(recording) is None:
            raise ValueError(f"Unexpected UBFC recording ID: {recording}")
    boundary = int(SPLIT_POINT * len(recordings))
    train_recordings = set(recordings[:boundary])
    train = [path for path in files if recording_id(path) in train_recordings]
    validation = [
        path for path in files if recording_id(path) not in train_recordings
    ]
    return train, validation


def write_manifest(path: Path, files: list[str]) -> None:
    if not files:
        raise RuntimeError(f"Refusing to create empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_files"])
        writer.writeheader()
        for input_file in sorted(files):
            writer.writerow({"input_files": input_file})
    os.replace(temporary, path)


def output_paths(full_manifest: Path) -> tuple[Path, Path]:
    suffix = "_0.0_1.0.csv"
    if not full_manifest.name.endswith(suffix):
        raise RuntimeError(f"Unexpected full-manifest name: {full_manifest}")
    prefix = full_manifest.name[: -len(suffix)]
    directory = full_manifest.parent
    return (
        directory / f"{prefix}_0.0_0.8.csv",
        directory / f"{prefix}_0.8_1.0.csv",
    )


def verify_partition(full: list[str], train: list[str], validation: list[str]) -> None:
    full_set = set(full)
    train_set = set(train)
    validation_set = set(validation)
    if train_set & validation_set:
        raise RuntimeError("Training and validation manifests overlap.")
    if train_set | validation_set != full_set:
        raise RuntimeError("80/20 manifests do not reproduce the full manifest.")


def create_for_dataset(name, experiment, splitter) -> None:
    full_manifest = find_file_list(experiment, 0.0, 1.0)
    full_files = read_manifest(full_manifest)
    train_files, validation_files = splitter(full_files)
    verify_partition(full_files, train_files, validation_files)

    train_path, validation_path = output_paths(full_manifest)
    write_manifest(train_path, train_files)
    write_manifest(validation_path, validation_files)

    train_recordings = {recording_id(path) for path in train_files}
    validation_recordings = {recording_id(path) for path in validation_files}

    expected = {
        "PURE": (47, 596, 12, 154, 750),
        "UBFC": (33, 378, 9, 105, 483),
    }[name]
    observed = (
        len(train_recordings),
        len(train_files),
        len(validation_recordings),
        len(validation_files),
        len(full_files),
    )
    if observed != expected:
        raise RuntimeError(
            f"{name} 80/20 split differs from the verified dataset. "
            f"Expected {expected}, observed {observed}."
        )

    print("=" * 78)
    print(f"{name} OFFICIAL CROSS-TRAINING MANIFESTS")
    print("=" * 78)
    print(f"Training recordings   : {len(train_recordings)}")
    print(f"Training clips        : {len(train_files)}")
    print(f"Validation recordings : {len(validation_recordings)}")
    print(f"Validation clips      : {len(validation_files)}")
    print(f"Full clips            : {len(full_files)}")
    print(f"Training manifest     : {train_path}")
    print(f"Validation manifest   : {validation_path}")
    print("Partition validation  : PASSED")


def main() -> None:
    create_for_dataset("PURE", PURE, split_pure)
    create_for_dataset("UBFC", UBFC, split_ubfc)
    print("=" * 78)
    print("PURE AND UBFC OFFICIAL 80/20 MANIFEST CREATION: PASSED")


if __name__ == "__main__":
    main()
