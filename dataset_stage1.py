"""Training loaders for the Stage-1 augmentation ablation."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import CachedClipDataset, find_file_list, seed_worker
from settings import BATCH_SIZE, SEED, TEST_WORKERS, TRAIN_WORKERS, Experiment


OFFLINE_ROOT = Path(
    "/home/rafsan/Documents/Data/Mamba_Hunt_Data/"
    "RhythmMamba_Offline_Augmentation"
)


def _identity(input_path: str) -> tuple[str, str]:
    item_name = os.path.basename(input_path)
    split_index = item_name.rindex("_")
    recording_id = item_name[:split_index]
    chunk_id = item_name[split_index + 6 :].split(".")[0]
    return recording_id, chunk_id


class Stage1TrainingDataset(Dataset):
    """Return one row per original clip, optionally sampling its offline pair."""

    def __init__(
        self,
        file_list: Path,
        dataset_name: str,
        use_offline: bool,
        offline_probability: float = 0.5,
    ):
        self.file_list = Path(file_list)
        self.dataset_name = dataset_name.upper()
        self.use_offline = bool(use_offline)
        self.offline_probability = float(offline_probability)
        if not 0.0 <= self.offline_probability <= 1.0:
            raise ValueError("offline_probability must be between 0 and 1")

        with self.file_list.open("r", newline="", encoding="utf-8") as handle:
            self.inputs = sorted(row["input_files"] for row in csv.DictReader(handle))
        if not self.inputs:
            raise RuntimeError(f"No training clips in {self.file_list}")
        self.labels = [path.replace("input", "label") for path in self.inputs]

        self.augmented: list[str] = []
        if self.use_offline:
            metadata_path = (
                OFFLINE_ROOT
                / self.dataset_name
                / f"{self.dataset_name}_offline_augmentation_metadata.csv"
            )
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Offline metadata is missing: {metadata_path}")
            with metadata_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            mapping = {
                str(Path(row["original_input"]).resolve()): row["augmented_input"]
                for row in rows
            }
            if len(mapping) != len(rows):
                raise RuntimeError(f"Duplicate original paths in {metadata_path}")
            normalized_inputs = [str(Path(path).resolve()) for path in self.inputs]
            absent = [
                path
                for path, normalized in zip(self.inputs, normalized_inputs)
                if normalized not in mapping
            ]
            if absent:
                raise RuntimeError(
                    f"Offline metadata does not cover {len(absent)} training clips. "
                    f"First: {absent[0]}"
                )
            self.augmented = [mapping[path] for path in normalized_inputs]

        required = self.inputs + self.labels + self.augmented
        missing = [path for path in required if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Stage-1 dataset contains {len(missing)} missing files. First: {missing[0]}"
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        use_augmented = self.use_offline and random.random() < self.offline_probability
        selected_path = self.augmented[index] if use_augmented else self.inputs[index]
        data = np.load(selected_path, mmap_mode="r")
        label = np.load(self.labels[index], mmap_mode="r")
        data = np.transpose(data, (0, 3, 1, 2)).astype(np.float32)
        label = np.asarray(label, dtype=np.float32)
        recording_id, chunk_id = _identity(self.inputs[index])
        source_kind = "offline" if use_augmented else "original"
        return data, label, recording_id, chunk_id, source_kind


def create_stage1_loaders(
    experiment: Experiment,
    source_dataset: str,
    use_offline: bool,
):
    train_csv = find_file_list(
        experiment, experiment.train_begin, experiment.train_end
    )
    validation_csv = find_file_list(
        experiment, experiment.test_begin, experiment.test_end
    )
    training = Stage1TrainingDataset(train_csv, source_dataset, use_offline)
    # Validation is deliberately the unchanged baseline dataset.
    validation = CachedClipDataset(validation_csv)

    train_generator = torch.Generator().manual_seed(SEED)
    validation_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        training,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_WORKERS,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=experiment.inference_batch_size,
        shuffle=False,
        num_workers=TEST_WORKERS,
        worker_init_fn=seed_worker,
        generator=validation_generator,
    )
    return train_loader, validation_loader
