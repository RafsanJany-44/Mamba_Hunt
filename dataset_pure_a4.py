"""PURE A4 loader: 50% original, otherwise one uniformly chosen A4 variant."""

from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import CachedClipDataset, find_file_list, seed_worker
from settings import BATCH_SIZE, SEED, TEST_WORKERS, TRAIN_WORKERS


A4_ROOT = Path(
    "/home/rafsan/Documents/Data/Mamba_Hunt_Data/"
    "RhythmMamba_Offline_Augmentation_A4/PURE"
)
METADATA = A4_ROOT / "PURE_A4_offline_augmentation_metadata.csv"
TRANSFORMS = ("jpeg", "blur", "gamma", "contrast")


def _identity(input_path):
    name = os.path.basename(input_path)
    split_index = name.rindex("_")
    return name[:split_index], name[split_index + 6:].split(".")[0]


class PURE_A4_TrainingDataset(Dataset):
    def __init__(self, file_list, offline_probability=0.5):
        self.offline_probability = float(offline_probability)
        with Path(file_list).open("r", newline="", encoding="utf-8") as handle:
            self.inputs = sorted(row["input_files"] for row in csv.DictReader(handle))
        if not self.inputs:
            raise RuntimeError(f"No training clips in {file_list}")
        self.labels = [path.replace("input", "label") for path in self.inputs]
        if not METADATA.is_file():
            raise FileNotFoundError(f"PURE A4 metadata is missing: {METADATA}")

        grouped = defaultdict(dict)
        with METADATA.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                original = str(Path(row["original_input"]).resolve())
                transform = row["transform"].lower()
                if transform in grouped[original]:
                    raise RuntimeError(f"Duplicate {transform} variant for {original}")
                grouped[original][transform] = row["augmented_input"]

        self.variants = []
        for path in self.inputs:
            found = grouped.get(str(Path(path).resolve()), {})
            missing = [name for name in TRANSFORMS if name not in found]
            if missing:
                raise RuntimeError(f"A4 variants missing for {path}: {missing}")
            self.variants.append(tuple(found[name] for name in TRANSFORMS))

        required = self.inputs + self.labels + [p for group in self.variants for p in group]
        missing_files = [path for path in required if not Path(path).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"PURE A4 dataset has {len(missing_files)} missing files. First: {missing_files[0]}"
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if random.random() < self.offline_probability:
            transform_index = random.randrange(len(TRANSFORMS))
            selected = self.variants[index][transform_index]
            source_kind = TRANSFORMS[transform_index]
        else:
            selected = self.inputs[index]
            source_kind = "original"
        data = np.load(selected, mmap_mode="r")
        label = np.load(self.labels[index], mmap_mode="r")
        data = np.transpose(data, (0, 3, 1, 2)).astype(np.float32)
        recording_id, chunk_id = _identity(self.inputs[index])
        return data, np.asarray(label, dtype=np.float32), recording_id, chunk_id, source_kind


def create_pure_a4_loaders(experiment):
    train_csv = find_file_list(experiment, experiment.train_begin, experiment.train_end)
    valid_csv = find_file_list(experiment, experiment.test_begin, experiment.test_end)
    training = PURE_A4_TrainingDataset(train_csv, 0.5)
    validation = CachedClipDataset(valid_csv)
    train_generator = torch.Generator().manual_seed(SEED)
    valid_generator = torch.Generator().manual_seed(SEED)
    return (
        DataLoader(training, batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=TRAIN_WORKERS, worker_init_fn=seed_worker,
                   generator=train_generator),
        DataLoader(validation, batch_size=experiment.inference_batch_size,
                   shuffle=False, num_workers=TEST_WORKERS,
                   worker_init_fn=seed_worker, generator=valid_generator),
    )


# Compatibility alias used by the shared verified A4 trainer.
create_ubfc_a4_loaders = create_pure_a4_loaders
