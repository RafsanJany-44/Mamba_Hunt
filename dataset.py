"""Minimal loader for the already-preprocessed RhythmMamba ``.npy`` clips."""

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from settings import BATCH_SIZE, SEED, TEST_WORKERS, TRAIN_WORKERS, Experiment


def _number(value: float) -> str:
    """Use the same split spelling as the official file-list generator."""
    return str(float(value))


def find_file_list(experiment: Experiment, begin: float, end: float) -> Path:
    """Find the one official CSV corresponding to a requested dataset split."""
    file_list_dir = experiment.cache_parent / "DataFileLists"
    suffix = f"_{_number(begin)}_{_number(end)}.csv"
    matches = sorted(file_list_dir.glob(f"*{suffix}"))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file list ending in {suffix!r} inside "
            f"{file_list_dir}, but found {len(matches)}: {matches}"
        )
    return matches[0]


class CachedClipDataset(Dataset):
    """Return ``video, label, recording_id, chunk_id`` exactly as the toolbox."""

    def __init__(self, file_list: Path):
        self.file_list = Path(file_list)
        with self.file_list.open("r", newline="") as handle:
            rows = csv.DictReader(handle)
            self.inputs = sorted(row["input_files"] for row in rows)

        if not self.inputs:
            raise RuntimeError(f"No cached clips were listed in {self.file_list}")

        self.labels = [path.replace("input", "label") for path in self.inputs]
        missing = [path for path in self.inputs + self.labels if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"The file list contains {len(missing)} missing cached files. "
                f"First missing file: {missing[0]}"
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        data = np.load(self.inputs[index], mmap_mode="r")
        label = np.load(self.labels[index], mmap_mode="r")

        # Official cache is [T,H,W,C]; RhythmMamba expects [T,C,H,W].
        data = np.transpose(data, (0, 3, 1, 2)).astype(np.float32)
        label = np.asarray(label, dtype=np.float32)

        item_name = os.path.basename(self.inputs[index])
        split_index = item_name.rindex("_")
        recording_id = item_name[:split_index]
        chunk_id = item_name[split_index + 6 :].split(".")[0]
        return data, label, recording_id, chunk_id


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loaders(experiment: Experiment):
    train_csv = find_file_list(
        experiment, experiment.train_begin, experiment.train_end
    )
    test_csv = find_file_list(
        experiment, experiment.test_begin, experiment.test_end
    )

    train_dataset = CachedClipDataset(train_csv)
    test_dataset = CachedClipDataset(test_csv)

    train_generator = torch.Generator().manual_seed(SEED)
    test_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_WORKERS,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=experiment.inference_batch_size,
        shuffle=False,
        num_workers=TEST_WORKERS,
        worker_init_fn=seed_worker,
        generator=test_generator,
    )
    return train_loader, test_loader


def create_test_loader(experiment: Experiment):
    test_csv = find_file_list(
        experiment, experiment.test_begin, experiment.test_end
    )
    test_dataset = CachedClipDataset(test_csv)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        test_dataset,
        batch_size=experiment.inference_batch_size,
        shuffle=False,
        num_workers=TEST_WORKERS,
        worker_init_fn=seed_worker,
        generator=generator,
    )
