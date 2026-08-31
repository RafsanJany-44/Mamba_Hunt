"""Minimal official-compatible training and evaluation loop."""

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from augmentation import OfficialAugmentation
from dataset import create_loaders, create_test_loader
from loss import HybridLoss
from metrics import calculate_metrics
from model import RhythmMamba
from settings import (
    CHUNK_LENGTH,
    DEVICE,
    EPOCHS,
    FS,
    LEARNING_RATE,
    NUM_GPUS,
    SAVE_EVERY_EPOCH,
    SEED,
    USE_AUGMENTATION,
    Experiment,
)


def set_reproducible(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_prediction(prediction):
    mean = torch.mean(prediction, dim=-1).view(-1, 1)
    std = torch.std(prediction, dim=-1).view(-1, 1)
    return (prediction - mean) / std


def build_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RhythmMamba baseline.")
    model = RhythmMamba().to(DEVICE)
    return nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))


def checkpoint_path(experiment: Experiment, epoch: int) -> Path:
    return experiment.model_dir / f"{experiment.name}_RhythmMamba_Epoch{epoch}.pth"


def save_model(model, experiment: Experiment, epoch: int):
    experiment.model_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(experiment, epoch)
    torch.save(model.state_dict(), path)
    print(f"Saved Model Path: {path}")


@torch.no_grad()
def test_model(model, test_loader):
    model.eval()
    predictions = {}
    labels = {}

    for batch in tqdm(test_loader, desc="Testing", ncols=80):
        data = batch[0].to(DEVICE)
        label = batch[1].to(DEVICE)
        prediction = normalize_prediction(model(data))

        batch_size = data.shape[0]
        prediction = prediction.view(-1, 1)
        label = label.view(-1, 1)

        for item in range(batch_size):
            recording_id = batch[2][item]
            chunk_id = int(batch[3][item])
            predictions.setdefault(recording_id, {})[chunk_id] = prediction[
                item * CHUNK_LENGTH : (item + 1) * CHUNK_LENGTH
            ]
            labels.setdefault(recording_id, {})[chunk_id] = label[
                item * CHUNK_LENGTH : (item + 1) * CHUNK_LENGTH
            ]

    print()
    return calculate_metrics(predictions, labels, fs=FS)


def train_experiment(experiment: Experiment):
    set_reproducible()
    train_loader, test_loader = create_loaders(experiment)
    model = build_model()
    criterion = HybridLoss()
    augmentation = OfficialAugmentation(fs=FS, diff_flag=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print("=" * 70)
    print(f"SIMPLIFIED RHYTHMMAMBA: {experiment.name}")
    print("=" * 70)
    print(f"Training clips : {len(train_loader.dataset)}")
    print(f"Testing clips  : {len(test_loader.dataset)}")
    print(f"Epochs         : {EPOCHS}")
    print(f"Batches/epoch  : {len(train_loader)}")
    print(f"Parameters     : {parameters:,}")
    print(f"Device         : {DEVICE}")
    print(f"Augmentation   : {USE_AUGMENTATION}")

    training_start = time.perf_counter()
    for epoch in range(EPOCHS):
        print(f"\n====Training Epoch: {epoch}====")
        model.train()
        progress = tqdm(train_loader, desc=f"Train epoch {epoch}", ncols=80)

        for batch in progress:
            data = batch[0].float()
            label = batch[1].float()
            batch_size = data.shape[0]

            if USE_AUGMENTATION:
                data, label = augmentation(data, label, batch[2], batch[3])

            data = data.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            prediction = normalize_prediction(model(data))

            loss = prediction.new_tensor(0.0)
            for item in range(batch_size):
                loss = loss + criterion(
                    prediction[item], label[item], epoch, FS, False
                )
            loss = loss / batch_size
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress.set_postfix(loss=loss.item())

        if SAVE_EVERY_EPOCH or epoch == EPOCHS - 1:
            save_model(model, experiment, epoch)

    elapsed = time.perf_counter() - training_start
    print(f"\nTraining completed in {elapsed / 3600:.2f} hours")

    final_path = checkpoint_path(experiment, EPOCHS - 1)
    model.load_state_dict(torch.load(final_path, map_location=DEVICE))
    print(f"Testing checkpoint: {final_path}")
    return test_model(model, test_loader)


def evaluate_experiment(experiment: Experiment, model_path=None):
    set_reproducible()
    path = Path(model_path) if model_path is not None else experiment.final_checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    test_loader = create_test_loader(experiment)
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f"Testing checkpoint: {path}")
    return test_model(model, test_loader)
