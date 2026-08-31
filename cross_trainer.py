"""Official-matched RhythmMamba cross-training with validation selection."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from augmentation import OfficialAugmentation
from dataset import create_loaders
from loss import HybridLoss
from settings import (
    DEVICE,
    EPOCHS,
    FS,
    LEARNING_RATE,
    SAVE_EVERY_EPOCH,
    USE_AUGMENTATION,
    Experiment,
)
from trainer import build_model, normalize_prediction, set_reproducible


def epoch_checkpoint(experiment: Experiment, epoch: int) -> Path:
    return (
        experiment.model_dir
        / f"{experiment.name}_RhythmMamba_Epoch{epoch}.pth"
    )


def best_checkpoint(experiment: Experiment) -> Path:
    return experiment.model_dir / f"{experiment.name}_RhythmMamba_Best.pth"


def history_path(experiment: Experiment) -> Path:
    return experiment.model_dir / f"{experiment.name}_training_history.csv"


def write_history(experiment: Experiment, rows: list[dict]) -> None:
    path = history_path(experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "training_loss",
                "validation_loss",
                "learning_rate",
                "is_best",
                "checkpoint",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def mean_batch_loss(prediction, label, criterion, epoch: int):
    loss = prediction.new_tensor(0.0)
    for item in range(prediction.shape[0]):
        loss = loss + criterion(
            prediction[item],
            label[item],
            epoch,
            FS,
            False,
        )
    return loss / prediction.shape[0]


@torch.no_grad()
def validation_loss(model, validation_loader, criterion, epoch: int) -> float:
    model.eval()
    total_loss = 0.0
    total_clips = 0

    progress = tqdm(
        validation_loader,
        desc=f"Validation epoch {epoch}",
        ncols=88,
    )
    for batch in progress:
        data = batch[0].float().to(DEVICE)
        label = batch[1].float().to(DEVICE)
        prediction = normalize_prediction(model(data))

        batch_loss = mean_batch_loss(
            prediction,
            label,
            criterion,
            epoch,
        )
        if not torch.isfinite(batch_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}.")

        batch_size = data.shape[0]
        total_loss += float(batch_loss.item()) * batch_size
        total_clips += batch_size
        progress.set_postfix(loss=batch_loss.item())

    if total_clips == 0:
        raise RuntimeError("Validation loader is empty.")
    return total_loss / total_clips


def train_cross_experiment(experiment: Experiment) -> Path:
    set_reproducible()
    train_loader, validation_loader = create_loaders(experiment)
    model = build_model()
    criterion = HybridLoss()
    augmentation = OfficialAugmentation(fs=FS, diff_flag=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    experiment.model_dir.mkdir(parents=True, exist_ok=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    best_loss = float("inf")
    best_epoch = None
    rows = []

    print("=" * 78)
    print(f"OFFICIAL-MATCHED CROSS TRAINING: {experiment.name}")
    print("=" * 78)
    print(f"Training split    : {experiment.train_begin}-{experiment.train_end}")
    print(f"Validation split  : {experiment.test_begin}-{experiment.test_end}")
    print(f"Training clips    : {len(train_loader.dataset)}")
    print(f"Validation clips  : {len(validation_loader.dataset)}")
    print(f"Epochs            : {EPOCHS}")
    print(f"Parameters        : {parameters:,}")
    print(f"Augmentation      : {USE_AUGMENTATION}")
    print(f"Checkpoint rule   : lowest validation loss")

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        training_loss_sum = 0.0
        training_clips = 0
        progress = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}",
            ncols=88,
        )

        for batch in progress:
            data = batch[0].float()
            label = batch[1].float()

            if USE_AUGMENTATION:
                data, label = augmentation(data, label, batch[2], batch[3])

            data = data.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            prediction = normalize_prediction(model(data))
            loss = mean_batch_loss(prediction, label, criterion, epoch)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")

            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = data.shape[0]
            training_loss_sum += float(loss.item()) * batch_size
            training_clips += batch_size
            progress.set_postfix(loss=loss.item())

        mean_training_loss = training_loss_sum / training_clips
        mean_validation_loss = validation_loss(
            model,
            validation_loader,
            criterion,
            epoch,
        )

        checkpoint = epoch_checkpoint(experiment, epoch)
        if SAVE_EVERY_EPOCH or epoch == EPOCHS - 1:
            torch.save(model.state_dict(), checkpoint)

        is_best = mean_validation_loss < best_loss
        if is_best:
            best_loss = mean_validation_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint(experiment))

        rows.append(
            {
                "epoch": epoch,
                "training_loss": mean_training_loss,
                "validation_loss": mean_validation_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "is_best": is_best,
                "checkpoint": str(checkpoint),
            }
        )
        write_history(experiment, rows)

        print(
            f"Epoch {epoch:02d} | train={mean_training_loss:.8f} | "
            f"validation={mean_validation_loss:.8f} | best={is_best}"
        )

    elapsed_hours = (time.perf_counter() - start_time) / 3600
    print("=" * 78)
    print(f"{experiment.name} CROSS TRAINING COMPLETED")
    print("=" * 78)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best valid loss : {best_loss:.8f}")
    print(f"Best checkpoint : {best_checkpoint(experiment)}")
    print(f"Training history: {history_path(experiment)}")
    print(f"Training time   : {elapsed_hours:.2f} hours")
    return best_checkpoint(experiment)
