"""Train A0/A2 RhythmMamba models with the harmonic-aware loss."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from augmentation import OfficialAugmentation
from cross_settings import PURE_CROSS_MATCHED, UBFC_CROSS_MATCHED
from dataset_stage1 import create_stage1_loaders
from loss_harmonic import (
    HARMONIC_BAND_HALF_WIDTH_BPM,
    HARMONIC_RATIOS,
    HARMONIC_WEIGHT,
    HarmonicAwareHybridLoss,
)
from settings import DEVICE, FS, LEARNING_RATE, OUTPUT_ROOT, SEED, Experiment
from trainer import build_model, normalize_prediction, set_reproducible


MAX_EPOCHS = 100
MINIMUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
MINIMUM_IMPROVEMENT = 0.0


def model_directory(name: str) -> Path:
    return OUTPUT_ROOT / "models" / "harmonic_loss_stage1" / name


def best_checkpoint(name: str) -> Path:
    return model_directory(name) / f"{name}_RhythmMamba_Best.pth"


def history_path(name: str) -> Path:
    return model_directory(name) / f"{name}_training_history.csv"


def configuration_path(name: str) -> Path:
    return model_directory(name) / f"{name}_configuration.json"


def completion_path(name: str) -> Path:
    return model_directory(name) / f"{name}_completion.json"


def atomic_model_save(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".pth.tmp", delete=False
        ) as handle:
            temporary = handle.name
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mean_loss_components(prediction, label, criterion, epoch: int):
    total = prediction.new_tensor(0.0)
    base = prediction.new_tensor(0.0)
    harmonic = prediction.new_tensor(0.0)
    for item in range(prediction.shape[0]):
        item_total, item_base, item_harmonic = criterion.components(
            prediction[item], label[item], epoch, FS, False
        )
        total = total + item_total
        base = base + item_base
        harmonic = harmonic + item_harmonic
    count = prediction.shape[0]
    return total / count, base / count, harmonic / count


@torch.no_grad()
def validate(model, loader, criterion, epoch: int) -> tuple[float, float, float]:
    model.eval()
    sums = [0.0, 0.0, 0.0]
    clips = 0
    progress = tqdm(loader, desc=f"Validation epoch {epoch}", ncols=96)
    for batch in progress:
        data = batch[0].float().to(DEVICE)
        label = batch[1].float().to(DEVICE)
        prediction = normalize_prediction(model(data))
        values = mean_loss_components(prediction, label, criterion, epoch)
        if not all(torch.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
        count = data.shape[0]
        for index, value in enumerate(values):
            sums[index] += float(value.item()) * count
        clips += count
        progress.set_postfix(total=values[0].item())
    if clips == 0:
        raise RuntimeError("Validation loader is empty")
    return tuple(value / clips for value in sums)


def train_one(
    source: Experiment,
    source_name: str,
    augmentation_name: str,
) -> Path:
    if augmentation_name not in ("A0", "A2"):
        raise ValueError("augmentation_name must be A0 or A2")
    use_offline = augmentation_name == "A2"
    name = f"{source_name}_{augmentation_name}_HARMONIC"
    directory = model_directory(name)
    best = best_checkpoint(name)
    completion = completion_path(name)
    if completion.is_file() and best.is_file():
        print(f"SKIPPING COMPLETED MODEL: {name}")
        print(f"Best checkpoint: {best}")
        return best
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            f"Incomplete prior run exists in {directory}. Preserve it for "
            "diagnosis and move/rename the directory before rerunning."
        )

    set_reproducible(SEED)
    training_loader, validation_loader = create_stage1_loaders(
        source, source_name, use_offline
    )
    model = build_model()
    criterion = HarmonicAwareHybridLoss(HARMONIC_WEIGHT)
    official_augmentation = OfficialAugmentation(fs=FS, diff_flag=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=MAX_EPOCHS,
        steps_per_epoch=len(training_loader),
    )

    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": name,
        "source_dataset": source_name,
        "augmentation": augmentation_name,
        "official_augmentation": True,
        "offline_augmentation": use_offline,
        "offline_probability": 0.5 if use_offline else 0.0,
        "online_rgb_gain": False,
        "loss": "verified_HybridLoss_plus_harmonic_competition",
        "harmonic_weight": HARMONIC_WEIGHT,
        "harmonic_ratios": list(HARMONIC_RATIOS),
        "harmonic_band_half_width_bpm": HARMONIC_BAND_HALF_WIDTH_BPM,
        "training_split": [source.train_begin, source.train_end],
        "validation_split": [source.test_begin, source.test_end],
        "validation_augmentation": False,
        "training_clips_per_epoch": len(training_loader.dataset),
        "validation_clips": len(validation_loader.dataset),
        "maximum_epochs": MAX_EPOCHS,
        "minimum_epochs_before_early_stopping": MINIMUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "checkpoint_rule": "lowest_clean_validation_total_loss",
        "saved_checkpoints": "best_only",
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
    }
    atomic_json(configuration_path(name), configuration)

    print("=" * 78)
    print(f"HARMONIC-AWARE TRAINING: {name}")
    print("=" * 78)
    print(f"Training clips      : {len(training_loader.dataset)}")
    print(f"Clean validation    : {len(validation_loader.dataset)}")
    print(f"Augmentation        : {augmentation_name}")
    print(f"Harmonic weight     : {HARMONIC_WEIGHT}")
    print(f"Maximum epochs      : {MAX_EPOCHS}")
    print(f"Minimum epochs      : {MINIMUM_EPOCHS}")
    print(f"Early-stop patience : {EARLY_STOPPING_PATIENCE}")

    best_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        model.train()
        sums = [0.0, 0.0, 0.0]
        clip_count = 0
        original_count = 0
        offline_count = 0
        progress = tqdm(training_loader, desc=f"{name} epoch {epoch}", ncols=100)
        for batch in progress:
            data = batch[0].float()
            label = batch[1].float()
            kinds = batch[4]
            original_count += sum(kind == "original" for kind in kinds)
            offline_count += sum(kind == "offline" for kind in kinds)
            data, label = official_augmentation(data, label, batch[2], batch[3])
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            prediction = normalize_prediction(model(data))
            values = mean_loss_components(prediction, label, criterion, epoch)
            if not all(torch.isfinite(value) for value in values):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            values[0].backward()
            optimizer.step()
            scheduler.step()

            count = data.shape[0]
            for index, value in enumerate(values):
                sums[index] += float(value.item()) * count
            clip_count += count
            progress.set_postfix(total=values[0].item())

        train_values = tuple(value / clip_count for value in sums)
        valid_values = validate(model, validation_loader, criterion, epoch)
        improved = valid_values[0] < best_loss - MINIMUM_IMPROVEMENT
        if improved:
            best_loss = valid_values[0]
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_model_save(model, best)
        else:
            epochs_without_improvement += 1

        rows.append(
            {
                "epoch": epoch,
                "training_total_loss": f"{train_values[0]:.10f}",
                "training_base_loss": f"{train_values[1]:.10f}",
                "training_harmonic_loss": f"{train_values[2]:.10f}",
                "validation_total_loss": f"{valid_values[0]:.10f}",
                "validation_base_loss": f"{valid_values[1]:.10f}",
                "validation_harmonic_loss": f"{valid_values[2]:.10f}",
                "learning_rate": f"{scheduler.get_last_lr()[0]:.12g}",
                "original_selected": original_count,
                "offline_selected": offline_count,
                "is_best": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        atomic_csv(history_path(name), rows)
        print(
            f"Epoch {epoch:03d} | train={train_values[0]:.8f} "
            f"(base={train_values[1]:.8f}, harmonic={train_values[2]:.8f}) | "
            f"valid={valid_values[0]:.8f} "
            f"(base={valid_values[1]:.8f}, harmonic={valid_values[2]:.8f}) | "
            f"best={improved} | original/offline="
            f"{original_count}/{offline_count}"
        )
        if (
            epoch + 1 >= MINIMUM_EPOCHS
            and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"Early stopping: no validation improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    if best_epoch is None or not best.is_file():
        raise RuntimeError("Training ended without a valid best checkpoint")
    hours = (time.perf_counter() - started) / 3600.0
    payload = {
        **configuration,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_completed": len(rows),
        "stopped_early": len(rows) < MAX_EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_loss,
        "best_checkpoint": str(best.resolve()),
        "history": str(history_path(name).resolve()),
        "training_hours": hours,
        "status": "PASSED",
    }
    atomic_json(completion, payload)
    print("=" * 78)
    print(f"{name} TRAINING COMPLETED")
    print("=" * 78)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best valid loss : {best_loss:.8f}")
    print(f"Epochs completed: {len(rows)}")
    print(f"Best checkpoint : {best}")
    print(f"Training history: {history_path(name)}")
    print(f"Training time   : {hours:.2f} hours")
    return best


def train_source(source_name: str) -> None:
    source_name = source_name.upper()
    if source_name == "PURE":
        source = PURE_CROSS_MATCHED
    elif source_name == "UBFC":
        source = UBFC_CROSS_MATCHED
    else:
        raise ValueError("source_name must be PURE or UBFC")
    paths = []
    for augmentation_name in ("A0", "A2"):
        paths.append(train_one(source, source_name, augmentation_name))
    print("=" * 78)
    print(f"{source_name} HARMONIC-LOSS TRAINING COMPLETED")
    print("=" * 78)
    for augmentation_name, path in zip(("A0", "A2"), paths):
        print(f"{augmentation_name}: {path}")

