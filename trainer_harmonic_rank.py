"""Train one A2 RhythmMamba model with conditional harmonic ranking."""

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
from harmonic_rank_loss import (
    HALF_WIDTH_BPM,
    HARMONIC_RATIOS,
    MARGIN,
    RANK_WEIGHT,
    OfficialWithConditionalHarmonicRank,
)
from settings import DEVICE, FS, LEARNING_RATE, OUTPUT_ROOT, SEED
from trainer import build_model, normalize_prediction, set_reproducible


MAX_EPOCHS = 100
MINIMUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10


def _paths(name):
    directory = OUTPUT_ROOT / "models" / "harmonic_rank_stage3" / name
    return {
        "directory": directory,
        "best": directory / f"{name}_RhythmMamba_Best.pth",
        "history": directory / f"{name}_training_history.csv",
        "configuration": directory / f"{name}_configuration.json",
        "completion": directory / f"{name}_completion.json",
    }


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_model(model, path):
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


def _mean_components(prediction, label, criterion, epoch):
    sums = [prediction.new_tensor(0.0) for _ in range(3)]
    for index in range(prediction.shape[0]):
        values = criterion.components(prediction[index], label[index], epoch, FS, False)
        for component, value in enumerate(values):
            sums[component] = sums[component] + value
    return tuple(value / prediction.shape[0] for value in sums)


@torch.no_grad()
def _validate(model, loader, criterion, epoch):
    model.eval()
    sums = [0.0, 0.0, 0.0]
    count = 0
    progress = tqdm(loader, desc=f"Validation epoch {epoch}", ncols=96)
    for batch in progress:
        data = batch[0].float().to(DEVICE)
        label = batch[1].float().to(DEVICE)
        prediction = normalize_prediction(model(data))
        values = _mean_components(prediction, label, criterion, epoch)
        if not all(torch.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
        size = data.shape[0]
        for component, value in enumerate(values):
            sums[component] += float(value.item()) * size
        count += size
        progress.set_postfix(total=values[0].item())
    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return tuple(value / count for value in sums)


def train_source(source_name):
    source_name = source_name.upper()
    if source_name == "PURE":
        source = PURE_CROSS_MATCHED
    elif source_name == "UBFC":
        source = UBFC_CROSS_MATCHED
    else:
        raise ValueError("source_name must be PURE or UBFC")

    name = f"{source_name}_A2_HARMONIC_RANK"
    paths = _paths(name)
    if paths["completion"].is_file() and paths["best"].is_file():
        payload = json.loads(paths["completion"].read_text(encoding="utf-8"))
        if payload.get("status") == "PASSED":
            print(f"SKIPPING VERIFIED COMPLETED MODEL: {name}")
            print(f"Best checkpoint: {paths['best']}")
            return
    if paths["directory"].exists() and any(paths["directory"].iterdir()):
        raise RuntimeError(
            f"Incomplete prior run exists: {paths['directory']}\n"
            "Preserve and rename that directory before restarting."
        )

    set_reproducible(SEED)
    training_loader, validation_loader = create_stage1_loaders(
        source, source_name, True
    )
    model = build_model()
    criterion = OfficialWithConditionalHarmonicRank()
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
        "augmentation": "A2",
        "official_augmentation": True,
        "offline_augmentation": True,
        "offline_probability": 0.5,
        "online_rgb_gain": False,
        "loss": "official_plus_conditional_harmonic_rank",
        "rank_weight": RANK_WEIGHT,
        "rank_margin": MARGIN,
        "rank_half_width_bpm": HALF_WIDTH_BPM,
        "harmonic_ratios": list(HARMONIC_RATIOS),
        "frequency_range_bpm": [45, 149],
        "spectral_engine": "official_sinusoidal_projection",
        "training_split": [source.train_begin, source.train_end],
        "validation_split": [source.test_begin, source.test_end],
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
    _atomic_json(paths["configuration"], configuration)

    print("=" * 88)
    print(f"STAGE-3 TRAINING: {name}")
    print("=" * 88)
    print("Loss                 : official + 0.02 conditional harmonic ranking")
    print(f"Harmonic ratios      : {HARMONIC_RATIOS}")
    print(f"Margin / half-width  : {MARGIN} / {HALF_WIDTH_BPM} BPM")
    print(f"Training clips       : {len(training_loader.dataset)}")
    print(f"Clean validation     : {len(validation_loader.dataset)}")
    print(f"Maximum epochs       : {MAX_EPOCHS}")
    print(f"Minimum epochs       : {MINIMUM_EPOCHS}")
    print(f"Early-stop patience  : {EARLY_STOPPING_PATIENCE}")

    best_loss = float("inf")
    best_epoch = None
    stale = 0
    rows = []
    started = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        model.train()
        sums = [0.0, 0.0, 0.0]
        clip_count = original_count = offline_count = 0
        progress = tqdm(training_loader, desc=f"{name} epoch {epoch}", ncols=112)
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
            values = _mean_components(prediction, label, criterion, epoch)
            if not all(torch.isfinite(value) for value in values):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            values[0].backward()
            optimizer.step()
            scheduler.step()

            size = data.shape[0]
            for component, value in enumerate(values):
                sums[component] += float(value.item()) * size
            clip_count += size
            progress.set_postfix(total=values[0].item(), rank=values[2].item())

        train_values = tuple(value / clip_count for value in sums)
        valid_values = _validate(model, validation_loader, criterion, epoch)
        improved = valid_values[0] < best_loss
        if improved:
            best_loss = valid_values[0]
            best_epoch = epoch
            stale = 0
            _atomic_model(model, paths["best"])
        else:
            stale += 1

        rows.append({
            "epoch": epoch,
            "training_total_loss": f"{train_values[0]:.10f}",
            "training_official_loss": f"{train_values[1]:.10f}",
            "training_rank_loss": f"{train_values[2]:.10f}",
            "validation_total_loss": f"{valid_values[0]:.10f}",
            "validation_official_loss": f"{valid_values[1]:.10f}",
            "validation_rank_loss": f"{valid_values[2]:.10f}",
            "learning_rate": f"{scheduler.get_last_lr()[0]:.12g}",
            "original_selected": original_count,
            "offline_selected": offline_count,
            "is_best": improved,
            "epochs_without_improvement": stale,
        })
        _atomic_csv(paths["history"], rows)
        print(
            f"Epoch {epoch:03d} | train={train_values[0]:.8f} | "
            f"valid={valid_values[0]:.8f} | "
            f"official/rank={valid_values[1]:.6f}/{valid_values[2]:.6f} | "
            f"best={improved} | original/offline={original_count}/{offline_count}"
        )
        if epoch + 1 >= MINIMUM_EPOCHS and stale >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {EARLY_STOPPING_PATIENCE} stale epochs.")
            break

    if best_epoch is None or not paths["best"].is_file():
        raise RuntimeError("Training ended without a valid best checkpoint")
    hours = (time.perf_counter() - started) / 3600.0
    completion = {
        **configuration,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_completed": len(rows),
        "stopped_early": len(rows) < MAX_EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_loss,
        "best_checkpoint": str(paths["best"].resolve()),
        "history": str(paths["history"].resolve()),
        "training_hours": hours,
        "status": "PASSED",
    }
    _atomic_json(paths["completion"], completion)
    print("=" * 88)
    print(f"{name} TRAINING COMPLETED")
    print("=" * 88)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best valid loss : {best_loss:.8f}")
    print(f"Epochs completed: {len(rows)}")
    print(f"Best checkpoint : {paths['best']}")
    print(f"Training history: {paths['history']}")
    print(f"Training time   : {hours:.2f} hours")
