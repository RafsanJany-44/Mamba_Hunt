"""Paired five-seed UBFC A2 training for L0 versus L5."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone

import torch
from tqdm import tqdm

from augmentation import OfficialAugmentation
from cross_settings import UBFC_CROSS_MATCHED
import dataset_stage1
from loss import HybridLoss
from loss_suite_stage2 import LossSuiteCriterion
from settings import DEVICE, FS, LEARNING_RATE, OUTPUT_ROOT
from trainer import build_model, normalize_prediction, set_reproducible


SEEDS = (100, 101, 102, 103, 104)
MAX_EPOCHS = 100
MINIMUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10

VARIANTS = {
    "L0": {
        "label": "UBFC_A2_L0",
        "description": "A2 plus official 0.2 Pearson + CE loss",
    },
    "L5": {
        "label": "UBFC_A2_L5_CE_CONCENTRATION",
        "description": "A2 plus 0.2 Pearson + CE + concentration",
    },
}


def _paths(name):
    directory = OUTPUT_ROOT / "models" / "ubfc_a2_multiseed" / name
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


class Criterion:
    def __init__(self, variant):
        self.variant = variant
        self.l0 = HybridLoss() if variant == "L0" else None
        self.l5 = LossSuiteCriterion("L5_CE_CONCENTRATION") if variant == "L5" else None

    def components(self, prediction, label, epoch):
        if self.variant == "L0":
            total = self.l0(prediction, label, epoch, FS, False)
            zero = total.detach() * 0.0
            return total, total, zero, zero
        total, pearson, ce, concentration, _ = self.l5.components(
            prediction, label, epoch, FS, False
        )
        official = 0.2 * pearson + ce
        return total, official, concentration, pearson


def _mean_components(prediction, label, criterion, epoch):
    sums = [prediction.new_tensor(0.0) for _ in range(4)]
    for index in range(prediction.shape[0]):
        values = criterion.components(prediction[index], label[index], epoch)
        for component, value in enumerate(values):
            sums[component] = sums[component] + value
    return tuple(value / prediction.shape[0] for value in sums)


@torch.no_grad()
def _validate(model, loader, criterion, epoch):
    model.eval()
    sums = [0.0] * 4
    clips = 0
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
        clips += size
        progress.set_postfix(total=values[0].item())
    if clips == 0:
        raise RuntimeError("Validation loader is empty")
    return tuple(value / clips for value in sums)


def train_one(variant, seed):
    if variant not in VARIANTS:
        raise ValueError("variant must be L0 or L5")
    label = VARIANTS[variant]["label"]
    name = f"{label}_SEED{seed}"
    paths = _paths(name)

    if paths["completion"].is_file() and paths["best"].is_file():
        payload = json.loads(paths["completion"].read_text(encoding="utf-8"))
        if payload.get("status") == "PASSED" and payload.get("seed") == seed:
            print(f"SKIPPING VERIFIED COMPLETED MODEL: {name}")
            return payload
    if paths["directory"].exists() and any(paths["directory"].iterdir()):
        raise RuntimeError(
            f"Incomplete prior run exists: {paths['directory']}\n"
            "Preserve and rename that directory before restarting."
        )

    set_reproducible(seed)
    # dataset_stage1 imported SEED as a module value. Set it explicitly so this
    # run's seed controls DataLoader shuffling, worker RNG state and the random
    # original/offline selection, in addition to model initialization.
    dataset_stage1.SEED = seed
    training_loader, validation_loader = dataset_stage1.create_stage1_loaders(
        UBFC_CROSS_MATCHED, "UBFC", True
    )
    model = build_model()
    criterion = Criterion(variant)
    official_augmentation = OfficialAugmentation(fs=FS, diff_flag=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=MAX_EPOCHS,
        steps_per_epoch=len(training_loader),
    )

    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": name,
        "source_dataset": "UBFC",
        "augmentation": "A2",
        "official_augmentation": True,
        "offline_augmentation": True,
        "offline_probability": 0.5,
        "online_rgb_gain": False,
        "loss_variant": variant,
        "loss_description": VARIANTS[variant]["description"],
        "training_split": [0.0, 0.8],
        "validation_split": [0.8, 1.0],
        "training_clips_per_epoch": len(training_loader.dataset),
        "validation_clips": len(validation_loader.dataset),
        "maximum_epochs": MAX_EPOCHS,
        "minimum_epochs_before_early_stopping": MINIMUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "checkpoint_rule": "lowest_clean_validation_total_loss",
        "saved_checkpoints": "best_only",
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "paired_seed_set": list(SEEDS),
    }
    _atomic_json(paths["configuration"], configuration)

    print("=" * 92)
    print(f"UBFC A2 MULTISEED TRAINING: {name}")
    print("=" * 92)
    print(f"Loss                 : {VARIANTS[variant]['description']}")
    print(f"Seed                 : {seed}")
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
        sums = [0.0] * 4
        clips = original_count = offline_count = 0
        progress = tqdm(training_loader, desc=f"{name} epoch {epoch}", ncols=114)
        for batch in progress:
            data = batch[0].float()
            label_batch = batch[1].float()
            kinds = batch[4]
            original_count += sum(kind == "original" for kind in kinds)
            offline_count += sum(kind == "offline" for kind in kinds)
            data, label_batch = official_augmentation(
                data, label_batch, batch[2], batch[3]
            )
            data = data.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            optimizer.zero_grad()
            prediction = normalize_prediction(model(data))
            values = _mean_components(prediction, label_batch, criterion, epoch)
            if not all(torch.isfinite(value) for value in values):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            values[0].backward()
            optimizer.step()
            scheduler.step()

            size = data.shape[0]
            for component, value in enumerate(values):
                sums[component] += float(value.item()) * size
            clips += size
            progress.set_postfix(total=values[0].item())

        train_values = tuple(value / clips for value in sums)
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
            "training_official_component": f"{train_values[1]:.10f}",
            "training_concentration_loss": f"{train_values[2]:.10f}",
            "training_pearson_loss": f"{train_values[3]:.10f}",
            "validation_total_loss": f"{valid_values[0]:.10f}",
            "validation_official_component": f"{valid_values[1]:.10f}",
            "validation_concentration_loss": f"{valid_values[2]:.10f}",
            "validation_pearson_loss": f"{valid_values[3]:.10f}",
            "learning_rate": f"{scheduler.get_last_lr()[0]:.12g}",
            "original_selected": original_count,
            "offline_selected": offline_count,
            "is_best": improved,
            "epochs_without_improvement": stale,
        })
        _atomic_csv(paths["history"], rows)
        print(
            f"Epoch {epoch:03d} | train={train_values[0]:.8f} | "
            f"valid={valid_values[0]:.8f} | best={improved} | "
            f"original/offline={original_count}/{offline_count}"
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
    print("=" * 92)
    print(f"{name} TRAINING COMPLETED")
    print("=" * 92)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best valid loss : {best_loss:.8f}")
    print(f"Epochs completed: {len(rows)}")
    print(f"Best checkpoint : {paths['best']}")
    print(f"Training history: {paths['history']}")
    print(f"Training time   : {hours:.2f} hours")
    return completion


def train_suite(variant):
    if variant not in VARIANTS:
        raise ValueError("variant must be L0 or L5")
    started = time.perf_counter()
    results = []
    for number, seed in enumerate(SEEDS, start=1):
        print("#" * 92)
        print(f"{VARIANTS[variant]['label']} — SEED {number}/{len(SEEDS)}: {seed}")
        print("#" * 92)
        results.append(train_one(variant, seed))

    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "label": VARIANTS[variant]["label"],
        "seeds": list(SEEDS),
        "successful_models": len(results),
        "planned_models": len(SEEDS),
        "suite_hours": (time.perf_counter() - started) / 3600.0,
        "runs": results,
        "status": "PASSED",
    }
    path = (
        OUTPUT_ROOT / "models" / "ubfc_a2_multiseed"
        / f"{VARIANTS[variant]['label']}_five_seed_summary.json"
    )
    _atomic_json(path, summary)
    print("=" * 92)
    print(f"{VARIANTS[variant]['label']} FIVE-SEED SUITE COMPLETED")
    print(f"Successful models: {len(results)}/{len(SEEDS)}")
    print(f"Summary: {path}")
    print("=" * 92)
