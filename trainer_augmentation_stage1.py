"""Train A1/A2/A3 while preserving the verified RhythmMamba baseline."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from augmentation import OfficialAugmentation
from augmentation_stage1 import OnlineRGBChannelGain
from cross_trainer import mean_batch_loss, validation_loss
from dataset_stage1 import create_stage1_loaders
from loss import HybridLoss
from settings import DEVICE, FS, LEARNING_RATE, OUTPUT_ROOT, SEED, Experiment
from trainer import build_model, normalize_prediction, set_reproducible


MAX_EPOCHS = 60
MINIMUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
MINIMUM_IMPROVEMENT = 0.0
ONLINE_RGB_PROBABILITY = 0.5
ONLINE_RGB_LOW = 0.85
ONLINE_RGB_HIGH = 1.15

CONFIGURATIONS = {
    "A1": {"use_offline": False, "use_online_rgb": True},
    "A2": {"use_offline": True, "use_online_rgb": False},
    "A3": {"use_offline": True, "use_online_rgb": True},
}


def _experiment(source: Experiment, source_name: str, setup: str) -> Experiment:
    return replace(source, name=f"{source_name}_{setup}")


def _model_dir(experiment: Experiment) -> Path:
    return OUTPUT_ROOT / "models" / "augmentation_stage1" / experiment.name


def _best_path(experiment: Experiment) -> Path:
    return _model_dir(experiment) / f"{experiment.name}_RhythmMamba_Best.pth"


def _history_path(experiment: Experiment) -> Path:
    return _model_dir(experiment) / f"{experiment.name}_training_history.csv"


def _configuration_path(experiment: Experiment) -> Path:
    return _model_dir(experiment) / f"{experiment.name}_configuration.json"


def _completion_path(experiment: Experiment) -> Path:
    return _model_dir(experiment) / f"{experiment.name}_completion.json"


def _atomic_torch_save(model, path: Path) -> None:
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def train_one(source: Experiment, source_name: str, setup: str) -> Path:
    if setup not in CONFIGURATIONS:
        raise ValueError(f"Unknown setup {setup}; expected {sorted(CONFIGURATIONS)}")
    options = CONFIGURATIONS[setup]
    experiment = _experiment(source, source_name, setup)
    completion = _completion_path(experiment)
    best_path = _best_path(experiment)
    if completion.is_file() and best_path.is_file():
        print(f"SKIPPING COMPLETED SETUP: {experiment.name}")
        print(f"Best checkpoint: {best_path}")
        return best_path
    if _model_dir(experiment).exists() and any(_model_dir(experiment).iterdir()):
        raise RuntimeError(
            f"Incomplete prior run exists in {_model_dir(experiment)}. "
            "Preserve it for diagnosis and move/rename that directory before rerunning."
        )

    set_reproducible(SEED)
    train_loader, validation_loader = create_stage1_loaders(
        source, source_name, bool(options["use_offline"])
    )
    model = build_model()
    criterion = HybridLoss()
    official_augmentation = OfficialAugmentation(fs=FS, diff_flag=False)
    online_rgb = OnlineRGBChannelGain(
        ONLINE_RGB_PROBABILITY, ONLINE_RGB_LOW, ONLINE_RGB_HIGH
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=MAX_EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment.name,
        "source_dataset": source_name,
        "setup": setup,
        "official_augmentation": True,
        "offline_augmentation": bool(options["use_offline"]),
        "offline_probability": 0.5 if options["use_offline"] else 0.0,
        "online_rgb_gain": bool(options["use_online_rgb"]),
        "online_rgb_probability": (
            ONLINE_RGB_PROBABILITY if options["use_online_rgb"] else 0.0
        ),
        "online_rgb_range": [ONLINE_RGB_LOW, ONLINE_RGB_HIGH],
        "augmentation_order": "offline_selection -> official_temporal_flip -> online_rgb",
        "training_split": [source.train_begin, source.train_end],
        "validation_split": [source.test_begin, source.test_end],
        "validation_augmentation": False,
        "training_clips_per_epoch": len(train_loader.dataset),
        "validation_clips": len(validation_loader.dataset),
        "max_epochs": MAX_EPOCHS,
        "minimum_epochs_before_early_stopping": MINIMUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "checkpoint_rule": "lowest_clean_validation_loss",
        "saved_checkpoints": "best_only",
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
    }
    _write_json(_configuration_path(experiment), configuration)

    print("=" * 78)
    print(f"RHYTHMMAMBA AUGMENTATION STAGE 1: {experiment.name}")
    print("=" * 78)
    print(f"Training clips       : {len(train_loader.dataset)}")
    print(f"Clean validation     : {len(validation_loader.dataset)}")
    print(f"Official augmentation: enabled")
    print(f"Offline augmentation : {options['use_offline']} (p=0.5)")
    print(f"Online RGB gain      : {options['use_online_rgb']} (p=0.5)")
    print(f"Maximum epochs       : {MAX_EPOCHS}")
    print(f"Minimum epochs       : {MINIMUM_EPOCHS}")
    print(f"Early-stop patience  : {EARLY_STOPPING_PATIENCE}")
    print(f"Checkpoint rule      : lowest clean validation loss")

    best_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        model.train()
        loss_sum = 0.0
        clip_count = 0
        original_count = 0
        offline_count = 0
        rgb_count = 0
        progress = tqdm(train_loader, desc=f"{experiment.name} epoch {epoch}", ncols=96)

        for batch in progress:
            data = batch[0].float()
            label = batch[1].float()
            kinds = batch[4]
            offline_count += sum(kind == "offline" for kind in kinds)
            original_count += sum(kind == "original" for kind in kinds)

            # The verified official augmentation remains enabled in A1/A2/A3.
            data, label = official_augmentation(data, label, batch[2], batch[3])
            if options["use_online_rgb"]:
                data, applied = online_rgb(data)
                rgb_count += applied

            data = data.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            prediction = normalize_prediction(model(data))
            loss = mean_batch_loss(prediction, label, criterion, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = data.shape[0]
            loss_sum += float(loss.item()) * batch_size
            clip_count += batch_size
            progress.set_postfix(loss=loss.item())

        training_loss = loss_sum / clip_count
        clean_validation_loss = validation_loss(
            model, validation_loader, criterion, epoch
        )
        improved = clean_validation_loss < best_loss - MINIMUM_IMPROVEMENT
        if improved:
            best_loss = clean_validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            _atomic_torch_save(model, best_path)
        else:
            epochs_without_improvement += 1

        rows.append(
            {
                "epoch": epoch,
                "training_loss": f"{training_loss:.10f}",
                "clean_validation_loss": f"{clean_validation_loss:.10f}",
                "learning_rate": f"{scheduler.get_last_lr()[0]:.12g}",
                "original_selected": original_count,
                "offline_selected": offline_count,
                "online_rgb_applied": rgb_count,
                "is_best": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        _write_csv(_history_path(experiment), rows)
        print(
            f"Epoch {epoch:02d} | train={training_loss:.8f} | "
            f"clean_valid={clean_validation_loss:.8f} | best={improved} | "
            f"original/offline={original_count}/{offline_count} | rgb={rgb_count}"
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

    if best_epoch is None or not best_path.is_file():
        raise RuntimeError("Training ended without a valid best checkpoint")
    elapsed_hours = (time.perf_counter() - started) / 3600.0
    completion_payload = {
        **configuration,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_completed": len(rows),
        "stopped_early": len(rows) < MAX_EPOCHS,
        "best_epoch": best_epoch,
        "best_clean_validation_loss": best_loss,
        "best_checkpoint": str(best_path.resolve()),
        "history": str(_history_path(experiment).resolve()),
        "training_hours": elapsed_hours,
        "status": "PASSED",
    }
    _write_json(completion, completion_payload)

    print("=" * 78)
    print(f"{experiment.name} TRAINING COMPLETED")
    print("=" * 78)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best valid loss : {best_loss:.8f}")
    print(f"Epochs completed: {len(rows)}")
    print(f"Best checkpoint : {best_path}")
    print(f"Training history: {_history_path(experiment)}")
    print(f"Training time   : {elapsed_hours:.2f} hours")
    return best_path


def train_source(source: Experiment, source_name: str) -> None:
    paths = []
    for setup in ("A1", "A2", "A3"):
        paths.append(train_one(source, source_name, setup))
    print("=" * 78)
    print(f"{source_name} STAGE-1 ABLATION COMPLETED")
    print("=" * 78)
    for setup, path in zip(("A1", "A2", "A3"), paths):
        print(f"{setup}: {path}")
