"""Train all new Stage-2 loss variants for one RhythmMamba source dataset."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from augmentation import OfficialAugmentation
from cross_settings import PURE_CROSS_MATCHED, UBFC_CROSS_MATCHED
from dataset_stage1 import create_stage1_loaders
from loss_suite_stage2 import (
    CONCENTRATION_HALF_WIDTH_BPM,
    HARMONIC_BAND_HALF_WIDTH_BPM,
    HARMONIC_RATIOS,
    LOSS_VARIANTS,
    LossSuiteCriterion,
)
from settings import DEVICE, FS, LEARNING_RATE, OUTPUT_ROOT, SEED, Experiment
from trainer import build_model, normalize_prediction, set_reproducible


MAX_EPOCHS = 100
MINIMUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
MINIMUM_IMPROVEMENT = 0.0
VARIANT_ORDER = (
    "L2_HARMONIC_REPLACE",
    "L3_CONCENTRATION",
    "L4_CONCENTRATION_HARMONIC",
    "L5_CE_CONCENTRATION",
)
AUGMENTATION_ORDER = ("A0", "A2")


def model_directory(name: str) -> Path:
    return OUTPUT_ROOT / "models" / "loss_suite_stage2" / name


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
    sums = [prediction.new_tensor(0.0) for _ in range(5)]
    for item in range(prediction.shape[0]):
        values = criterion.components(
            prediction[item], label[item], epoch, FS, False
        )
        for index, value in enumerate(values):
            sums[index] = sums[index] + value
    count = prediction.shape[0]
    return tuple(value / count for value in sums)


@torch.no_grad()
def validate(model, loader, criterion, epoch: int) -> tuple[float, ...]:
    model.eval()
    sums = [0.0] * 5
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
    variant_code: str,
) -> Path:
    if augmentation_name not in AUGMENTATION_ORDER:
        raise ValueError("augmentation_name must be A0 or A2")
    if variant_code not in LOSS_VARIANTS:
        raise ValueError(f"Unknown loss variant: {variant_code}")

    use_offline = augmentation_name == "A2"
    variant = LOSS_VARIANTS[variant_code]
    name = f"{source_name}_{augmentation_name}_{variant_code}"
    directory = model_directory(name)
    best = best_checkpoint(name)
    completion = completion_path(name)
    if completion.is_file() and best.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("status") != "PASSED":
            raise RuntimeError(f"Invalid completion marker: {completion}")
        print(f"SKIPPING VERIFIED COMPLETED MODEL: {name}")
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
    criterion = LossSuiteCriterion(variant_code)
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
        "loss_variant": variant.code,
        "loss_description": variant.description,
        "pearson_weight": variant.pearson_weight,
        "ce_weight": variant.ce_weight,
        "concentration_weight": variant.concentration_weight,
        "harmonic_weight": variant.harmonic_weight,
        "concentration_half_width_bpm": CONCENTRATION_HALF_WIDTH_BPM,
        "harmonic_ratios": list(HARMONIC_RATIOS),
        "harmonic_band_half_width_bpm": HARMONIC_BAND_HALF_WIDTH_BPM,
        "frequency_range_bpm": [45, 149],
        "spectral_engine": "official_sinusoidal_projection",
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

    print("=" * 88)
    print(f"LOSS-SUITE TRAINING: {name}")
    print("=" * 88)
    print(f"Loss                 : {variant.description}")
    print(f"Training clips       : {len(training_loader.dataset)}")
    print(f"Clean validation     : {len(validation_loader.dataset)}")
    print(f"Augmentation         : {augmentation_name}")
    print(f"Maximum epochs       : {MAX_EPOCHS}")
    print(f"Minimum epochs       : {MINIMUM_EPOCHS}")
    print(f"Early-stop patience  : {EARLY_STOPPING_PATIENCE}")

    best_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        model.train()
        sums = [0.0] * 5
        clip_count = 0
        original_count = 0
        offline_count = 0
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
                "training_pearson_loss": f"{train_values[1]:.10f}",
                "training_ce_loss": f"{train_values[2]:.10f}",
                "training_concentration_loss": f"{train_values[3]:.10f}",
                "training_harmonic_loss": f"{train_values[4]:.10f}",
                "validation_total_loss": f"{valid_values[0]:.10f}",
                "validation_pearson_loss": f"{valid_values[1]:.10f}",
                "validation_ce_loss": f"{valid_values[2]:.10f}",
                "validation_concentration_loss": f"{valid_values[3]:.10f}",
                "validation_harmonic_loss": f"{valid_values[4]:.10f}",
                "learning_rate": f"{scheduler.get_last_lr()[0]:.12g}",
                "original_selected": original_count,
                "offline_selected": offline_count,
                "is_best": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        atomic_csv(history_path(name), rows)
        print(
            f"Epoch {epoch:03d} | train={train_values[0]:.8f} | "
            f"valid={valid_values[0]:.8f} | "
            f"P/CE/C/H={valid_values[1]:.5f}/{valid_values[2]:.5f}/"
            f"{valid_values[3]:.5f}/{valid_values[4]:.5f} | "
            f"best={improved} | original/offline={original_count}/{offline_count}"
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
    print("=" * 88)
    print(f"{name} TRAINING COMPLETED")
    print("=" * 88)
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

    jobs = [
        (variant_code, augmentation_name)
        for variant_code in VARIANT_ORDER
        for augmentation_name in AUGMENTATION_ORDER
    ]
    successes: list[tuple[str, Path]] = []
    failures: list[dict[str, str]] = []
    suite_started = time.perf_counter()

    for job_number, (variant_code, augmentation_name) in enumerate(jobs, start=1):
        name = f"{source_name}_{augmentation_name}_{variant_code}"
        print("#" * 88)
        print(f"{source_name} LOSS SUITE — MODEL {job_number}/{len(jobs)}: {name}")
        print("#" * 88)
        try:
            path = train_one(source, source_name, augmentation_name, variant_code)
            successes.append((name, path))
        except Exception as error:
            failures.append(
                {
                    "experiment": name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print("!" * 88)
            print(f"FAILED MODEL: {name}")
            print(traceback.format_exc())
            print("The suite will continue to the next model.")
            print("!" * 88)

    suite_directory = OUTPUT_ROOT / "models" / "loss_suite_stage2"
    suite_directory.mkdir(parents=True, exist_ok=True)
    suite_summary = {
        "source_dataset": source_name,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "planned_models": len(jobs),
        "successful_models": len(successes),
        "failed_models": len(failures),
        "successes": [{"experiment": n, "checkpoint": str(p)} for n, p in successes],
        "failures": failures,
        "suite_hours": (time.perf_counter() - suite_started) / 3600.0,
        "status": "PASSED" if not failures else "COMPLETED_WITH_FAILURES",
    }
    summary_path = suite_directory / f"{source_name}_loss_suite_summary.json"
    atomic_json(summary_path, suite_summary)

    print("=" * 88)
    print(f"{source_name} LOSS SUITE FINISHED")
    print("=" * 88)
    print(f"Successful models: {len(successes)}/{len(jobs)}")
    print(f"Failed models    : {len(failures)}/{len(jobs)}")
    print(f"Suite summary    : {summary_path}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} model(s) failed. Share {summary_path} for diagnosis."
        )
