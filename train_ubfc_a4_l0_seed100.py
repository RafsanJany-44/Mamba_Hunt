"""Single-seed UBFC A4 pilot using the unchanged official L0 loss."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone

import torch
from tqdm import tqdm

from augmentation import OfficialAugmentation
from cross_settings import UBFC_CROSS_MATCHED
import dataset_ubfc_a4
from loss import HybridLoss
from settings import DEVICE, FS, LEARNING_RATE, OUTPUT_ROOT
from trainer import build_model, normalize_prediction, set_reproducible


NAME = "UBFC_A4_L0_SEED100"
SOURCE_DATASET = "UBFC"
MODEL_FAMILY_DIRECTORY = "ubfc_a4_pilot"
SEED = 100
MAX_EPOCHS = 100
MINIMUM_EPOCHS = 30
PATIENCE = 10


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_model(model, path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pth.tmp", delete=False) as handle:
            temporary = handle.name
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def mean_loss(prediction, label, criterion, epoch):
    total = prediction.new_tensor(0.0)
    for index in range(prediction.shape[0]):
        total = total + criterion(prediction[index], label[index], epoch, FS, False)
    return total / prediction.shape[0]


@torch.no_grad()
def validate(model, loader, criterion, epoch):
    model.eval()
    total = 0.0
    clips = 0
    progress = tqdm(loader, desc=f"Validation epoch {epoch}", ncols=96)
    for batch in progress:
        data = batch[0].float().to(DEVICE)
        label = batch[1].float().to(DEVICE)
        loss = mean_loss(normalize_prediction(model(data)), label, criterion, epoch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
        total += float(loss.item()) * data.shape[0]
        clips += data.shape[0]
        progress.set_postfix(total=loss.item())
    if clips == 0:
        raise RuntimeError("Validation loader is empty")
    return total / clips


def main():
    directory = OUTPUT_ROOT / "models" / MODEL_FAMILY_DIRECTORY / NAME
    best_path = directory / f"{NAME}_RhythmMamba_Best.pth"
    history_path = directory / f"{NAME}_training_history.csv"
    config_path = directory / f"{NAME}_configuration.json"
    completion_path = directory / f"{NAME}_completion.json"

    if completion_path.is_file() and best_path.is_file():
        completed = json.loads(completion_path.read_text(encoding="utf-8"))
        if completed.get("status") == "PASSED":
            print(f"SKIPPING VERIFIED COMPLETED MODEL: {NAME}")
            return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            f"Incomplete prior run exists: {directory}\n"
            "Preserve and rename that directory before restarting."
        )
    directory.mkdir(parents=True, exist_ok=True)

    set_reproducible(SEED)
    dataset_ubfc_a4.SEED = SEED
    train_loader, valid_loader = dataset_ubfc_a4.create_ubfc_a4_loaders(UBFC_CROSS_MATCHED)
    model = build_model()
    criterion = HybridLoss()
    official_augmentation = OfficialAugmentation(fs=FS, diff_flag=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, epochs=MAX_EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": NAME,
        "source_dataset": SOURCE_DATASET,
        "augmentation": "A4",
        "offline_sampling": "0.5 original; otherwise uniform jpeg/blur/gamma/contrast",
        "official_training_augmentation": True,
        "loss": "L0 official HybridLoss",
        "training_split": [0.0, 0.8],
        "validation_split": [0.8, 1.0],
        "maximum_epochs": MAX_EPOCHS,
        "minimum_epochs": MINIMUM_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "checkpoint_rule": "lowest clean validation loss",
        "saved_checkpoints": "best only",
        "seed": SEED,
    }
    atomic_json(config_path, configuration)

    print("=" * 92)
    print(f"{SOURCE_DATASET} A4 PILOT TRAINING: {NAME}")
    print("=" * 92)
    print(f"Training clips/epoch : {len(train_loader.dataset)}")
    print(f"Clean validation     : {len(valid_loader.dataset)}")
    print("Sampling             : 50% original; 50% uniformly selected A4 variant")
    print(f"Maximum epochs       : {MAX_EPOCHS}")
    print(f"Early-stop patience  : {PATIENCE}")

    best_loss = float("inf")
    best_epoch = None
    stale = 0
    rows = []
    started = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        model.train()
        total = 0.0
        clips = 0
        selections = Counter()
        progress = tqdm(train_loader, desc=f"{NAME} epoch {epoch}", ncols=118)
        for batch in progress:
            data = batch[0].float()
            label = batch[1].float()
            selections.update(batch[4])
            data, label = official_augmentation(data, label, batch[2], batch[3])
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            loss = mean_loss(normalize_prediction(model(data)), label, criterion, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.item()) * data.shape[0]
            clips += data.shape[0]
            progress.set_postfix(total=loss.item())

        train_loss = total / clips
        valid_loss = validate(model, valid_loader, criterion, epoch)
        improved = valid_loss < best_loss
        if improved:
            best_loss = valid_loss
            best_epoch = epoch
            stale = 0
            atomic_model(model, best_path)
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "training_total_loss": f"{train_loss:.10f}",
            "validation_total_loss": f"{valid_loss:.10f}",
            "learning_rate": f"{scheduler.get_last_lr()[0]:.12g}",
            "original_selected": selections["original"],
            "jpeg_selected": selections["jpeg"],
            "blur_selected": selections["blur"],
            "gamma_selected": selections["gamma"],
            "contrast_selected": selections["contrast"],
            "is_best": improved,
            "epochs_without_improvement": stale,
        }
        rows.append(row)
        atomic_csv(history_path, rows)
        print(
            f"Epoch {epoch:03d} | train={train_loss:.8f} | valid={valid_loss:.8f} | "
            f"best={improved} | selections={dict(selections)}"
        )
        if epoch + 1 >= MINIMUM_EPOCHS and stale >= PATIENCE:
            print(f"Early stopping after {PATIENCE} stale epochs.")
            break

    if best_epoch is None or not best_path.is_file():
        raise RuntimeError("Training ended without a valid best checkpoint")
    completion = {
        **configuration,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_completed": len(rows),
        "stopped_early": len(rows) < MAX_EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_loss,
        "best_checkpoint": str(best_path.resolve()),
        "training_history": str(history_path.resolve()),
        "training_hours": (time.perf_counter() - started) / 3600.0,
        "status": "PASSED",
    }
    atomic_json(completion_path, completion)
    print("=" * 92)
    print(f"{NAME} TRAINING COMPLETED")
    print("=" * 92)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best valid loss : {best_loss:.8f}")
    print(f"Best checkpoint : {best_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
