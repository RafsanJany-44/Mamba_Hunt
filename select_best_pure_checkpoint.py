"""Select the best existing PURE checkpoint using PURE validation data only.

Place this file directly inside ``Mamba_Hunt`` and run it without arguments.
The external datasets, including COHFACE, are never used for selection.
"""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset import create_test_loader
from loss import HybridLoss
from metrics import calculate_video_metrics
from settings import DEVICE, EPOCHS, FS, PURE
from trainer import build_model, normalize_prediction, set_reproducible


RESULTS_CSV = PURE.model_dir / "PURE_epoch_validation_results.csv"
BEST_CHECKPOINT = PURE.model_dir / "PURE_RhythmMamba_Best.pth"


def checkpoint_path(epoch: int) -> Path:
    return PURE.model_dir / f"PURE_RhythmMamba_Epoch{epoch}.pth"


def save_results(rows: list[dict]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "validation_loss",
                "validation_mae_bpm",
                "checkpoint",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def reconstructed_mae(
    predictions: dict[str, dict[int, torch.Tensor]],
    labels: dict[str, dict[int, torch.Tensor]],
) -> float:
    errors = []

    for recording_id in predictions:
        prediction = torch.cat(
            [
                value
                for _, value in sorted(predictions[recording_id].items())
            ]
        ).numpy()
        label = torch.cat(
            [value for _, value in sorted(labels[recording_id].items())]
        ).numpy()

        ground_truth_hr, predicted_hr, _ = calculate_video_metrics(
            prediction,
            label,
            fs=FS,
            diff_flag=False,
        )
        errors.append(abs(predicted_hr - ground_truth_hr))

    return float(np.mean(errors))


@torch.no_grad()
def evaluate_checkpoint(model, loader, criterion, epoch: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_clips = 0
    predictions: dict[str, dict[int, torch.Tensor]] = {}
    labels: dict[str, dict[int, torch.Tensor]] = {}

    progress = tqdm(
        loader,
        desc=f"Validating epoch {epoch:02d}",
        ncols=88,
        leave=False,
    )

    for batch in progress:
        data = batch[0].float().to(DEVICE)
        label = batch[1].float().to(DEVICE)
        prediction = normalize_prediction(model(data))

        for item in range(data.shape[0]):
            loss = criterion(
                prediction[item],
                label[item],
                epoch,
                FS,
                False,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite validation loss at epoch {epoch}, "
                    f"batch item {item}."
                )

            total_loss += float(loss.item())
            total_clips += 1

            recording_id = str(batch[2][item])
            chunk_id = int(batch[3][item])
            predictions.setdefault(recording_id, {})[chunk_id] = (
                prediction[item].detach().cpu().reshape(-1)
            )
            labels.setdefault(recording_id, {})[chunk_id] = (
                label[item].detach().cpu().reshape(-1)
            )

    if total_clips == 0:
        raise RuntimeError("PURE validation loader is empty.")

    mean_loss = total_loss / total_clips
    mae_bpm = reconstructed_mae(predictions, labels)

    if not math.isfinite(mean_loss) or not math.isfinite(mae_bpm):
        raise RuntimeError(f"Non-finite result at epoch {epoch}.")

    return mean_loss, mae_bpm


def main() -> None:
    set_reproducible()

    missing = [checkpoint_path(epoch) for epoch in range(EPOCHS)]
    missing = [path for path in missing if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} PURE checkpoints. "
            f"First missing checkpoint: {missing[0]}"
        )

    validation_loader = create_test_loader(PURE)
    model = build_model()
    criterion = HybridLoss()
    rows = []

    print("=" * 78)
    print("PURE CHECKPOINT SELECTION")
    print("=" * 78)
    print(f"Validation split : {PURE.test_begin}-{PURE.test_end}")
    print(f"Validation clips : {len(validation_loader.dataset)}")
    print(f"Epochs checked   : 0-{EPOCHS - 1}")
    print("Selection metric : lowest mean PURE validation loss")
    print("COHFACE used     : False")

    for epoch in range(EPOCHS):
        path = checkpoint_path(epoch)
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)

        validation_loss, validation_mae = evaluate_checkpoint(
            model,
            validation_loader,
            criterion,
            epoch,
        )

        row = {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_mae_bpm": validation_mae,
            "checkpoint": str(path),
        }
        rows.append(row)
        save_results(rows)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={validation_loss:.8f} | "
            f"MAE={validation_mae:.6f} BPM"
        )

    best = min(rows, key=lambda row: row["validation_loss"])
    best_source = Path(best["checkpoint"])
    shutil.copy2(best_source, BEST_CHECKPOINT)

    print("=" * 78)
    print("BEST PURE CHECKPOINT SELECTED")
    print("=" * 78)
    print(f"Best epoch       : {best['epoch']}")
    print(f"Validation loss  : {best['validation_loss']:.8f}")
    print(f"Validation MAE   : {best['validation_mae_bpm']:.6f} BPM")
    print(f"Original file    : {best_source}")
    print(f"Best checkpoint  : {BEST_CHECKPOINT}")
    print(f"All epoch results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
