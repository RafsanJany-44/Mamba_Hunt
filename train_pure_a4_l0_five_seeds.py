"""Train PURE A4 L0 with seeds 100-104 using the verified A4 trainer."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import dataset_pure_a4
import train_ubfc_a4_l0_seed100 as trainer
from cross_settings import PURE_CROSS_MATCHED
from settings import OUTPUT_ROOT


SEEDS = (100, 101, 102, 103, 104)
MODEL_FAMILY_DIRECTORY = "pure_a4_pilot"


def main():
    started = time.perf_counter()
    statuses = []

    trainer.SOURCE_DATASET = "PURE"
    trainer.MODEL_FAMILY_DIRECTORY = MODEL_FAMILY_DIRECTORY
    trainer.UBFC_CROSS_MATCHED = PURE_CROSS_MATCHED
    trainer.dataset_ubfc_a4 = dataset_pure_a4

    for number, seed in enumerate(SEEDS, start=1):
        name = f"PURE_A4_L0_SEED{seed}"
        print("#" * 92)
        print(f"PURE A4 L0 — SEED {number}/{len(SEEDS)}: {seed}")
        print("#" * 92)

        trainer.SEED = seed
        trainer.NAME = name
        trainer.main()

        completion_path = (
            OUTPUT_ROOT / "models" / MODEL_FAMILY_DIRECTORY / name
            / f"{name}_completion.json"
        )
        checkpoint_path = (
            OUTPUT_ROOT / "models" / MODEL_FAMILY_DIRECTORY / name
            / f"{name}_RhythmMamba_Best.pth"
        )
        if not completion_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"Missing completed model files for {name}")

        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "PASSED"
            or payload.get("seed") != seed
            or payload.get("source_dataset") != "PURE"
        ):
            raise RuntimeError(f"Invalid completion record: {completion_path}")
        statuses.append(payload)

    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_family": "PURE_A4_L0",
        "seeds": list(SEEDS),
        "successful_models": len(statuses),
        "planned_models": len(SEEDS),
        "suite_hours": (time.perf_counter() - started) / 3600.0,
        "runs": statuses,
        "status": "PASSED",
    }
    summary_path = (
        OUTPUT_ROOT / "models" / MODEL_FAMILY_DIRECTORY
        / "PURE_A4_L0_five_seed_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)

    print("=" * 92)
    print("PURE A4 L0 FIVE-SEED TRAINING COMPLETED")
    print("=" * 92)
    print(f"Successful models: {len(statuses)}/{len(SEEDS)}")
    print("Seeds: 100, 101, 102, 103, 104")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
