"""Train the four remaining UBFC A4 L0 seeds using the verified pilot trainer."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import train_ubfc_a4_l0_seed100 as trainer
from settings import OUTPUT_ROOT


SEEDS = (101, 102, 103, 104)


def main():
    started = time.perf_counter()
    statuses = []
    for number, seed in enumerate(SEEDS, start=1):
        name = f"UBFC_A4_L0_SEED{seed}"
        print("#" * 92)
        print(f"UBFC A4 L0 — REMAINING SEED {number}/{len(SEEDS)}: {seed}")
        print("#" * 92)
        trainer.SEED = seed
        trainer.NAME = name
        trainer.main()

        completion_path = (
            OUTPUT_ROOT / "models" / "ubfc_a4_pilot" / name
            / f"{name}_completion.json"
        )
        if not completion_path.is_file():
            raise RuntimeError(f"Missing completion record after training: {completion_path}")
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASSED" or payload.get("seed") != seed:
            raise RuntimeError(f"Invalid completion record: {completion_path}")
        statuses.append(payload)

    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_family": "UBFC_A4_L0",
        "seeds_trained_by_this_launcher": list(SEEDS),
        "existing_pilot_seed": 100,
        "successful_models": len(statuses),
        "planned_models": len(SEEDS),
        "suite_hours": (time.perf_counter() - started) / 3600.0,
        "runs": statuses,
        "status": "PASSED",
    }
    summary_path = (
        OUTPUT_ROOT / "models" / "ubfc_a4_pilot"
        / "UBFC_A4_L0_remaining_four_seed_summary.json"
    )
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)

    print("=" * 92)
    print("UBFC A4 L0 REMAINING FOUR-SEED TRAINING COMPLETED")
    print("=" * 92)
    print(f"Successful models: {len(statuses)}/{len(SEEDS)}")
    print("Complete seed family: 100, 101, 102, 103, 104")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
