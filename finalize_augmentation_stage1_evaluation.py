"""Validate and combine the two parallel Stage-1 evaluation summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import eval_protocols as core
from settings import MAMBA_HUNT_ROOT


ROOT = MAMBA_HUNT_ROOT / "results" / "evaluation_protocols_augmentation_stage1"
SOURCES = (
    ROOT / "all_results_summary_augmentation_stage1_pure_3x6x3.csv",
    ROOT / "all_results_summary_augmentation_stage1_ubfc_3x6x3.csv",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Source evaluation summary is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 54:
        raise RuntimeError(f"Expected 54 rows in {path}, found {len(rows)}")
    return rows


def main() -> None:
    rows = read_rows(SOURCES[0]) + read_rows(SOURCES[1])
    keys = [(row["run_name"], row["protocol"]) for row in rows]
    if len(rows) != 108 or len(set(keys)) != 108:
        raise RuntimeError(
            f"Expected 108 unique run/protocol results; rows={len(rows)}, "
            f"unique={len(set(keys))}"
        )
    core.write_csv(ROOT / "all_results_summary.csv", rows)
    final_path = ROOT / "all_results_summary_augmentation_stage1_6x6x3.csv"
    core.write_csv(final_path, rows)
    print("=" * 84)
    print("AUGMENTATION STAGE-1 6 x 6 x 3 EVALUATION FINALIZED")
    print("Evaluations: 108")
    print("Summary:", final_path)
    print("=" * 84)


if __name__ == "__main__":
    main()

