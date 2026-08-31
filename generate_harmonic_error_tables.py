#!/usr/bin/env python3
"""Generate one harmonic-error CSV from RhythmMamba evaluation folders.

The script reads every FAILURE_TYPE_SUMMARY.csv found below the supplied
result roots. It does not rerun inference and does not modify result files.
"""

import csv
from pathlib import Path


FAILURE_COLUMNS = {
    "correct": "correct_count",
    "super_harmonic_1p5x": "harmonic_1p5x_count",
    "super_harmonic_2x": "harmonic_2x_count",
    "sub_harmonic_half": "harmonic_0p5x_count",
    "large_error": "other_large_error_count",
}

PROTOCOL_NAMES = {
    "official_mamba": "Official",
    "old": "Old",
    "prism": "PRISM",
}

DATASET_NAMES = {
    "PURE": "PURE",
    "UBFC": "UBFC-rPPG",
    "TOKYOTECH": "TokyoTech",
    "BH": "BH-rPPG",
    "UBFC_PHYS": "UBFC-PHYS",
    "COHFACE": "COHFACE",
}

KNOWN_CHECKPOINT_NAMES = {
    "PURE_CHECKPOINT": "Our PURE intra (epoch 29)",
    "UBFC_CHECKPOINT": "Our UBFC intra (epoch 29)",
    "OFFICIAL_PURE_CHECKPOINT": "Released PURE cross",
    "OFFICIAL_UBFC_CHECKPOINT": "Released UBFC cross",
    "PURE_CROSS_MATCHED": "Our PURE cross-matched",
    "UBFC_CROSS_MATCHED": "Our UBFC cross-matched",
}

OUTPUT_COLUMNS = [
    "evaluation_run",
    "checkpoint",
    "dataset",
    "protocol",
    "evaluation_unit",
    "total_count",
    "correct_count",
    "harmonic_1p5x_count",
    "harmonic_2x_count",
    "harmonic_0p5x_count",
    "other_large_error_count",
]

# The complete results directory is scanned recursively. Therefore, any new
# evaluation folder is discovered automatically without changing this file.
RESULTS_ROOT = Path("results")
OUTPUT_PATH = Path(
    "results/error_analysis/HARMONIC_ERROR_COUNTS_ALL_SETUPS.csv"
)


def checkpoint_name(folder_name):
    return KNOWN_CHECKPOINT_NAMES.get(folder_name, folder_name)


def read_failure_counts(csv_path):
    counts = {name: 0 for name in FAILURE_COLUMNS.values()}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            output_name = FAILURE_COLUMNS.get(row["failure_type"].strip())
            if output_name:
                counts[output_name] = int(float(row["count"]))
    return counts


def parse_result_path(results_root, csv_path):
    relative = csv_path.relative_to(results_root)
    parts = relative.parts
    try:
        eval_index = next(i for i, part in enumerate(parts)
                          if part.startswith("Eval_On_"))
    except StopIteration:
        return None

    if eval_index == 0 or eval_index + 1 >= len(parts):
        return None

    evaluation_run = "/".join(parts[:eval_index - 1]) or "results"
    checkpoint_folder = parts[eval_index - 1]
    dataset_key = parts[eval_index].replace("Eval_On_", "", 1)
    protocol_folder = parts[eval_index + 1]
    protocol = PROTOCOL_NAMES.get(protocol_folder)
    if protocol is None:
        return None

    return evaluation_run, checkpoint_folder, dataset_key, protocol_folder, protocol


def collect_rows(results_root):
    rows = []
    seen_files = set()

    if not results_root.exists():
        raise SystemExit("Results folder was not found: " + str(results_root))

    for csv_path in results_root.rglob("FAILURE_TYPE_SUMMARY.csv"):
        resolved = csv_path.resolve()
        if resolved in seen_files:
            continue
        seen_files.add(resolved)

        parsed = parse_result_path(results_root, csv_path)
        if parsed is None:
            print("Skipped unrecognized path:", csv_path)
            continue

        evaluation_run, checkpoint_folder, dataset_key, protocol_folder, protocol = parsed
        counts = read_failure_counts(csv_path)
        total = sum(counts.values())

        row = {
            "evaluation_run": evaluation_run,
            "checkpoint": checkpoint_name(checkpoint_folder),
            "dataset": DATASET_NAMES.get(dataset_key, dataset_key),
            "protocol": protocol,
            "evaluation_unit": (
                "recording" if protocol_folder == "official_mamba"
                else "window"
            ),
            "total_count": total,
            **counts,
        }

        category_sum = sum(row[name] for name in FAILURE_COLUMNS.values())
        if category_sum != total:
            raise ValueError("Category-count mismatch in " + str(csv_path))
        rows.append(row)

    protocol_order = {"Official": 0, "Old": 1, "PRISM": 2}
    rows.sort(key=lambda row: (
        row["evaluation_run"],
        row["checkpoint"],
        row["dataset"],
        protocol_order.get(row["protocol"], 99),
    ))
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    results_root = RESULTS_ROOT.resolve()
    output_path = OUTPUT_PATH.resolve()
    rows = collect_rows(results_root)

    if not rows:
        raise SystemExit("No FAILURE_TYPE_SUMMARY.csv files were found.")

    write_csv(rows, output_path)
    checkpoint_count = len({row["checkpoint"] for row in rows})
    dataset_count = len({row["dataset"] for row in rows})

    print("=" * 78)
    print("HARMONIC-ERROR CSV GENERATED")
    print("=" * 78)
    print("Checkpoints :", checkpoint_count)
    print("Datasets    :", dataset_count)
    print("Runs        :", len({row["evaluation_run"] for row in rows}))
    print("Setups      :", len(rows))
    print("Output      :", output_path)


if __name__ == "__main__":
    main()
