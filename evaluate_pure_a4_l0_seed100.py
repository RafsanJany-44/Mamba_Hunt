"""Evaluate PURE_A4_L0_SEED100 on six datasets and three protocols."""

from __future__ import annotations

import json

import torch

import eval_protocols as core
from dataset import find_file_list
from settings import BH, COHFACE, MAMBA_HUNT_ROOT, PURE, TOKYOTECH, UBFC, UBFC_PHYS


MODEL_NAME = "PURE_A4_L0_SEED100"
DATASETS = ("PURE", "UBFC", "TOKYOTECH", "BH", "UBFC_PHYS", "COHFACE")
PROTOCOLS = ("official_mamba", "old", "prism")
EXPERIMENTS = {
    "PURE": PURE,
    "UBFC": UBFC,
    "TOKYOTECH": TOKYOTECH,
    "BH": BH,
    "UBFC_PHYS": UBFC_PHYS,
    "COHFACE": COHFACE,
}

CHECKPOINT = (
    MAMBA_HUNT_ROOT / "results" / "models" / "pure_a4_pilot" / MODEL_NAME
    / f"{MODEL_NAME}_RhythmMamba_Best.pth"
)
core.OUTPUT_ROOT = (
    MAMBA_HUNT_ROOT / "results" / "evaluation_protocols_pure_a4_pilot"
)
core.GENERATE_SIGNAL_PLOTS = True
core.GENERATE_PSD_DIAGNOSTICS = True
core.GENERATE_SUMMARY_PLOTS = True
core.SAVE_SIGNAL_SAMPLE_TABLES = True


def split_for(target):
    # PURE is the source dataset, so report only its untouched 0.8-1.0 split.
    # Every other dataset is external and is evaluated in full.
    return (0.8, 1.0) if target == "PURE" else (0.0, 1.0)


def completed_summary(run, protocol):
    path = core.OUTPUT_ROOT / run.name / protocol.name / "tables" / "summary.json"
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("run_name") != run.name or summary.get("protocol") != protocol.name:
        return None
    if int(summary.get("number_of_recordings", 0)) <= 0:
        return None
    if int(summary.get("number_of_measurements", 0)) <= 0:
        return None
    print(f"REUSING COMPLETED: {run.name}/{protocol.name}")
    return summary


def validate():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")
    for target in DATASETS:
        begin, end = split_for(target)
        find_file_list(EXPERIMENTS[target], begin, end)


def main():
    validate()
    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    core.write_output_guide(core.OUTPUT_ROOT / "README.txt")
    model = core.load_model(CHECKPOINT)
    all_summaries = []

    print("=" * 98)
    print(f"PURE A4 PILOT EVALUATION: {MODEL_NAME}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Output: {core.OUTPUT_ROOT}")
    print("=" * 98)

    for target in DATASETS:
        experiment = EXPERIMENTS[target]
        begin, end = split_for(target)
        split_description = (
            "held-out PURE 0.8-1.0 source validation split"
            if target == "PURE" else "complete 0.0-1.0 external dataset"
        )
        run = core.EvaluationRun(
            name=f"{MODEL_NAME}/Eval_On_{target}",
            checkpoint=CHECKPOINT,
            experiment=experiment,
            split_begin=begin,
            split_end=end,
            description=(
                f"PURE A4 L0 seed-100 pilot evaluated on {target}: "
                f"{split_description}"
            ),
        )
        file_list = find_file_list(experiment, begin, end)
        recordings = core.read_manifest(file_list)
        run_summaries = []
        for protocol_name in PROTOCOLS:
            protocol = core.PROTOCOLS[protocol_name]
            summary = completed_summary(run, protocol)
            if summary is None:
                summary = core.run_protocol(model, run, protocol, file_list, recordings)
            run_summaries.append(summary)
            all_summaries.append(summary)

        comparison = core.OUTPUT_ROOT / run.name / "protocol_comparison"
        comparison.mkdir(parents=True, exist_ok=True)
        core.write_csv(comparison / "PROTOCOL_COMPARISON.csv", run_summaries)
        core.make_protocol_comparison_plot(
            run_summaries,
            comparison / "protocol_comparison.html",
            f"Protocol comparison — {run.name}",
        )
        core.write_protocol_comparison_note(comparison / "INTERPRETATION.txt")

    expected = len(DATASETS) * len(PROTOCOLS)
    if len(all_summaries) != expected:
        raise RuntimeError(f"Expected {expected} summaries, obtained {len(all_summaries)}")
    summary_path = core.OUTPUT_ROOT / "all_results_summary_PURE_A4_L0_SEED100_1x6x3.csv"
    core.write_csv(summary_path, all_summaries)
    del model
    torch.cuda.empty_cache()
    print("=" * 98)
    print("PURE A4 L0 SEED100 — 1 x 6 x 3 EVALUATION COMPLETED")
    print(f"Evaluations: {len(all_summaries)}")
    print(f"Summary: {summary_path}")
    print("=" * 98)


if __name__ == "__main__":
    main()
