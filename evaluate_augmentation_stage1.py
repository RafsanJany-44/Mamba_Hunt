"""Shared 3 x 6 x 3 evaluator for one Stage-1 source dataset.

This module deliberately reuses ``eval_protocols.py`` without modifying its
metric, FFT, plotting, failure-category, or signal-diagnostic behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import eval_protocols as core
from dataset import find_file_list
from settings import BH, COHFACE, MAMBA_HUNT_ROOT, PURE, TOKYOTECH, UBFC, UBFC_PHYS


DATASETS_TO_EVALUATE = (
    "PURE",
    "UBFC",
    "TOKYOTECH",
    "BH",
    "UBFC_PHYS",
    "COHFACE",
)

PROTOCOLS_TO_EVALUATE = (
    "official_mamba",
    "old",
    "prism",
)

EXPERIMENTS = {
    "PURE": PURE,
    "UBFC": UBFC,
    "TOKYOTECH": TOKYOTECH,
    "BH": BH,
    "UBFC_PHYS": UBFC_PHYS,
    "COHFACE": COHFACE,
}

core.OUTPUT_ROOT = (
    MAMBA_HUNT_ROOT / "results" / "evaluation_protocols_augmentation_stage1"
)
core.GENERATE_SIGNAL_PLOTS = True
core.GENERATE_PSD_DIAGNOSTICS = True
core.GENERATE_SUMMARY_PLOTS = True
core.SAVE_SIGNAL_SAMPLE_TABLES = True


def checkpoint_path(source_dataset: str, setup: str) -> Path:
    name = f"{source_dataset}_{setup}"
    return (
        MAMBA_HUNT_ROOT
        / "results"
        / "models"
        / "augmentation_stage1"
        / name
        / f"{name}_RhythmMamba_Best.pth"
    )


def split_for(source_dataset: str, target_dataset: str) -> tuple[float, float]:
    if source_dataset == target_dataset:
        return 0.8, 1.0
    return 0.0, 1.0


def validate(source_dataset: str) -> None:
    if source_dataset not in ("PURE", "UBFC"):
        raise ValueError("source_dataset must be PURE or UBFC")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RhythmMamba evaluation")

    for setup in ("A1", "A2", "A3"):
        checkpoint = checkpoint_path(source_dataset, setup)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    for target_dataset in DATASETS_TO_EVALUATE:
        begin, end = split_for(source_dataset, target_dataset)
        find_file_list(EXPERIMENTS[target_dataset], begin, end)


def completed_summary(run: core.EvaluationRun, protocol) -> dict | None:
    """Return a verified completed summary so interrupted jobs can resume."""
    path = (
        core.OUTPUT_ROOT
        / run.name
        / protocol.name
        / "tables"
        / "summary.json"
    )
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("run_name") != run.name:
        return None
    if summary.get("protocol") != protocol.name:
        return None
    if int(summary.get("number_of_recordings", 0)) <= 0:
        return None
    if int(summary.get("number_of_measurements", 0)) <= 0:
        return None
    print(f"REUSING COMPLETED: {run.name}/{protocol.name}")
    return summary


def evaluate_source(source_dataset: str) -> None:
    source_dataset = source_dataset.upper()
    validate(source_dataset)
    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    core.write_output_guide(core.OUTPUT_ROOT / "README.txt")
    all_summaries = []

    for setup in ("A1", "A2", "A3"):
        checkpoint_name = f"{source_dataset}_{setup}"
        checkpoint = checkpoint_path(source_dataset, setup)
        print("=" * 84)
        print("LOADING", checkpoint_name, ":", checkpoint)
        print("=" * 84)
        model = core.load_model(checkpoint)

        for target_dataset in DATASETS_TO_EVALUATE:
            experiment = EXPERIMENTS[target_dataset]
            begin, end = split_for(source_dataset, target_dataset)
            split_description = (
                "held-out 0.8-1.0 clean source validation split"
                if source_dataset == target_dataset
                else "complete 0.0-1.0 external dataset"
            )
            run = core.EvaluationRun(
                name=f"{checkpoint_name}/Eval_On_{target_dataset}",
                checkpoint=checkpoint,
                experiment=experiment,
                split_begin=begin,
                split_end=end,
                description=(
                    f"Stage-1 {setup} local {source_dataset} checkpoint "
                    f"evaluated on {target_dataset}: {split_description}"
                ),
            )
            file_list = find_file_list(experiment, begin, end)
            recordings = core.read_manifest(file_list)
            run_summaries = []

            for protocol_name in PROTOCOLS_TO_EVALUATE:
                protocol = core.PROTOCOLS[protocol_name]
                summary = completed_summary(run, protocol)
                if summary is None:
                    summary = core.run_protocol(
                        model, run, protocol, file_list, recordings
                    )
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

        del model
        torch.cuda.empty_cache()

    if len(all_summaries) != 54:
        raise RuntimeError(
            f"Expected 54 {source_dataset} summaries, obtained {len(all_summaries)}"
        )
    source_name = source_dataset.lower()
    summary_path = (
        core.OUTPUT_ROOT
        / f"all_results_summary_augmentation_stage1_{source_name}_3x6x3.csv"
    )
    core.write_csv(summary_path, all_summaries)
    print("=" * 84)
    print(f"{source_dataset} STAGE-1 3 x 6 x 3 EVALUATION COMPLETED")
    print("Evaluations:", len(all_summaries))
    print("Summary:", summary_path)
    print("=" * 84)

