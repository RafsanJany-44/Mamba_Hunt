"""Evaluate the four harmonic-loss checkpoints on 6 datasets x 3 protocols.

This module reuses ``eval_protocols.py`` unchanged so its metrics, FFT rules,
plots, diagnostic tables, and failure categories remain directly comparable
with the baseline and Stage-1 augmentation evaluations.
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

SETUPS_TO_EVALUATE = ("A0", "A2")

EXPERIMENTS = {
    "PURE": PURE,
    "UBFC": UBFC,
    "TOKYOTECH": TOKYOTECH,
    "BH": BH,
    "UBFC_PHYS": UBFC_PHYS,
    "COHFACE": COHFACE,
}

core.OUTPUT_ROOT = MAMBA_HUNT_ROOT / "results" / "evaluation_protocols_harmonic_loss"
core.GENERATE_SIGNAL_PLOTS = True
core.GENERATE_PSD_DIAGNOSTICS = True
core.GENERATE_SUMMARY_PLOTS = True
core.SAVE_SIGNAL_SAMPLE_TABLES = True


def checkpoint_name(source_dataset: str, setup: str) -> str:
    return f"{source_dataset}_{setup}_HARMONIC"


def checkpoint_path(source_dataset: str, setup: str) -> Path:
    name = checkpoint_name(source_dataset, setup)
    return (
        MAMBA_HUNT_ROOT
        / "results"
        / "models"
        / "harmonic_loss_stage1"
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

    for setup in SETUPS_TO_EVALUATE:
        checkpoint = checkpoint_path(source_dataset, setup)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    for target_dataset in DATASETS_TO_EVALUATE:
        begin, end = split_for(source_dataset, target_dataset)
        find_file_list(EXPERIMENTS[target_dataset], begin, end)


def completed_summary(run: core.EvaluationRun, protocol) -> dict | None:
    """Return a verified summary so an interrupted evaluation can resume."""
    path = core.OUTPUT_ROOT / run.name / protocol.name / "tables" / "summary.json"
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

    for setup in SETUPS_TO_EVALUATE:
        name = checkpoint_name(source_dataset, setup)
        checkpoint = checkpoint_path(source_dataset, setup)
        print("=" * 84)
        print("LOADING", name, ":", checkpoint)
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
                name=f"{name}/Eval_On_{target_dataset}",
                checkpoint=checkpoint,
                experiment=experiment,
                split_begin=begin,
                split_end=end,
                description=(
                    f"Harmonic-loss {setup} local {source_dataset} checkpoint "
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

    expected = len(SETUPS_TO_EVALUATE) * len(DATASETS_TO_EVALUATE) * len(PROTOCOLS_TO_EVALUATE)
    if len(all_summaries) != expected:
        raise RuntimeError(
            f"Expected {expected} {source_dataset} summaries, "
            f"obtained {len(all_summaries)}"
        )

    source_name = source_dataset.lower()
    summary_path = (
        core.OUTPUT_ROOT
        / f"all_results_summary_harmonic_loss_{source_name}_2x6x3.csv"
    )
    core.write_csv(summary_path, all_summaries)
    print("=" * 84)
    print(f"{source_dataset} HARMONIC-LOSS 2 x 6 x 3 EVALUATION COMPLETED")
    print("Evaluations:", len(all_summaries))
    print("Summary:", summary_path)
    print("=" * 84)
