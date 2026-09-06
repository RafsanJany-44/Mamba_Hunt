"""Evaluate one Stage-3 harmonic-rank checkpoint on 6 datasets x 3 protocols."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import eval_protocols as core
from dataset import find_file_list
from settings import BH, COHFACE, MAMBA_HUNT_ROOT, PURE, TOKYOTECH, UBFC, UBFC_PHYS


DATASETS_TO_EVALUATE = (
    "PURE", "UBFC", "TOKYOTECH", "BH", "UBFC_PHYS", "COHFACE"
)
PROTOCOLS_TO_EVALUATE = ("official_mamba", "old", "prism")
EXPERIMENTS = {
    "PURE": PURE,
    "UBFC": UBFC,
    "TOKYOTECH": TOKYOTECH,
    "BH": BH,
    "UBFC_PHYS": UBFC_PHYS,
    "COHFACE": COHFACE,
}

core.OUTPUT_ROOT = MAMBA_HUNT_ROOT / "results" / "evaluation_protocols_harmonic_rank_stage3"
core.GENERATE_SIGNAL_PLOTS = True
core.GENERATE_PSD_DIAGNOSTICS = True
core.GENERATE_SUMMARY_PLOTS = True
core.SAVE_SIGNAL_SAMPLE_TABLES = True


def model_name(source):
    return f"{source}_A2_HARMONIC_RANK"


def checkpoint_path(source):
    name = model_name(source)
    return (
        MAMBA_HUNT_ROOT / "results" / "models" / "harmonic_rank_stage3"
        / name / f"{name}_RhythmMamba_Best.pth"
    )


def split_for(source, target):
    return (0.8, 1.0) if source == target else (0.0, 1.0)


def completed_summary(run, protocol):
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


def evaluate_source(source):
    source = source.upper()
    if source not in ("PURE", "UBFC"):
        raise ValueError("source must be PURE or UBFC")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = checkpoint_path(source)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    for target in DATASETS_TO_EVALUATE:
        begin, end = split_for(source, target)
        find_file_list(EXPERIMENTS[target], begin, end)

    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    core.write_output_guide(core.OUTPUT_ROOT / "README.txt")
    name = model_name(source)
    model = core.load_model(checkpoint)
    all_summaries = []

    for target in DATASETS_TO_EVALUATE:
        experiment = EXPERIMENTS[target]
        begin, end = split_for(source, target)
        split_description = (
            "held-out 0.8-1.0 clean source validation split"
            if source == target else "complete 0.0-1.0 external dataset"
        )
        run = core.EvaluationRun(
            name=f"{name}/Eval_On_{target}",
            checkpoint=checkpoint,
            experiment=experiment,
            split_begin=begin,
            split_end=end,
            description=(
                f"Stage-3 A2 official plus conditional harmonic-ranking "
                f"{source} checkpoint evaluated on {target}: {split_description}"
            ),
        )
        file_list = find_file_list(experiment, begin, end)
        recordings = core.read_manifest(file_list)
        run_summaries = []
        for protocol_name in PROTOCOLS_TO_EVALUATE:
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

    expected = len(DATASETS_TO_EVALUATE) * len(PROTOCOLS_TO_EVALUATE)
    if len(all_summaries) != expected:
        raise RuntimeError(f"Expected {expected} summaries, obtained {len(all_summaries)}")
    summary_path = core.OUTPUT_ROOT / f"all_results_summary_harmonic_rank_{source.lower()}_1x6x3.csv"
    core.write_csv(summary_path, all_summaries)
    print("=" * 96)
    print(f"{source} STAGE-3 HARMONIC-RANK 1 x 6 x 3 COMPLETED")
    print(f"Evaluations: {len(all_summaries)}")
    print(f"Summary: {summary_path}")
    print("=" * 96)
