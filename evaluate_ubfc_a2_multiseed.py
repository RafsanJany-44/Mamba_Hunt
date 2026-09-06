"""Evaluate five paired UBFC A2 L0/L5 seeds on 6 datasets x 3 protocols."""

from __future__ import annotations

import json

import torch

import eval_protocols as core
from dataset import find_file_list
from settings import BH, COHFACE, MAMBA_HUNT_ROOT, PURE, TOKYOTECH, UBFC, UBFC_PHYS


SEEDS = (100, 101, 102, 103, 104)
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
LABELS = {
    "L0": "UBFC_A2_L0",
    "L5": "UBFC_A2_L5_CE_CONCENTRATION",
}

core.OUTPUT_ROOT = MAMBA_HUNT_ROOT / "results" / "evaluation_protocols_ubfc_a2_multiseed"
core.GENERATE_SIGNAL_PLOTS = True
core.GENERATE_PSD_DIAGNOSTICS = True
core.GENERATE_SUMMARY_PLOTS = True
core.SAVE_SIGNAL_SAMPLE_TABLES = True


def model_name(variant, seed):
    return f"{LABELS[variant]}_SEED{seed}"


def checkpoint_path(variant, seed):
    name = model_name(variant, seed)
    return (
        MAMBA_HUNT_ROOT / "results" / "models" / "ubfc_a2_multiseed"
        / name / f"{name}_RhythmMamba_Best.pth"
    )


def split_for(target):
    return (0.8, 1.0) if target == "UBFC" else (0.0, 1.0)


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


def validate(variant):
    if variant not in LABELS:
        raise ValueError("variant must be L0 or L5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for seed in SEEDS:
        checkpoint = checkpoint_path(variant, seed)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    for target in DATASETS:
        begin, end = split_for(target)
        find_file_list(EXPERIMENTS[target], begin, end)


def evaluate_variant(variant):
    variant = variant.upper()
    validate(variant)
    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    core.write_output_guide(core.OUTPUT_ROOT / "README.txt")
    all_summaries = []

    for number, seed in enumerate(SEEDS, start=1):
        name = model_name(variant, seed)
        checkpoint = checkpoint_path(variant, seed)
        print("=" * 98)
        print(f"{LABELS[variant]} EVALUATION — SEED {number}/{len(SEEDS)}: {seed}")
        print(f"Checkpoint: {checkpoint}")
        print("=" * 98)
        model = core.load_model(checkpoint)

        for target in DATASETS:
            experiment = EXPERIMENTS[target]
            begin, end = split_for(target)
            split_description = (
                "held-out UBFC 0.8-1.0 source validation split"
                if target == "UBFC" else "complete 0.0-1.0 external dataset"
            )
            run = core.EvaluationRun(
                name=f"{name}/Eval_On_{target}",
                checkpoint=checkpoint,
                experiment=experiment,
                split_begin=begin,
                split_end=end,
                description=(
                    f"Paired UBFC A2 five-seed experiment; {variant}, seed {seed}, "
                    f"evaluated on {target}: {split_description}"
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

        del model
        torch.cuda.empty_cache()

    expected = len(SEEDS) * len(DATASETS) * len(PROTOCOLS)
    if len(all_summaries) != expected:
        raise RuntimeError(f"Expected {expected} summaries, obtained {len(all_summaries)}")
    summary_path = core.OUTPUT_ROOT / f"all_results_summary_{LABELS[variant]}_5x6x3.csv"
    core.write_csv(summary_path, all_summaries)
    print("=" * 98)
    print(f"{LABELS[variant]} FIVE-SEED 5 x 6 x 3 EVALUATION COMPLETED")
    print(f"Evaluations: {len(all_summaries)}")
    print(f"Summary: {summary_path}")
    print("=" * 98)
