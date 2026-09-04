"""Evaluate Stage-2 loss-suite checkpoints on 6 datasets x 3 protocols."""

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

VARIANTS_TO_EVALUATE = (
    "L2_HARMONIC_REPLACE",
    "L3_CONCENTRATION",
    "L4_CONCENTRATION_HARMONIC",
    "L5_CE_CONCENTRATION",
)

AUGMENTATIONS_TO_EVALUATE = ("A0", "A2")

EXPERIMENTS = {
    "PURE": PURE,
    "UBFC": UBFC,
    "TOKYOTECH": TOKYOTECH,
    "BH": BH,
    "UBFC_PHYS": UBFC_PHYS,
    "COHFACE": COHFACE,
}

core.OUTPUT_ROOT = (
    MAMBA_HUNT_ROOT / "results" / "evaluation_protocols_loss_suite_stage2"
)
core.GENERATE_SIGNAL_PLOTS = True
core.GENERATE_PSD_DIAGNOSTICS = True
core.GENERATE_SUMMARY_PLOTS = True
core.SAVE_SIGNAL_SAMPLE_TABLES = True


def checkpoint_name(source_dataset: str, augmentation: str, variant: str) -> str:
    return f"{source_dataset}_{augmentation}_{variant}"


def checkpoint_path(source_dataset: str, augmentation: str, variant: str) -> Path:
    name = checkpoint_name(source_dataset, augmentation, variant)
    return (
        MAMBA_HUNT_ROOT
        / "results"
        / "models"
        / "loss_suite_stage2"
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

    for variant in VARIANTS_TO_EVALUATE:
        for augmentation in AUGMENTATIONS_TO_EVALUATE:
            checkpoint = checkpoint_path(source_dataset, augmentation, variant)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    for target_dataset in DATASETS_TO_EVALUATE:
        begin, end = split_for(source_dataset, target_dataset)
        find_file_list(EXPERIMENTS[target_dataset], begin, end)


def completed_summary(run: core.EvaluationRun, protocol) -> dict | None:
    """Reuse only a structurally verified completed protocol summary."""
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

    jobs = [
        (variant, augmentation)
        for variant in VARIANTS_TO_EVALUATE
        for augmentation in AUGMENTATIONS_TO_EVALUATE
    ]

    for job_number, (variant, augmentation) in enumerate(jobs, start=1):
        name = checkpoint_name(source_dataset, augmentation, variant)
        checkpoint = checkpoint_path(source_dataset, augmentation, variant)
        print("=" * 96)
        print(
            f"{source_dataset} LOSS-SUITE EVALUATION — "
            f"CHECKPOINT {job_number}/{len(jobs)}"
        )
        print("LOADING", name, ":", checkpoint)
        print("=" * 96)
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
                    f"Stage-2 {variant}, {augmentation}, local {source_dataset} "
                    f"checkpoint evaluated on {target_dataset}: {split_description}"
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

    expected = (
        len(VARIANTS_TO_EVALUATE)
        * len(AUGMENTATIONS_TO_EVALUATE)
        * len(DATASETS_TO_EVALUATE)
        * len(PROTOCOLS_TO_EVALUATE)
    )
    if len(all_summaries) != expected:
        raise RuntimeError(
            f"Expected {expected} {source_dataset} summaries, "
            f"obtained {len(all_summaries)}"
        )

    source_name = source_dataset.lower()
    summary_path = (
        core.OUTPUT_ROOT
        / f"all_results_summary_loss_suite_stage2_{source_name}_8x6x3.csv"
    )
    core.write_csv(summary_path, all_summaries)
    print("=" * 96)
    print(f"{source_dataset} STAGE-2 LOSS SUITE 8 x 6 x 3 COMPLETED")
    print("Evaluations:", len(all_summaries))
    print("Summary:", summary_path)
    print("=" * 96)
