"""Evaluate the eight new final-candidate seeds on 6 datasets x 3 protocols."""

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
SEEDS_TO_EVALUATE = (101, 102, 103, 104)
FINAL_CODES = {
    "PURE": "L4_C100_H100",
    "UBFC": "L3_C050",
}
EXPECTED_WEIGHTS = {
    "PURE": {"pearson_weight": 0.2, "ce_weight": 0.0,
             "concentration_weight": 1.0, "harmonic_weight": 1.0},
    "UBFC": {"pearson_weight": 0.2, "ce_weight": 0.0,
             "concentration_weight": 0.5, "harmonic_weight": 0.0},
}
EXPERIMENTS = {
    "PURE": PURE,
    "UBFC": UBFC,
    "TOKYOTECH": TOKYOTECH,
    "BH": BH,
    "UBFC_PHYS": UBFC_PHYS,
    "COHFACE": COHFACE,
}

core.OUTPUT_ROOT = (
    MAMBA_HUNT_ROOT / "results" / "evaluation_protocols_a4_final_multiseed"
)
core.GENERATE_SIGNAL_PLOTS = False
core.GENERATE_PSD_DIAGNOSTICS = False
core.GENERATE_SUMMARY_PLOTS = True
core.SAVE_SIGNAL_SAMPLE_TABLES = False


def checkpoint_name(source_dataset: str, seed: int) -> str:
    return f"{source_dataset}_A4_{FINAL_CODES[source_dataset]}_SEED{seed}"


def model_directory(source_dataset: str, seed: int) -> Path:
    return (
        MAMBA_HUNT_ROOT / "results" / "models" / "a4_final_multiseed"
        / checkpoint_name(source_dataset, seed)
    )


def checkpoint_path(source_dataset: str, seed: int) -> Path:
    name = checkpoint_name(source_dataset, seed)
    return model_directory(source_dataset, seed) / f"{name}_RhythmMamba_Best.pth"


def split_for(source_dataset: str, target_dataset: str) -> tuple[float, float]:
    return (0.8, 1.0) if source_dataset == target_dataset else (0.0, 1.0)


def verify_model_metadata(source_dataset: str, seed: int) -> None:
    name = checkpoint_name(source_dataset, seed)
    directory = model_directory(source_dataset, seed)
    checkpoint = checkpoint_path(source_dataset, seed)
    completion = directory / f"{name}_completion.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not completion.is_file():
        raise FileNotFoundError(f"Missing completion marker: {completion}")
    payload = json.loads(completion.read_text(encoding="utf-8"))
    expected = {
        "status": "PASSED",
        "source_dataset": source_dataset,
        "seed": seed,
        **EXPECTED_WEIGHTS[source_dataset],
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Model metadata mismatch for {name}: {mismatches}")


def validate(source_dataset: str) -> None:
    if source_dataset not in FINAL_CODES:
        raise ValueError("source_dataset must be PURE or UBFC")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RhythmMamba evaluation")
    for seed in SEEDS_TO_EVALUATE:
        verify_model_metadata(source_dataset, seed)
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
    checks = (
        summary.get("run_name") == run.name,
        summary.get("protocol") == protocol.name,
        summary.get("completion_status") == "PASSED",
        Path(summary.get("checkpoint", "")).resolve() == run.checkpoint.resolve(),
        int(summary.get("number_of_recordings", 0)) > 0,
        int(summary.get("number_of_measurements", 0)) > 0,
    )
    if not all(checks):
        return None
    print(f"REUSING COMPLETED: {run.name}/{protocol.name}")
    return summary


def evaluate_source(source_dataset: str) -> None:
    source_dataset = source_dataset.upper()
    validate(source_dataset)
    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    core.write_output_guide(core.OUTPUT_ROOT / "README.txt")
    all_summaries = []

    for job_number, seed in enumerate(SEEDS_TO_EVALUATE, start=1):
        name = checkpoint_name(source_dataset, seed)
        checkpoint = checkpoint_path(source_dataset, seed)
        print("=" * 96)
        print(
            f"{source_dataset} A4 FINAL MULTI-SEED EVALUATION — "
            f"SEED {job_number}/{len(SEEDS_TO_EVALUATE)}: {seed}"
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
                    f"A4 {FINAL_CODES[source_dataset]}, seed {seed}, local "
                    f"{source_dataset} checkpoint evaluated on {target_dataset}: "
                    f"{split_description}"
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
        len(SEEDS_TO_EVALUATE)
        * len(DATASETS_TO_EVALUATE)
        * len(PROTOCOLS_TO_EVALUATE)
    )
    if len(all_summaries) != expected:
        raise RuntimeError(
            f"Expected {expected} {source_dataset} summaries, "
            f"obtained {len(all_summaries)}"
        )
    source_name = source_dataset.lower()
    summary_path = core.OUTPUT_ROOT / (
        f"all_results_summary_a4_final_multiseed_{source_name}_4x6x3.csv"
    )
    core.write_csv(summary_path, all_summaries)
    print("=" * 96)
    print(f"{source_dataset} A4 FINAL MULTI-SEED 4 x 6 x 3 COMPLETED")
    print("Evaluations:", len(all_summaries))
    print("Summary:", summary_path)
    print("=" * 96)
