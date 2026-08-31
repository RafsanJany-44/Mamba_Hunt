"""Evaluate three UBFC cross-matched seed checkpoints on six datasets.

The UBFC source dataset uses its held-out 0.8-1.0 validation partition.
Every external dataset uses its complete 0.0-1.0 manifest. Existing
evaluation folders are untouched.
"""

from __future__ import annotations

import torch

import eval_protocols as core
from dataset import find_file_list
from settings import BH, COHFACE, MAMBA_HUNT_ROOT, PURE, TOKYOTECH, UBFC, UBFC_PHYS


CHECKPOINTS_TO_EVALUATE = (
    "UBFC_CROSS_MATCHED",
    "UBFC_CROSS_MATCHED_SEED101",
    "UBFC_CROSS_MATCHED_SEED102",
)

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

core.OUTPUT_ROOT = (
    MAMBA_HUNT_ROOT / "results" / "evaluation_protocols_ubfc_seed_stability"
)
core.GENERATE_SIGNAL_PLOTS = True
core.GENERATE_PSD_DIAGNOSTICS = True
core.GENERATE_SUMMARY_PLOTS = True
core.SAVE_SIGNAL_SAMPLE_TABLES = True


EXPERIMENTS = {
    "PURE": PURE,
    "UBFC": UBFC,
    "TOKYOTECH": TOKYOTECH,
    "BH": BH,
    "UBFC_PHYS": UBFC_PHYS,
    "COHFACE": COHFACE,
}

CHECKPOINTS = {
    checkpoint_name: (
        MAMBA_HUNT_ROOT
        / "results"
        / "models"
        / checkpoint_name
        / f"{checkpoint_name}_RhythmMamba_Best.pth",
        "UBFC",
    )
    for checkpoint_name in CHECKPOINTS_TO_EVALUATE
}


def split_for(source_dataset: str, target_dataset: str) -> tuple[float, float]:
    if source_dataset == target_dataset:
        return 0.8, 1.0
    return 0.0, 1.0


def validate() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RhythmMamba evaluation")

    for checkpoint_name in CHECKPOINTS_TO_EVALUATE:
        checkpoint, source_dataset = CHECKPOINTS[checkpoint_name]
        if not checkpoint.is_file():
            raise FileNotFoundError("Missing checkpoint: " + str(checkpoint))

        for target_dataset in DATASETS_TO_EVALUATE:
            begin, end = split_for(source_dataset, target_dataset)
            find_file_list(EXPERIMENTS[target_dataset], begin, end)


def main() -> None:
    validate()
    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    core.write_output_guide(core.OUTPUT_ROOT / "README.txt")
    all_summaries = []

    for checkpoint_name in CHECKPOINTS_TO_EVALUATE:
        checkpoint, source_dataset = CHECKPOINTS[checkpoint_name]
        print("=" * 84)
        print("LOADING", checkpoint_name, ":", checkpoint)
        print("=" * 84)
        model = core.load_model(checkpoint)

        for target_dataset in DATASETS_TO_EVALUATE:
            experiment = EXPERIMENTS[target_dataset]
            begin, end = split_for(source_dataset, target_dataset)
            split_description = (
                "held-out 0.8-1.0 UBFC validation split"
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
                    f"UBFC cross-matched seed checkpoint evaluated on "
                    f"{target_dataset}: {split_description}"
                ),
            )

            file_list = find_file_list(experiment, begin, end)
            recordings = core.read_manifest(file_list)
            run_summaries = []

            for protocol_name in PROTOCOLS_TO_EVALUATE:
                summary = core.run_protocol(
                    model,
                    run,
                    core.PROTOCOLS[protocol_name],
                    file_list,
                    recordings,
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

    core.write_csv(core.OUTPUT_ROOT / "all_results_summary.csv", all_summaries)
    core.write_csv(
        core.OUTPUT_ROOT / "all_results_summary_ubfc_seed_stability_3x6x3.csv",
        all_summaries,
    )
    print("=" * 84)
    print("UBFC SEED-STABILITY 3 x 6 x 3 EVALUATION COMPLETED")
    print("Evaluations:", len(all_summaries))
    print("Summary:", core.OUTPUT_ROOT / "all_results_summary.csv")
    print("=" * 84)


if __name__ == "__main__":
    main()
