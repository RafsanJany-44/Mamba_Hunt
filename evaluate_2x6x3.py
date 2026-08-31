"""Evaluate two locally trained checkpoints on six datasets and 3 protocols.

Protocol policy:
* A checkpoint's native dataset uses the held-out native test split.
* Every external target dataset uses its complete 0.0-1.0 manifest.
* The same model prediction is assessed by official_mamba, old, and prism.

No YAML or command-line arguments are used. Edit the constants below only if
you intentionally want a smaller trial run.
"""

from __future__ import annotations

import torch

import eval_protocols as core
from dataset import find_file_list
from settings import BH, COHFACE, PURE, TOKYOTECH, UBFC, UBFC_PHYS


CHECKPOINTS_TO_EVALUATE = ("PURE_CHECKPOINT", "UBFC_CHECKPOINT")
DATASETS_TO_EVALUATE = ("PURE", "UBFC", "TOKYOTECH", "BH", "UBFC_PHYS", "COHFACE")
PROTOCOLS_TO_EVALUATE = ("official_mamba", "old", "prism")

# Switch plots off only for a quick computational trial. Tables are always saved.
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
    "PURE_CHECKPOINT": (PURE.final_checkpoint, "PURE"),
    "UBFC_CHECKPOINT": (UBFC.final_checkpoint, "UBFC"),
}


def split_for(checkpoint_dataset: str, target_name: str) -> tuple[float, float]:
    if checkpoint_dataset == target_name:
        experiment = EXPERIMENTS[target_name]
        return experiment.test_begin, experiment.test_end
    return 0.0, 1.0


def validate() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RhythmMamba evaluation")
    for checkpoint_name in CHECKPOINTS_TO_EVALUATE:
        path, _ = CHECKPOINTS[checkpoint_name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing local trained checkpoint: {path}")
    for checkpoint_name in CHECKPOINTS_TO_EVALUATE:
        _, source_dataset = CHECKPOINTS[checkpoint_name]
        for target_name in DATASETS_TO_EVALUATE:
            begin, end = split_for(source_dataset, target_name)
            find_file_list(EXPERIMENTS[target_name], begin, end)


def main() -> None:
    validate()
    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    core.write_output_guide(core.OUTPUT_ROOT / "README.txt")
    all_summaries = []

    for checkpoint_name in CHECKPOINTS_TO_EVALUATE:
        checkpoint, source_dataset = CHECKPOINTS[checkpoint_name]
        print("=" * 84)
        print(f"LOADING {checkpoint_name}: {checkpoint}")
        print("=" * 84)
        model = core.load_model(checkpoint)

        for target_name in DATASETS_TO_EVALUATE:
            experiment = EXPERIMENTS[target_name]
            begin, end = split_for(source_dataset, target_name)
            run = core.EvaluationRun(
                name=f"{checkpoint_name}/Eval_On_{target_name}",
                checkpoint=checkpoint,
                experiment=experiment,
                split_begin=begin,
                split_end=end,
                description=(
                    f"Locally trained {source_dataset} checkpoint evaluated on "
                    f"{target_name} ({'native held-out split' if source_dataset == target_name else 'complete external dataset'})"
                ),
            )
            file_list = find_file_list(experiment, begin, end)
            recordings = core.read_manifest(file_list)
            run_summaries = []
            for protocol_name in PROTOCOLS_TO_EVALUATE:
                summary = core.run_protocol(
                    model, run, core.PROTOCOLS[protocol_name], file_list, recordings
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
    core.write_csv(core.OUTPUT_ROOT / "all_results_summary_2x6x3.csv", all_summaries)
    print("=" * 84)
    print("2 x 6 x 3 EVALUATION COMPLETED")
    print(f"Summary: {core.OUTPUT_ROOT / 'all_results_summary.csv'}")


if __name__ == "__main__":
    main()
