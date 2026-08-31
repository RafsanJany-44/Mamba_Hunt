"""Evaluate the validation-selected PURE checkpoint on COHFACE only."""

from __future__ import annotations

import torch

import eval_protocols as core
from dataset import find_file_list
from settings import COHFACE, PURE


CHECKPOINT = PURE.model_dir / "PURE_RhythmMamba_Best.pth"
PROTOCOLS = ("official_mamba", "old", "prism")

# Keep this diagnosis separate from the completed 2 x 6 x 3 results.
core.OUTPUT_ROOT = (
    core.MAMBA_HUNT_ROOT / "results" / "checkpoint_diagnosis"
)

# Tables and summaries are sufficient for this checkpoint diagnosis.
core.GENERATE_SIGNAL_PLOTS = False
core.GENERATE_PSD_DIAGNOSTICS = False
core.GENERATE_SUMMARY_PLOTS = False
core.SAVE_SIGNAL_SAMPLE_TABLES = False


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RhythmMamba evaluation.")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing best PURE checkpoint: {CHECKPOINT}")

    file_list = find_file_list(COHFACE, 0.0, 1.0)
    recordings = core.read_manifest(file_list)
    model = core.load_model(CHECKPOINT)

    run = core.EvaluationRun(
        name="PURE_BEST_EPOCH22/Eval_On_COHFACE",
        checkpoint=CHECKPOINT,
        experiment=COHFACE,
        split_begin=0.0,
        split_end=1.0,
        description=(
            "PURE 0.0-0.6 model selected by the lowest PURE 0.6-1.0 "
            "validation loss, evaluated on complete COHFACE"
        ),
    )

    summaries = []
    for protocol_name in PROTOCOLS:
        summary = core.run_protocol(
            model,
            run,
            core.PROTOCOLS[protocol_name],
            file_list,
            recordings,
        )
        summaries.append(summary)

    comparison = core.OUTPUT_ROOT / run.name / "protocol_comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    core.write_csv(comparison / "PROTOCOL_COMPARISON.csv", summaries)
    core.write_protocol_comparison_note(comparison / "INTERPRETATION.txt")
    core.write_csv(core.OUTPUT_ROOT / "PURE_BEST_ON_COHFACE.csv", summaries)

    print("=" * 84)
    print("PURE BEST-CHECKPOINT -> COHFACE DIAGNOSIS COMPLETED")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Summary   : {core.OUTPUT_ROOT / 'PURE_BEST_ON_COHFACE.csv'}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
