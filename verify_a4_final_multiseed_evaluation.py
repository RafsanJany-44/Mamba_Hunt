"""Verify checkpoints, metadata, manifests, and evaluation cardinality."""

from evaluate_a4_final_multiseed import (
    DATASETS_TO_EVALUATE,
    PROTOCOLS_TO_EVALUATE,
    SEEDS_TO_EVALUATE,
    checkpoint_name,
    core,
    validate,
)


def main() -> None:
    validate("PURE")
    validate("UBFC")
    names = {
        checkpoint_name(source, seed)
        for source in ("PURE", "UBFC")
        for seed in SEEDS_TO_EVALUATE
    }
    if len(names) != 8:
        raise RuntimeError("Expected eight unique checkpoint names")
    per_source = (
        len(SEEDS_TO_EVALUATE)
        * len(DATASETS_TO_EVALUATE)
        * len(PROTOCOLS_TO_EVALUATE)
    )
    if per_source != 72:
        raise RuntimeError(f"Expected 72 setups per source, found {per_source}")
    print("=" * 88)
    print("A4 FINAL MULTI-SEED EVALUATION VERIFICATION: PASSED")
    print("=" * 88)
    print("Checkpoints       : 8")
    print("Seeds             : 101, 102, 103, 104")
    print("Setups per source : 72")
    print("Total setups      : 144")
    print(f"Output folder     : {core.OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
