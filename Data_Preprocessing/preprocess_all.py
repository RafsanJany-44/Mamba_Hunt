"""Preprocess every dataset selected in settings.py."""

try:
    from .common import run_dataset
    from .settings import DATASETS_TO_PROCESS, RUN_MODE, selected_cache_root
except ImportError:
    from common import run_dataset
    from settings import DATASETS_TO_PROCESS, RUN_MODE, selected_cache_root


def main() -> None:
    print("=" * 70)
    print("MAMBA_HUNT INDEPENDENT PREPROCESSING")
    print("=" * 70)
    print(f"Run mode   : {RUN_MODE}")
    print(f"Datasets   : {DATASETS_TO_PROCESS}")
    print(f"Output root: {selected_cache_root()}")

    summaries = [run_dataset(name) for name in DATASETS_TO_PROCESS]

    print("=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    for summary in summaries:
        print(
            f"{summary['dataset']}: {summary['recordings']} recordings, "
            f"{summary['clips']} clips"
        )
    print("Independent preprocessing completed successfully.")


if __name__ == "__main__":
    main()

