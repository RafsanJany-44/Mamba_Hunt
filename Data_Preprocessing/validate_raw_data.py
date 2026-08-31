"""Fast native-layout validation that does not preprocess complete recordings."""

try:
    from .dataset_registry import get_adapter
    from .settings import DATASET_SETTINGS, DATASETS_TO_PROCESS
except ImportError:
    from dataset_registry import get_adapter
    from settings import DATASET_SETTINGS, DATASETS_TO_PROCESS


def main() -> None:
    for dataset_name in DATASETS_TO_PROCESS:
        adapter = get_adapter(dataset_name)
        dataset_settings = DATASET_SETTINGS[dataset_name]
        recordings = adapter.discover(dataset_settings.raw_root)
        if not recordings:
            raise RuntimeError(
                f"No {dataset_name} recordings found in {dataset_settings.raw_root}"
            )
        probe = adapter.probe(recordings[0])

        print("=" * 70)
        print(f"{dataset_name} RAW-DATA VALIDATION")
        print("=" * 70)
        print(f"Raw root         : {dataset_settings.raw_root}")
        print(f"Recordings found : {len(recordings)}")
        for key, value in probe.items():
            print(f"{key.replace('_', ' ').title():17}: {value}")
        print(f"{dataset_name} raw-data validation: PASSED")


if __name__ == "__main__":
    main()

