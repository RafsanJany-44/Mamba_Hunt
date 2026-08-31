"""Read-only discovery and approximate cache-space estimate."""

from __future__ import annotations

import shutil

try:
    from .dataset_registry import get_adapter
    from .settings import DATASET_SETTINGS, DATASETS_TO_PROCESS, FULL_CACHE_ROOT, TRANSFORM
except ImportError:
    from dataset_registry import get_adapter
    from settings import DATASET_SETTINGS, DATASETS_TO_PROCESS, FULL_CACHE_ROOT, TRANSFORM


KNOWN_SECONDS = {
    "PURE": 59 * 60,
    "UBFC": 42 * 60,
    "BH": 105 * 60,
    "UBFC_PHYS": 60 * 180,
    "COHFACE": 160 * 63,
    "TOKYOTECH": 9 * 180,
}


def main() -> None:
    total = 0.0
    print("=" * 70)
    print("MULTI-DATASET PREPROCESSING PREFLIGHT")
    print("=" * 70)
    for name in DATASETS_TO_PROCESS:
        settings = DATASET_SETTINGS[name]
        recordings = get_adapter(name).discover(settings.raw_root)
        seconds = KNOWN_SECONDS.get(name, 0)
        bytes_per_sample = 8 if settings.cache_dtype == "float64" else 4
        estimate = seconds * 30 * TRANSFORM.height * TRANSFORM.width * 3 * bytes_per_sample
        estimate *= 1.02
        total += estimate
        print(f"{name:12}: {len(recordings):4d} recordings | estimated {estimate / 2**30:7.2f} GiB")
    free = shutil.disk_usage(FULL_CACHE_ROOT.parent).free
    print(f"Total estimate: {total / 2**30:.2f} GiB")
    print(f"Free space    : {free / 2**30:.2f} GiB")
    if free < total * 1.15:
        raise RuntimeError("Insufficient safety margin. Process fewer datasets or free disk space.")
    print("Preflight: PASSED")


if __name__ == "__main__":
    main()
