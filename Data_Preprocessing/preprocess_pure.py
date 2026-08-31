"""Preprocess PURE using only the independent Mamba_Hunt package."""

try:
    from .common import run_dataset
except ImportError:
    from common import run_dataset


if __name__ == "__main__":
    run_dataset("PURE")

