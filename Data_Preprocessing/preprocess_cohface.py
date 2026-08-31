"""Preprocess only COHFACE using settings.py."""
try:
    from .common import run_dataset
except ImportError:
    from common import run_dataset

if __name__ == "__main__":
    run_dataset("COHFACE")
