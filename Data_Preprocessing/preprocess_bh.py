"""Preprocess only BH-rPPG using settings.py."""
try:
    from .common import run_dataset
except ImportError:
    from common import run_dataset

if __name__ == "__main__":
    run_dataset("BH")
