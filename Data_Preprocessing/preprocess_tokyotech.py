"""Preprocess only TokyoTech after the synchronization audit is accepted."""
try:
    from .common import run_dataset
except ImportError:
    from common import run_dataset

if __name__ == "__main__":
    run_dataset("TOKYOTECH")
