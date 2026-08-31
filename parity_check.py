"""Check the simplified model and loss against the pinned official source."""

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from loss import HybridLoss
from model import RhythmMamba as SimplifiedRhythmMamba
from settings import DEVICE, SEED
from trainer import normalize_prediction, set_reproducible


# Only this optional development check refers to the official source.
# Training and inference do not use this path.
OFFICIAL_ROOT = Path(
    "/media/data/rPPG/Code/GitHub/Catch_The_Mamba/official/RhythmMamba"
)


def load_module(module_name, file_path):
    specification = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    if not OFFICIAL_ROOT.is_dir():
        raise FileNotFoundError(f"Official submodule was not found: {OFFICIAL_ROOT}")

    # Needed by the official loss module's absolute evaluation import.
    sys.path.insert(0, str(OFFICIAL_ROOT))
    official_model_module = load_module(
        "official_rhythmmamba_model",
        OFFICIAL_ROOT / "neural_methods" / "model" / "RhythmMamba.py",
    )
    official_loss_module = load_module(
        "official_rhythmmamba_loss",
        OFFICIAL_ROOT / "neural_methods" / "loss" / "TorchLossComputer.py",
    )

    set_reproducible(SEED)
    official_model = official_model_module.RhythmMamba().to(DEVICE).eval()
    simplified_model = SimplifiedRhythmMamba().to(DEVICE).eval()
    simplified_model.load_state_dict(official_model.state_dict(), strict=True)

    official_parameters = sum(p.numel() for p in official_model.parameters())
    simplified_parameters = sum(p.numel() for p in simplified_model.parameters())

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    video = torch.randn(
        1, 160, 3, 128, 128, generator=generator, device=DEVICE
    )

    with torch.no_grad():
        official_output = official_model(video)
        simplified_output = simplified_model(video)

    maximum_output_difference = torch.max(
        torch.abs(official_output - simplified_output)
    ).item()

    time = torch.arange(160, device=DEVICE, dtype=torch.float32) / 30
    label = torch.sin(2 * torch.pi * 1.2 * time)
    official_prediction = normalize_prediction(official_output)[0]
    simplified_prediction = normalize_prediction(simplified_output)[0]

    official_loss = official_loss_module.Hybrid_Loss()(
        official_prediction, label, 0, 30, False
    )
    simplified_loss = HybridLoss()(
        simplified_prediction, label, 0, 30, False
    )
    loss_difference = abs(official_loss.item() - simplified_loss.item())

    print("=" * 70)
    print("OFFICIAL vs SIMPLIFIED PARITY CHECK")
    print("=" * 70)
    print(f"Official parameters       : {official_parameters:,}")
    print(f"Simplified parameters     : {simplified_parameters:,}")
    print(f"Official output shape     : {tuple(official_output.shape)}")
    print(f"Simplified output shape   : {tuple(simplified_output.shape)}")
    print(f"Maximum output difference : {maximum_output_difference:.12g}")
    print(f"Official loss             : {official_loss.item():.12g}")
    print(f"Simplified loss           : {simplified_loss.item():.12g}")
    print(f"Loss difference           : {loss_difference:.12g}")

    passed = (
        official_parameters == simplified_parameters
        and official_output.shape == simplified_output.shape
        and maximum_output_difference <= 1e-7
        and loss_difference <= 1e-6
    )
    if not passed:
        raise AssertionError("Parity check FAILED")
    print("\nSimplified RhythmMamba parity check: PASSED")


if __name__ == "__main__":
    main()
