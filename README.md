# Mamba Hunt: Simplified RhythmMamba Baseline (Version 1.0.0)


# This version contain the offline and + light online augmentaiton

Mamba Hunt contains the code required to train and evaluate RhythmMamba using the following datasets:

* PURE
* UBFC-rPPG
* BH-rPPG
* UBFC-PHYS
* COHFACE
* TokyoTech

The project uses a flat structure without YAML configuration files or command-line argument parsing. Editable paths and experimental settings are defined directly in the Python configuration files.

## Important Baseline Rule

Do not modify the official RhythmMamba implementation while verifying the baseline.

First establish parity between this simplified implementation and the pinned official implementation. Architecture changes, augmentation improvements, and other model corrections belong to the later development phase.

## Project Structure

```text
Mamba_Hunt/
├── augmentation.py
├── create_official_cross_manifests.py
├── cross_settings.py
├── cross_trainer.py
├── dataset.py
├── eval_protocols.py
├── evaluate_2x6x3.py
├── evaluate_cross_matched_2x6x3.py
├── evaluate_official_2x6x3.py
├── evaluate_pure.py
├── evaluate_pure_best_on_cohface.py
├── evaluate_ubfc.py
├── generate_harmonic_error_tables.py
├── generate_mae_analysis.py
├── infer_pure_to_ubfc.py
├── infer_ubfc_to_pure.py
├── loss.py
├── metrics.py
├── model.py
├── parity_check.py
├── select_best_pure_checkpoint.py
├── settings.py
├── trainer.py
├── train_pure.py
├── train_pure_cross_matched.py
├── train_ubfc.py
├── train_ubfc_cross_matched.py
├── train_ubfc_cross_matched_seed101.py
├── train_ubfc_cross_matched_seed102.py
├── Data_Preprocessing/
├── Official_Checkpoints/
└── results → external results folder
```

## File Roles

* `settings.py`: dataset paths, cache paths, training settings, and hyperparameters.
* `cross_settings.py`: settings for matched cross-dataset training.
* `dataset.py`: cached `.npy` clip loader and dataset file-list handling.
* `model.py`: public RhythmMamba architecture from commit `1533ad2`.
* `augmentation.py`: official RhythmMamba training augmentation.
* `loss.py`: negative Pearson and frequency-domain training objectives.
* `metrics.py`: FFT-based MAE, RMSE, MAPE, Pearson correlation, and SNR.
* `trainer.py`: standard training, checkpoint saving, and evaluation.
* `cross_trainer.py`: matched cross-dataset training and validation-based checkpoint selection.
* `parity_check.py`: fixed-weight comparison between the official and simplified implementations.
* `train_*.py`: training entry points.
* `evaluate_*.py`: evaluation entry points.
* `infer_pure_to_ubfc.py`: evaluates UBFC using the released PURE cross checkpoint.
* `infer_ubfc_to_pure.py`: evaluates PURE using the released UBFC cross checkpoint.
* `evaluate_2x6x3.py`: evaluates two local checkpoints on six datasets using three protocols.
* `evaluate_official_2x6x3.py`: evaluates the released checkpoints on six datasets using three protocols.
* `evaluate_cross_matched_2x6x3.py`: evaluates the locally reproduced cross-matched checkpoints.
* `generate_harmonic_error_tables.py`: generates harmonic-error tables from available evaluation folders.
* `generate_mae_analysis.py`: generates consolidated MAE tables and visual reports.
* `Data_Preprocessing/`: independent preprocessing code and dataset metadata.
* `Official_Checkpoints/`: released RhythmMamba checkpoints required for official comparison.

## External Results Folder

Generated results are stored outside the code repository. This keeps the GitHub code repository lightweight while preserving all checkpoints, plots, diagnostics, signal tables, and evaluation outputs locally.

The scripts continue to use the relative path:

```text
results/
```

A symbolic link or directory junction connects this name to the external results folder.

### External results structure

```text
results/
├── checkpoint_diagnosis/
├── error_analysis/
├── evaluation_protocols/
├── evaluation_protocols_cross_matched/
├── evaluation_protocols_official/
├── logs/
├── mae_analysis/
├── models/
└── preprocessing_checks/
```

## Connecting the Results Folder on Linux

The current external results location is:

```text
/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Mamba_Results/results
```

Open a terminal and run:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt

ln -s \
  /media/data/rPPG/Code/GitHub/Project_rPPG_Result/Mamba_Results/results \
  results
```

Verify the connection:

```bash
readlink -f results
```

Expected output:

```text
/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Mamba_Results/results
```

You can also verify that the existing result folders are visible:

```bash
ls results
```

## Connecting the Results Folder on Windows

Open PowerShell inside the `Mamba_Hunt` repository.

Run:

```powershell
New-Item -ItemType Junction `
  -Path ".\results" `
  -Target "D:\path\to\Mamba_Results\results"
```

Replace:

```text
D:\path\to\Mamba_Results\results
```

with the actual Windows location of the external results folder.

Example:

```powershell
New-Item -ItemType Junction `
  -Path ".\results" `
  -Target "D:\rPPG\Project_rPPG_Result\Mamba_Results\results"
```

Verify the connection:

```powershell
(Get-Item ".\results").Target
```

Verify that the existing results are accessible:

```powershell
Get-ChildItem ".\results"
```

## Important Results-Folder Rules

* Do not create a second physical `results` folder inside the code repository.
* Do not move or rename the external results folder after creating the connection.
* Do not commit the `results` link or its contents to the code repository.
* The external results are not copied into the code repository.
* Existing results remain in their original external location.
* New training and evaluation outputs are automatically written to the external results folder.
* Relative paths such as `results/models` and `results/evaluation_protocols` work without modification.

## Git Exclusion

Create a `.gitignore` file in the repository root and include:

```gitignore
/results
/push_diagnostics/
/__pycache__/
/Temp/
*.pyc
```

Verify the Git status:

```bash
git status --short
```

The external `results` link and the files inside it should not appear in the Git status.

## Environment

Activate the project environment before running any experiment:

```bash
conda activate mamba_hunting
```

Then enter the repository:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
```

## Official Checkpoints

The following released checkpoints are required:

```text
Official_Checkpoints/
├── PURE_cross_RhythmMamba.pth
└── UBFC_cross_RhythmMamba.pth
```

If the official RhythmMamba repository is available locally, copy the checkpoints into this repository once:

```bash
cp /path/to/official/RhythmMamba/PreTrainedModels/PURE_cross_RhythmMamba.pth \
  Official_Checkpoints/

cp /path/to/official/RhythmMamba/PreTrainedModels/UBFC_cross_RhythmMamba.pth \
  Official_Checkpoints/
```

After copying them, official-checkpoint inference does not depend on the official code directory.

## Baseline Verification

Run the parity test before starting full training:

```bash
conda activate mamba_hunting
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt

python parity_check.py
```

Continue only after the parity check passes.

## Standard Training

Train the original intra-dataset PURE model:

```bash
python train_pure.py 2>&1 | tee results/logs/PURE_training.log
```

Train the original intra-dataset UBFC model:

```bash
python train_ubfc.py 2>&1 | tee results/logs/UBFC_training.log
```

## Official-Matched Cross Training

Create the official 80/20 training and validation manifests:

```bash
python create_official_cross_manifests.py
```

Train the PURE cross-matched model:

```bash
python train_pure_cross_matched.py \
  2>&1 | tee results/logs/PURE_CROSS_MATCHED_training.log
```

Train the UBFC cross-matched model:

```bash
python train_ubfc_cross_matched.py \
  2>&1 | tee results/logs/UBFC_CROSS_MATCHED_training.log
```

Optional UBFC repeated-seed experiments:

```bash
python train_ubfc_cross_matched_seed101.py \
  2>&1 | tee results/logs/UBFC_CROSS_MATCHED_SEED101_training.log

python train_ubfc_cross_matched_seed102.py \
  2>&1 | tee results/logs/UBFC_CROSS_MATCHED_SEED102_training.log
```

## Standalone Cross-Dataset Inference

Evaluate UBFC using the released PURE cross checkpoint:

```bash
python infer_pure_to_ubfc.py
```

Evaluate PURE using the released UBFC cross checkpoint:

```bash
python infer_ubfc_to_pure.py
```

## Complete Evaluation

Evaluate the original locally trained checkpoints:

```bash
python evaluate_2x6x3.py
```

Evaluate the released official checkpoints:

```bash
python evaluate_official_2x6x3.py
```

Evaluate the locally reproduced cross-matched checkpoints:

```bash
python evaluate_cross_matched_2x6x3.py
```

These evaluations cover:

```text
2 checkpoints × 6 datasets × 3 evaluation protocols
```

The three evaluation protocols are:

* Official complete-recording protocol
* Old 8-second window protocol
* PRISM 10-second window protocol

## Analysis Generation

Generate the harmonic-error tables:

```bash
python generate_harmonic_error_tables.py
```

Generate the consolidated MAE tables and report:

```bash
python generate_mae_analysis.py
```

Generated analysis files are written to:

```text
results/error_analysis/
results/mae_analysis/
```

Because `results` is linked to the external results location, these files are stored outside the code repository automatically.

## Notes

* Dataset caches are not included in this repository.
* Raw datasets are not included in this repository.
* Full generated results are maintained separately from the source code.
* Official checkpoints are kept in `Official_Checkpoints/`.
* Locally trained usable checkpoints are stored under `results/models/`.
* Per-epoch checkpoints, plots, diagnostics, and signal tables remain in the external results folder.
* Do not modify the baseline architecture until parity and reproduction experiments are complete.
