# Mamba_Hunt verified execution guide

This guide documents the source snapshot in `Mamba_Hunt_Complete_Source.tar.gz`. It explains how a colleague can run the experiments that already exist in the repository on the same machine. It does not introduce a new experiment or change the implementation.

## 1. Scope and terminology

- **Released checkpoint** means `Official_Checkpoints/PURE_cross_RhythmMamba.pth` or `Official_Checkpoints/UBFC_cross_RhythmMamba.pth`.
- **Official-complete evaluation protocol** means the complete-recording RhythmMamba/rPPG-Toolbox-compatible evaluation implemented as `official_mamba`.
- **Old protocol** means overlapping 8-second windows with 1-second stride.
- **PRISM protocol** means non-overlapping 10-second windows.
- **A0** is the verified baseline augmentation/loss setup.
- **A1** adds online RGB-channel gain.
- **A2** uses one stored offline variant per source clip with 50% original/offline sampling.
- **A3** combines A2 with online RGB-channel gain.
- **A4** stores four variants per source clip and samples 50% original plus 12.5% each JPEG, blur, gamma, and contrast.

All evaluation families use the same six targets: PURE, UBFC-rPPG, BH-rPPG, UBFC-PHYS, COHFACE, and TokyoTech.

## 2. Verified archive status

The supplied archive contains 141 files. It intentionally excludes:

- Git history (`.git/`)
- the linked `results/` data
- released and locally trained `.pth` checkpoints
- cached `.npy`/`.npz` arrays

Therefore, source code is present, but evaluation cannot start until the required checkpoints and caches exist at the paths described below.

## 3. Same-machine paths

Repository:

```text
/media/data/rPPG/Code/GitHub/Mamba_Hunt
```

External results directory:

```text
/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Result_Lab_Mamba_Results/results
```

Preprocessed caches configured in `settings.py`:

```text
/home/rafsan/Documents/Data/Mamba_Hunt_Data/RhythmMamba_Preprocessed_Independent
```

Offline A2 cache root configured in `offline_augmentation_generator.py`:

```text
/home/rafsan/Documents/Data/Mamba_Hunt_Data/RhythmMamba_Offline_Augmentation
```

Offline A4 cache root used by the A4 generators/loaders:

```text
/home/rafsan/Documents/Data/Mamba_Hunt_Data/RhythmMamba_Offline_Augmentation_A4
```

The older path printed in the repository's current `README.md`, ending in `Mamba_Results/results`, is stale. Use the `Result_Lab_Mamba_Results/results` path above.

## 4. One-time startup checks

Run all commands from the repository root:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
```

Confirm the external results link:

```bash
ls -ld results
readlink -f results
```

The second command must print:

```text
/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Result_Lab_Mamba_Results/results
```

If the link is missing, recreate it safely:

```bash
RESULT_ROOT=/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Result_Lab_Mamba_Results/results
mkdir -p "$RESULT_ROOT"
ln -s "$RESULT_ROOT" results
```

If a `results` path already exists, inspect it before changing it. Never recursively delete `results` without first running `readlink -f results`, because it is a link to the external results store.

Check the six cache directories:

```bash
for dataset in PURE UBFC BH UBFC_PHYS COHFACE TOKYOTECH; do
  test -d "/home/rafsan/Documents/Data/Mamba_Hunt_Data/RhythmMamba_Preprocessed_Independent/$dataset" \
    && echo "PASS $dataset" || echo "MISSING $dataset"
done
```

Check the released checkpoints:

```bash
ls -lh Official_Checkpoints/PURE_cross_RhythmMamba.pth \
       Official_Checkpoints/UBFC_cross_RhythmMamba.pth
```

Create the log directory and preserve pipeline failure codes:

```bash
mkdir -p results/logs
set -o pipefail
```

`set -o pipefail` makes a command such as `python ... | tee ...` fail when Python fails, rather than reporting only the successful exit status of `tee`.

## 5. Core fixed configuration

The baseline values in `settings.py` are:

| Setting | Value |
|---|---:|
| Device inside process | `cuda:0` |
| Sampling rate | 30 Hz |
| Clip length | 160 frames |
| Frame size | 128 × 128 |
| Batch size | 16 |
| Learning rate | `3e-4` |
| Baseline epochs | 30 |
| Baseline seed | 100 |
| Baseline online augmentation | enabled |

`CUDA_VISIBLE_DEVICES=0` or `1` exposes one physical GPU to the process; inside that process it is still addressed as `cuda:0`.

## 6. Preprocessing workflow

Only run preprocessing when a required cache is absent. Raw dataset paths and preprocessing controls are in `Data_Preprocessing/settings.py`.

Safe order:

```bash
python Data_Preprocessing/validate_raw_data.py
python Data_Preprocessing/preflight_multidataset.py
python Data_Preprocessing/preprocess_all.py
python Data_Preprocessing/validate_cache.py
```

Before `preprocess_all.py`, select the desired dataset(s) and `RUN_MODE` in `Data_Preprocessing/settings.py`. Start with `RUN_MODE = "smoke"`; change to `"full"` only after smoke validation.

Dataset-specific entry points are:

```text
preprocess_pure.py
preprocess_ubfc.py
preprocess_bh.py
preprocess_ubfc_phys.py
preprocess_cohface.py
preprocess_tokyotech.py
```

TokyoTech has an additional synchronization gate:

```bash
python Data_Preprocessing/audit_tokyotech_sync.py
```

Review its CSV/JSON audit before enabling `TOKYOTECH_ACCEPT_AUDIT_RECOMMENDATIONS` in `Data_Preprocessing/settings.py`.

Explicit preprocessing dependencies are recorded in `Data_Preprocessing/requirements.txt`:

```text
numpy==1.26.4
opencv-python
tqdm
scipy==1.11.4
h5py
```

The complete project additionally imports PyTorch, `mamba_ssm`, `timm`, `einops`, and Plotly. The source snapshot does not contain a verified root-level environment lock file, so do not claim that a fresh Windows/Linux installation is reproducible from this archive alone. On the current machine, use the existing `mamba_hunting` environment.

## 7. Experiment execution map

The following sections are ordered historically. Each training section states the checkpoint location required by its evaluation.

### 7.1 Baseline parity and original intra-dataset models

Purpose: verify the simplified architecture and train the original source-specific intra-dataset models.

Splits:

- PURE: train 0.0–0.6; test 0.6–1.0.
- UBFC: train 0.0–0.72; test 0.72–1.0.

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python train_pure.py 2>&1 \
  | tee results/logs/PURE_training.log
```
```bash
CUDA_VISIBLE_DEVICES=0 python train_ubfc.py 2>&1 \
  | tee results/logs/UBFC_training.log
```

The baseline trainer saves every epoch under `results/models/PURE/` and `results/models/UBFC/`. The standard final checkpoints are epoch 29.

Optional native evaluations:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_pure.py
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc.py
```

Full two-checkpoint evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_2x6x3.py 2>&1 \
  | tee results/logs/evaluate_2x6x3.log
```

Output:

```text
results/evaluation_protocols/
```

### 7.2 Released-checkpoint evaluation

Purpose: evaluate the two released cross-dataset checkpoints with the same six datasets and three protocols.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_official_2x6x3.py 2>&1 \
  | tee results/logs/evaluate_released_checkpoints_2x6x3.log
```

Output:

```text
results/evaluation_protocols_official/
```

Standalone released-checkpoint inference entry points:

```bash
python infer_pure_to_ubfc.py
python infer_ubfc_to_pure.py
```

### 7.3 Locally reproduced 80/20 cross-matched models

Purpose: match the source-data split and validation-based checkpoint selection used for cross-dataset training.

```bash
python create_official_cross_manifests.py

CUDA_VISIBLE_DEVICES=1 python train_pure_cross_matched.py 2>&1 \
  | tee results/logs/PURE_CROSS_MATCHED_training.log

CUDA_VISIBLE_DEVICES=0 python train_ubfc_cross_matched.py 2>&1 \
  | tee results/logs/UBFC_CROSS_MATCHED_training.log
```

Training uses source 0.0–0.8 and clean validation 0.8–1.0 for 30 epochs, retaining the lowest-validation-loss checkpoint.

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_cross_matched_2x6x3.py 2>&1 \
  | tee results/logs/evaluate_cross_matched_2x6x3.log
```

Output:

```text
results/evaluation_protocols_cross_matched/
```

### 7.4 PURE checkpoint-selection diagnosis

Purpose: select the best of the 30 original PURE epochs using only held-out PURE data, then test that checkpoint on COHFACE.

```bash
CUDA_VISIBLE_DEVICES=1 python select_best_pure_checkpoint.py
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_best_on_cohface.py
```

Outputs:

```text
results/models/PURE/PURE_RhythmMamba_Best.pth
results/models/PURE/PURE_epoch_validation_results.csv
results/checkpoint_diagnosis/
```

### 7.5 UBFC cross-matched seed stability

Purpose: add seeds 101 and 102 to the seed-100 cross-matched UBFC model.

```bash
CUDA_VISIBLE_DEVICES=0 python train_ubfc_cross_matched_seed101.py 2>&1 \
  | tee results/logs/UBFC_CROSS_MATCHED_SEED101_training.log
CUDA_VISIBLE_DEVICES=0 python train_ubfc_cross_matched_seed102.py 2>&1 \
  | tee results/logs/UBFC_CROSS_MATCHED_SEED102_training.log

CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_seed_stability_3x6x3.py 2>&1 \
  | tee results/logs/UBFC_cross_matched_seed_stability_evaluation.log
```

Output:

```text
results/evaluation_protocols_ubfc_seed_stability/
```

### 7.6 Stage 1: A1/A2/A3 augmentation ablation

First generate one deterministic offline variant per training clip for PURE and UBFC:

```bash
python generate_pure_offline_augmentation.py
python generate_ubfc_offline_augmentation.py
```

Each original training clip receives exactly one of mild JPEG compression, Gaussian blur, gamma, or contrast. The generator validates shapes, finite values, and label identity and writes metadata/summary files.

Train the three configurations per source:

```bash
CUDA_VISIBLE_DEVICES=1 python train_pure_augmentation_stage1.py 2>&1 \
  | tee results/logs/PURE_augmentation_stage1.log
CUDA_VISIBLE_DEVICES=0 python train_ubfc_augmentation_stage1.py 2>&1 \
  | tee results/logs/UBFC_augmentation_stage1.log
```

Policy: 80/20 source split, seed 100, maximum 60 epochs, minimum 30, patience 10, clean validation, best checkpoint only.

Evaluate in parallel and finalize:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_augmentation_stage1_3x6x3.py 2>&1 \
  | tee results/logs/PURE_augmentation_stage1_evaluation.log
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_augmentation_stage1_3x6x3.py 2>&1 \
  | tee results/logs/UBFC_augmentation_stage1_evaluation.log
python finalize_augmentation_stage1_evaluation.py
```

Outputs:

```text
results/models/augmentation_stage1/
results/evaluation_protocols_augmentation_stage1/
```

### 7.7 Harmonic-aware loss experiment

Purpose: compare A0 and A2 with the first harmonic-aware extension.

```bash
python verify_harmonic_loss.py
CUDA_VISIBLE_DEVICES=1 python train_pure_harmonic_loss.py 2>&1 \
  | tee results/logs/PURE_harmonic_loss.log
CUDA_VISIBLE_DEVICES=0 python train_ubfc_harmonic_loss.py 2>&1 \
  | tee results/logs/UBFC_harmonic_loss.log
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_harmonic_loss.py
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_harmonic_loss.py
```

Outputs:

```text
results/models/harmonic_loss_stage1/
results/evaluation_protocols_harmonic_loss/
```

### 7.8 Stage 2: A0/A2 loss suite

Purpose: independently test L2–L5 under A0 and A2 for both source datasets.

| ID | Effective loss |
|---|---|
| L2 | `0.2 Pearson + 0.1 harmonic` |
| L3 | `0.2 Pearson + 1.0 concentration` |
| L4 | `0.2 Pearson + 1.0 concentration + 0.1 harmonic` |
| L5 | `0.2 Pearson + 1.0 CE + 1.0 concentration` |

```bash
python verify_loss_suite.py
CUDA_VISIBLE_DEVICES=1 python train_pure_loss_suite.py 2>&1 \
  | tee results/logs/PURE_loss_suite_stage2.log
CUDA_VISIBLE_DEVICES=0 python train_ubfc_loss_suite.py 2>&1 \
  | tee results/logs/UBFC_loss_suite_stage2.log
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_loss_suite_stage2.py
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_loss_suite_stage2.py
```

Outputs:

```text
results/models/loss_suite_stage2/
results/evaluation_protocols_loss_suite_stage2/
```

### 7.9 Stage 3: A2 conditional harmonic ranking

```bash
python verify_harmonic_rank_loss.py
CUDA_VISIBLE_DEVICES=1 python train_pure_harmonic_rank.py 2>&1 \
  | tee results/logs/PURE_harmonic_rank.log
CUDA_VISIBLE_DEVICES=0 python train_ubfc_harmonic_rank.py 2>&1 \
  | tee results/logs/UBFC_harmonic_rank.log

CUDA_VISIBLE_DEVICES=1 python evaluate_pure_harmonic_rank.py
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_harmonic_rank.py
```

Outputs:

```text
results/models/harmonic_rank_stage3/
results/evaluation_protocols_harmonic_rank_stage3/
```

### 7.10 UBFC A2 paired five-seed experiment

Purpose: compare `UBFC_A2_L0` and `UBFC_A2_L5_CE_CONCENTRATION` using seeds 100–104.

```bash
python verify_ubfc_a2_multiseed.py
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a2_l0_five_seeds.py 2>&1 \
  | tee results/logs/UBFC_A2_L0_five_seeds.log
CUDA_VISIBLE_DEVICES=1 python train_ubfc_a2_l5_five_seeds.py 2>&1 \
  | tee results/logs/UBFC_A2_L5_five_seeds.log

CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a2_l0_five_seeds.py
CUDA_VISIBLE_DEVICES=1 python evaluate_ubfc_a2_l5_five_seeds.py
```

Outputs:

```text
results/models/ubfc_a2_multiseed/
results/evaluation_protocols_ubfc_a2_multiseed/
```

### 7.11 A4 cache generation

A4 creates four offline versions of every training clip: JPEG, blur, gamma, and contrast.

```bash
python generate_pure_a4_offline_augmentation.py
python generate_ubfc_a4_offline_augmentation.py
```

Expected totals from the fixed 80/20 manifests:

| Source | Original training clips | A4 augmented clips |
|---|---:|---:|
| PURE | 596 | 2,384 |
| UBFC | 378 | 1,512 |

The training loaders do not use all five versions simultaneously. For each training row, they choose the original with probability 0.5 or one of the four A4 variants with probability 0.125 each. All 160 frames of the selected clip use the same selected version.

### 7.12 V8: A4 + L0 five-seed experiment

Purpose: evaluate A4 with the baseline L0 loss over seeds 100–104 for both source families.

PURE:

```bash
python verify_pure_a4_training.py
CUDA_VISIBLE_DEVICES=1 python train_pure_a4_l0_five_seeds.py 2>&1 \
  | tee results/logs/PURE_A4_L0_five_seeds.log
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_a4_l0_five_seeds.py 2>&1 \
  | tee results/logs/PURE_A4_L0_five_seed_evaluation.log
```

UBFC:

```bash
python verify_ubfc_a4_training.py
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_l0_seed100.py 2>&1 \
  | tee results/logs/UBFC_A4_L0_seed100.log
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_remaining_seeds.py 2>&1 \
  | tee results/logs/UBFC_A4_L0_seeds101_104.log
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a4_l0_five_seeds.py 2>&1 \
  | tee results/logs/UBFC_A4_L0_five_seed_evaluation.log
```

Outputs:

```text
results/models/pure_a4_pilot/
results/models/ubfc_a4_pilot/
results/evaluation_protocols_pure_a4_pilot/
results/evaluation_protocols_ubfc_a4_pilot/
```

The directory names retain the historical word `pilot`, even when they contain all five seeds.

### 7.13 V9: A4 + L1–L4 controlled loss comparison

Purpose: hold A4, seed 100, the 80/20 split, optimizer, and training policy fixed while changing the loss.

| ID | Effective loss |
|---|---|
| L1 | `0.2 Pearson + 1.0 CE + 1.0 harmonic` |
| L2 | `0.2 Pearson + 1.0 harmonic` |
| L3 | `0.2 Pearson + 1.0 concentration` |
| L4 | `0.2 Pearson + 1.0 concentration + 1.0 harmonic` |

```bash
python verify_a4_loss_suite.py
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_loss_suite.py 2>&1 \
  | tee results/logs/UBFC_A4_L1_L4_training.log
CUDA_VISIBLE_DEVICES=1 python train_pure_a4_loss_suite.py 2>&1 \
  | tee results/logs/PURE_A4_L1_L4_training.log

CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a4_loss_suite.py
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_a4_loss_suite.py
```

Outputs:

```text
results/models/a4_loss_suite/
results/evaluation_protocols_a4_loss_suite/
```

### 7.14 V10: A4 L3/L4 loss-weight sweep

Purpose: tune concentration and harmonic weights with seed 100.

| Experiment | Pearson | CE | Concentration | Harmonic |
|---|---:|---:|---:|---:|
| `L3_C010` | 0.2 | 0 | 0.10 | 0 |
| `L3_C025` | 0.2 | 0 | 0.25 | 0 |
| `L3_C050` | 0.2 | 0 | 0.50 | 0 |
| `L4_C025_H100` | 0.2 | 0 | 0.25 | 1.0 |
| `L4_C025_H200` | 0.2 | 0 | 0.25 | 2.0 |
| `L4_C050_H100` | 0.2 | 0 | 0.50 | 1.0 |
| `L4_C050_H200` | 0.2 | 0 | 0.50 | 2.0 |
| `L4_C100_H200` | 0.2 | 0 | 1.00 | 2.0 |

```bash
python verify_a4_loss_weight_tuning.py
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_loss_weight_tuning.py 2>&1 \
  | tee results/logs/UBFC_A4_loss_weight_tuning.log
CUDA_VISIBLE_DEVICES=1 python train_pure_a4_loss_weight_tuning.py 2>&1 \
  | tee results/logs/PURE_A4_loss_weight_tuning.log

CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a4_loss_weight_tuning.py
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_a4_loss_weight_tuning.py
```

Outputs:

```text
results/models/a4_loss_weight_tuning/
results/evaluation_protocols_a4_loss_weight_tuning/
```

### 7.15 V11: final A4 candidates over new seeds

Purpose: confirm the selected candidates with seeds 101–104. Seed 100 is reused from the earlier V9/V10 experiment and is intentionally not retrained here.

| Source | Candidate | Effective loss | Newly trained seeds |
|---|---|---|---|
| PURE | `PURE_A4_L4_C100_H100` | `0.2 Pearson + 1.0 concentration + 1.0 harmonic` | 101–104 |
| UBFC | `UBFC_A4_L3_C050` | `0.2 Pearson + 0.5 concentration` | 101–104 |

```bash
python verify_a4_final_multiseed.py
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_final_multiseed.py 2>&1 \
  | tee results/logs/UBFC_A4_L3_C050_seeds101_104.log
CUDA_VISIBLE_DEVICES=1 python train_pure_a4_final_multiseed.py 2>&1 \
  | tee results/logs/PURE_A4_L4_C100_H100_seeds101_104.log

python verify_a4_final_multiseed_evaluation.py
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a4_final_multiseed.py 2>&1 \
  | tee results/logs/UBFC_A4_final_multiseed_evaluation.log
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_a4_final_multiseed.py 2>&1 \
  | tee results/logs/PURE_A4_final_multiseed_evaluation.log
```

Outputs:

```text
results/models/a4_final_multiseed/
results/evaluation_protocols_a4_final_multiseed/
```

The V11 evaluation folder contains only seeds 101–104. The matching seed-100 evidence remains in the V9 PURE folder and V10 UBFC folder.

## 8. Evaluation output structure and reading results

Every multi-protocol evaluator writes a hierarchy resembling:

```text
results/evaluation_protocols_<experiment>/
└── <checkpoint>/
    └── Eval_On_<dataset>/
        ├── official_mamba/
        ├── old/
        └── prism/
```

The most important files are:

| File | Meaning |
|---|---|
| `tables/summary.json` | Macro MAE, MAE standard error, RMSE, accuracy within 5 BPM, counts, protocol identity, and completion status |
| `FAILURE_TYPE_SUMMARY.csv` | Counts of correct, 1.5×, 2×, 0.5×, 1/3×, and other-large-error cases |
| `all_results_summary_*.csv` | Combined summary produced by a specific evaluator |
| `protocol_comparison/` | Per-checkpoint protocol-comparison presentation outputs |
| `*_training_history.csv` | Epoch-level training/validation loss and component history |
| `*_configuration.json` | Fixed experimental configuration and effective loss weights |
| `*_completion.json` | Completion status, best epoch/loss, checkpoint path, and timing |

For the official-complete evaluation protocol, failure counts are recording counts. For Old and PRISM, they are window counts.

## 9. Generating MAE and harmonic-error CSV/HTML reports

Run only after the intended evaluation jobs finish:

```bash
python generate_mae_analysis.py
python generate_harmonic_error_tables.py
```

Outputs are versioned automatically:

```text
results/mae_analysis/MAE_ALL_SETUPS_vN.csv
results/mae_analysis/MAE_ANALYSIS_vN.html
results/error_analysis/HARMONIC_ERROR_COUNTS_ALL_SETUPS_vN.csv
results/error_analysis/HARMONIC_ERROR_ANALYSIS_vN.html
```

The scripts do not rerun inference:

- `generate_mae_analysis.py` reads every recognizable `summary.json` below `results`.
- `generate_harmonic_error_tables.py` discovers every top-level directory named `evaluation_protocols*` and reads each `FAILURE_TYPE_SUMMARY.csv` below it.

### Critical report-isolation rule

The generators do **not** automatically know which historical version is intended. If several `evaluation_protocols*` folders are present, their compatible rows are combined.

To create a report containing only one current experiment family, temporarily keep only that desired evaluation directory under the linked `results` root. Move other evaluation directories to a separate archive location; do not delete them unless they are independently backed up.

Example for V10 only:

```bash
RESULT_ROOT=/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Result_Lab_Mamba_Results/results
ARCHIVE_ROOT=/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Result_Lab_Mamba_Results/evaluation_archive
mkdir -p "$ARCHIVE_ROOT"

find -L "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d \
  -name 'evaluation_protocols*' \
  ! -name 'evaluation_protocols_a4_loss_weight_tuning' \
  -print
```

Inspect that printed list before moving anything. The source snapshot intentionally provides no automatic destructive cleanup utility.

The version number is selected from existing analysis filenames. Keeping `v9` outputs in `mae_analysis/` and `error_analysis/` causes the next successful generation to use `v10`; the existing files are not overwritten.

Validate a generated version:

```bash
ls -lh results/mae_analysis/MAE_ALL_SETUPS_vN.csv \
       results/mae_analysis/MAE_ANALYSIS_vN.html \
       results/error_analysis/HARMONIC_ERROR_COUNTS_ALL_SETUPS_vN.csv \
       results/error_analysis/HARMONIC_ERROR_ANALYSIS_vN.html
```

Open the HTML dashboards on Linux:

```bash
xdg-open results/mae_analysis/MAE_ANALYSIS_vN.html
xdg-open results/error_analysis/HARMONIC_ERROR_ANALYSIS_vN.html
```

## 10. Safe restart and overwrite behavior

- Most later trainers skip a model only when both its completion record and best checkpoint validate.
- A nonempty incomplete model directory generally causes a deliberate stop; preserve it for diagnosis rather than overwriting it.
- Later evaluators validate completed summaries and resume compatible work.
- Do not rename model directories: evaluators derive checkpoint identity and paths from the fixed experiment names.
- Do not select a checkpoint using an external target such as COHFACE; selection is based on the clean source validation split.
- Do not run two evaluators that write the same checkpoint subtree simultaneously.
- Training and evaluation outputs are physically written through the `results` link to the external results repository.

## 11. File-role catalog

### 11.1 Core model, data, and evaluation files

| File | Role |
|---|---|
| `settings.py` | Shared paths, six dataset definitions, baseline hyperparameters, and released-checkpoint paths |
| `cross_settings.py` | PURE/UBFC 80/20 cross-matched experiment objects |
| `dataset.py` | Cached clip loader and manifest discovery |
| `model.py` | Simplified RhythmMamba architecture |
| `augmentation.py` | Verified baseline online augmentation |
| `loss.py` | Baseline Pearson plus spectral CE objective |
| `metrics.py` | HR estimation and evaluation metrics |
| `trainer.py` | Baseline training and native evaluation engine |
| `cross_trainer.py` | 80/20 cross-training and best-validation checkpoint selection |
| `eval_protocols.py` | Shared six-dataset, three-protocol evaluation engine |
| `parity_check.py` | Simplified-versus-reference implementation parity check |
| `create_official_cross_manifests.py` | Creates and validates the 0.0–0.8/0.8–1.0 manifests |
| `select_best_pure_checkpoint.py` | Selects the best existing PURE epoch on held-out PURE |
| `generate_mae_analysis.py` | Auto-discovers evaluation summaries and produces versioned MAE CSV/HTML |
| `generate_harmonic_error_tables.py` | Auto-discovers failure summaries and produces versioned error CSV/HTML |

### 11.2 Baseline/cross entry points

| Files | Role |
|---|---|
| `train_pure.py`, `train_ubfc.py` | Original intra-dataset training wrappers |
| `evaluate_pure.py`, `evaluate_ubfc.py` | Original native held-out evaluation wrappers |
| `train_pure_cross_matched.py`, `train_ubfc_cross_matched.py` | 80/20 cross-matched training wrappers |
| `train_ubfc_cross_matched_seed101.py`, `train_ubfc_cross_matched_seed102.py` | Additional UBFC cross-matched seeds |
| `evaluate_2x6x3.py` | Original local checkpoints × six datasets × three protocols |
| `evaluate_official_2x6x3.py` | Released checkpoints × six datasets × three protocols |
| `evaluate_cross_matched_2x6x3.py` | Cross-matched checkpoints × six datasets × three protocols |
| `evaluate_ubfc_seed_stability_3x6x3.py` | Three UBFC cross-matched seeds × six datasets × three protocols |
| `evaluate_pure_best_on_cohface.py` | PURE best-epoch diagnostic on COHFACE |
| `infer_pure_to_ubfc.py`, `infer_ubfc_to_pure.py` | Standalone released-checkpoint cross inference |

### 11.3 A1/A2/A3 and harmonic-stage files

| Files | Role |
|---|---|
| `offline_augmentation_generator.py` | Shared one-of-four A2 offline generator |
| `generate_pure_offline_augmentation.py`, `generate_ubfc_offline_augmentation.py` | Dataset-specific A2 generation wrappers |
| `augmentation_stage1.py` | Online RGB-channel gain implementation |
| `dataset_stage1.py` | A0/A2 data selection loader |
| `trainer_augmentation_stage1.py` | A1/A2/A3 trainer |
| `train_pure_augmentation_stage1.py`, `train_ubfc_augmentation_stage1.py` | Stage-1 training wrappers |
| `evaluate_augmentation_stage1.py` | Shared Stage-1 evaluator |
| `evaluate_pure_augmentation_stage1_3x6x3.py`, `evaluate_ubfc_augmentation_stage1_3x6x3.py` | Source-specific Stage-1 evaluators |
| `finalize_augmentation_stage1_evaluation.py` | Validates and combines Stage-1 summaries |
| `loss_harmonic.py` | First harmonic-aware loss implementation |
| `trainer_harmonic_loss.py` | A0/A2 harmonic-aware trainer |
| `verify_harmonic_loss.py` | Harmonic loss behavior/parity/gradient checks |
| `train_pure_harmonic_loss.py`, `train_ubfc_harmonic_loss.py` | Harmonic-aware training wrappers |
| `evaluate_harmonic_loss.py`, `evaluate_pure_harmonic_loss.py`, `evaluate_ubfc_harmonic_loss.py` | Shared and source-specific harmonic evaluators |
| `loss_suite_stage2.py` | L2–L5 definitions and spectral components |
| `trainer_loss_suite.py` | A0/A2 L2–L5 trainer |
| `verify_loss_suite.py` | L0 parity, spectral behavior, and gradient verification |
| `train_pure_loss_suite.py`, `train_ubfc_loss_suite.py` | Stage-2 suite wrappers |
| `evaluate_loss_suite_stage2.py`, `evaluate_pure_loss_suite_stage2.py`, `evaluate_ubfc_loss_suite_stage2.py` | Stage-2 evaluation engine/wrappers |
| `harmonic_rank_loss.py` | Conditional harmonic-ranking objective |
| `trainer_harmonic_rank.py` | A2 harmonic-ranking trainer |
| `verify_harmonic_rank_loss.py` | Harmonic-ranking behavior and gradient verification |
| `train_pure_harmonic_rank.py`, `train_ubfc_harmonic_rank.py` | Stage-3 training wrappers |
| `evaluate_harmonic_rank.py`, `evaluate_pure_harmonic_rank.py`, `evaluate_ubfc_harmonic_rank.py` | Stage-3 evaluation engine/wrappers |

### 11.4 A2 multi-seed and A4 files

| Files | Role |
|---|---|
| `trainer_ubfc_a2_multiseed.py` | Paired UBFC A2 L0/L5 multi-seed trainer |
| `verify_ubfc_a2_multiseed.py` | Validates seed set, loaders, and experiment registry |
| `train_ubfc_a2_l0_five_seeds.py`, `train_ubfc_a2_l5_five_seeds.py` | Paired five-seed training wrappers |
| `evaluate_ubfc_a2_multiseed.py` | Shared UBFC A2 multi-seed evaluator |
| `evaluate_ubfc_a2_l0_five_seeds.py`, `evaluate_ubfc_a2_l5_five_seeds.py` | Variant-specific evaluation wrappers |
| `generate_pure_a4_offline_augmentation.py`, `generate_ubfc_a4_offline_augmentation.py` | Generate all four stored variants per source clip |
| `dataset_pure_a4.py`, `dataset_ubfc_a4.py` | A4 50/12.5/12.5/12.5/12.5 sampling loaders |
| `train_pure_a4_l0_seed100.py`, `train_ubfc_a4_l0_seed100.py` | Historical A4 L0 seed-100 entry points |
| `train_pure_a4_l0_five_seeds.py` | PURE A4 L0 seeds 100–104 |
| `train_ubfc_a4_remaining_seeds.py` | UBFC A4 L0 seeds 101–104 after seed 100 |
| `verify_pure_a4_training.py`, `verify_ubfc_a4_training.py` | A4 cache/loader/training checks |
| `evaluate_pure_a4_l0_seed100.py`, `evaluate_ubfc_a4_l0_seed100.py` | Historical seed-100 evaluators |
| `evaluate_pure_a4_l0_five_seeds.py`, `evaluate_ubfc_a4_l0_five_seeds.py` | A4 L0 five-seed evaluators |
| `trainer_a4_loss_suite.py` | V9 A4 L1–L4 trainer |
| `verify_a4_loss_suite.py` | V9 loss/loader/gradient verification |
| `train_pure_a4_loss_suite.py`, `train_ubfc_a4_loss_suite.py` | V9 source runners |
| `evaluate_a4_loss_suite.py`, `evaluate_pure_a4_loss_suite.py`, `evaluate_ubfc_a4_loss_suite.py` | V9 evaluation engine/wrappers |
| `trainer_a4_loss_weight_tuning.py` | V10 eight-configuration weight-sweep trainer |
| `verify_a4_loss_weight_tuning.py` | V10 configuration/gradient/data verification |
| `train_pure_a4_loss_weight_tuning.py`, `train_ubfc_a4_loss_weight_tuning.py` | V10 source runners |
| `evaluate_a4_loss_weight_tuning.py`, `evaluate_pure_a4_loss_weight_tuning.py`, `evaluate_ubfc_a4_loss_weight_tuning.py` | V10 evaluation engine/wrappers |
| `trainer_a4_final_multiseed.py` | V11 final-candidate seeds 101–104 trainer |
| `verify_a4_final_multiseed.py` | V11 loss, seed, loader, and training preflight |
| `train_pure_a4_final_multiseed.py`, `train_ubfc_a4_final_multiseed.py` | V11 source runners |
| `evaluate_a4_final_multiseed.py`, `evaluate_pure_a4_final_multiseed.py`, `evaluate_ubfc_a4_final_multiseed.py` | V11 evaluation engine/wrappers |
| `verify_a4_final_multiseed_evaluation.py` | V11 checkpoint/evaluation preflight |

### 11.5 Documentation and preprocessing files

The root `*_README.md` files are experiment-specific notes retained from each development stage. This guide consolidates them; their historical GPU assignments or path examples should not override the verified same-machine paths in Sections 3–4.

Inside `Data_Preprocessing/`:

| File/group | Role |
|---|---|
| `settings.py` | Raw paths, output roots, run mode, dataset selection, TokyoTech gate |
| `common.py` | Shared decoding, face crop, temporal alignment, standardization, chunking, and manifests |
| `dataset_registry.py` | Dataset adapter registry |
| `datasets/shared.py` | Shared dataset-adapter helpers |
| `datasets/pure.py`, `ubfc.py`, `bh.py`, `ubfc_phys.py`, `cohface.py`, `tokyotech.py` | Native-layout adapters |
| `preprocess_all.py` | Selected-dataset orchestration |
| `preprocess_*.py` | Dataset-specific preprocessing entry points |
| `validate_raw_data.py` | Read-only raw-data audit |
| `preflight_multidataset.py` | Dependency/path/layout checks |
| `validate_cache.py` | Cache shape, dtype, finite-value, chunk, and manifest checks |
| `parity_check.py`, `full_parity_check.py` | PURE/UBFC preprocessing parity checks |
| `audit_tokyotech_sync.py` | TokyoTech waveform/video synchronization audit |
| `assets/haarcascade_frontalface_default.xml` | OpenCV face detector data |
| `THIRD_PARTY_LICENSES/` | Haar cascade and third-party notices |
| `requirements.txt` | Explicit preprocessing dependency list |

## 12. Colleague handoff checklist

Before running an experiment, the colleague should confirm:

1. `conda activate mamba_hunting` succeeds.
2. `readlink -f results` resolves to the intended external results directory.
3. The required cache and offline augmentation directory exist.
4. The required input checkpoints exist.
5. The experiment's `verify_*.py` script passes when one is provided.
6. Training completion JSON and best checkpoint exist before evaluation.
7. Both source evaluations finish before generating a combined report.
8. Only the intended `evaluation_protocols*` roots are visible when an isolated report version is required.
9. The generated CSV row count and HTML dropdowns match the intended model family.
10. Logs and completion/configuration JSON files are retained with the results.

## 13. Source limitations discovered during documentation

- The archive is a source snapshot, not a complete runnable package: checkpoints, caches, and results are external.
- The root README contains an outdated external-results target.
- There is no verified root-level environment specification for recreating `mamba_hunting`, especially on Windows.
- Several historical folders retain names such as `pilot` or `official`; interpret them using this guide's terminology and experiment mapping.
- The analysis generators aggregate all compatible evaluation roots they can see. Version numbers identify report files, not scientific experiment membership.

These limitations do not invalidate the completed experiments, but they must be understood before another person reruns or reports them.

