# Independent RhythmMamba preprocessing

This package reads the native layouts of PURE, UBFC-rPPG, BH-rPPG,
UBFC-PHYS, COHFACE, and TokyoTech. It creates the 160-frame `.npy` clips and
CSV manifests consumed directly by `Mamba_Hunt/dataset.py`. It does not import
the official repository and it does not use YAML.

## What remains identical

PURE and UBFC retain the already verified bit-identical route: first-frame
Haar face detection, 1.5x fixed face box, 128x128 resize, per-recording global
standardization, standardized waveform, and non-overlapping 160-frame chunks.

The additional datasets use those same image/signal transformations after an
explicit conversion to the model's 30 Hz time base:

| Dataset | Native video / label | Evaluation unit | Saved metadata |
|---|---|---|---|
| BH-rPPG | variable ~15 fps PNG / ~59 Hz wave | subject-condition | low, medium, high |
| UBFC-PHYS | 35.138 fps AVI / 64 Hz BVP | subject-task | T1/T2/T3, ctrl/test |
| COHFACE | 20 fps AVI / 256 Hz HDF5 pulse | subject-session | session, lamp/natural |
| TokyoTech | nine consecutive 30 fps AVI / 2048 Hz MAT | complete 180 s subject | sync method |

The new caches use float32 to control storage. The model loader converts both
historical float64 and new caches to float32, so model input dtype is
unchanged. A `recording_metadata.csv` file is written for stratified
condition/task/illumination evaluation.

## Safe order

1. Edit raw paths in `settings.py`.
2. Select one dataset in `DATASETS_TO_PROCESS` and keep `RUN_MODE = "smoke"`.
3. Run the read-only checks:

   ```bash
   python Mamba_Hunt/Data_Preprocessing/validate_raw_data.py
   python Mamba_Hunt/Data_Preprocessing/preflight_multidataset.py
   ```

4. Create one-record smoke output:

   ```bash
   python Mamba_Hunt/Data_Preprocessing/preprocess_all.py
   ```

5. Change `RUN_MODE = "full"`, run a dataset-specific script, then validate:

   ```bash
   python Mamba_Hunt/Data_Preprocessing/preprocess_bh.py
   python Mamba_Hunt/Data_Preprocessing/validate_cache.py
   ```

   Equivalent scripts exist for `ubfc_phys`, `cohface`, and `tokyotech`.

6. Repeat for the next dataset. Existing raw data and
   `RhythmMamba_Preprocessed` are read-only; outputs go only to
   `RhythmMamba_Preprocessed_Independent`.

## TokyoTech synchronization gate

TokyoTech subjects 05-09 contain more PPG time than the 180 seconds of video.
The code refuses to guess an offset. Run:

```bash
python Mamba_Hunt/Data_Preprocessing/audit_tokyotech_sync.py
```

Inspect `TokyoTech_Synchronization_Audit.csv` and `.json` under `DATA_ROOT`.
The audit compares first 180 seconds, last 180 seconds, and full-duration
rescaling using nine 20-second video/PPG HR comparisons plus lag-tolerant
waveform correlation. Only after inspection set:

```python
TOKYOTECH_ACCEPT_AUDIT_RECOMMENDATIONS = True
```

Then run the TokyoTech smoke preprocess. This gate prevents an undocumented
alignment assumption from contaminating the cross-dataset result.

## Full 2 x 6 x 3 evaluation

After all six full caches validate, run from the `Mamba_Hunt` directory:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_2x6x3.py \
  2>&1 | tee results/evaluation_protocols/evaluate_2x6x3.log
```

PURE and UBFC checkpoints use their held-out native test split on their source
dataset. Every external target uses the complete dataset. Results are written
under `results/evaluation_protocols/{PURE_CHECKPOINT,UBFC_CHECKPOINT}` with
separate `official_mamba`, `old`, and `prism` folders.

## Dependencies

Install `requirements.txt`. COHFACE needs `h5py`; TokyoTech and evaluation need
SciPy. The original PURE/UBFC parity scripts remain available and unchanged.
