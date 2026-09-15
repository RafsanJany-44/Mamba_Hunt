# A4 final-candidate multi-seed confirmation

This suite confirms the two selected A4 candidates with four new seeds. Seed
100 already exists and is not retrained.

| Source | Candidate | Effective loss | New seeds |
|---|---|---|---|
| PURE | `PURE_A4_L4_C100_H100` | `0.2 Pearson + 1.0 concentration + 1.0 harmonic` (CE removed) | 101-104 |
| UBFC | `UBFC_A4_L3_C050` | `0.2 Pearson + 0.5 concentration` (CE removed) | 101-104 |

Both use A4 sampling: 50% original and 12.5% each JPEG, blur, gamma, and
contrast. Validation is clean. Training uses at most 100 epochs, at least 30
epochs, patience 10, and saves only the lowest-validation-loss checkpoint.

## Install and verify

Extract this archive directly inside `Mamba_Hunt`, then run:

```bash
python verify_a4_final_multiseed.py
```

Do not begin training unless the final line reports `PASSED`.

## Train concurrently

Terminal 1, GPU 0 — UBFC:

```bash
mkdir -p results/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_final_multiseed.py 2>&1 \
  | tee results/logs/UBFC_A4_L3_C050_seeds101_104.log
```

Terminal 2, GPU 1 — PURE:

```bash
mkdir -p results/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python train_pure_a4_final_multiseed.py 2>&1 \
  | tee results/logs/PURE_A4_L4_C100_H100_seeds101_104.log
```

Outputs are saved beneath:

```text
results/models/a4_final_multiseed/
```

Every experiment has its own configuration JSON, training-history CSV,
completion JSON, and best checkpoint. An incomplete nonempty experiment folder
causes a deliberate stop instead of silently overwriting evidence.
