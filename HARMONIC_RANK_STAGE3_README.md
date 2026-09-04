# Stage-3 conditional harmonic-ranking experiment

Copy these four Python files into the root of `Mamba_Hunt`. They reuse the
existing A2 loaders and do not modify or overwrite previous experiments.

## 1. Verify the loss

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
python verify_harmonic_rank_loss.py
```

Do not train unless the final output says `PASSED`.

## 2. Train on two GPUs

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 python train_pure_harmonic_rank.py
```

Terminal 2:

```bash
CUDA_VISIBLE_DEVICES=1 python train_ubfc_harmonic_rank.py
```

Outputs are saved below:

```text
results/models/harmonic_rank_stage3/PURE_A2_HARMONIC_RANK/
results/models/harmonic_rank_stage3/UBFC_A2_HARMONIC_RANK/
```

Each run saves only its best checkpoint, training history, configuration and
completion record. Existing A2 and Stage-2 results remain unchanged.
