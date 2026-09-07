# PURE A4 L0 five-seed overnight training

Place `train_pure_a4_l0_five_seeds.py` directly inside `Mamba_Hunt`.

The launcher trains seeds 100, 101, 102, 103 and 104 sequentially. Each run
uses PURE 0.0-0.8 for training, clean PURE 0.8-1.0 for validation, A4 offline
sampling, the official online augmentation, the official L0 loss, a maximum of
100 epochs, a minimum of 30 epochs and early-stopping patience 10. Only the
lowest-clean-validation-loss checkpoint is saved.

Run on physical GPU 1:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=1 python train_pure_a4_l0_five_seeds.py 2>&1 | \
  tee results/logs/PURE_A4_L0_five_seeds.log
```

Outputs:

```text
results/models/pure_a4_pilot/PURE_A4_L0_SEED100/
results/models/pure_a4_pilot/PURE_A4_L0_SEED101/
results/models/pure_a4_pilot/PURE_A4_L0_SEED102/
results/models/pure_a4_pilot/PURE_A4_L0_SEED103/
results/models/pure_a4_pilot/PURE_A4_L0_SEED104/
```

The launcher skips a seed only when both its PASSED completion record and best
checkpoint already exist. It refuses to overwrite a nonempty incomplete run.
