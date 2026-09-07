# UBFC A4 L0 seed-100 pilot

Copy these three Python files into the root of `Mamba_Hunt`.

Verify the A4 dataset and loader first:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
python verify_ubfc_a4_training.py
```

After verification passes, train on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_l0_seed100.py
```

The best checkpoint is saved at:

```text
results/models/ubfc_a4_pilot/UBFC_A4_L0_SEED100/UBFC_A4_L0_SEED100_RhythmMamba_Best.pth
```

The run uses 100 maximum epochs, a 30-epoch minimum, patience 10, clean
validation, official L0 loss, and saves only the best checkpoint.
