# UBFC A4 L0 remaining seeds

This launcher trains seeds 101, 102, 103 and 104 sequentially. It reuses the
already verified `train_ubfc_a4_l0_seed100.py` implementation and does not
modify or repeat seed 100.

Required files already in the `Mamba_Hunt` root:

```text
dataset_ubfc_a4.py
train_ubfc_a4_l0_seed100.py
```

Run on GPU 0:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_remaining_seeds.py
```

Each seed uses the same configuration as the pilot: official L0 loss, 100
maximum epochs, 30 minimum epochs, patience 10, 50% original sampling and 50%
uniform A4 sampling. Only the best checkpoint is saved.

Verified completed seeds are skipped safely if the launcher is rerun.
