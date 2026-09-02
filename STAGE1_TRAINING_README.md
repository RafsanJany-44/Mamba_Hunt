# RhythmMamba Stage-1 augmentation ablation training

Place all five Python files directly inside `Mamba_Hunt`.

## Setups

| Setup | Verified official augmentation | Offline sampling | Online RGB gain |
|---|---:|---:|---:|
| A1 | Yes | No | Yes, p=0.5 |
| A2 | Yes | Yes, p=0.5 | No |
| A3 | Yes | Yes, p=0.5 | Yes, p=0.5 |

The existing verified cross-matched checkpoints serve as A0 and are not
retrained. Validation always uses unchanged original clips.

## Training policy

- Source split: 0.0–0.8 training, 0.8–1.0 clean validation
- Maximum duration: 60 epochs
- Minimum duration: 30 epochs
- Early stopping after epoch 30: 10 epochs without lower validation loss
- Selection: lowest clean validation loss
- Saved checkpoint: best only
- Seed: 100
- A1, A2, and A3 each contain the same number of training rows per epoch
- For A2/A3, every row independently selects original or offline input at p=0.5

## Run on two GPUs

Terminal 1:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=0 python train_pure_augmentation_stage1.py
```

Terminal 2:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=1 python train_ubfc_augmentation_stage1.py
```

Each source script trains A1, A2, and A3 sequentially. A completed setup is
skipped safely on rerun. A non-empty incomplete setup directory causes a stop
so partial evidence is not silently overwritten.

## Outputs

```text
results/models/augmentation_stage1/
├── PURE_A1/
├── PURE_A2/
├── PURE_A3/
├── UBFC_A1/
├── UBFC_A2/
└── UBFC_A3/
```

Every directory contains:

- `<setup>_RhythmMamba_Best.pth`
- `<setup>_training_history.csv`
- `<setup>_configuration.json`
- `<setup>_completion.json`

The history records actual original/offline selections and online-RGB counts
for every epoch.
