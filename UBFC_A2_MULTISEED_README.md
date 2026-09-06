# UBFC A2 paired five-seed experiment

This package trains the paired seed set `100, 101, 102, 103, 104` for:

- `UBFC_A2_L0`: official RhythmMamba loss.
- `UBFC_A2_L5_CE_CONCENTRATION`: official loss plus concentration.

Both suites use the same UBFC 80/20 split, A2 data, optimizer, scheduler,
100-epoch maximum, 30-epoch minimum and early-stopping patience of 10.

## Run

First run the preflight check:

```bash
python verify_ubfc_a2_multiseed.py
```

Do not start training unless it reports `PASSED`.

Terminal/GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a2_l0_five_seeds.py
```

Terminal/GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python train_ubfc_a2_l5_five_seeds.py
```

Outputs are isolated under:

```text
results/models/ubfc_a2_multiseed/
```

Only the best checkpoint, history, configuration and completion record are
saved for each seed. A rerun skips verified complete seeds. An incomplete
existing seed directory causes a safe stop so partial evidence is preserved.
