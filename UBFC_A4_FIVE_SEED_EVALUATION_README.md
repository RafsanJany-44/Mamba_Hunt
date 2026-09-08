# UBFC A4 L0 five-seed evaluation

Place `evaluate_ubfc_a4_l0_five_seeds.py` directly inside `Mamba_Hunt`.

Run on GPU 0 after all five UBFC A4 checkpoints complete:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a4_l0_five_seeds.py 2>&1 | \
  tee results/logs/UBFC_A4_L0_five_seed_evaluation.log
```

This performs 5 checkpoints x 6 datasets x 3 protocols = 90 evaluations.
UBFC uses its held-out 0.8-1.0 source split; the other five datasets use their
complete 0.0-1.0 data. Results are written to:

```text
results/evaluation_protocols_ubfc_a4_pilot/
```

Existing complete protocol results are reused safely if execution is resumed.
