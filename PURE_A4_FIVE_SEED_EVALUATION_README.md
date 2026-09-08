# PURE A4 L0 five-seed evaluation

Place `evaluate_pure_a4_l0_five_seeds.py` directly inside `Mamba_Hunt`.

Run on GPU 1 after all five PURE A4 checkpoints complete:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_a4_l0_five_seeds.py 2>&1 | \
  tee results/logs/PURE_A4_L0_five_seed_evaluation.log
```

This performs 5 checkpoints x 6 datasets x 3 protocols = 90 evaluations.
PURE uses its held-out 0.8-1.0 source split; the other five datasets use their
complete 0.0-1.0 data. Results are written to:

```text
results/evaluation_protocols_pure_a4_pilot/
```

Existing complete protocol results are reused safely if execution is resumed.
