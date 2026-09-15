# A4 Loss-Weight Tuning Evaluation

This evaluator processes eight weight-tuning checkpoints from one source model
on six datasets with the Official, Old, and PRISM protocols: 144 setups per
source.

The raw output directory is:

```text
results/evaluation_protocols_a4_loss_weight_tuning/
```

Large per-signal plots, PSD diagnostics, and signal sample tables are disabled.
Summary JSON files, failure-type CSV files, protocol summaries, and compact
summary plots are retained, so `generate_mae_analysis.py` and
`generate_harmonic_error_tables.py` remain compatible.

Evaluate UBFC:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a4_loss_weight_tuning.py
```

Evaluate PURE after its training completes:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_a4_loss_weight_tuning.py
```

Do not run the MAE and harmonic generators until both source evaluations have
finished if the intended report must contain all 16 new tuning checkpoints.
