# RhythmMamba Stage-1 augmentation evaluation

Place the four Python files directly inside `Mamba_Hunt`.

The evaluation contains six checkpoints, six datasets and three unchanged
protocols: 108 model–dataset–protocol setups. Signal plots, PSD diagnostics,
summary plots and signal tables remain enabled exactly as in the verified
evaluator.

## Parallel execution

Terminal 1 — PURE A1/A2/A3 on GPU 0:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=0 python evaluate_pure_augmentation_stage1_3x6x3.py
```

Terminal 2 — UBFC A1/A2/A3 on GPU 1:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=1 python evaluate_ubfc_augmentation_stage1_3x6x3.py
```

Both jobs write to non-overlapping checkpoint directories under:

```text
results/evaluation_protocols_augmentation_stage1/
```

Completed protocol summaries are validated and reused on restart. A partial
protocol without a valid `summary.json` is evaluated again.

## Finalize after both jobs finish

```bash
python finalize_augmentation_stage1_evaluation.py
```

Expected final summary:

```text
results/evaluation_protocols_augmentation_stage1/
all_results_summary_augmentation_stage1_6x6x3.csv
```

Then regenerate the automatic presentation analyses:

```bash
python generate_mae_analysis.py
python generate_harmonic_error_tables.py
```

Because both generators automatically discover `evaluation_protocols*`, no
experiment root should need to be added manually.
