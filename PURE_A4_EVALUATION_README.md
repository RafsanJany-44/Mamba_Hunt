# PURE A4 seed-100 pilot evaluation

Place `evaluate_pure_a4_l0_seed100.py` directly inside `Mamba_Hunt`.

Run:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_a4_l0_seed100.py
```

The source-dataset result uses the untouched PURE `0.8-1.0` split. The other
five datasets use their complete `0.0-1.0` manifests. Every dataset is evaluated
with Official, Old, and PRISM protocols.

Outputs are written to:

```text
results/evaluation_protocols_pure_a4_pilot/
```

The evaluator resumes safely by reusing a protocol only when its saved summary
is complete and valid.
