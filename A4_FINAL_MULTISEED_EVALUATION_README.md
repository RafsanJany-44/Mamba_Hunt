# A4 final multi-seed evaluation

This evaluates only the eight newly trained checkpoints (seeds 101-104). Seed
100 is intentionally excluded because its evaluation already exists.

| GPU | Source candidate | Checkpoints | Evaluations |
|---|---|---:|---:|
| 0 | UBFC A4 + L3, C=0.5 | 4 | 72 |
| 1 | PURE A4 + L4, C=1.0, H=1.0 | 4 | 72 |

Each checkpoint is evaluated on six datasets using official-complete, Old, and
PRISM protocols. Total: 144 setups.

Raw outputs are isolated in:

```text
results/evaluation_protocols_a4_final_multiseed/
```

Signal plots, PSD diagnostics, and signal sample tables are disabled. Summary
plots, protocol-comparison HTML files, `summary.json`, and
`FAILURE_TYPE_SUMMARY.csv` remain enabled. Completed outputs are safely resumed
only after their identity and structure are verified.

## Run concurrently

First verify:

```bash
python verify_a4_final_multiseed_evaluation.py
```

Proceed only if it reports `PASSED`.

GPU 0 — UBFC:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a4_final_multiseed.py 2>&1 \
  | tee results/logs/UBFC_A4_final_multiseed_evaluation.log
```

GPU 1 — PURE:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_a4_final_multiseed.py 2>&1 \
  | tee results/logs/PURE_A4_final_multiseed_evaluation.log
```

Expected final summaries:

```text
all_results_summary_a4_final_multiseed_ubfc_4x6x3.csv
all_results_summary_a4_final_multiseed_pure_4x6x3.csv
```
