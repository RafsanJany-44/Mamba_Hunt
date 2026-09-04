# Stage-2 loss-suite evaluation

The evaluator runs the eight new loss-suite checkpoints from one source across
six datasets and the Official, Old and PRISM protocols. It reuses
`eval_protocols.py` without changing metrics, spectral settings, diagnostic
tables, plots or failure definitions.

Run PURE and UBFC on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_pure_loss_suite_stage2.py
```

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_ubfc_loss_suite_stage2.py
```

Each command produces 144 summaries. Together they produce 288 evaluations in:

```text
results/evaluation_protocols_loss_suite_stage2/
```

The scripts validate all checkpoints and manifests before inference. On rerun,
they reuse only completed protocol folders whose summary identity and counts
are valid.
