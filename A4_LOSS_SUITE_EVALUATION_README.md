# A4 + L1-L4 evaluation

This evaluation processes four PURE checkpoints and four UBFC checkpoints on
six datasets using the Official, Old, and PRISM protocols.

Run PURE on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_pure_a4_loss_suite.py
```

Run UBFC on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a4_loss_suite.py
```

Each source produces 72 summaries (`4 checkpoints x 6 datasets x 3
protocols`). Together, the two runs produce 144 summaries. Results are stored
under `results/evaluation_protocols_a4_loss_suite`.
