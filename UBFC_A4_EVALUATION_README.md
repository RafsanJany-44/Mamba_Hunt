# UBFC A4 pilot evaluation

Copy `evaluate_ubfc_a4_l0_seed100.py` into the `Mamba_Hunt` root and run:

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
CUDA_VISIBLE_DEVICES=0 python evaluate_ubfc_a4_l0_seed100.py
```

It evaluates one checkpoint on six datasets with the Official, Old and PRISM
protocols (18 evaluations). Existing evaluation folders are not overwritten.

Physical output through the linked `results` directory:

```text
/media/data/rPPG/Code/GitHub/Project_rPPG_Result/Result_Lab_Mamba_Results/
results/evaluation_protocols_ubfc_a4_pilot
```

The script can safely resume: a protocol result is reused only when its summary
exists and passes the basic completeness checks.
