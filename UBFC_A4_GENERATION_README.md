# UBFC A4 offline-cache generation

A4 creates four stored variants for every UBFC 0.0–0.8 training clip:
JPEG, Gaussian blur, gamma and contrast. The verified label is reused.

Run from `Mamba_Hunt`:

```bash
conda activate mamba_hunting
python generate_ubfc_a4_offline_augmentation.py
```

Expected final counts:

```text
Recordings      : 33
Original clips  : 378
Augmented clips : 1512
Each transform  : 378
```

Do not start A4 training until the generator reports
`Shape/finite/labels: PASSED` and its final output has been reviewed.
