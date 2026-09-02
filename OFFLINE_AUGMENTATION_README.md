# Stage-1 offline augmentation cache generation

Place these three Python files directly inside `Mamba_Hunt`.

The fixed policy generates one augmented input for each clip in the verified
0.0–0.8 training manifest. Each clip receives exactly one deterministic,
clip-consistent transform:

- JPEG quality: 70–90
- Gaussian blur sigma: 0.3–0.8
- Gamma: 0.85–1.15
- Contrast: 0.90–1.10

The processing order is:

```text
raw frames -> verified face crop/resize -> one offline transform per clip
           -> global recording standardization -> 160-frame augmented clips
```

Original caches and labels are never modified. Augmented inputs are saved as
float32, while metadata references the verified original label paths.

Run from `Mamba_Hunt`:

```bash
conda activate mamba_hunting
python generate_pure_offline_augmentation.py
python generate_ubfc_offline_augmentation.py
```

These jobs use CPU/OpenCV, not the GPU. They may run in two terminals, but
sequential execution is safer if host RAM is limited.

Expected outputs:

```text
/home/rafsan/Documents/Data/Mamba_Hunt_Data/RhythmMamba_Offline_Augmentation/
├── PURE/
│   ├── <cache>_OfflineOneOfFour/
│   ├── PURE_offline_augmentation_metadata.csv
│   └── PURE_offline_augmentation_summary.json
└── UBFC/
    ├── <cache>_OfflineOneOfFour/
    ├── UBFC_offline_augmentation_metadata.csv
    └── UBFC_offline_augmentation_summary.json
```

Expected counts:

| Dataset | Training recordings | Augmented clips |
|---|---:|---:|
| PURE | 47 | 596 |
| UBFC | 33 | 378 |

The generator is restart-safe when a recording is complete: existing outputs
are shape/dtype/finite validated and reused. If only part of one recording is
present, it stops and identifies that recording instead of silently mixing an
incomplete cache.
