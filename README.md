# FNO Climate Downscaling
### Fourier Neural Operator for ERA5 Temperature & Precipitation Downscaling over Europe

This repository provides a complete starter implementation for the bachelor thesis project
**"Neural Operator for Climate Data Restoration"** (LUT University).

The code is adapted from [ClimateDiffuse](https://arxiv.org/abs/2404.17752) (Watt & Mansfield, 2024),
retaining its data download and preprocessing pipeline, and replacing the diffusion / UNet model
with a **Fourier Neural Operator (FNO)** (Li et al., ICLR 2021).

---

## Project structure

```
fno_climate/
├── data/                          ← ERA5 NetCDF files go here (created by download script)
├── checkpoints/                   ← Saved model weights
├── results/                       ← Training figures and evaluation plots
├── runs/                          ← TensorBoard logs
├── download_ERA5/
│   ├── download_era5_europe.sh    ← Master download + preprocess script
│   ├── preprocess_subsample.py    ← Per-month subsampling and cropping
│   └── preprocess_concat_year.py  ← Concatenate monthly → yearly files
├── src/
│   ├── dataset.py                 ← ERA5EuropeDataset: paired coarse/fine loader
│   ├── model.py                   ← FNO2d: Fourier Neural Operator
│   ├── train.py                   ← Training script
│   └── evaluate.py                ← Evaluation + plotting script
└── environment.yml                ← Conda environment
```

---

## Quick start

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate fno_climate
```

### 2. Download ERA5 data

Register for a free account at [NCAR RDA](https://rda.ucar.edu/), then:

```bash
export NCAR_USER="your_email@example.com"
export NCAR_PASS="your_password"
bash download_ERA5/download_era5_europe.sh
```

This downloads ERA5 over **Europe** (−25°W to 45°E, 34°N to 72°N) at 0.25° resolution
for two variables:
- `VAR_2T` — 2-metre temperature [K]
- `TP` — total precipitation [m]

Data is sub-sampled to 30 random time steps per month and saved as `data/samples_{year}.nc`.

> **Disk space:** approximately 5–10 GB for 1979–2022 after sub-sampling.

You will also need the static ERA5 fields (geopotential + land-sea mask).
Download them from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu)
by selecting `geopotential (z)` and `land-sea mask (lsm)` under *Other* variables,
save as `data/ERA5_const_sfc_variables.nc`.

### 3. Train the FNO

```bash
cd src
python train.py
```

Default settings:
- Training years: 1979–2016
- Validation year: 2017
- Downscaling factor: 4× (0.25° → 1.00° → 0.25°)
- FNO: 4 layers, 64 channels, 16 Fourier modes per dimension

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `../data` | Path to ERA5 data directory |
| `--coarsen_factor` | `4` | Spatial downscaling factor (try 2, 4, 8) |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `8` | Batch size |
| `--hidden` | `64` | FNO hidden channel width |
| `--n_modes` | `16` | Fourier modes per spatial dimension |
| `--n_layers` | `4` | Number of FNO blocks |
| `--with_dem` | off | Add DEM + land-sea mask as input channels |
| `--lr` | `3e-4` | Learning rate |

**Example: 8× downscaling with DEM conditioning**
```bash
python train.py --coarsen_factor 8 --with_dem --epochs 200 --hidden 96
```

Training produces:
- `checkpoints/best.pt` — best validation checkpoint
- `checkpoints/last.pt` — last epoch checkpoint
- `results/sample_epoch{N}.png` — visual comparison every 10 epochs
- `results/train_loss.png` — loss curve
- TensorBoard logs in `runs/`

Monitor training with:
```bash
tensorboard --logdir runs/
```

### 4. Evaluate

```bash
python evaluate.py --checkpoint ../checkpoints/best.pt --year_test 2018
```

Outputs written to `results/eval/`:
- `spatial_mae_T2m.png` / `spatial_mae_TP.png` — spatial maps of MAE
- `example_t0_T2m.png` — coarse vs. FNO vs. truth for one time step
- `spectrum_T2m.png` — power spectrum comparison

Console output example:
```
──────────────────────────────────────────────
Method               Metric  T2m         TP
──────────────────────────────────────────────
Bilinear (coarse)    RMSE   2.3104      4.12e-05
                     MAE    1.7843      2.98e-05
FNO                  RMSE   1.4211      2.87e-05
                     MAE    1.0932      2.04e-05
──────────────────────────────────────────────
```

---

## Understanding the code

### Dataset (`src/dataset.py`)

The dataset follows the same residual formulation as ClimateDiffuse:

1. Load fine-resolution ERA5 fields (0.25°) as the **target**
2. Coarsen by factor `k` using bilinear interpolation → **coarse** field
3. Bilinearly upsample the coarse field back to the fine grid → **input to the model**
4. The **target** for the network is the **residual** = fine − upsampled coarse
5. Both input and residual are normalised channel-wise before training

The model thus learns to predict the high-frequency detail missing from the
coarsened input, which is a standard residual super-resolution formulation.

### Model (`src/model.py`)

The FNO2d model has three parts:

```
Input (B, C_in, H, W)
    │
    ▼
Lifting layer  [Conv 1×1: C_in → C_hidden]
    │
    ▼  × n_layers
┌─────────────────────────────────────────────┐
│  Spectral branch: FFT → R(ξ) · F(v) → iFFT │
│  +                                          │
│  Local branch: Conv 1×1                     │
│  → InstanceNorm → GELU                      │
└─────────────────────────────────────────────┘
    │
    ▼
Projection  [Conv 1×1: C_hidden → 128 → C_out]
    │
    ▼
Output residual (B, 2, H, W)
```

The key operation in each FNO block (spectral branch):

```
v'(x) = F^{-1}[ R(ξ) · F(v)(ξ) ]  +  W v(x)
```

where `R(ξ)` are learnable complex weights for the lowest `n_modes` frequency components.

### Training (`src/train.py`)

- Loss: MSE on the normalised residual
- Optimiser: AdamW with cosine annealing scheduler
- Mixed-precision training with `torch.cuda.amp` on GPU
- Gradient accumulation supported via `--accum`

---

## Thesis research plan

This code covers **Step 1** of the thesis research plan.
The remaining steps build directly on it:

| Step | Task | What to change |
|------|------|----------------|
| **1** | FNO baseline *(this code)* | Run as-is |
| **2** | Compare U-FNO, U-Net, EDSR | Add new model classes to `src/model.py`, train each with the same script |
| **3** | Physics-informed constraints (PINO) | Add a PDE residual loss term in `src/train.py` |
| **4** | DEM as boundary condition | Use `--with_dem` flag; analyse the ablation |

---

## References

1. Li, Z. et al. (2021). Fourier Neural Operator for Partial Differential Equations. *ICLR 2021*. https://arxiv.org/abs/2010.08895
2. Watt, R. A. & Mansfield, L. A. (2024). Generative Diffusion-based Downscaling for Climate. *arXiv:2404.17752*. https://arxiv.org/abs/2404.17752
3. Li, Z. et al. (2023). Physics-Informed Neural Operator for Learning Partial Differential Equations. *arXiv:2111.03794*. https://arxiv.org/abs/2111.03794
4. Harder, P. et al. (2024). Hard-Constrained Deep Learning for Climate Downscaling. *IEEE TNNLS*. https://arxiv.org/abs/2208.05424
5. Mardani, M. et al. (2024). Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling. *arXiv:2309.15214*. https://arxiv.org/abs/2309.15214

---

*LUT University · Supervisor: Zhi-Song Liu (zhisong.liu@lut.fi) · Co-supervisor: Tapio Helin*
