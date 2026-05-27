# FNO Climate Downscaling
### Fourier Neural Operator for ERA5 Temperature & Precipitation Downscaling over Europe

This repository provides a complete starter implementation for the bachelor thesis project
**"Neural Operator for Climate Data Restoration"** (LUT University).

The code is adapted from [ClimateDiffuse](https://arxiv.org/abs/2404.17752) (Watt & Mansfield, 2024),
retaining its preprocessing structure and residual training formulation, and replacing
the diffusion/UNet model with a **Fourier Neural Operator (FNO)** (Li et al., ICLR 2021).

---

## Data: real fine and coarse ERA5 from ECMWF

Unlike synthetic coarsening approaches, this project downloads **two separate ERA5 products**
at their native resolutions directly from the Copernicus Climate Data Store (CDS):

| | Resolution | Grid points (Europe) | Size / year |
|---|---|---|---|
| **Fine** | 0.25° × 0.25° | ~149 × 161 | ~18 MB |
| **Coarse** | 1.50° × 1.50° | ~26 × 27 | ~0.4 MB |

Both are real ECMWF reanalysis products — the coarse field is not an artificial
degradation of the fine field.  The downscaling factor is **6×**.

**Domain:** Western and Central Europe — 10°W to 30°E, 35°N to 72°N

**Variables:** 2-metre temperature (T2m) and total precipitation (TP)

**Dataset size:** 8 random time steps per month × 12 months = ~96 samples/year.
For 1979–2021 (43 years) this gives ~4,100 training samples — manageable on a single moderate GPU.

---

## Project structure

```
fno_climate/
├── data/                              ← ERA5 NetCDF files (created by download script)
├── checkpoints/                       ← Saved model weights
├── results/                           ← Training figures and evaluation plots
├── runs/                              ← TensorBoard logs
├── download_ERA5/
│   ├── download_era5_europe.sh        ← Master loop: calls the two scripts below
│   ├── download_month_cds.py          ← Downloads one month at 0.25° AND 1.5° from CDS
│   └── preprocess_concat_year.py      ← Merges monthly → yearly files
├── src/
│   ├── dataset.py                     ← ERA5EuropeDataset: real paired coarse/fine loader
│   ├── model.py                       ← FNO2d: Fourier Neural Operator
│   ├── train.py                       ← Training script
│   └── evaluate.py                    ← Evaluation + plotting script
└── environment.yml                    ← Conda environment
```

---

## Quick start

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate fno_climate
```

### 2. Configure CDS API access

Register for a free account at [Copernicus CDS](https://cds.climate.copernicus.eu),
then create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <your-UID>:<your-API-key>
```

Your UID and API key are shown on your CDS profile page.

### 3. Download ERA5 data

```bash
bash download_ERA5/download_era5_europe.sh
```

This downloads ERA5 for 1979–2021 at both 0.25° and 1.5° over Europe,
sub-samples 8 random time steps per month, and saves paired files:

```
data/
├── samples_fine_1979.nc      ← 0.25°  T2m + TP, Europe, 1979
├── samples_coarse_1979.nc    ← 1.5°   T2m + TP, Europe, 1979
├── samples_fine_1980.nc
├── samples_coarse_1980.nc
...
```

> **Tip:** To download a single year for testing before running the full loop:
> ```bash
> python download_ERA5/download_month_cds.py --year 2000 --month 1 --n_samples 8
> python download_ERA5/preprocess_concat_year.py --year 2000
> ```

You also need the static ERA5 fields (geopotential + land-sea mask) from CDS:
select `geopotential (z)` and `land-sea mask (lsm)` under *Other* variables,
same domain, and save as `data/ERA5_const_sfc_variables.nc`.

### 4. Train

```bash
cd src
python train.py
```

Default settings:
- Training years: 1979–2016  (~3,700 samples)
- Validation year: 2017  (~96 samples)
- FNO: 4 layers, 64 hidden channels, 16 Fourier modes per dimension

Key options:

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `../data` | Path to ERA5 data |
| `--year_start` | `1979` | First training year |
| `--year_val` | `2017` | First validation year |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `8` | Batch size |
| `--hidden` | `64` | FNO hidden channel width |
| `--n_modes` | `16` | Fourier modes per spatial dimension |
| `--n_layers` | `4` | Number of FNO blocks |
| `--with_dem` | off | Add DEM + land-sea mask as input channels |
| `--lr` | `3e-4` | Learning rate |

**Example: train with DEM conditioning (Step 4)**
```bash
python train.py --with_dem --epochs 150 --hidden 96
```

Training produces:
- `checkpoints/best.pt` — best validation checkpoint
- `results/sample_epoch{N}.png` — visual comparison every 10 epochs
- `results/train_loss.png` — loss curve
- TensorBoard logs in `runs/`

```bash
tensorboard --logdir runs/
```

### 5. Evaluate

```bash
python evaluate.py --checkpoint ../checkpoints/best.pt --year_test 2018
```

Console output example:
```
──────────────────────────────────────────────
Method               Metric  T2m [K]     TP [m]
──────────────────────────────────────────────
Bilinear (coarse)    RMSE    2.31        4.1e-05
                     MAE     1.78        3.0e-05
FNO                  RMSE    1.42        2.9e-05
                     MAE     1.09        2.0e-05
──────────────────────────────────────────────
```

Outputs saved to `results/eval/`:
- `spatial_mae_T2m.png` — spatial map of MAE over Europe
- `example_t0_T2m.png` — coarse / FNO / truth comparison
- `spectrum_T2m.png` — power spectrum: does FNO recover fine-scale energy?

---

## How the dataset works

```
CDS download
    │
    ├── samples_fine_{year}.nc    (0.25°)  ─── fine field          (N, 2, 149, 161)
    └── samples_coarse_{year}.nc  (1.5°)   ─── coarse field        (N, 2,  26,  27)
                                                        │
                                           bilinear interpolation to 0.25° grid
                                                        │
                                              coarse_interp         (N, 2, 149, 161)
                                                        │
                            residual = fine − coarse_interp         (N, 2, 149, 161)
                                                        │
                                          normalise per channel
                                                        │
                                  ┌─────────────────────┴─────────────────────┐
                              model input                               model target
                           (normalised coarse_interp)           (normalised residual)
```

The model predicts the residual.  At inference time the final fine-resolution
output is reconstructed as:

```
prediction = coarse_interp + denormalise(predicted_residual)
```

---

## Thesis research plan

| Step | Task | What to modify |
|---|---|---|
| **1** | FNO baseline *(this code)* | Run as-is |
| **2** | Compare U-FNO, U-Net, EDSR | Add new classes to `src/model.py`; same train script |
| **3** | Physics-informed constraints (PINO) | Add PDE residual loss term in `src/train.py` |
| **4** | DEM as boundary condition | Enable `--with_dem`; run ablation |

---

## References

1. Li, Z. et al. (2021). Fourier Neural Operator for Partial Differential Equations. *ICLR 2021*. https://arxiv.org/abs/2010.08895
2. Watt, R. A. & Mansfield, L. A. (2024). Generative Diffusion-based Downscaling for Climate. *arXiv:2404.17752*. https://arxiv.org/abs/2404.17752
3. Li, Z. et al. (2023). Physics-Informed Neural Operator for Learning Partial Differential Equations. *arXiv:2111.03794*. https://arxiv.org/abs/2111.03794
4. Harder, P. et al. (2024). Hard-Constrained Deep Learning for Climate Downscaling. *IEEE TNNLS*. https://arxiv.org/abs/2208.05424
5. Mardani, M. et al. (2024). Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling. *arXiv:2309.15214*. https://arxiv.org/abs/2309.15214

---

*LUT University · Supervisor: Zhi-Song Liu (zhisong.liu@lut.fi) · Co-supervisor: Tapio Helin*
