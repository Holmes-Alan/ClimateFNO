"""
ERA5 Europe downscaling dataset.
Adapted from ClimateDiffuse (Watt & Mansfield, 2024).

Key difference from ClimateDiffuse:
  Both the fine (0.25°) and coarse (1.5°) fields are REAL ERA5 products
  downloaded separately from the Copernicus CDS.  No synthetic coarsening
  is performed.  The coarse field is bilinearly interpolated onto the fine
  grid before being fed to the model, and the model learns the residual
  between the fine field and this interpolated coarse field.

Resolution:
  Fine   : 0.25° × 0.25°  → 149 × 161 grid points over Europe (35–72°N, 10°W–30°E)
  Coarse : 1.50° × 1.50°  →  26 ×  27 grid points  (6× downscaling factor)

Dataset size per year (8 samples/month × 12 months = 96 samples):
  Each fine sample  (2, 149, 161) float32 ≈ 190 KB
  Total per year ≈ 18 MB  →  manageable on a single moderate GPU.
"""

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr


# ── Default normalisation statistics ─────────────────────────────────────────
# Rough estimates for Europe; run compute_stats.py on your data to refine.
DEFAULT_MEAN = torch.tensor([281.0,  0.0])    # T2m [K],  TP [m]
DEFAULT_STD  = torch.tensor([ 14.0,  3e-4])   # T2m [K],  TP [m]


class ERA5EuropeDataset(torch.utils.data.Dataset):
    """
    Paired real coarse (1.5°) / fine (0.25°) ERA5 dataset over Europe.

    Variables
    ---------
    VAR_2T : 2-metre temperature [K]
    TP     : total precipitation [m]

    The input to the FNO is the 1.5° coarse field bilinearly interpolated
    to the 0.25° fine grid.  The training target is the residual:
        residual = fine_field − interpolated_coarse_field

    This residual formulation keeps the learning problem focused on the
    high-frequency spatial detail that the coarse field cannot represent,
    and follows the same strategy used in ClimateDiffuse.

    Parameters
    ----------
    data_dir : str
        Directory containing  samples_fine_{year}.nc
        and  samples_coarse_{year}.nc  files.
    year_start, year_end : int
        Year range [year_start, year_end) to load.
    mean, std : torch.Tensor  (2,)
        Per-channel normalisation statistics for the raw fine-resolution data.
        Defaults are rough estimates; compute from training data for best results.
    dem_file : str or None
        Path to ERA5_const_sfc_variables.nc containing geopotential (z) and
        land-sea mask (lsm).  When provided, both are appended as additional
        input channels — useful for Step 4 of the thesis (DEM conditioning).
    """

    # European domain as downloaded from CDS
    # lat: 72°N → 35°N (descending, ERA5 convention)
    # lon: −10°E → 30°E
    LAT_SLICE = slice(72, 35)
    LON_SLICE = slice(-10, 30)

    def __init__(
        self,
        data_dir:   str,
        year_start: int = 1979,
        year_end:   int = 2017,
        mean: torch.Tensor = DEFAULT_MEAN,
        std:  torch.Tensor = DEFAULT_STD,
        dem_file: str | None = None,
    ):
        self.mean = mean
        self.std  = std

        # ── Load fine and coarse yearly files ─────────────────────────────────
        print(f"Loading ERA5 Europe data for years {year_start}–{year_end - 1} …")
        fine_list, coarse_list, time_list = [], [], []

        for year in range(year_start, year_end):
            fine_path   = f"{data_dir}/samples_fine_{year}.nc"
            coarse_path = f"{data_dir}/samples_coarse_{year}.nc"

            ds_fine   = xr.open_dataset(fine_path,   engine="netcdf4")
            ds_coarse = xr.open_dataset(coarse_path, engine="netcdf4")

            # Crop to European domain (in case files are global)
            ds_fine   = ds_fine.sel(latitude=self.LAT_SLICE,
                                    longitude=self.LON_SLICE)
            ds_coarse = ds_coarse.sel(latitude=self.LAT_SLICE,
                                      longitude=self.LON_SLICE)

            # Align on shared time steps (should already match from download)
            common = np.intersect1d(ds_fine.time.values, ds_coarse.time.values)
            ds_fine   = ds_fine.sel(time=common)
            ds_coarse = ds_coarse.sel(time=common)

            fine_list.append(ds_fine)
            coarse_list.append(ds_coarse)
            time_list.append(common)

        ds_fine   = xr.concat(fine_list,   dim="time")
        ds_coarse = xr.concat(coarse_list, dim="time")

        self.lat_fine   = ds_fine.latitude.values
        self.lon_fine   = ds_fine.longitude.values
        self.lat_coarse = ds_coarse.latitude.values
        self.lon_coarse = ds_coarse.longitude.values

        self.H = len(self.lat_fine)   # e.g. 149
        self.W = len(self.lon_fine)   # e.g. 161
        self.fine_shape   = (self.H, self.W)
        self.coarse_shape = (len(self.lat_coarse), len(self.lon_coarse))

        print(f"  Fine grid  : {self.fine_shape}  (0.25°)")
        print(f"  Coarse grid: {self.coarse_shape}  (1.5°)")
        print(f"  Time steps : {len(ds_fine.time)}")

        # ── Build tensors ──────────────────────────────────────────────────────
        # Fine field  (N, 2, H_fine, W_fine)
        t2m_fine = torch.from_numpy(ds_fine["VAR_2T"].values).float()
        tp_fine  = torch.from_numpy(ds_fine["TP"].values).float()
        fine     = torch.stack([t2m_fine, tp_fine], dim=1)

        # Coarse field  (N, 2, H_coarse, W_coarse)
        t2m_coarse = torch.from_numpy(ds_coarse["VAR_2T"].values).float()
        tp_coarse  = torch.from_numpy(ds_coarse["TP"].values).float()
        coarse_native = torch.stack([t2m_coarse, tp_coarse], dim=1)

        # Interpolate coarse → fine grid using bilinear interpolation
        # so both fields live on the same 0.25° grid
        coarse_interp = F.interpolate(
            coarse_native,
            size=self.fine_shape,
            mode="bilinear",
            align_corners=True,
        )                                              # (N, 2, H_fine, W_fine)

        # Residual: what the FNO must learn to predict
        residual = fine - coarse_interp               # (N, 2, H_fine, W_fine)

        # Store un-normalised copies for evaluation and plotting
        self.fine          = fine
        self.coarse_native = coarse_native
        self.coarse_interp = coarse_interp

        # ── Normalisation ──────────────────────────────────────────────────────
        # Normalise the coarse input using raw-data statistics
        coarse_norm = (coarse_interp - mean[None, :, None, None]) \
                    /  std[None, :, None, None]

        # Normalise the residual using its own mean/std (computed from training data)
        self.residual_mean = residual.mean(dim=(0, 2, 3))
        self.residual_std  = residual.std(dim=(0, 2, 3)).clamp(min=1e-8)
        residual_norm = (residual - self.residual_mean[None, :, None, None]) \
                      /  self.residual_std[None, :, None, None]

        self.inputs  = coarse_norm    # (N, 2, H, W)  model input
        self.targets = residual_norm  # (N, 2, H, W)  model target

        # ── Optional DEM + land-sea mask (Step 4) ─────────────────────────────
        self.n_input_channels = 2
        if dem_file is not None:
            print("  Loading DEM / land-sea mask …")
            ds_dem = xr.open_dataset(dem_file, engine="netcdf4")
            ds_dem = ds_dem.sel(latitude=self.LAT_SLICE, longitude=self.LON_SLICE)

            # Geopotential: normalise to zero mean, unit variance
            z = torch.from_numpy(ds_dem["z"].values).float()
            z = (z - z.mean()) / z.std().clamp(min=1e-8)

            # Land-sea mask: already in [0, 1]
            lsm = torch.from_numpy(ds_dem["lsm"].values).float()

            # Interpolate to fine grid if needed
            def _interp_static(field_2d: torch.Tensor) -> torch.Tensor:
                """Interpolate a (H', W') static field to (H, W)."""
                return F.interpolate(
                    field_2d.unsqueeze(0).unsqueeze(0),
                    size=self.fine_shape,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()

            z   = _interp_static(z)
            lsm = _interp_static(lsm)

            N = self.inputs.shape[0]
            aux = torch.stack([
                z.expand(N, -1, -1),
                lsm.expand(N, -1, -1),
            ], dim=1)                                  # (N, 2, H, W)

            self.inputs = torch.cat([self.inputs, aux], dim=1)  # (N, 4, H, W)
            self.n_input_channels = 4
            print("  DEM channels appended — input channels: 4")

        # ── Time metadata ──────────────────────────────────────────────────────
        time = ds_fine.time.dt
        self.doy_norm  = torch.from_numpy(
            ((time.month.values - 1) * 30 + (time.day.values - 1)) / 360.0
        ).float()
        self.hour_norm = torch.from_numpy(time.hour.values / 24.0).float()

        print("Dataset ready.\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def inverse_normalise_residual(self, res_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised residual back to physical units [K] / [m]."""
        return (res_norm
                * self.residual_std[None, :, None, None]
                + self.residual_mean[None, :, None, None])

    def residual_to_fine(self, res_norm: torch.Tensor,
                         coarse_interp: torch.Tensor) -> torch.Tensor:
        """Reconstruct fine-resolution field from predicted normalised residual."""
        return coarse_interp + self.inverse_normalise_residual(res_norm)

    # ── PyTorch interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "inputs":        self.inputs[idx],        # normalised coarse (on fine grid)
            "targets":       self.targets[idx],        # normalised residual
            "fine":          self.fine[idx],           # raw 0.25° fine field
            "coarse_interp": self.coarse_interp[idx],  # raw 1.5° coarse on fine grid
            "doy":           self.doy_norm[idx],
            "hour":          self.hour_norm[idx],
        }
