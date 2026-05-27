"""
ERA5 Europe downscaling dataset.
Adapted from ClimateDiffuse (Watt & Mansfield, 2024) for the European domain,
targeting 2-metre temperature (T2m) and total precipitation (TP).

The dataset produces paired (low-resolution input, high-resolution target) samples
by coarsening the native 0.25° ERA5 fields to a lower resolution and using the
original fields as the supervision target.
"""

import numpy as np
import torch
import torchvision.transforms as T
import xarray as xr


# ── Default normalisation statistics (computed over ERA5 Europe 1979-2017) ──────
# Update these by running `compute_stats.py` on your own data.
DEFAULT_MEAN = torch.tensor([281.0,  0.0])   # T2m [K], TP [m]
DEFAULT_STD  = torch.tensor([ 14.0,  3e-4])  # rough estimates


class ERA5EuropeDataset(torch.utils.data.Dataset):
    """
    Paired coarse/fine ERA5 dataset over Europe for two variables:
      - VAR_2T  : 2-metre temperature [K]
      - TP      : total precipitation [m]

    Resolution pairs (default 4×):
      fine   : 0.25°  → ~128 × 96  grid points over Europe bounding box
      coarse : 1.00°  → ~ 32 × 24  grid points  (coarsen_factor=4)

    The input to the FNO is the coarse field bilinearly interpolated back to
    the fine grid (standard super-resolution setup).  The target is the
    residual between the fine field and this interpolated coarse field,
    following the same residual formulation as ClimateDiffuse.

    Parameters
    ----------
    data_dir : str
        Path to directory containing `samples_{year}.nc` files.
    year_start : int
        First year (inclusive) to load.
    year_end : int
        Last year (exclusive) to load.
    coarsen_factor : int
        Spatial downscaling factor (default 4 → 0.25° fine, 1.00° coarse).
    mean, std : torch.Tensor of shape (n_var,)
        Per-channel normalisation statistics for the raw fine-resolution data.
    dem_file : str or None
        Path to a NetCDF file containing the `z` (geopotential) and `lsm`
        (land-sea mask) fields.  When provided, both fields are appended to
        the input as extra conditioning channels (as in ClimateDiffuse).
    """

    # European bounding box (N→S latitude ordering in ERA5)
    LAT_SLICE = slice(72.0, 34.0)
    LON_SLICE = slice(345.0, 45.0)   # wraps: 345°E–360°E + 0°E–45°E
    # For simplicity we use a non-wrapping slice; adjust if your download
    # covers −25°W–45°E directly:
    LAT_SLICE = slice(72.0, 34.0)
    LON_SLICE = slice(335.0, 45.0)   # covers −25°W to 45°E as 335°E–405°E

    VAR_NAMES = ["VAR_2T", "TP"]

    def __init__(
        self,
        data_dir: str,
        year_start: int = 1979,
        year_end:   int = 2017,
        coarsen_factor: int = 4,
        mean: torch.Tensor = DEFAULT_MEAN,
        std:  torch.Tensor = DEFAULT_STD,
        dem_file: str | None = None,
    ):
        self.data_dir      = data_dir
        self.coarsen_factor = coarsen_factor
        self.mean = mean
        self.std  = std

        # ── Load all yearly files ────────────────────────────────────────────
        print(f"Loading ERA5 Europe data for years {year_start}–{year_end-1} …")
        datasets = []
        for year in range(year_start, year_end):
            path = f"{data_dir}/samples_{year}.nc"
            ds   = xr.open_dataset(path, engine="netcdf4")
            ds   = self._crop_europe(ds)
            datasets.append(ds)
        ds_all = xr.concat(datasets, dim="time")
        print(f"  Loaded {len(ds_all.time)} time steps.")

        self.lat = ds_all.latitude
        self.lon = ds_all.longitude
        self.H   = len(self.lat)
        self.W   = len(self.lon)
        self.fine_shape   = (self.H, self.W)
        self.coarse_shape = (self.H // coarsen_factor, self.W // coarsen_factor)

        # ── Build fine-resolution tensor (ntime, 2, H, W) ───────────────────
        t2m = torch.from_numpy(ds_all["VAR_2T"].values).float()   # (T, H, W)
        tp  = torch.from_numpy(ds_all["TP"].values).float()        # (T, H, W)
        fine = torch.stack([t2m, tp], dim=1)                       # (T, 2, H, W)

        # ── Build coarse input by resizing down then back up ─────────────────
        coarsen  = T.Resize(self.coarse_shape,
                            interpolation=T.InterpolationMode.BILINEAR,
                            antialias=True)
        upsample = T.Resize(self.fine_shape,
                            interpolation=T.InterpolationMode.BILINEAR,
                            antialias=True)
        coarse = upsample(coarsen(fine))                           # (T, 2, H, W)

        # ── Residual target ──────────────────────────────────────────────────
        residual = fine - coarse                                   # (T, 2, H, W)

        # Keep un-normalised copies for plotting / evaluation
        self.fine   = fine
        self.coarse = coarse

        # ── Normalise ────────────────────────────────────────────────────────
        norm_raw      = T.Normalize(mean.tolist(), std.tolist())
        residual_mean = residual.mean(dim=(0, 2, 3))
        residual_std  = residual.std(dim=(0, 2, 3)).clamp(min=1e-6)
        self.residual_mean = residual_mean
        self.residual_std  = residual_std
        norm_residual = T.Normalize(residual_mean.tolist(), residual_std.tolist())

        self.inputs  = norm_raw(coarse)       # (T, 2, H, W)  normalised coarse
        self.targets = norm_residual(residual) # (T, 2, H, W)  normalised residual

        # ── Optional: DEM + land-sea mask ────────────────────────────────────
        self.n_input_channels = 2
        if dem_file is not None:
            print("  Loading DEM / land-sea mask …")
            ds_dem = xr.open_dataset(dem_file, engine="netcdf4")
            ds_dem = self._crop_europe(ds_dem)

            # Geopotential → normalise
            z   = torch.from_numpy(ds_dem["z"].values).float()
            z   = (z - z.mean()) / z.std().clamp(min=1e-6)
            # Land-sea mask → keep as-is (0/1)
            lsm = torch.from_numpy(ds_dem["lsm"].values).float()

            # Both are (H, W) → broadcast over time
            T_   = self.inputs.shape[0]
            z    = z.unsqueeze(0).expand(T_, -1, -1)    # (T, H, W)
            lsm  = lsm.unsqueeze(0).expand(T_, -1, -1)  # (T, H, W)
            aux  = torch.stack([z, lsm], dim=1)         # (T, 2, H, W)
            self.inputs = torch.cat([self.inputs, aux], dim=1)  # (T, 4, H, W)
            self.n_input_channels = 4
            print("  DEM channels added.  Input channels: 4.")

        # ── Time metadata ────────────────────────────────────────────────────
        time = ds_all.time.dt
        self.doy_norm  = torch.from_numpy(
            ((time.month.values - 1) * 30 + (time.day.values - 1)) / 360.0
        ).float()
        self.hour_norm = torch.from_numpy(time.hour.values / 24.0).float()

        print("Dataset ready.")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _crop_europe(self, ds: xr.Dataset) -> xr.Dataset:
        """Crop dataset to the European bounding box."""
        return ds.sel(
            latitude=self.LAT_SLICE,
            longitude=self.LON_SLICE,
        )

    def inverse_normalise_residual(self, residual_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised residual back to physical units."""
        return (residual_norm
                * self.residual_std[:, None, None]
                + self.residual_mean[:, None, None])

    def residual_to_fine(self, residual_norm: torch.Tensor,
                         coarse: torch.Tensor) -> torch.Tensor:
        """Reconstruct fine-resolution field from predicted residual."""
        return coarse + self.inverse_normalise_residual(residual_norm)

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "inputs":  self.inputs[idx],          # normalised coarse (+DEM)
            "targets": self.targets[idx],          # normalised residual
            "fine":    self.fine[idx],             # raw fine field
            "coarse":  self.coarse[idx],           # raw coarse field
            "doy":     self.doy_norm[idx],
            "hour":    self.hour_norm[idx],
        }
