"""
download_month_cds.py
---------------------
Downloads one month of ERA5 data at BOTH 0.25° (fine) and 1.5° (coarse)
from the Copernicus Climate Data Store (CDS) using cdsapi, then sub-samples
to a fixed number of random time steps and saves paired NetCDF files.

Output files (written to ../data/):
    samples_fine_{year}{month:02d}.nc    – 0.25° grid, Europe
    samples_coarse_{year}{month:02d}.nc  – 1.5°  grid, Europe

Both files share the same randomly-selected time indices so every coarse
sample has an exactly paired fine-resolution counterpart.

Domain: 10°W–30°E, 35°N–72°N  (Western + Central Europe)
Variables: 2m temperature (t2m), total precipitation (tp)

Usage
-----
    python download_ERA5/download_month_cds.py \\
        --year 1979 --month 1 --n_samples 8 --remove_raw

Requirements
------------
    pip install cdsapi
    # Configure ~/.cdsapirc:
    #   url: https://cds.climate.copernicus.eu/api/v2
    #   key: <UID>:<API-KEY>
"""

import argparse
import os
import random

import cdsapi
import numpy as np
import xarray as xr

# ── Domain ───────────────────────────────────────────────────────────────────
# Europe bounding box: [N, W, S, E]  (CDS convention)
AREA = [72, -10, 35, 30]            # 35°N–72°N, 10°W–30°E
# As xarray slices (ERA5 lat is descending N→S, lon is 0–360 or −180–180
# depending on the product; CDS returns −180–180 when area includes negatives)
LAT_SLICE = slice(72, 35)           # N → S
LON_SLICE = slice(-10, 30)          # W → E

DATADIR   = "../data/"
VARIABLES = ["2m_temperature", "total_precipitation"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year",       type=int, required=True)
    p.add_argument("--month",      type=int, required=True)
    p.add_argument("--n_samples",  type=int, default=8,
                   help="Number of random time steps to keep per month")
    p.add_argument("--remove_raw", action="store_true",
                   help="Delete the full raw CDS downloads after sub-sampling")
    return p.parse_args()


def cds_request(c: cdsapi.Client, year: int, month: int,
                grid: str, out_path: str) -> None:
    """
    Submit one CDS request for a full month at the requested grid resolution.

    Parameters
    ----------
    grid : str
        CDS grid specification, e.g. "0.25/0.25" or "1.5/1.5".
    """
    if os.path.exists(out_path):
        print(f"    {out_path} already exists, skipping download.")
        return

    print(f"    Requesting {grid} data for {year}-{month:02d} …")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year":  str(year),
            "month": f"{month:02d}",
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  ["00:00", "06:00", "12:00", "18:00"],
            "area":  AREA,
            "grid":  grid,
            "format": "netcdf",
        },
        out_path,
    )
    print(f"    Saved raw: {out_path}")


def deaccumulate_tp(da: xr.DataArray) -> xr.DataArray:
    """
    ERA5 total precipitation is a forecast accumulation that resets every
    24 hours (at 00 UTC).  Convert to per-step values by differencing within
    each daily accumulation cycle and clipping negatives at reset points.
    """
    diff = da.diff(dim="valid_time")
    diff = diff.where(diff >= 0, 0.0)
    return diff


def load_and_crop(path: str) -> xr.Dataset:
    """Open a CDS NetCDF file, rename the time dimension if needed, and crop."""
    ds = xr.open_dataset(path, engine="netcdf4")
    # CDS may use 'valid_time' or 'time' depending on the API version
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    ds = ds.sel(latitude=LAT_SLICE, longitude=LON_SLICE)
    return ds


def subsample_and_save(ds_fine: xr.Dataset, ds_coarse: xr.Dataset,
                       year: int, month: int, n_samples: int) -> None:
    """
    Align fine and coarse on a common time axis, randomly sub-sample
    n_samples time steps, and save paired monthly files.
    """
    # De-accumulate TP in both datasets
    for ds in [ds_fine, ds_coarse]:
        ds["tp"] = deaccumulate_tp(ds["tp"])

    # Common time steps after de-accumulation (both lose first step)
    common = np.intersect1d(ds_fine.time.values, ds_coarse.time.values)
    ds_fine   = ds_fine.sel(time=common)
    ds_coarse = ds_coarse.sel(time=common)

    # Random sub-sample (reproducible per year/month)
    random.seed(year * 12 + month)
    indices = sorted(random.sample(range(len(common)), min(n_samples, len(common))))

    ds_fine   = ds_fine.isel(time=indices)
    ds_coarse = ds_coarse.isel(time=indices)

    # Rename variables to consistent internal names
    rename = {"t2m": "VAR_2T", "tp": "TP"}
    ds_fine   = ds_fine.rename({k: v for k, v in rename.items() if k in ds_fine})
    ds_coarse = ds_coarse.rename({k: v for k, v in rename.items() if k in ds_coarse})

    out_fine   = DATADIR + f"samples_fine_{year}{month:02d}.nc"
    out_coarse = DATADIR + f"samples_coarse_{year}{month:02d}.nc"
    ds_fine.to_netcdf(out_fine)
    ds_coarse.to_netcdf(out_coarse)
    print(f"    Saved: {out_fine}  ({len(indices)} steps)")
    print(f"    Saved: {out_coarse}")


def main():
    args = parse_args()
    os.makedirs(DATADIR, exist_ok=True)

    c = cdsapi.Client()

    raw_fine   = DATADIR + f"raw_fine_{args.year}{args.month:02d}.nc"
    raw_coarse = DATADIR + f"raw_coarse_{args.year}{args.month:02d}.nc"

    # Download fine (0.25°) and coarse (1.5°) from CDS
    cds_request(c, args.year, args.month, "0.25/0.25", raw_fine)
    cds_request(c, args.year, args.month, "1.5/1.5",   raw_coarse)

    # Load, crop to Europe, de-accumulate TP, sub-sample, save
    ds_fine   = load_and_crop(raw_fine)
    ds_coarse = load_and_crop(raw_coarse)
    subsample_and_save(ds_fine, ds_coarse, args.year, args.month, args.n_samples)

    # Optionally remove raw full-month files
    if args.remove_raw:
        for path in [raw_fine, raw_coarse]:
            if os.path.exists(path):
                os.remove(path)
                print(f"    Removed raw: {path}")


if __name__ == "__main__":
    main()
