"""
preprocess_subsample.py
-----------------------
Adapted from ClimateDiffuse (Watt & Mansfield, 2024).

For a given year/month:
  1. Open the raw ERA5 NetCDF files for T2m and TP.
  2. Crop to the European bounding box.
  3. De-accumulate TP from accumulated hourly totals to per-step values.
  4. Randomly sub-sample 30 time steps to keep file sizes manageable.
  5. Save as  data/samples_{year}{month:02d}.nc

Usage
-----
    python download_ERA5/preprocess_subsample.py \\
        --year 1979 --month 1 --last_day 31 --remove_files
"""

import argparse
import os
import random

import numpy as np
import xarray as xr

# ── European bounding box ────────────────────────────────────────────────────
# ERA5 longitudes run 0–360°E;  −25°W = 335°E
LAT_SLICE = slice(72.0, 34.0)   # N → S (ERA5 latitude is descending)
LON_SLICE = slice(335.0, 45.0)  # 335°E → 45°E  (covers −25°W to 45°E)

N_SAMPLES_PER_MONTH = 30        # number of time steps to keep per month
DATADIR = "../data/"

# ── Variable name mapping (NCAR RDA filenames → xarray variable names) ───────
T2M_CODE = "128_167_2t"
TP_CODE  = "128_228_tp"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year",         type=int, required=True)
    p.add_argument("--month",        type=int, required=True)
    p.add_argument("--last_day",     type=int, required=True)
    p.add_argument("--remove_files", action="store_true",
                   help="Delete raw downloaded files after processing")
    return p.parse_args()


def deaccumulate(da: xr.DataArray) -> xr.DataArray:
    """
    Convert accumulated TP (m) to per-step precipitation by differencing.
    ERA5 TP resets every 6 hours; we compute the forward difference and
    clip negatives (which arise at reset points) to zero.
    """
    diff = da.diff(dim="time")
    diff = diff.where(diff >= 0, 0.0)
    # Drop the first time step (no predecessor)
    return diff


def main():
    args  = parse_args()
    year  = args.year
    month = args.month

    # reproducible but different per year/month
    random.seed(year * 12 + month)

    # ── Open T2m ──────────────────────────────────────────────────────────────
    t2m_file = (f"e5.oper.an.sfc.{T2M_CODE}.ll025sc."
                f"{year}{month:02d}0100_{year}{month:02d}{args.last_day}23.nc")
    ds_t2m = xr.open_dataset(DATADIR + t2m_file, engine="netcdf4")
    ds_t2m = ds_t2m.sel(latitude=LAT_SLICE, longitude=LON_SLICE)

    # ── Open TP and de-accumulate ─────────────────────────────────────────────
    tp_file = (f"e5.oper.fc.sfc.accumu.{TP_CODE}.ll025sc."
               f"{year}{month:02d}0106_{year}{month:02d}{args.last_day}06.nc")
    ds_tp = xr.open_dataset(DATADIR + tp_file, engine="netcdf4")
    ds_tp = ds_tp.sel(latitude=LAT_SLICE, longitude=LON_SLICE)
    ds_tp["TP"] = deaccumulate(ds_tp["TP"])

    # ── Align on common time axis ─────────────────────────────────────────────
    common_times = np.intersect1d(ds_t2m.time.values, ds_tp.time.values)
    ds_t2m = ds_t2m.sel(time=common_times)
    ds_tp  = ds_tp.sel(time=common_times)

    # ── Random sub-sample ─────────────────────────────────────────────────────
    n_times  = len(common_times)
    indices  = list(range(n_times))
    random.shuffle(indices)
    indices  = sorted(indices[:N_SAMPLES_PER_MONTH])

    ds_t2m = ds_t2m.isel(time=indices)
    ds_tp  = ds_tp.isel(time=indices)

    # ── Merge and save ────────────────────────────────────────────────────────
    # Rename T2m variable for consistency with ClimateDiffuse convention
    ds_t2m = ds_t2m.rename({"VAR_2T": "VAR_2T"})   # already correct
    ds_merged = xr.merge([ds_t2m[["VAR_2T"]], ds_tp[["TP"]]])

    out_file = DATADIR + f"samples_{year}{month:02d}.nc"
    ds_merged.to_netcdf(out_file)
    print(f"  Saved: {out_file}  ({len(indices)} time steps)")

    # ── Optionally remove raw files ───────────────────────────────────────────
    if args.remove_files:
        for fname in [t2m_file, tp_file]:
            fpath = DATADIR + fname
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"  Removed: {fpath}")


if __name__ == "__main__":
    main()
