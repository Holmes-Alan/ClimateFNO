"""
preprocess_concat_year.py
-------------------------
Adapted from ClimateDiffuse (Watt & Mansfield, 2024).

Concatenates monthly  samples_{year}{month:02d}.nc  files into a single
yearly  samples_{year}.nc  file and optionally removes the monthly files.

Usage
-----
    python download_ERA5/preprocess_concat_year.py --year 1979 --remove_files
"""

import argparse
import os

import xarray as xr

DATADIR = "../data/"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year",         type=int, required=True)
    p.add_argument("--remove_files", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    year = args.year

    datasets = []
    for m in range(1, 13):
        path = DATADIR + f"samples_{year}{m:02d}.nc"
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping.")
            continue
        ds = xr.open_dataset(path, engine="netcdf4")
        datasets.append(ds)

    if not datasets:
        print(f"No monthly files found for {year}. Aborting.")
        return

    ds_year = xr.concat(datasets, dim="time")
    out_path = DATADIR + f"samples_{year}.nc"
    ds_year.to_netcdf(out_path)
    print(f"  Saved: {out_path}  ({len(ds_year.time)} time steps)")

    if args.remove_files:
        for m in range(1, 13):
            path = DATADIR + f"samples_{year}{m:02d}.nc"
            if os.path.exists(path):
                os.remove(path)
                print(f"  Removed: {path}")


if __name__ == "__main__":
    main()
