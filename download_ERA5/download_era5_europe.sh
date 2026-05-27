#!/bin/bash
# ============================================================
#  download_era5_europe.sh
#
#  Downloads ERA5 over Europe at TWO native resolutions from
#  the Copernicus Climate Data Store (CDS) via cdsapi:
#
#    Fine   : 0.25° × 0.25°  (ERA5 full resolution)
#    Coarse : 1.50° × 1.50°  (ERA5 coarse grid — same product,
#                              requested at lower resolution)
#
#  Variables:
#    - 2m temperature (t2m)
#    - Total precipitation (tp)
#
#  Domain: Europe  10°W–30°E, 35°N–72°N
#  Time:   monthly files, 8 random time steps per month
#          → ~360 samples/year after concat (manageable on one GPU)
#
#  Usage:
#    bash download_ERA5/download_era5_europe.sh
#
#  Requirements:
#    - cdsapi  (pip install cdsapi)
#    - A CDS account with ~/.cdsapirc configured:
#        url: https://cds.climate.copernicus.eu/api/v2
#        key: <UID>:<API-KEY>
#      See: https://cds.climate.copernicus.eu/api-how-to
# ============================================================

set -e

YEAR_START=1979
YEAR_END=2022   # exclusive → downloads 1979-2021
N_PER_MONTH=8   # random time steps kept per month (GPU-friendly size)

echo "Downloading ERA5 Europe (fine 0.25° + coarse 1.5°) for ${YEAR_START}–$((YEAR_END-1))"
echo "Samples per month: ${N_PER_MONTH}"

for year in $(seq ${YEAR_START} 1 $((YEAR_END - 1))); do
    for m in $(seq 1 1 12); do
        month=$(printf "%02d" ${m})
        echo "── ${year}-${month} ──"

        python3 download_ERA5/download_month_cds.py \
            --year ${year} --month ${m} \
            --n_samples ${N_PER_MONTH} \
            --remove_raw

        echo "  Done: ${year}-${month}"
    done

    echo "Concatenating months → samples_{fine,coarse}_${year}.nc …"
    python3 download_ERA5/preprocess_concat_year.py --year ${year} --remove_files
    echo "Done: ${year}"
done

echo "All downloads complete."
