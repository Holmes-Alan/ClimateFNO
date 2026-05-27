#!/bin/bash
# ============================================================
#  download_era5_europe.sh
#  Downloads ERA5 surface variables over Europe from NCAR RDA
#  (ds633.0) and preprocesses them into yearly sample files.
#
#  Variables downloaded:
#    - VAR_2T  : 2-metre temperature
#    - TP      : total precipitation
#
#  Usage:
#    bash download_ERA5/download_era5_europe.sh
#
#  Requirements:
#    - wget, python3, xarray, netcdf4
#    - Free NCAR RDA account: https://rda.ucar.edu/
#      (set NCAR_USER and NCAR_PASS below, or export them beforehand)
# ============================================================

set -e

NCAR_USER="${NCAR_USER:-your_email@example.com}"
NCAR_PASS="${NCAR_PASS:-your_password}"

DIR="../data"
mkdir -p "${DIR}"

YEAR_START=1979
YEAR_END=2022

# Log in once to get a session cookie
echo "Logging in to NCAR RDA …"
wget --save-cookies "${DIR}/.rda_cookies" \
     --keep-session-cookies \
     --post-data="email=${NCAR_USER}&passwd=${NCAR_PASS}&action=login" \
     -q -O /dev/null \
     "https://rda.ucar.edu/cgi-bin/login"

for year in $(seq ${YEAR_START} 1 ${YEAR_END}); do
    for m in $(seq 1 1 12); do
        month=$(printf "%02d" ${m})

        # Determine last day of month
        case ${m} in
            4|6|9|11) last_day=30 ;;
            2)
                if [ $(( year % 4 )) -eq 0 ] && \
                   ( [ $(( year % 100 )) -ne 0 ] || \
                     [ $(( year % 400 )) -eq 0 ] ); then
                    last_day=29
                else
                    last_day=28
                fi ;;
            *) last_day=31 ;;
        esac

        echo "Downloading ${year}-${month} …"

        BASE="https://data.rda.ucar.edu/ds633.0/e5.oper.an.sfc/${year}${month}"
        OPTS="--load-cookies ${DIR}/.rda_cookies -N -c -P ${DIR}"

        # 2-metre temperature
        wget ${OPTS} \
          "${BASE}/e5.oper.an.sfc.128_167_2t.ll025sc.${year}${month}0100_${year}${month}${last_day}23.nc"

        # Total precipitation (accumulated, needs de-accumulation — see note)
        wget ${OPTS} \
          "${BASE}/e5.oper.fc.sfc.accumu.128_228_tp.ll025sc.${year}${month}0106_${year}${month}${last_day}06.nc"

        echo "  Subsampling ${year}-${month} …"
        python3 download_ERA5/preprocess_subsample.py \
            --year ${year} --month ${m} --last_day ${last_day} --remove_files

        echo "  Done: ${year}-${month}"
    done

    echo "Concatenating months → samples_${year}.nc …"
    python3 download_ERA5/preprocess_concat_year.py --year ${year} --remove_files
    echo "Done: ${year}"
done

echo "All downloads complete."
