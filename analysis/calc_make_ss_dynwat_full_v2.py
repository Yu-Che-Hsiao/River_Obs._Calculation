#!/usr/bin/env python3
# ============================================================
# make_ss_dynwat_full_v2.py
#
# Convert output calc_dynwat_obs_trends_v2.py
#     data/dynwat_monthly_trends_v2_{N}y.csv
# into the format required by the station-violin figure
#     data/ss_dynwat_full_trends_{N}y.csv
#
# This is purely a column-name conversion; no trend is recomputed.
# This works because calc_dynwat_obs_trends_v2.py uses dynwat_station_monthly_v1.pkl,
# which contains all available years and is not filtered by the GEMStat observation months,
# so its output is already the full DynWat record at the station locations (the ALL form).
#
# Usage
#   python make_ss_dynwat_full_v2.py                # default min_years=5
#   python make_ss_dynwat_full_v2.py --min_years 8
# ============================================================
import os, sys, argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--min_years', type=int, default=5)
args = ap.parse_args()
MY = args.min_years

DATA = '/work/home/H.Jason421/water_temp_for_publish/data/'
SRC  = DATA + f'dynwat_monthly_trends_v2_{MY}y.csv'
OUT  = DATA + f'ss_dynwat_full_trends_{MY}y.csv'

# Standard column order for ss_*.csv (consistent with save() in calc_station_scale.py)
COLS = ['station_id', 'latitude', 'longitude', 'water_type',
        'month', 'n_years', 'trend_dec', 'p_value', 'significant', 'r2']

RENAME = {'dynwat_trend_dec': 'trend_dec',
          'dynwat_p':         'p_value',
          'dynwat_r2':        'r2'}

print('=' * 60)
print(f'make_ss_dynwat_full_v2   MIN_YEARS={MY}')
print(f'Reading:{SRC}')
print(f'Output: {OUT}')
print('=' * 60)

if not os.path.exists(SRC):
    sys.exit(f'[Abort] {SRC}not found')

df = pd.read_csv(SRC)
df['station_id'] = df['station_id'].astype(str)

need = set(RENAME) | {'station_id', 'latitude', 'longitude', 'water_type', 'month', 'n_years'}
missing = need - set(df.columns)
if missing:
    sys.exit(f'[Abort] Missing columns:{sorted(missing)}')

out = df.rename(columns=RENAME)

# Rebuild the 'significant' column from p_value if it does not exist
if 'significant' not in out.columns:
    out['significant'] = (out['p_value'] < 0.05).map({True: 'Yes', False: 'No'})

out = out[COLS]

# ── Sanity checks ────────────────────────────────────────────────
y = out['n_years']
print(f'\nRows        :{len(out):,}')
print(f'Stations        :{out["station_id"].nunique()}')
print(f'n_years     :median={y.median():.0f}  p25={y.quantile(.25):.0f}  '
      f'p75={y.quantile(.75):.0f}  max={y.max()}')
print(f'Median monthly trend:{out["trend_dec"].median():+.3f} °C/dec')

# Frozen reaches: DynWat holds winter water temperature at 0 °C, giving a constant
# series for which linregress cannot return a p value.
n_const = int(out['p_value'].isna().sum())
if n_const:
    print(f'Constant series:{n_const} records (trend_dec=0, p_value=NaN; frozen reaches, expected)')

if y.median() < 18:
    print('\n[Warning] Median n_years is low; this dataset appears to have been filtered by observation months,')
    print('  not a full record. Please verify the source of dynwat_station_monthly_v1.pkl.')

# Compare against SUB (if available)
OBS = DATA + f'ss_dynwat_obs_trends_{MY}y.csv'
if os.path.exists(OBS):
    sub = pd.read_csv(OBS)
    sub['station_id'] = sub['station_id'].astype(str)
    n_all, n_sub = out['station_id'].nunique(), sub['station_id'].nunique()
    print(f'\nCompare SUB    :ALL {n_all} Stations vs SUB {n_sub} Stations')
    print(f'             ALL n_years median={y.median():.0f}  '
          f'SUB n_years median={sub["n_years"].median():.0f}')
    if n_all <= n_sub:
        print('[Warning] ALL does not have more stations than SUB; they may be the same dataset.')

out.to_csv(OUT, index=False)
print(f'\nDone: {OUT}')
