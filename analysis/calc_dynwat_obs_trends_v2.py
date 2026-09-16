#!/usr/bin/env python3
# ============================================================
# calc_dynwat_obs_trends_v2.py
# DynWat obs. trend calculation, suppporting --min_years parameter
#
# Usage:
#   python calc_dynwat_obs_trends_v2.py --min_years 5
#   python calc_dynwat_obs_trends_v2.py --min_years 8
#   python calc_dynwat_obs_trends_v2.py --min_years 10
#
# Output: dynwat_monthly_trends_v2_Xy.csv（X = min_years）
# ============================================================
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import argparse
import warnings
warnings.filterwarnings('ignore')

# ── parameter ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--min_years', type=int, default=5,
                    help='minimum number of valid years required to compute a trend（default 5）')
args = parser.parse_args()
MIN_YEARS = args.min_years

# ── path ──────────────────────────────────────────────────────────────────────
DATA_DIR        = '/work/home/H.Jason421/water_temp_for_publish/data/'
CACHE_DYNWAT    = DATA_DIR + 'dynwat_station_monthly_v1.pkl'   # original cache unchanging
WTEMP_CSV       = DATA_DIR + f'global_wtemp_monthly_trends_v{MIN_YEARS}y.csv'
OUT_PATH        = DATA_DIR + f'dynwat_monthly_trends_v2_{MIN_YEARS}y.csv'

MIN_DATA_FOR_TREND = 1   # DynWat is monthly data even if one data per year is usable.
# ─────────────────────────────────────────────────────────────────────────────

print(f'DynWat obs. trend calculated MIN_YEARS={MIN_YEARS}')
print(f'Read TR trend：{WTEMP_CSV}')
print(f'Output：{OUT_PATH}')

# ── Step 1：Read GEMStat station(correspond versions)───────────────────────────────────────
print('\n[1/3] Read GEMStat station...')
df_wt = pd.read_csv(WTEMP_CSV)
stations_all = (df_wt[['station_id','latitude','longitude','water_type']]
                .drop_duplicates('station_id'))

# Estabilish station_id → row dict(speeding searching and enforce str)
station_dict = {
    str(row['station_id']): row
    for _, row in stations_all.iterrows()
}
river_ids = set(
    str(sid) for sid, row in station_dict.items()
    if row['water_type'] == 'River station'
)
print(f'  River station number：{len(river_ids)}')

# ── Step 2：Read DynWat cache ───────────────────────────────────────────────────
print('\n[2/3] Read DynWat cache...')
with open(CACHE_DYNWAT, 'rb') as f:
    dynwat_station_data = pickle.load(f)
print(f'  DynWat cache grid numbers：{len(dynwat_station_data)}')

# check intersection
overlap = set(str(k) for k in dynwat_station_data.keys()) & river_ids
print(f'  DynWat ∩ River station:{len(overlap)} stations')

# ── Step 3：calculate the trend every station and every month────────────────────────────────────────────────
print(f'\n[3/3] calculate trend（MIN_YEARS={MIN_YEARS}）...')
results         = []
skipped_notriver = 0
skipped_nan      = 0

for k, (sid, df_s) in enumerate(dynwat_station_data.items()):
    if k % 200 == 0:
        print(f'  Progress: {k}/{len(dynwat_station_data)}  results={len(results)}')

    sid_str = str(sid)

    # Only deal with the River station（correspond versions）
    if sid_str not in station_dict:
        skipped_notriver += 1
        continue
    row = station_dict[sid_str]
    if row['water_type'] != 'River station':
        skipped_notriver += 1
        continue

    # Clean outlier
    df_s = df_s.copy()
    df_s.loc[df_s['dynwat_wtemp'].abs() > 100, 'dynwat_wtemp'] = np.nan

    if df_s['dynwat_wtemp'].notna().sum() == 0:
        skipped_nan += 1
        continue

    for month in range(1, 13):
        md = df_s[df_s['month'] == month].dropna(subset=['dynwat_wtemp'])
        ym = md.groupby('year')['dynwat_wtemp'].mean().reset_index()
        ym.columns = ['year', 'val_mean']

        if len(ym) < MIN_YEARS:
            continue

        x = ym['year'].values
        y = ym['val_mean'].values
        mask = np.isfinite(y)
        if mask.sum() < MIN_YEARS:
            continue

        sl, _, r, p, _ = stats.linregress(x[mask], y[mask])
        results.append({
            'station_id':       sid_str,
            'latitude':         float(row['latitude']),
            'longitude':        float(row['longitude']),
            'water_type':       row['water_type'],
            'month':            month,
            'n_years':          int(mask.sum()),
            'dynwat_trend_yr':  sl,
            'dynwat_trend_dec': sl * 10,
            'dynwat_r2':        r ** 2,
            'dynwat_p':         p,
            'significant':      'Yes' if p < 0.05 else 'No',
        })

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_PATH, index=False)

print(f'\nFinish! {len(df_out):,} records with station-month trends')
print(f'The stations with trends: {df_out["station_id"].nunique() if len(df_out) > 0 else 0}')
print(f'Skip (Non-River/cannot found): {skipped_notriver}')
print(f'Skip (All Non-River): {skipped_nan}')
if len(df_out) > 0:
    print(f'Median for monthly trends: {df_out["dynwat_trend_dec"].median():.3f} °C/dec')
print(f'Output:{OUT_PATH}')
