#!/usr/bin/env python3
# ============================================================
# calc_skt_obs_trends_v1.py
# ERA5 SKT obs. trend — correspond to daily
#
# Same logic wirh calc_era5_obs_trends_v2.py, but the data source is from: 
#   ERA5 skin temperature（era5_skin_temperature_1980_2022.nc）
#   → monthly average data connot do daily average, transfers into:
#     Only take GEMStat measured month and use the monthly SKT average of the month
#     → Same logic as ERA5 obs.(Only take values when it observted
#
# Usage: 
#   python calc_skt_obs_trends_v1.py --min_years 5
#   python calc_skt_obs_trends_v1.py --min_years 8
#   python calc_skt_obs_trends_v1.py --min_years 10
#
# Output: era5_skt_obs_trends_v1_Xy.csv（X = min_years）
# ============================================================
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import time, argparse, warnings
warnings.filterwarnings('ignore')

# ── parameters ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--min_years', type=int, default=5)
args = parser.parse_args()
MIN_YEARS = args.min_years

DATA_DIR    = '/work/home/H.Jason421/water_temp_for_publish/data/'
SKT_PATH    = '/work5/ERA5/era5_skin_temperature_1980_2022.nc'
GEMSTAT_RAW = '/work5/H.Jason421/GEMStat_new/Water_Temperature.csv'
OUT_PATH    = DATA_DIR + f'era5_skt_obs_trends_v1_{MIN_YEARS}y.csv'

ANALYSIS_PERIOD = (1990, 2020)

t0 = time.time()
print(f'ERA5 SKT obs. trend v1  MIN_YEARS={MIN_YEARS}', flush=True)
print(f'輸出：{OUT_PATH}', flush=True)

# ── Step 1: Read GEMStat to build every (year, month) at each station whether observated ──────────────────
print('\n[1/4] Read GEMStat Origin Data...', flush=True)
df_raw = pd.read_csv(
    GEMSTAT_RAW, encoding='latin-1',
    usecols=['GEMS Station Number', 'Sample Date', 'Value'],
    low_memory=False
)
df_raw.columns = ['station_id', 'date', 'value']
df_raw = df_raw.dropna(subset=['value', 'date'])
df_raw['date']  = pd.to_datetime(df_raw['date'], errors='coerce')
df_raw = df_raw.dropna(subset=['date'])
df_raw['year']  = df_raw['date'].dt.year
df_raw['month'] = df_raw['date'].dt.month
df_raw = df_raw[
    (df_raw['year'] >= ANALYSIS_PERIOD[0]) &
    (df_raw['year'] <= ANALYSIS_PERIOD[1])
]
df_raw['value'] = pd.to_numeric(df_raw['value'], errors='coerce')
df_raw = df_raw[df_raw['value'].between(-5, 40)]

# The set of (year, month) for every measurement
obs_ym = set(
    zip(df_raw['station_id'], df_raw['year'], df_raw['month'])
)
print(f'  Stations: {df_raw["station_id"].nunique()}', flush=True)
print(f'  the set of (station, year, month): {len(obs_ym):,}', flush=True)

# ── Step 2: station coordinate───────────────────────────────────────────────────────────
print('\n[2/4] station coordinate...', flush=True)
df_wt = pd.read_csv(DATA_DIR + f'global_wtemp_monthly_trends_v{MIN_YEARS}y.csv')
si = (df_wt[['station_id', 'latitude', 'longitude']]
      .drop_duplicates('station_id')
      .reset_index(drop=True))

valid_sids = set(df_raw['station_id'].unique())
si = si[si['station_id'].isin(valid_sids)].reset_index(drop=True)
print(f'  Target stations: {len(si)}', flush=True)

# ── Step 3: opened SKT, and batch extract time series data for all monitoring stations. ─────────────────────────────
print('\n[3/4] Read ERA5 SKT...', flush=True)
ds = xr.open_dataset(SKT_PATH)
ds = ds.sel(time=slice(f'{ANALYSIS_PERIOD[0]}-01', f'{ANALYSIS_PERIOD[1]}-12'))

lats  = xr.DataArray(si['latitude'].values,              dims='station')
lons  = xr.DataArray((si['longitude'].values % 360),     dims='station')
skt_all = ds['skt'].sel(latitude=lats, longitude=lons,
                         method='nearest').values - 273.15
# shape: (n_months, n_stations)

times      = pd.to_datetime(ds.time.values)
years_arr  = times.year.values
months_arr = times.month.values
print(f'  extract shape done = {skt_all.shape}', flush=True)

# ── Step 4: Only keep (year, month) with GEMStat observations and calculate trend ────────────────
print('\n[4/4] Calculate trend...', flush=True)

sid_list = si['station_id'].values
lat_list = si['latitude'].values
lon_list = si['longitude'].values

results = []
for s_idx, sid in enumerate(sid_list):
    lat = lat_list[s_idx]
    lon = lon_list[s_idx]

    for mo in range(1, 13):
        mo_mask = (months_arr == mo)
        yrs = years_arr[mo_mask]
        vals = skt_all[mo_mask, s_idx]

        # Take the year only GEMStat observated
        keep = np.array([
            (sid, int(yr), int(mo)) in obs_ym
            for yr in yrs
        ])
        yrs_use  = yrs[keep]
        vals_use = vals[keep]

        valid = np.isfinite(vals_use)
        if valid.sum() < MIN_YEARS:
            continue

        sl, _, _, p, _ = stats.linregress(
            yrs_use[valid].astype(float), vals_use[valid]
        )
        results.append({
            'station_id':  sid,
            'latitude':    lat,
            'longitude':   lon,
            'month':       mo,
            'n_years':     int(valid.sum()),
            'trend_dec':   sl * 10,
            'p_value':     p,
            'significant': 'Yes' if p < 0.05 else 'No',
        })

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_PATH, index=False)

elapsed = time.time() - t0
print(f'\nFinish! {len(df_out):,} records, {df_out["station_id"].nunique()} station', flush=True)
print(f'Output: {OUT_PATH}', flush=True)
print(f'Total time spent: {elapsed:.0f} seconds', flush=True)
print(df_out[['trend_dec', 'p_value']].describe().round(4))
