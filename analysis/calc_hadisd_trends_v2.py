#!/usr/bin/env python3
# ============================================================
# calc_hadisd_trends_v2.py
# Calculate HadISD TA trend (Month by month version which the logic is same as ERA5 obs.)
#
# Usage:
#   python calc_hadisd_trends_v2.py --min_years 5
#   python calc_hadisd_trends_v2.py --min_years 8
#   python calc_hadisd_trends_v2.py --min_years 10
#
# Output:hadisd_airtemp_trends_v3_Xy.csv（X = min_years）
# Field: station_id, latitude, longitude, water_type,
#        hadisd_id, dist_km, month, n_years,
#        trend_dec, r2, p_value, significant
# ============================================================
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.spatial import cKDTree
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

# ── parameters ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--min_years', type=int, default=5,
                    help='minimum number of valid years required to compute a trend（default 5）')
args = parser.parse_args()
MIN_YEARS = args.min_years

# ── path ──────────────────────────────────────────────────────────────────────
HADISD_DIR  = '/work7/L.chshih/ERA_Analysis/data/HadISD/nc/'
DATA_DIR    = '/work/home/H.Jason421/water_temp_for_publish/data/'
WTEMP_CSV   = DATA_DIR + f'global_wtemp_monthly_trends_v{MIN_YEARS}y.csv'
OUT_PATH    = DATA_DIR + f'hadisd_airtemp_trends_v3_{MIN_YEARS}y.csv'

YEAR_START       = 1990
YEAR_END         = 2020
MAX_DIST_KM      = 100.0
MIN_MONTHS_PER_YEAR = 3
# ─────────────────────────────────────────────────────────────────────────────

print(f'Calculate HadISD trend month by month MIN_YEARS={MIN_YEARS}')
print(f'Read TR trend: {WTEMP_CSV}')
print(f'Output:{OUT_PATH}')

# ── Step 1: Read GEMStat TR station coordinate ────────────────────────────────────────────
print('\n[1/4] Read GEMStat station location...')
df_wt = pd.read_csv(WTEMP_CSV)
stations = (df_wt[['station_id','latitude','longitude','water_type']]
            .drop_duplicates('station_id')
            .reset_index(drop=True))
print(f'  GEMStat station: {len(stations)}')

# ── Step 2: Build HadISD station location metadata ─────────────────────────────────────────
print('\n[2/4] Build HadISD station location list...')
hadisd_files = sorted(glob.glob(HADISD_DIR + '*.nc'))
print(f'  HadISD file numbers:{len(hadisd_files)}')

hadisd_meta = []
for fpath in hadisd_files:
    fname = os.path.basename(fpath)
    try:
        ds = xr.open_dataset(fpath)
        lat = float(ds['latitude'].values.flat[0])
        lon = float(ds['longitude'].values.flat[0])
        hadisd_id = fname.replace('.nc', '')
        ds.close()
        hadisd_meta.append({
            'hadisd_id':   hadisd_id,
            'hadisd_path': fpath,
            'lat':         lat,
            'lon':         lon,
        })
    except Exception:
        continue

df_hadisd = pd.DataFrame(hadisd_meta)
print(f'  Efficient HadISD station: {len(df_hadisd)}')

# ── Step 3: Spatial pairing (GEMStat ↔ HadISD, distance < MAX_DIST_KM)──────────────────
print(f'\n[3/4] Spatial pairing(Maximum Distance {MAX_DIST_KM} km)...')

def latlon_to_xyz(lat, lon):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])

EARTH_R = 6371.0
hadisd_xyz   = latlon_to_xyz(df_hadisd['lat'].values, df_hadisd['lon'].values)
tree         = cKDTree(hadisd_xyz)
station_xyz  = latlon_to_xyz(stations['latitude'].values, stations['longitude'].values)
dists_chord, idxs = tree.query(station_xyz, k=1)
dists_km = 2 * EARTH_R * np.arcsin(np.clip(dists_chord / 2, 0, 1))

pairs = []
for i, row in stations.iterrows():
    dist = dists_km[i]
    if dist > MAX_DIST_KM:
        continue
    h = df_hadisd.iloc[idxs[i]]
    pairs.append({
        'station_id':  row['station_id'],
        'latitude':    row['latitude'],
        'longitude':   row['longitude'],
        'water_type':  row['water_type'],
        'hadisd_id':   h['hadisd_id'],
        'hadisd_path': h['hadisd_path'],
        'dist_km':     dist,
    })

pairs_df = pd.DataFrame(pairs)
print(f'  Pairing succeed{len(pairs_df)}stations')

# ── Step 4: Calculate HadISD TA trend every station month by month───────────────────────────────────
print(f'\n[4/4] Calculate HadISD TA trend every station month by month (MIN_YEARS={MIN_YEARS}...)')
results = []
total   = len(pairs_df)

for i, (_, row) in enumerate(pairs_df.iterrows()):
    if i % 200 == 0:
        print(f'  Proscess: {i}/{total}')
    try:
        ds = xr.open_dataset(row['hadisd_path'])

        # Find TA variables
        temp_var = None
        for vname in ['temperatures', 'T', 'temp', 'air_temperature', 't2m']:
            if vname in ds:
                temp_var = vname
                break
        if temp_var is None:
            ds.close()
            continue

        # Find the time dimension
        time_dim = None
        for tdim in ['time', 'dates', 'date']:
            if tdim in ds.dims or tdim in ds.coords:
                time_dim = tdim
                break
        if time_dim is None:
            ds.close()
            continue

        df_t = ds[temp_var].to_dataframe().reset_index()
        df_t = df_t.rename(columns={temp_var: 'temp', time_dim: 'time'})
        df_t['time'] = pd.to_datetime(df_t['time'], errors='coerce')
        df_t = df_t.dropna(subset=['time', 'temp'])
        df_t['year']  = df_t['time'].dt.year
        df_t['month'] = df_t['time'].dt.month
        df_t = df_t[(df_t['year'] >= YEAR_START) & (df_t['year'] <= YEAR_END)]

        # K → °C
        if df_t['temp'].median() > 200:
            df_t['temp'] = df_t['temp'] - 273.15

        # Remove missing value markers
        df_t = df_t[df_t['temp'].abs() < 100]

        if len(df_t) == 0:
            ds.close()
            continue

        # Monthly average
        df_monthly = df_t.groupby(['year','month'])['temp'].mean().reset_index()

        # ── Calculate trend month by month (Same logic with ERA5 obs.)────────────────────────────
        for month in range(1, 13):
            md = df_monthly[df_monthly['month'] == month]

            # QC of years: Data for at least this month must be available every year.
            if len(md) < MIN_YEARS:
                continue

            x = md['year'].values.astype(float)
            y = md['temp'].values
            mask = np.isfinite(y)
            if mask.sum() < MIN_YEARS:
                continue

            sl, _, r, p, _ = stats.linregress(x[mask], y[mask])
            results.append({
                'station_id':  row['station_id'],
                'latitude':    row['latitude'],
                'longitude':   row['longitude'],
                'water_type':  row['water_type'],
                'hadisd_id':   row['hadisd_id'],
                'dist_km':     row['dist_km'],
                'month':       month,
                'n_years':     int(mask.sum()),
                'trend_dec':   sl * 10,
                'r2':          r ** 2,
                'p_value':     p,
                'significant': 'Yes' if p < 0.05 else 'No',
            })

        ds.close()
    except Exception:
        continue

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_PATH, index=False)

print(f'\nFinish!{len(df_out):,} records(station*month)')
print(f'The station with trend:{df_out["station_id"].nunique() if len(df_out) > 0 else 0}')
print(f'Monthly Distribution:')
print(df_out['month'].value_counts().sort_index().to_string())
print(f'Output: {OUT_PATH}')
