#!/usr/bin/env python3
# ============================================================
# calc_era5_obs_trends_v2.py
# ERA5 obs. trend — date-matched version
#
# Changes (vs. v1):
#   v1: the years with water-temperature records at each station were
#       identified, and the monthly-mean air temperature of those years
#       was taken.
#   v2: the dates of each individual water-temperature record are
#       identified; ERA5 is first averaged to daily values, and only the
#       daily-mean air temperatures on the sampled days are averaged to
#       represent that month.
#       -> sampling variance matches GEMStat, giving an equally broad
#          distribution.
#
# Usage:
#   python calc_era5_obs_trends_v2.py --min_years 5
#   python calc_era5_obs_trends_v2.py --min_years 8
#   python calc_era5_obs_trends_v2.py --min_years 10
#
# Output: era5_obs_airtemp_trends_v2_Xy.csv（X = min_years）
# ============================================================
import numpy as np
import pandas as pd
import netCDF4 as nc4
from scipy import stats
import glob, os, time, argparse, warnings
warnings.filterwarnings('ignore')

# ── Parameters ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--min_years', type=int, default=5,
                    help='minimum number of valid years required to compute a trend（default 5）')
args = parser.parse_args()
MIN_YEARS = args.min_years

DATA_DIR    = '/work/home/H.Jason421/water_temp_for_publish/data/'
ERA5_DIR    = '/work5/ERA5/t2m/'
GEMSTAT_RAW = '/work5/H.Jason421/GEMStat_new/Water_Temperature.csv'
OUT_PATH    = DATA_DIR + f'era5_obs_airtemp_trends_v2_{MIN_YEARS}y.csv'

ANALYSIS_PERIOD = (1990, 2020)
# ─────────────────────────────────────────────────────────────────────────────

t0 = time.time()
print(f'ERA5 obs. trend calculation. v2（date-matched）MIN_YEARS={MIN_YEARS}', flush=True)
print(f'Output:{OUT_PATH}', flush=True)

# ── Step 1: Read GEMStat and bulid the daily measured set with (year, month) for every station ───────────
print('\n[1/4] Read GEMStat Data...', flush=True)
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
df_raw['day']   = df_raw['date'].dt.day
df_raw = df_raw[
    (df_raw['year'] >= ANALYSIS_PERIOD[0]) &
    (df_raw['year'] <= ANALYSIS_PERIOD[1])
]
# Clean TR outlier (just to get the date, but also clean the outlier)
df_raw['value'] = pd.to_numeric(df_raw['value'], errors='coerce')
df_raw = df_raw[df_raw['value'].between(-5, 40)]

# (year, month) for every station → the set of measured date
# station_dates[(sid, yr, mo)] = {1, 6, 10, ...}
print('Build an index of measurement dates by station and month...', flush=True)
station_dates = {}
for row in df_raw[['station_id','year','month','day']].itertuples(index=False):
    key = (row.station_id, row.year, row.month)
    if key not in station_dates:
        station_dates[key] = set()
    station_dates[key].add(row.day)

print(f'  Stations:{df_raw["station_id"].nunique()}', flush=True)
print(f'  (Station, Year, Month) Combinations:{len(station_dates):,}', flush=True)

# ── Step 2: Stations coordinate + ERA5 grid index ────────────────────────────────────────
print('\n[2/4] Stations coordinate + ERA5 grid index...', flush=True)
df_at = pd.read_csv(DATA_DIR + f'global_wtemp_monthly_trends_v{MIN_YEARS}y.csv')
df_at = df_at[df_at['water_type'] == 'River station'].copy()
si = (df_at[['station_id','latitude','longitude']]
      .drop_duplicates('station_id')
      .reset_index(drop=True))

# Only keep the stations with daily measured data
valid_sids = set(df_raw['station_id'].unique())
si = si[si['station_id'].isin(valid_sids)].reset_index(drop=True)
print(f' Target station:{len(si)}', flush=True)

# ERA5 grid coordinate
era5_files = sorted(glob.glob(ERA5_DIR + 'T_*.nc'))
era5_files_sel = [f for f in era5_files
                  if ANALYSIS_PERIOD[0] <= int(os.path.basename(f).split('_')[1]) <= ANALYSIS_PERIOD[1]]
print(f'  ERA5 File numbers:{len(era5_files_sel)}', flush=True)

ds_tmp = nc4.Dataset(era5_files_sel[0])
lats_grid = np.array(ds_tmp.variables['latitude'][:])
lons_grid = np.array(ds_tmp.variables['longitude'][:])
ds_tmp.close()

lat_idxs = np.array([int(np.argmin(np.abs(lats_grid - v))) for v in si['latitude'].values])
lon_idxs = np.array([int(np.argmin(np.abs(lons_grid - v))) for v in si['longitude'].values])
sid_list = si['station_id'].values
print(f'  The calculation on grids index complete', flush=True)

# ── Step 3: Read ERA5 month by month. Take the daily average first and then correspond the measured date.──────────────────────
print(f'\n[3/4] Read ERA5 month by month（Total: {len(era5_files_sel)} files）...', flush=True)

# Save Results: {station_id: {month: [(year, mean_t2m), ...]}}
# Use dict of dict of list to avoid not know the size
from collections import defaultdict
sta_monthly = defaultdict(lambda: defaultdict(list))
# sta_monthly[sid][month] = [(yr, t2m_mean_on_obs_days), ...]

n_stations = len(si)

for k, fpath in enumerate(era5_files_sel):
    fname = os.path.basename(fpath)
    parts = fname.replace('.nc', '').split('_')
    yr = int(parts[1])
    mo = int(parts[2])

    # Read ERA5 by hourrs → Take daily average first
    ds = nc4.Dataset(fpath)
    t2m_raw   = np.array(ds.variables['t2m'][:])    # (n_hours, 721, 1440)
    times_raw = np.array(ds.variables['valid_time'][:])  # unix seconds
    ds.close()

    # Transfer into the date
    times_dt = pd.to_datetime(times_raw, unit='s', utc=True)
    days_arr  = times_dt.day.values   # Day corresponding to each time step

    # daily average: take the average to each day
    unique_days = np.unique(days_arr)
    # daily_t2m shape: (n_days, n_stations)
    daily_t2m = np.full((len(unique_days), n_stations), np.nan)
    for d_idx, day in enumerate(unique_days):
        hour_mask = days_arr == day
        daily_t2m[d_idx, :] = t2m_raw[hour_mask][:, lat_idxs, lon_idxs].mean(axis=0)

    # To every station, find the (yr, mo) measured date and take the corresponding daily average
    for s_idx, sid in enumerate(sid_list):
        key = (sid, yr, mo)
        if key not in station_dates:
            continue
        obs_days = station_dates[key]   # e.g. {1, 6, 10}

        # Find the obs_days index in the unique_days
        day_indices = [
            np.where(unique_days == d)[0][0]
            for d in obs_days
            if d in unique_days
        ]
        if len(day_indices) == 0:
            continue

        t2m_vals = daily_t2m[day_indices, s_idx]  # (n_obs_days,)
        valid = np.isfinite(t2m_vals)
        if valid.sum() == 0:
            continue

        mean_t2m = float(t2m_vals[valid].mean()) - 273.15  # K → °C
        sta_monthly[sid][mo].append((yr, mean_t2m))

    if (k + 1) % 12 == 0:
        elapsed = time.time() - t0
        print(f'  {yr} Complete [{k+1}/{len(era5_files_sel)}]  {elapsed:.0f}s', flush=True)

print(' Read ERA5 completed', flush=True)

# ── Step 4: calculate trends for every station month by month.───────────────────────────────────────────
print('\n[4/4] calculate trend...', flush=True)

# Build sid → (lat, lon) lookup table
sid_to_meta = {
    row['station_id']: (float(row['latitude']), float(row['longitude']))
    for _, row in si.iterrows()
}

results = []
for sid in sid_list:
    lat, lon_val = sid_to_meta[sid]
    for mo in range(1, 13):
        pts = sta_monthly[sid][mo]   # [(yr, t2m), ...]
        if len(pts) < MIN_YEARS:
            continue
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < MIN_YEARS:
            continue
        sl, _, _, p, _ = stats.linregress(x[valid], y[valid])
        results.append({
            'station_id':  sid,
            'latitude':    lat,
            'longitude':   lon_val,
            'month':       mo,
            'n_years':     int(valid.sum()),
            'trend_dec':   sl * 10,
            'p_value':     p,
            'significant': 'Yes' if p < 0.05 else 'No',
        })

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_PATH, index=False)

elapsed = time.time() - t0
print(f'\nFinish !{len(df_out):,} records, {df_out["station_id"].nunique()} stations', flush=True)
print(f'output: {OUT_PATH}', flush=True)
print(f'Total time spent: {elapsed:.0f} seconds', flush=True)
