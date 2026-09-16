#!/usr/bin/env python3
# ============================================================
# calc_station_scale.py  v2
# Unified station-scale trend computation
# All datasets recomputed from the raw files
# All based on the stations in global_wtemp_monthly_trends_v{MIN_YEARS}y.csv
#
# Usage
#   python calc_station_scale.py --min_years 5
#
# Output (data/ directory):
#   ss_gemstat_trends_{N}y.csv
#   ss_era5_t2m_obs_trends_{N}y.csv
#   ss_era5_t2m_full_trends_{N}y.csv
#   ss_era5_skt_obs_trends_{N}y.csv
#   ss_era5_skt_full_trends_{N}y.csv
#   ss_hadisd_obs_trends_{N}y.csv
#   ss_hadisd_full_trends_{N}y.csv
#   ss_dynwat_obs_trends_{N}y.csv
# ============================================================
import os, glob, time, argparse, warnings
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
from scipy import stats
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--min_years', type=int, default=5)
args      = parser.parse_args()
MIN_YEARS = args.min_years

DATA_DIR    = '/work/home/H.Jason421/water_temp_for_publish/data/'
GEMSTAT_RAW = '/work5/H.Jason421/GEMStat_new/Water_Temperature.csv'
ERA5_T2M_DIR= '/work5/ERA5/t2m/'
ERA5_SKT_NC = '/work5/ERA5/era5_skin_temperature_1980_2022.nc'
HADISD_DIR  = '/work7/L.chshih/ERA_Analysis/data/HadISD/nc/'
DYNWAT_NC   = '/work5/H.Jason421/DynWat/waterTemperature_monthly_1981-2014.nc'

YEAR_START  = 1990
YEAR_END    = 2020
MAX_DIST_KM = 100.0

t0 = time.time()
def elapsed(): return f'{time.time()-t0:.0f}s'
print(f'calc_station_scale.py  MIN_YEARS={MIN_YEARS}', flush=True)

# ── Utility functions ──────────────────────────────────────────────────
EARTH_R = 6371.0

def latlon_to_xyz(lat, lon):
    lr = np.radians(np.asarray(lat, float))
    lo = np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(lr)*np.cos(lo),
                            np.cos(lr)*np.sin(lo),
                            np.sin(lr)])

def chord_to_km(chord):
    return 2 * EARTH_R * np.arcsin(np.clip(np.asarray(chord)/2, 0, 1))

def calc_trend(years, vals, min_n):
    years = np.asarray(years, float)
    vals  = np.asarray(vals,  float)
    mask  = np.isfinite(vals)
    if mask.sum() < min_n: return None
    sl, _, r, p, _ = stats.linregress(years[mask], vals[mask])
    return {'trend_dec': sl*10, 'p_value': p, 'r2': r**2,
            'n_years': int(mask.sum()),
            'significant': 'Yes' if p < 0.05 else 'No'}

def save(results, path):
    if not results:
        print(f'  → 0 stations (no data) {os.path.basename(path)}', flush=True)
        return
    df   = pd.DataFrame(results)
    cols = ['station_id','latitude','longitude','water_type',
            'month','n_years','trend_dec','p_value','significant']
    extra = [c for c in df.columns if c not in cols]
    df[cols+extra].to_csv(path, index=False)
    print(f'  → {df["station_id"].nunique()} stations {os.path.basename(path)}  ({elapsed()})', flush=True)

# ── Reference stations ──────────────────────────────────────────────────
WTEMP_CSV = DATA_DIR + f'global_wtemp_monthly_trends_v{MIN_YEARS}y.csv'
df_base   = pd.read_csv(WTEMP_CSV)
df_base   = df_base[df_base['water_type']=='River station'].copy()
STATIONS  = (df_base[['station_id','latitude','longitude','water_type']]
             .drop_duplicates('station_id').reset_index(drop=True))
STATIONS['station_id'] = STATIONS['station_id'].astype(str)
STATION_IDS = set(STATIONS['station_id'])
sid_arr = STATIONS['station_id'].values
lat_arr = STATIONS['latitude'].values
lon_arr = STATIONS['longitude'].values
print(f'Reference stations: {len(STATIONS)} stations ({elapsed()})', flush=True)

# ══════════════════════════════════════════════════════════════
# 1. GEMStat water temperature (monthly median)
# ══════════════════════════════════════════════════════════════
print(f'\n[1/8] GEMStat TR...', flush=True)
df_raw = pd.read_csv(GEMSTAT_RAW, encoding='latin-1',
                     usecols=['GEMS Station Number','Sample Date','Value'],
                     low_memory=False)
df_raw.columns = ['station_id','date','value']
df_raw = df_raw.dropna(subset=['value','date'])
df_raw['date']       = pd.to_datetime(df_raw['date'], errors='coerce')
df_raw               = df_raw.dropna(subset=['date'])
df_raw['station_id'] = df_raw['station_id'].astype(str)
df_raw['year']       = df_raw['date'].dt.year
df_raw['month']      = df_raw['date'].dt.month
df_raw['day']        = df_raw['date'].dt.day
df_raw = df_raw[df_raw['value'].between(-5, 40)]
df_raw = df_raw[(df_raw['year']>=YEAR_START) & (df_raw['year']<=YEAR_END) &
                (df_raw['station_id'].isin(STATION_IDS))]
print(f'  Records after filtering:{len(df_raw):,}', flush=True)

# Monthly median (avoids reservoir-release extremes)
gem_monthly = df_raw.groupby(['station_id','year','month'])['value'].median().reset_index()
sid_meta    = STATIONS.set_index('station_id').to_dict('index')
results     = []
for sid, grp in gem_monthly.groupby('station_id'):
    meta = sid_meta.get(sid, {})
    for month in range(1, 13):
        md  = grp[grp['month']==month].sort_values('year')
        res = calc_trend(md['year'].values, md['value'].values, MIN_YEARS)
        if res is None: continue
        results.append({'station_id': sid,
                        'latitude':   meta.get('latitude', np.nan),
                        'longitude':  meta.get('longitude', np.nan),
                        'water_type': meta.get('water_type',''),
                        'month': month, **res})
save(results, DATA_DIR + f'ss_gemstat_trends_{MIN_YEARS}y.csv')

# Build observation index
obs_dates = {}
obs_ym    = set()
for row in df_raw[['station_id','year','month','day']].itertuples(index=False):
    key = (row.station_id, row.year, row.month)
    obs_dates.setdefault(key, set()).add(row.day)
    obs_ym.add(key)

# ══════════════════════════════════════════════════════════════
# 2. ERA5 t2m obs. (day-matched, only the days GEMStat sampled)
# ══════════════════════════════════════════════════════════════
print(f'\n[2/8] ERA5 t2m obs. ...', flush=True)
era5_files = sorted(glob.glob(ERA5_T2M_DIR + '*.nc'))
# Read 1990-2020 only
era5_files = [f for f in era5_files
              if any(f'_{yr}_' in f for yr in range(YEAR_START, YEAR_END+1))]
print(f' Number of 1990-2020 files: {len(era5_files)}', flush=True)

buf = {sid: {m: {} for m in range(1,13)} for sid in STATION_IDS}
for fi, fpath in enumerate(era5_files):
    if fi % 30 == 0:
        print(f'  Progress {fi}/{len(era5_files)}  ({elapsed()})', flush=True)
    try:
        ds    = nc4.Dataset(fpath)
        times = nc4.num2date(ds.variables['valid_time'][:],
                             ds.variables['valid_time'].units,
                             only_use_cftime_datetimes=False)
        lats  = ds.variables['latitude'][:]
        lons  = ds.variables['longitude'][:]
        t2m   = ds.variables['t2m'][:] - 273.15  # K→°C
        ds.close()
        for ti, t in enumerate(times):
            yr, mo, dy = t.year, t.month, t.day
            if not (YEAR_START <= yr <= YEAR_END): continue
            slc = t2m[ti]
            for si_i, sid in enumerate(sid_arr):
                if (sid, yr, mo) not in obs_dates: continue
                if dy not in obs_dates[(sid, yr, mo)]: continue
                li  = int(np.argmin(np.abs(lats - lat_arr[si_i])))
                gi  = int(np.argmin(np.abs(lons - lon_arr[si_i])))
                buf[sid][mo].setdefault(yr, []).append(float(slc[li, gi]))
    except Exception as e:
        print(f'  skip {os.path.basename(fpath)}: {e}', flush=True)

results = []
for _, row in STATIONS.iterrows():
    sid = row['station_id']
    for month in range(1, 13):
        yr_vals = {yr: np.mean(v) for yr, v in buf[sid][month].items()}
        if not yr_vals: continue
        yrs  = np.array(sorted(yr_vals))
        vals = np.array([yr_vals[y] for y in yrs])
        res  = calc_trend(yrs, vals, MIN_YEARS)
        if res is None: continue
        results.append({'station_id': sid, 'latitude': row['latitude'],
                        'longitude': row['longitude'], 'water_type': row['water_type'],
                        'month': month, **res})
save(results, DATA_DIR + f'ss_era5_t2m_obs_trends_{MIN_YEARS}y.csv')

# ══════════════════════════════════════════════════════════════
# 3. ERA5 t2m full (all days in each month)
# ══════════════════════════════════════════════════════════════
print(f'\n[3/8] ERA5 t2m full ...', flush=True)
buf_f = {sid: {m: {} for m in range(1,13)} for sid in STATION_IDS}
for fi, fpath in enumerate(era5_files):
    if fi % 30 == 0:
        print(f'  Progress {fi}/{len(era5_files)}  ({elapsed()})', flush=True)
    try:
        ds    = nc4.Dataset(fpath)
        times = nc4.num2date(ds.variables['valid_time'][:],
                             ds.variables['valid_time'].units,
                             only_use_cftime_datetimes=False)
        lats  = ds.variables['latitude'][:]
        lons  = ds.variables['longitude'][:]
        t2m   = ds.variables['t2m'][:] - 273.15
        ds.close()
        for ti, t in enumerate(times):
            yr, mo = t.year, t.month
            if not (YEAR_START <= yr <= YEAR_END): continue
            slc = t2m[ti]
            for si_i, sid in enumerate(sid_arr):
                li  = int(np.argmin(np.abs(lats - lat_arr[si_i])))
                gi  = int(np.argmin(np.abs(lons - lon_arr[si_i])))
                buf_f[sid][mo].setdefault(yr, []).append(float(slc[li, gi]))
    except Exception as e:
        print(f'  skip {os.path.basename(fpath)}: {e}', flush=True)

results = []
for _, row in STATIONS.iterrows():
    sid = row['station_id']
    for month in range(1, 13):
        yr_vals = {yr: np.mean(v) for yr, v in buf_f[sid][month].items()}
        if not yr_vals: continue
        yrs  = np.array(sorted(yr_vals))
        vals = np.array([yr_vals[y] for y in yrs])
        res  = calc_trend(yrs, vals, MIN_YEARS)
        if res is None: continue
        results.append({'station_id': sid, 'latitude': row['latitude'],
                        'longitude': row['longitude'], 'water_type': row['water_type'],
                        'month': month, **res})
save(results, DATA_DIR + f'ss_era5_t2m_full_trends_{MIN_YEARS}y.csv')

# ══════════════════════════════════════════════════════════════
# 4. ERA5 SKT obs. (values only in observed months)
# ══════════════════════════════════════════════════════════════
print(f'\n[4/8] ERA5 SKT obs. ...', flush=True)
ds_skt  = xr.open_dataset(ERA5_SKT_NC)
skt_var = next((v for v in ds_skt.data_vars
                if 'skt' in v.lower() or 'skin' in v.lower()),
               list(ds_skt.data_vars)[0])
print(f'  SKT variable:{skt_var}', flush=True)

results = []
for si_i, row in STATIONS.iterrows():
    if si_i % 50 == 0:
        print(f'  Progress {si_i}/{len(STATIONS)}  ({elapsed()})', flush=True)
    sid = row['station_id']
    monthly = {}
    for yr in range(YEAR_START, YEAR_END+1):
        for mo in range(1, 13):
            if (sid, yr, mo) not in obs_ym: continue
            try:
                val = float(ds_skt[skt_var].sel(
                    time=f'{yr}-{mo:02d}',
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    method='nearest').values)
                if val > 200: val -= 273.15
                monthly.setdefault(mo, {})[yr] = val
            except Exception: continue
    for month, yr_dict in monthly.items():
        yrs  = np.array(sorted(yr_dict))
        vals = np.array([yr_dict[y] for y in yrs])
        res  = calc_trend(yrs, vals, MIN_YEARS)
        if res is None: continue
        results.append({'station_id': sid, 'latitude': row['latitude'],
                        'longitude': row['longitude'], 'water_type': row['water_type'],
                        'month': month, **res})
ds_skt.close()
save(results, DATA_DIR + f'ss_era5_skt_obs_trends_{MIN_YEARS}y.csv')

# ══════════════════════════════════════════════════════════════
# 5. ERA5 SKT full (every month)
# ══════════════════════════════════════════════════════════════
print(f'\n[5/8] ERA5 SKT full ...', flush=True)
ds_skt = xr.open_dataset(ERA5_SKT_NC)
results = []
for si_i, row in STATIONS.iterrows():
    if si_i % 50 == 0:
        print(f'  Progress {si_i}/{len(STATIONS)}  ({elapsed()})', flush=True)
    sid = row['station_id']
    for month in range(1, 13):
        pts = {}
        for yr in range(YEAR_START, YEAR_END+1):
            try:
                val = float(ds_skt[skt_var].sel(
                    time=f'{yr}-{month:02d}',
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    method='nearest').values)
                if val > 200: val -= 273.15
                pts[yr] = val
            except Exception: continue
        if not pts: continue
        yrs  = np.array(sorted(pts))
        vals = np.array([pts[y] for y in yrs])
        res  = calc_trend(yrs, vals, MIN_YEARS)
        if res is None: continue
        results.append({'station_id': sid, 'latitude': row['latitude'],
                        'longitude': row['longitude'], 'water_type': row['water_type'],
                        'month': month, **res})
ds_skt.close()
save(results, DATA_DIR + f'ss_era5_skt_full_trends_{MIN_YEARS}y.csv')

# ══════════════════════════════════════════════════════════════
# 6 & 7. HadISD obs. + full (same pairing, files read once)
# ══════════════════════════════════════════════════════════════
print(f'\n[6-7/8] HadISD pairing + computation...', flush=True)
hadisd_files = sorted(glob.glob(HADISD_DIR + '*.nc'))
hadisd_meta  = []
for fpath in hadisd_files:
    try:
        ds  = xr.open_dataset(fpath)
        lat = float(ds['latitude'].values.flat[0])
        lon = float(ds['longitude'].values.flat[0])
        ds.close()
        hadisd_meta.append({'path': fpath, 'lat': lat, 'lon': lon})
    except Exception: continue

df_h = pd.DataFrame(hadisd_meta)
tree = cKDTree(latlon_to_xyz(df_h['lat'].values, df_h['lon'].values))
dists, idxs = tree.query(latlon_to_xyz(lat_arr, lon_arr), k=1)
dists_km    = chord_to_km(dists)

pairs = []
for i, row in STATIONS.iterrows():
    if dists_km[i] > MAX_DIST_KM: continue
    pairs.append({**row.to_dict(),
                  'hadisd_path': df_h.iloc[idxs[i]]['path'],
                  'dist_km':     dists_km[i]})
pairs_df = pd.DataFrame(pairs)
print(f'  Paired: {len(pairs_df)} stations', flush=True)

def read_hadisd(fpath):
    ds = xr.open_dataset(fpath)
    tv = next((v for v in ['temperatures','T','temp','air_temperature','t2m']
               if v in ds), None)
    td = next((d for d in ['time','dates','date']
               if d in ds.dims or d in ds.coords), None)
    if not tv or not td: ds.close(); return None
    df_t = ds[tv].to_dataframe().reset_index().rename(columns={tv:'temp', td:'time'})
    df_t['time'] = pd.to_datetime(df_t['time'], errors='coerce')
    df_t = df_t.dropna(subset=['time','temp'])
    df_t['year']  = df_t['time'].dt.year
    df_t['month'] = df_t['time'].dt.month
    df_t = df_t[(df_t['year']>=YEAR_START) & (df_t['year']<=YEAR_END)]
    if len(df_t) > 0 and df_t['temp'].median() > 200: df_t['temp'] -= 273.15
    df_t = df_t[df_t['temp'].abs() < 100]
    ds.close()
    return df_t.groupby(['year','month'])['temp'].mean().reset_index()

results_obs  = []
results_full = []
for i, (_, row) in enumerate(pairs_df.iterrows()):
    if i % 100 == 0:
        print(f'  HadISD progress {i}/{len(pairs_df)}  ({elapsed()})', flush=True)
    sid = str(row['station_id'])
    try:
        df_monthly = read_hadisd(row['hadisd_path'])
        if df_monthly is None: continue
        for month in range(1, 13):
            md = df_monthly[df_monthly['month']==month]
            # full: all years
            res = calc_trend(md['year'].values, md['temp'].values, MIN_YEARS)
            if res:
                results_full.append({'station_id': sid,
                                     'latitude':   row['latitude'],
                                     'longitude':  row['longitude'],
                                     'water_type': row['water_type'],
                                     'dist_km':    row['dist_km'],
                                     'month': month, **res})
            # obs: keep only years GEMStat sampled
            valid = md[md.apply(
                lambda r: (sid, int(r['year']), month) in obs_ym, axis=1)]
            res = calc_trend(valid['year'].values, valid['temp'].values, MIN_YEARS)
            if res:
                results_obs.append({'station_id': sid,
                                    'latitude':   row['latitude'],
                                    'longitude':  row['longitude'],
                                    'water_type': row['water_type'],
                                    'dist_km':    row['dist_km'],
                                    'month': month, **res})
    except Exception: continue

save(results_obs,  DATA_DIR + f'ss_hadisd_obs_trends_{MIN_YEARS}y.csv')
save(results_full, DATA_DIR + f'ss_hadisd_full_trends_{MIN_YEARS}y.csv')

# ══════════════════════════════════════════════════════════════
# 8. DynWat obs. (from raw nc, only observed months, years through 2014)
# ══════════════════════════════════════════════════════════════
print(f'\n[8/8] DynWat obs. ...', flush=True)
DW_YEAR_END = min(YEAR_END, 2014)
print(f'  DynWat year range:{YEAR_START}–{DW_YEAR_END}', flush=True)

ds_dw   = xr.open_dataset(DYNWAT_NC)
dw_lats = ds_dw['lat'].values
dw_lons = ds_dw['lon'].values

results = []
for si_i, row in STATIONS.iterrows():
    if si_i % 50 == 0:
        print(f'  Progress {si_i}/{len(STATIONS)}  ({elapsed()})', flush=True)
    sid = str(row['station_id'])
    li  = int(np.argmin(np.abs(dw_lats - row['latitude'])))
    gi  = int(np.argmin(np.abs(dw_lons - row['longitude'])))
    monthly = {}
    for yr in range(YEAR_START, DW_YEAR_END+1):
        for mo in range(1, 13):
            if (sid, yr, mo) not in obs_ym: continue
            try:
                val = float(ds_dw['waterTemp'].sel(
                    time=f'{yr}-{mo:02d}',
                    method='nearest').values[li, gi])
                if np.isnan(val): continue
                monthly.setdefault(mo, {})[yr] = val
            except Exception: continue
    for month, yr_dict in monthly.items():
        yrs  = np.array(sorted(yr_dict))
        vals = np.array([yr_dict[y] for y in yrs])
        res  = calc_trend(yrs, vals, MIN_YEARS)
        if res is None: continue
        results.append({'station_id': sid,
                        'latitude':   row['latitude'],
                        'longitude':  row['longitude'],
                        'water_type': row['water_type'],
                        'month': month, **res})
ds_dw.close()
save(results, DATA_DIR + f'ss_dynwat_obs_trends_{MIN_YEARS}y.csv')

print(f'\nAll done! Total time {elapsed()}', flush=True)
