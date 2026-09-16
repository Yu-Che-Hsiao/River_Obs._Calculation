#!/usr/bin/env python3
# ============================================================
# calc_dynwat_global_trends.py
# Calculate DynWat global grid point trend（1990–2014）
# Use：python calc_dynwat_global_trends.py
# ------------------------------------------------------------
# FIX (repo cleanup): lat_vals / lon_vals were originally read after ds_dw.close();
# accessing coordinates after the dataset is closed can fail or return empty values.
# The coordinate reads have been moved to before close().
# ============================================================
import numpy as np
import pandas as pd
import xarray as xr
import os, time, warnings
warnings.filterwarnings('ignore')

DATA_DIR    = '/work/home/H.Jason421/water_temp_for_publish/data/'
DYNWAT_PATH = '/work5/H.Jason421/DynWat/waterTemperature_monthly_1981-2014.nc'
OUT_PATH    = DATA_DIR + 'dynwat_global_trends_v1.csv'
MIN_YEARS   = 3

t0 = time.time()
print('=' * 60)
print('DynWat global trend calculating start...')
print('=' * 60)

print('\nLoad DynWat...')
ds_dw = xr.open_dataset(DYNWAT_PATH).sel(
    time=slice('1990-01-01', '2014-12-31')
)
times  = pd.to_datetime(ds_dw['time'].values)
years  = np.array(times.year)   # ← Transfer into numpy array to avaid Index.mean() error
months = np.array(times.month)

print(f'Times: {len(times)} months, Grids: {dict(ds_dw.dims)}')

# --- FIX: Read all the needed Array and Coordinates before "close()" ---
wt       = ds_dw['waterTemp'].values  # (300, 2160, 4320)
lat_vals = ds_dw['lat'].values
lon_vals = ds_dw['lon'].values
ds_dw.close()
print(f'waterTemp loading completed, shape={wt.shape}')

results = []
for mo in range(1, 13):
    t1 = time.time()
    mo_mask = months == mo
    wt_mo   = wt[mo_mask]           # (25, 2160, 4320)
    yrs_mo  = years[mo_mask]        # numpy array withe 25 years

    # Effective Grids: At least MIN_YEARS non-NaN values greater than 0.
    valid_mask = np.sum((wt_mo > 0) & np.isfinite(wt_mo), axis=0) >= MIN_YEARS
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        print(f'  Month {mo:2d}: 0 Effective Grids, Jupmed')
        continue

    # Vectorization OLS
    wt_flat = wt_mo[:, valid_mask]   # (25, n_valid)
    x = yrs_mo.astype(float)
    x_mean = float(x.mean())
    x_c    = x - x_mean
    ss_xx  = float((x_c**2).sum())

    # Dealing with NaN：use nanmean
    y_mean = np.nanmean(wt_flat, axis=0)   # (n_valid,)
    ss_xy  = np.nansum(
        x_c[:, None] * (wt_flat - y_mean[None, :]),
        axis=0
    )                                        # (n_valid,)
    slopes = ss_xy / ss_xx                   # °C/year

    lat_idxs, lon_idxs = np.where(valid_mask)

    for j in range(n_valid):
        results.append({
            'latitude':  float(lat_vals[lat_idxs[j]]),
            'longitude': float(lon_vals[lon_idxs[j]]),
            'month':     mo,
            'trend_dec': float(slopes[j]) * 10,
        })

    elapsed = time.time() - t1
    print(f'  Month {mo:2d}: {n_valid:,} Effective Grids ({elapsed:.1f}s)')

df_out = pd.DataFrame(results)
df_out.to_csv(OUT_PATH, index=False)

total = time.time() - t0
print(f'\nFinish！{len(df_out):,} records')
print(f'Outout：{OUT_PATH}')
print(f'Total duration: {total:.0f} seconds')
