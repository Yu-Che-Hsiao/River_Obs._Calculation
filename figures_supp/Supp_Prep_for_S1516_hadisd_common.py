#!/usr/bin/env python3
# ============================================================
# Supp_hadisd_common.py
#   HadISD data processing shared by Supplementary Fig. S5 and S6
#
#   Extracted from Cells 0–4, 6 of old_bash/Fig_HadISD_validation_v2.ipynb.
#   The numerical logic is identical to the notebook, only parallel processing and caching are added.
#
#   Does not plot anything itself; imported by:
#     Supp_FigS15_hadisd_era5_validation.py   (Fig S15)
#     Supp_FigS16_wtemp_vs_hadisd.py          (Fig S16)
# ============================================================
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = '/work/home/H.Jason421/water_temp_for_publish/'
DATA_DIR = BASE_DIR + 'data/'
FIG_DIR  = BASE_DIR + 'figures/Supplementary_correct/'

PAIRS_CSV     = DATA_DIR + 'hadisd_gemstat_pairs.csv'
CACHE_ANNUAL  = DATA_DIR + 'hadisd_airtemp_trends_v1.csv'
CACHE_SEASON  = DATA_DIR + 'hadisd_airtemp_seasonal_trends_v1.csv'

os.makedirs(FIG_DIR, exist_ok=True)

# ── Analysis parameters (same as the notebook, do not change) ─────────────────────────────────────
YEAR_START          = 1990
YEAR_END            = 2020
MAX_DIST_KM         = 100.0
MIN_YEARS_FOR_TREND = 10
MIN_MONTHS_PER_YEAR = 3      # annual trend: at least 3 months per year
MIN_MONTHS_SEASON   = 2      # seasonal trend: at least 2 months per season

SEASONS = {
    'Boreal Summer': {'NH': [6, 7, 8],  'SH': [12, 1, 2]},
    'Boreal Winter': {'NH': [12, 1, 2], 'SH': [6, 7, 8]},
}

N_WORKERS = 8

# ── Shared figure settings ─────────────────────────────────────────────────────────────
DPI = 300


def apply_white_style(plt):
    """All Supplementary figures use a uniform white background."""
    plt.rcParams['figure.facecolor']  = 'white'
    plt.rcParams['axes.facecolor']    = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'


# ============================================================
# HadISD single-station time series
# ============================================================
def _read_station_series(path):
    """Read a single HadISD nc, return DataFrame(year, month, temp); return None on failure."""
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            tvar = next((v for v in ['temperatures', 'T', 'temp',
                                     'air_temperature', 't2m'] if v in ds), None)
            tdim = next((d for d in ['time', 'dates', 'date']
                         if d in ds.dims or d in ds.coords), None)
            if tvar is None or tdim is None:
                return None
            df = ds[tvar].to_dataframe().reset_index()

        df = df.rename(columns={tvar: 'temp', tdim: 'time'})
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time', 'temp'])
        df['year']  = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df = df[df['year'].between(YEAR_START, YEAR_END)]
        if df.empty:
            return None
        if df['temp'].median() > 200:          # K -> degC
            df['temp'] = df['temp'] - 273.15
        return df[df['temp'].abs() < 100]      # filter out missing-value flags
    except Exception:
        return None


def _fit(years, values):
    m = np.isfinite(values)
    if m.sum() < MIN_YEARS_FOR_TREND:
        return None
    sl, ic, r, p, _ = stats.linregress(years[m], values[m])
    return sl, r ** 2, p, int(m.sum())


def _annual_job(row):
    df = _read_station_series(row['hadisd_path'])
    if df is None:
        return None
    monthly = df.groupby(['year', 'month'])['temp'].mean().reset_index()
    annual = (monthly.groupby('year')
              .agg(temp_mean=('temp', 'mean'), n_months=('month', 'nunique'))
              .reset_index())
    annual = annual[annual['n_months'] >= MIN_MONTHS_PER_YEAR]
    if len(annual) < MIN_YEARS_FOR_TREND:
        return None
    res = _fit(annual['year'].values, annual['temp_mean'].values)
    if res is None:
        return None
    sl, r2, p, n = res
    return {'station_id': row['station_id'], 'hadisd_id': row['hadisd_id'],
            'dist_km': row['dist_km'], 'hadisd_trend_yr': sl,
            'hadisd_r2': r2, 'hadisd_p': p, 'hadisd_n_years': n}


def _seasonal_job(row):
    df = _read_station_series(row['hadisd_path'])
    if df is None:
        return None
    out = []
    for season, mdef in SEASONS.items():
        months = mdef['NH'] if row['latitude'] > 0 else mdef['SH']
        sub = df[df['month'].isin(months)]
        annual = (sub.groupby('year')
                  .agg(temp_mean=('temp', 'mean'), n_months=('month', 'nunique'))
                  .reset_index())
        annual = annual[annual['n_months'] >= MIN_MONTHS_SEASON]
        if len(annual) < MIN_YEARS_FOR_TREND:
            continue
        res = _fit(annual['year'].values, annual['temp_mean'].values)
        if res is None:
            continue
        sl, r2, p, _ = res
        out.append({'station_id': row['station_id'], 'hadisd_id': row['hadisd_id'],
                    'season': season, 'hadisd_trend_yr': sl,
                    'hadisd_r2': r2, 'hadisd_p': p})
    return out or None


# ============================================================
# Public interface
# ============================================================
def load_pairs():
    if not os.path.exists(PAIRS_CSV):
        raise FileNotFoundError(
            f'not found {PAIRS_CSV}\n'
            'This pairing table is produced by Cell 3 of old_bash/Fig_HadISD_validation_v2.ipynb;'
            'run that cell first.')
    pairs = pd.read_csv(PAIRS_CSV)
    return pairs[pairs['dist_km'] <= MAX_DIST_KM].copy()


def _build(cache, job, flatten, refresh):
    if not refresh and os.path.exists(cache):
        df = pd.read_csv(cache)
        print(f'Using cache:{os.path.basename(cache)}（{len(df):,} records)', flush=True)
        return df

    pairs = load_pairs()
    rows = [r for _, r in pairs.iterrows()]
    print(f'Computing HadISD trends（{len(rows):,} stations, {N_WORKERS} workers)...', flush=True)

    t0, out, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for res in ex.map(job, rows, chunksize=8):
            done += 1
            if res is not None:
                out.extend(res) if flatten else out.append(res)
            if done % 100 == 0:
                print(f'  [{time.time()-t0:6.1f}s] {done}/{len(rows)}', flush=True)

    df = pd.DataFrame(out)
    df.to_csv(cache, index=False)
    print(f'Finish! {len(df):,} records, written to  {cache}', flush=True)
    return df


def hadisd_annual_trends(refresh=False):
    return _build(CACHE_ANNUAL, _annual_job, False, refresh)


def hadisd_seasonal_trends(refresh=False):
    return _build(CACHE_SEASON, _seasonal_job, True, refresh)


def era5_annual_trends():
    """ERA5 per-station annual trend = mean of the individual calendar-month trends (same as the notebook)."""
    df = pd.read_csv(DATA_DIR + 'global_airtemp_monthly_trends_v1.csv')
    out = df.groupby('station_id')['trend_per_year'].mean().reset_index()
    out.columns = ['station_id', 'era5_trend_yr']
    return out
