#!/usr/bin/env python3
# ============================================================
# Supp_FigS1_station_coverage.py  (v2, accelerated)
#   Supplementary Fig. S1: Station coverage of GEMStat and HadISD
#   air-temperature references (four-panel global map)
#
#   v2 speedups over v1:
#     1. Temperature.csv: read only the 4 needed columns, in chunks, filter TEMP-Air before parsing dates
#     2. HadISD 9,667 nc files: read coordinates directly with netCDF4 and process in parallel
#     3. Both intermediate results are cached; subsequent runs skip the file reads
#
#   Usage:
#     python Supp_FigS1_station_coverage.py            # use the cache if present
#     python Supp_FigS1_station_coverage.py --refresh  # force-rebuild the cache
#
#   Output -> figures/Supplementary_correct/Supplementary_FigS1_station_coverage.png
# ============================================================
import os
import glob
import time
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore')

# ── Path ─────────────────────────────────────────────────────────────────────
BASE_DIR  = '/work/home/H.Jason421/water_temp_for_publish/'
DATA_DIR  = BASE_DIR + 'data/'
FIG_DIR   = BASE_DIR + 'figures/Supplementary_correct/'
CACHE_DIR = BASE_DIR + 'data/cache/'

TEMP_CSV      = '/work5/H.Jason421/GEMstat_WaterQuality/Temperature.csv'
GEMS_META_XLS = '/work5/H.Jason421/GEMStat_new/GEMS-Water_data_request.xls'
HADISD_NC_DIR = '/work7/L.chshih/ERA_Analysis/data/HadISD/nc/'

OUT           = FIG_DIR + 'Supplementary_FigS1_station_coverage.png'
CACHE_GEMSTAT = CACHE_DIR + 'gemstat_tempair_stations.csv'
CACHE_HADISD  = CACHE_DIR + 'hadisd_all_station_coords.csv'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Analysis parameters  ─────────────────────────────────────────────────────────────────
YEAR_START = 1990
YEAR_END   = 2020
MIN_YEARS  = 5
PAIR_KM    = 100

CHUNKSIZE  = 2_000_000   # Temperature.csv chunk size
N_WORKERS  = 8           # parallelism for reading HadISD nc files (keep low on the login node)

# ── Figure settings ─────────────────────────────────────────────────────────────────
DPI          = 300
FIGSIZE      = (22, 12)
EXTENT       = [-180, 180, -57, 80]

FS_TITLE     = 16
FS_LEGEND    = 13
FS_PANEL_LAB = 18

C_GEMSTAT    = '#FF8C00'
C_HADISD     = '#1E90FF'
C_BASE       = '#9e9e9e'
C_LAND       = '#ffffff'
C_OCEAN      = '#ffffff'
C_COAST      = '#8a8a8a'
LW_COAST     = 0.6

plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size']         = 12


# ============================================================
# Tools
# ============================================================
def tick(msg, t0):
    print(f'  [{time.time() - t0:6.1f}s] {msg}', flush=True)


def _hadisd_coord(fp):
    """Read (lat, lon) from a single HadISD nc file, using netCDF4 directly and skipping CF decoding."""
    try:
        import netCDF4
        with netCDF4.Dataset(fp) as ds:
            for la, lo in (('latitude', 'longitude'), ('lat', 'lon')):
                if la in ds.variables and lo in ds.variables:
                    return (float(np.ravel(ds.variables[la][:])[0]),
                            float(np.ravel(ds.variables[lo][:])[0]))
            for la, lo in (('latitude', 'longitude'), ('lat', 'lon')):
                if hasattr(ds, la) and hasattr(ds, lo):
                    return float(getattr(ds, la)), float(getattr(ds, lo))
    except Exception:
        pass
    return None


# ============================================================
# 1. GEMStat TEMP-Air stations (cached)
# ============================================================
def build_gemstat_cache():
    t0 = time.time()
    print('Build GEMStat TEMP-Air cache ...', flush=True)

    header = pd.read_csv(TEMP_CSV, sep=';', encoding='latin-1', nrows=0)
    want = ['GEMS.Station.Number', 'Sample.Date', 'Parameter.Code', 'Value']
    usecols = [c for c in want if c in header.columns]
    if len(usecols) < 4:
        raise KeyError(f'Temperature.csv columns do not match; found:{list(header.columns)}')

    parts, n_read = [], 0
    for chunk in pd.read_csv(TEMP_CSV, sep=';', encoding='latin-1',
                             usecols=usecols, dtype=str, chunksize=CHUNKSIZE):
        n_read += len(chunk)
        sub = chunk[chunk['Parameter.Code'] == 'TEMP-Air']
        if len(sub):
            parts.append(sub)
        tick(f'Scanned {n_read:,} rows, accumulated TEMP-Air {sum(len(p) for p in parts):,} records', t0)

    air = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
    air['Value'] = pd.to_numeric(air['Value'], errors='coerce')
    air['year']  = pd.to_datetime(air['Sample.Date'], errors='coerce').dt.year
    tick('Date parsing complete', t0)

    air = air[(air['year'].between(YEAR_START, YEAR_END)) &
              (air['Value'].between(-50, 60))]
    air['sid'] = air['GEMS.Station.Number'].astype(str)
    sids = air[['sid']].drop_duplicates()
    tick(f'After filtering: {len(air):,} records / {len(sids):,} stations', t0)

    meta_cols = {'GEMS Station Number': 'sid', 'Latitude': 'lat', 'Longitude': 'lon'}
    gems_meta = (pd.read_excel(GEMS_META_XLS, sheet_name='Station_Metadata')
                 [list(meta_cols)].rename(columns=meta_cols))
    gems_meta['sid'] = gems_meta['sid'].astype(str)

    out = sids.merge(gems_meta, on='sid', how='left')
    out.to_csv(CACHE_GEMSTAT, index=False)
    tick(f'Wrote cache {CACHE_GEMSTAT}', t0)
    return out


def build_hadisd_cache():
    t0 = time.time()
    files = sorted(glob.glob(HADISD_NC_DIR + '*.nc'))
    print(f'Build HadISD cache（{len(files):,} files, {N_WORKERS} workers)...', flush=True)

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for res in ex.map(_hadisd_coord, files, chunksize=64):
            done += 1
            if res is not None:
                rows.append({'lat': res[0], 'lon': res[1]})
            if done % 1000 == 0:
                tick(f'{done:,}/{len(files):,}', t0)

    out = pd.DataFrame(rows)
    out.to_csv(CACHE_HADISD, index=False)
    tick(f'Succeeded  {len(out):,} stations, wrote {CACHE_HADISD}', t0)
    return out


def load_cached(path, builder, refresh):
    if not refresh and os.path.exists(path):
        df = pd.read_csv(path)
        print(f'Using cache: {os.path.basename(path)}({len(df):,} rows)', flush=True)
        return df
    return builder()


# ============================================================
# 2. Main Process
# ============================================================
def main(refresh=False):
    t_all = time.time()

    gem_air    = load_cached(CACHE_GEMSTAT, build_gemstat_cache, refresh)
    hadisd_all = load_cached(CACHE_HADISD,  build_hadisd_cache,  refresh)

    gem_air['sid'] = gem_air['sid'].astype(str)
    airtemp_all = gem_air.dropna(subset=['lat', 'lon'])

    base = pd.read_csv(DATA_DIR + f'global_wtemp_monthly_trends_v{MIN_YEARS}y.csv')
    base = base[base['water_type'] == 'River station']
    base_meta = base[['station_id', 'latitude', 'longitude']].drop_duplicates('station_id')
    base_meta['station_id'] = base_meta['station_id'].astype(str)
    base_ids = set(base_meta['station_id'])

    onsite_meta = base_meta[base_meta['station_id'].isin(set(gem_air['sid']) & base_ids)]

    pairs = pd.read_csv(DATA_DIR + 'hadisd_gemstat_pairs.csv')
    pairs['station_id'] = pairs['station_id'].astype(str)
    paired_meta = base_meta[base_meta['station_id'].isin(set(pairs['station_id']) & base_ids)]

    # ── Plot ────────────────────────────────────────────────────────────────
    PROJ = ccrs.PlateCarree()
    fig, axs = plt.subplots(2, 2, figsize=FIGSIZE,
                            subplot_kw={'projection': ccrs.Mercator()})
    fig.patch.set_facecolor('white')

    def setup_ax(ax, title, panel_label):
        ax.set_extent(EXTENT, crs=PROJ)
        ax.add_feature(cfeature.LAND,      facecolor=C_LAND,  zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor=C_OCEAN, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=LW_COAST, color=C_COAST, zorder=1)
        ax.set_title(title, fontsize=FS_TITLE, pad=8)
        ax.text(0.005, 1.02, panel_label, transform=ax.transAxes,
                fontsize=FS_PANEL_LAB, fontweight='bold', va='bottom', ha='left')
        try:
            ax.spines['geo'].set_visible(False)
        except Exception:
            pass

    setup_ax(axs[0, 0],
             f'All GEMStat on-site air-temperature stations (n = {len(airtemp_all):,})',
             '(a)')
    axs[0, 0].scatter(airtemp_all['lon'], airtemp_all['lat'], s=2, c=C_GEMSTAT,
                      alpha=0.55, linewidths=0, transform=PROJ, zorder=2)

    setup_ax(axs[0, 1],
             'Trend-analysis river stations with on-site air temperature', '(b)')
    axs[0, 1].scatter(base_meta['longitude'], base_meta['latitude'], s=14, c=C_BASE,
                      alpha=0.7, linewidths=0, transform=PROJ, zorder=2,
                      label=f'Trend-analysis stations (n = {len(base_meta)})')
    axs[0, 1].scatter(onsite_meta['longitude'], onsite_meta['latitude'], s=34,
                      c=C_GEMSTAT, alpha=0.95, linewidths=0, transform=PROJ, zorder=3,
                      label=f'With on-site air temperature (n = {len(onsite_meta)})')
    axs[0, 1].legend(loc='lower left', fontsize=FS_LEGEND, framealpha=0.9,
                     facecolor='white', edgecolor='#cccccc')

    setup_ax(axs[1, 0], f'All HadISD stations (n = {len(hadisd_all):,})', '(c)')
    axs[1, 0].scatter(hadisd_all['lon'], hadisd_all['lat'], s=2, c=C_HADISD,
                      alpha=0.45, linewidths=0, transform=PROJ, zorder=2)

    setup_ax(axs[1, 1],
             f'Trend-analysis river stations matched to HadISD within {PAIR_KM} km',
             '(d)')
    axs[1, 1].scatter(base_meta['longitude'], base_meta['latitude'], s=14, c=C_BASE,
                      alpha=0.7, linewidths=0, transform=PROJ, zorder=2,
                      label=f'Trend-analysis stations (n = {len(base_meta)})')
    axs[1, 1].scatter(paired_meta['longitude'], paired_meta['latitude'], s=34,
                      c=C_HADISD, alpha=0.95, linewidths=0, transform=PROJ, zorder=3,
                      label=f'Matched to HadISD (n = {len(paired_meta)})')
    axs[1, 1].legend(loc='lower left', fontsize=FS_LEGEND, framealpha=0.9,
                     facecolor='white', edgecolor='#cccccc')

    plt.tight_layout()
    fig.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'\nSaved: {OUT}')
    print(f'Total time {time.time() - t_all:.1f}s')

    print('\n- numbers to check against the caption -')
    print(f'  Total TEMP-Air stations (incl. no coords):{len(gem_air):,}')
    print(f'  (a) Actually plotted (with coords)       : {len(airtemp_all):,}')
    print(f'  (b) Trend-analysis stations              : {len(base_meta)}')
    print(f'      of which have on-site air temp       : {len(onsite_meta)}')
    print(f'  (c) HadISD total                         : {len(hadisd_all):,}')
    print(f'  (d) Paired to HadISD                     : {len(paired_meta)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='orce-rebuild the cache')
    main(**vars(ap.parse_args()))
