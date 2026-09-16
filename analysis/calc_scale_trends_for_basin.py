#!/usr/bin/env python3
# ============================================================
# calc_scale_trends_v1.py
#
# Design:
#   scale:
#     - basin       :Main Basin of metadata
#     - continental :5 continental (use lat/lon of assign_region)
#     - global      :All station one group
#   version:
#     - GEMStat(Water): Single (It is obs original)
#     - ERA5 TA / ERA5 TS / HadISD / DynWat: each ALL + SUB
#   SUB aligns GEM = (A), Each source is aligned to the finest allowable data.
#     - ERA5 t2m (hourly)  :Corresponding to each day - Only the daily average of the actual measurement days for that station in that month is taken from the GEMStat data.
#     - HadISD (daily)     :Corresponding to each day
#     - ERA5 SKT (Monthly) :Corresponding to each month - Only retain observations from GEMStat (station, year, month).
#     - DynWat (Monthly)   :Corresponding to each month
#   Aggregation used median:
#     - Monthly Representative Value within the Site:
#               GEMStat = Median of measurements for the current month;
#               Daily corresponding source = median of daily average of the measurement days;
#               Monthly source data = value for this month
#     - Cross Station:For each (group, year, month), take the median of the stations.
#   trend: Perform OLS (≥MIN_YEARS years) on each (group, calendar month) year, outputting °C/decade.
#
# Output (New folder and do not cover old file):
#   scale_trends_v1/{dataset}_{scale}_{version}.csv
#     dataset ∈ {gemstat, era5_t2m, era5_skt, hadisd, dynwat}
#     scale   ∈ {basin, continental, global}
#     version ∈ {all, sub}(gemstat without version)
#
# Usage:
#   python calc_scale_trends_v1.py                  # All
#   python calc_scale_trends_v1.py --dataset era5_skt dynwat gemstat   # Run for some part
#   python calc_scale_trends_v1.py --min_years 8
#   python calc_scale_trends_v1.py --station_set qc219   # Use v5y 219 stations
#
# Note: ERA5 t2m time-based processing is very resource-intensive and will take a long time to run; use --dataset to run the fast one (gemstat/dynwat/era5_skt).
# ============================================================
import os, glob, pickle, argparse, warnings
import numpy as np
import pandas as pd
import xarray as xr
from collections import defaultdict
from scipy import stats
warnings.filterwarnings('ignore')

try:
    import netCDF4 as nc4
except Exception:
    nc4 = None

# ── Path ──────────────────────────────────────────────────────
DATA_DIR    = '/work/home/H.Jason421/water_temp_for_publish/data/'
GEMSTAT_RAW = '/work5/H.Jason421/GEMStat_new/Water_Temperature.csv'
META_XLS    = '/work5/H.Jason421/GEMStat_new/GEMS-Water_data_request.xls'
T2M_DIR     = '/work5/ERA5/t2m/'                                   # hourly T_YYYY_MM.nc
ERA5_SKT    = '/work5/ERA5/era5_skin_temperature_1980_2022.nc'     # Monthly file 
DYNWAT_PKL  = DATA_DIR + 'dynwat_station_monthly_v1.pkl'
HADISD_PAIRS= DATA_DIR + 'hadisd_gemstat_pairs.csv'
QC_LIST_CSV = DATA_DIR + 'global_wtemp_monthly_trends_v5y.csv'
OUT_DIR     = DATA_DIR + 'scale_trends_v1/'
FNAME_SUFFIX = ''   # Use main() by --min_years setting; 5y is empty, and else are _Ny

PERIOD    = (1990, 2020)
DW_PERIOD = (1990, 2014)
SCALES    = ['basin', 'continental', 'global']
# ╔══════════════════════════════════════════════════════════╗
# ║  Shared: Area Weighted Function                          ║
# ╚══════════════════════════════════════════════════════════╝
def weighted_median(values, weights):
    """cos latitude-weighted median: find the values whose cumulative weight reaches 50% of the total weight after sorting."""
    import numpy as np
    values  = np.asarray(values,  float)
    weights = np.asarray(weights, float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[m], weights[m]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    return float(values[np.searchsorted(cum, weights.sum() / 2.0)])
 
def weighted_mean(values, weights):
    """cos latitudinal weighted average."""
    import numpy as np
    values  = np.asarray(values,  float)
    weights = np.asarray(weights, float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[m], weights[m]
    if len(values) == 0:
        return np.nan
    return float(np.average(values, weights=weights))

# ── 5 continents (Same as the assign_region of Fig2 combine)──────────────
def assign_region(lat, lon):
    if lon > 180:
        lon -= 360
    if   lat > 20  and -170 < lon < -50:  return 'North America'
    elif lat <= 20 and -120 < lon < -30:  return 'Latin America'
    elif -10 < lat < 80 and -15 < lon < 60: return 'Europe'
    elif lat > 35  and  60 < lon < 180:   return 'Asia'
    elif lat <= 35 and  60 < lon < 180:   return 'South/SE Asia'
    return None   # Other → Not included continental


def detect_var(ds, keywords):
    for v in ds.data_vars:
        if any(k in v.lower() for k in keywords):
            return v
    cands = [v for v in ds.data_vars if ds[v].ndim >= 2]
    if len(cands) == 1:
        return cands[0]
    raise KeyError(f'Cannot find {keywords}, now have {list(ds.data_vars)}')


def detect_time_dim(ds):
    for c in ('time', 'valid_time', 'date', 'dates'):
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError(f'cannot find the time dimension, and coords={list(ds.coords)}')


# ============================================================
# 0. Station list + group (basin / continental)
# ============================================================
def load_stations(station_set):
    print('Read station metadata...', flush=True)
    meta = pd.read_excel(pd.ExcelFile(META_XLS), sheet_name='Station_Metadata',
                         usecols=['GEMS Station Number', 'Country Name', 'Water Type',
                                  'Main Basin', 'Latitude', 'Longitude'])
    meta.columns = ['station_id', 'country', 'water_type', 'basin', 'lat', 'lon']
    meta['station_id'] = meta['station_id'].astype(str)
    meta = (meta[meta['water_type'] == 'River station']
            .dropna(subset=['lat', 'lon']).drop_duplicates('station_id'))

    if station_set == 'qc219':
        qc = pd.read_csv(QC_LIST_CSV)
        qc_ids = set(qc['station_id'].astype(str))
        meta = meta[meta['station_id'].isin(qc_ids)]
        print(f'  Apply v5y 219 station list{len(meta)} station', flush=True)
    else:
        print(f'  All of the River station: {len(meta)} station', flush=True)

    si = meta[['station_id', 'lat', 'lon', 'basin', 'country']].reset_index(drop=True)
    si['continental'] = [assign_region(la, lo) for la, lo in zip(si['lat'], si['lon'])]
    return si


# ============================================================
# 1. GEMStat observation content (for SUB to correspond) + GEMStat monthly present values
# ============================================================
def build_gemstat(si):
    print('[GEMStat] Read original data and build observation content...', flush=True)
    df = pd.read_csv(GEMSTAT_RAW, encoding='latin-1', low_memory=False,
                     usecols=['GEMS Station Number', 'Sample Date', 'Value'])
    df.columns = ['station_id', 'date', 'value']
    df['station_id'] = df['station_id'].astype(str)
    df = df[df['station_id'].isin(set(si['station_id']))]
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['value', 'date'])
    df = df[df['value'].between(-5, 40)]
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df[(df['year'] >= PERIOD[0]) & (df['year'] <= PERIOD[1])]

    # Set of "Measuring Days" for each station and each month (for SUB daily corresponding)
    station_dates = defaultdict(set)
    for r in df[['station_id', 'year', 'month', 'day']].itertuples(index=False):
        station_dates[(r.station_id, r.year, r.month)].add(r.day)
    station_months = set(station_dates.keys())      # for SUB monthly corresponding
    print(f'  The set numbers of (Station, Year, Month):{len(station_months):,}', flush=True)

    # GEMStat monthly present values = the median in that measuring month
    gem_long = (df.groupby(['station_id', 'year', 'month'])['value']
                  .median().reset_index().rename(columns={'value': 'val'}))
    return gem_long, station_dates, station_months


# ============================================================
# 2. Extraction from each source(Return long: station_id, year, month, val)
# ============================================================
def extract_era5_t2m(si, station_dates):
    print('[ERA5 t2m] Extract hourly files (ALL monthly average + SUB daily correspondence)...', flush=True)
    if nc4 is None:
        raise RuntimeError('Need netCDF4 kit')
    files = sorted(glob.glob(T2M_DIR + 'T_*.nc'))
    files = [f for f in files
             if PERIOD[0] <= int(os.path.basename(f).split('_')[1]) <= PERIOD[1]]
    print(f'  Files number:{len(files)}', flush=True)

    ds0 = nc4.Dataset(files[0])
    glats = np.array(ds0.variables['latitude'][:])
    glons = np.array(ds0.variables['longitude'][:])
    ds0.close()
    lat_i = np.array([int(np.argmin(np.abs(glats - v))) for v in si['lat']])
    lon_i = np.array([int(np.argmin(np.abs(glons - (v % 360)))) for v in si['lon']])
    sids = si['station_id'].values

    all_rows, sub_rows = [], []
    for k, f in enumerate(files):
        p = os.path.basename(f).replace('.nc', '').split('_')
        yr, mo = int(p[1]), int(p[2])
        ds = nc4.Dataset(f)
        arr = np.array(ds.variables['t2m'][:])            # (hours, nlat, nlon)
        vt  = np.array(ds.variables['valid_time'][:])
        ds.close()
        cell = arr[:, lat_i, lon_i] - 273.15              # (hours, nsta)
        # Do daily average first (hour→day)
        days = pd.to_datetime(vt, unit='s', utc=True).day.values
        uniq = np.unique(days)
        daily = np.full((len(uniq), len(sids)), np.nan)
        for di, d in enumerate(uniq):
            daily[di] = np.nanmean(cell[days == d], axis=0)
        # ALL: The median of all daily averages for the month.(Using median, same as SUB and GEMStat)
        monthly = np.nanmedian(daily, axis=0)
        for s, sid in enumerate(sids):
            if np.isfinite(monthly[s]):
                all_rows.append((sid, yr, mo, float(monthly[s])))
        # SUB: Only the days when measurements were taken at that station in that month are considered, and the median is used.
        for s, sid in enumerate(sids):
            obs = station_dates.get((sid, yr, mo))
            if not obs:
                continue
            idx = [np.where(uniq == d)[0][0] for d in obs if d in uniq]
            if not idx:
                continue
            v = daily[idx, s]
            v = v[np.isfinite(v)]
            if len(v):
                sub_rows.append((sid, yr, mo, float(np.median(v))))
        if (k + 1) % 24 == 0:
            print(f'    {yr}-{mo:02d} [{k+1}/{len(files)}]', flush=True)

    cols = ['station_id', 'year', 'month', 'val']
    return pd.DataFrame(all_rows, columns=cols), pd.DataFrame(sub_rows, columns=cols)


def extract_era5_skt(si, station_months):
    print('[ERA5 SKT] Extracting from monthly file (ALL + SUB month-matched)...', flush=True)
    ds = xr.open_dataset(ERA5_SKT)
    tdim = detect_time_dim(ds)
    var = detect_var(ds, ['skt', 'skin'])
    ds = ds.sel({tdim: slice(f'{PERIOD[0]}-01', f'{PERIOD[1]}-12')})
    lat_da = xr.DataArray(si['lat'].values, dims='s')
    lon_da = xr.DataArray((si['lon'].values % 360), dims='s')
    vals = ds[var].sel(latitude=lat_da, longitude=lon_da, method='nearest').values  # (time, nsta)
    vals = np.where(vals > 200, vals - 273.15, vals)
    times = pd.to_datetime(ds[tdim].values)
    ds.close()
    yrs, mos = times.year.values, times.month.values
    sids = si['station_id'].values

    all_rows, sub_rows = [], []
    for ti in range(len(times)):
        yr, mo = int(yrs[ti]), int(mos[ti])
        for s, sid in enumerate(sids):
            v = vals[ti, s]
            if not np.isfinite(v):
                continue
            all_rows.append((sid, yr, mo, float(v)))
            if (sid, yr, mo) in station_months:
                sub_rows.append((sid, yr, mo, float(v)))
    cols = ['station_id', 'year', 'month', 'val']
    return pd.DataFrame(all_rows, columns=cols), pd.DataFrame(sub_rows, columns=cols)


def extract_dynwat(si, station_months):
    print('[DynWat] Extracting from station-month pkl (ALL + SUB month-matched, through 2014)...', flush=True)
    with open(DYNWAT_PKL, 'rb') as fh:
        cache = pickle.load(fh)
    keep = set(si['station_id'])
    all_rows, sub_rows = [], []
    for sid, dfs in cache.items():
        sid = str(sid)
        if sid not in keep:
            continue
        d = dfs.copy()
        d.loc[d['dynwat_wtemp'].abs() > 100, 'dynwat_wtemp'] = np.nan
        d = d[(d['year'] >= DW_PERIOD[0]) & (d['year'] <= DW_PERIOD[1])]
        d = d.dropna(subset=['dynwat_wtemp'])
        for r in d[['year', 'month', 'dynwat_wtemp']].itertuples(index=False):
            yr, mo, v = int(r.year), int(r.month), float(r.dynwat_wtemp)
            all_rows.append((sid, yr, mo, v))
            if (sid, yr, mo) in station_months:
                sub_rows.append((sid, yr, mo, v))
    cols = ['station_id', 'year', 'month', 'val']
    return pd.DataFrame(all_rows, columns=cols), pd.DataFrame(sub_rows, columns=cols)


def extract_hadisd(si, station_dates):
    print('[HadISD] Reading per station (ALL monthly mean + SUB day-matched)...', flush=True)
    pairs = pd.read_csv(HADISD_PAIRS)
    pairs['station_id'] = pairs['station_id'].astype(str)
    pairs = pairs[pairs['station_id'].isin(set(si['station_id']))]

    # Each HadISD file → daily-mean table (date→temp), read once
    daily_by_path = {}
    for hpath in pairs['hadisd_path'].dropna().unique():
        if not os.path.exists(hpath):
            continue
        try:
            ds = xr.open_dataset(hpath)
            tv = detect_var(ds, ['temperatures', 'temp', 'air_temperature', 't2m'])
            td = detect_time_dim(ds)
            dft = ds[[tv]].to_dataframe().reset_index().rename(columns={tv: 'temp', td: 'time'})
            ds.close()
        except Exception:
            continue
        dft['time'] = pd.to_datetime(dft['time'], errors='coerce')
        dft = dft.dropna(subset=['time', 'temp'])
        dft['year'] = dft['time'].dt.year
        dft['month'] = dft['time'].dt.month
        dft['day'] = dft['time'].dt.day
        dft = dft[(dft['year'] >= PERIOD[0]) & (dft['year'] <= PERIOD[1])]
        if dft.empty:
            continue
        if dft['temp'].median() > 200:
            dft['temp'] -= 273.15
        dft = dft[dft['temp'].abs() < 100]
        if dft.empty:
            continue
        daily = dft.groupby(['year', 'month', 'day'])['temp'].mean().reset_index()
        daily_by_path[hpath] = daily

    all_rows, sub_rows = [], []
    for _, pr in pairs.iterrows():
        sid, hpath = pr['station_id'], pr['hadisd_path']
        daily = daily_by_path.get(hpath)
        if daily is None:
            continue
        # ALL: median of all daily means in the month (median, consistent with t2m/GEMStat)
        mon = daily.groupby(['year', 'month'])['temp'].median().reset_index()
        for r in mon.itertuples(index=False):
            all_rows.append((sid, int(r.year), int(r.month), float(r.temp)))
        # SUB: median of daily means on sampled days for that station-month
        gb = daily.groupby(['year', 'month'])
        for (yr, mo), sub in gb:
            obs = station_dates.get((sid, int(yr), int(mo)))
            if not obs:
                continue
            v = sub[sub['day'].isin(obs)]['temp'].values
            v = v[np.isfinite(v)]
            if len(v):
                sub_rows.append((sid, int(yr), int(mo), float(np.median(v))))
    cols = ['station_id', 'year', 'month', 'val']
    return pd.DataFrame(all_rows, columns=cols), pd.DataFrame(sub_rows, columns=cols)


# ============================================================
# 3. Aggregation + per-month OLS (cross-station median first, then regression)
# ============================================================
def scale_trends(long_df, si, min_years):
    """Returns {scale: DataFrame}. long_df: station_id, year, month, val."""
    if long_df.empty:
        return {sc: pd.DataFrame() for sc in SCALES}
    gmap = si.set_index('station_id')
    g = long_df.copy()
    g['basin'] = g['station_id'].map(gmap['basin'])
    g['continental'] = g['station_id'].map(gmap['continental'])
    g['global'] = 'GLOBAL'
    g['lat'] = g['station_id'].map(gmap['lat'])
    g['w']   = np.cos(np.radians(g['lat']))
    # Per-station, per-calendar-month anomaly: each station minus its own climatological median for that calendar month.
    # Aggregating anomalies across stations removes the effect of absolute offsets between stations, so SUB's year-to-year change in station composition
    # is not mistaken for a trend (when station composition is stable, the anomaly trend matches the absolute trend).
    clim = g.groupby(['station_id', 'month'])['val'].transform('median')
    g['anom'] = g['val'] - clim

    out = {}
    for scale in SCALES:
        gg = g.dropna(subset=[scale]) if scale != 'global' else g
        if gg.empty:
            out[scale] = pd.DataFrame()
            continue
        # Cross-station median (anomaly) → one value per (group, year, month); also keep the absolute median for reference
        def _wagg_forbasin(sub):
            import pandas as pd
            return pd.Series({
                'anom':   weighted_median(sub['anom'].values, sub['w'].values),
                'absmed': weighted_median(sub['val'].values,  sub['w'].values),
            })
        agg = (gg.groupby([scale, 'year', 'month']).apply(_wagg_forbasin).reset_index())
        nst = gg.groupby(scale)['station_id'].nunique().to_dict()

        rows = []
        for grp, sub in agg.groupby(scale):
            for m in range(1, 13):
                s = sub[sub['month'] == m].dropna(subset=['anom'])
                if len(s) < min_years:
                    continue
                x = s['year'].values.astype(float)
                y = s['anom'].values
                v = np.isfinite(y)
                if v.sum() < min_years:
                    continue
                sl, _, r, p, _ = stats.linregress(x[v], y[v])
                rows.append({
                    scale: grp, 'month': m, 'n_years': int(v.sum()),
                    'n_stations': int(nst.get(grp, 0)),
                    'mean_temp': float(s['absmed'].values[v].mean()),
                    'trend_per_year': sl, 'trend_per_decade': sl * 10,
                    'r_squared': r ** 2, 'p_value': p,
                    'significant': int(p < 0.05),
                })
        out[scale] = pd.DataFrame(rows)
    return out


# Note: the station scale is not computed in this pipeline.
# The station scale uses the legacy logic (per-station QC + >=5 years) (ss_*_trends_5y.csv, ~219 stations),
# produced by an existing script; this file handles only the basin / continental / global (area-based) scales.

CONTINENTS_TS = ['Asia', 'Europe', 'South/SE Asia', 'Latin America', 'North America']
SEAS_MONTHS_TS = {'Boreal Summer': {'NH': [6, 7, 8], 'SH': [12, 1, 2]},
                  'Boreal Winter': {'NH': [12, 1, 2], 'SH': [6, 7, 8]}}

def regional_series(long_df, si):
    """Cross-station median anomaly for each (region, season, year). region = 5 continents + Global.
       Season months are assigned per station by hemisphere (lat>0 uses NH months, otherwise SH months), fixing hemisphere mixing in cross-equator regions.
       Returns [region, season, year, value]。"""
    cols = ['region', 'season', 'year', 'value']
    if long_df.empty:
        return pd.DataFrame(columns=cols)
    gmap = si.set_index('station_id')
    g = long_df.copy()
    g['continental'] = g['station_id'].map(gmap['continental'])
    g['lat'] = g['station_id'].map(gmap['lat'])
    g['w'] = np.cos(np.radians(g['lat']))
    g = g.dropna(subset=['lat'])
    # Per-station, per-calendar-month anomaly
    clim = g.groupby(['station_id', 'month'])['val'].transform('median')
    g['anom'] = g['val'] - clim
    g['hemi'] = np.where(g['lat'] > 0, 'NH', 'SH')

    out = []
    for season, hd in SEAS_MONTHS_TS.items():
        gs = g[((g['hemi'] == 'NH') & (g['month'].isin(hd['NH']))) |
               ((g['hemi'] == 'SH') & (g['month'].isin(hd['SH'])))].copy()
        if gs.empty:
            continue
        # DJF spans years: December is assigned to the following year (both NH winter and SH summer include DJF)
        gs.loc[gs['month'] == 12, 'year'] = gs.loc[gs['month'] == 12, 'year'] + 1
        # Per-station seasonal anomaly = mean of that season's monthly anomalies
        sta = (gs.groupby(['station_id','continental','year'])
               .agg(anom=('anom','mean'), w=('w','first')).reset_index())
        sta = sta[(sta['year'] >= PERIOD[0]) & (sta['year'] <= PERIOD[1])]
        for region in CONTINENTS_TS:
            rr = sta[sta['continental'] == region]
            for yr, sub in rr.groupby('year'):
                v = weighted_median(sub['anom'].values, sub['w'].values)
                out.append({'region':region,'season':season,'year':int(yr),'value':float(v)})
        for yr, sub in sta.groupby('year'):    # Global
            v = weighted_median(sub['anom'].values, sub['w'].values)
            out.append({'region':'Global','season':season,'year':int(yr),'value':float(v)})
    return pd.DataFrame(out, columns=cols)


def write_out(trends, dataset, version):
    os.makedirs(OUT_DIR, exist_ok=True)
    for scale, df in trends.items():
        tag = f'{dataset}_{scale}' + (f'_{version}' if version else '') + FNAME_SUFFIX
        path = OUT_DIR + tag + '.csv'
        df.to_csv(path, index=False)
        med = df['trend_per_decade'].median() if len(df) else float('nan')
        n = df[scale].nunique() if (len(df) and scale in df) else 0
        print(f'  → {tag}.csv  ({len(df)} records, {n} , median={med:.3f})', flush=True)


CACHE_DIR = OUT_DIR + '_stationmonth_cache/'

def cached_extract(name, extractor, eargs, from_cache):
    """Returns (all_long, sub_long). If a cache exists and --from_cache is set, read it directly; otherwise extract and save the cache.
    Purpose: when rerunning with changed aggregation logic (e.g. anomalies), avoids re-reading the hourly ERA5."""
    pa = CACHE_DIR + f'{name}_all.pkl'
    ps = CACHE_DIR + f'{name}_sub.pkl'
    if from_cache and os.path.exists(pa) and os.path.exists(ps):
        print(f'  [cache] Read {name} station-year-month intermediate table (skip re-extraction)', flush=True)
        return pd.read_pickle(pa), pd.read_pickle(ps)
    a, s = extractor(*eargs)
    os.makedirs(CACHE_DIR, exist_ok=True)
    a.to_pickle(pa); s.to_pickle(ps)
    return a, s


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', nargs='+',
                    default=['gemstat', 'era5_t2m', 'era5_skt', 'hadisd', 'dynwat'],
                    choices=['gemstat', 'era5_t2m', 'era5_skt', 'hadisd', 'dynwat'])
    ap.add_argument('--min_years', type=int, default=5)
    ap.add_argument('--station_set', choices=['river', 'qc219'], default='river')
    ap.add_argument('--from_cache', action='store_true',
                    help='If a station-year-month cache exists, read it and skip re-extraction (saves a lot when rerunning with changed aggregation, especially t2m)')
    args = ap.parse_args()
    MY = args.min_years
    global FNAME_SUFFIX
    FNAME_SUFFIX = '' if MY == 5 else f'_{MY}y'
    FC = args.from_cache

    si = load_stations(args.station_set)
    gem_long, station_dates, station_months = build_gemstat(si)

    # Keep only river stations that actually have water-temperature observations: metadata has ~18,000 stations, most without water temperature.
    # All datasets are sampled on the same "water-temperature network" stations, so ALL/SUB is a full-time vs observed-time comparison on the same stations,
    # and this greatly reduces the ERA5 extraction load.
    data_sids = set(gem_long['station_id'].unique())
    n0 = len(si)
    si = si[si['station_id'].isin(data_sids)].reset_index(drop=True)
    print(f' River stations with water-temperature obs: {len(si)} / {n0}(the rest have no obs and are excluded from extraction)', flush=True)

    reg_parts = []   # (colname, df[region,season,year,value]) → later assembled into regional_series_v1.csv

    def add_series(long_df, col):
        rs = regional_series(long_df, si)
        if not rs.empty:
            reg_parts.append((col, rs))

    if 'gemstat' in args.dataset:
        print('\n== GEMStat (single obs)==', flush=True)
        write_out(scale_trends(gem_long, si, MY), 'gemstat', None)
        add_series(gem_long, 'gemstat')

    if 'era5_skt' in args.dataset:
        print('\n== ERA5 SKT ==', flush=True)
        a, s = cached_extract('era5_skt', extract_era5_skt, (si, station_months), FC)
        write_out(scale_trends(a, si, MY), 'era5_skt', 'all')
        write_out(scale_trends(s, si, MY), 'era5_skt', 'sub')
        add_series(s, 'era5_skt'); add_series(a, 'era5_skt_full')

    if 'dynwat' in args.dataset:
        print('\n== DynWat ==', flush=True)
        a, s = cached_extract('dynwat', extract_dynwat, (si, station_months), FC)
        write_out(scale_trends(a, si, MY), 'dynwat', 'all')
        write_out(scale_trends(s, si, MY), 'dynwat', 'sub')
        add_series(s, 'dynwat'); add_series(a, 'dynwat_full')

    if 'hadisd' in args.dataset:
        print('\n== HadISD ==', flush=True)
        a, s = cached_extract('hadisd', extract_hadisd, (si, station_dates), FC)
        write_out(scale_trends(a, si, MY), 'hadisd', 'all')
        write_out(scale_trends(s, si, MY), 'hadisd', 'sub')
        add_series(s, 'hadisd'); add_series(a, 'hadisd_full')

    if 'era5_t2m' in args.dataset:
        print('\n== ERA5 t2m(hourly, slower)==', flush=True)
        a, s = cached_extract('era5_t2m', extract_era5_t2m, (si, station_dates), FC)
        write_out(scale_trends(a, si, MY), 'era5_t2m', 'all')
        write_out(scale_trends(s, si, MY), 'era5_t2m', 'sub')
        add_series(s, 'era5_t2m'); add_series(a, 'era5_t2m_full')

    # Per-year regional anomaly series (for time series a–f): merge each dataset's column into a wide table
    if reg_parts:
        base = None
        for col, df in reg_parts:
            d = df.rename(columns={'value': col})
            base = d if base is None else base.merge(d, on=['region', 'season', 'year'], how='outer')
        base = base.sort_values(['region', 'season', 'year']).reset_index(drop=True)
        base.to_csv(OUT_DIR + 'regional_series_v1.csv', index=False)
        scols = [c for c in base.columns if c not in ('region', 'season', 'year')]
        print(f'\n  → regional_series_v1.csv  ({len(base)} records, columns={scols})', flush=True)

    print(f'\nAll done! Output {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()

# ============================================================
# NOTES / assumptions to verify on the cluster:
# - ERA5 t2m hourly files: variable 't2m', time 'valid_time' (following calc_era5_obs_trends_v2).
# - ERA5 SKT monthly file: variable/time via detect_*; coordinate names latitude/longitude.
# - DynWat pkl: dict[station_id] -> DataFrame (with year, month, dynwat_wtemp).
# - HadISD: pairs have columns station_id, hadisd_path; nc variable 'temperatures', time 'time'.
# - Station list defaults to all river stations (consistent with existing basin scripts); --station_set qc219 switches to the 219-station set.
# - SUB matching: t2m/HadISD per day (median of daily means on sampled days); SKT/DynWat per month (keep observed station-year-month).
# - Cross-station always median; OLS over years for each (group, calendar month), >=MIN_YEARS.
# - continental = 5 continents (assign_region); out-of-box stations (Other) are excluded from continental but still count in global / their basin.
# ============================================================
