#!/usr/bin/env python3
# ============================================================
# calc_global_monthly_trends_v5y.py
#
# Per-station, per-calendar-month linear trends (1990-2020) for the GEMStat
# water-quality variables, plus start-vs-end delta values. This is the
# upstream computation for the station- and basin-scale analyses and for the
# DO / specific-conductance figures (Figs S13, S17).
#
# Outputs (in data/):
#   global_wtemp_monthly_trends_v5y.csv    water temperature
#   global_do_monthly_trends_v1.csv        dissolved oxygen
#   global_airtemp_monthly_trends_v1.csv   ERA5 t2m air temperature at stations
#   global_turb_monthly_trends_v1.csv      turbidity (NTU only)
#   global_cond_monthly_trends_v1.csv      specific conductance
#   global_wtemp_delta_v1.csv, global_do_delta_v1.csv
#   global_turb_delta_v1.csv, global_cond_delta_v1.csv
#
# Each trend block reads back an existing output file if present; delete the
# file to force recomputation.
#
# Converted from the notebook calc_global_monthly_trends_v5y.ipynb.
# NOTE: the water-temperature filename is unified to _v5y here (the notebook
# wrote _v5y in the trend step but read _v1 in the air-temperature and delta
# steps).
# ============================================================
import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---------- Paths ----------
GEMSTAT_PATH = '/work5/H.Jason421/GEMStat_new/'
ERA5_T2M_DIR = '/work5/ERA5/t2m/'
OUTPUT_DIR   = '/work/home/H.Jason421/water_temp_for_publish/data/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Parameters ----------
ANALYSIS_PERIOD     = (1990, 2020)
MIN_DATA_PER_YEAR   = 3
MIN_MONTHS_PER_YEAR = 3
MIN_DATA_FOR_TREND  = 3
MIN_YEARS_FOR_TREND = 5
INCLUDE_WATER_TYPES = ['River station', 'Lake station']
N_YEARS_DELTA       = 3   # number of years at each end for the delta calculation

# Unified water-temperature trend filename (see NOTE in header)
WTEMP_TREND_CSV = OUTPUT_DIR + 'global_wtemp_monthly_trends_v5y.csv'


# ---------- Shared functions ----------
def check_station_quality(df_station):
    """Criterion 1: each year needs >=3 records and >=3 distinct months;
    returns (ok, list of valid years)."""
    if len(df_station) == 0:
        return False, None
    yearly = df_station.groupby('year').agg(
        count    = ('Value', 'count'),
        n_months = ('month', 'nunique')
    ).reset_index()
    valid_years = yearly.loc[
        (yearly['count']    >= MIN_DATA_PER_YEAR) &
        (yearly['n_months'] >= MIN_MONTHS_PER_YEAR),
        'year'
    ].values
    if len(valid_years) == 0:
        return False, None
    return True, valid_years


def calc_monthly_trends(df_filtered, station_row):
    """Compute per-month trends for a single station; returns a list of dicts.
    Criterion 2: each year-month needs >=3 records; needs >=5 valid years."""
    results = []
    for month in range(1, 13):
        md = df_filtered[df_filtered['month'] == month]
        if len(md) == 0:
            continue
        ym = md.groupby('year').agg(
            val_mean = ('Value', 'mean'),
            count    = ('Value', 'count')
        ).reset_index()
        valid = ym[ym['count'] >= MIN_DATA_FOR_TREND]
        if len(valid) < MIN_YEARS_FOR_TREND:
            continue
        x = valid['year'].values
        y = valid['val_mean'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        results.append({
            'station_id'      : station_row['station_id'],
            'country'         : station_row['country'],
            'latitude'        : station_row['latitude'],
            'longitude'       : station_row['longitude'],
            'elevation_m'     : station_row.get('elevation_m', np.nan),
            'water_type'      : station_row['water_type'],
            'month'           : month,
            'n_years'         : len(valid),
            'n_data_points'   : int(valid['count'].sum()),
            'mean_value'      : float(y.mean()),
            'trend_per_decade': slope * 10,
            'p_value'         : p_value,
            'r_squared'       : r_value ** 2,
            'significant'     : 'Yes' if p_value < 0.05 else 'No'
        })
    return results


def trend_for_variable(raw_csv, value_range, output_path, df_meta,
                       unit_filter=None, label=''):
    """Generic per-station monthly-trend computation for one variable.
    Reads back an existing output file if present."""
    if os.path.exists(output_path):
        print(f'Exists, reading back: {output_path}')
        df_tr = pd.read_csv(output_path)
        print(f'  Rows: {len(df_tr)}, stations: {df_tr["station_id"].nunique()}')
        return df_tr

    print(f'Reading {label} data...')
    raw = pd.read_csv(GEMSTAT_PATH + raw_csv, sep=',', encoding='latin-1', low_memory=False)
    raw['date']  = pd.to_datetime(raw['Sample Date'], errors='coerce')
    raw['year']  = raw['date'].dt.year
    raw['month'] = raw['date'].dt.month
    raw['Value'] = pd.to_numeric(raw['Value'], errors='coerce')
    if unit_filter is not None:
        raw = raw[raw['Unit'] == unit_filter]
    raw = raw.dropna(subset=['Value', 'year', 'month'])
    raw = raw[raw['Value'].between(value_range[0], value_range[1])]
    raw = raw[(raw['year'] >= ANALYSIS_PERIOD[0]) & (raw['year'] <= ANALYSIS_PERIOD[1])]
    target_ids = set(df_meta['station_id'])
    raw = raw[raw['GEMS Station Number'].isin(target_ids)]
    print(f'  Valid {label} records: {len(raw):,}')

    print(f'\nComputing {label} trends per station...')
    all_results = []
    passed = skipped = 0
    total = len(df_meta)
    for i, row in df_meta.iterrows():
        if i % 1000 == 0:
            print(f'  {i}/{total}  (passed={passed}, skipped={skipped})')
        df = raw[raw['GEMS Station Number'] == row['station_id']].copy()
        ok, valid_years = check_station_quality(df)
        if not ok:
            skipped += 1
            continue
        passed += 1
        all_results.extend(calc_monthly_trends(df[df['year'].isin(valid_years)], row))

    df_tr = pd.DataFrame(all_results)
    df_tr.to_csv(output_path, index=False)
    print(f'\n{label} trend computation done')
    print(f'  Stations passing QC: {passed}')
    print(f'  Stations skipped: {skipped}')
    print(f'  Rows (station x month): {len(df_tr)}')
    print(f'  Stations with a trend: {df_tr["station_id"].nunique()}')
    print(f'  Saved: {output_path}')
    return df_tr


def calc_delta(csv_filename, value_range, trend_csv, output_path, unit_filter=None):
    """Start-vs-end delta: mean of the earliest N years minus mean of the latest
    N years, per station-month. mu is the mean_value from the trend CSV."""
    df_trend   = pd.read_csv(trend_csv)
    target_ids = set(df_trend['station_id'])
    print(f'Target stations: {len(target_ids)}')

    print(f'Reading {csv_filename}...')
    raw = pd.read_csv(GEMSTAT_PATH + csv_filename, sep=',', encoding='latin-1', low_memory=False)
    raw['date']  = pd.to_datetime(raw['Sample Date'], errors='coerce')
    raw['year']  = raw['date'].dt.year
    raw['month'] = raw['date'].dt.month
    raw['Value'] = pd.to_numeric(raw['Value'], errors='coerce')
    if unit_filter is not None:
        raw = raw[raw['Unit'] == unit_filter]
    raw = raw.dropna(subset=['Value', 'year', 'month'])
    raw = raw[raw['Value'].between(value_range[0], value_range[1])]
    raw = raw[raw['GEMS Station Number'].isin(target_ids)]
    print(f'  Valid records: {len(raw):,}')

    results = []
    for sid, grp in raw.groupby('GEMS Station Number'):
        for month in range(1, 13):
            gm = grp[grp['month'] == month]
            if len(gm) == 0:
                continue
            ym = gm.groupby('year')['Value'].mean().reset_index()
            ym.columns = ['year', 'val']
            ym = ym.sort_values('year')
            if len(ym) < N_YEARS_DELTA * 2:
                continue
            early_years = ym.head(N_YEARS_DELTA)
            late_years  = ym.tail(N_YEARS_DELTA)
            val_early = early_years['val'].mean()
            val_late  = late_years['val'].mean()
            delta     = val_late - val_early
            year_early = early_years['year'].mean()
            year_late  = late_years['year'].mean()
            trend_row = df_trend[(df_trend['station_id'] == sid) & (df_trend['month'] == month)]
            if len(trend_row) == 0:
                continue
            mu = trend_row.iloc[0]['mean_value']
            results.append({
                'station_id' : sid,
                'country'    : trend_row.iloc[0]['country'],
                'latitude'   : trend_row.iloc[0]['latitude'],
                'longitude'  : trend_row.iloc[0]['longitude'],
                'water_type' : trend_row.iloc[0]['water_type'],
                'month'      : month,
                'year_early' : round(year_early, 1),
                'year_late'  : round(year_late,  1),
                'val_early'  : val_early,
                'val_late'   : val_late,
                'delta'      : delta,
                'mu'         : mu,
                'delta_pct'  : delta / mu * 100 if mu != 0 else np.nan,
                'n_early_yrs': len(early_years),
                'n_late_yrs' : len(late_years),
            })

    df_out = pd.DataFrame(results).replace([np.inf, -np.inf], np.nan).dropna(subset=['delta_pct'])
    df_out.to_csv(output_path, index=False)
    print(f'  Valid station-months: {len(df_out)}, stations: {df_out["station_id"].nunique()}')
    print(f'  Saved: {output_path}')
    return df_out


def airtemp_trends(df_meta):
    """ERA5 t2m per-station, per-month trends (nearest grid point to each
    station that has a water-temperature trend)."""
    import xarray as xr
    OUTPUT_AT = OUTPUT_DIR + 'global_airtemp_monthly_trends_v1.csv'
    if os.path.exists(OUTPUT_AT):
        print(f'Exists, reading back: {OUTPUT_AT}')
        df_at = pd.read_csv(OUTPUT_AT)
        print(f'  Rows: {len(df_at)}, stations: {df_at["station_id"].nunique()}')
        return df_at

    # Stations that have a water-temperature trend
    df_wtemp = pd.read_csv(WTEMP_TREND_CSV)
    stations = df_wtemp[['station_id', 'latitude', 'longitude',
                         'water_type', 'country']].drop_duplicates('station_id')
    print(f'Target stations: {len(stations)}')

    years  = list(range(ANALYSIS_PERIOD[0], ANALYSIS_PERIOD[1] + 1))
    months = list(range(1, 13))

    # Read one file to determine the longitude convention
    ds_test  = xr.open_dataset(f'{ERA5_T2M_DIR}T_{years[0]}_{1:02d}.nc')
    lon_raw  = ds_test['longitude'].values
    is_0_360 = lon_raw.max() > 180
    ds_test.close()
    print(f'ERA5 longitude convention: {"0-360" if is_0_360 else "-180-180"}')

    sta_lons = stations['longitude'].values.copy()
    if is_0_360:
        sta_lons = sta_lons % 360

    print('Extracting ERA5 t2m per station...')
    sta_data = {sid: {m: [] for m in months} for sid in stations['station_id']}
    total_steps = len(years) * len(months)
    count = 0
    for yr in years:
        for mo in months:
            count += 1
            if count % 50 == 0:
                print(f'  {count}/{total_steps}  ({yr}-{mo:02d})')
            fpath = f'{ERA5_T2M_DIR}T_{yr}_{mo:02d}.nc'
            try:
                ds = xr.open_dataset(fpath)
                t2m_mean = ds['t2m'].mean(dim='valid_time')  # (lat, lon)
                for idx, row in stations.iterrows():
                    slat = row['latitude']
                    slon = sta_lons[stations.index.get_loc(idx)]
                    t_val = float(t2m_mean.sel(latitude=slat, longitude=slon,
                                               method='nearest').values) - 273.15
                    sta_data[row['station_id']][mo].append((yr, t_val))
                ds.close()
            except Exception:
                pass  # skip missing files

    print('\nComputing air-temperature monthly trends...')
    all_results = []
    for idx, row in stations.iterrows():
        sid = row['station_id']
        for mo in months:
            pts = sta_data[sid][mo]
            if len(pts) < 3:
                continue
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            slope, _, r_val, p_val, _ = stats.linregress(x, y)
            all_results.append({
                'station_id'      : sid,
                'country'         : row['country'],
                'latitude'        : row['latitude'],
                'longitude'       : row['longitude'],
                'water_type'      : row['water_type'],
                'month'           : mo,
                'n_years'         : len(pts),
                'trend_per_decade': slope * 10,
                'trend_per_year'  : slope,
                'p_value'         : p_val,
                'r_squared'       : r_val ** 2,
                'significant'     : 'Yes' if p_val < 0.05 else 'No'
            })
    df_at = pd.DataFrame(all_results)
    df_at.to_csv(OUTPUT_AT, index=False)
    print(f'\nAir-temperature trend done')
    print(f'  Rows: {len(df_at)}, stations: {df_at["station_id"].nunique()}')
    print(f'  Saved: {OUTPUT_AT}')
    return df_at


def main():
    # --- Station list ---
    print('Reading full station list...')
    df_meta = pd.read_excel(
        GEMSTAT_PATH + 'GEMS-Water_data_request.xls', sheet_name='Station_Metadata'
    ).rename(columns={
        'GEMS Station Number': 'station_id', 'Water Type': 'water_type',
        'Latitude': 'latitude', 'Longitude': 'longitude',
        'Country Name': 'country', 'Elevation': 'elevation_m',
    })
    df_meta = df_meta[df_meta['water_type'].isin(INCLUDE_WATER_TYPES)] \
        .dropna(subset=['latitude', 'longitude']).copy().reset_index(drop=True)
    print(f'  River + Lake stations: {len(df_meta)}')
    for wt, cnt in df_meta['water_type'].value_counts().items():
        print(f'    {wt}: {cnt}')

    # --- Monthly trends ---
    print('\n=== Water temperature ===')
    trend_for_variable('Water_Temperature.csv', (-5, 40), WTEMP_TREND_CSV, df_meta, label='water temperature')

    print('\n=== Dissolved oxygen ===')
    trend_for_variable('Dissolved_Oxygen.csv', (0, 20),
                       OUTPUT_DIR + 'global_do_monthly_trends_v1.csv', df_meta, label='DO')

    print('\n=== Air temperature (ERA5 t2m) ===')
    airtemp_trends(df_meta)

    print('\n=== Turbidity ===')
    trend_for_variable('Turbidity.csv', (0, 3000),
                       OUTPUT_DIR + 'global_turb_monthly_trends_v1.csv', df_meta,
                       unit_filter='NTU', label='turbidity')

    print('\n=== Specific conductance ===')
    trend_for_variable('Conductivity.csv', (0, 100000),
                       OUTPUT_DIR + 'global_cond_monthly_trends_v1.csv', df_meta, label='specific conductance')

    # --- Delta (start vs end) ---
    print('\n=== Water-temperature delta ===')
    calc_delta('Water_Temperature.csv', (-5, 40), WTEMP_TREND_CSV,
               OUTPUT_DIR + 'global_wtemp_delta_v1.csv')

    print('\n=== DO delta ===')
    calc_delta('Dissolved_Oxygen.csv', (0, 20),
               OUTPUT_DIR + 'global_do_monthly_trends_v1.csv',
               OUTPUT_DIR + 'global_do_delta_v1.csv')

    print('\n=== Turbidity delta ===')
    calc_delta('Turbidity.csv', (0, 3000),
               OUTPUT_DIR + 'global_turb_monthly_trends_v1.csv',
               OUTPUT_DIR + 'global_turb_delta_v1.csv', unit_filter='NTU')

    print('\n=== Specific-conductance delta ===')
    calc_delta('Conductivity.csv', (0, 100000),
               OUTPUT_DIR + 'global_cond_monthly_trends_v1.csv',
               OUTPUT_DIR + 'global_cond_delta_v1.csv')

    print('\nAll done!')


if __name__ == '__main__':
    main()
