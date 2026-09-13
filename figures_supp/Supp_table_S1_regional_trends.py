#!/usr/bin/env python3
# ============================================================
# Supp_table_S1_regional_trends.py
#   Supplementary Table: trends and p-values for Fig. 2, by region / season / dataset
#
#   Background
#   The Fig. 2 caption originally said "a circle is ringed in black where the trend is
#   significant at p < 0.05", but the linregress in Main_Text_Figure2_time_series_v1.4.py
#   discarded the p value (sl,_,_,_,_), so black rings were never drawn. Presented in this table instead.
#
#   Also, the Fig. 2 right axis clips trends at ±0.5 °C/decade, drawing anything beyond at the axis boundary.
#   This table gives the true unclipped values; after running, note the few entries with |trend| > 0.5.
#
#   The trend calculation is identical to Fig. 2
#     - uses the raw annual anomaly series (not the 5-year moving average)
#     - sscipy.stats.linregress, slope × 10 converted to °C/decade
#     - stimated only when the number of valid years >= MIN_YEARS
#
#   Usage
#     python Supp_table_S1_regional_trends.py                # 5y
#     python Supp_table_S1_regional_trends.py --min_years 8
#
#   Output -> figures/Supplementary_correct/
#     Supplementary_Table_regional_trends_{ny}y_long.csv   (per row, includes n and R²)
#     Supplementary_Table_regional_trends_{ny}y_wide.csv   (ready to paste into Word)
# ============================================================
import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import linregress

ap = argparse.ArgumentParser()
ap.add_argument('--min_years', type=int, default=5)
ap.add_argument('--alpha', type=float, default=0.05)
args = ap.parse_args()
MIN_YEARS = args.min_years
ALPHA     = args.alpha

BASE_DIR  = '/work/home/H.Jason421/water_temp_for_publish/'
SCALE_DIR = BASE_DIR + 'data/scale_trends_v1/'
OUT_DIR   = BASE_DIR + 'figures/Supplementary_correct/'
os.makedirs(OUT_DIR, exist_ok=True)

SUFFIX = '' if MIN_YEARS == 5 else f'_{MIN_YEARS}y'

# ── Same column mapping as Fig. 2 ─────────────────────────────────────────────────
TS_COL_OBS = {'gemstat': 'GEMStat TR', 'era5_t2m': 'ERA5 TA SUB',
              'era5_skt': 'ERA5 TS SUB', 'dynwat': 'DynWat TR SUB',
              'hadisd': 'HadISD TA SUB'}
TS_COL_FULL = {'era5_t2m_full': 'ERA5 TA ALL', 'era5_skt_full': 'ERA5 TS ALL',
               'dynwat_full': 'DynWat TR ALL', 'hadisd_full': 'HadISD TA ALL'}
COL2LBL = {**TS_COL_OBS, **TS_COL_FULL}

# output order follows the Fig. 2 legend
ROW_ORDER = ['ERA5 TA SUB', 'ERA5 TA ALL', 'ERA5 TS SUB', 'ERA5 TS ALL',
             'HadISD TA SUB', 'HadISD TA ALL', 'DynWat TR SUB', 'DynWat TR ALL',
             'GEMStat TR']
REGIONS = ['Global', 'Asia', 'Europe', 'South/SE Asia',
           'Latin America', 'North America']
SEASONS = ['Boreal Summer', 'Boreal Winter']


def pretty(lbl):
    """ERA5 TA SUB -> ERA5 TA_SUB (manually change to subscript later in Word)"""
    return lbl.replace(' ALL', '_ALL').replace(' SUB', '_SUB')


def fmt_p(p):
    if not np.isfinite(p):
        return 'n/a'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def fmt_cell(row):
    if row is None or not np.isfinite(row['trend_per_decade']):
        return '—'
    star = '*' if row['p_value'] < ALPHA else ''
    return f"{row['trend_per_decade']:+.2f}{star} ({fmt_p(row['p_value'])})"


# ── Read data ───────────────────────────────────────────────────────────────────
pref, base = SCALE_DIR + f'regional_series_v1{SUFFIX}.csv', SCALE_DIR + 'regional_series_v1.csv'
path = pref if os.path.exists(pref) else base
if path == base and SUFFIX:
    print(f'⚠ {os.path.basename(pref)}not found, using {os.path.basename(base)} instead', flush=True)
print(f'Read {os.path.basename(path)}  (MIN_YEARS={MIN_YEARS}, alpha={ALPHA})', flush=True)
df = pd.read_csv(path)

# ── Estimate trends per region × season × dataset ─────────────────────────────────────
rows = []
for region in REGIONS:
    for season in SEASONS:
        d = df[(df['region'] == region) & (df['season'] == season)]
        if d.empty:
            print(f'  ⚠ No data: {region} / {season}', flush=True)
            continue
        for col, lbl in COL2LBL.items():
            if col not in d.columns:
                continue
            sub = d.dropna(subset=[col])
            x = sub['year'].values.astype(float)
            y = sub[col].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < MIN_YEARS:
                continue
            sl, ic, r, p, se = linregress(x[ok], y[ok])
            rows.append({
                'region': region, 'season': season, 'dataset': pretty(lbl),
                'trend_per_decade': sl * 10,
                'se_per_decade': se * 10,
                'p_value': p,
                'r_squared': r ** 2,
                'n_years': int(ok.sum()),
                'significant': 'Yes' if p < ALPHA else 'No',
            })

long = pd.DataFrame(rows)
if long.empty:
    raise SystemExit('No trends were computed; check the column names in regional_series.')

long['dataset'] = pd.Categorical(long['dataset'],
                                 [pretty(l) for l in ROW_ORDER], ordered=True)
long['region'] = pd.Categorical(long['region'], REGIONS, ordered=True)
long = long.sort_values(['season', 'dataset', 'region']).reset_index(drop=True)

out_long = OUT_DIR + f'Supplementary_Table_regional_trends_{MIN_YEARS}y_long.csv'
long.to_csv(out_long, index=False, float_format='%.4f')

# ── Wide table: rows = dataset × season, columns = region域 ────────────────────────────────────
wide_rows = []
for season in SEASONS:
    for lbl in ROW_ORDER:
        rec = {'Season': season, 'Dataset': pretty(lbl)}
        for region in REGIONS:
            m = long[(long['season'] == season) &
                     (long['dataset'] == pretty(lbl)) &
                     (long['region'] == region)]
            rec[region] = fmt_cell(m.iloc[0] if len(m) else None)
        wide_rows.append(rec)

wide = pd.DataFrame(wide_rows)
out_wide = OUT_DIR + f'Supplementary_TableS1_regional_trends_{MIN_YEARS}y_wide.csv'
wide.to_csv(out_wide, index=False)

# ── Terminal view ───────────────────────────────────────────────────────────────
pd.set_option('display.width', 200, 'display.max_columns', 20)
for season in SEASONS:
    print(f'\n=== {season} ===  values in °C per decade, p in parentheses, * marks p < {ALPHA}')
    print(wide[wide['Season'] == season].drop(columns='Season').to_string(index=False))

big = long[long['trend_per_decade'].abs() > 0.5]
print(f'\n--- |trend| > 0.5 °C/decade (points clipped to the boundary on the Fig. 2 right axis): {len(big)} entries ---')
if len(big):
    print(big[['season', 'region', 'dataset', 'trend_per_decade', 'p_value']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

n_sig = (long['significant'] == 'Yes').sum()
print(f'\nSignificant entries {n_sig} / {len(long)}')
print(f'\nSaved:\n  {out_long}\n  {out_wide}')
