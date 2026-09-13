#!/usr/bin/env python3
# ============================================================
# print_region_trends_v1.py
#
# Purpose: tabulate the trend rates (°C/decade) that Fig 2 already computes but never prints.
#       The [XX] [YY] [ZZ] in the Abstract / Results global section are copied from here.
#
# The logic is identical to draw_ts in Main_Text_Figure2_time_series_v1.1.py:
#   for each region×season×dataset, take the linregress slope of (year, value) ×10.
#
# Usage: python print_region_trends_v1.py
# ============================================================
import pandas as pd
import numpy as np
from scipy.stats import linregress

SCALE_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/scale_trends_v1/'
CSV = SCALE_DIR + 'regional_series_v1.csv'

# Same column mapping as Fig2
TS_COL_OBS = {'gemstat':'GEMStat TR','era5_t2m':'ERA5 TA SUB','era5_skt':'ERA5 TS SUB',
              'dynwat':'DynWat TR SUB','hadisd':'HadISD TA SUB'}
TS_COL_FULL = {'era5_t2m_full':'ERA5 TA ALL','era5_skt_full':'ERA5 TS ALL',
               'dynwat_full':'DynWat TR ALL','hadisd_full':'HadISD TA ALL'}
ALLCOLS = {**TS_COL_OBS, **TS_COL_FULL}

df = pd.read_csv(CSV)
print(f'Read:{CSV}')
print(f'Columns:{list(df.columns)}\n')

regions = ['Global','Asia','Europe','South/SE Asia','Latin America','North America']
seasons = ['Boreal Summer','Boreal Winter']

def trend_of(sub, col):
    """eturn °C/decade and the p value; return None if fewer than 5 points."""
    if col not in sub.columns: return None, None
    s = sub.dropna(subset=[col])
    x = s['year'].values.astype(float); y = s[col].values
    v = np.isfinite(y)
    if v.sum() < 5: return None, None
    sl,_,_,p,_ = linregress(x[v], y[v])
    return sl*10, p

# Print a table per region
for region in regions:
    print('='*70)
    print(f'  {region}')
    print('='*70)
    for season in seasons:
        sub = df[(df['region']==region) & (df['season']==season)]
        if sub.empty:
            print(f'  [{season}] No data'); continue
        print(f'  [{season}]')
        for col, lbl in ALLCOLS.items():
            t, p = trend_of(sub, col)
            if t is None: continue
            star = '*' if (p is not None and p < 0.05) else ' '
            print(f'      {lbl:16s} {t:+.3f} °C/dec {star}  (p={p:.3f})')
    print()

# Separately list the three numbers needed for the Abstract
print('#'*70)
print('#  The [XX] [YY] [ZZ] in the Abstract / Results global section')
print('#  = the three atmospheric references for Global (which form depends on your main-text wording, usually ALL)')
print('#'*70)
g = df[df['region']=='Global']
for season in seasons:
    s = g[g['season']==season]
    if s.empty: continue
    print(f'\n[{season}]')
    for col, lbl in [('era5_t2m_full','ERA5 TA (ALL)'),
                     ('era5_skt_full','ERA5 TS (ALL)'),
                     ('hadisd_full','HadISD TA (ALL)'),
                     ('era5_t2m','ERA5 TA (SUB)'),
                     ('era5_skt','ERA5 TS (SUB)'),
                     ('hadisd','HadISD TA (SUB)')]:
        t, p = trend_of(s, col)
        if t is None:
            print(f'   {lbl:16s} (insufficient data)'); continue
        print(f'   {lbl:16s} {t:+.3f} °C/dec  (p={p:.3f})')
