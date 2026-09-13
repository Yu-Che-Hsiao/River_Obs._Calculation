#!/usr/bin/env python3
# ============================================================
# count_gemstat_basin_signs.py
#
# Count how many GEMStat TR basins are significantly warming vs significantly cooling in the Fig 3 heatmap.
# The logic is identical to load_basin_seasonal in Main_Text_Figure3_heatmap_v1_2.py:
#   - trend = mean of trend_per_decade over the months of that season for that basin
#   - significant = any month in that season has p_value < 0.05
#   - positive/negative = sign of the mean trend
# So the counts here = the number of red crosses / blue crosses in the GEMStat row of the heatmap.
#
# Usage: python count_gemstat_basin_signs.py
# ------------------------------------------------------------
# FIX (repo cleanup): BLAT was originally an empty dict, so BLAT.get(basin,45) always returned 45,
#   treating every basin as northern hemisphere, flipping southern-hemisphere seasons and making the numbers inconsistent with the heatmap.
#   Now matches Supp_basin_choropleth.py / the heatmap: reads each basin's median
#   station latitude from the GEMStat Excel as BLAT.
# ============================================================
import pandas as pd
import numpy as np

SCALE_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/scale_trends_v1/'
DATA_DIR  = '/work/home/H.Jason421/water_temp_for_publish/data/'
META_XLS  = '/work5/H.Jason421/GEMStat_new/GEMS-Water_data_request.xls'

# Same season definition as the heatmap
SEASONS = {'Boreal Summer': {'NH':[6,7,8],  'SH':[12,1,2]},
           'Boreal Winter': {'NH':[12,1,2], 'SH':[6,7,8]}}

# --- BLAT: exactly the same method as Supp_basin_choropleth.py ---
#   Read River station latitudes from Station_Metadata, take the median per basin.
meta = pd.read_excel(pd.ExcelFile(META_XLS), sheet_name='Station_Metadata')
meta = meta.rename(columns={'Main Basin': 'basin', 'Latitude': 'lat', 'Water Type': 'wt'})
meta = meta[meta['wt'] == 'River station'].dropna(subset=['lat', 'basin'])
BLAT = meta.groupby('basin')['lat'].median().to_dict()
print(f'BLAT: {len(BLAT)} basins basins have latitude info (from Station_Metadata)\n')


def load_basin_seasonal(fname, tcol='trend_per_decade'):
    df = pd.read_csv(SCALE_DIR + fname)
    rows = []
    for basin, grp in df.groupby('basin'):
        lat = BLAT.get(basin, 45)   # fall back to 45 (northern hemisphere) only if latitude not found; normally it should be
        for sea, hemi in SEASONS.items():
            mo = hemi['NH'] if lat >= 0 else hemi['SH']
            sub = grp[grp['month'].isin(mo)]
            if sub.empty: continue
            rows.append({'basin': basin, 'season': sea,
                         'trend': sub[tcol].mean(),
                         'sig': bool((sub['p_value'] < 0.05).any()) if 'p_value' in sub.columns else False})
    return pd.DataFrame(rows)


print('Read gemstat_basin.csv, applying the Fig 3 heatmap aggregation logic...\n')
d = load_basin_seasonal('gemstat_basin.csv')

# Check how many basins fell back to the default because latitude was not found (ideally 0)
covered = pd.read_csv(SCALE_DIR + 'gemstat_basin.csv')['basin'].unique()
missing = [b for b in covered if b not in BLAT]
if missing:
    print(f'[Warning] {len(missing)} basins had no latitude in BLAT and fell back to the northern-hemisphere default: {missing[:10]}...\n')

print(f'Number of basin-season combinations with GEMStat seasonal data:{len(d)}')
print(f'Number of unique basins::{d["basin"].nunique()}\n')

print('='*56)
print('  GEMStat TR basins significantly warm / significantly cool (= heatmap red/blue crosses)')
print('='*56)
for sea in ['Boreal Summer', 'Boreal Winter']:
    s = d[d['season'] == sea]
    sig = s[s['sig']]
    warm = int((sig['trend'] > 0).sum())
    cool = int((sig['trend'] < 0).sum())
    total_sig = len(sig)
    total = len(s)
    print(f'\n[{sea}]')
    print(f'  Significantly warm (red cross) : {warm} basins')
    print(f'  Significantly cool (blue cross): {cool} basins')
    print(f'  Significant total              : {total_sig} / {total} basins ({100*total_sig/total:.0f}% significant)')

# Both seasons combined (if the main text needs a single total)
print('\n' + '='*56)
print('  Both seasons combined (if the main text uses a single number)')
print('='*56)
sig_all = d[d['sig']]
print(f'  Significantly warm: {int((sig_all["trend"]>0).sum())} Significantly cool: {int((sig_all["trend"]<0).sum())}')
print('\nFill into main text: splits between [warm] warming and [cool] cooling basins')
print(' (choose the matching number depending on whether the main text splits by season; the basin paragraph usually uses summer or one of the two seasons, consistent with the figure)')
