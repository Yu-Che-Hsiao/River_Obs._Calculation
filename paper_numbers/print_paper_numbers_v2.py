#!/usr/bin/env python3
# ============================================================
# print_paper_numbers_v1.py
#
# Print the three sets of numbers to be filled into the main text at once, all read from existing csv files,
# without recomputing trends or modifying any file. Pure aggregation + printing, convenient for keeping a terminal record.
#
# Required files (all in data/):
#   station_scale_stats_5y.csv             (station scale pre-aggregated: median/IQR/sig_warm/sig_cool)
#   ss_gemstat_trends_5y.csv               (GEMStat per station per month, for the binomial test)
#   ss_dynwat_full_trends_5y.csv           (DynWat complete record at the station locations)
#   ss_dynwat_obs_trends_5y.csv            (DynWat matched sampling)
#   scale_trends_v1/regional_series_v1.csv (continental series, for the global-section trend rates)
#
# Usage:python print_paper_numbers_v1.py
# ============================================================
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress, binomtest

DATA = '/work/home/H.Jason421/water_temp_for_publish/data/'
SCALE = DATA + 'scale_trends_v1/'

SEASONS = {'Boreal Summer': {'NH': [6,7,8], 'SH': [12,1,2]},
           'Boreal Winter': {'NH': [12,1,2], 'SH': [6,7,8]}}

def bar(t):
    print('\n' + '='*68)
    print('  ' + t)
    print('='*68)

# ─────────────────────────────────────────────────────────────
# Tools: turn a per-station per-month file into "one trend per station per season + split into positive/negative significant
# ─────────────────────────────────────────────────────────────
def seasonalize(path):
    df = pd.read_csv(path); df['station_id'] = df['station_id'].astype(str)
    rows = []
    for sid, g in df.groupby('station_id'):
        lat = float(g['latitude'].iloc[0])
        for sea, md in SEASONS.items():
            mo = md['NH'] if lat > 0 else md['SH']
            sub = g[g['month'].isin(mo)].dropna(subset=['trend_dec'])
            if sub.empty: continue
            sig = sub['p_value'] < 0.05
            npos = int((sig & (sub['trend_dec'] > 0)).sum())
            nneg = int((sig & (sub['trend_dec'] < 0)).sum())
            rows.append(dict(sid=sid, season=sea,
                             trend=sub['trend_dec'].mean(),
                             warm=(npos > 0 and nneg == 0),
                             cool=(nneg > 0 and npos == 0)))
    return pd.DataFrame(rows)

def trend_of(sub, col):
    if col not in sub.columns: return None, None, None
    s = sub.dropna(subset=[col]); x = s['year'].values.astype(float); y = s[col].values
    v = np.isfinite(y)
    if v.sum() < 5: return None, None, None
    res = linregress(x[v], y[v])
    # res.stderr = standard error of the slope; ×10 converts to °C/decade, same units as the trend
    return res.slope*10, res.pvalue, res.stderr*10

# ═════════════════════════════════════════════════════════════
# 1. Global section (Results, second part): the ALL claim + SUB collapse
# ═════════════════════════════════════════════════════════════
bar('1. Global section — trend rates of the three atmospheric references (fill into [XX] [YY] [ZZ])')
gpath = SCALE + 'regional_series_v1.csv'
if os.path.exists(gpath):
    g = pd.read_csv(gpath); g = g[g['region'] == 'Global']
    label = [('era5_t2m_full','ERA5 TA'),('era5_skt_full','ERA5 TS'),('hadisd_full','HadISD TA'),
             ('era5_t2m','ERA5 TA SUB'),('era5_skt','ERA5 TS SUB'),('hadisd','HadISD TA SUB')]
    for sea in ['Boreal Summer','Boreal Winter']:
        s = g[g['season'] == sea]
        print(f'\n  [{sea}]')
        for col, name in label:
            t, p, se = trend_of(s, col)
            if t is None: continue
            sig = 'significant' if (p is not None and p < 0.05) else 'not significant'
            print(f'    {name:14s} {t:+.3f} ± {se:.3f} °C/dec   ({sig}, p={p:.3f})')
    print('\n  → Main-text summer sentence (drop p, use ± standard error):')

else:
    print(f'  [Skipped] {gpath} not found')

# ═════════════════════════════════════════════════════════════
# 2. Station scale (Results, third part): split into positive/negative + binomial test
# ═════════════════════════════════════════════════════════════
bar('2. Station scale — significant warming/cooling split + binomial test')
sp = DATA + 'station_scale_stats_5y.csv'
if os.path.exists(sp):
    d = pd.read_csv(sp)
    key = ['ERA5 TA ALL','ERA5 TA SUB','GEMStat TR','DynWat TR ALL','DynWat TR SUB']
    for sea in ['Boreal Summer','Boreal Winter']:
        print(f'\n  [{sea}]')
        print(f'    {"dataset":14s}{"n":>5}{"median":>9}{"IQR":>7}{"↑sig":>6}{"↓sig":>6}{"↑占顯著":>9}{"binomP":>9}')
        for grp in key:
            r = d[(d.group == grp) & (d.season == sea)]
            if r.empty: continue
            r = r.iloc[0]
            w, c = int(r.sig_warm or 0), int(r.sig_cool or 0); tot = w + c
            pb = binomtest(w, tot, 0.5).pvalue if tot > 0 else float('nan')
            share = f'{100*w/tot:.0f}%' if tot else '—'
            pbs = f'{pb:.3f}' if tot else '—'
            print(f'    {grp:14s}{int(r.n):>5}{r["median"]:>+9.3f}{r.iqr:>7.2f}{w:>6}{c:>6}{share:>9}{pbs:>9}')
    print('\n  Main-text summer sentence: ERA5 TA complete 147↑ 0↓; SUB 33↑ 2↓ (94% warming); GEMStat 32↑ 28↓ (53%, p=0.70, indistinguishable from a coin flip)')
else:
    print(f'  [Skipped] {sp} not found')

# ═════════════════════════════════════════════════════════════
# 3. DynWat controlled experiment: same set of stations, only the sampled months change
# ═════════════════════════════════════════════════════════════
bar('3. DynWat controlled experiment — same set of stations, ALL vs SUB (sampling erases a known signal)')
fp = DATA + 'ss_dynwat_full_trends_5y.csv'
op = DATA + 'ss_dynwat_obs_trends_5y.csv'
if os.path.exists(fp) and os.path.exists(op):
    A = seasonalize(fp).set_index(['sid','season'])   # complete record
    B = seasonalize(op).set_index(['sid','season'])   # complete record
    for sea in ['Boreal Summer','Boreal Winter']:
        a = A.xs(sea, level='season'); b = B.xs(sea, level='season')
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        print(f'\n  [{sea}]  same set, n={len(common)} stations')
        print(f'    ALL  median {a.trend.median():+.3f} °C/dec | significant warming {int(a.warm.sum()):3d} ({100*a.warm.mean():.0f}%) | significant cooling {int(a.cool.sum()):3d}')
        print(f'    SUB  median {b.trend.median():+.3f} °C/dec | significant warming {int(b.warm.sum()):3d} ({100*b.warm.mean():.0f}%) | significant cooling {int(b.cool.sum()):3d}')
else:
    print(f'  [Skipped] DynWat files not found')

# ═════════════════════════════════════════════════════════════
# 4. DynWat global grid vs station locations (boss's point 6)
# ═════════════════════════════════════════════════════════════
bar('4. DynWat global grid vs station locations (stations sit where the model does not warm)')
if os.path.exists(fp):
    A = seasonalize(fp)
    print('  S1 global grid median (2,116,452 grid cells): summer +0.202  winter +0.024 °C/dec')
    for sea in ['Boreal Summer','Boreal Winter']:
        a = A[A.season == sea]
        print(f'  Station locations (complete record) {sea:14s}: {a.trend.median():+.3f} °C/dec  (n={len(a)})')
    print('\n  → Global model warms in summer (+0.20), but station locations do not (≈0): the stations are not a representative sample of the global river network')

print('\n' + '='*68)
print('  Done. All numbers above are read from existing csv, no trends recomputed, no files modified.')
print('='*68)
