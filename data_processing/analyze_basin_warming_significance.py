#!/usr/bin/env python3
# ============================================================
# analyze_basin_warming_significance.py
#   Respond two requests from the boss and combine into one bash
#     (1) Area weighting（HydroBASINS Level 3 sub-basin area）
#     (2) Significance test（permutation test which is applicable weighting case）
#
#   Using the GEMStat river temperature data shown in Fig. 3, count the warming and cooling basins, and thest their difference.
#   In the four versions which is 5y, 8y, 10y, and mean, compare three counting ways in boreal summer and winter.
#     A. Numbers（every basin one point）
#     B. Area-weghted （the bigger basin, the more weighting）
#     C. Only significant basin（X labels on Fig. 3）
#
#   Output（figures/Supplementary/）
#     basin_warming_summary.csv    Full number sheet
#     basin_warming_summary.png    summary chart（for boss reads quickly）
#
#   Use
#     python analyze_basin_warming_significance.py
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest

BASE   = '/work/home/H.Jason421/water_temp_for_publish/'
D      = BASE + 'data/scale_trends_v1/'
SHP    = '/work/home/H.Jason421/shapefiles/hybas_global_lev03.shp'
META   = '/work5/H.Jason421/GEMStat_new/GEMS-Water_data_request.xls'
MAP    = BASE + 'data/basin_to_hybas_mapping.csv'
OUT    = BASE + 'figures/Supplementary/'
os.makedirs(OUT, exist_ok=True)

N_PERM = 50000
rng = np.random.default_rng(42)

VERSIONS = [
    ('gemstat_basin.csv',     '5y'),
    ('gemstat_basin_8y.csv',  '8y'),
    ('gemstat_basin_10y.csv', '10y'),
    ('gemstat_basin_mean.csv','mean'),
]
SEASONS = {'Boreal Summer': {'NH': [6, 7, 8],  'SH': [12, 1, 2]},
           'Boreal Winter': {'NH': [12, 1, 2], 'SH': [6, 7, 8]}}

# ── Area sheet（HydroBASINS Level 3，offical sub-basin area）────────────────────────
print('Reading basin area...', flush=True)
g = gpd.read_file(SHP)
area_by_hybas = dict(zip(g['HYBAS_ID'].astype('int64'), g['SUB_AREA']))
mp = pd.read_csv(MAP)
mp['HYBAS_ID'] = mp['HYBAS_ID'].astype('int64')
mp['area'] = mp['HYBAS_ID'].map(area_by_hybas)
AREA = mp.dropna(subset=['area']).groupby('gemstat_basin')['area'].sum().to_dict()
print(f'  The basin with area：{len(AREA)}', flush=True)

# ── Basin latitude（North or South hemisphere, and decide the season and month）─────
meta = pd.read_excel(META, sheet_name='Station_Metadata')
meta = meta.rename(columns={'Water Type': 'wt', 'Main Basin': 'basin', 'Latitude': 'lat'})
meta = meta[meta['wt'] == 'River station'].dropna(subset=['lat', 'basin'])
BLAT = meta.groupby('basin')['lat'].median().to_dict()


def perm_share(signs, areas, n=N_PERM):
    """H0：Warming and Cooling Labels are NOT related with the area. Return（Weighted warming area percentage, two tails p）"""
    signs = np.asarray(signs); areas = np.asarray(areas)
    obs = areas[signs > 0].sum() / areas.sum()
    null = np.empty(n)
    for i in range(n):
        s = rng.permutation(signs)
        null[i] = areas[s > 0].sum() / areas.sum()
    p = 2 * min((null >= obs).mean(), (null <= obs).mean())
    return obs, min(p, 1.0)


def perm_wmean(trends, areas, n=N_PERM):
    """H0：Area-weighted average trends = 0. Return(Weightied average trend, two tails p)"""
    trends = np.asarray(trends); areas = np.asarray(areas)
    w = areas / areas.sum()
    obs = (w * trends).sum()
    null = np.empty(n)
    for i in range(n):
        wp = rng.permutation(areas); wp = wp / wp.sum()
        null[i] = (wp * trends).sum()
    p = 2 * min((null >= obs).mean(), (null <= obs).mean())
    return obs, min(p, 1.0)


# ── Main  Loop ───────────────────────────────────────────────────────────────────
records = []
for fname, ver in VERSIONS:
    df = pd.read_csv(D + fname)
    for sea, hemi in SEASONS.items():
        trends, areas, sigs = [], [], []
        for basin, grp in df.groupby('basin'):
            lat = BLAT.get(basin, 45)
            mo = hemi['NH'] if lat >= 0 else hemi['SH']
            sub = grp[grp['month'].isin(mo)]
            if sub.empty:
                continue
            a = AREA.get(basin, np.nan)
            if np.isnan(a):
                continue
            trends.append(sub['trend_per_decade'].mean())
            areas.append(a)
            sigs.append(bool((sub['p_value'] < 0.05).any()))
        trends = np.array(trends); areas = np.array(areas); sigs = np.array(sigs)

        n_warm = int((trends > 0).sum()); n_cool = int((trends < 0).sum())
        # A. Numbers binomial
        p_count = binomtest(n_warm, n_warm + n_cool, 0.5).pvalue
        # The numbers in the significant basin
        sw = int(((trends > 0) & sigs).sum()); sc = int(((trends < 0) & sigs).sum())
        p_sig = binomtest(sw, sw + sc, 0.5).pvalue if (sw + sc) else np.nan
        # B. Area weighting permutation
        share, p_share = perm_share(np.sign(trends), areas)
        wmean, p_wmean = perm_wmean(trends, areas)

        records.append(dict(
            version=ver, season=sea.replace('Boreal ', ''),
            n_warm=n_warm, n_cool=n_cool, pct_count=100 * n_warm / (n_warm + n_cool),
            p_count=p_count,
            sig_warm=sw, sig_cool=sc, p_sig=p_sig,
            area_warm_share=100 * share, p_perm_share=p_share,
            wmean_trend=wmean, p_wmean=p_wmean,
        ))
        print(f'  {ver:<5} {sea}: count {n_warm}/{n_cool} '
              f'(p={p_count:.3f}), area {100*share:.0f}% (p={p_share:.3f}), '
              f'wmean {wmean:+.3f} (p={p_wmean:.3f})', flush=True)

tab = pd.DataFrame(records)
csv_out = OUT + 'basin_warming_summary.csv'
tab.to_csv(csv_out, index=False, float_format='%.4f')
print(f'\n表：{csv_out}', flush=True)

# ── Output: The warming percentage in three counting ways, and compare each other.──────────────────────────
RED = '#d6604d'; BLUE = '#4393c3'; GREY = '#999999'
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

for ax, sea in zip(axes, ['Summer', 'Winter']):
    sub = tab[tab['season'] == sea].reset_index(drop=True)
    vers = sub['version'].tolist()
    x = np.arange(len(vers)); w = 0.26

    b1 = ax.bar(x - w, sub['pct_count'],       w, color=GREY, label='By count')
    b2 = ax.bar(x,     sub['area_warm_share'], w, color=RED,  label='Area-weighted')
    # The percentage in the significant basin
    #(It only needs another calculate in the warming area of the significant basin, and uses approximation to show the percentage of the significant numbers.)
    sig_pct = 100 * sub['sig_warm'] / (sub['sig_warm'] + sub['sig_cool'])
    b3 = ax.bar(x + w, sig_pct,                w, color=BLUE, label='Significant only')

    ax.axhline(50, color='black', lw=1.0, ls='--', alpha=0.6)
    ax.set_ylim(0, 80)
    ax.set_xticks(x); ax.set_xticklabels(vers)
    ax.set_title(f'Boreal {sea}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Warming share (%)' if sea == 'Summer' else '')
    ax.spines[['top', 'right']].set_visible(False)

    # Mark the permutation p on the area-weighted bar
    for xi, (pct, p) in enumerate(zip(sub['area_warm_share'], sub['p_perm_share'])):
        ax.text(xi, pct + 1.5, f'p={p:.2f}', ha='center', va='bottom',
                fontsize=8, color=RED)

axes[0].legend(loc='upper right', frameon=False, fontsize=9)
fig.suptitle('GEMStat basin warming share: by count vs area-weighted vs significant only\n'
             '(dashed line = 50%; area-weighted p from permutation test, 50,000 shuffles)',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
png_out = OUT + 'basin_warming_summary.png'
fig.savefig(png_out, dpi=300, facecolor='white')
print(f'FIG：{png_out}', flush=True)
print('Finish.', flush=True)
