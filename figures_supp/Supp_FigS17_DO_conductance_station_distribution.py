#!/usr/bin/env python3
# ============================================================
# Supp_FigS17_DO_conductance_station_distribution.py   (Supplementary Fig. S17)
#   Distribution of GEMStat river stations used in Fig. S13, July.
#   Two categories:
#     - River stations with dissolved oxygen (DO) data          (blue)
#     - River stations with specific conductance data           (orange)
#   Each split into: has-trend (filled) / no-trend (hollow).
#
#   Data sources are identical to Fig. S13:
#     global_do_monthly_trends_v1.csv, global_cond_monthly_trends_v1.csv
#   (river stations only), so this figure documents the station coverage
#   behind Fig. S13.
#
#   Usage: python Supp_FigS17_DO_conductance_station_distribution.py
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/'
OUT_DIR  = '/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary_correct/'
os.makedirs(OUT_DIR, exist_ok=True)
OUT      = OUT_DIR + 'Supplementary_FigS17_DO_conductance_station_distribution.png'

PLOT_MONTH = 7   # July
PROJ     = ccrs.PlateCarree()
ROBINSON = ccrs.Robinson()

COLOR_DO   = '#1f6fb4'   # DO stations (blue)
COLOR_COND = '#e08214'   # specific conductance stations (orange)

def load_month_stations(fname, val_col_decade='trend_per_decade'):
    """回傳該檔Return the River station in PLOT_MONTH of that files, and distinguish has-trend / no-trend.
       has-trend = There is effective trends of the stations in that month; no-trend = There is a stations but no effective trends in that month."""
    df = pd.read_csv(DATA_DIR + fname)
    df = df[df['water_type'] == 'River station'].copy()
    # coordinate of every stations
    coords = df.groupby('station_id')[['latitude', 'longitude']].first()
    # There is effective trends of the stations in that month
    mo = df[(df['month'] == PLOT_MONTH) & df[val_col_decade].notna()]
    trend_ids = set(mo['station_id'].unique())
    all_ids   = set(df['station_id'].unique())
    notrend_ids = all_ids - trend_ids
    trend   = coords.loc[sorted(trend_ids)]
    notrend = coords.loc[sorted(notrend_ids)]
    return trend, notrend


print('Loading DO and specific conductance stations (July)...', flush=True)
do_trend,   do_notrend   = load_month_stations('global_do_monthly_trends_v1.csv')
cond_trend, cond_notrend = load_month_stations('global_cond_monthly_trends_v1.csv')
print(f'  DO   : trend {len(do_trend)}, no-trend {len(do_notrend)}', flush=True)
print(f'  Cond : trend {len(cond_trend)}, no-trend {len(cond_notrend)}', flush=True)


# ── Plot ──
fig = plt.figure(figsize=(18, 9))
ax = fig.add_subplot(1, 1, 1, projection=ROBINSON)
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='#f0ede8', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#d6e8f2', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)
ax.add_feature(cfeature.BORDERS, linewidth=0.25, alpha=0.5, zorder=3)

def scat(df, color, filled):
    if len(df) == 0:
        return
    if filled:
        ax.scatter(df['longitude'], df['latitude'], s=10, c=color,
                   edgecolors='white', linewidths=0.3, alpha=0.9,
                   transform=PROJ, zorder=10)
    else:
        ax.scatter(df['longitude'], df['latitude'], s=10, facecolors='none',
                   edgecolors=color, linewidths=0.6, alpha=0.7,
                   transform=PROJ, zorder=9)

# no-trend drawa first (below) and trend draws later (up)
scat(do_notrend,   COLOR_DO,   False)
scat(cond_notrend, COLOR_COND, False)
scat(do_trend,     COLOR_DO,   True)
scat(cond_trend,   COLOR_COND, True)

ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, linestyle='--', color='gray')

legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_DO,
           markeredgecolor='white', markersize=7, label='DO — trend'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=COLOR_DO, markersize=7, label='DO — no trend'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_COND,
           markeredgecolor='white', markersize=7, label='Specific conductance — trend'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=COLOR_COND, markersize=7, label='Specific conductance — no trend'),
]
ax.legend(handles=legend, loc='lower right', fontsize=8, framealpha=1.0,
          facecolor='white', edgecolor='#aaaaaa', title='Stations (Fig. S13)',
          title_fontsize=8, ncol=2)

ax.set_title('River-station distribution — July (1990–2020)\n'
             'DO and specific conductance stations used in Fig. S13',
             fontsize=13, fontweight='bold', pad=10)

fig.savefig(OUT, dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved:', OUT)
