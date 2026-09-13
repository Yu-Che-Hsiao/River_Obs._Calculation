#!/usr/bin/env python3
# ============================================================
# Supp_FigS2_dynwat_global_map.py  (white background, no main title)
# DynWat global TR trend map
# Showed Boreal Summer / Boreal Winter side by side.
#   Changes: into white bckground and remove the suptitle. Everything else remains the same.
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

def weighted_median(values, weights):
    """cos latitude weighted median."""
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[m], weights[m]
    if len(values) == 0: return np.nan
    o = np.argsort(values); values, weights = values[o], weights[o]
    cum = np.cumsum(weights)
    return float(values[np.searchsorted(cum, weights.sum()/2.0)])


DATA_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/'
FIG_DIR  = '/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary_correct/'

# ── Read Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR + 'dynwat_global_trends_v1.csv')
print(f'Loaded: {len(df):,} rows')
print(f'Column: {df.columns.tolist()}')
print(f'trend_dec: min={df["trend_dec"].min():.3f}, max={df["trend_dec"].max():.3f}')

# ── Seasons difinitions ──────────────────────────────────────────────────────────────────
SEASONS = {
    'Boreal Summer': {'NH': [6,7,8], 'SH': [12,1,2]},
    'Boreal Winter': {'NH': [12,1,2], 'SH': [6,7,8]},
}

def get_seasonal_mean(df):
    """The average of every grid in Boreal Summer / Winter"""
    results = {}
    for season, mdef in SEASONS.items():
        mask = (
            ((df['latitude'] > 0) & df['month'].isin(mdef['NH'])) |
            ((df['latitude'] <= 0) & df['month'].isin(mdef['SH']))
        )
        grp = df[mask].groupby(['latitude','longitude'])['trend_dec'].mean().reset_index()
        results[season] = grp
    return results

season_data = get_seasonal_mean(df)

# ── Color Setting ──────────────────────────────────────────────────────────────────
VMIN, VMAX = -1.0, 1.0
CMAP = plt.cm.RdBu_r

# The background in the white background version map
LAND_C  = '#eaeaea'      # land light gray
OCEAN_C = 'white'        # ocean white
COAST_C = '#555555'      # coastline dark gray
BORDER_C = '#c0c0c0'     # national border light gray
TXT_C   = '#222222'      # text darl color

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2, figsize=(18, 7),
    subplot_kw={'projection': ccrs.Robinson()}
)
fig.patch.set_facecolor('white')

for ax, (season, grp) in zip(axes, season_data.items()):
    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor=LAND_C,  zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor=OCEAN_C, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=COAST_C, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.2, edgecolor=BORDER_C, zorder=2)

    sc = ax.scatter(
        grp['longitude'], grp['latitude'],
        c=grp['trend_dec'],
        cmap=CMAP, vmin=VMIN, vmax=VMAX,
        s=0.3, alpha=0.7, zorder=3,
        transform=ccrs.PlateCarree()
    )

    n = len(grp)
    v = grp['trend_dec'].values
    w = np.cos(np.radians(grp['latitude'].values))
    keep = np.isfinite(v) & (np.abs(v) <= 5)   # Filter the outliers (|trend|>5 °C/decade is impossible in physics)
    median = weighted_median(v[keep], w[keep])
    ax.set_title(f'{season}\n(n={n:,} grid points,  area-weighted median={median:+.3f} °C/decade)',
                 fontsize=12, fontweight='bold', pad=8, color=TXT_C)

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(vmin=VMIN, vmax=VMAX))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cb.set_label('Water Temperature Trend (°C/decade)', fontsize=11, color=TXT_C)
cb.ax.tick_params(labelsize=9, colors=TXT_C)
cb.outline.set_edgecolor('#999999')

# (Remove the suptitle already)

plt.tight_layout(rect=[0, 0.1, 1, 1])
out = FIG_DIR + 'DynWat_global_trend_map.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
