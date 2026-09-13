#!/usr/bin/env python3
# ============================================================
# Main_Text_Figure1_map_v1.3.py
# Global River Temperature Monitoring Stations + HydroRIVERS network
#   Change from Main_Text_Figure1_map_v1.0.py。
#   v1.3 Changed: 
#     - legend shipped into Indian Ocean, remove the broader, and change the text saming as Fig 1 caption
#       (Red=stations analysed at all scales; yellow=stations analysed at basin scale and above;
#        River networks = River network (HydroRIVERS, Strahler order ≥ 5))
#     - Remove brackets from Histogram X axis (Stations per basin）and Y axis No.→Number.
#   v1.2 Changed:
#     - Corrected the issue of legend color blocks being out of sync with actual map colors: site colors are now set to a constant.
#       GREY_C / TREND_C and legend shared with the points; River networks legend follows river_c.
#   v1.1 Changed:
#     - Add a small histogram in the bottom left corner (where the legend was originally located): to supplement the explanation of the basin scale in FIG2 below.
#         X = The station numbers included into basin-scale analysis (All River Stations with or without data, not considering QC）
#         Y = Basins numbers
#     - legend moved up to give the place(the location and histogram location designed into adjustable constant: LEG_ANCHOR / HIST_RECT).
#   Stations categories remain unchanged: gray = data is not enough; blue = pass QC and have trend (219).
#   Black and white background each one.
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
warnings.filterwarnings('ignore')

# ── Path ──────────────────────────────────────────────────
DATA_DIR  = '/work/home/H.Jason421/water_temp_for_publish/data/'
FIG_DIR   = '/work/home/H.Jason421/water_temp_for_publish/figures/final/'
META_XLS  = '/work5/H.Jason421/GEMStat_new/GEMS-Water_data_request.xls'   # take Main Basin
RIVER_DIR = '/work/home/H.Jason421/HydroRIVERS_v10_shp/'
RIVER_SHP = RIVER_DIR + 'HydroRIVERS_v10.shp'
RIVER_MIN_STRAHLER = 5      # Only major rivers of this scale (inclusive) and above should be drawn globally; to increase the density, lower the pixel density (4 or 3).
os.makedirs(FIG_DIR, exist_ok=True)

# ── Station colors (legend and scatter share one set to avoid mismatch) ──
GREY_C  = 'khaki'      # Data is not enough
TREND_C = 'crimson'    # Pass QC and have trend (219)

# ── Histogram / legend location (map-axes fractional coordinates; adjust these two to move them)──
HIST_RECT  = [0.025, 0.075, 0.185, 0.255]   # small histogram at lower-left [x0,y0,w,h]
LEG_ANCHOR = (0.45, 0.10)                   # legend placed over the open southern Indian Ocean (south of Madagascar, west of Australia); loc='upper left' anchor. 
                                            # The longitude span here is wide enough that the legend does not overlap land.
LEG_FS     = 13                             # legend font size
# Histogram bins: stations-per-basin is highly skewed, so use range-based bins
HIST_EDGES  = [0.5,1.5,2.5,3.5,5.5,10.5,20.5,50.5,100.5, 1e12]
HIST_LABELS = ['1','2','3','4–5','6–10','11–20','21–50','51–100','100+']

ROB  = ccrs.Mercator()      # (variable name kept from the original file; this is actually a Mercator projection)
PROJ = ccrs.PlateCarree()

# ── HydroRIVERS River Networks───────────────────────────────────────
def load_rivers(min_strahler=RIVER_MIN_STRAHLER, bbox=None):
    """Read only river reaches with ORD_STRA >= threshold (filtered with 'where' at the driver level, avoiding loading all 8.5 million reaches)."""
    kw = {'engine': 'pyogrio', 'columns': ['ORD_STRA']}
    if min_strahler is not None:
        kw['where'] = f'ORD_STRA >= {int(min_strahler)}'
    if bbox is not None:
        kw['bbox'] = tuple(bbox)
    return gpd.read_file(RIVER_SHP, **kw)

def add_rivers(ax, gdf=None, color='#7fb0d4', base_lw=0.22, lw_step=0.14,
               alpha=0.85, zorder=1):
    """Overlay the river network on the cartopy GeoAxes; line width increases with Strahler order (larger rivers thicker, tributaries thinner)."""
    if gdf is None:
        gdf = load_rivers()
    if len(gdf) == 0:
        return
    mo = int(gdf['ORD_STRA'].min())
    for order, sub in gdf.groupby('ORD_STRA'):
        lw = base_lw + (int(order) - mo) * lw_step
        ax.add_geometries(list(sub.geometry.values), crs=PROJ,
                          facecolor='none', edgecolor=color,
                          linewidth=lw, alpha=alpha,
                          zorder=zorder + (int(order) - mo) * 0.01)

# ── Read station data ────────────────────────
print('eading station data...')
# v5y: 219 stations have trends; all others treated as no-data
df_trends   = pd.read_csv(DATA_DIR + 'global_wtemp_monthly_trends_v5y.csv')
df_trends   = df_trends[df_trends['water_type'] == 'River station'].copy()

# Stations with trends (219)
wt_trend_r  = (df_trends[['station_id','latitude','longitude','water_type']]
               .drop_duplicates('station_id').reset_index(drop=True))

# All GEMStat stations
wt_all      = pd.read_csv(DATA_DIR + 'wtemp_stations_failed.csv')
wt_no_trend_r2 = pd.read_csv(DATA_DIR + 'wtemp_stations_no_trend.csv')
wt_trend_old   = pd.read_csv(DATA_DIR + 'wtemp_stations_trend.csv')
wt_all_r    = pd.concat([
    wt_all[wt_all['water_type']=='River station'],
    wt_no_trend_r2[wt_no_trend_r2['water_type']=='River station'],
    wt_trend_old[wt_trend_old['water_type']=='River station']
], ignore_index=True).drop_duplicates('station_id')

# Grey = all stations minus the 219 with trends
trend_ids   = set(wt_trend_r['station_id'].astype(str))
wt_grey_r   = wt_all_r[~wt_all_r['station_id'].astype(str).isin(trend_ids)].copy()

n_grey  = len(wt_grey_r)
n_trend = len(wt_trend_r)

# Annual-mean trend
df_annual = (
    df_trends[df_trends['water_type'] == 'River station']
    .groupby('station_id')['trend_per_decade'].mean()
    .reset_index().rename(columns={'trend_per_decade': 'annual_trend_dec'})
)
wt_trend_r = wt_trend_r.merge(df_annual, on='station_id', how='left')

vmax = round(np.nanpercentile(wt_trend_r['annual_trend_dec'].abs(), 95), 1)
vmin = -vmax
CMAP = plt.cm.RdBu_r
NORM = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
print(f'Colorbar range: {vmin:.2f} ~ {vmax:.2f} °C/dec')

# ── Number of stations per basin included in the basin-scale analysis (all river stations with data, mapped via Main Basin in metadata)──
print('Reading metadata to compute stations per basin (basin-scale coverage)...')
try:
    _meta = pd.read_excel(pd.ExcelFile(META_XLS), sheet_name='Station_Metadata')
    _meta = _meta.rename(columns={'GEMS Station Number':'station_id','Main Basin':'basin'})
    _meta['station_id'] = _meta['station_id'].astype(str)
    _sid2basin = (_meta.dropna(subset=['basin']).drop_duplicates('station_id')
                  .set_index('station_id')['basin'])
    _ab = wt_all_r['station_id'].astype(str).map(_sid2basin).dropna()
    per_basin_counts = _ab.value_counts()          # index=basin, value=number of stations in that basin included in the analysis
    print(f'  {per_basin_counts.size} basins, {int(per_basin_counts.sum())} stations in total;'
          f'median {int(per_basin_counts.median())}, max {int(per_basin_counts.max())} stations/basins')
except Exception as e:
    print(f'  Failed to read metadata; histogram left blank:{e}')
    per_basin_counts = pd.Series(dtype=int)

def hist_bars(counts):
    v = np.asarray(counts.values, float)
    h, _ = np.histogram(v, bins=HIST_EDGES)
    last = len(h)
    while last > 1 and h[last-1] == 0: last -= 1     # trim trailing empty bins
    return np.arange(last), h[:last], HIST_LABELS[:last]

def draw_basin_hist(axh, txt, panel_bg, bar_c):
    axh.set_facecolor(panel_bg); axh.patch.set_alpha(0.88)
    if len(per_basin_counts) == 0:
        axh.text(0.5,0.5,'no basin metadata',ha='center',va='center',
                 transform=axh.transAxes,color=txt,fontsize=9)
        axh.set_xticks([]); axh.set_yticks([]); return
    xpos, h, labels = hist_bars(per_basin_counts)
    axh.bar(xpos, h, width=0.86, color=bar_c, edgecolor='none', alpha=0.95, zorder=3)
    axh.set_xticks(xpos)
    axh.set_xticklabels(labels, fontsize=9, color=txt, rotation=30, ha='right')
    axh.set_xlabel('Stations per basin', fontsize=10, color=txt, labelpad=1.5)
    axh.set_ylabel('Number of basins', fontsize=10, color=txt, labelpad=1.5)
    axh.set_title(f'Basin-scale coverage  ',
#                  f'(n={int(per_basin_counts.sum())} stations / {per_basin_counts.size} basins)',
                  fontsize=10, color=txt, pad=2)
    axh.tick_params(colors=txt, labelsize=9, length=1.5, pad=1)
    for sp in ['top','right']: axh.spines[sp].set_visible(False)
    for sp in ['left','bottom']: axh.spines[sp].set_color(txt); axh.spines[sp].set_linewidth(0.5)

# ── egend handles (colors passed in by the caller to stay consistent with the map)──
def make_legend_handles(grey_c, trend_c, river_c):
    return [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=trend_c, markeredgecolor='white',
               markeredgewidth=0.5, markersize=8,
               label=f'Stations analysed at all scales (n = {n_trend:,})'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=grey_c, markeredgecolor=grey_c,
               markersize=6, label=f'Stations analysed at basin scale and above (n = {n_grey:,})'),
        Line2D([0],[0], color=river_c, linewidth=1.4,
               label=f'River network (HydroRIVERS, Strahler order \u2265 {RIVER_MIN_STRAHLER})'),
    ]

# ── Plotting function ───────────────────────────────────────────────
def make_fig(dark=True):
    bg      = '#0d1b2a' if dark else 'white'
    land_c  = '#111111' if dark else 'dimgray'
    ocean_c = '#0d1b2a' if dark else 'white'
    coast_c = '#555555' if dark else 'white'
    river_c = '#5a9bd4' if dark else 'aqua'
    txt     = 'white' if dark else 'black'
    leg_bg  = '#111111' if dark else 'white'
    leg_ec  = '#666666' if dark else '#aaaaaa'
    bar_c   = '#111111' if dark else 'dimgray'

    fig, ax = plt.subplots(1, 1, figsize=(24, 9),
                           subplot_kw={'projection': ROB})
    ax.set_extent([-20037508, 20037508, -7724253, 15496571], crs=ROB)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(ocean_c)

    ax.add_feature(cfeature.LAND,      facecolor=land_c,  zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor=ocean_c, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color=coast_c, zorder=6)

    # Layer 1: HydroRIVERS network (beneath the stations as a base)
    add_rivers(ax, color=river_c, zorder=1)

    # Layer 2: insufficient-data stations (grey)
    ax.scatter(wt_grey_r['longitude'], wt_grey_r['latitude'],
               s=1, c=GREY_C, marker='o', alpha=0.6,
               transform=PROJ, zorder=2)

    # Layer 3: QC-passed stations with a trend (219)
    valid = wt_trend_r.copy()
    ax.scatter(
        valid['longitude'], valid['latitude'],
        s=16, c=TREND_C, marker='o',
        edgecolors='white', linewidths=0.5,
        alpha=0.95, transform=PROJ, zorder=5
    )

    ax.gridlines(draw_labels=False, linewidth=0.2,
                 alpha=0.3, linestyle='--', color='#888888')

    # Legend (moved up above the histogram; position = LEG_ANCHOR)
    leg = ax.legend(
        handles=make_legend_handles(GREY_C, TREND_C, river_c),
        loc='upper left',
        bbox_to_anchor=LEG_ANCHOR,
        fontsize=LEG_FS, frameon=False,
        labelcolor=txt,
        title='River Stations & River Network',
        title_fontsize=LEG_FS + 1,
        ncol=1, borderpad=0.5, handlelength=1.8,
    )
    leg.get_title().set_color(txt)

    # small histogram (former legend position; supplements the Fig 2 basin scale)
    axh = ax.inset_axes(HIST_RECT)
    draw_basin_hist(axh, txt, leg_bg, bar_c)

    ax.spines['geo'].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR + f"Main_Text_Figure1_map_v1.3_{'black' if dark else 'white'}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {out}')

print('Plotting dark background version...')
make_fig(dark=True)
print('Plotting white background version...')
make_fig(dark=False)
print('All Finish!')
