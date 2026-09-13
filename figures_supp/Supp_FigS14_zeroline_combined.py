#!/usr/bin/env python3
# ============================================================
# Supp_zeroline_combined.py
#   Supplementary Fig. S8 (new numbering): 0 degC isotherm, global + regional insets
#   in ONE figure.
#     top row     : global maps, January (boreal winter) and July (boreal summer)
#     bottom rows : regional insets (same two months per region)
#   Station symbology matches Main Text Fig. 1 (two classes only):
#     red    = stations analysed at all scales            (n = 219)
#     khaki  = stations analysed at basin scale and above (the rest)
#   Zero-degree isotherm: decade means of ERA5 t2m, contoured at 273.15 K,
#   dashed density increasing with decade (1990s -> 2020), as in the original notebooks.
# ============================================================
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

# ---------- paths ----------
BASE_DIR  = '/work/home/H.Jason421/water_temp_for_publish/'
DATA_DIR  = BASE_DIR + 'data/'
FIG_DIR   = BASE_DIR + 'figures/Supplementary_correct/'
CACHE     = DATA_DIR + 't2m_data_v2.3.pkl'          # (t2m_data, lat_global, lon_global)
OUT       = FIG_DIR + 'Supplementary_FigS14_zeroline_combined.png'
os.makedirs(FIG_DIR, exist_ok=True)

#   Station source: same as Main Text Fig. 1
#   Red(219): 5-year threshold trend file (Note: wtemp_stations_trend.csv is 3-year threshold trend file with 384 stations. Do not use)
F_TREND5   = DATA_DIR + 'global_wtemp_monthly_trends_v5y.csv'
F_TREND    = DATA_DIR + 'wtemp_stations_trend.csv'
F_NO_TREND = DATA_DIR + 'wtemp_stations_no_trend.csv'
F_FAILED   = DATA_DIR + 'wtemp_stations_failed.csv'

# ---------- style (Align Main Text Fig. 1)----------
GREY_C  = 'khaki'      # basin scale and above
TREND_C = 'crimson'    # all scales (219)

PROJ = ccrs.PlateCarree()
ROB  = ccrs.Robinson()
T_LEVEL = [273.15]

PLOT_MONTHS = [1, 7]
MONTH_LABEL = {1: 'Boreal Winter (January)', 7: 'Boreal Summer (July)'}

DECADE_RANGES = {1990: list(range(1990, 2000)),
                 2000: list(range(2000, 2010)),
                 2010: list(range(2010, 2020)),
                 2020: [2020]}
ZEROLINE_STYLES = {
    1990: {'color': 'black', 'lw': 0.8, 'linestyle': (0, (8, 8)), 'label': '1990s', 'alpha': 0.55},
    2000: {'color': 'black', 'lw': 0.9, 'linestyle': (0, (5, 5)), 'label': '2000s', 'alpha': 0.65},
    2010: {'color': 'black', 'lw': 1.0, 'linestyle': (0, (3, 3)), 'label': '2010s', 'alpha': 0.80},
    2020: {'color': 'black', 'lw': 1.3, 'linestyle': 'solid',     'label': '2020',  'alpha': 1.00},
}

REGIONS = {
    'North America':   {'lon': (-130, -60), 'lat': (15,  60)},
    'Europe':          {'lon': (-10,   40), 'lat': (35,  70)},
    'India':           {'lon': ( 65,  100), 'lat': ( 5,  35)},
    'NE Asia':         {'lon': (100,  145), 'lat': (30,  55)},
    'South America':   {'lon': (-85,  -30), 'lat': (-60, 15)},
    'Southern Africa': {'lon': ( 10,   60), 'lat': (-40,-10)},
}
REGION_BOX_C = '#444444'

LAND_C, OCEAN_C = '#f2f0ec', '#e3eef5'


# ---------- helpers ----------
def in_reg(df, lon_min, lon_max, lat_min, lat_max, pad=1.0):
    return df[(df['longitude'] >= lon_min - pad) & (df['longitude'] <= lon_max + pad) &
              (df['latitude']  >= lat_min - pad) & (df['latitude']  <= lat_max + pad)]


def draw_region_box(ax, lon_r, lat_r, color=REGION_BOX_C):
    n = 60
    for lat_v in lat_r:
        ax.plot(np.linspace(*lon_r, n), np.full(n, lat_v), transform=PROJ,
                color=color, lw=1.0, ls='--', alpha=0.85, zorder=7)
    for lon_v in lon_r:
        ax.plot(np.full(n, lon_v), np.linspace(*lat_r, n), transform=PROJ,
                color=color, lw=1.0, ls='--', alpha=0.85, zorder=7)


def draw_zerolines(ax, t2m_data, lat_g, lon_g, mo):
    for decade, yrs in DECADE_RANGES.items():
        st = ZEROLINE_STYLES[decade]
        frames = [t2m_data[mo][y] for y in yrs
                  if y in t2m_data.get(mo, {}) and t2m_data[mo][y] is not None]
        if not frames:
            continue
        mean_dec = np.mean(np.stack(frames, axis=0), axis=0)
        try:
            ax.contour(lon_g, lat_g, mean_dec, levels=T_LEVEL,
                       colors=[st['color']], linewidths=st['lw'],
                       linestyles=[st['linestyle']], alpha=st['alpha'],
                       transform=PROJ, zorder=4)
        except Exception:
            pass


def draw_stations(ax, red, yell, s_yell, s_red):
    ax.scatter(yell['longitude'], yell['latitude'], s=s_yell, c=GREY_C,
               marker='o', alpha=0.6, transform=PROJ, zorder=2, linewidths=0)
    ax.scatter(red['longitude'], red['latitude'], s=s_red, c=TREND_C,
               marker='o', edgecolors='white', linewidths=0.5,
               alpha=0.95, transform=PROJ, zorder=5)


def main():
    # ---- t2m cache ----
    with open(CACHE, 'rb') as f:
        t2m_data, lat_g, lon_g = pickle.load(f)
    # key normalized to int (tolerate str/float key)
    t2m_data = {int(float(mk)): {int(float(yk)): v for yk, v in yd.items()}
                for mk, yd in t2m_data.items()}
    print('t2m months:', sorted(t2m_data.keys()))

    # ---- stations: only take River station, which two types to convert Fig.1. ----
    def riv(path):
        d = pd.read_csv(path)
        return d[d['water_type'] == 'River station']

    # Red dot: Same as Fig.1 and take 5-year threshold with 219 station
    df5 = pd.read_csv(F_TREND5)
    df5 = df5[df5['water_type'] == 'River station']
    wt_red = (df5[['station_id', 'latitude', 'longitude', 'water_type']]
              .drop_duplicates('station_id').reset_index(drop=True))

    # Yellow dot: all river station with data and not include red dot(failed + no_trend + trend Union of Three Files).
    wt_all = pd.concat([riv(F_FAILED), riv(F_NO_TREND), riv(F_TREND)],
                       ignore_index=True).drop_duplicates('station_id')
    wt_yell = wt_all[~wt_all['station_id'].astype(str)
                     .isin(wt_red['station_id'].astype(str))]
    print(f'stations: all-scales(red)={len(wt_red):,}  basin-and-above(khaki)={len(wt_yell):,}')

    # ---- layout: 1 global row (2 cols) + inset rows (2 cols x 3 regions) ----
    n_reg = len(REGIONS)
    fig = plt.figure(figsize=(20, 8 + 4.2 * n_reg // 2 + 2), facecolor='white')
    gs = gridspec.GridSpec(1 + (n_reg + 1) // 2, 2, figure=fig,
                           height_ratios=[1.35] + [1.0] * ((n_reg + 1) // 2),
                           hspace=0.12, wspace=0.05,
                           left=0.03, right=0.97, top=0.96, bottom=0.09)

    # -- top: global, two months --
    for j, mo in enumerate(PLOT_MONTHS):
        ax = fig.add_subplot(gs[0, j], projection=ROB)
        ax.set_global()
        ax.add_feature(cfeature.LAND,      facecolor=LAND_C,  zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor=OCEAN_C, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=6)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, alpha=0.6, zorder=6)
        draw_stations(ax, wt_red, wt_yell, s_yell=1.0, s_red=14)
        draw_zerolines(ax, t2m_data, lat_g, lon_g, mo)
        for r, cfg in REGIONS.items():
            draw_region_box(ax, cfg['lon'], cfg['lat'])
        ax.set_title(MONTH_LABEL[mo], fontsize=14, fontweight='bold', pad=6)

    # -- insets: each region, two months side by side --
    for i, (rname, cfg) in enumerate(REGIONS.items()):
        row, col = 1 + i // 2, i % 2
        sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row, col], wspace=0.03)
        for j, mo in enumerate(PLOT_MONTHS):
            ax = fig.add_subplot(sub[0, j], projection=PROJ)
            lon_min, lon_max = cfg['lon']; lat_min, lat_max = cfg['lat']
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=PROJ)
            ax.add_feature(cfeature.LAND,      facecolor=LAND_C,  zorder=0)
            ax.add_feature(cfeature.OCEAN,     facecolor=OCEAN_C, zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=6)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.25, alpha=0.6, zorder=6)
            draw_stations(ax,
                          in_reg(wt_red,  lon_min, lon_max, lat_min, lat_max),
                          in_reg(wt_yell, lon_min, lon_max, lat_min, lat_max),
                          s_yell=3, s_red=26)
            draw_zerolines(ax, t2m_data, lat_g, lon_g, mo)
            if j == 0:
                ax.text(-0.04, 0.5, rname, transform=ax.transAxes, rotation=90,
                        ha='center', va='center', fontsize=11, fontweight='bold')
            ax.set_title(MONTH_LABEL[mo].split('(')[1].rstrip(')'), fontsize=9, pad=3)

    # -- legend (bottom, no frame) --
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TREND_C,
               markeredgecolor='white', markeredgewidth=0.5, markersize=8,
               label=f'Stations analysed at all scales (n = {len(wt_red):,})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREY_C,
               markeredgecolor=GREY_C, markersize=6,
               label=f'Stations analysed at basin scale and above (n = {len(wt_yell):,})'),
    ] + [Line2D([0], [0], color=s['color'], lw=s['lw'], ls=s['linestyle'], alpha=s['alpha'],
                label=f"0 °C isotherm, {s['label']}") for s in ZEROLINE_STYLES.values()]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.012))

    fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
