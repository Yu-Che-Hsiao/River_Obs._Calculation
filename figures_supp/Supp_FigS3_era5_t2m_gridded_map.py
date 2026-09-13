#!/usr/bin/env python3
# ============================================================
# Supp_FigS3_era5_t2m_gridded_map.py
#   Supplementary Fig. S11a: ERA5 2-m air temperature (T2m) GLOBAL GRIDDED
#   trend map, Boreal Summer / Winter, white background, land+ocean filled.
#   Data: t2m_data_v2.3.pkl = tuple( {month:{year: array(nlat,nlon)}}, lat, lon )
#   Method: per grid cell, per calendar month, OLS trend over 1990-2020
#           (x10 -> C/decade); then per hemisphere average the season's month trends
#           (same as the DynWat map).
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

def weighted_median(values, weights):
    """cos latitude weighted median"""
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[m], weights[m]
    if len(values) == 0: return np.nan
    o = np.argsort(values); values, weights = values[o], weights[o]
    cum = np.cumsum(weights)
    return float(values[np.searchsorted(cum, weights.sum()/2.0)])


DATA_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/'
FIG_DIR  = '/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary_correct/'
PKL      = DATA_DIR + 't2m_data_v2.3.pkl'
OUT      = 'Supplementary_FigS3_ERA5_TA_global_trend.png'
CBAR     = 'Air Temperature Trend (°C/decade)'

YEARS = list(range(1990, 2021))
VMIN, VMAX = -1.0, 1.0
CMAP = plt.cm.RdBu_r
LAND_C, OCEAN_C, COAST_C, BORDER_C, TXT_C = '#eaeaea', 'white', '#555555', '#c0c0c0', '#222222'


def load_grid(pkl):
    obj = pd.read_pickle(pkl)
    data = obj[0] if isinstance(obj, tuple) else obj      # {month:{year:grid}}
    # Normalize the key to int (accepts str / float / numpy types; does not copy arrays)
    data = {int(float(mk)): {int(float(yk)): v for yk, v in yd.items()}
            for mk, yd in data.items()}
    months = sorted(data.keys())
    yrs_all = sorted(set().union(*[set(data[m].keys()) for m in data]))
    print(f'pkl: months={months}  years={yrs_all[0]}..{yrs_all[-1]} (n={len(yrs_all)})')
    sample = next(iter(next(iter(data.values())).values()))
    nlat, nlon = np.asarray(sample).shape
    lat = lon = None
    if isinstance(obj, tuple):
        for x in obj[1:]:
            arr = np.asarray(x).ravel()
            if arr.size == nlat and lat is None:   lat = arr.astype(float)
            elif arr.size == nlon and lon is None: lon = arr.astype(float)
    if lat is None: lat = np.linspace(90, -90, nlat)
    if lon is None: lon = np.linspace(0, 360, nlon, endpoint=False)
    return data, lat, lon


def month_trend(data, m, years, order):
    yrs = [y for y in years if y in data.get(m, {})]
    if not yrs:
        avail = sorted(data.get(m, {}).keys())
        raise SystemExit(f'Month {m} in {years[0]}-{years[-1]} is no yearly data;'
                         f'The actual available year for this month = {avail[:3]}...{avail[-3:] if len(avail)>3 else ""}。'
                         f'Please adjust YEARS or confirm the pkl year range.')
    stack = np.stack([np.asarray(data[m][y], dtype='float32') for y in yrs])
    stack = stack[:, :, order]
    yv = np.array(yrs, dtype='float64'); yy = yv - yv.mean()
    num = (yy[:, None, None] * (stack - stack.mean(0))).sum(0)
    den = (yy ** 2).sum()
    return (num / den) * 10.0


def main():
    data, lat, lon = load_grid(PKL)
    lon = ((lon + 180) % 360) - 180
    order = np.argsort(lon); lon = lon[order]
    print(f'grid: nlat={len(lat)} nlon={len(lon)}  lat[{lat.min():.1f},{lat.max():.1f}]')

    REP_SUMMER, REP_WINTER = 7, 1          # Present month: NH summer = July and winter = January (Opposite in SH)
    t_jul = month_trend(data, REP_SUMMER, YEARS, order)
    t_jan = month_trend(data, REP_WINTER, YEARS, order)
    latcol = lat[:, None]
    season_grid = {
        'Boreal Summer': np.where(latcol > 0, t_jul, t_jan),   # NH used July and SH used January
        'Boreal Winter': np.where(latcol > 0, t_jan, t_jul),
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             subplot_kw={'projection': ccrs.Robinson()})
    fig.patch.set_facecolor('white')
    from cartopy.util import add_cyclic_point
    levels = np.linspace(VMIN, VMAX, 21)
    for ax, (season, grid) in zip(axes, season_grid.items()):
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_C, zorder=0)
        grid_c, lon_c = add_cyclic_point(grid, coord=lon)   # Repairing longitude seams
        LONc, LATc = np.meshgrid(lon_c, lat)
        ax.contourf(LONc, LATc, grid_c, levels=levels, cmap=CMAP, extend='both',
                    transform=ccrs.PlateCarree(), zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=COAST_C,  zorder=2)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.2, edgecolor=BORDER_C, zorder=2)
        w2d = np.cos(np.radians(lat))[:, None] * np.ones_like(grid)
        med = weighted_median(grid.ravel(), w2d.ravel())
        ax.set_title(f'{season}\n(area-weighted median={med:+.3f} °C/decade)',
                     fontsize=12, fontweight='bold', pad=8, color=TXT_C)

    cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.025])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(vmin=VMIN, vmax=VMAX)); sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
    cb.set_label(CBAR, fontsize=11, color=TXT_C)
    cb.ax.tick_params(labelsize=9, colors=TXT_C)
    cb.outline.set_edgecolor('#999999')

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(FIG_DIR + OUT, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', FIG_DIR + OUT)


if __name__ == '__main__':
    main()
