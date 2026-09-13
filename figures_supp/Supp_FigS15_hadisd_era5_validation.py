#!/usr/bin/env python3
# ============================================================
# Supp_hadisd_era5_validation.py
#   Supplementary Fig. S5: Validation of ERA5 air-temperature trends
#   against HadISD in-situ observations
#
# Change from Cell 5 from old_bash/Fig_HadISD_validation_v2.ipynb 
#
#     c27  Zoom in; xlim(-0.25,0.25), ylim(-0.25,0.25)
#          -> Axis change to °C/year from °C/decade (±0.25 is no meaning under °C/year. 
#             °C/decade is the resonable range and same unit with the main text).
#             Use AXIS_LIM also. 。Points outside the range will have their printouts printed on the terminal.。
#     c28  Use Spearman R
#          -> The main statistics were revised to Spearman rho; Pearson r is still calculated but only printed on the terminal.
#     c29  Delete panel c
#          -> Change to two lattice from three
#
#   Output -> figures/Supplementary_correct/Supplementary_FigS15_HadISD_ERA5_validation.png
# ============================================================
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

import Supp_Prep_for_S1516_hadisd_common as C

OUT = C.FIG_DIR + 'Supplementary_FigS15_HadISD_ERA5_validation.png'

# ── Setting of the figure ─────────────────────────────────────────────────────────────────
FIGSIZE   = (13, 5.6)
AXIS_LIM  = 2.5          # c27：°C per decade

FS_LABEL  = 14
FS_TICK   = 12
FS_LEGEND = 12
FS_STAT   = 12
FS_PANEL  = 17

C_POINT   = '#555555'
C_ONE2ONE = '#000000'
C_OLS     = '#D55E00'     
C_HADISD  = '#0072B2'
C_ERA5    = '#E69F00'

C.apply_white_style(plt)
plt.rcParams['font.size'] = FS_TICK


def main(refresh=False):
    had = C.hadisd_annual_trends(refresh=refresh)
    era = C.era5_annual_trends()
    cmp_ = had.merge(era, on='station_id', how='inner')

    #  °C per decade
    x = cmp_['hadisd_trend_yr'].values * 10
    y = cmp_['era5_trend_yr'].values * 10
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)

    rho, p_rho = stats.spearmanr(x, y)              # c28
    r_pearson, _ = stats.pearsonr(x, y)             # for compare
    slope, intercept, _, _, _ = stats.linregress(x, y)
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    bias = float(np.mean(y - x))

    print(f'\n=== HadISD vs ERA5（°C per decade）===')
    print(f'  N            = {n}')
    print(f'  Spearman rho = {rho:.3f}  (p = {p_rho:.2e})')
    print(f'  Pearson r    = {r_pearson:.3f}  (R2 = {r_pearson**2:.3f})  <- old caption')
    print(f'  RMSE         = {rmse:.2f} °C/decade')
    print(f'  Bias         = {bias:+.2f} °C/decade  (ERA5 - HadISD)')

    outside = int((np.abs(x) > AXIS_LIM).sum() + (np.abs(y) > AXIS_LIM).sum())
    print(f'  The number of coordinate values clipped exceeding ±{AXIS_LIM} is {outside}'
          f'({2*n} numbers of coordinate in total)')

    # ── Plot: c29, Delete panel c, lefting two lattices ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.patch.set_facecolor('white')

    # (a) HadISD vs ERA5 Scatter
    ax = axes[0]
    ax.axhline(0, color='#cccccc', lw=0.8, zorder=1)
    ax.axvline(0, color='#cccccc', lw=0.8, zorder=1)
    ax.scatter(x, y, s=22, c=C_POINT, alpha=0.5, linewidths=0, zorder=3)

    grid = np.linspace(-AXIS_LIM, AXIS_LIM, 300)
    ax.plot(grid, grid, ls='--', color=C_ONE2ONE, lw=1.3, alpha=0.7,
            zorder=4, label='1:1 line')
    ax.plot(grid, slope * grid + intercept, color=C_OLS, lw=2.0,
            zorder=5, label='OLS fit')

    ax.set_xlim(-AXIS_LIM, AXIS_LIM)          # c27
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)          # c27
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('HadISD air-temperature trend (°C per decade)', fontsize=FS_LABEL)
    ax.set_ylabel('ERA5 air-temperature trend (°C per decade)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.04, 0.96,
            f'N = {n}\n'
            r'Spearman $\rho$ = ' + f'{rho:.3f}\n'
            f'RMSE = {rmse:.2f} °C/dec\n'
            f'Bias = {bias:+.2f} °C/dec',
            transform=ax.transAxes, fontsize=FS_STAT, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#dddddd', pad=4))
    ax.legend(fontsize=FS_LEGEND, loc='lower right', framealpha=0.95,
              facecolor='white', edgecolor='#cccccc')
    ax.text(-0.14, 1.02, '(a)', transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='left')

    # (b) Trend distribution of the two data sets
    ax = axes[1]
    for vals, label, color in [(x, 'HadISD', C_HADISD), (y, 'ERA5', C_ERA5)]:
        kde = gaussian_kde(vals, bw_method=0.3)
        g = np.linspace(vals.min(), vals.max(), 400)
        ax.plot(g, kde(g), color=color, lw=2.4, label=label, zorder=3)
        ax.fill_between(g, kde(g), alpha=0.16, color=color, zorder=2)
    ax.axvline(0, color='#cccccc', lw=0.9, ls='--', zorder=1)
    ax.set_xlabel('Air-temperature trend (°C per decade)', fontsize=FS_LABEL)
    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylabel('Density', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.95,
              facecolor='white', edgecolor='#cccccc')
    ax.text(-0.12, 1.02, '(b)', transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='left')

    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT, dpi=C.DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='recalculate HadISD TA trend cache')
    main(**vars(ap.parse_args()))
