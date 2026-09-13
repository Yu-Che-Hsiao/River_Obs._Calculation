#!/usr/bin/env python3
# ============================================================
# Supp_wtemp_vs_hadisd.py
#   Supplementary Fig. S6: River water-temperature trend against
#   HadISD in-situ air-temperature trend
#
#   changes from cell 7 in old_bash/Fig_HadISD_validation_v2.ipynb.
#
#   Output -> figures/Supplementary_correct/Supplementary_FigS16_wtemp_vs_HadISD.png
# ============================================================
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

import Supp_Prep_for_S1516_hadisd_common as C

OUT = C.FIG_DIR + 'Supplementary_FigS16_wtemp_vs_HadISD.png'

# ── Figure Setting ─────────────────────────────────────────────────────────────────
FIGSIZE = (24, 8)

FS_LABEL  = 16
FS_TICK   = 14
FS_TITLE  = 18
FS_LEGEND = 14
FS_STAT   = 14
FS_PANEL  = 20

COLOR_DEEP  = {'River station': '#2166ac', 'Lake station': '#d6604d'}
COLOR_LIGHT = {'River station': '#92c5de', 'Lake station': '#f4a582'}
SIZE        = {'River station': 55,        'Lake station': 65}

C_ONE2ONE = '#f1c40f'
C_OLS     = '#6a3d9a'     # Originally green #27ae60, changed to purple to avoid the red and green combination.

X_LIM = 2.5
Y_MIN = -0.05
Y_MAX = 0.10

C.apply_white_style(plt)
plt.rcParams['font.size'] = FS_TICK


def seasonal_mean_wt(df, months_nh, months_sh):
    """For each station, the average trend of each month in the quarter is taken, and it is marked whether any month is significant."""
    rec = []
    for sid, g in df.groupby('station_id'):
        lat = g['latitude'].iloc[0]
        months = months_nh if lat > 0 else months_sh
        sub = g[g['month'].isin(months)].dropna(subset=['wtemp_trend_yr'])
        if len(sub) == 0:
            continue
        rec.append({
            'station_id': sid,
            'latitude': lat,
            'longitude': g['longitude'].iloc[0],
            'water_type': g['water_type'].iloc[0],
            'wtemp_trend_yr': sub['wtemp_trend_yr'].mean(),
            'significant': 'Yes' if (sub['significant'] == 'Yes').any() else 'No',
        })
    return pd.DataFrame(rec)


def plot_panel(fig, gs_main, gs_kde_y, gs_kde_x, df, season_label, panel_label):
    ax  = fig.add_subplot(gs_main)
    aky = fig.add_subplot(gs_kde_y)
    akx = fig.add_subplot(gs_kde_x)

    ax.axhline(0, color='#999999', lw=1.5, zorder=1)
    ax.axvline(0, color='#999999', lw=1.5, zorder=1)
    ax.plot([-X_LIM, X_LIM], [-X_LIM, X_LIM], color=C_ONE2ONE,
            lw=1.8, ls='--', alpha=0.9, zorder=1)

    for wt in ['River station', 'Lake station']:
        sub = df[df['water_type'] == wt]
        if len(sub) == 0:
            continue
        near = sub[sub['station_id'].isin(GLACIER_IDS)]
        far  = sub[~sub['station_id'].isin(GLACIER_IDS)]
        for pts, lw in [(far, 0.5), (near, 2.2)]:
            if len(pts) == 0:
                continue
            sig = pts[pts['significant'] == 'Yes']
            if len(sig):
                ax.scatter(sig['wtemp_trend_yr'], sig['airtemp_trend_yr'],
                           c=COLOR_DEEP[wt], s=SIZE[wt], marker='o',
                           edgecolors='#444444', linewidths=lw,
                           alpha=0.85, zorder=3)
            ns = pts[pts['significant'] != 'Yes']
            if len(ns):
                ax.scatter(ns['wtemp_trend_yr'], ns['airtemp_trend_yr'],
                           facecolors='none', edgecolors=COLOR_LIGHT[wt],
                           s=SIZE[wt], marker='o', linewidths=lw,
                           alpha=0.75, zorder=2)

    xr_ = df['wtemp_trend_yr'].values
    yr_ = df['airtemp_trend_yr'].values
    m = np.isfinite(xr_) & np.isfinite(yr_)
    if m.sum() >= 3:
        sl, ic, r, p, _ = stats.linregress(xr_[m], yr_[m])
        xl = np.linspace(np.nanmin(xr_[m]), np.nanmax(xr_[m]), 300)
        ax.plot(xl, sl * xl + ic, color=C_OLS, lw=2.2, zorder=4)
        pstr = 'p < 0.001' if p < 0.001 else f'p = {p:.2f}'
        ax.text(0.975, 0.965,
                f'slope = {sl:.4f}\n' + r'$R^2$ = ' + f'{r**2:.2f}\n{pstr}',
                transform=ax.transAxes, fontsize=FS_STAT, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.85,
                          edgecolor='#dddddd', pad=4))

    ax.set_xlim(-X_LIM, X_LIM)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel('')
    ax.set_ylabel('HadISD air-temperature trend (°C per year)', fontsize=FS_LABEL)
    ax.set_title(f'{season_label}  (n = {len(df)})', fontsize=FS_TITLE, pad=8)
    ax.tick_params(axis='both', which='major', width=1.4, length=5,
                   labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_linewidth(1.4)
    ax.text(-0.02, 1.06, panel_label, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='right')

    # broader KDE
    for wt in ['River station', 'Lake station']:
        v = df[df['water_type'] == wt]['wtemp_trend_yr'].values
        v = v[np.isfinite(v) & (np.abs(v) <= X_LIM)]
        if len(v) >= 3:
            k = gaussian_kde(v, bw_method=0.3)
            g = np.linspace(-X_LIM, X_LIM, 300)
            akx.plot(g, k(g), color=COLOR_DEEP[wt], lw=1.6)
            akx.fill_between(g, k(g), alpha=0.15, color=COLOR_DEEP[wt])
    akx.set_xlim(-X_LIM, X_LIM)
    akx.axvline(0, color='#cccccc', lw=0.6)
    akx.set_xlabel('River water-temperature trend (°C per year)', fontsize=FS_LABEL)

    for wt in ['River station', 'Lake station']:
        v = df[df['water_type'] == wt]['airtemp_trend_yr'].values
        v = v[np.isfinite(v) & (v >= Y_MIN) & (v <= Y_MAX)]
        if len(v) >= 3:
            k = gaussian_kde(v, bw_method=0.3)
            g = np.linspace(Y_MIN, Y_MAX, 300)
            aky.plot(k(g), g, color=COLOR_DEEP[wt], lw=1.6)
            aky.fill_betweenx(g, k(g), alpha=0.15, color=COLOR_DEEP[wt])
    aky.set_ylim(Y_MIN, Y_MAX)
    aky.axhline(0, color='#cccccc', lw=0.6)
    aky.invert_xaxis()

    for a in (akx, aky):
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.tick_params(left=False, bottom=False)
        for sp in a.spines.values():
            sp.set_visible(False)


def main(refresh=False):
    global GLACIER_IDS

    wt = pd.read_csv(C.DATA_DIR + 'global_wtemp_monthly_trends_v5y.csv')
    wt['wtemp_trend_yr'] = wt['trend_per_decade'] / 10
    GLACIER_IDS = set(pd.read_csv(
        C.DATA_DIR + 'station_near_glacier_v2.6.csv')['station_id'])

    seasonal = C.hadisd_seasonal_trends(refresh=refresh)

    combined = {}
    for season, mdef in C.SEASONS.items():
        w = seasonal_mean_wt(wt, mdef['NH'], mdef['SH'])
        h = (seasonal[seasonal['season'] == season][['station_id', 'hadisd_trend_yr']]
             .rename(columns={'hadisd_trend_yr': 'airtemp_trend_yr'}))
        combined[season] = w.merge(h, on='station_id', how='inner')
        print(f'{season}: {len(combined[season])} station comparable')

    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor('white')
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.16,
                                 left=0.06, right=0.98, top=0.90, bottom=0.20)

    for i, (season, plab) in enumerate(zip(['Boreal Summer', 'Boreal Winter'],
                                           ['(a)', '(b)'])):
        gsi = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_outer[i],
            width_ratios=[0.1, 0.9], height_ratios=[0.9, 0.1],
            wspace=0.18, hspace=0.18)
        plot_panel(fig, gsi[0, 1], gsi[0, 0], gsi[1, 1],
                   combined[season], season, plab)

    handles = [
        Line2D([0], [0], marker='o', color='w', markersize=11,
               markerfacecolor=COLOR_DEEP['River station'],
               markeredgecolor='#444444', markeredgewidth=0.5,
               label='River, significant'),
        Line2D([0], [0], marker='o', color='w', markersize=11,
               markerfacecolor='none',
               markeredgecolor=COLOR_LIGHT['River station'],
               label='River, not significant'),
        Line2D([0], [0], marker='o', color='w', markersize=11,
               markerfacecolor=COLOR_DEEP['River station'],
               markeredgecolor='#444444', markeredgewidth=2.2,
               label='River, near glacier'),
        Line2D([0], [0], color=C_ONE2ONE, lw=1.8, ls='--', label='1:1 line'),
        Line2D([0], [0], marker='o', color='w', markersize=11,
               markerfacecolor=COLOR_DEEP['Lake station'],
               markeredgecolor='#444444', markeredgewidth=0.5,
               label='Lake, significant'),
        Line2D([0], [0], marker='o', color='w', markersize=11,
               markerfacecolor='none',
               markeredgecolor=COLOR_LIGHT['Lake station'],
               label='Lake, not significant'),
        Line2D([0], [0], marker='o', color='w', markersize=11,
               markerfacecolor=COLOR_DEEP['Lake station'],
               markeredgecolor='#444444', markeredgewidth=2.2,
               label='Lake, near glacier'),
        Line2D([0], [0], color=C_OLS, lw=2.2, label='OLS fit'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=FS_LEGEND, framealpha=1.0, facecolor='white',
               edgecolor='#cccccc', bbox_to_anchor=(0.52, 0.005),
               columnspacing=2.0, handletextpad=0.8)

    fig.savefig(OUT, dpi=C.DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved: {OUT}')

    print('\n--- caption numbers needed to check ---')
    for season in ['Boreal Summer', 'Boreal Winter']:
        d = combined[season]
        m = np.isfinite(d['wtemp_trend_yr']) & np.isfinite(d['airtemp_trend_yr'])
        sl, _, r, p, _ = stats.linregress(d.loc[m, 'wtemp_trend_yr'],
                                          d.loc[m, 'airtemp_trend_yr'])
        print(f'  {season}: n = {len(d)}, slope = {sl:.4f}, '
              f'R2 = {r**2:.2f}, p = {p:.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='Recalculate HadISD seasonal trends cache')
    main(**vars(ap.parse_args()))
