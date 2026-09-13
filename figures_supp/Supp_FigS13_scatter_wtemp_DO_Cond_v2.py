# ============================================================
# Fig3_scatter_wtemp_DO_Cond_v2
# Cell 0 — set environment
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/'

# ★ Changed Version：3 / 5 / 8 / 10
MIN_YEARS = 5
FIG_DIR  = '/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary_correct/'
os.makedirs(FIG_DIR, exist_ok=True)

SEASONS = {
    'Boreal Summer': {'NH': [6,7,8], 'SH': [12,1,2]},
    'Boreal Winter': {'NH': [12,1,2], 'SH': [6,7,8]},
}

# Light theme
BG_COLOR    = 'white'
AX_COLOR    = 'white'
GRID_COLOR  = '#dddddd'
TEXT_COLOR  = 'black'

# Season-based colors (Summer=orange, Winter=blue)
COLOR_SEASON = {
    'Boreal Summer': {'sig': '#e8824a', 'ns': '#e8824a', 'glacier_edge': '#dddddd'},
    'Boreal Winter': {'sig': '#4ab3e8', 'ns': '#4ab3e8', 'glacier_edge': '#dddddd'},
}

# Keep for KDE
COLOR_DEEP  = {'River station': '#e8824a', 'Lake station': '#d6604d'}
COLOR_LIGHT = {'River station': '#4ab3e8', 'Lake station': '#f4a582'}
MARKER      = {'River station': 'o',       'Lake station': 'o'}
SIZE        = {'River station': 40,        'Lake station': 50}

X_LIM      = 2.5
Y_LIM_DO   = 0.65
Y_LIM_COND = 80.0

# ============================================================
# Cell 1 — Read + calculate season present value
# ============================================================
df_wt   = pd.read_csv(DATA_DIR + f'global_wtemp_monthly_trends_v{MIN_YEARS}y.csv')
df_wt   = df_wt[df_wt['water_type'] == 'River station'].copy()  # only keep River station
df_do   = pd.read_csv(DATA_DIR + 'global_do_monthly_trends_v1.csv')
df_do   = df_do[df_do['water_type'] == 'River station'].copy()
df_cond = pd.read_csv(DATA_DIR + 'global_cond_monthly_trends_v1.csv')
df_cond = df_cond[df_cond['water_type'] == 'River station'].copy()
df_glacier = pd.read_csv(DATA_DIR + 'station_near_glacier_v3_100km.csv')
GLACIER_IDS = set(df_glacier['station_id'])

df_wt['wtemp_trend_yr']  = df_wt['trend_per_decade']   / 10
df_do['do_trend_yr']     = df_do['trend_per_decade']    / 10
df_cond['cond_trend_yr'] = df_cond['trend_per_decade']  / 10

def seasonal_mean(df, val_col, sig_col, station_col,
                  lat_col, lon_col, wt_col, months_nh, months_sh):
    records = []
    for sid, grp in df.groupby(station_col):
        lat    = grp[lat_col].iloc[0]
        months = months_nh if lat > 0 else months_sh
        sub    = grp[grp['month'].isin(months)].dropna(subset=[val_col])
        if len(sub) == 0:
            continue
        val_mean = sub[val_col].mean()
        sig = 'Yes' if (sub[sig_col] == 'Yes').any() else 'No'
        records.append({
            'station_id': sid,
            'latitude':   lat,
            'longitude':  grp[lon_col].iloc[0],
            'water_type': grp[wt_col].iloc[0],
            val_col:      val_mean,
            'significant': sig,
            'n_months':   len(sub),
        })
    return pd.DataFrame(records)

# season present value(shared)
wt_seasonal = {}
for season, mdef in SEASONS.items():
    wt_s = seasonal_mean(
        df_wt, 'wtemp_trend_yr', 'significant',
        'station_id', 'latitude', 'longitude', 'water_type',
        mdef['NH'], mdef['SH']
    )
    wt_seasonal[season] = wt_s

# DO season present value
dfs_do = {}
for season, mdef in SEASONS.items():
    do_s = seasonal_mean(
        df_do, 'do_trend_yr', 'significant',
        'station_id', 'latitude', 'longitude', 'water_type',
        mdef['NH'], mdef['SH']
    )
    merged = wt_seasonal[season][
        ['station_id','latitude','longitude','water_type','wtemp_trend_yr']
    ].merge(
        do_s[['station_id','do_trend_yr','significant']],
        on='station_id', how='inner'
    )
    # p95 clip
    merged = merged[merged['do_trend_yr'].abs() <= Y_LIM_DO]
    dfs_do[season] = merged
    print(f'DO {season}: {len(merged)} 站')

# Cond season present value
dfs_cond = {}
for season, mdef in SEASONS.items():
    cond_s = seasonal_mean(
        df_cond, 'cond_trend_yr', 'significant',
        'station_id', 'latitude', 'longitude', 'water_type',
        mdef['NH'], mdef['SH']
    )
    merged = wt_seasonal[season][
        ['station_id','latitude','longitude','water_type','wtemp_trend_yr']
    ].merge(
        cond_s[['station_id','cond_trend_yr','significant']],
        on='station_id', how='inner'
    )
    # p95 clip
    merged = merged[merged['cond_trend_yr'].abs() <= Y_LIM_COND]
    dfs_cond[season] = merged
    print(f'Cond {season}: {len(merged)} station')


# ============================================================
# Cell 2 — Figure 3 (dark theme, season colors)
# ============================================================

def plot_panel(fig, gs_main, gs_kde_y,
               df, y_col, y_label, y_lim, season_label, panel_label):

    ax_main  = fig.add_subplot(gs_main)
    ax_kde_y = fig.add_subplot(gs_kde_y)

    # Dark background
    ax_main.set_facecolor(AX_COLOR)
    ax_kde_y.set_facecolor(AX_COLOR)

    # Zero lines
    ax_main.axhline(0, color='#999999', linewidth=1.2, zorder=1)
    ax_main.axvline(0, color='#999999', linewidth=1.2, zorder=1)

    sc = COLOR_SEASON[season_label]
    col_sig  = sc['sig']
    col_ns   = sc['ns']
    edge_glc = sc['glacier_edge']

    sub = df[df['water_type'] == 'River station']
    sub_near = sub[ sub['station_id'].isin(GLACIER_IDS)]
    sub_far  = sub[~sub['station_id'].isin(GLACIER_IDS)]

    # Far stations: significant (solid) and not significant (hollow)
    sig_far = sub_far[sub_far['significant'] == 'Yes']
    ns_far  = sub_far[sub_far['significant'] != 'Yes']
    if len(sig_far):
        ax_main.scatter(sig_far['wtemp_trend_yr'], sig_far[y_col],
                        c=col_sig, s=SIZE['River station'],
                        marker=MARKER['River station'],
                        edgecolors='none', alpha=0.85, zorder=3)
    if len(ns_far):
        ax_main.scatter(ns_far['wtemp_trend_yr'], ns_far[y_col],
                        facecolors='none', edgecolors=col_ns,
                        s=SIZE['River station'],
                        marker=MARKER['River station'],
                        linewidths=0.6, alpha=0.6, zorder=2)

    # Near-glacier stations: white-grey thick border
    sig_near = sub_near[sub_near['significant'] == 'Yes']
    ns_near  = sub_near[sub_near['significant'] != 'Yes']
    if len(sig_near):
        ax_main.scatter(sig_near['wtemp_trend_yr'], sig_near[y_col],
                        c=col_sig, s=SIZE['River station'],
                        marker=MARKER['River station'],
                        edgecolors=edge_glc, linewidths=1.8,
                        alpha=0.95, zorder=5)
    if len(ns_near):
        ax_main.scatter(ns_near['wtemp_trend_yr'], ns_near[y_col],
                        facecolors='none', edgecolors=edge_glc,
                        s=SIZE['River station'],
                        marker=MARKER['River station'],
                        linewidths=1.8, alpha=0.8, zorder=4)

    # OLS regression line
    x_reg = df['wtemp_trend_yr'].values
    y_reg = df[y_col].values
    mask  = np.isfinite(x_reg) & np.isfinite(y_reg)
    if mask.sum() >= 3:
        slope, intercept, r, p, _ = stats.linregress(x_reg[mask], y_reg[mask])
        x_line = np.linspace(-X_LIM, X_LIM, 300)
        ax_main.plot(x_line, slope * x_line + intercept,
                     color='#27ae60', linewidth=1.5,
                     linestyle='-', alpha=0.9, zorder=6)
        p_str = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
        ax_main.text(0.97, 0.97,
                     f'slope={slope:.4f}\n$R^2$={r**2:.2f}\n{p_str}',
                     transform=ax_main.transAxes, fontsize=8,
                     ha='right', va='top', color=TEXT_COLOR,
                     bbox=dict(facecolor='white', alpha=0.7,
                               edgecolor='none', pad=2))

    # Theory lines
    x_theory = np.linspace(-X_LIM, X_LIM, 300)
    if y_col == 'do_trend_yr':
        DO_SLOPE = -0.38
        ax_main.plot(x_theory, DO_SLOPE * x_theory,
                     color='#e67e22', linewidth=1.5,
                     linestyle='--', alpha=0.9, zorder=7)
        ax_main.text(-1.5, DO_SLOPE * -1.5, "Henry's Law",
                     color='#e67e22', fontsize=8,
                     ha='left', va='bottom', zorder=8)
    elif y_col == 'cond_trend_yr':
        HAYASHI_SLOPE = 0.02 * 200
        ax_main.plot(x_theory, HAYASHI_SLOPE * x_theory,
                     color='#e67e22', linewidth=1.5,
                     linestyle='--', alpha=0.9, zorder=7)
        ax_main.text(X_LIM * 0.98, HAYASHI_SLOPE * X_LIM * 0.98,
                     'Hayashi (2004)', color='#e67e22', fontsize=8,
                     ha='right', va='bottom', zorder=8)

    ax_main.set_xlim(-X_LIM, X_LIM)
    ax_main.set_ylim(-y_lim, y_lim)
    ax_main.set_xlabel('Water Temperature Trend (°C/year)', fontsize=10, color=TEXT_COLOR)
    ax_main.set_ylabel(y_label, fontsize=10, color=TEXT_COLOR)
    ax_main.tick_params(colors=TEXT_COLOR)
    for sp in ax_main.spines.values():
        sp.set_color('#aaaaaa')
        sp.set_linewidth(1.2)
    ax_main.grid(color=GRID_COLOR, linestyle='--', linewidth=0.4, alpha=0.6)
    ax_main.set_title(f'({panel_label}) {season_label}  (n={len(df)})',
                      fontsize=11, fontweight='bold', color=TEXT_COLOR)
    ax_main.text(0.98, 0.02, f'n = {len(df)}',
                 transform=ax_main.transAxes, fontsize=8,
                 ha='right', va='bottom', color=TEXT_COLOR,
                 bbox=dict(facecolor='white', alpha=0.7,
                           edgecolor='none', pad=2))

    # Left Y-KDE
    kde_color = col_sig
    sub_kde = df[df['water_type'] == 'River station'].dropna(subset=[y_col])
    if len(sub_kde) >= 3:
        y_vals = sub_kde[y_col].values
        y_vals = y_vals[np.isfinite(y_vals)]
        y_vals = y_vals[(y_vals >= -y_lim) & (y_vals <= y_lim)]
        if len(y_vals) >= 3:
            kde    = gaussian_kde(y_vals, bw_method=0.3)
            y_grid = np.linspace(-y_lim, y_lim, 300)
            ax_kde_y.plot(kde(y_grid), y_grid,
                          color=kde_color, linewidth=1.2)
            ax_kde_y.fill_betweenx(y_grid, kde(y_grid),
                                    alpha=0.2, color=kde_color)

    ax_kde_y.set_ylim(-y_lim, y_lim)
    ax_kde_y.set_xticklabels([])
    ax_kde_y.set_yticklabels([])
    ax_kde_y.tick_params(left=False, bottom=False)
    ax_kde_y.axhline(0, color='#999999', linewidth=0.5)
    ax_kde_y.invert_xaxis()
    for sp in ax_kde_y.spines.values():
        sp.set_visible(False)

    return ax_main


PANELS = [
    ('DO',   'Boreal Summer', 'do_trend_yr',   Y_LIM_DO,
     'DO Trend (mg/L/year)', 'a', dfs_do),
    ('DO',   'Boreal Winter', 'do_trend_yr',   Y_LIM_DO,
     'DO Trend (mg/L/year)', 'b', dfs_do),
    ('Cond', 'Boreal Summer', 'cond_trend_yr', Y_LIM_COND,
     'Specific conductance Trend (µS/cm/year)', 'c', dfs_cond),
    ('Cond', 'Boreal Winter', 'cond_trend_yr', Y_LIM_COND,
     'Specific conductance Trend (µS/cm/year)', 'd', dfs_cond),
]

plt.rcParams.update({
    'text.color':      TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color':     TEXT_COLOR,
    'ytick.color':     TEXT_COLOR,
})

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor(BG_COLOR)

gs_outer = gridspec.GridSpec(
    2, 2, figure=fig,
    wspace=0.15, hspace=0.30,
    left=0.10, right=0.97,
    top=0.92, bottom=0.12
)

for idx, (var, season, y_col, y_lim, y_label, plabel, dfs_var) in         enumerate(PANELS):
    row = idx // 2
    col = idx  % 2
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs_outer[row, col],
        width_ratios=[0.1, 0.9],
        wspace=0.20
    )
    plot_panel(
        fig,
        gs_main  = gs_inner[0, 1],
        gs_kde_y = gs_inner[0, 0],
        df       = dfs_var[season],
        y_col    = y_col,
        y_label  = y_label,
        y_lim    = y_lim,
        season_label = season,
        panel_label  = plabel
    )

# Legend
legend_handles = [
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor='#e8824a', markeredgecolor='none',
           markersize=8, label='River (significant)'),
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor='none', markeredgecolor='#e8824a',
           markersize=8, label='River (not significant)'),
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor='#e8824a', markeredgecolor='#dddddd',
           markeredgewidth=1.8, markersize=8, label='River (near glacier)'),
    Line2D([0],[0], color='#27ae60', linewidth=1.5,
           linestyle='-', label='OLS regression'),
    Line2D([0],[0], color='#e67e22', linewidth=1.5,
           linestyle='--', label="Henry's Law / Hayashi (2004)"),
]
fig.legend(handles=legend_handles, loc='lower center',
           fontsize=9, framealpha=0.9,
           facecolor='white', edgecolor='#aaaaaa',
           labelcolor='black',
           ncol=5, bbox_to_anchor=(0.52, 0.01))

#fig.suptitle(
#    'Water Temperature Trend vs DO / Specific conductance\n'
#    'Solid = significant (p<0.05)  |  thick border = near glacier  |  '
#    '1990–2020  |  Global',
#    fontsize=13, fontweight='bold', color=TEXT_COLOR
#)

out = FIG_DIR + f'FigS13_scatter_wtemp_DO_Cond_v2_{MIN_YEARS}y.png'
plt.savefig(out, dpi=200, bbox_inches='tight',
            facecolor='white')
plt.show()
print(f'Saved: {out}')

