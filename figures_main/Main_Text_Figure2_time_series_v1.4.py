#!/usr/bin/env python3
# ============================================================
# Main_Text_Figure2_time_series_v1.3.py - standalone time-series figure (a–f)
# v1.3:Anomaly/Trend labels changed from once per summer/winter → once centered per block; side padding; larger font
# v1.1 changed:
#   - Summer/Winter moved from the left sidebar into the top-left of each summer/winter subplot (SW_ROT controls rotation)
#   - The freed left space goes into the block gap (GAP_W widened); each subplot has ticks on both sides (left anomaly, right trend)
#   - Right-side trend ticks made more visible (TREND_TICK_FS / TREND_TICK_LEN / thicker axis line)
#   - legend moved to the bottom and its frame removed
# Split out from FIG2_combine_figure_v2.5.py, with v2.7 changes applied:
#   #2 legend's ALL / SUB changed to subscripts
#   #3 trend changed to a separate right-hand y-axis, range -1..+1 (different unit from the left anomaly)
#   #4 block frame thinned (lw 1.5), top/bottom lines aligned exactly to y=+2 / y=-2, tick marks at ±2 removed
#   #8 red/blue use the violin base colors (RED_BASE / BLUE_BASE)
# Width = 10 in (same as combined v2.5); height is set here; left/right margins aligned with the other two figures.
# ============================================================
import matplotlib
matplotlib.use('Agg')
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import linregress
warnings.filterwarnings('ignore')

DATA_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/'
FIG_DIR  = '/work/home/H.Jason421/water_temp_for_publish/figures/final/'
os.makedirs(FIG_DIR, exist_ok=True)
MIN_YEARS = 5
SCALE_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/scale_trends_v1/'

# ── Shared base colors (from the violin red/blue; #8) ────────────────
RED_BASE  = mcolors.to_hex(plt.cm.RdBu_r(0.80))   # #d6604d  summer red / warm
BLUE_BASE = mcolors.to_hex(plt.cm.RdBu_r(0.15))   # #327cb7  winter blue / cool
RED_DARK  = mcolors.to_hex(plt.cm.RdBu_r(0.93))   # dark red (ERA5 TS, to distinguish from TA)
AQUA      = '#17BECF'   # DynWat aqua
PINK      = '#F4A259'   # HadISD light orange
BLACK     = '#111111'   # GEMStat

C = {
    'GEMStat TR':   BLACK,
    'ERA5 TA SUB':  RED_BASE, 'ERA5 TA ALL':  RED_BASE,
    'ERA5 TS SUB':  RED_DARK, 'ERA5 TS ALL':  RED_DARK,
    'DynWat TR SUB': AQUA,    'DynWat TR ALL': AQUA,
    'HadISD TA SUB': PINK,    'HadISD TA ALL': PINK,
}
LW_THIN = 0.8; LW_THICK = 1.7; LW_GEM = 3.2   # GEMStat black line = focus, extra thick
TS_COL_OBS = {'gemstat':'GEMStat TR','era5_t2m':'ERA5 TA SUB','era5_skt':'ERA5 TS SUB',
              'dynwat':'DynWat TR SUB','hadisd':'HadISD TA SUB'}
TS_COL_FULL = {'era5_t2m_full':'ERA5 TA ALL','era5_skt_full':'ERA5 TS ALL',
               'dynwat_full':'DynWat TR ALL','hadisd_full':'HadISD TA ALL'}

def fmt_sub(g):
    return g.replace(' ALL', '$_{ALL}$').replace(' SUB', '$_{SUB}$')

# ════════════════════════════ Data ════════════════════════════
print('Reading regional_series_v1 (new pipeline: per-station anomaly, per-station hemisphere)...', flush=True)
df_ts = pd.read_csv(SCALE_DIR + 'regional_series_v1.csv')

# ════════════════════════════ Plotting functions ════════════════════════
TREND_YLIM = 0.5   # #3 trend right-axis range ±1

# ── v1.1 adjustable layout constants ──
SW_ROT         = 0     # Summer/Winter corner label rotation (0=horizontal; use 90 for vertical)
GAP_W          = 0.7   # block gap width (was 0.55), widened to fit ticks on both sides
TREND_TICK_FS  = 9     # right-side trend tick font size (was 6.5), made more visible
TREND_TICK_LEN = 3.5   # right-side trend tick length (was 2), made more visible
# ── v1.3: Anomaly / Trend labels changed to once centered per block ──
LBL_X_OFF_L    = 0.045  # distance from Anomaly label to the block's left frame (increase if it overlaps the tick numbers)
LBL_X_OFF_R    = 0.047  # distance from Trend label to the block's right frame
LBL_FS_L       = 13     # Anomaly label font size (must be smaller than the panel name, 16)
LBL_FS_R       = 12     # Trend label font size (must be smaller than the panel name, 16)

def draw_ts(ax, ax_trend, region, season, bg, txt,
            show_xlabels=True, show_yticks=True, show_ytitle=True,
            show_trend_ticks=True, show_trend_title=False):
    ax.set_facecolor(bg)
    ax.axhline(0, color=txt, lw=0.4, alpha=0.5, zorder=0)
    ax.set_yticks([-1,-0.5,0,0.5,1])
    ax.grid(axis='y', color=txt, alpha=0.07, lw=0.3, zorder=0)
    df_r = df_ts[(df_ts['region']==region) & (df_ts['season']==season)]
    trend_obs = {}; trend_full = {}

    for col, lbl in TS_COL_OBS.items():
        if col not in df_r.columns: continue
        sub = df_r.dropna(subset=[col])
        if sub.empty: continue
        ax.plot(sub['year'], sub[col], color=C[lbl], lw=LW_THIN, ls='-', alpha=0.3, zorder=2)
        roll = sub.set_index('year')[col].rolling(5, center=True, min_periods=3).mean()
        _is_gem = (lbl == 'GEMStat TR')
        ax.plot(roll.index, roll.values, color=C[lbl],
                lw=(LW_GEM if _is_gem else LW_THICK), ls='-',
                alpha=(1.0 if _is_gem else 0.9),
                zorder=(6 if _is_gem else 3),
                solid_capstyle='round')
        x = sub['year'].values.astype(float); y = sub[col].values; valid = np.isfinite(y)
        if valid.sum() >= 5:
            sl,_,_,_,_ = linregress(x[valid], y[valid]); trend_obs[lbl] = sl * 10

    for col, lbl in TS_COL_FULL.items():
        if col not in df_r.columns: continue
        sub = df_r.dropna(subset=[col])
        if sub.empty: continue
        ax.plot(sub['year'], sub[col], color=C[lbl], lw=LW_THIN, ls='--', alpha=0.2, zorder=2)
        roll = sub.set_index('year')[col].rolling(5, center=True, min_periods=3).mean()
        ax.plot(roll.index, roll.values, color=C[lbl], lw=LW_THICK, ls='--', alpha=0.65, zorder=3)
        x = sub['year'].values.astype(float); y = sub[col].values; valid = np.isfinite(y)
        if valid.sum() >= 5:
            sl,_,_,_,_ = linregress(x[valid], y[valid]); trend_full[lbl] = sl * 10

    ax.set_xlim(1988, 2021)
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_xticklabels(["1990","2000","2010","2020"] if show_xlabels else [], fontsize=10, color=txt)
    ax.set_ylim(-1.0, 1.0)
    if show_yticks:
        ax.set_yticklabels(['-1','','0','','1'], fontsize=12, color=txt)
    else:
        ax.set_yticklabels([])
    if show_ytitle:
        ax.set_ylabel('Anomaly (°C)', color=txt, fontsize=12, labelpad=2)
    # #4 remove tick marks: disable tick lines (length=0), keep labels only
#    ax.tick_params(colors=txt, labelsize=12, length=1, pad=2)
    ax.tick_params(axis='y', colors=txt, labelsize=12, length=1, pad=3)
    ax.tick_params(axis='x', colors=txt, labelsize=12, length=1, pad=5)   # ← move year labels downward
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ── trend: obs filled (x=0), full hollow (x=1); #3 uses a separate right axis ±1 ──
    ax_trend.set_facecolor(bg)
    ax_trend.set_xlim(-0.5, 1.5)
    ax_trend.set_ylim(-TREND_YLIM, TREND_YLIM)
    ax_trend.axhline(0, color=txt, lw=0.5, alpha=0.5, zorder=0)
    ax_trend.set_xticks([0, 1]); ax_trend.set_xticklabels([])
    for tval in trend_obs.values(): pass
    for lbl, tval in trend_obs.items():
        ax_trend.scatter(0, np.clip(tval,-TREND_YLIM,TREND_YLIM), color=C[lbl], s=22, zorder=5,
                         marker="o", edgecolors="white", linewidths=0.3)
    for lbl, tval in trend_full.items():
        ax_trend.scatter(1, np.clip(tval,-TREND_YLIM,TREND_YLIM), color=C[lbl], s=22, zorder=5,
                         marker="o", facecolors="none", edgecolors=C[lbl], linewidths=0.8)
    # Hide the other spines; the right axis shows ticks and title only on the rightmost block
    for side in ['top','left','bottom']:
        ax_trend.spines[side].set_visible(False)
    if show_trend_ticks:
        ax_trend.spines['right'].set_visible(True)
        ax_trend.spines['right'].set_color(txt); ax_trend.spines['right'].set_linewidth(0.9)
        ax_trend.yaxis.set_label_position('right'); ax_trend.yaxis.tick_right()
        ax_trend.set_yticks([-0.5, 0, 0.5])
        ax_trend.set_yticklabels(['-0.5','0','0.5'], fontsize=TREND_TICK_FS, color=txt)
        ax_trend.tick_params(axis='y', which='both', right=True, left=False,
                             labelright=True, length=TREND_TICK_LEN, width=0.9, pad=1, colors=txt)
        if show_trend_title:
            ax_trend.set_ylabel('Trend (°C/dec)', color=txt, fontsize=11, rotation=270, labelpad=8)
    else:
        ax_trend.spines['right'].set_visible(False)
        ax_trend.set_yticks([]); ax_trend.tick_params(left=False, right=False, bottom=False)

# ════════════════════════════ Layout ════════════════════════════
def make_fig():
    bg = 'white'; txt = '#111111'
    TS_SUPERROWS = [
        [('Global','a'),('Asia','b')],
        [('Europe','c'),('South/SE Asia','d')],
        [('Latin America','e'),('North America','f')],
    ]
    fig = plt.figure(figsize=(9, 11), facecolor=bg)
    # 4 rows: tsA, ts_gap, tsB, legend (bottom) — legend moved to the bottom, frame removed
    gs = gridspec.GridSpec(6, 1, height_ratios=[1.0, 0.001, 1.0, 0.001, 1.0, 0.10],
                           hspace=0.20, left=0.065, right=0.935, top=0.97, bottom=0.01)
    leg_handles = None
    frames = []
    for sr_i, regions in enumerate(TS_SUPERROWS):
        gs_sr = gridspec.GridSpecFromSubplotSpec(
            3, 5, subplot_spec=gs[sr_i*2], hspace=0.0, wspace=0.04,
            height_ratios=[1, 0.16, 1],
            width_ratios=[4,0.7,GAP_W, 4,0.7])
        for r_i,(region,letter) in enumerate(regions):
            c_ts = 3*r_i; c_tr = 3*r_i+1
            sum_ts = fig.add_subplot(gs_sr[0,c_ts]); sum_tr = fig.add_subplot(gs_sr[0,c_tr])
            win_ts = fig.add_subplot(gs_sr[2,c_ts]); win_tr = fig.add_subplot(gs_sr[2,c_tr])
            leftmost = (r_i==0); rightmost = (r_i==1)
            draw_ts(sum_ts, sum_tr, region, 'Boreal Summer', bg, txt,
                    show_xlabels=False, show_yticks=True, show_ytitle=False,
                    show_trend_ticks=True, show_trend_title=False)
            draw_ts(win_ts, win_tr, region, 'Boreal Winter', bg, txt,
                    show_xlabels=True, show_yticks=True, show_ytitle=False,
                    show_trend_ticks=True, show_trend_title=False)
            frames.append({'axes':[sum_ts,sum_tr,win_ts,win_tr],
                           'rightmost':rightmost,
                           'leftmost':leftmost,
                           'label':f'{letter}  {region}'})
            if sr_i==0 and r_i==0:
                _ls = {'-':'-','--':(0,(5,2.5))}
                leg_handles = [Line2D([0],[0],color=C[k],lw=1.7,ls=_ls[ss],label=fmt_sub(k))
                               for k,ss in [('ERA5 TA SUB','-'),('ERA5 TA ALL','--'),('ERA5 TS SUB','-'),('ERA5 TS ALL','--'),                                            
                                            ('HadISD TA SUB','-'),('HadISD TA ALL','--'),('DynWat TR SUB','-'),('DynWat TR ALL','--'),('GEMStat TR','-')]]

    # ── Block frame: #4 thin frame (lw 1.5), top/bottom aligned exactly to y=±2 (no y-direction pad) ──
    for fr in frames:
        a_sum_ts,a_sum_tr,a_win_ts,a_win_tr = fr['axes']
        poss = [a.get_position() for a in fr['axes']]
        x0 = min(p.x0 for p in poss); x1 = max(p.x1 for p in poss)
        y0 = min(p.y0 for p in poss); y1 = max(p.y1 for p in poss)   # align to y=±2
        fig.add_artist(mpatches.Rectangle((x0,y0), x1-x0, y1-y0, transform=fig.transFigure,
                       fill=False, edgecolor='black', lw=1.5, zorder=20, clip_on=False))
        fig.text(x0+0.002, y1+0.006, fr['label'], ha='left', va='bottom',
                 fontsize=16, fontweight='bold', color=txt)
        # ── v1.3: Anomaly / Trend labels once centered per block ──
        ymid = (y0 + y1) / 2
        if fr['leftmost']:
            fig.text(x0 - LBL_X_OFF_L, ymid, 'Anomaly (°C)', rotation=90,
                     ha='center', va='center', fontsize=LBL_FS_L, color=txt)
        if fr['rightmost']:
            fig.text(x1 + LBL_X_OFF_R, ymid, 'Trend (°C/dec)', rotation=270,
                     ha='center', va='center', fontsize=LBL_FS_R, color=txt)
        y_gap_top = a_sum_ts.get_position().y0
        y_gap_bot = a_win_ts.get_position().y1
        for yy in (y_gap_top, y_gap_bot):
            fig.add_artist(Line2D([x0,x1],[yy,yy], transform=fig.transFigure,
                           color='black', lw=1.5, zorder=20, clip_on=False))
        if True:   # v1.4 comment 25: label Summer/Winter on every panel (originally only a, c, e)
            a_sum_ts.text(0.035, 0.90, 'Summer', transform=a_sum_ts.transAxes,
                          rotation=SW_ROT, ha='left', va='top',
                          fontsize=11, fontweight='bold', color=txt, zorder=25)
            a_win_ts.text(0.035, 0.90, 'Winter', transform=a_win_ts.transAxes,
                          rotation=SW_ROT, ha='left', va='top',
                          fontsize=11, fontweight='bold', color=txt, zorder=25)
    # ── legend (#2 subscripts) ──
    ax_leg = fig.add_subplot(gs[5]); ax_leg.set_facecolor(bg); ax_leg.axis('off')
    if leg_handles:
        ax_leg.legend(handles=leg_handles, loc='center', ncol=5,
                      frameon=False, labelcolor=txt,
                      fontsize=10, handlelength=2.8, columnspacing=0.6,
                      borderpad=0.3, handletextpad=0.4)
    return fig

print('Plotting (time series)...', flush=True)
fw = make_fig()
out = FIG_DIR + f'Main_Text_Figure2_time_series_v1.4_{MIN_YEARS}y_white.png'
fw.savefig(out, dpi=800, facecolor='white')   # fixed 10-in width, no cropping → same width as the other two figures
print(f'Saved: {out}'); plt.close(fw)
print('Finish!')
