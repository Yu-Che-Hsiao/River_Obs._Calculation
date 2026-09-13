#!/usr/bin/env python3
# ============================================================
# Supp_minyears_coverage.py
#   Supplementary Fig. S4: Station counts and spatial coverage as a
#   function of the record-length threshold
#
#   Rewritten from old_bash/Fig4_plot_minyears_summary_v2.py
#
#     c17  background changed to white -> global rcParams + explicit facecolor
#     c20  what is MIN_YEARS -> the variable name no longer appears in the figure, replaced by reader-friendly wording
#
#   Output -> figures/Supplementary_correct/Supplementary_FigS18_minyears_coverage.png
# ============================================================
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# ── Path  ─────────────────────────────────────────────────────────────────────
BASE_DIR = '/work/home/H.Jason421/water_temp_for_publish/'
DATA_DIR = BASE_DIR + 'data/'
FIG_DIR  = BASE_DIR + 'figures/Supplementary_correct/'
OUT      = FIG_DIR + 'Supplementary_FigS18_minyears_coverage.png'

os.makedirs(FIG_DIR, exist_ok=True)

# ── Analysis parameters ─────────────────────────────────────────────────────────────────
THRESHOLDS = [3, 5, 8, 10]          # minimum valid years required per calendar month
CONTINENTS = ['N. America', 'S. America', 'Europe', 'Africa', 'Asia', 'Oceania']

# ── Figure settings ─────────────────────────────────────────────────────────────────
DPI      = 300
FIGSIZE  = (16, 6)

FS_LABEL  = 14   # axis titles
FS_TICK   = 12   # ticks
FS_LEGEND = 11   # legend
FS_ANNOT  = 10   # numbers on points/bars
FS_PANEL  = 16   # (a)(b)

# c18: no red-green pairing. Instead:
#   GEMStat = black (consistent with Fig 2, observations as the lead)
#   ERA5    = blues   HadISD = oranges   DynWat = magenta
C_GEMSTAT      = '#000000'
C_ERA5_OBS     = '#0072B2'
C_ERA5_FULL    = '#56B4E9'
C_HADISD_FULL  = '#E69F00'
C_HADISD_MATCH = '#B35A00'
C_DYNWAT       = '#CC79A7'

CONT_COLORS = {
    'N. America': '#0072B2',
    'S. America': '#56B4E9',
    'Europe':     '#E69F00',
    'Africa':     '#D55E00',
    'Asia':       '#CC79A7',
    'Oceania':    '#4D4D4D',
}

# c17: globally force a white background
plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size']         = FS_TICK


# ── Continent classification ─────────────────────────────────────────────────────────────────
def assign_continent(lat, lon):
    if lat > 15 and lon < -30:
        return 'N. America'
    elif -60 < lat <= 15 and lon < -30:
        return 'S. America'
    elif lat > 35 and -30 <= lon < 60:
        return 'Europe'
    elif -40 < lat <= 35 and -20 <= lon < 55:
        return 'Africa'
    elif 55 <= lon < 180 and lat > -15:
        return 'Asia'
    elif lat <= -15 or (lat < 35 and lon >= 110):
        return 'Oceania'
    return 'Other'


def read_csv(name):
    path = DATA_DIR + name
    if not os.path.exists(path):
        raise FileNotFoundError(f'Data file not found:{path}')
    return pd.read_csv(path)


def text_color_on(hex_color):
    """Number inside a bar is black or white, decided by background brightness."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return '#000000' if lum > 0.55 else '#ffffff'


# ============================================================
# 1. Read data
# ============================================================
print('Read station counts for each threshold ...', flush=True)

gemstat_counts    = {}
continent_counts  = {ny: {} for ny in THRESHOLDS}
era5_obs_counts   = {}
hadisd_full_counts = {}
hadisd_counts     = {}
dynwat_obs_counts = {}

for ny in THRESHOLDS:
    df = read_csv(f'global_wtemp_monthly_trends_v{ny}y.csv')
    df = df[df['water_type'] == 'River station'].drop_duplicates('station_id')
    gemstat_counts[ny] = len(df)
    df['continent'] = df.apply(
        lambda r: assign_continent(r['latitude'], r['longitude']), axis=1)
    for cont in CONTINENTS:
        continent_counts[ny][cont] = int((df['continent'] == cont).sum())

    era5_obs_counts[ny]    = read_csv(f'era5_obs_airtemp_trends_v2_{ny}y.csv')['station_id'].nunique()
    hadisd_full_counts[ny] = read_csv(f'fig2_hadisd_full_seasonal_{ny}y.csv')['station_id'].nunique()
    hadisd_counts[ny]      = read_csv(f'hadisd_airtemp_trends_v2_{ny}y.csv')['station_id'].nunique()
    dynwat_obs_counts[ny]  = read_csv(f'dynwat_monthly_trends_v2_{ny}y.csv')['station_id'].nunique()

# ERA5 has a complete record at every station, so the station count does not change with the threshold
era5_full_n = len(read_csv('global_airtemp_monthly_trends_v1.csv')
                  .pipe(lambda d: d[d['water_type'] == 'River station'])
                  .drop_duplicates('station_id'))

print('\nStation counts per threshold:')
for ny in THRESHOLDS:
    print(f'  {ny:>2}y  GEMStat={gemstat_counts[ny]:>4}  ERA5obs={era5_obs_counts[ny]:>4}  '
          f'HadISDfull={hadisd_full_counts[ny]:>4}  HadISD={hadisd_counts[ny]:>4}  '
          f'DynWat={dynwat_obs_counts[ny]:>4}')
print(f'  ERA5 full (fixed)= {era5_full_n}')


# ============================================================
# 2. Plotting
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
fig.patch.set_facecolor('white')

XLABEL = 'Minimum years of record required per calendar month'

# ── (a) Station count vs threshold ───────────────────────────────────────────────────────
series = [
    ('GEMStat',      gemstat_counts,     C_GEMSTAT,      'o', '-',  2.6),
    ('ERA5',         era5_obs_counts,    C_ERA5_OBS,     's', '-',  2.0),
    ('HadISD, all',  hadisd_full_counts, C_HADISD_FULL,  '^', '-',  2.0),
    ('HadISD, matched to river stations',
                     hadisd_counts,      C_HADISD_MATCH, 'v', '--', 2.0),
    ('DynWat',       dynwat_obs_counts,  C_DYNWAT,       'D', '-',  2.0),
]

ax1.axhline(era5_full_n, color=C_ERA5_FULL, linestyle=':', linewidth=1.8,
            zorder=2, label=f'ERA5, complete record (n = {era5_full_n}, fixed)')

for label, counts, color, marker, ls, lw in series:
    y = [counts[ny] for ny in THRESHOLDS]
    ax1.plot(THRESHOLDS, y, color=color, marker=marker, linestyle=ls,
             linewidth=lw, markersize=8, label=label, zorder=3)
    for xi, yi in zip(THRESHOLDS, y):
        ax1.annotate(f'{yi}', (xi, yi), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=FS_ANNOT, color=color)

ymax = max([max(c.values()) for _, c, *_ in series] + [era5_full_n])
ax1.set_xlabel(XLABEL, fontsize=FS_LABEL)
ax1.set_ylabel('Number of stations', fontsize=FS_LABEL)
ax1.set_xticks(THRESHOLDS)
ax1.set_xticklabels([str(ny) for ny in THRESHOLDS], fontsize=FS_TICK)
ax1.set_xlim(THRESHOLDS[0] - 1, THRESHOLDS[-1] + 1)
ax1.set_ylim(0, ymax * 1.18)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(50))
ax1.tick_params(axis='y', labelsize=FS_TICK)
ax1.grid(axis='y', color='#e6e6e6', linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(fontsize=FS_LEGEND, loc='upper right',
           framealpha=0.95, facecolor='white', edgecolor='#cccccc')
ax1.text(-0.06, 1.03, '(a)', transform=ax1.transAxes,
         fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='left')

# ── (b) GEMStat station count broken down by continent ─────────────────────────────────────────────
x2      = np.arange(len(THRESHOLDS))
width2  = 0.5
bottoms = np.zeros(len(THRESHOLDS))

for cont in CONTINENTS:
    vals = np.array([continent_counts[ny].get(cont, 0) for ny in THRESHOLDS])
    ax2.bar(x2, vals, width2, bottom=bottoms, label=cont,
            color=CONT_COLORS[cont], zorder=3)
    tcol = text_color_on(CONT_COLORS[cont])
    for xi, (val, bot) in enumerate(zip(vals, bottoms)):
        if val > 5:
            ax2.text(x2[xi], bot + val / 2, str(val), ha='center', va='center',
                     fontsize=FS_ANNOT, color=tcol, fontweight='bold')
    bottoms += vals

for xi, ny in enumerate(THRESHOLDS):
    ax2.text(x2[xi], bottoms[xi] + max(bottoms) * 0.015,
             f'n = {gemstat_counts[ny]}', ha='center', va='bottom',
             fontsize=FS_ANNOT + 1, fontweight='bold', color='#333333')

ax2.set_xlabel(XLABEL, fontsize=FS_LABEL)
ax2.set_ylabel('Number of GEMStat stations', fontsize=FS_LABEL)
ax2.set_xticks(x2)
ax2.set_xticklabels([str(ny) for ny in THRESHOLDS], fontsize=FS_TICK)
ax2.set_ylim(0, max(bottoms) * 1.15)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(50))
ax2.tick_params(axis='y', labelsize=FS_TICK)
ax2.grid(axis='y', color='#e6e6e6', linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.95,
           facecolor='white', edgecolor='#cccccc',
           title='Continent', title_fontsize=FS_LEGEND)
ax2.text(-0.06, 1.03, '(b)', transform=ax2.transAxes,
         fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='left')

plt.tight_layout()
fig.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f'\nSaved: {OUT}')

# ── caption 對照用 ───────────────────────────────────────────────────────────
print('\n--- Numbers to verify against the caption ---')
print(f'  Fixed station count for the ERA5 complete record : {era5_full_n}')
print(f'  GEMStat station count at the 3-year threshold    : {gemstat_counts[3]}')
print(f'  Whether the two are equal                        : {era5_full_n == gemstat_counts[3]}')
