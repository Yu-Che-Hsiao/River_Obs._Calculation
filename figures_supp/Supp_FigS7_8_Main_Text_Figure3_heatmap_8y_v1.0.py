#!/usr/bin/env python3
# ============================================================
# Supp_FigS7_8_Main_Text_Figure3_heatmap_8y_v1.0.py
#   Supplementary version: record-length threshold robustness check (8y / 10y)
#
#   Usage (one script runs both thresholds)
#     python Supp_FigS7_8_Main_Text_Figure3_heatmap_8y_v1.0.py --min_years 8
#     python Supp_FigS7_8_Main_Text_Figure3_heatmap_8y_v1.0.py --min_years 10
#
#   Prerequisites
#   This figure reads {dataset}_basin_{version}.csv produced by calc_scale_trends_v1.py.
#   That script's output filenames do not include the threshold, so running --min_years 10 directly overwrites the 5y results.
#   Apply the FNAME_SUFFIX change first (5y keeps the original filename, the rest get _8y / _10y), then run
#     python calc_scale_trends_v1.py --min_years 8  --from_cache
#     python calc_scale_trends_v1.py --min_years 10 --from_cache
#
#   Unlike Fig 2, the basin trends in this figure do change with the threshold —— sparse basins are dropped
#   at higher thresholds for having too few years, so grey cells (no data) become more numerous.
#
#   Original v1.2 notes follow
# ------------------------------------------------------------
# Main_Text_Figure3_heatmap_v1.2.py   —  basin scale standalone file (formerly Fig 2g, renamed to Fig 3 after splitting)
# v1.2 changes: title drops "g" and keeps only basin scale; colorbar moved down (CB_YSHIFT) to avoid covering the bottom row of basin labels;
#           colorbar narrowed (CB_WFRAC) and thinned (CB_HFRAC).
# Split out from FIG2_combine_figure_v2.5.py, with v2.7 changes applied:
#   #5 outer frame thinned (lw 1.5); summer/winter thick dividers thinned in sync; removed bold Summer/Winter on the far right
#   #6 colorbar switched to the colormaps package's nrl_sirkes (or rdbu)
#   #8 when CB_STYLE='violin', the colorbar ends use the violin blue/red (#327cb7 / #d6604d)
# width = 10 in (same as combined v2.5); left/right margins aligned with the other two figures.
# Requires: pip install colormaps
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
warnings.filterwarnings('ignore')

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--min_years', type=int, default=8,
                 help='record-length threshold, 8 or 10; for 5 use the main text v1.2 directly')
_args = _ap.parse_args()
MIN_YEARS = _args.min_years

DATA_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/'
FIG_DIR  = '/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary_correct/'
os.makedirs(FIG_DIR, exist_ok=True)
SCALE_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/scale_trends_v1/'   # basin/continental/global trends all live here

# After adding FNAME_SUFFIX to calc_scale_trends_v1.py, 5y keeps the original filename and the rest get a year suffix
SUFFIX = '' if MIN_YEARS == 5 else f'_{MIN_YEARS}y'

# ── Shared base colors for the whole figure (taken from the violin red/blue; #8)────────────────
RED_BASE  = mcolors.to_hex(plt.cm.RdBu_r(0.80))   # #d6604d
BLUE_BASE = mcolors.to_hex(plt.cm.RdBu_r(0.15))   # #327cb7

# ── #6/#8 colorbar palette: 'nrl_sirkes'(default) | 'rdbu' | 'violin' ──
CB_STYLE = 'rdbu'
def build_hcmap(style):
    if style == 'nrl_sirkes':
        import colormaps as cmaps; return cmaps.nrl_sirkes          # blue(negative)→white→red(positive)
    if style == 'rdbu':
        import colormaps as cmaps; return cmaps.rdbu.reversed()     # reversed to blue(negative)→red(positive)
    if style == 'violin':                                           # ends use violin blue/red
        return LinearSegmentedColormap.from_list('violin_bwr', [BLUE_BASE, '#ffffff', RED_BASE])
    return plt.cm.RdBu_r
def drop_white(cmap, gap=0.16, n=256):
    # Cut out the white band at the center of the palette and recombine; a larger gap means the center is less white and blue/red are more distinct (gap=0 means no white removal)
    left  = np.linspace(0.0, 0.5-gap/2, n//2)
    right = np.linspace(0.5+gap/2, 1.0, n-n//2)
    return LinearSegmentedColormap.from_list('noWhite', cmap(np.concatenate([left, right])))
# ── Adjustable colorbar layout constants (v1.2: moved down, narrowed, to avoid covering the bottom row of basin labels) ──
CB_WFRAC  = 0.50   # colorbar length = fraction of that row's width (smaller is narrower; was 2/3)
CB_YSHIFT = 0.0    # whitespace is handled mainly by the spacer row, no extra shift here
CB_HFRAC  = 0.80   # colorbar thickness = fraction of the gs row height (smaller is thinner)

HVMAX = 1.5
# v1.1: the color cells and the colorbar share a "discrete" color scale, with boundaries at the equal-division points of -1.5..1.5,
#       so the -1.5/-1.0/-0.5/0/0.5/1.0/1.5 tick lines fall exactly on the color transitions (and transition lines appear only at ticks).
N_LEVELS = 6      # 6 segments (0.5 each): every tick is exactly one transition line. For a finer gradient use 12 (0.25 each, ticks land on every other 
HBOUNDS    = np.linspace(-HVMAX, HVMAX, N_LEVELS + 1)
HCMAP_FULL = drop_white(build_hcmap(CB_STYLE), gap=0.16)                     # continuous parent palette
HCMAP      = mcolors.ListedColormap(HCMAP_FULL(np.linspace(0, 1, N_LEVELS))) # discrete: shared by color cells and colorbar
HCMAP.set_over(HCMAP_FULL(1.0)); HCMAP.set_under(HCMAP_FULL(0.0))
HNORM      = mcolors.BoundaryNorm(HBOUNDS, N_LEVELS, clip=True)              # colors the cells (out-of-range clipped to the two ends)

HROWS_ORDER = ['ERA5 TA ALL','ERA5 TA SUB','HadISD TA ALL','HadISD TA SUB',
               'ERA5 TS ALL','ERA5 TS SUB','DynWat TR ALL','DynWat TR SUB','GEMStat TR']
HROWS_FILES = {
    'ERA5 TA ALL':   (f'era5_t2m_basin_all{SUFFIX}.csv','trend_per_decade'),
    'ERA5 TA SUB':   (f'era5_t2m_basin_sub{SUFFIX}.csv','trend_per_decade'),
    'HadISD TA ALL': (f'hadisd_basin_all{SUFFIX}.csv','trend_per_decade'),
    'HadISD TA SUB': (f'hadisd_basin_sub{SUFFIX}.csv','trend_per_decade'),
    'ERA5 TS ALL':   (f'era5_skt_basin_all{SUFFIX}.csv','trend_per_decade'),
    'ERA5 TS SUB':   (f'era5_skt_basin_sub{SUFFIX}.csv','trend_per_decade'),
    'DynWat TR ALL': (f'dynwat_basin_all{SUFFIX}.csv','trend_per_decade'),
    'DynWat TR SUB': (f'dynwat_basin_sub{SUFFIX}.csv','trend_per_decade'),
    'GEMStat TR':    (f'gemstat_basin{SUFFIX}.csv','trend_per_decade'),
}
REGION_ORDER = ['Asia','Europe','South/SE Asia','Latin America','North America']
SEASONS = {'Boreal Summer':{'NH':[6,7,8],'SH':[12,1,2]},
           'Boreal Winter':{'NH':[12,1,2],'SH':[6,7,8]}}

def assign_region(lat, lon):
    if lon>180: lon-=360
    if   lat>20  and -170<lon<-50:   return 'North America'
    elif lat<=20 and -120<lon<-30:   return 'Latin America'
    elif -10<lat<80 and -15<lon<60:  return 'Europe'
    elif lat>35  and  60<lon<180:    return 'Asia'
    elif lat<=35 and  60<lon<180:    return 'South/SE Asia'
    else:                            return 'Other'

def fmt_sub(g):
    return g.replace(' ALL','$_{ALL}$').replace(' SUB','$_{SUB}$')

# ════════════════════════════ Data (DATA-BLOCK)════════════════
print('Read metadata...', flush=True)
xl   = pd.ExcelFile('/work5/H.Jason421/GEMStat_new/GEMS-Water_data_request.xls')
meta = pd.read_excel(xl, sheet_name='Station_Metadata')
meta = meta.rename(columns={'GEMS Station Number':'station_id','Water Type':'water_type',
                              'Main Basin':'basin','Latitude':'lat','Longitude':'lon'})
meta = meta[meta['water_type']=='River station'].dropna(subset=['lat','lon']).drop_duplicates('station_id')
BLAT = meta.dropna(subset=['basin']).groupby('basin')['lat'].median().to_dict()
BLON = meta.dropna(subset=['basin']).groupby('basin')['lon'].median().to_dict()
BABBR = dict(zip(*[pd.read_csv(DATA_DIR+'basin_abbrev_v1.csv')[c] for c in ['basin','abbrev']]))

print('Preparing heatmap data...', flush=True)
def load_basin_seasonal(fname, tcol):
    df = pd.read_csv(SCALE_DIR+fname); rows=[]
    for basin,grp in df.groupby('basin'):
        lat=BLAT.get(basin,45)
        for sea,hemi in SEASONS.items():
            mo=hemi['NH'] if lat>=0 else hemi['SH']
            sub=grp[grp['month'].isin(mo)]
            if sub.empty: continue
            rows.append({'basin':basin,'season':sea,'trend':sub[tcol].mean(),
                         'sig':bool((sub['p_value']<0.05).any()) if 'p_value' in sub.columns else False})
    return pd.DataFrame(rows)

print(f'MIN_YEARS = {MIN_YEARS}(reading suffix "{SUFFIX or "無"}")', flush=True)
_missing=[f for f,_ in HROWS_FILES.values() if not os.path.exists(SCALE_DIR+f)]
if _missing:
    raise FileNotFoundError(
        'The following basin trend files are missing:\n  ' + '\n  '.join(_missing) +
        f'\n\nApply the FNAME_SUFFIX change first, then run:\n'
        f'  python calc_scale_trends_v1.py --min_years {MIN_YEARS} --from_cache')

bdata={}
for lbl,(fname,tcol) in HROWS_FILES.items():
    bdata[lbl]=load_basin_seasonal(fname,tcol)
    print(f'  {lbl:<15} {len(bdata[lbl]["basin"].unique()) if len(bdata[lbl]) else 0:>4} basins', flush=True)

gem_basins=sorted(bdata['GEMStat TR']['basin'].unique())
brl=[(b,assign_region(BLAT.get(b,0),BLON.get(b,0)),BLAT.get(b,0)) for b in gem_basins]
rom={r:i for i,r in enumerate(REGION_ORDER+['Other'])}
brl.sort(key=lambda x:(rom.get(x[1],99),-x[2]))
BASIN_ORDER=[b for b,r,l in brl]
BASIN_REGION={b:r for b,r,l in brl}
REGION_COLS={reg:[i for i,b in enumerate(BASIN_ORDER) if BASIN_REGION.get(b)==reg] for reg in REGION_ORDER}

def build_hm(season):
    n=len(HROWS_ORDER); m=len(BASIN_ORDER)
    HM=np.full((n,m),np.nan); SIG=np.zeros((n,m),dtype=bool)
    bidx={b:i for i,b in enumerate(BASIN_ORDER)}
    for ri,lbl in enumerate(HROWS_ORDER):
        df_ds=bdata.get(lbl,pd.DataFrame())
        if df_ds.empty: continue
        sub=df_ds[df_ds['season']==season]
        for _,r in sub.iterrows():
            if r['basin'] in bidx:
                HM[ri,bidx[r['basin']]]=r['trend']; SIG[ri,bidx[r['basin']]]=r.get('sig',False)
    return HM,SIG
HM_SUM,SIG_SUM=build_hm('Boreal Summer')
HM_WIN,SIG_WIN=build_hm('Boreal Winter')
# ════════════════════════════ /DATA-BLOCK ════════════════════

def draw_heatmap_region(ax,HM_s,SIG_s,HM_w,SIG_w,col_indices,title,txt_c,bg_c,
                        show_ylabel=True,show_sw_label=False,show_xlabel=True):
    ax.set_facecolor(bg_c)
    n_r=len(HROWS_ORDER); n_c=len(col_indices); n_total=n_r*2+1
    sub_s=HM_s[:,col_indices]; sig_s=SIG_s[:,col_indices]
    sub_w=HM_w[:,col_indices]; sig_w=SIG_w[:,col_indices]
    def draw_half(hm,sig,row_offset):
        for ri in range(n_r):
            for ci in range(n_c):
                y=ri+row_offset
                fc='#bbbbbb' if np.isnan(hm[ri,ci]) else HCMAP(HNORM(hm[ri,ci]))
                ax.add_patch(mpatches.Rectangle((ci-0.5,y-0.5),1,1,facecolor=fc,edgecolor='none',
                             zorder=1 if np.isnan(hm[ri,ci]) else 2))
                if sig[ri,ci] and not np.isnan(hm[ri,ci]):
                    ax.plot([ci-0.35,ci+0.35],[y-0.35,y+0.35],color='black',lw=0.5,zorder=4)
                    ax.plot([ci-0.35,ci+0.35],[y+0.35,y-0.35],color='black',lw=0.5,zorder=4)
    draw_half(sub_s,sig_s,0); draw_half(sub_w,sig_w,n_r+1)
    # #5 summer/winter thick divider 2.5→1.5
    ax.axhline(n_r-0.5,color=txt_c,lw=1.5,alpha=1.0,zorder=5)
    ax.axhline(n_r+0.5,color=txt_c,lw=1.5,alpha=1.0,zorder=5)
    for y_off in [0,n_r+1]:
        for y in [1.5,3.5,5.5,7.5]:
            ax.axhline(y+y_off,color=txt_c,lw=0.4,alpha=0.5,zorder=3)
    ax.set_xlim(-0.5,n_c-0.5); ax.set_ylim(n_total-0.5,-0.5)
    basins_sub=[BASIN_ORDER[i] for i in col_indices]
    ax.set_xticks(range(n_c))
    if show_xlabel:
        ax.set_xticklabels([BABBR.get(b,b[:3]) for b in basins_sub],fontsize=5,rotation=90,color=txt_c)
    else:
        ax.set_xticklabels([])
    ax.tick_params(colors=txt_c,length=1,pad=0)
    if show_ylabel:
        ax.set_yticks(list(range(n_r))+list(range(n_r+1,n_total)))
        ax.set_yticklabels([fmt_sub(x) for x in HROWS_ORDER]*2,fontsize=7,color=txt_c)
    else:
        ax.set_yticks([])
#    if show_sw_label:   # # #5 remove bold → fontweight='normal'
#        ax.text(n_c-0.5+0.3,n_r/2-0.5,'Summer',ha='left',va='center',
#                fontsize=8,color=txt_c,fontweight='normal',rotation=270,transform=ax.transData)
#        ax.text(n_c-0.5+0.3,n_r+1+n_r/2-0.5,'Winter',ha='left',va='center',
#                fontsize=8,color=txt_c,fontweight='normal',rotation=270,transform=ax.transData)
#    ax.set_title(title,color=txt_c,fontsize=9,pad=2,loc='left',fontweight='normal')
    if show_sw_label:                                                                          # Build a blended coordinate system: X uses the figure's absolute fraction (0~1), 
                                                                                               # Y uses the heatmap's data coordinates
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.figure.transFigure, ax.transData)     # Set X to 0.985. Because the figure's right boundary (right) is 0.96,
        x_pos = 0.985                                                                          # lus the previous figure's Trend Y-axis ticks and labelpad, it lands roughly between 0.98 ~ 0.99.
        ax.text(x_pos, n_r/2 - 0.5, 'Summer', ha='left', va='center',
                fontsize=8, color=txt_c, fontweight='normal', rotation=270, transform=trans)
        ax.text(x_pos, n_r + 1 + n_r/2 - 0.5, 'Winter', ha='left', va='center',
                fontsize=8, color=txt_c, fontweight='normal', rotation=270, transform=trans)
    ax.set_title(title,color=txt_c,fontsize=9,pad=2,loc='left',fontweight='normal')
        
    # #5 outer frame 2.5→1.5
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(1.5)

def make_fig():
    bg='white'; ax_bg='white'; txt='#111111'
    fig=plt.figure(figsize=(10,6.4),facecolor=bg)
    # 5 rows: hm_top, hm_gap, hm_bot, label_spacer (for the bottom row of basin labels), colorbar
    #   —— left/right margins aligned with the other two figures; the spacer keeps the colorbar from covering the basin labels
    gs=gridspec.GridSpec(5,1,height_ratios=[1.0,0.01,1.0,0.07,0.05],
                         hspace=0.16,left=0.08,right=0.98,top=0.94,bottom=0.075)

    # Top row:Asia | Europe
    asia_cols=REGION_COLS['Asia']; europe_cols=REGION_COLS['Europe']
    na=len(asia_cols); ne=len(europe_cols)
    gs_t=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[0],wspace=0.02,width_ratios=[na,1,ne])
    ax_ha=fig.add_subplot(gs_t[0]); ax_hg=fig.add_subplot(gs_t[2])
    ax_g0=fig.add_subplot(gs_t[1]); ax_g0.set_facecolor(bg); ax_g0.axis('off')
    draw_heatmap_region(ax_ha,HM_SUM[:,asia_cols],SIG_SUM[:,asia_cols],HM_WIN[:,asia_cols],SIG_WIN[:,asia_cols],
                        list(range(na)),f'Asia ({na})',txt,ax_bg,show_ylabel=True)
    draw_heatmap_region(ax_hg,HM_SUM[:,europe_cols],SIG_SUM[:,europe_cols],HM_WIN[:,europe_cols],SIG_WIN[:,europe_cols],
                        list(range(ne)),f'Europe ({ne})',txt,ax_bg,show_ylabel=False,show_sw_label=True)

    # Middle whitespace
    ax_gap=fig.add_subplot(gs[1]); ax_gap.set_facecolor(bg); ax_gap.axis('off')

    # Bottom row:South/SE Asia | Latin America | North America
    sse_cols=REGION_COLS['South/SE Asia']; la_cols=REGION_COLS['Latin America']; na_cols=REGION_COLS['North America']
    nsse=len(sse_cols); nla=len(la_cols); nna=len(na_cols)
    gs_b=gridspec.GridSpecFromSubplotSpec(1,5,subplot_spec=gs[2],wspace=0.02,width_ratios=[nsse,1,nla,1,nna])
    ax_hsse=fig.add_subplot(gs_b[0]); ax_hla=fig.add_subplot(gs_b[2]); ax_hna=fig.add_subplot(gs_b[4])
    for gi in [1,3]:
        ag=fig.add_subplot(gs_b[gi]); ag.set_facecolor(bg); ag.axis('off')
    draw_heatmap_region(ax_hsse,HM_SUM[:,sse_cols],SIG_SUM[:,sse_cols],HM_WIN[:,sse_cols],SIG_WIN[:,sse_cols],
                        list(range(nsse)),f'South/SE Asia ({nsse})',txt,ax_bg,show_ylabel=True)
    draw_heatmap_region(ax_hla,HM_SUM[:,la_cols],SIG_SUM[:,la_cols],HM_WIN[:,la_cols],SIG_WIN[:,la_cols],
                        list(range(nla)),f'Latin America ({nla})',txt,ax_bg,show_ylabel=False)
    draw_heatmap_region(ax_hna,HM_SUM[:,na_cols],SIG_SUM[:,na_cols],HM_WIN[:,na_cols],SIG_WIN[:,na_cols],
                        list(range(nna)),f'North America ({nna})',txt,ax_bg,show_ylabel=False,show_sw_label=True)

    # colorbar: length shrunk to 2/3 of the full row width, centered; shares the same discrete color scale as the cells → transition lines only at ticks
    _cbpos = gs[4].get_position(fig)
    _full_w = _cbpos.x1 - _cbpos.x0
    _cb_w  = _full_w * CB_WFRAC
    _cb_x0 = _cbpos.x0 + (_full_w - _cb_w) / 2.0
    _cb_h  = _cbpos.height * CB_HFRAC
    ax_cb = fig.add_axes([_cb_x0, _cbpos.y0 - CB_YSHIFT, _cb_w, _cb_h]); ax_cb.set_facecolor(bg)
    cb_norm = mcolors.BoundaryNorm(HBOUNDS, N_LEVELS)
    cb = fig.colorbar(ScalarMappable(cmap=HCMAP, norm=cb_norm), cax=ax_cb,
                      orientation='horizontal', extend='both')
    cb.set_ticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    cb.set_label('Trend (°C/decade)',color=txt,fontsize=7,labelpad=2)
    cb.ax.xaxis.set_tick_params(color=txt,labelsize=6)
    plt.setp(cb.ax.xaxis.get_ticklabels(),color=txt,fontsize=6)
    cb.outline.set_edgecolor(txt); cb.outline.set_linewidth(0.5)

    # section label
#    p_hm=ax_ha.get_position()
#    fig.text(0.012, p_hm.y1+0.025, 'basin scale',
#             color=txt, fontsize=10, fontweight='bold', ha='left', va='bottom')
    return fig

print('Plotting (basin heatmap)...', flush=True)
fw=make_fig()
out=FIG_DIR+f'Supplementary_FigS7_or_S8__heatmap_{MIN_YEARS}y.png'
fw.savefig(out,dpi=800,facecolor='white')   # fixed 10in width, no cropping → equal width with the other two figures
print(f'Saved: {out}'); plt.close(fw)
print('Finish!')
