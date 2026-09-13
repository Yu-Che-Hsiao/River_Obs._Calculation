#!/usr/bin/env python3
# ============================================================
# Main_Text_Figure4_station_violin_v1.2.py - standalone station-scale figure (violin + box) (formerly Fig 3)
#   v1.3: (1) significance changed from "any month p<0.05" to "significant months within a season share the same sign", split into sig_warm / sig_cool
#         (2) added station-level DynWat TR ALL (reads ss_dynwat_full_trends_5y.csv; falls back to the global grid if missing)
#         (3) outputs station_scale_stats.csv: n, median, IQR, and significant warming/cooling fractions per group and season
#   v1.2: keep only the section label 'a' (removed the "station scale" text).
#   v1.1: legend frame removed.
# Split out from FIG2_combine_figure_v2.5.py, with v2.7 changes applied:
#   #7 removed the "← Air / Water →" text and the central dashed divider
#   #8 keep the current best red/blue pairing and fix it as the shared base (RED_BASE / BLUE_BASE)
# Width = 10 in (same as combined v2.5); left/right margins aligned with the other two figures.
# ============================================================
import matplotlib
matplotlib.use('Agg')
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
warnings.filterwarnings('ignore')

DATA_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/'
FIG_DIR  = '/work/home/H.Jason421/water_temp_for_publish/figures/final/'
os.makedirs(FIG_DIR, exist_ok=True)
MIN_YEARS = 5

# ── Shared base colors (violin red/blue; #8)──────────────────────
RED_BASE  = mcolors.to_hex(plt.cm.RdBu_r(0.80))   # #d6604d  summer
BLUE_BASE = mcolors.to_hex(plt.cm.RdBu_r(0.15))   # #327cb7  winter
SC = {'Boreal Summer':RED_BASE,'Boreal Winter':BLUE_BASE}

GROUP_ORDER = ['ERA5 TA ALL','ERA5 TA SUB','HadISD TA ALL','HadISD TA SUB',
               'ERA5 TS ALL','ERA5 TS SUB','DynWat TR ALL','DynWat TR SUB','GEMStat TR']
SEASONS = {'Boreal Summer':{'NH':[6,7,8],'SH':[12,1,2]},
           'Boreal Winter':{'NH':[12,1,2],'SH':[6,7,8]}}

def fmt_sub(g):
    return g.replace(' ALL','$_{ALL}$').replace(' SUB','$_{SUB}$')

# ════════════════════════════ Data ════════════════════════════
print('reparing violin data...', flush=True)
def ss_to_seasonal(path, group):
    """Seasonal significance: a station is judged significantly warming or cooling only when all significant months in that season share the same sign.
    The old version (v1.2) used `(p<0.05).any()`, running 3 tests over 3 months without regard to sign,
    giving a family-wise error rate of ~1-0.95**3 = 14% under the null, which counts noise as significant."""
    df = pd.read_csv(path); df['station_id'] = df['station_id'].astype(str); rows=[]
    for sid, grp in df.groupby('station_id'):
        lat = float(grp['latitude'].iloc[0])
        for sea, mdef in SEASONS.items():
            mo  = mdef['NH'] if lat > 0 else mdef['SH']
            sub = grp[grp['month'].isin(mo)].dropna(subset=['trend_dec'])
            if sub.empty: continue
            has_p = 'p_value' in sub.columns
            if has_p:
                sig_m = sub['p_value'] < 0.05
                n_pos = int((sig_m & (sub['trend_dec'] > 0)).sum())
                n_neg = int((sig_m & (sub['trend_dec'] < 0)).sum())
                sig_warm = (n_pos > 0) and (n_neg == 0)
                sig_cool = (n_neg > 0) and (n_pos == 0)
                sig_any  = bool(sig_m.any())          # old definition, kept for comparison only
            else:
                sig_warm = sig_cool = sig_any = None
            rows.append({'station_id':sid,'season':sea,'trend_dec':sub['trend_dec'].mean(),
                         'sig_warm':sig_warm,'sig_cool':sig_cool,'sig_any':sig_any,
                         'n_months':int(len(sub))})
    r = pd.DataFrame(rows); r['group'] = group
    print(f'  {group}: {r["station_id"].nunique()} 站', flush=True); return r

vframes=[]
vframes.append(ss_to_seasonal(DATA_DIR+f'ss_era5_t2m_full_trends_{MIN_YEARS}y.csv','ERA5 TA ALL'))
vframes.append(ss_to_seasonal(DATA_DIR+f'ss_era5_t2m_obs_trends_{MIN_YEARS}y.csv','ERA5 TA SUB'))
vframes.append(ss_to_seasonal(DATA_DIR+f'ss_hadisd_full_trends_{MIN_YEARS}y.csv','HadISD TA ALL'))
vframes.append(ss_to_seasonal(DATA_DIR+f'ss_hadisd_obs_trends_{MIN_YEARS}y.csv','HadISD TA SUB'))
vframes.append(ss_to_seasonal(DATA_DIR+f'ss_era5_skt_full_trends_{MIN_YEARS}y.csv','ERA5 TS ALL'))
vframes.append(ss_to_seasonal(DATA_DIR+f'ss_era5_skt_obs_trends_{MIN_YEARS}y.csv','ERA5 TS SUB'))
# DynWat TR ALL: prefer the full record at the 219 station locations, so it can be compared as a control with the other columns in the figure.
# If the file is missing, fall back to the global grid (= v1.2 behavior) and print a warning.
DW_FULL = DATA_DIR + f'ss_dynwat_full_trends_{MIN_YEARS}y.csv'
if os.path.exists(DW_FULL):
    vframes.append(ss_to_seasonal(DW_FULL, 'DynWat TR ALL'))
    DW_ALL_IS_STATION = True
else:
    print(f'  [Warning] {DW_FULL}} not found; DynWat TR ALL falls back to the global grid, '
          f'which has a different spatial population from the other columns and cannot be used as a control.', flush=True)
    df=pd.read_csv(DATA_DIR+'dynwat_global_trends_v1.csv'); dw=[]
    for sea,mdef in SEASONS.items():
        for hl,mo in [('NH',mdef['NH']),('SH',mdef['SH'])]:
            lm=df['latitude']>0 if hl=='NH' else df['latitude']<=0
            a=df[lm&df['month'].isin(mo)].groupby(['latitude','longitude'])['trend_dec'].mean().reset_index()
            a['season']=sea; dw.append(a)
    s4=pd.concat(dw,ignore_index=True); s4['group']='DynWat TR ALL'; vframes.append(s4)
    print(f'  DynWat global grid: {len(s4):,} rows')
    DW_ALL_IS_STATION = False
vframes.append(ss_to_seasonal(DATA_DIR+f'ss_dynwat_obs_trends_{MIN_YEARS}y.csv','DynWat TR SUB'))
vframes.append(ss_to_seasonal(DATA_DIR+f'ss_gemstat_trends_{MIN_YEARS}y.csv','GEMStat TR'))
df_vio=pd.concat(vframes,ignore_index=True)

# ════════════════════════════ Plotting functions ════════════════════════
def box_stats(vals):
    v=np.array(vals.dropna())
    if len(v)<3: return None
    return {'n':len(v),'vals':v,'mean':float(np.mean(v)),'median':float(np.percentile(v,50)),
            'q25':float(np.percentile(v,25)),'q75':float(np.percentile(v,75)),
            'w_lo':float(np.percentile(v,5)),'w_hi':float(np.percentile(v,95))}

VW=0.10; BW=0.035

def draw_vb(ax,xc,st,color,txt_c):
    if st is None or len(st['vals'])<5: return
    v=st['vals']; lo,hi=max(-6,st['w_lo']-0.15),min(6,st['w_hi']+0.15)
    yg=np.linspace(lo,hi,200)
    try:
        kde=gaussian_kde(v,bw_method='scott'); d=kde(yg); d=d/d.max()*VW
    except: return
    ax.fill_betweenx(yg,xc-d,xc+d,color=color,alpha=0.4,zorder=2)
    ax.plot(xc-d,yg,color=color,lw=0.4,alpha=0.6,zorder=3)
    ax.plot(xc+d,yg,color=color,lw=0.4,alpha=0.6,zorder=3)
    x0,x1=xc-BW,xc+BW
    ax.plot([xc,xc],[st['w_lo'],st['q25']],color=color,lw=0.7,alpha=0.9,zorder=4)
    ax.plot([xc,xc],[st['q75'],st['w_hi']],color=color,lw=0.7,alpha=0.9,zorder=4)
    for yy in [st['w_lo'],st['w_hi']]:
        ax.plot([xc-BW*.5,xc+BW*.5],[yy,yy],color=color,lw=0.7,zorder=4)
    h=max(st['q75']-st['q25'],1e-6)
    ax.add_patch(mpatches.FancyBboxPatch((x0,st['q25']),BW*2,h,boxstyle='square,pad=0',
                 facecolor=color,alpha=0.75,edgecolor=color,lw=0.5,zorder=5))
    ax.plot([x0,x1],[st['median'],st['median']],color='black',lw=0.8,zorder=6)
    ax.plot(xc,st['mean'],marker='*',ms=4,color='black',zorder=7,markeredgewidth=0.3)

def draw_violin(ax, txt_c):
    ax.set_facecolor('white')
    ax.axhline(0,color='#9a9a9a',lw=1.1,ls=':',zorder=1)
    ax.grid(axis='y',color='black',alpha=0.07,lw=0.3)
    INNER=0.28; GAP=0.85
    GXPOS={g:i*GAP for i,g in enumerate(GROUP_ORDER)}
    VXPOS={(g,'Boreal Summer'):GXPOS[g]-INNER/2 for g in GROUP_ORDER}
    VXPOS.update({(g,'Boreal Winter'):GXPOS[g]+INNER/2 for g in GROUP_ORDER})
    NO_STRIP={'DynWat TR ALL','ERA5 TA ALL','ERA5 TS ALL','HadISD TA ALL'}
    for g in GROUP_ORDER:
        for sea,sc in [('Boreal Summer',SC['Boreal Summer']),('Boreal Winter',SC['Boreal Winter'])]:
            xcg=VXPOS[(g,sea)]
            sub=df_vio[(df_vio['group']==g)&(df_vio['season']==sea)]['trend_dec']
            st=box_stats(sub); draw_vb(ax,xcg,st,sc,txt_c)
            if g not in NO_STRIP and st is not None:
                rng=np.random.default_rng(42)
                jit=rng.uniform(-BW*.7,BW*.7,len(st['vals']))
                ax.scatter(xcg+jit,st['vals'],s=1.5,color=sc,alpha=0.2,zorder=1,linewidths=0)
    # #7 remove the central dashed divider (former sep axvline) — intentionally not drawn here
    ax.set_xticks([GXPOS[g] for g in GROUP_ORDER])
    ax.set_xticklabels([fmt_sub(g) for g in GROUP_ORDER], rotation=0, ha='center', fontsize=6.5, color='black')
    # label colors matched to the violins
    SUMMER_COLOR = '#D6604D'   # boreal summer red
    WINTER_COLOR = '#4393C3'   # boreal winter blue
    # label y position (data coordinates); axis bottom -5, labels placed below with extra line spacing
    LBL_Y_SUM = -5.9
    LBL_Y_WIN = -6.75
    for g in GROUP_ORDER:
        xc = GXPOS[g]
        sub_sum = df_vio[(df_vio['group']==g)&(df_vio['season']=='Boreal Summer')]
        sub_win = df_vio[(df_vio['group']==g)&(df_vio['season']=='Boreal Winter')]
        n_sum = len(sub_sum.dropna(subset=['trend_dec'])); n_win = len(sub_win.dropna(subset=['trend_dec']))
        has_sig = ('sig_warm' in df_vio.columns) and (not sub_sum['sig_warm'].dropna().empty)
        xs = VXPOS[(g,'Boreal Summer')]   # summer violin x
        xw = VXPOS[(g,'Boreal Winter')]   # winter violin x
        if has_sig:
            w_s = int(sub_sum['sig_warm'].fillna(False).sum()); c_s = int(sub_sum['sig_cool'].fillna(False).sum())
            w_w = int(sub_win['sig_warm'].fillna(False).sum()); c_w = int(sub_win['sig_cool'].fillna(False).sum())
            lbl_sum = f'{w_s}↑ {c_s}↓ / {n_sum}'
            lbl_win = f'{w_w}↑ {c_w}↓ / {n_win}'
        else:
            lbl_sum = f'{n_sum}'
            lbl_win = f'{n_win}'
        # x aligned to the violins; summer (upper row, red), winter (lower row, blue). y in data coordinates, line spacing 0.75
        ax.text(xs, LBL_Y_SUM, lbl_sum, ha='center', va='top', fontsize=6.0,
                color=SUMMER_COLOR, alpha=0.95, clip_on=False)
        ax.text(xw, LBL_Y_WIN, lbl_win, ha='center', va='top', fontsize=6.0,
                color=WINTER_COLOR, alpha=0.95, clip_on=False)
    ax.set_xlim(min(VXPOS.values())-0.3,max(VXPOS.values())+0.3)
    ax.set_ylim(-7.3,6.0)
    ax.set_ylabel('Trend (°C/dec)',color='black',fontsize=10,labelpad=2)
    ax.tick_params(colors='black',labelsize=9,pad=3)
    ax.spines[['top','right','bottom']].set_visible(False)
    ax.spines['left'].set_color('#888888'); ax.spines['left'].set_linewidth(0.8)
    # #7 remove the "← Air / Water →" text — intentionally not drawn here

# ════════════════════════════ Layout  ════════════════════════════
def make_fig():
    bg='white'; txt='#111111'
    fig=plt.figure(figsize=(10,4.5),facecolor=bg)
    # Single panel; left/right margins aligned with the other two figures; bottom padding for x labels, station counts, and legend
    ax=fig.add_axes([0.08, 0.20, 0.98-0.08, 0.66])
    draw_violin(ax, txt)
    vleg=[mpatches.Patch(color=SC['Boreal Summer'],alpha=0.7,label='Boreal Summer'),
          mpatches.Patch(color=SC['Boreal Winter'],alpha=0.7,label='Boreal Winter'),
          Line2D([0],[0],color='black',lw=0.8,label='Median'),
          Line2D([0],[0],marker='*',color='black',ms=5,lw=0,label='Mean')]
    ax.legend(handles=vleg,loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=4,
              frameon=False,labelcolor='black',fontsize=8,
              handlelength=1.2,columnspacing=0.8,borderpad=0.4,handletextpad=0.4)
    # section label
    fig.text(0.012, 0.93, 'a', color=txt, fontsize=10,
             fontweight='bold', ha='left', va='center')
    return fig

# ════════════════════════ Statistics summary output ════════════════════════
def dump_stats(path):
    recs=[]
    for g in GROUP_ORDER:
        for sea in ['Boreal Summer','Boreal Winter']:
            sub = df_vio[(df_vio['group']==g)&(df_vio['season']==sea)]
            v = sub['trend_dec'].dropna().values
            if len(v)==0: continue
            has_sig = ('sig_warm' in sub.columns) and (not sub['sig_warm'].dropna().empty)
            rec = dict(group=g, season=sea, n=len(v),
                       median=np.median(v), q25=np.percentile(v,25), q75=np.percentile(v,75),
                       iqr=np.percentile(v,75)-np.percentile(v,25), mean=np.mean(v))
            if has_sig:
                w=int(sub['sig_warm'].fillna(False).sum()); c=int(sub['sig_cool'].fillna(False).sum())
                a=int(sub['sig_any'].fillna(False).sum())
                rec.update(sig_warm=w, sig_cool=c, pct_sig_warm=100*w/len(v), pct_sig_cool=100*c/len(v),
                           sig_any_oldrule=a, pct_sig_any_oldrule=100*a/len(v))
            recs.append(rec)
    out = pd.DataFrame(recs)
    out.to_csv(path, index=False, float_format='%.4f')
    print(f'Statistics summary:{path}')
    return out

stats = dump_stats(FIG_DIR + f'station_scale_stats_{MIN_YEARS}y.csv')
print(stats[['group','season','n','median','iqr','pct_sig_warm','pct_sig_cool','pct_sig_any_oldrule']].to_string(index=False))

print('Plotting (station violin)...', flush=True)
fw=make_fig()
out=FIG_DIR+f'Main_Text_Figure4_station_violin_v1.3_{MIN_YEARS}y_white.png'
fw.savefig(out,dpi=800,facecolor='white')   # fixed 10-in width, no cropping → same width as the other two figures
print(f'Saved: {out}'); plt.close(fw)
print('Finish!')
