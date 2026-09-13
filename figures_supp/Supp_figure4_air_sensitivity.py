#!/usr/bin/env python3
# ============================================================
# Supp_figure4_air_sensitivity.py
#   Supplementary version: Fig. 4b river-air sensitivity β, record-length threshold robustness check
#
#   Usage
#     python Supp_figure4_air_sensitivity.py --min_years 8
#     python Supp_figure4_air_sensitivity.py --min_years 10
#
#     while the Methods require "at least five paired stations" to keep a regression. This version lists, at the end,
#     the combinations dropped for having too few paired stations, for cross-checking against the caption.
#     Note that "at least five stations" and --min_years are two independent criteria; the former is fixed at 5.
#   Output (figures/Supplementary/)
#
#     Supplementary_Fig4b_air_sensitivity_{ny}y.png
#     Supplementary_Fig4b_air_sensitivity_table_{ny}y.csv
#
#   Original v1.1–v1.5 notes follow
# ------------------------------------------------------------
# Main_Text_Figure4_air_sensitivity_v1.5.py   (main text Fig4 second row; formerly Fig3)
#   v1.5 changes: save at full 10in width (removed bbox_inches='tight'), equal width with the violin, for easier combine alignment.
#   v1.4 changes: added panel letter b (top-left, same idea as the violin's 'a').
#   v1.3 changes: season line styles swapped and thickened (winter=solid, summer=dashed, so short CIs are still visible);
#             removed the repeated air-temperature text below each pair (now shown by color + legend).
#   v1.2 changes:
#     - Reworked the x-axis to align with the violin: water-temperature group (left DynWat / right GEMStat) → air temperature (T2M/HadISD/Skt) → season (summer/winter paired)
#     - Season distinguished by line style (summer=solid, winter=dashed), color still = air temperature; bottom labels air temperature (per pair) + water-temperature group
#     - β=0 / β=1 reference lines added back
#   v1.1 changes:
#     - Removed significance asterisks above points (redundant with solid/open)
#     - Removed β=0 / β=1 reference lines, the decoupled/1:1 text labels and their legend entries
#     - Moved Summer/Winter season labels to the bottom, using the same violin red/blue (summer #d6604, winter #327cb)
#     - Made the season divider line more prominent
#   Adapted from FIG3_sensitivity_forest_v1.2.py:
#     - Kept only the station scale (removed the basin column and its metadata/file reading).
#     - Rotated the whole thing 90°: categories (season × water × air) laid along the horizontal axis, β moved to the vertical axis,
#       forming a wide horizontal band, easy to stack vertically with the violin and the DO/Cond rows into FIG3.
#   β = Δ(water)/Δ(air) (water-temperature trend regressed on air-temperature trend) + 95% CI + p.
#   solid=significant (p<0.05), open=not significant; color=air-temperature dataset; reference lines β=0 (decoupled), β=1 (1:1).
#   Includes a CSV table of β/CI/p/R²/n. Units: trends in °C/decade (β is dimensionless).
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
from matplotlib.lines import Line2D
from scipy.stats import linregress, t as tdist
warnings.filterwarnings("ignore")

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--min_years", type=int, default=8,
                 help="record-length threshold, 8 or 10; for 5 use the main text v1.5")
MIN_YEARS = _ap.parse_args().min_years

DATA_DIR = "/work/home/H.Jason421/water_temp_for_publish/data/"
FIG_DIR  = "/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary/"
os.makedirs(FIG_DIR, exist_ok=True)

MIN_PAIRED_STATIONS = 5   # required by Methods, independent of --min_years, fixed at 5

SEASONS = {"Boreal Summer":{"NH":[6,7,8],"SH":[12,1,2]},
           "Boreal Winter":{"NH":[12,1,2],"SH":[6,7,8]}}

C_AIR = {"T2M":"#d6604d","HadISD":"#F4A259","Skt":"#991027"}
AIR_NAME = {"T2M":"ERA5 TA$_{SUB}$","HadISD":"HadISD TA$_{SUB}$","Skt":"ERA5 TS$_{SUB}$"}
WAT_NAME = {"GEM":"GEMStat TR","Dyn":"DynWat TR$_{SUB}$"}
AIRS  = ["T2M","HadISD","Skt"]
# Season colors: aligned with the violin (summer=red, winter=blue)
RED_BASE  = "#d6604d"
BLUE_BASE = "#327cb7"
SEASON_C  = {"Boreal Summer":RED_BASE,"Boreal Winter":BLUE_BASE}
# Category order: water-temperature group (left DynWat / right GEMStat) → air temperature (T2M/HadISD/Skt) → season (summer/winter paired)
ROWS = [(wat, air, se) for wat in ["Dyn","GEM"]
        for air in AIRS for se in ["Boreal Summer","Boreal Winter"]]

# ════════════════════════════ Data (station only)════════════════
print(f"REad station trends（MIN_YEARS={MIN_YEARS}）...", flush=True)
STATION_FILES = {"T2M":f"ss_era5_t2m_obs_trends_{MIN_YEARS}y.csv",
                 "HadISD":f"ss_hadisd_obs_trends_{MIN_YEARS}y.csv",
                 "Skt":f"ss_era5_skt_obs_trends_{MIN_YEARS}y.csv",
                 "GEM":f"ss_gemstat_trends_{MIN_YEARS}y.csv",
                 "Dyn":f"ss_dynwat_obs_trends_{MIN_YEARS}y.csv"}
def station_seasonal(fname):
    df = pd.read_csv(DATA_DIR+fname); df["station_id"]=df["station_id"].astype(str); rows=[]
    for sid,grp in df.groupby("station_id"):
        lat = float(grp["latitude"].iloc[0])
        for sea,hemi in SEASONS.items():
            mo = hemi["NH"] if lat>0 else hemi["SH"]
            sub = grp[grp["month"].isin(mo)].dropna(subset=["trend_dec"])
            if sub.empty: continue
            rows.append({"key":sid,"season":sea,"trend":sub["trend_dec"].mean()})
    return pd.DataFrame(rows)

station_tr = {ds: station_seasonal(f) for ds,f in STATION_FILES.items()}

def fit_w_on_a(air_vals, water_vals):
    x=np.asarray(air_vals,float); y=np.asarray(water_vals,float)
    m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]; n=len(x)
    if n<MIN_PAIRED_STATIONS or np.ptp(x)==0: return None
    sl,ic,r,p,se=linregress(x,y)
    ci=tdist.ppf(0.975,n-2)*se
    return dict(beta=sl,lo=sl-ci,hi=sl+ci,p=p,r2=r**2,n=n)

DROPPED=[]

def combo_fit(tr_dict, season, water_ds, air_ds):
    a=tr_dict[air_ds]; w=tr_dict[water_ds]
    aa=a[a["season"]==season][["key","trend"]].rename(columns={"trend":"air"})
    ww=w[w["season"]==season][["key","trend"]].rename(columns={"trend":"wat"})
    m=aa.merge(ww,on="key").dropna()
    if len(m)<MIN_PAIRED_STATIONS:
        DROPPED.append((WAT_NAME[water_ds], AIR_NAME[air_ds], season, len(m)))
        return None
    return fit_w_on_a(m["air"].values, m["wat"].values)

EST={}; records=[]
for (wat,air,se) in ROWS:
    r=combo_fit(station_tr, se, wat, air); EST[(wat,air,se)]=r
    if r: records.append(dict(scale="Station",season=se,
                              water=WAT_NAME[wat].replace("$_{SUB}$","_SUB"),
                              air=AIR_NAME[air].replace("$_{SUB}$","_SUB"),
                              beta=r["beta"],ci_lo=r["lo"],ci_hi=r["hi"],
                              p_value=r["p"],r2=r["r2"],n=r["n"]))
tab=pd.DataFrame(records)
tab_out=FIG_DIR+f"Supplementary_Fig4b_air_sensitivity_table_{MIN_YEARS}y.csv"
tab.to_csv(tab_out,index=False); print(f"Saved table: {tab_out}")

def stars(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

# ── x coordinates (whitespace between categories: season > water > air) ──
def make_xpos():
    xp=[]; x=0.0; prev=None
    for cur in ROWS:
        wat,air,se = cur
        if prev is not None:
            if   prev[0]!=wat: x+=2.4   # switching TR group: large gap
            elif prev[1]!=air: x+=1.2   # switching TA : medium gap
            else:              x+=0.5   # summer→winter within the same air temperature: small paired gap
        xp.append(x); prev=cur
    return np.array(xp)
XP=make_xpos()

def make_fig():
    allci=[v for v in EST.values() if v]
    if allci:
        lo=min(min(v["lo"] for v in allci), -0.2); hi=max(max(v["hi"] for v in allci), 1.05)
        pad=0.10*(hi-lo); ylim=(max(-2.5,lo-pad), min(3.5,hi+pad))
    else:
        ylim=(-0.5,1.5)

    fig=plt.figure(figsize=(10,3.9)); fig.patch.set_facecolor("white")
    ax=fig.add_axes([0.08,0.15,0.90,0.75])   # left/right margins aligned with the violin (0.08~0.98) so combine lines up

    # panel letter b (same idea as the violin's 'a')
    fig.text(0.012, 0.93, 'b', color="#111111", fontsize=10, fontweight="bold",
             ha="left", va="center")

    # reference lines added back: β=0 (decoupled) dashed, β=1 (1:1) dotted
    ax.axhline(0,color="#9a9a9a",lw=1.1,ls=":",zorder=1)
    ax.axhline(1,color="#cccccc",lw=1.0,ls=":",zorder=1)

    # season distinguished by line style (winter=solid, summer=dashed; thickened dashes so short CIs are still visible)
    SEASON_LS={"Boreal Summer":(0,(2,4)),"Boreal Winter":"-"}
    for (wat,air,se),xp in zip(ROWS,XP):
        r=EST[(wat,air,se)]
        if r is None: continue
        col=C_AIR[air]
        # significance is already shown by whether the 95% CI crosses 0; solid/open would be redundant -> all solid
        ax.plot([xp,xp],[r["lo"],r["hi"]],color=col,lw=2.0,ls=SEASON_LS[se],
                zorder=3)
        ax.scatter(xp,r["beta"],s=66,facecolor=col,edgecolor=col,
                   linewidths=1.8,zorder=4)

    ax.set_ylim(*ylim); ax.set_xlim(XP.min()-0.9, XP.max()+0.9)
    ax.set_xticks([])
    ax.set_ylabel(r"$\beta$",fontsize=10)
    ax.tick_params(labelsize=9)
    for s in ["top","right","bottom"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#888888")

    # group labels: keep only the water-temperature group (DynWat/GEMStat); air temperature shown by color + legend, no repeated text
    trans=mt.blended_transform_factory(ax.transData, fig.transFigure)
    g_wat={}
    for (wat,air,se),xp in zip(ROWS,XP):
        g_wat.setdefault(wat,[]).append(xp)
    for wat,xs in g_wat.items():
        fig.text(np.mean(xs),0.150,WAT_NAME[wat],ha="center",va="top",fontsize=10.5,
                 fontweight="bold",color="#111111",transform=trans)
    # divider line between the two water-temperature groups (made prominent)
    dxs=g_wat["Dyn"]; gxs=g_wat["GEM"]
    xdiv=(max(dxs)+min(gxs))/2
    ax.axvline(xdiv,color="#888888",lw=1.4,ls="--",zorder=1)

    # legend
    H=[Line2D([0],[0],color="#444",lw=2.0,ls=(0,(2,4)),label="Boreal Summer"),
       Line2D([0],[0],color="#444",lw=2.0,ls="-",label="Boreal Winter")]
    H+=[Line2D([0],[0],marker="o",color="w",markerfacecolor=C_AIR[a],markeredgecolor=C_AIR[a],
               markersize=9,linestyle="None",label=AIR_NAME[a]) for a in AIRS]
    fig.legend(handles=H,loc="lower center",bbox_to_anchor=(0.5,0.005),ncol=5,fontsize=8,
               frameon=False,columnspacing=1.1,handletextpad=0.4)

    out=FIG_DIR+f"Supplementary_Fig4b_air_sensitivity_{MIN_YEARS}y.png"
    plt.savefig(out,dpi=800,facecolor="white")   # full 10in width (no cropping) → equal width with the violin so combine lines up
    plt.close(); print(f"Saved: {out}")

print("Plotting...",flush=True)
make_fig()

# ── combinations dropped for too few paired stations  ─────────────────────────────────
print(f"\n--- {len(tab)}/{len(ROWS)} regression combinations valid ---")
if DROPPED:
    print(f"Not kept due to fewer than {MIN_PAIRED_STATIONS} paired stations:")
    for wat,air,se,n in DROPPED:
        print(f"  {wat:<16} x {air:<20} {se:<14} n = {n}")
else:
    print("All combinations reached the minimum number of paired stations.")
if len(tab):
    print("\n--- Number of paired stations per combination ---")
    print(tab[["water","air","season","n","beta","p_value"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("All done!")
