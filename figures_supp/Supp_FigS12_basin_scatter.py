#!/usr/bin/env python3
# ============================================================
# Supp_FigS12_basin_scatter.py
#   Supplementary: basin-scale water vs air temperature trend scatter (formerly Fig. S11)
#
#   ⚠ Key correction relative to the original FIG3_redesign_scatter_v1.2.py
#   The original put air temperature on y and water temperature on x, so the OLS slope was Δair/Δwater,
#   the opposite direction to the Methods definition β = Δwater/Δair, and not the same quantity as the
#   basin rows of Table S1. This version swaps the two axes so the slope is the basin-scale β.
#
#   Usage
#     python Supp_FigS12_basin_scatter.py                # 5y
#     python Supp_FigS12_basin_scatter.py --min_years 8
#
#
#   Original v1.2 notes follow
# ------------------------------------------------------------
# FIG3_redesign_scatter_v1.2.py   (white-background editorial version)
#   Visual upgrade over the v1.0 multi-scale scatter:
#     - Basin: grey crosses (kept from v1.0)
#     - Inverted visual hierarchy: Continental as white-haloed open circles, Global as a large solid anchor
#     - Added basin OLS regression line + 95% confidence-interval band (extended toward both ends)
#     - Very faint warm/cool quadrant background (Fig2 red/blue), faint 1:1 corridor band
#     - Open axes, outward thin ticks, enlarged tiered axis labels, generous whitespace
#   Background: white. Main structure (12 panels, three scales, season labels, dataset-named axis labels) unchanged.
#   Units: °C/decade.
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import linregress
from scipy import stats
warnings.filterwarnings("ignore")

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--min_years", type=int, default=5)
MIN_YEARS = _ap.parse_args().min_years
SUFFIX = "" if MIN_YEARS == 5 else f"_{MIN_YEARS}y"

DATA_DIR = "/work/home/H.Jason421/water_temp_for_publish/data/"
FIG_DIR  = "/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary/"
os.makedirs(FIG_DIR, exist_ok=True)

SEASONS = {"Boreal Summer":{"NH":[6,7,8],"SH":[12,1,2]},
           "Boreal Winter":{"NH":[12,1,2],"SH":[6,7,8]}}
CONTINENTS = ["Asia","Europe","South/SE Asia","Latin America","North America"]

# ── Colors (Okabe-Ito)──────────────────────
C_GLOBAL = "#111111"
C_CONT   = {"Asia":"#0072B2","Europe":"#009E73","South/SE Asia":"#CC79A7",
            "Latin America":"#E69F00","North America":"#D55E00"}
C_BASIN  = "#9e9e9e"
C_REG    = "#c2185b"            # egression line / CI band
C_ZERO   = "#d8d8d8"
C_ONE2ONE= "#caa200"
C_WARM   = "#d6604d"; C_COOL = "#327cb7"     # quadrant background (Fig2 red/blue)
SHOW_ONE2ONE = True

# ── Axis ranges (°C/decade); adjust per data ──────────────────
X_LIM = 2.5   # TR trend
Y_LIM = 2.5   # TA trend

AIR_COL  = {"T2M":"era5_t2m","HadISD":"hadisd","Skt":"era5_skt"}
WAT_COL  = {"GEM":"gemstat","Dyn":"dynwat"}
AIR_NAME = {"T2M":"ERA5 TA$_{SUB}$","HadISD":"HadISD TA$_{SUB}$","Skt":"ERA5 TS$_{SUB}$"}
WAT_NAME = {"GEM":"GEMStat TR","Dyn":"DynWat TR$_{SUB}$"}
COLS = ["T2M","HadISD","Skt"]
ROWS = [("Boreal Summer","GEM"),("Boreal Summer","Dyn"),
        ("Boreal Winter","GEM"),("Boreal Winter","Dyn")]

# ════════════════════════════ Data (DATA-BLOCK)════════════════
print("Setting up scale_trends_v1 reading layer (continental/global)...", flush=True)
SCALE_DIR = '/work/home/H.Jason421/water_temp_for_publish/data/scale_trends_v1/'
REG_LATC  = {"Asia":45,"Europe":50,"South/SE Asia":10,"Latin America":-15,
             "North America":45,"Global":45}                 # Global uses northern-hemisphere (boreal) months
SEAS_MO_RG = {"Boreal Summer":{"NH":[6,7,8],"SH":[12,1,2]},
              "Boreal Winter":{"NH":[12,1,2],"SH":[6,7,8]}}
_TCACHE = {}
def _load_scale(col, scale):
    ver = "" if col == "gemstat" else "_sub"     # GEMStat TR has no version; the rest use sub
    fn  = f"{SCALE_DIR}{col}_{scale}{ver}{SUFFIX}.csv"
    if fn not in _TCACHE:
        try:
            _TCACHE[fn] = pd.read_csv(fn)
        except FileNotFoundError:
            print(f"  ⚠ {fn} not found", flush=True); _TCACHE[fn] = None
    return _TCACHE[fn]

print("Read metadata（basin latitudes）...", flush=True)
xl   = pd.ExcelFile("/work5/H.Jason421/GEMStat_new/GEMS-Water_data_request.xls")
meta = pd.read_excel(xl, sheet_name="Station_Metadata")
meta = meta.rename(columns={"GEMS Station Number":"station_id","Water Type":"water_type",
                            "Main Basin":"basin","Latitude":"lat","Longitude":"lon"})
meta = meta[meta["water_type"]=="River station"].dropna(subset=["lat","lon"]).drop_duplicates("station_id")
BLAT = meta.dropna(subset=["basin"]).groupby("basin")["lat"].median().to_dict()

print("Read basin trends...", flush=True)
BASIN_FILES = {"T2M":f"era5_t2m_basin_sub{SUFFIX}.csv","HadISD":f"hadisd_basin_sub{SUFFIX}.csv",
               "Skt":f"era5_skt_basin_sub{SUFFIX}.csv","GEM":f"gemstat_basin{SUFFIX}.csv",
               "Dyn":f"dynwat_basin_sub{SUFFIX}.csv"}
def basin_seasonal(fname, tcol="trend_per_decade"):
    df = pd.read_csv(SCALE_DIR+fname); rows=[]
    for basin,grp in df.groupby("basin"):
        lat = BLAT.get(basin, 45)
        for sea,hemi in SEASONS.items():
            mo = hemi["NH"] if lat>=0 else hemi["SH"]
            sub = grp[grp["month"].isin(mo)]
            if sub.empty: continue
            rows.append({"basin":basin,"season":sea,"trend":sub[tcol].mean()})
    return pd.DataFrame(rows)
basin_tr = {ds: basin_seasonal(f) for ds,f in BASIN_FILES.items()}
# ════════════════════════════ /DATA-BLOCK ════════════════════

def regional_trend(region, season, col):
    scale = "global" if region == "Global" else "continental"
    df = _load_scale(col, scale)
    if df is None: return np.nan
    grp = "GLOBAL" if region == "Global" else region
    sub = df[df[scale] == grp]
    if sub.empty: return np.nan
    hemi = "NH" if REG_LATC[region] >= 0 else "SH"
    s = sub[sub["month"].isin(SEAS_MO_RG[season][hemi])]["trend_per_decade"].dropna()
    return float(s.mean()) if len(s) else np.nan

def panel_points(season, water_ds, air_ds):
    acol, wcol = AIR_COL[air_ds], WAT_COL[water_ds]
    # x = TA, y = TR, so the OLS slope = beta = dWater/dAir (consistent with Methods and Table S1)
    g = (regional_trend("Global",season,acol), regional_trend("Global",season,wcol))
    cont = [(regional_trend(c,season,acol), regional_trend(c,season,wcol), C_CONT[c]) for c in CONTINENTS]
    a = basin_tr[air_ds]; w = basin_tr[water_ds]
    aa = a[a["season"]==season][["basin","trend"]].rename(columns={"trend":"x"})
    ww = w[w["season"]==season][["basin","trend"]].rename(columns={"trend":"y"})
    m  = aa.merge(ww, on="basin").dropna()
    return g, cont, m["x"].values.astype(float), m["y"].values.astype(float)

def basin_scatter(ax, x, y):
    ax.scatter(x, y, marker="x", c=C_BASIN, s=18, linewidths=0.8, alpha=0.6, zorder=3)

def basin_regression(ax, x, y):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return None
    sl, ic, r, p, se = linregress(x, y)
    n = len(x); xbar = x.mean(); Sxx = np.sum((x-xbar)**2)
    if Sxx <= 0: return
    resid = y - (sl*x + ic); s_err = np.sqrt(np.sum(resid**2)/max(n-2, 1))
    tval = stats.t.ppf(0.975, max(n-2, 1))
    xg = np.linspace(-X_LIM, X_LIM, 120); yh = sl*xg + ic
    ci = tval*s_err*np.sqrt(1.0/n + (xg-xbar)**2/Sxx)
    ax.fill_between(xg, yh-ci, yh+ci, color=C_REG, alpha=0.15, zorder=4, lw=0)
    ax.plot(xg, yh, color=C_REG, lw=1.7, zorder=5)
    hw = stats.t.ppf(0.975, max(n-2, 1)) * se
    return dict(beta=sl, lo=sl-hw, hi=sl+hw, p=p, r2=r**2, n=n)

def draw_panel(ax, season, water_ds, air_ds, show_yticklab, xlabel, ylabel):
    ax.set_facecolor("white")
    # quadrant background (very faint)
    ax.add_patch(Rectangle((0,0),X_LIM,Y_LIM,color=C_WARM,alpha=0.05,zorder=0,lw=0))
    ax.add_patch(Rectangle((-X_LIM,-Y_LIM),X_LIM,Y_LIM,color=C_COOL,alpha=0.05,zorder=0,lw=0))
    # 1:1 corridor + line
    if SHOW_ONE2ONE:
        xs = np.linspace(-X_LIM, X_LIM, 2)
#        ax.fill_between(xs, xs-0.25, xs+0.25, color="#f1c40f", alpha=0.10, zorder=1, lw=0)
        ax.plot([-X_LIM,X_LIM],[-X_LIM,X_LIM], color=C_ONE2ONE, lw=1.0, ls=(0,(6,3)), alpha=0.85, zorder=1)
    ax.axhline(0, color=C_ZERO, lw=1.0, zorder=1); ax.axvline(0, color=C_ZERO, lw=1.0, zorder=1)
    g, cont, bx, by = panel_points(season, water_ds, air_ds)
    basin_scatter(ax, bx, by)
    _fit = basin_regression(ax, bx, by)
    # Continental: white halo + colored open circle
    for cx, cy, col in cont:
        if np.isfinite(cx) and np.isfinite(cy):
            ax.scatter(cx, cy, s=150, c="white", zorder=6, linewidths=0, alpha=0.9)
            ax.scatter(cx, cy, s=82, facecolors="none", edgecolors=col, linewidths=1, zorder=7)
    # Global: large solid anchor + white halo
    if np.isfinite(g[0]) and np.isfinite(g[1]):
        ax.scatter(g[0], g[1], s=230, c="white", zorder=8, linewidths=0, alpha=0.9)
        ax.scatter(g[0], g[1], s=130, c=C_GLOBAL, edgecolors="white", linewidths=1.2, zorder=9)
    ax.set_xlim(-X_LIM, X_LIM); ax.set_ylim(-Y_LIM, Y_LIM)
    ax.set_xticks([-2,-1,0,1,2]); ax.set_yticks([-2,-1,0,1,2])
    ax.tick_params(colors="#444444", labelsize=8, length=3.5, width=0.8, direction="out")
    if not show_yticklab: ax.set_yticklabels([])
    if xlabel: ax.set_xlabel(xlabel, fontsize=10.5, color="#1a1a1a", labelpad=4)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10.5, color="#1a1a1a", labelpad=5)
    # open axes
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]:
        ax.spines[s].set_color("#8a8a8a"); ax.spines[s].set_linewidth(0.9)
    return _fit

FITS = []

def make_fig():
    fig = plt.figure(figsize=(10, 10)); fig.patch.set_facecolor("white")
    # Refined proportions: generous whitespace, non-square panels
    gs = gridspec.GridSpec(4, 3, left=0.085, right=0.905, top=0.965, bottom=0.115,
                           hspace=0.24, wspace=0.13)
    A = np.empty((4,3), dtype=object)
    for r,(season,water_ds) in enumerate(ROWS):
        for c,air_ds in enumerate(COLS):
            ax = fig.add_subplot(gs[r,c]); A[r,c] = ax
            FITS.append((water_ds, air_ds, season,
                         draw_panel(ax, season, water_ds, air_ds, show_yticklab=(c==0),
                       xlabel=f"{AIR_NAME[air_ds]}  (°C/dec)",
                       ylabel=f"{WAT_NAME[water_ds]}  (°C/dec)")))
    # Right-side season labels (each spanning two rows)
    for rp, lab in [((0,1),"Boreal Summer"), ((2,3),"Boreal Winter")]:
        pt = A[rp[0],2].get_position(); pb = A[rp[1],2].get_position()
        fig.text(pt.x1+0.018, (pt.y1+pb.y0)/2, lab, rotation=270, ha="left", va="center",
                 fontsize=12, fontweight="bold", color="#1a1a1a")
    # legend
    H = [Line2D([0],[0],marker="o",color="w",markerfacecolor=C_GLOBAL,markeredgecolor="white",
                markersize=11,linestyle="None",label="Global")]
    H += [Line2D([0],[0],marker="o",color="w",markerfacecolor="none",markeredgecolor=C_CONT[c],
                 markeredgewidth=2.4,markersize=10,linestyle="None",label=c) for c in CONTINENTS]
    H += [Line2D([0],[0],marker="x",color=C_BASIN,markersize=9,linestyle="None",label="Basin"),
          Line2D([0],[0],color=C_REG,lw=1.7,label="OLS (basin) + 95% CI"),
          Line2D([0],[0],color=C_ONE2ONE,lw=1.0,ls="--",label="1:1 line")]
    fig.legend(handles=H, loc="lower center", bbox_to_anchor=(0.5,0.008), ncol=5, fontsize=9,
               labelcolor="#1a1a1a", facecolor="white", edgecolor="#bbbbbb", borderpad=0.7,
               handletextpad=0.5, columnspacing=1.2, framealpha=0.95)
    out = FIG_DIR + f"Supplementary_Fig_basin_scatter_{MIN_YEARS}y.png"
    plt.savefig(out, dpi=800, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(); print(f"Saved: {out}")

print("Plotting...", flush=True)
make_fig()

print("\n--- basin-scale beta (compare against the Basin rows of Table S1) ---")
print(f'{"water":<16}{"air":<20}{"season":<16}{"beta":>7}{"":<2}{"95% CI":>18}{"n":>5}')
for wat, air, sea, f in FITS:
    if f is None:
        print(f'{WAT_NAME[wat]:<16}{AIR_NAME[air]:<20}{sea:<16}{"n/a":>7}')
        continue
    star = "*" if f["p"] < 0.05 else ""
    print(f'{WAT_NAME[wat].replace("$_{SUB}$","_SUB"):<16}'
          f'{AIR_NAME[air].replace("$_{SUB}$","_SUB"):<20}{sea:<16}'
          f'{f["beta"]:>7.2f}{star:<2}[{f["lo"]:>6.2f},{f["hi"]:>6.2f}]{f["n"]:>5}')
print("All done!")