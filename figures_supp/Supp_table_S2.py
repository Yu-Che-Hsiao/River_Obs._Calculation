#!/usr/bin/env python3
# ============================================================
# Supp_table_S2.py
#   Generate Table S2: River–air temperature sensitivity β = ΔTR/ΔTA
#   at both the station and basin scales.
#
#   The logic follows the two existing scripts directly, to stay consistent with the figures:
#     - Station β：同 Main_Text_Figure4_air_sensitivity_v1.6.py
#       (reads ss_*_trends_5y.csv, averages trend_dec per station per season,
#       pairs water/air by station_id, linregress(air, water))
#     - Basin β: same as Supp_figure_basin_scatter.py
#       (reads *_basin_sub.csv, averages trend_per_decade per basin per season,
#       pairs water/air by basin, linregress(air, water))
#   x = TA、y = TR, OLS slope = β = ΔTR/ΔTA (consistent with Methods).
#
#   Usage: python Supp_table_S2.py
#   Output: figures/Supplementary_correct/Supplementary_TableS2_sensitivity_basin_station.csv
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress, t as tdist
warnings.filterwarnings("ignore")

DATA_DIR  = "/work/home/H.Jason421/water_temp_for_publish/data/"
SCALE_DIR = DATA_DIR + "scale_trends_v1/"
META_XLS  = "/work5/H.Jason421/GEMStat_new/GEMS-Water_data_request.xls"
OUT_DIR   = "/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary_correct/"
os.makedirs(OUT_DIR, exist_ok=True)
OUT       = OUT_DIR + "Supplementary_TableS2_sensitivity_basin_station.csv"
MIN_YEARS = 5

SEASONS = {"Boreal Summer": {"NH": [6,7,8],  "SH": [12,1,2]},
           "Boreal Winter": {"NH": [12,1,2], "SH": [6,7,8]}}

# Display names (consistent with the existing table)
WAT_NAME = {"GEM": "GEMStat TR",       "Dyn": "DynWat TR SUB"}
AIR_NAME = {"T2M": "ERA5 TA SUB", "HadISD": "HadISD TA SUB", "Skt": "ERA5 TS SUB"}
AIRS = ["T2M", "HadISD", "Skt"]
# Order: water (GEM first → Dyn) × air (T2M/HadISD/Skt) × season (summer/winter), aligned with the existing table
COMBOS = [(wat, air, se) for wat in ["GEM", "Dyn"]
          for air in AIRS for se in ["Boreal Summer", "Boreal Winter"]]


def fit(air_vals, water_vals):
    """x=TA y=TR; Returns β/CI/p/R²/n。"""
    x = np.asarray(air_vals, float); y = np.asarray(water_vals, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = len(x)
    if n < 5 or np.ptp(x) == 0:
        return None
    sl, ic, r, p, se = linregress(x, y)
    ci = tdist.ppf(0.975, n-2) * se
    return dict(beta=sl, lo=sl-ci, hi=sl+ci, p=p, r2=r**2, n=n)


# ═══════════════ Station scale (same as air_sensitivity) ═══════════════
print("Read station trends（219 QC）...", flush=True)
STATION_FILES = {"T2M": f"ss_era5_t2m_obs_trends_{MIN_YEARS}y.csv",
                 "HadISD": f"ss_hadisd_obs_trends_{MIN_YEARS}y.csv",
                 "Skt": f"ss_era5_skt_obs_trends_{MIN_YEARS}y.csv",
                 "GEM": f"ss_gemstat_trends_{MIN_YEARS}y.csv",
                 "Dyn": f"ss_dynwat_obs_trends_{MIN_YEARS}y.csv"}

def station_seasonal(fname):
    df = pd.read_csv(DATA_DIR + fname); df["station_id"] = df["station_id"].astype(str)
    rows = []
    for sid, grp in df.groupby("station_id"):
        lat = float(grp["latitude"].iloc[0])
        for sea, hemi in SEASONS.items():
            mo = hemi["NH"] if lat > 0 else hemi["SH"]
            sub = grp[grp["month"].isin(mo)].dropna(subset=["trend_dec"])
            if sub.empty: continue
            rows.append({"key": sid, "season": sea, "trend": sub["trend_dec"].mean()})
    return pd.DataFrame(rows)

station_tr = {ds: station_seasonal(f) for ds, f in STATION_FILES.items()}

def station_fit(season, wat, air):
    a = station_tr[air]; w = station_tr[wat]
    aa = a[a["season"] == season][["key", "trend"]].rename(columns={"trend": "air"})
    ww = w[w["season"] == season][["key", "trend"]].rename(columns={"trend": "wat"})
    m = aa.merge(ww, on="key").dropna()
    if len(m) < 5: return None
    return fit(m["air"].values, m["wat"].values)


# ═══════════════ Basin scale (same as basin_scatter)═══════════════
print("Read metadata (basin latitudes)...", flush=True)
xl = pd.ExcelFile(META_XLS)
meta = pd.read_excel(xl, sheet_name="Station_Metadata")
meta = meta.rename(columns={"GEMS Station Number": "station_id", "Water Type": "water_type",
                            "Main Basin": "basin", "Latitude": "lat", "Longitude": "lon"})
meta = meta[meta["water_type"] == "River station"].dropna(subset=["lat", "lon"]).drop_duplicates("station_id")
BLAT = meta.dropna(subset=["basin"]).groupby("basin")["lat"].median().to_dict()

print("Read basin trends (weighted)..", flush=True)
BASIN_FILES = {"T2M": "era5_t2m_basin_sub.csv", "HadISD": "hadisd_basin_sub.csv",
               "Skt": "era5_skt_basin_sub.csv", "GEM": "gemstat_basin.csv",
               "Dyn": "dynwat_basin_sub.csv"}

def basin_seasonal(fname, tcol="trend_per_decade"):
    df = pd.read_csv(SCALE_DIR + fname); rows = []
    for basin, grp in df.groupby("basin"):
        lat = BLAT.get(basin, 45)
        for sea, hemi in SEASONS.items():
            mo = hemi["NH"] if lat >= 0 else hemi["SH"]
            sub = grp[grp["month"].isin(mo)]
            if sub.empty: continue
            rows.append({"basin": basin, "season": sea, "trend": sub[tcol].mean()})
    return pd.DataFrame(rows)

basin_tr = {ds: basin_seasonal(f) for ds, f in BASIN_FILES.items()}

def basin_fit(season, wat, air):
    a = basin_tr[air]; w = basin_tr[wat]
    aa = a[a["season"] == season][["basin", "trend"]].rename(columns={"trend": "air"})
    ww = w[w["season"] == season][["basin", "trend"]].rename(columns={"trend": "wat"})
    m = aa.merge(ww, on="basin").dropna()
    if len(m) < 5: return None
    return fit(m["air"].values, m["wat"].values)


# ═══════════════ Assemble table ═══════════════
def fmt(f):
    """β (95% CI) format, with significance asterisk."""
    if f is None: return None
    star = "*" if f["p"] < 0.05 else ""
    return (f'{f["beta"]:.2f}{star} ({f["lo"]:.2f}, {f["hi"]:.2f})',
            f'{f["p"]:.3f}' if f["p"] >= 0.001 else "<0.001",
            f'{f["r2"]:.2f}', f['n'])

records = []
seas_short = {"Boreal Summer": "Summer", "Boreal Winter": "Winter"}
for wat, air, se in COMBOS:
    for scale, fitter in [("Basin", basin_fit), ("Station", station_fit)]:
        f = fitter(se, wat, air)
        row = {"River dataset": WAT_NAME[wat], "Air reference": AIR_NAME[air],
               "Season": seas_short[se], "Scale": scale}
        if f is None:
            row.update({"β (95% CI)": "n/a", "p": "", "R²": "", "n": 0})
        else:
            b, p, r2, n = fmt(f)
            row.update({"β (95% CI)": b, "p": p, "R²": r2, "n": n})
        records.append(row)

tab = pd.DataFrame(records)
tab.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}\n")
print(tab.to_string(index=False))
