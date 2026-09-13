#!/usr/bin/env python3
# ============================================================
# Supp_FigS5_basin_choropleth.py (v2:correct join and correspond the row of GEMStat in Main text heatmap)
#   Apply the GEMStat TR basin trend from the bottom row of the main text heatmap to the world map.
#   Join: gemstat_basin(name) -> basin_to_hybas_mapping(HYBAS_ID) -> HydroBASINS L3 polygon.
#   Base figures: All the basin L3 colored light gray and the GEMStat basins with trends colored with RdBu (Same colors levels as heatmap). 
#   Seasons/ Significat/ hemisphere follows the definition of heatmap (load_basin_seasonal).
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

try:
    import colormaps as cmaps
    CMAP_BASE = cmaps.rdbu.reversed()
except Exception:
    CMAP_BASE = plt.cm.RdBu_r

# -- CONFIG (Follow the path in main text heatmap) --------------------------
DATA_DIR   = "/work/home/H.Jason421/water_temp_for_publish/data/"
SCALE_DIR  = DATA_DIR + "scale_trends_v1/"
META_XLS   = "/work5/H.Jason421/GEMStat_new/GEMS-Water_data_request.xls"
TREND_CSV  = SCALE_DIR + "gemstat_basin.csv"
MAP_CSV    = DATA_DIR + "basin_to_hybas_mapping.csv"
ABBREV_CSV = DATA_DIR + "basin_abbrev_v1.csv"
SHP        = "/work/home/H.Jason421/shapefiles/hybas_global_lev03.shp"
SHP_KEY    = "HYBAS_ID"
FIG_DIR    = "/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary_correct/"
OUT        = FIG_DIR + "Supplementary_FigS5_basin_choropleth.png"

TREND_COL = "trend_per_decade"
HVMAX     = 1.5
N_LEVELS  = 6
LABEL_FS  = 3.5
PROJ      = "ESRI:54030"
DPI       = 400

SEASONS = {"Boreal Summer": {"NH": [6, 7, 8], "SH": [12, 1, 2]},
           "Boreal Winter": {"NH": [12, 1, 2], "SH": [6, 7, 8]}}


def load_basin_seasonal(trend_csv, blat):
    df = pd.read_csv(trend_csv)
    out = {}
    for basin, grp in df.groupby("basin"):
        lat = blat.get(basin, 45)
        for sea, hemi in SEASONS.items():
            mo = hemi["NH"] if lat >= 0 else hemi["SH"]
            sub = grp[grp["month"].isin(mo)]
            if len(sub):
                out[(str(basin), sea)] = (
                    float(sub[TREND_COL].mean()),
                    bool((sub["p_value"] < 0.05).any()) if "p_value" in sub.columns else False,
                )
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    meta = pd.read_excel(pd.ExcelFile(META_XLS), sheet_name="Station_Metadata")
    meta = meta.rename(columns={"Main Basin": "basin", "Latitude": "lat", "Water Type": "wt"})
    meta = meta[meta["wt"] == "River station"].dropna(subset=["lat", "basin"])
    BLAT = meta.groupby("basin")["lat"].median().to_dict()

    seas = load_basin_seasonal(TREND_CSV, BLAT)

    mp = pd.read_csv(MAP_CSV)
    mp["HYBAS_ID"] = mp["HYBAS_ID"].astype(float).astype("int64")
    name2hyb = dict(zip(mp["gemstat_basin"].astype(str), mp["HYBAS_ID"]))
    hyb2name = {v: k for k, v in name2hyb.items()}

    gdf = gpd.read_file(SHP)[[SHP_KEY, "geometry"]].rename(columns={SHP_KEY: "HYBAS_ID"})
    gdf["HYBAS_ID"] = gdf["HYBAS_ID"].astype("int64")
    gdf = gdf.to_crs(PROJ)
    gdf["rep"] = gdf.geometry.representative_point()

    abbr = {}
    if ABBREV_CSV:
        a = pd.read_csv(ABBREV_CSV)
        abbr = dict(zip(a["basin"].astype(str), a["abbrev"].astype(str)))

    bounds = np.linspace(-HVMAX, HVMAX, N_LEVELS + 1)
    cmap = mcolors.ListedColormap(CMAP_BASE(np.linspace(0, 1, N_LEVELS)))
    cmap.set_over(CMAP_BASE(1.0)); cmap.set_under(CMAP_BASE(0.0))
    norm = mcolors.BoundaryNorm(bounds, N_LEVELS, clip=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), facecolor="white")
    for ax, season in zip(axes, ["Boreal Summer", "Boreal Winter"]):
        gdf.plot(ax=ax, facecolor="#eeeeee", edgecolor="#cfcfcf", linewidth=0.15, zorder=1)

        vals, sig = [], []
        for _, row in gdf.iterrows():
            nm = hyb2name.get(int(row["HYBAS_ID"]))
            rec = seas.get((nm, season)) if nm else None
            vals.append(rec[0] if rec else np.nan)
            sig.append(bool(rec[1]) if rec else False)
        gdf_c = gdf.assign(_v=vals, _sig=sig).dropna(subset=["_v"])
        gdf_c.plot(ax=ax, column="_v", cmap=cmap, norm=norm,
                   edgecolor="#7a7a7a", linewidth=0.2, zorder=2)
        sg = gdf_c[gdf_c["_sig"]]
        if len(sg):
            sg.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=3)

        if abbr:
            for _, row in gdf_c.iterrows():
                nm = hyb2name.get(int(row["HYBAS_ID"]))
                lab = abbr.get(nm)
                if lab:
                    ax.annotate(lab, (row["rep"].x, row["rep"].y), ha="center", va="center",
                                fontsize=LABEL_FS, color="#222222", zorder=4)

        ax.set_title(season, fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.09, hspace=0.06)
    cax = fig.add_axes([0.30, 0.05, 0.40, 0.018])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", boundaries=bounds, ticks=bounds, extend="both")
    cb.set_label("GEMStat river water-temperature trend (°C/decade)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.savefig(OUT, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved:", OUT, "| colored basins (summer):",
          sum(1 for k in seas if k[1] == "Boreal Summer"))


if __name__ == "__main__":
    main()
