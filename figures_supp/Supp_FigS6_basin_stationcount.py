#!/usr/bin/env python3
# ============================================================
# Supp_FigS6_basin_stationcount.py
#   Take every station numbers of GEMStat basin into global map (single map, one color, log color levels).
#   join / projection / shapefile all follow Supp_FigS5_basin_choropleth.py and make sure same as the map in main text.
#   Difference with trend map: coloered value = n_stations (Regardless of season), single color LogNorm, single map, no significnt black broader.
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

# -- CONFIG (Follow the path in main text heatmap) --------------------------
DATA_DIR   = "/work/home/H.Jason421/water_temp_for_publish/data/"
SCALE_DIR  = DATA_DIR + "scale_trends_v1/"
TREND_CSV  = SCALE_DIR + "gemstat_basin.csv"          # Include n_stations
MAP_CSV    = DATA_DIR + "basin_to_hybas_mapping.csv"
ABBREV_CSV = DATA_DIR + "basin_abbrev_v1.csv"
SHP        = "/work/home/H.Jason421/shapefiles/hybas_global_lev03.shp"
SHP_KEY    = "HYBAS_ID"
FIG_DIR    = "/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary_correct/"
OUT        = FIG_DIR + "Supplementary_FigS6_basin_stationcount.png"

LABEL_FS  = 3.5
PROJ      = "ESRI:54030"
DPI       = 400
CMAP      = plt.cm.viridis        # Monochrome continuous (can be replaced with plt.cm.YlGnBu, etc.)


def load_basin_nstations(trend_csv):
    """Station numbers in every basin (Regardless of season). gemstat_basin is monthly rows, 
    which n_stations may be a different in the same basin each month, so taking median to be present values."""
    df = pd.read_csv(trend_csv)
    return df.groupby("basin")["n_stations"].median().to_dict()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    nstn = load_basin_nstations(TREND_CSV)

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

    # Number of stations for each polygon to the basin
    vals = []
    for _, row in gdf.iterrows():
        nm = hyb2name.get(int(row["HYBAS_ID"]))
        vals.append(nstn.get(nm, np.nan) if nm else np.nan)
    gdf_c = gdf.assign(_v=vals).dropna(subset=["_v"])
    gdf_c = gdf_c[gdf_c["_v"] >= 1]        # At least one station for coloring.

    vmin = 1
    vmax = float(gdf_c["_v"].max())
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(11, 6), facecolor="white")

    # Base: Light gray of global L3 watersheds
    gdf.plot(ax=ax, facecolor="#eeeeee", edgecolor="#cfcfcf", linewidth=0.15, zorder=1)

    # Basins with stations are colored using a single log color.
    gdf_c.plot(ax=ax, column="_v", cmap=CMAP, norm=norm,
               edgecolor="#7a7a7a", linewidth=0.2, zorder=2)

    if abbr:
        for _, row in gdf_c.iterrows():
            nm = hyb2name.get(int(row["HYBAS_ID"]))
            lab = abbr.get(nm)
            if lab:
                ax.annotate(lab, (row["rep"].x, row["rep"].y), ha="center", va="center",
                            fontsize=LABEL_FS, color="#222222", zorder=4)

    ax.axis("off")

    # colorbar（log）
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.12)
    cax = fig.add_axes([0.30, 0.08, 0.40, 0.02])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_label("Number of GEMStat river stations per basin (log scale)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.savefig(OUT, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved:", OUT, "| colored basins:", len(gdf_c),
          "| max stations:", int(vmax))


if __name__ == "__main__":
    main()
