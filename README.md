# Current Observations Do Not Establish Global River Warming
Analysis and figure-generation code for the manuscript *Current Observations Do Not Establish Global River Warming*

The scripts provided here compute long-term temperature trends for observed river temperatures and atmospheric reference data, 
aggregate them across different spatial scales, and reproduce all figures and tables presented in the main text and supplementary materials.

## Repository Structure
- `analysis/`: Scripts for upstream trend calculations. Generates intermediate CSV files used by the figure scripts.
- `figures_main/`: Code to generate Main Text Figures 1–4.
- `figures_supp/`: Code to generate Supplementary Figures and Tables.
- `paper_numbers/`: Utility scripts that output the exact numerical values cited in the abstract, results, and figure captions.

## Terminology & Naming Conventions
### Temperature Variables:
- `TR` = River Temperature
- `TA` = Air Temperature
- `TS` = Skin Temperature
### Sampling Methods:
- `ALL` = Full available record.
- `SUB` = Matched sampling (restricted only to months with co-located river-temperature observations).

*Note: Trends are expressed in °C per decade. Seasons follow the boreal convention based on station/basin latitude (e.g., Boreal Summer = JJA in the Northern Hemisphere, DJF in the Southern Hemisphere).*

## Data Requirements
The input datasets are not included in this repository and must be downloaded directly from their respective providers:
- GEMStat: Observed river water temperature (GEMS/Water Programme).
- ERA5: 2m air temperature (t2m) and skin temperature (skt) (Copernicus Climate Change Service).
- HadISD: Station air temperature (Met Office Hadley Centre).
- DynWat: Modelled water temperature (monthly, 1981–2014).
  
*Important: Before running the pipeline, you must update the file paths at the top of each script to point to your local data directories.*

## Environment Setup
The code requires **Python 3.11.** Install the required dependencies:
`pip install numpy pandas scipy matplotlib xarray geopandas Pillow colormaps`

*(Note: `geopandas` is used for basin/shapefile handling, `Pillow` for figure compositing, and `colormaps` for heatmap generation).*

## Reproduction Workflow
The workflow relies on intermediate CSV files, so you must run the data processing pipeline before generating figures.

### 1. Data Processing
Calculate station-scale trends (used in Figure 4):
`python data_processing/calc_station_scale.py --min_years 5`
`python data_processing/calc_make_ss_dynwat_full_v2.py --min_years 5`

Calculate aggregated trends at basin, continental, and global scales (used in Figures 2 and 3):
`python data_processing/calc_scale_trends_v1.py

**Parameters:**
- `--min_years`: Sets the minimum number of valid years required per calendar month. The default (`5`) reproduces the main-text results. For robustness checks, use `8` or `10`. Outputs for these checks will be automatically suffixed with `_8y` or `_10y` to prevent overwriting the main results.
- `--agg`: Available in `calc_scale_trends_mean.py`. Switches cross-station aggregation from `median` (default) to `mean`. This only affects cross-station aggregation; within-station monthly baselines always use the median.

*(Optional)* Run the remaining scripts in `data_processing/` to prepare individual atmospheric/modelled inputs and perform area-weighted basin significance analysis.

### 2. Main-Text Figures
Run these scripts to generate the primary figures. Figure 4 requires a composite step.
- `python figures_main/Main_Text_Figure1_map_v1.3.py`            # Fig 1: Station map
- `python figures_main/Main_Text_Figure2_time_series_v1.4.py`    # Fig 2: Regional time series
- `python figures_main/Main_Text_Figure3_heatmap_v1.2.py`        # Fig 3: Basin heatmap
- `python figures_main/Main_Text_Figure4_station_violin_v1.4.py` # Fig 4a
- `python figures_main/Main_Text_Figure4_air_sensitivity_v1.6.py` # Fig 4b
- `python figures_main/Main_Text_Figure4_combine_v1.0.py`        # Assembles Fig 4

*Fig 4 requires generating components first, then compositing:*

### 3. Supplementary Figures & Tables
- Scripts for supplementary materials are located in `figures_supp/` and follow the `Supp_FigSN_*` and `Supp_table_*` naming conventions.
- Scripts for threshold/mean-based robustness checks accept the same `--min_years` and `--agg flags` as the processing pipeline.
- Dependency note: Ensure `Supp_Prep_for_S1516_hadisd_common.py remains` in the same directory as the HadISD validation figure scripts, as it acts as a shared import module.

### 4. Extracting Reported Values
To verify the exact numbers cited in the paper without modifying any files or recomputing trends, run:

- `python paper_numbers/print_region_trends_v1.py`
- `python paper_numbers/print_paper_numbers_v2.py`
- `python paper_numbers/count_gemstat_basin_signs.py`
