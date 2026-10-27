# A third of the world's threatened land vertebrates exposed to tropical urbanization

Rakibul Ahasan<sup>1</sup>, Burak Güneralp<sup>2</sup>, Shawan Chowdhury<sup>3</sup>

<sup>1</sup> Esri, Redlands, CA, USA<br>
<sup>2</sup> Department of Geography, Texas A&M University, College Station, TX, USA<br>
<sup>3</sup> School of Biological Sciences, Monash University, Melbourne, VIC, Australia

This repository holds the code for the paper above, which delineates 7,229 tropical urban agglomerations from 30 m impervious-surface dynamics and measures the expert-mapped range of threatened and range-restricted land vertebrates that their expansion converted between 2000 and 2020. It includes the Python code of the measurement and the analysis, the scripts that draw the figures of the paper and its Supplementary Information with the figures they draw, and the instructions for obtaining the source data.

The code is archived on Zenodo (DOI to be added at deposit). Neither this repository nor the Zenodo deposit redistributes the source data, which are available from their providers under their own terms (see Source data).

## Source data

The analysis reads the products below. None is included here. Download each from its provider, accept the provider's terms, and place the files at the paths given, relative to the repository folder. The environment variables in the last column point the code to copies held elsewhere.

| Product | Used for | Source | Path in the repository | Variable |
|---|---|---|---|---|
| GISD30, global 30 m impervious-surface dynamics, 1985 to 2020<sup>1</sup> | Urban extent at each epoch, agglomeration delineation, land added | https://doi.org/10.5281/zenodo.5220816 (twelve RAR archives; extract the 5° GeoTIFF tiles) | data/source/gisd30/tiles/GISD30_1985-2020_<tile>.tif | TROPICS_GISD30 |
| GLC_FCS30D, global 30 m land cover, 1985 to 2022<sup>2</sup> | Land cover the expansion replaced | https://zenodo.org/records/8239305 (read over the network by tropics_form.py; a local copy of a longitude band can be placed in data/source/GLC_FCS30D_19852022maps_E90-E95) | none | TROPICS_DOWNLOADS |
| IUCN Red List spatial data, version 2026-1<sup>3</sup>: terrestrial mammals, amphibians and reptiles | Species ranges and Red List categories | https://www.iucnredlist.org/resources/spatial-data-download | data/source/MAMMALS_TERRESTRIAL_ONLY/MAMMALS_TERRESTRIAL_ONLY.shp, data/source/AMPHIBIANS/AMPHIBIANS_PART1.shp and _PART2.shp, data/source/REPTILES/REPTILES_PART1.shp and _PART2.shp | TROPICS_DOWNLOADS |
| Bird species distribution maps of the world, 2025 release, with the HBW and BirdLife International Digital Checklist of the Birds of the World, version 10<sup>4</sup> | Bird ranges and Red List categories | http://datazone.birdlife.org | data/source/species/BOTW_2025.gpkg (layer all_species) and data/source/species/Handbook of the Birds of the World and BirdLife International Digital Checklist of the Birds of the World_Version_10.xlsx | TROPICS_DOWNLOADS |
| Natural Earth 1:10m admin-0 countries | Country of each agglomeration, map outlines | https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/ | data/source/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp | TROPICS_DOWNLOADS |
| RESOLVE Ecoregions 2017<sup>5</sup> | Biome and biogeographic realm of each agglomeration | https://ecoregions.appspot.com/ | data/source/ecoregions/Ecoregions2017.shp | TROPICS_DOWNLOADS |
| Natural Earth 1:10m populated places | Labels of named agglomerations in the maps (place_names.py writes the place table data/raw/tropical_places.csv from it) | https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/ | data/source/ne_10m_populated_places/ne_10m_populated_places.shp | TROPICS_DOWNLOADS |

The folder data is not part of the repository. The code writes its intermediate files to data/interim and data/processed, its results to outputs/tables, the Supplementary Tables to manuscript/tables and the figures to the folder of each figure script. TROPICS_PROJECT moves all of these to another folder, and TROPICS_SCRATCH moves the temporary files of the measurement.

## Analysis

The folder analysis holds the code that turns the source data into the numbers, tables and figures of the paper. tropics_config.py holds the paths, the extent of the tropical band, the codes of the source products and the constants of the analysis that every module shares. The modules import one another by name, so the folder is run as one unit.

| Stage | Modules |
|---|---|
| Measurement | place_names.py (place names for the map labels), tropics_grid.py (cell areas on the WGS84 ellipsoid), tropics_urban.py (GISD30 mosaic on a 30 arcsec lattice), tropics_cities.py (delineation by the City Clustering Algorithm), tropics_measure.py (30 m measurement inside each footprint), tropics_species.py and tropics_rangecache.py (range ingest and a local cache of the tropical part of every range), tropics_cost.py (range converted by each agglomeration), tropics_flags.py (Red List category, taxon and range size of each species), tropics_assemble.py (agglomeration table), tropics_setting.py and setting_rebuild.py (threatened species within 50 km of the year 2000 footprint), birdlife_union.py (bird ranges as spatial unions), tropics_form.py and tropics_habpass.py (expansion form and land cover replaced), tropics_surfaces.py (display surfaces of the maps) |
| Sensitivity of the measurement | tropics_delin.py and delineation_fixed.py (delineation rules), tropics_radius.py (radius of the setting covariate), resolution_test.py and resolution_resume.py (lattice and boundary rule), species_boundary.py (boundary rule and the share of one range converted), tropics_crosscheck.py (setting covariate from disjoint taxonomic groups) |
| Five-year periods | tropics_temporal.py (urban land at 2000, 2005, 2010, 2015 and 2020 and the range each period converted), temporal_analysis.py, temporal_size.py, temporal_subregion.py and temporal_species.py (period totals, size distribution, species accrual by period and by sub-region), temporal_verify.py (checks) |
| Models and statistics | tropics_scaling.py (estimation of the size exponent and the spatial diagnostic), tropics_results.py, tropics_numbers.py, tropics_refit.py, tropics_analysis.py, tropics_models.py and tropics_sizevars.py (primary specification, response definitions, quantile regression, variance decomposition, alternative measures of size), tropics_cluster.py (typology of agglomerations), tropics_sensitivity.py, r2_components.py (marginal and conditional R²), spatial_cv_blocks.py, spatial_cv_run.py and validate_setting.py (buffered spatial block cross-validation), landcover_disagreement.py, drivers.py, delineation_note.py, threshold_isolate.py, setting_definition_compare.py, species_ledger.py, species_concentration.py, weighting_vs_sample.py, bird_union_effect.py, bird_sensitivity.py, region_overlap.py, region_search.py, recency_numbers.py (timing of the impervious surface and its relation to rarity-weighted exposure) |
| Reallocation of growth within countries | tropics_counterfactual.py and tropics_cfrobust.py (allocation of each country's added land by the power of year 2000 extent), cf_release.py, cf_run.py, cf_bands.py, cf_influence.py, cf_boundary_check.py and cf_fix.py (country bootstrap, response and realm bands, leave-one-country-out influence) |
| Supplementary Tables | tropics_sitables.py, si_print.py, si_tables_two_part.py, table_s5_quantiles.py |
| Figures | tropics_style.py, tropics_frame.py, tropics_robinson.py, tropics_basemap.py, tropics_bandmaps.py, tropics_panels.py, tropics_main_figures.py, tropics_si_figures.py, tropics_citymaps.py, tropics_speciesmaps.py, render_figures.py, region_panels.py, two_part_figs.py, si_spatialcv.py and the fig_*.py modules |
| Checks | tropics_verify.py |

run_pipeline.py runs the stages in order. From the repository folder,

    python analysis/run_pipeline.py measurement
    python analysis/run_pipeline.py temporal
    python analysis/run_pipeline.py analysis
    python analysis/run_pipeline.py tables
    python analysis/run_pipeline.py figures
    python analysis/run_pipeline.py verify

or `python analysis/run_pipeline.py all` for every stage. The measurement and temporal stages read the full range files and the 30 m rasters of the tropical band and run for many hours. The steps that take a time budget resume from their last completed tile or species when they are started again.

## Figures

The folder figures holds the script behind each figure of the paper and its Supplementary Information, one folder per figure, with the figure it draws as PNG and PDF. The scripts read the outputs of the analysis stages. Run a script from the repository folder without arguments, for example

    python figures/Fig3_size_growth_setting/make_fig3_size_growth_setting.py

Each script writes the figure to its own folder, or to the folder given by --out. The scripts do not overwrite an existing figure and add _r1, _r2 and so on to the name of a new one.

| Figure | Content | Script (in figures/) |
|---|---|---|
| Figure 1 | Tropical urban expansion, 2000 to 2020 | Fig1_urban_expansion/make_fig1_urban_expansion.py |
| Figure 2 | Exposure intensity and its spread | Fig2_exposure_intensity/make_fig2_exposure_intensity.py |
| Figure 3 | Exposure intensity by size, growth and setting | Fig3_size_growth_setting/make_fig3_size_growth_setting.py |
| Figure 4 | Reallocation of urban growth within countries | Fig4_reallocation/make_fig4_reallocation.py |
| Figure 5 | Range rarity, the size relationship and three regional windows | Fig5_range_rarity/make_fig5_range_rarity.py |
| Figure 6 | Urban growth and accumulating range conversion, 2000 to 2020 | Fig6_temporal_accrual/make_fig6_temporal_accrual.py |
| Supplementary Fig. 1 | Delineated agglomerations | SFig01_agglomerations/make_sfig01_agglomerations.py |
| Supplementary Fig. 2 | Agglomerations matched on 2000 size and contrasted in exposure | SFig02_matched_pairs/make_sfig02_matched_pairs.py |
| Supplementary Fig. 3 | Occurrence and intensity of narrow-range conversion against agglomeration size | SFig03_two_part/make_sfig03_two_part.py |
| Supplementary Fig. 4 | Concentration of a species' converted range across agglomerations | SFig04_species_concentration/make_sfig04_species_concentration.py |
| Supplementary Fig. 5 | Threatened species reached by taxon and Red List category | SFig05_threatened_taxa/make_sfig05_threatened_taxa.py |
| Supplementary Fig. 6 | Range-restricted species and new impervious surface in four regions | SFig06_narrow_range_regions/make_sfig06_narrow_range_regions.py |
| Supplementary Fig. 7 | Impervious surface by 2000 against surface added since | SFig07_surface_by_period/make_sfig07_surface_by_period.py |
| Supplementary Fig. 8 | The 2020 impervious surface split by period | SFig08_surface_split/make_sfig08_surface_split.py |
| Supplementary Fig. 9 | When the agglomerated surface was added | SFig09_surface_timing/make_sfig09_surface_timing.py |
| Supplementary Fig. 10 | The size distribution of agglomerations through time | SFig10_size_distribution/make_sfig10_size_distribution.py |
| Supplementary Fig. 11 | Agglomeration size and the period in which its surface was added | SFig11_size_and_period/make_sfig11_size_and_period.py |
| Supplementary Fig. 12 | Exposure intensity by country and by when an agglomeration formed | SFig12_country_intensity/make_sfig12_country_intensity.py |
| Supplementary Fig. 13 | Recency of urban surface and the rarity of the ranges it converts | SFig13_recency_rarity/make_sfig13_recency_rarity.py |

The same list is in figures/figure_scripts.csv. The maps are drawn in the Robinson projection.

## Environment

The code was run with Python 3.10, numpy 2.2, pandas 2.3, scipy 1.15, matplotlib 3.10, rasterio 1.4 (GDAL 3.10), pyogrio 0.13, pyproj 3.7, shapely 2.1, scikit-learn 1.7, statsmodels 0.15 and openpyxl. environment.yml creates a conda environment with these versions (`conda env create -f environment.yml`), and requirements.txt lists the packages for pip.

## Sources

The source products keep the licenses and terms of their providers. GISD30 and GLC_FCS30D are distributed on Zenodo by their authors. The IUCN Red List spatial data and the BirdLife International distribution maps are distributed by IUCN and BirdLife International under their terms of use, which do not permit redistribution. Natural Earth is in the public domain. RESOLVE Ecoregions 2017 is distributed by RESOLVE.

## References

<sup>1</sup> Zhang, X., Liu, L., Zhao, T., Gao, Y., Chen, X., & Mi, J. (2022). GISD30: global 30 m impervious-surface dynamic dataset from 1985 to 2020 using time-series Landsat imagery on the Google Earth Engine platform. *Earth System Science Data, 14*, 1831–1856. https://doi.org/10.5194/essd-14-1831-2022

<sup>2</sup> Zhang, X., et al. (2024). GLC_FCS30D: the first global 30 m land-cover dynamics monitoring product with a fine classification system for the period from 1985 to 2022 generated using dense-time-series Landsat imagery and the continuous change-detection method. *Earth System Science Data, 16*, 1353–1381. https://doi.org/10.5194/essd-16-1353-2024

<sup>3</sup> IUCN (2026). *The IUCN Red List of Threatened Species*, spatial data, version 2026-1. https://www.iucnredlist.org

<sup>4</sup> BirdLife International and Handbook of the Birds of the World (2025). *Bird species distribution maps of the world*, 2025 release, with the HBW and BirdLife International Digital Checklist of the Birds of the World, version 10. http://datazone.birdlife.org

<sup>5</sup> Dinerstein, E., et al. (2017). An ecoregion-based approach to protecting half the terrestrial realm. *BioScience, 67*(6), 534–545. https://doi.org/10.1093/biosci/bix014

## License

The code and figures are released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (see LICENSE). The source data are not covered by this license.
