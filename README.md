# The geography of threatened species exposure to tropical urban growth

Code and figure data for a pan-tropical assessment of how urban expansion between 2000 and 2020 converted the ranges of threatened and range-restricted land vertebrates. The analysis delineates 7,229 urban agglomerations from 30 m impervious-surface dynamics, measures the expert-mapped range their growth converted, and relates that exposure to agglomeration size, growth, setting and timing. The source products are not redistributed here (see Source data).

**Keywords:** urban expansion, tropics, biodiversity, threatened species, range-restricted species, impervious surface, urban agglomerations

## Analyses

| Folder | Contents |
|---|---|
| common | Shared settings and helpers. tropics_config.py holds the paths, the extent of the tropical band, the codes of the source products and the constants of the analysis. tropics_grid.py (cell areas on the WGS84 ellipsoid), tropics_flags.py (Red List category, taxon and range size of each species), tropics_style.py, tropics_frame.py, tropics_robinson.py and tropics_basemap.py (map projection, base maps and figure style). |
| measurement | From the source products to one table of agglomerations. place_names.py (place names for the map labels), tropics_urban.py (GISD30 mosaic on a 30 arcsec lattice), tropics_cities.py (delineation by the City Clustering Algorithm), tropics_measure.py (30 m measurement inside each footprint), tropics_species.py and tropics_rangecache.py (range ingest and a local cache of the tropical part of every range at full source resolution), tropics_cost.py (global range areas and the range converted by each agglomeration), tropics_assemble.py and tropics_results.py (agglomeration table), tropics_setting.py and setting_rebuild.py (threatened species within 50 km of the year 2000 footprint), tropics_form.py and tropics_habpass.py (expansion form and land cover replaced), tropics_surfaces.py (gridded surfaces of the maps), tropics_verify.py (checks). |
| sensitivity | Sensitivity of the measurement. tropics_delin.py, delineation_fixed.py, delineation_note.py and threshold_isolate.py (delineation rules), tropics_radius.py (radius of the setting covariate), resolution_test.py and resolution_resume.py (lattice and boundary rule), species_boundary.py (boundary rule and the share of one range converted), tropics_crosscheck.py (setting covariate from disjoint taxonomic groups), setting_definition_compare.py, landcover_disagreement.py, bird_sensitivity.py, tropics_sensitivity.py, weighting_vs_sample.py and outside_agglomerations.py (range converted by impervious surface added outside the agglomerations). |
| temporal | Five-year periods. tropics_temporal.py (urban land at 2000, 2005, 2010, 2015 and 2020 and the range each period converted), temporal_analysis.py, temporal_size.py, temporal_subregion.py and temporal_species.py (period totals, size distribution, species accrual by period and by sub-region), temporal_disagreement_check.py (species reached and species above 1 percent of range converted under each treatment of the land the two land-cover products date differently), recency_numbers.py (timing of the impervious surface and its relation to rarity-weighted exposure), temporal_verify.py (checks). |
| models | Scaling models and statistics. tropics_scaling.py (estimation of the size exponent and the spatial diagnostic), tropics_numbers.py, tropics_refit.py, tropics_analysis.py, tropics_models.py and tropics_sizevars.py (primary specification, response definitions, quantile regression, variance decomposition, alternative measures of size), tropics_cluster.py (typology of agglomerations), r2_components.py (marginal and conditional R²), spatial_cv_blocks.py, spatial_cv_run.py and validate_setting.py (buffered spatial block cross-validation), residual_moran.py (Moran's I of the residuals with and without the spatial block component), drivers.py, species_ledger.py and species_accounting_by_class.py (species between the range assembly and the analysis pool, and by taxonomic class), species_concentration.py, region_overlap.py and region_search.py. |
| reallocation | Reallocation of growth within countries. tropics_counterfactual.py and tropics_cfrobust.py (allocation of each country's added land by the power of year 2000 extent), cf_release.py, cf_run.py, cf_bands.py, cf_influence.py, cf_boundary_check.py and cf_fix.py (country bootstrap, response and realm bands, leave-one-country-out influence), cf_pool_bootstrap.py (land pooled within realms and across the tropics before reallocation). |
| supplementary_tables | The tables of the paper. tropics_sitables.py, si_print.py, si_tables_two_part.py and table_s5_quantiles.py (formatted tables), supplementary_data.py (Supplementary Data 1 to 3 as Excel workbooks). |
| figures | The script and the output of each figure, Extended Data figure and Extended Data table of the paper (see Figures). figures/plotting holds the plotting code the scripts share, with extended_data.py drawing the panels and tables added for the Extended Data. |

run_pipeline.py runs the stages in order, from the source products to the tables, the figures and the checks. From the repository folder,

    python run_pipeline.py measurement
    python run_pipeline.py temporal
    python run_pipeline.py analysis
    python run_pipeline.py tables
    python run_pipeline.py figures
    python run_pipeline.py verify

or `python run_pipeline.py all` for every stage. One module runs on its own with `python run_pipeline.py module <name> [arguments]`, for example `python run_pipeline.py module temporal_species`. The measurement and temporal stages read the full range files and the 30 m rasters of the tropical band and run for many hours. The steps that take a time budget resume from their last completed tile or species when they are started again.

The analysis writes its intermediate files to data/interim and data/processed, its results to outputs/tables, and the tables and Supplementary Data to manuscript/tables and manuscript/supplementary_data, all inside the repository folder. TROPICS_PROJECT moves these folders elsewhere, and TROPICS_SCRATCH moves the temporary files of the measurement.

## Figures

The folder figures holds the script behind each figure, Extended Data figure and Extended Data table of the paper, one folder per item, with the figure it draws as PNG (400 dpi) and PDF. The PDF keeps its text as embedded TrueType. Text is set in Arial, or Helvetica where Arial is absent, and a script stops with a message if neither font is installed. The folder data/figures holds the inputs the scripts read, so the figures can be drawn without the source products or the analysis. Run a script from the repository folder without arguments, for example

    python figures/Fig2_size_growth_setting/make_fig2_size_growth_setting.py

Each script writes the figure to its own folder, or to the folder given by --out, and adds _r1, _r2 and so on to the name rather than overwrite an existing figure. --data points a script to another copy of the figure inputs, and --pipeline draws the figure from the outputs of run_pipeline.py instead.

| Figure | Content | Script (in figures/) | Inputs (in data/figures) |
|---|---|---|---|
| Figure 1 | Threatened land vertebrates reached by tropical urban expansion, 2000 to 2020 | Fig1_species_reached/make_fig1_species_reached.py | tables/cities_cost.csv, tables/species_accounting_by_class.csv, tables/species_concentration.csv, tables/temporal, interim/city_form.csv |
| Figure 2 | Exposure intensity by size, growth and setting | Fig2_size_growth_setting/make_fig2_size_growth_setting.py | tables/cities_cost.csv, tables/plan4.json, tables/scaling_spatial.json, tables/setting_footprint_v3.csv, interim/city_form.csv |
| Figure 3 | Reallocating the same urban land among agglomerations | Fig3_reallocation/make_fig3_reallocation.py | tables/cf_influence.csv, tables/cf_pool_bootstrap.json, tables/cf_release.json, tables/cf_robust.json, tables/cities_cost.csv, interim/city_form.csv |
| Figure 4 | Urban growth and accumulating range conversion, 2000 to 2020 | Fig4_temporal_accrual/make_fig4_temporal_accrual.py | tables/cities_cost.csv, tables/temporal, interim/city_form.csv |
| Extended Data Fig. 1 | Delineated agglomerations and their size distribution through time | EDFig1_agglomerations/make_edfig1_agglomerations.py | tables/cities_cost.csv, tables/temporal, interim/city_form.csv, interim/temporal/delineation |
| Extended Data Fig. 2 | Tropical urban expansion, 2000 to 2020 | EDFig2_urban_expansion/make_edfig2_urban_expansion.py | tables/cities_cost.csv, interim/city_form.csv, interim/city_habitat.csv, interim/surfaces |
| Extended Data Fig. 3 | Threatened species reached by taxon, Red List category and boundary rule | EDFig3_threatened_taxa/make_edfig3_threatened_taxa.py | tables/species_accounting_by_class.csv, tables/species_boundary_shares.csv, tables/temporal, interim/rangecache/ranges_idx.jsonl, interim/sp_flags_merged.npz, interim/species_city_pairs.npz, interim/temporal/pairs |
| Extended Data Fig. 4 | Concentration of a species' converted range across agglomerations | EDFig4_species_concentration/make_edfig4_species_concentration.py | tables/species_concentration.csv, tables/species_concentration_strata.json |
| Extended Data Fig. 5 | Exposure intensity, its spread and its prediction in held-out regions | EDFig5_exposure_intensity/make_edfig5_exposure_intensity.py | tables/_spatial_cv_buffered.csv, tables/cities_cost.csv, tables/si/tableS5_cross_taxon.csv, interim/city_form.csv |
| Extended Data Fig. 6 | Rarity-weighted exposure, its decomposition and narrow-range species in five regions | EDFig6_range_rarity/make_edfig6_range_rarity.py | tables/cities_cost.csv, tables/region_window_overlap.json, tables/si/tableS15_objective_decomposition.csv, interim/city_form.csv, interim/surfaces |
| Extended Data Fig. 7 | Occurrence and intensity of narrow-range conversion against agglomeration size | EDFig7_two_part/make_edfig7_two_part.py | tables/narrow_two_part_B.csv, tables/narrow_two_part_contrasts_B.json, tables/narrow_two_part_meta.json, tables/narrow_two_part_validation_A.json, tables/narrow_two_part_validation_B.json |
| Extended Data Fig. 8 | Exposure intensity by country, by when an agglomeration formed and by the recency of its surface | EDFig8_country_recency/make_edfig8_country_recency.py | tables/cities_cost.csv, tables/setting_footprint_v3.csv, tables/temporal, interim/city_form.csv, interim/city_habitat.csv, interim/city_measurements.npz, naturalearth |
| Extended Data Table 1 | Primary estimates of the size exponent and setting coefficient | EDTable1_primary_estimates/make_edtable1_primary_estimates.py | tables/si/tableS2_response_definitions.csv |
| Extended Data Table 2 | Robustness of the size exponent and setting coefficient | EDTable2_robustness/make_edtable2_robustness.py | tables/si/tableS2_response_definitions.csv, tables/si/tableS3_realm_biome.csv, tables/quantile_scaling.csv, tables/si/tableS4_delineation.csv, tables/si/tableS7_size_variables.csv, tables/resolution_test_v2.csv, tables/si/tableS14_landcover_disagreement.csv, tables/setting_radius.csv, tables/si/tableS5_cross_taxon.csv |

The same list is in figures/figure_scripts.csv. The maps are drawn in the Robinson projection.

## Figure inputs

| File or folder in data/figures | Contents | Written by |
|---|---|---|
| tables/cities_cost.csv | One row per agglomeration: location, country, realm and biome, impervious extent in 2000 and 2020, land added, range converted for all species of concern, threatened and narrow-range species, rarity-weighted conversion, exposure intensity and setting richness | measurement/tropics_assemble.py, measurement/tropics_results.py |
| tables/setting_footprint_v3.csv | Setting covariate of each agglomeration, the threatened species within 50 km of the year 2000 footprint | measurement/setting_rebuild.py |
| tables/plan4.json | Fitted models of the primary specification and its alternatives | models/tropics_models.py |
| tables/scaling_spatial.json | The size exponent and its interval in the primary specification | models/tropics_numbers.py |
| tables/cf_release.json, tables/cf_robust.json | Reallocation curves and their bootstrap, response and realm bands | reallocation/cf_release.py, cf_run.py, cf_bands.py |
| tables/cf_influence.csv, tables/cf_pool_bootstrap.json | Reallocation with each country left out in turn, and with land pooled within realms and across the tropics | reallocation/cf_influence.py, cf_pool_bootstrap.py |
| tables/species_accounting_by_class.csv | Species mapped, in the analysis pool and reached, by taxonomic class | models/species_accounting_by_class.py |
| tables/narrow_two_part_B.csv, tables/narrow_two_part_contrasts_B.json | Two-part model of narrow-range conversion | models/species_concentration.py |
| tables/species_concentration.csv, tables/species_concentration_strata.json | Concentration of each species' converted range across agglomerations | models/species_concentration.py |
| tables/_spatial_cv_buffered.csv | Held-out R² of each specification under buffered spatial block cross-validation | models/spatial_cv_blocks.py, spatial_cv_run.py |
| tables/si/tableS5_cross_taxon.csv | Setting measured on one taxonomic group and its prediction of the exposure of the other two | sensitivity/tropics_crosscheck.py |
| tables/narrow_two_part_meta.json, tables/narrow_two_part_validation_A.json, tables/narrow_two_part_validation_B.json | Coefficients of the two-part model and its validation on held-out blocks | models/species_concentration.py |
| tables/si/tableS15_objective_decomposition.csv | Unweighted, rarity-weighted and mean-range exponents by species subset and region | models/tropics_models.py |
| tables/region_window_overlap.json | New impervious surface over narrow-range species in each map window | models/region_overlap.py |
| tables/si/tableS2_response_definitions.csv, tables/si/tableS3_realm_biome.csv, tables/quantile_scaling.csv, tables/si/tableS4_delineation.csv, tables/si/tableS7_size_variables.csv, tables/resolution_test_v2.csv, tables/si/tableS14_landcover_disagreement.csv, tables/setting_radius.csv | The fits summarized in Extended Data Tables 1 and 2 | models/tropics_models.py, tropics_sizevars.py, sensitivity/tropics_delin.py, resolution_test.py, landcover_disagreement.py, tropics_radius.py |
| tables/temporal | Period totals, size classes, transitions, realms, sub-regions, the agglomeration-by-period table (temporal_city_periods.csv.gz), threatened species reached by agglomeration, region and epoch, and the threatened range each agglomeration converted by 2005 and afterward | temporal/temporal_analysis.py, temporal_size.py, temporal_subregion.py, temporal_species.py, figures/plotting/fig_threat_taxa.py, fig_headline.py, fig_deepening.py |
| interim/city_form.csv, interim/city_habitat.csv | Expansion form (infill, edge, leapfrog) and land cover replaced, per agglomeration | measurement/tropics_form.py |
| interim/city_measurements.npz | Impervious area of each agglomeration by epoch | measurement/tropics_cities.py, measurement/tropics_measure.py |
| interim/species_city_pairs.npz | Range converted for each species and agglomeration, in km² | measurement/tropics_cost.py |
| interim/sp_flags_merged.npz, interim/rangecache/ranges_idx.jsonl | Red List category, taxon, scientific name and global range area of each species (no geometry) | common/tropics_flags.py, measurement/tropics_rangecache.py |
| interim/surfaces | Impervious surface at 0.05° and richness of threatened and narrow-range species at 0.1° | measurement/tropics_surfaces.py |
| interim/temporal | The agglomerations of 2000, 2005, 2010, 2015 and 2020 (delineation) and the range converted for each species, agglomeration and period (pairs) | temporal/tropics_temporal.py |
| naturalearth | Natural Earth 1:10m admin-0 countries, version 5.1.1. The scripts write base-map caches beside the files in data/figures/interim on the first run. | Natural Earth |

## Supplementary Data

The folder data/supplementary_data holds the Supplementary Data of the paper, written by supplementary_tables/supplementary_data.py from the outputs of the analysis.

| File | Contents |
|---|---|
| Supplementary_Data_1.xlsx | The per-agglomeration analysis table, one row per agglomeration (7,229 rows), with a second sheet defining each column. |
| Supplementary_Data_2.xlsx | Extended Data Tables 1 and 2 and every fit summarized in Extended Data Table 2, the tables behind the panels of Extended Data Figs. 3 to 7, the concentration of each species' converted range across agglomerations with its summary for threatened species, the reallocation with each country left out, with land pooled and with capped allocations (Fig. 3e and 3f and Supplementary Note 2), and the setting buffer and residual Moran's I checks, one sheet per table. |
| Supplementary_Data_3.xlsx | The tables of the temporal evaluation, one sheet per table: totals by five-year period, the size exponent and setting coefficient fitted in each period, agglomerations at each epoch, the agglomeration-by-period table, exposure by class and period, the size distribution through time, species reached and change in intensity by sub-region, the threatened range each agglomeration converted by 2005 and afterward (Fig. 4c), the counts under each treatment of the land the two land-cover products date differently, the impervious surface of the tropical band, and the species reached and range converted by impervious surface added outside the agglomerations. |
Each workbook opens on a README sheet that describes its sheets.

## Figure source data

The workbook data/workbook/Source_Data_Figures.xlsx holds the values plotted in every figure and Extended Data figure of the paper, one sheet per figure (Fig1 to Fig4, EDFig1 to EDFig8) and one row per plotted value. Its README sheet defines the columns, its Contents sheet lists the panels of each figure with their axes and the number of values, and its Inputs sheet lists the files in data/figures each figure reads. Map panels give the longitude, latitude and agglomeration identifier of each marker. Surfaces drawn as images are listed by their size, and their values are in the gridded files named in Inputs. figures/make_workbook.py writes the workbook by drawing each figure from data/figures and reading the values from the drawing:

    python figures/make_workbook.py

## Source data

The measurement stage reads the products below. None is included in full. Download each from its provider, accept the provider's terms, and place the files at the paths given, relative to the repository folder. The environment variables in the last column point the code to copies held elsewhere.

| Product | Used for | Source | Path in the repository | Variable |
|---|---|---|---|---|
| GISD30, global 30 m impervious-surface dynamics, 1985 to 2020<sup>1</sup> | Urban extent at each epoch, agglomeration delineation, land added | https://doi.org/10.5281/zenodo.5220816 (twelve RAR archives, from which the 5° GeoTIFF tiles are extracted) | data/source/gisd30/tiles/GISD30_1985-2020_<tile>.tif | TROPICS_GISD30 |
| GLC_FCS30D, global 30 m land cover, 1985 to 2022<sup>2</sup> | Land cover the expansion replaced | https://zenodo.org/records/8239305 (read over the network by measurement/tropics_form.py, or from a local copy of a longitude band in data/source/GLC_FCS30D_19852022maps_E90-E95) | none | TROPICS_DOWNLOADS |
| IUCN Red List spatial data, version 2023-1<sup>3</sup>: terrestrial mammals, amphibians and reptiles | Species ranges and Red List categories | https://www.iucnredlist.org/resources/spatial-data-download | data/source/MAMMALS_TERRESTRIAL_ONLY/MAMMALS_TERRESTRIAL_ONLY.shp, data/source/AMPHIBIANS/AMPHIBIANS_PART1.shp and _PART2.shp, data/source/REPTILES/REPTILES_PART1.shp and _PART2.shp | TROPICS_DOWNLOADS |
| Bird species distribution maps of the world, 2025 release, with the HBW and BirdLife International Digital Checklist of the Birds of the World, version 10<sup>4</sup> | Bird ranges and Red List categories | http://datazone.birdlife.org | data/source/species/BOTW_2025.gpkg (layer all_species) and data/source/species/Handbook of the Birds of the World and BirdLife International Digital Checklist of the Birds of the World_Version_10.xlsx | TROPICS_DOWNLOADS |
| Natural Earth 1:10m admin-0 countries | Country of each agglomeration, map outlines | https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/ | data/source/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp | TROPICS_DOWNLOADS |
| RESOLVE Ecoregions 2017<sup>5</sup> | Biome and biogeographic realm of each agglomeration | https://ecoregions.appspot.com/ | data/source/ecoregions/Ecoregions2017.shp | TROPICS_DOWNLOADS |
| Natural Earth 1:10m populated places | Labels of named agglomerations in the maps (measurement/place_names.py writes the place table data/raw/tropical_places.csv from it) | https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/ | data/source/ne_10m_populated_places/ne_10m_populated_places.shp | TROPICS_DOWNLOADS |

## Environment

The code was run with Python 3.10, numpy 2.2, pandas 2.3, scipy 1.15, matplotlib 3.10, rasterio 1.4 (GDAL 3.10), pyogrio 0.13, pyproj 3.7, shapely 2.1, scikit-learn 1.7, statsmodels 0.15, openpyxl and xlsxwriter. environment.yml creates a conda environment with these versions (`conda env create -f environment.yml`), and requirements.txt lists the packages for pip.

## Sources

The source products keep the licenses and terms of their providers. The other files in data/figures are derived from the source products, and none of them holds a species range polygon or a range raster. GISD30 and GLC_FCS30D are distributed on Zenodo by their authors. The IUCN Red List spatial data and the BirdLife International distribution maps are distributed by IUCN and BirdLife International under their terms of use, which do not permit redistribution. Natural Earth is in the public domain. RESOLVE Ecoregions 2017 is distributed by RESOLVE.

## References

<sup>1</sup> Zhang, X., Liu, L., Zhao, T., Gao, Y., Chen, X., & Mi, J. (2022). GISD30: global 30 m impervious-surface dynamic dataset from 1985 to 2020 using time-series Landsat imagery on the Google Earth Engine platform. *Earth System Science Data, 14*, 1831–1856. https://doi.org/10.5194/essd-14-1831-2022

<sup>2</sup> Zhang, X., et al. (2024). GLC_FCS30D: the first global 30 m land-cover dynamics monitoring product with a fine classification system for the period from 1985 to 2022 generated using dense-time-series Landsat imagery and the continuous change-detection method. *Earth System Science Data, 16*, 1353–1381. https://doi.org/10.5194/essd-16-1353-2024

<sup>3</sup> IUCN (2023). *The IUCN Red List of Threatened Species*, spatial data, version 2023-1. https://www.iucnredlist.org

<sup>4</sup> BirdLife International and Handbook of the Birds of the World (2025). *Bird species distribution maps of the world*, 2025 release, with the HBW and BirdLife International Digital Checklist of the Birds of the World, version 10. http://datazone.birdlife.org

<sup>5</sup> Dinerstein, E., et al. (2017). An ecoregion-based approach to protecting half the terrestrial realm. *BioScience, 67*(6), 534–545. https://doi.org/10.1093/biosci/bix014

## License

The code and figures are released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (see LICENSE). The source data are not covered by this license.
