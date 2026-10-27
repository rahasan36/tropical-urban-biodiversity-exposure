# A third of the world's threatened land vertebrates exposed to tropical urbanization

Code and figure data for a pan-tropical assessment of how urban expansion between 2000 and 2020 converted the ranges of threatened and range-restricted land vertebrates. The analysis delineates 7,229 urban agglomerations from 30 m impervious-surface dynamics, measures the expert-mapped range their growth converted, and relates that exposure to agglomeration size, growth, setting and timing. The source products are not redistributed here (see Source data).

**Keywords:** urban expansion, tropics, biodiversity, threatened species, range-restricted species, impervious surface, urban agglomerations

## Analyses

| Folder | Contents |
|---|---|
| common | Shared settings and helpers. tropics_config.py holds the paths, the extent of the tropical band, the codes of the source products and the constants of the analysis. tropics_grid.py (cell areas on the WGS84 ellipsoid), tropics_flags.py (Red List category, taxon and range size of each species), tropics_style.py, tropics_frame.py, tropics_robinson.py and tropics_basemap.py (map projection, base maps and figure style). |
| measurement | From the source products to one table of agglomerations. place_names.py (place names for the map labels), tropics_urban.py (GISD30 mosaic on a 30 arcsec lattice), tropics_cities.py (delineation by the City Clustering Algorithm), tropics_measure.py (30 m measurement inside each footprint), tropics_species.py and tropics_rangecache.py (range ingest and a local cache of the tropical part of every range), tropics_cost.py (range converted by each agglomeration), tropics_assemble.py and tropics_results.py (agglomeration table), tropics_setting.py and setting_rebuild.py (threatened species within 50 km of the year 2000 footprint), birdlife_union.py (bird ranges as spatial unions), tropics_form.py and tropics_habpass.py (expansion form and land cover replaced), tropics_surfaces.py (gridded surfaces of the maps), tropics_verify.py (checks). |
| sensitivity | Sensitivity of the measurement. tropics_delin.py, delineation_fixed.py, delineation_note.py and threshold_isolate.py (delineation rules), tropics_radius.py (radius of the setting covariate), resolution_test.py and resolution_resume.py (lattice and boundary rule), species_boundary.py (boundary rule and the share of one range converted), tropics_crosscheck.py (setting covariate from disjoint taxonomic groups), setting_definition_compare.py, landcover_disagreement.py, bird_union_effect.py, bird_sensitivity.py, tropics_sensitivity.py and weighting_vs_sample.py. |
| temporal | Five-year periods. tropics_temporal.py (urban land at 2000, 2005, 2010, 2015 and 2020 and the range each period converted), temporal_analysis.py, temporal_size.py, temporal_subregion.py and temporal_species.py (period totals, size distribution, species accrual by period and by sub-region), recency_numbers.py (timing of the impervious surface and its relation to rarity-weighted exposure), temporal_verify.py (checks). |
| models | Scaling models and statistics. tropics_scaling.py (estimation of the size exponent and the spatial diagnostic), tropics_numbers.py, tropics_refit.py, tropics_analysis.py, tropics_models.py and tropics_sizevars.py (primary specification, response definitions, quantile regression, variance decomposition, alternative measures of size), tropics_cluster.py (typology of agglomerations), r2_components.py (marginal and conditional R²), spatial_cv_blocks.py, spatial_cv_run.py and validate_setting.py (buffered spatial block cross-validation), drivers.py, species_ledger.py, species_concentration.py, region_overlap.py and region_search.py. |
| reallocation | Reallocation of growth within countries. tropics_counterfactual.py and tropics_cfrobust.py (allocation of each country's added land by the power of year 2000 extent), cf_release.py, cf_run.py, cf_bands.py, cf_influence.py, cf_boundary_check.py and cf_fix.py (country bootstrap, response and realm bands, leave-one-country-out influence). |
| supplementary_tables | The Supplementary Tables. tropics_sitables.py, si_print.py, si_tables_two_part.py and table_s5_quantiles.py. |
| figures | The script and the figure of each figure of the paper and its Supplementary Information (see Figures). figures/plotting holds the plotting code the scripts share. |

run_pipeline.py runs the stages in order, from the source products to the Supplementary Tables, the figures and the checks. From the repository folder,

    python run_pipeline.py measurement
    python run_pipeline.py temporal
    python run_pipeline.py analysis
    python run_pipeline.py tables
    python run_pipeline.py figures
    python run_pipeline.py verify

or `python run_pipeline.py all` for every stage. One module runs on its own with `python run_pipeline.py module <name> [arguments]`, for example `python run_pipeline.py module temporal_species`. The measurement and temporal stages read the full range files and the 30 m rasters of the tropical band and run for many hours. The steps that take a time budget resume from their last completed tile or species when they are started again.

The analysis writes its intermediate files to data/interim and data/processed, its results to outputs/tables and the Supplementary Tables to manuscript/tables, all inside the repository folder. TROPICS_PROJECT moves these folders elsewhere, and TROPICS_SCRATCH moves the temporary files of the measurement.

## Figures

The folder figures holds the script behind each figure of the paper and its Supplementary Information, one folder per figure, with the figure it draws as PNG and PDF. The folder data/figures holds the inputs the scripts read, so the figures can be drawn without the source products or the analysis. Run a script from the repository folder without arguments, for example

    python figures/Fig3_size_growth_setting/make_fig3_size_growth_setting.py

Each script writes the figure to its own folder, or to the folder given by --out, and adds _r1, _r2 and so on to the name rather than overwrite an existing figure. --data points a script to another copy of the figure inputs, and --pipeline draws the figure from the outputs of run_pipeline.py instead.

| Figure | Content | Script (in figures/) | Inputs (in data/figures) |
|---|---|---|---|
| Figure 1 | Tropical urban expansion, 2000 to 2020 | Fig1_urban_expansion/make_fig1_urban_expansion.py | tables/cities_cost.csv, interim/city_form.csv, interim/city_habitat.csv, interim/surfaces, naturalearth |
| Figure 2 | Exposure intensity and its spread | Fig2_exposure_intensity/make_fig2_exposure_intensity.py | tables/cities_cost.csv, interim/city_form.csv, naturalearth |
| Figure 3 | Exposure intensity by size, growth and setting | Fig3_size_growth_setting/make_fig3_size_growth_setting.py | tables/cities_cost.csv, tables/plan4.json, tables/setting_footprint_v3.csv, interim/city_form.csv |
| Figure 4 | Reallocation of urban growth within countries | Fig4_reallocation/make_fig4_reallocation.py | tables/cities_cost.csv, tables/cf_release.json, tables/cf_robust.json, interim/city_form.csv |
| Figure 5 | Range rarity, the size relationship and three regional windows | Fig5_range_rarity/make_fig5_range_rarity.py | tables/cities_cost.csv, tables/plan4.json, interim/city_form.csv, interim/species_city_pairs.npz, interim/sp_flags_merged.npz, interim/surfaces, naturalearth |
| Figure 6 | Urban growth and accumulating range conversion, 2000 to 2020 | Fig6_temporal_accrual/make_fig6_temporal_accrual.py | tables/temporal |
| Supplementary Fig. 1 | Delineated agglomerations | SFig01_agglomerations/make_sfig01_agglomerations.py | tables/cities_cost.csv, interim/city_form.csv, naturalearth |
| Supplementary Fig. 2 | Agglomerations matched on 2000 size and contrasted in exposure | SFig02_matched_pairs/make_sfig02_matched_pairs.py | tables/cities_cost.csv, tables/plan4.json, tables/setting_footprint_v3.csv, interim/city_form.csv, interim/city_id_30arcsec.npz, interim/city_measurements.npz, raw/tropical_places.csv, gisd30 |
| Supplementary Fig. 3 | Occurrence and intensity of narrow-range conversion against agglomeration size | SFig03_two_part/make_sfig03_two_part.py | tables/narrow_two_part_B.csv, tables/narrow_two_part_contrasts_B.json |
| Supplementary Fig. 4 | Concentration of a species' converted range across agglomerations | SFig04_species_concentration/make_sfig04_species_concentration.py | tables/species_concentration.csv, tables/species_concentration_strata.json |
| Supplementary Fig. 5 | Threatened species reached by taxon and Red List category | SFig05_threatened_taxa/make_sfig05_threatened_taxa.py | interim/species_city_pairs.npz, interim/sp_flags_merged.npz, interim/rangecache/ranges_idx.jsonl, interim/temporal/pairs, tables/temporal |
| Supplementary Fig. 6 | Range-restricted species and new impervious surface in four regions | SFig06_narrow_range_regions/make_sfig06_narrow_range_regions.py | interim/surfaces, naturalearth |
| Supplementary Fig. 7 | Impervious surface by 2000 against surface added since | SFig07_surface_by_period/make_sfig07_surface_by_period.py | interim/temporal/lattice_cum_counts.npz, naturalearth |
| Supplementary Fig. 8 | The 2020 impervious surface split by period | SFig08_surface_split/make_sfig08_surface_split.py | interim/temporal/lattice_cum_counts.npz, naturalearth |
| Supplementary Fig. 9 | When the agglomerated surface was added | SFig09_surface_timing/make_sfig09_surface_timing.py | tables/cities_cost.csv, interim/city_form.csv, interim/city_id_30arcsec.npz, interim/city_measurements.npz, interim/temporal/cell_epoch_area.npz, raw/tropical_places.csv, naturalearth |
| Supplementary Fig. 10 | The size distribution of agglomerations through time | SFig10_size_distribution/make_sfig10_size_distribution.py | interim/temporal/delineation, tables/temporal |
| Supplementary Fig. 11 | Agglomeration size and the period in which its surface was added | SFig11_size_and_period/make_sfig11_size_and_period.py | tables/cities_cost.csv, interim/city_form.csv, interim/city_id_30arcsec.npz, interim/city_measurements.npz, raw/tropical_places.csv, naturalearth |
| Supplementary Fig. 12 | Exposure intensity by country and by when an agglomeration formed | SFig12_country_intensity/make_sfig12_country_intensity.py | tables/cities_cost.csv, tables/setting_footprint_v3.csv, tables/temporal/temporal_city_periods.csv.gz, interim/city_form.csv, interim/city_habitat.csv, naturalearth |
| Supplementary Fig. 13 | Recency of urban surface and the rarity of the ranges it converts | SFig13_recency_rarity/make_sfig13_recency_rarity.py | tables/cities_cost.csv, interim/city_form.csv, interim/city_measurements.npz |

The same list is in figures/figure_scripts.csv. The maps are drawn in the Robinson projection.

## Figure inputs

| File or folder in data/figures | Contents | Written by |
|---|---|---|
| tables/cities_cost.csv | One row per agglomeration: location, country, realm and biome, impervious extent in 2000 and 2020, land added, range converted for all species of concern, threatened and narrow-range species, rarity-weighted conversion, exposure intensity and setting richness | measurement/tropics_assemble.py, measurement/tropics_results.py |
| tables/setting_footprint_v3.csv | Setting covariate of each agglomeration, the threatened species within 50 km of the year 2000 footprint | measurement/setting_rebuild.py |
| tables/plan4.json | Fitted models of the primary specification and its alternatives | models/tropics_models.py |
| tables/cf_release.json, tables/cf_robust.json | Reallocation curves and their bootstrap, response and realm bands | reallocation/cf_release.py, cf_run.py, cf_bands.py |
| tables/narrow_two_part_B.csv, tables/narrow_two_part_contrasts_B.json | Two-part model of narrow-range conversion | models/species_concentration.py |
| tables/species_concentration.csv, tables/species_concentration_strata.json | Concentration of each species' converted range across agglomerations | models/species_concentration.py |
| tables/temporal | Period totals, size classes, transitions, realms, sub-regions and the agglomeration-by-period table (temporal_city_periods.csv.gz) | temporal/temporal_analysis.py, temporal_size.py, temporal_subregion.py, temporal_species.py, figures/plotting/fig_threat_taxa.py |
| interim/city_form.csv, interim/city_habitat.csv | Expansion form (infill, edge, leapfrog) and land cover replaced, per agglomeration | measurement/tropics_form.py |
| interim/city_id_30arcsec.npz, interim/city_measurements.npz | Agglomeration of each 30 arcsec cell and the impervious area of each agglomeration by epoch | measurement/tropics_cities.py, measurement/tropics_measure.py |
| interim/species_city_pairs.npz | Range converted for each species and agglomeration, in km² | measurement/tropics_cost.py |
| interim/sp_flags_merged.npz, interim/rangecache/ranges_idx.jsonl | Red List category, taxon, scientific name and global range area of each species (no geometry) | common/tropics_flags.py, measurement/tropics_rangecache.py |
| interim/surfaces | Impervious surface at 0.05° and richness of threatened and narrow-range species at 0.1° | measurement/tropics_surfaces.py |
| interim/temporal | Impervious area of each 30 arcsec cell by epoch (cell_epoch_area.npz, lattice_cum_counts.npz), the agglomerations of 2000, 2005, 2010, 2015 and 2020 (delineation) and the range converted for each species, agglomeration and period (pairs) | temporal/tropics_temporal.py |
| raw/tropical_places.csv | Names, population and location of the populated places of the tropical band, from Natural Earth | measurement/place_names.py |
| gisd30 | GISD30 extracts around the six agglomerations of Supplementary Fig. 2, under the tile names of the product | Extracted from GISD30 |
| naturalearth | Natural Earth 1:10m admin-0 countries, version 5.1.1. The scripts write base-map caches beside the files in data/figures/interim on the first run. | Natural Earth |

## Source data

The measurement stage reads the products below. None is included in full. Download each from its provider, accept the provider's terms, and place the files at the paths given, relative to the repository folder. The environment variables in the last column point the code to copies held elsewhere.

| Product | Used for | Source | Path in the repository | Variable |
|---|---|---|---|---|
| GISD30, global 30 m impervious-surface dynamics, 1985 to 2020<sup>1</sup> | Urban extent at each epoch, agglomeration delineation, land added | https://doi.org/10.5281/zenodo.5220816 (twelve RAR archives, from which the 5° GeoTIFF tiles are extracted) | data/source/gisd30/tiles/GISD30_1985-2020_<tile>.tif | TROPICS_GISD30 |
| GLC_FCS30D, global 30 m land cover, 1985 to 2022<sup>2</sup> | Land cover the expansion replaced | https://zenodo.org/records/8239305 (read over the network by measurement/tropics_form.py, or from a local copy of a longitude band in data/source/GLC_FCS30D_19852022maps_E90-E95) | none | TROPICS_DOWNLOADS |
| IUCN Red List spatial data, version 2026-1<sup>3</sup>: terrestrial mammals, amphibians and reptiles | Species ranges and Red List categories | https://www.iucnredlist.org/resources/spatial-data-download | data/source/MAMMALS_TERRESTRIAL_ONLY/MAMMALS_TERRESTRIAL_ONLY.shp, data/source/AMPHIBIANS/AMPHIBIANS_PART1.shp and _PART2.shp, data/source/REPTILES/REPTILES_PART1.shp and _PART2.shp | TROPICS_DOWNLOADS |
| Bird species distribution maps of the world, 2025 release, with the HBW and BirdLife International Digital Checklist of the Birds of the World, version 10<sup>4</sup> | Bird ranges and Red List categories | http://datazone.birdlife.org | data/source/species/BOTW_2025.gpkg (layer all_species) and data/source/species/Handbook of the Birds of the World and BirdLife International Digital Checklist of the Birds of the World_Version_10.xlsx | TROPICS_DOWNLOADS |
| Natural Earth 1:10m admin-0 countries | Country of each agglomeration, map outlines | https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/ | data/source/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp | TROPICS_DOWNLOADS |
| RESOLVE Ecoregions 2017<sup>5</sup> | Biome and biogeographic realm of each agglomeration | https://ecoregions.appspot.com/ | data/source/ecoregions/Ecoregions2017.shp | TROPICS_DOWNLOADS |
| Natural Earth 1:10m populated places | Labels of named agglomerations in the maps (measurement/place_names.py writes the place table data/raw/tropical_places.csv from it) | https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/ | data/source/ne_10m_populated_places/ne_10m_populated_places.shp | TROPICS_DOWNLOADS |

## Environment

The code was run with Python 3.10, numpy 2.2, pandas 2.3, scipy 1.15, matplotlib 3.10, rasterio 1.4 (GDAL 3.10), pyogrio 0.13, pyproj 3.7, shapely 2.1, scikit-learn 1.7, statsmodels 0.15 and openpyxl. environment.yml creates a conda environment with these versions (`conda env create -f environment.yml`), and requirements.txt lists the packages for pip.

## Sources

The source products keep the licenses and terms of their providers. The GISD30 extracts in data/figures/gisd30 are redistributed under the CC BY 4.0 license of GISD30. The other files in data/figures are derived from the source products, and none of them holds a species range polygon or a range raster. GISD30 and GLC_FCS30D are distributed on Zenodo by their authors. The IUCN Red List spatial data and the BirdLife International distribution maps are distributed by IUCN and BirdLife International under their terms of use, which do not permit redistribution. Natural Earth is in the public domain. RESOLVE Ecoregions 2017 is distributed by RESOLVE.

## References

<sup>1</sup> Zhang, X., Liu, L., Zhao, T., Gao, Y., Chen, X., & Mi, J. (2022). GISD30: global 30 m impervious-surface dynamic dataset from 1985 to 2020 using time-series Landsat imagery on the Google Earth Engine platform. *Earth System Science Data, 14*, 1831–1856. https://doi.org/10.5194/essd-14-1831-2022

<sup>2</sup> Zhang, X., et al. (2024). GLC_FCS30D: the first global 30 m land-cover dynamics monitoring product with a fine classification system for the period from 1985 to 2022 generated using dense-time-series Landsat imagery and the continuous change-detection method. *Earth System Science Data, 16*, 1353–1381. https://doi.org/10.5194/essd-16-1353-2024

<sup>3</sup> IUCN (2026). *The IUCN Red List of Threatened Species*, spatial data, version 2026-1. https://www.iucnredlist.org

<sup>4</sup> BirdLife International and Handbook of the Birds of the World (2025). *Bird species distribution maps of the world*, 2025 release, with the HBW and BirdLife International Digital Checklist of the Birds of the World, version 10. http://datazone.birdlife.org

<sup>5</sup> Dinerstein, E., et al. (2017). An ecoregion-based approach to protecting half the terrestrial realm. *BioScience, 67*(6), 534–545. https://doi.org/10.1093/biosci/bix014

## License

The code and figures are released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (see LICENSE). The source data are not covered by this license.
