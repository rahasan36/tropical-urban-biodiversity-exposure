"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import json
import zipfile
import pandas as pd
import tropics_numbers as tn
from tropics_config import PROJECT, TABLE_DIR

OUT = PROJECT / "manuscript" / "supplementary_data"
SI = TABLE_DIR / "si"
T = TABLE_DIR / "temporal"

SD1_COLUMNS = ["city_id", "lon", "lat", "country", "continent", "biome", "realm", "n_cells",
               "urban_km2_base", "urban_km2_final", "urban_added_km2",
               "cost_km2", "cost_threat_km2", "cost_narrow_km2", "cost_all_km2", "rarity_converted",
               "n_threat_affected", "n_narrow_affected", "cost_per_km2", "rarity_per_km2",
               "setting_footprint", "setting_richness", "setting_richness_all",
               "form_infill_km2", "form_edge_km2", "form_leapfrog_km2",
               "form_infill_share", "form_edge_share", "form_leapfrog_share"]

SD1_DEFINITIONS = {
    "city_id": "Agglomeration identifier",
    "lon": "Longitude of the agglomeration centroid (degrees)",
    "lat": "Latitude of the agglomeration centroid (degrees)",
    "country": "ISO 3166-1 alpha-3 code of the country holding the centroid (Natural Earth ADM0_A3)",
    "continent": "Continent of that country (Natural Earth)",
    "biome": "Biome of the ecoregion holding the centroid (RESOLVE Ecoregions 2017)",
    "realm": "Biogeographic realm of that ecoregion (RESOLVE Ecoregions 2017)",
    "n_cells": "Number of 30 arcsecond lattice cells in the 2020 footprint",
    "urban_km2_base": "Impervious area in 2000 (km2)",
    "urban_km2_final": "Impervious area in 2020 (km2)",
    "urban_added_km2": "Impervious area added between 2000 and 2020 (km2)",
    "cost_km2": "Mapped range converted, summed over the species of concern (km2)",
    "cost_threat_km2": "Mapped range converted, summed over threatened species (km2)",
    "cost_narrow_km2": "Mapped range converted, summed over narrow-range species (km2)",
    "cost_all_km2": "Mapped range converted, summed over every species of the analysis pool (km2)",
    "rarity_converted": "Converted range of the species of concern, each divided by the species' global range",
    "n_threat_affected": "Threatened species with mapped range converted",
    "n_narrow_affected": "Narrow-range species with mapped range converted",
    "cost_per_km2": "Exposure intensity, cost_km2 divided by urban_added_km2 (km2 of range per km2 added)",
    "rarity_per_km2": "Rarity-weighted exposure intensity, rarity_converted divided by urban_added_km2 (km-2)",
    "setting_footprint": "Setting richness, threatened species whose mapped range comes within 50 km of the 2000 footprint",
    "setting_richness": "Threatened species whose mapped range comes within 50 km of the 2020 footprint",
    "setting_richness_all": "Species of the analysis pool whose mapped range comes within 50 km of the 2020 footprint",
    "form_infill_km2": "Impervious area added by infill (km2)",
    "form_edge_km2": "Impervious area added by edge expansion (km2)",
    "form_leapfrog_km2": "Impervious area added by leapfrog development (km2)",
    "form_infill_share": "Share of the added area added by infill",
    "form_edge_share": "Share of the added area added by edge expansion",
    "form_leapfrog_share": "Share of the added area added by leapfrog development",
}

SOURCES = {1: "tableS1_city_summary", 3: "tableS2_response_definitions", 4: "tableS9_spatial_cv",
           8: "species_boundary_shares",
           11: "resolution_test_v2", 12: "tableS4_delineation", 15: "tableS7_size_variables"}

SD4_FILES = ["temporal_periods.csv", "temporal_agglomerations.csv", "temporal_city_periods.csv.gz",
             "temporal_classes.csv", "temporal_emergence.csv", "temporal_realms.csv",
             "temporal_size_distribution.csv", "temporal_size_classes.csv",
             "temporal_transitions.csv", "temporal_class_transitions.csv", "temporal_gibrat.csv",
             "temporal_species_periods.csv", "temporal_species_cumulative.csv",
             "temporal_subregion_repeat.csv", "temporal_subregion_cumulative.csv",
             "temporal_subregions.csv"]


def _source(stem):
    for d in (SI, TABLE_DIR):
        p = d / f"{stem}.csv"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


TERMS = {"ls": "log10 year 2000 extent", "lset": "log10 setting richness + 1",
         "ladd": "log10 urban land added"}


def _block(stem, block):
    d = pd.read_csv(_source(stem))
    return d[d.block == block].drop(columns="block").dropna(axis=1, how="all")


def table_s7():
    d = pd.read_csv(_source("tableS15_objective_decomposition"))
    d["block"] = d.block.map({"responses": "species subset", "regions": "region"}).fillna(d.block)
    return d.rename(columns={"block": "grouping"})


def table_s13():
    d = _block("tableS11_species_ledger", "ledger")
    d["note"] = d.note.replace({
        "Taxa the study does not analyse; empty in the current configuration":
            "Taxa the study does not analyze, none in the current configuration",
        "Species with a non-zero converted area; the rest are in the pool through the setting covariate only":
            "Species with a non-zero converted area. The rest are in the pool through the setting covariate only"})
    return d


def table_s5():
    a = pd.read_csv(_source("tableS3_realm_biome"))
    q = pd.read_csv(_source("quantile_scaling")).rename(columns={"quantile": "group", "gamma": "gamma_setting"})
    q["level"] = "quantile"
    return pd.concat([a, q], ignore_index=True)[["level", "group", "n", "beta", "se", "ci_lower", "ci_upper",
                                                 "gamma_setting", "R2", "total_cost", "excludes_zero"]]


def table_s6():
    d = pd.read_csv(_source("tableS5_cross_taxon"))
    cor = d[d.block == "rank correlation between settings"]
    rho = {}
    for r in cor.itertuples():
        rho[(r.setting_from, r.exposure_of)] = rho[(r.exposure_of, r.setting_from)] = r.statistic
    rows = []
    for r in d[d.block == "setting predicts the excluded groups"].itertuples():
        row = dict(setting_measured_on=r.setting_from, predicting=r.exposure_of, n=r.n)
        for g in ("birds", "mammals", "herps"):
            row[f"spearman_with_{g}"] = rho.get((r.setting_from, g))
        row.update(setting_coefficient=r.statistic, se=r.se, p=r.p_value, size_exponent=r.beta_size)
        rows.append(row)
    return pd.DataFrame(rows)


def table_s9():
    m = json.load(open(TABLE_DIR / "narrow_two_part_meta.json"))
    con = json.load(open(TABLE_DIR / "narrow_two_part_contrasts_B.json"))
    rows = []
    for spec, d in m.items():
        if spec == "sample":
            continue
        sp = spec.split(":")[0]
        for part in ("occurrence", "intensity"):
            for k, (b, se, p) in d[part].items():
                rows.append(dict(specification=sp, part=part, quantity=f"coefficient, {TERMS[k]}",
                                 estimate=b, se=se, p=p))
        if sp == "A":
            r, lo, hi = d["combined_ratio"]
            rows.append(dict(specification=sp, part="combined",
                             quantity="fold change, 1st to 99th percentile of year 2000 size",
                             estimate=r, lower=lo, upper=hi))
        rows.append(dict(specification=sp, part="combined", quantity="Duan smearing factor", estimate=d["smearing"]))
        rows.append(dict(specification=sp, part="combined",
                         quantity="share of replicates higher at the largest size than the smallest",
                         estimate=d["share_rising"]))
        rows.append(dict(specification=sp, part="combined",
                         quantity="share of replicates rising across the whole size grid",
                         estimate=d["share_monotone"]))
    for cov in con:
        for q, (r, lo, hi) in con[cov].items():
            rows.append(dict(specification="B", part={"p": "occurrence", "intensity_pos": "intensity"}.get(q, q),
                             quantity=f"fold change, 1st to 99th percentile of {cov}",
                             estimate=r, lower=lo, upper=hi))
    return pd.DataFrame(rows)


def table_s17():
    rows = []
    for spec in ("A", "B"):
        f = TABLE_DIR / f"narrow_two_part_validation_{spec}.json"
        if f.exists():
            v = json.load(open(f))
            for k in ("auc", "cal_slope", "brier", "intensity_r2_log10", "combined_r2"):
                rows.append(dict(specification=spec, metric=k, value=v[k],
                                 agglomerations_scored=v["n_positive"] if k == "intensity_r2_log10" else v["n_scored"],
                                 block_degrees=v["deg"], buffer_km=v["buffer_km"]))
    return pd.DataFrame(rows)


def table_s10():
    s = json.load(open(TABLE_DIR / "species_concentration_strata.json"))
    rows = []
    for b in s["bands"]:
        r = dict(grouping="converted share of mapped range", group=b["band"], **{k: v for k, v in b.items() if k != "band"})
        if b["band"] == "all":
            r.update(rho=s["rho_unstratified"][0], p=s["rho_unstratified"][1])
        rows.append(r)
    for r in s["strata"]:
        rows.append(dict(grouping="range-size quintile", group=r["stratum"], **{k: v for k, v in r.items() if k != "stratum"}))
    for r in s.get("taxa", []):
        rows.append(dict(grouping="taxonomic class", group=r["taxon"], **{k: v for k, v in r.items() if k != "taxon"}))
    return pd.DataFrame(rows)


def table_s18():
    s = json.load(open(TABLE_DIR / "species_concentration_strata.json"))
    return pd.DataFrame(s["rule_sensitivity"])


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    d = tn.table()
    d1 = d[SD1_COLUMNS].sort_values("city_id")
    d1.to_csv(OUT / "Supplementary_Data_1.csv", index=False)
    pd.DataFrame({"column": SD1_COLUMNS, "definition": [SD1_DEFINITIONS[c] for c in SD1_COLUMNS]}).to_csv(
        OUT / "Supplementary_Data_1_columns.csv", index=False)
    built = {2: lambda: _block("tableS14_landcover_disagreement", "geography"),
             5: table_s5, 6: table_s6, 9: table_s9, 10: table_s10,
             7: table_s7,
             13: table_s13,
             14: lambda: _block("tableS11_species_ledger", "composition"),
             16: lambda: _block("tableS14_landcover_disagreement", "estimates"),
             17: table_s17, 18: table_s18}
    with zipfile.ZipFile(OUT / "Supplementary_Data_2.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for k in range(1, 19):
            if k in built:
                z.writestr(f"Supplementary_Table_S{k}.csv", built[k]().to_csv(index=False))
            else:
                z.write(_source(SOURCES[k]), f"Supplementary_Table_S{k}.csv")
    b = pd.read_csv(TABLE_DIR / "birdlife_union_summary.csv")
    b.to_csv(OUT / "Supplementary_Data_3.csv", index=False)
    with zipfile.ZipFile(OUT / "Supplementary_Data_4.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in SD4_FILES:
            z.write(T / f, f)
    return dict(sd1=d1.shape, sd3=b.shape, out=str(OUT))


if __name__ == "__main__":
    print(build())
