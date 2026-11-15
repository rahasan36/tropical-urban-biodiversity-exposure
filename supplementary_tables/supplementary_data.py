"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tropics_numbers as tn
from tropics_config import PROJECT, TABLE_DIR

OUT = PROJECT / "manuscript" / "supplementary_data"
SI = TABLE_DIR / "si"
T = TABLE_DIR / "temporal"

CITY_COLUMNS = ["city_id", "lon", "lat", "country", "continent", "biome", "realm", "n_cells",
               "urban_km2_base", "urban_km2_final", "urban_added_km2",
               "cost_km2", "cost_threat_km2", "cost_narrow_km2", "cost_all_km2", "rarity_converted",
               "n_threat_affected", "n_narrow_affected", "cost_per_km2", "rarity_per_km2",
               "setting_footprint", "setting_richness", "setting_richness_all",
               "form_infill_km2", "form_edge_km2", "form_leapfrog_km2",
               "form_infill_share", "form_edge_share", "form_leapfrog_share"]

CITY_DEFINITIONS = {
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

TEMPORAL_FILES = ["temporal_periods.csv", "temporal_agglomerations.csv", "temporal_city_periods.csv.gz",
                  "temporal_classes.csv", "temporal_emergence.csv", "temporal_realms.csv",
                  "temporal_size_distribution.csv", "temporal_size_classes.csv",
                  "temporal_transitions.csv", "temporal_class_transitions.csv", "temporal_gibrat.csv",
                  "temporal_species_periods.csv", "temporal_species_cumulative.csv",
                  "temporal_subregion_repeat.csv", "temporal_subregion_cumulative.csv",
                  "temporal_subregions.csv", "agglomeration_deepening.csv", "agglomeration_deepening_summary.csv",
                  "disagreement_check.csv"]

TEMPORAL_TEXT = {
    "temporal_periods": "Land added, converted range and species reached in each five-year period",
    "temporal_agglomerations": "Agglomerations, delineated area and impervious area at each epoch",
    "temporal_city_periods": "Each agglomeration in each period: area at the start, land added and range converted",
    "temporal_classes": "Exposure by impervious-area class and period",
    "temporal_emergence": "Agglomerations present at the period start and formed within it: land added and intensity",
    "temporal_realms": "Exposure by realm and period",
    "temporal_size_distribution": "Size distribution and rank-size exponent at each epoch (Extended Data Fig. 1b)",
    "temporal_size_classes": "Land added and range converted by size class at period start (Extended Data Fig. 1d)",
    "temporal_transitions": "Agglomerations continued, newly formed and formed by merger in each period",
    "temporal_class_transitions": "Agglomerations moving between size classes in each period",
    "temporal_gibrat": "Slope of growth rate on size in each period (Extended Data Fig. 1c)",
    "temporal_species_periods": "Each species reached: the number of periods and the first period in which it was reached",
    "temporal_species_cumulative": "Quantiles, by species group and epoch, of the share of each species' 2000 to 2020 converted range that had been converted by the end of the epoch (percent of the 2000 to 2020 total, not of global range)",
    "temporal_subregion_repeat": "Species reached and passing 1 percent of mapped range converted, by sub-region and period",
    "temporal_subregion_cumulative": "Cumulative conversion by sub-region",
    "temporal_subregions": "Change in concentration and rarity-weighted intensity by sub-region",
    "agglomeration_deepening": "Threatened range each agglomeration converted by 2005 and afterward (Fig. 4c)",
    "agglomeration_deepening_summary": "Summary statistics of Fig. 4c",
    "disagreement_check": "Species reached and above 1 percent under each treatment of disputed early land (Supplementary Note 2)",
    "temporal_fits": "Size exponent and setting coefficient of the primary specification fitted separately in each five-year period and over 2000 to 2020, with the Shapley shares of size and setting in the variance of log exposure intensity",
    "band_totals": "Impervious surface in the tropical band, the share added inside agglomerations, and the addition outside agglomerations by the 2020 built-up fraction of its 30 arcsecond cell (Supplementary Note 1)",
    "outside_species": "Species of concern, threatened and narrow-range species, and threatened species by class, reached and converted by impervious surface added inside and outside the delineated agglomerations, 2000 to 2020",
    "outside_builtup": "Impervious surface added outside the agglomerations, the range of species of concern it converted and the range converted per km2 added, by the 2020 built-up fraction of its 30 arcsecond cell",
}

CHECK_FILES = {"Checks_setting_buffer.csv": "setting_radius.csv",
               "Checks_residual_moran.csv": "residual_moran.csv"}

CV_MODELS = ["size only", "setting only", "size + setting", "size + setting + biome", "+ country"]

RESPONSES = {"Threatened, unweighted": ("Threatened", "Unweighted"),
             "Narrow range, unweighted": ("Narrow range", "Unweighted"),
             "Species of concern, rarity weighted": ("Species of concern", "Rarity weighted"),
             "Threatened, rarity weighted": ("Threatened", "Rarity weighted"),
             "Narrow range, rarity weighted": ("Narrow range", "Rarity weighted")}


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


def decomposition():
    d = pd.read_csv(_source("tableS15_objective_decomposition")).rename(columns={"grouping": "region_type"})
    d["block"] = d.block.map({"responses": "species subset", "regions": "region"}).fillna(d.block)
    return d.rename(columns={"block": "grouping"})


def realm_biome_quantile():
    a = pd.read_csv(_source("tableS3_realm_biome"))
    q = pd.read_csv(_source("quantile_scaling")).rename(columns={"quantile": "group", "gamma": "gamma_setting"})
    q["level"] = "quantile"
    return pd.concat([a, q], ignore_index=True)[["level", "group", "n", "beta", "se", "ci_lower", "ci_upper",
                                                 "gamma_setting", "R2", "total_cost", "excludes_zero"]]


def cross_taxon():
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


def two_part():
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


def two_part_validation():
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


def rule_sensitivity():
    s = json.load(open(TABLE_DIR / "species_concentration_strata.json"))
    return pd.DataFrame(s["rule_sensitivity"])


def responses():
    d = pd.read_csv(_source("tableS2_response_definitions"))
    d["specification"] = d["specification"].astype(str).str.replace(" (as drafted)", "", regex=False)
    return d


def species_sets():
    s = responses()
    rows = {("Species of concern", "Unweighted"): s.iloc[0]}
    for _, r in s.iloc[6:].iterrows():
        rows[RESPONSES[r.response]] = r
    out = []
    for sset in ("Species of concern", "Threatened", "Narrow range"):
        for wt in ("Unweighted", "Rarity weighted"):
            r = rows[(sset, wt)]
            out.append(dict(species_set=sset, weighting=wt, n=r.n, beta=r.beta, ci_lower=r.ci_lower,
                            ci_upper=r.ci_upper, p_value=r.p_value, R2=r.R2))
    return pd.DataFrame(out)


def spatial_cv():
    cv = pd.read_csv(TABLE_DIR / "_spatial_cv_buffered.csv")
    w = cv.pivot_table(index=["block_deg", "buffer_km", "n_blocks", "n_scored"], columns="model", values="R2_oos",
                       aggfunc="first")
    return w[CV_MODELS].reset_index().sort_values(["block_deg", "buffer_km"]).reset_index(drop=True)


def realm_biome():
    s = realm_biome_quantile()
    return s[s.level != "quantile"].drop(columns=["se"]).reset_index(drop=True)


def quantiles():
    s = realm_biome_quantile()
    q = s[s.level == "quantile"].rename(columns={"group": "quantile"})
    return q[["quantile", "n", "beta", "se", "ci_lower", "ci_upper", "gamma_setting"]].reset_index(drop=True)


def two_part_coefficients():
    s = two_part()
    c = s[s.quantity.str.startswith("coefficient")].copy()
    c["term"] = c.quantity.str.replace("coefficient, ", "", regex=False)
    return c[["specification", "part", "term", "estimate", "se", "p"]].reset_index(drop=True)


def two_part_fold_changes():
    s = two_part()
    f = s[s.quantity.str.startswith("fold change")].copy()
    f["covariate_varied"] = np.where(f.quantity.str.contains("setting"), "setting richness", "year 2000 size")
    return f[["specification", "covariate_varied", "part", "estimate", "lower", "upper"]].rename(
        columns={"estimate": "fold_change"}).reset_index(drop=True)


def concentration():
    d = pd.read_csv(TABLE_DIR / "species_concentration.csv")
    d = d[d.rule == "centroid"]

    def row(grouping, group, g, lo=np.nan, hi=np.nan):
        r, p = spearmanr(g.frac_converted, g.top1_share)
        return dict(grouping=grouping, group=group, n=len(g), range_median_km2=g.range_km2.median(), range_lo_km2=lo,
                    range_hi_km2=hi, n_contrib_median=g.n_city.median(), top1_median=g.top1_share.median(),
                    top3_median=g.top3_share.median(), share_top1_gt50=(g.top1_share > 0.5).mean(), rho=r, p=p)

    rows = [row("converted share of mapped range", lab, d if cut is None else d[d.frac_converted >= cut])
            for lab, cut in (("all", None), (">=0.1%", 0.001), (">=1%", 0.01), (">=5%", 0.05))]
    q = pd.qcut(d.range_km2, 5, labels=False)
    for i in range(5):
        g = d[q == i]
        rows.append(row("range-size quintile", str(i + 1), g, g.range_km2.min(), g.range_km2.max()))
    for taxon, g in d.groupby("taxon"):
        rows.append(row("taxonomic class", taxon.capitalize(), g))
    return pd.DataFrame(rows)


def _conc_species():
    d = pd.read_csv(TABLE_DIR / "species_concentration.csv")
    return d[d.rule == "centroid"].copy()


def concentration_threatened():
    d = _conc_species()
    t = d[d.is_threat.astype(bool)]

    def row(grouping, group, g):
        return dict(grouping=grouping, group=group, species_reached=len(g),
                    share_reached_by_two_or_more=(g.n_city >= 2).mean(),
                    share_five_or_fewer_hold_90pct=(g.n_city_90 <= 5).mean(),
                    share_one_agglomeration_holds_all=(g.top1_share >= 1 - 1e-9).mean(),
                    agglomerations_median=g.n_city.median(),
                    agglomerations_for_90pct_median=g.n_city_90.median(),
                    largest_share_median=g.top1_share.median())

    rows = [row("all threatened species reached", "all", t)]
    for c in ("CR", "EN", "VU"):
        rows.append(row("Red List category", c, t[t.category.astype(str).str.upper() == c]))
    for taxon, g in t.groupby("taxon"):
        rows.append(row("taxonomic class", taxon.capitalize(), g))
    rows.append(row("all species of concern with converted range", "all", d))
    return pd.DataFrame(rows)


def concentration_species():
    d = _conc_species()
    cols = {"sp_id": "species_id", "sci_name": "scientific_name", "taxon": "class", "category": "red_list_category",
            "is_threat": "threatened", "is_narrow": "narrow_range", "range_km2": "global_range_km2",
            "converted_km2": "converted_km2", "frac_converted": "share_of_range_converted",
            "n_city": "agglomerations_converting", "top1_share": "largest_agglomeration_share",
            "top3_share": "three_largest_share", "n_city_90": "agglomerations_for_90pct"}
    return d[list(cols)].rename(columns=cols).sort_values("converted_km2", ascending=False).reset_index(drop=True)


def temporal_fits():
    rows = []
    for j in range(5):
        z = json.loads((T / f"temporal_fit_{j}.json").read_text())
        for resp, lab in (("cost_per_km2", "exposure intensity"), ("rarity_per_km2", "rarity-weighted intensity")):
            r = z[resp]
            own = resp == "cost_per_km2"
            rows.append(dict(period=z["period"], response=lab, n=r["n"], size_exponent=r["beta"],
                             ci_lower=r["ci_lower"], ci_upper=r["ci_upper"], p_value=r["p_value"],
                             setting_coefficient=r["gamma_setting"], R2=r["R2"],
                             shapley_size=z["shapley"]["size"] if own else np.nan,
                             shapley_setting=z["shapley"]["setting"] if own else np.nan,
                             shapley_R2=z["shapley_R2"] if own else np.nan))
    return pd.DataFrame(rows)


def left_out():
    return pd.read_csv(TABLE_DIR / "cf_influence.csv").rename(columns=dict(
        dropped="country_left_out", n_cities="agglomerations_remaining", n_dropped="agglomerations_removed",
        measured_dispersed="measured_change_dispersed_pct", measured_concentrated="measured_change_concentrated_pct",
        predicted_dispersed="fitted_change_dispersed_pct", predicted_concentrated="fitted_change_concentrated_pct",
        paired_concentrated="fitted_minus_measured_concentrated_pp",
        paired_dispersed="fitted_minus_measured_dispersed_pp"))


def pooled():
    r = json.loads((TABLE_DIR / "cf_release.json").read_text())
    p = json.loads((TABLE_DIR / "cf_pool_bootstrap.json").read_text())
    reps = np.asarray(r["reps_m"], float)
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    rows = [dict(pool="within country", alpha=a, change_pct=r["point"]["measured"][i], ci_lower_pct=lo[i],
                 ci_upper_pct=hi[i], bootstrap_replicates=len(reps)) for i, a in enumerate(r["alphas"])]
    for key, lab in (("realm", "within realm"), ("tropics", "across the tropics")):
        d = p[key]["minus_country"]
        for i, a in enumerate(p["alphas"]):
            rows.append(dict(pool=lab, alpha=a, change_pct=p[key]["pct"][i], ci_lower_pct=p[key]["ci_lo"][i],
                             ci_upper_pct=p[key]["ci_hi"][i], bootstrap_replicates=p["n_boot"],
                             minus_within_country_pp=d["pct"][i], minus_within_country_ci_lower_pp=d["ci_lo"][i],
                             minus_within_country_ci_upper_pp=d["ci_hi"][i]))
    return pd.DataFrame(rows)


def capped():
    r = json.loads((TABLE_DIR / "cf_robust.json").read_text())
    return pd.DataFrame(dict(alpha=r["alphas"], uncapped_pct=r["aggregate"]["pct"],
                             cap_two_times_pct=r["cap2"]["pct"], cap_five_times_pct=r["cap5"]["pct"]))


def source(stem):
    return lambda: pd.read_csv(_source(stem))


TABLES = {
    1: source("tableS1_city_summary"),
    2: lambda: _block("tableS14_landcover_disagreement", "geography"),
    3: source("species_accounting_by_class"),
    4: concentration,
    5: rule_sensitivity,
    6: source("species_boundary_shares"),
    7: lambda: responses().iloc[:6].reset_index(drop=True),
    8: spatial_cv,
    9: realm_biome,
    10: quantiles,
    11: source("tableS4_delineation"),
    12: source("tableS7_size_variables"),
    13: cross_taxon,
    14: species_sets,
    15: two_part_coefficients,
    16: two_part_fold_changes,
    17: source("resolution_test_v2"),
    18: source("species_ledger"),
    19: lambda: _block("tableS14_landcover_disagreement", "estimates"),
    20: decomposition,
    21: two_part_validation,
    22: concentration_threatened,
    23: concentration_species,
}


SHEETS = [
    ("Agglomeration summary", 1, "Summary of the per-agglomeration table (Supplementary Data 1)"),
    ("Land cover disagreement", 2, "Share of land GISD30 dates as new that GLC_FCS30D labels impervious in 2000"),
    ("Species accounting by class", 3, "Species and threatened species by class and boundary rule (Extended Data Fig. 3g)"),
    ("Concentration by group", 4, "Concentration of converted range by exposure band, range size and class (Extended Data Fig. 4c)"),
    ("Threatened concentration", 22, "Threatened species reached by two or more agglomerations and those for which five or fewer agglomerations hold 90 percent of the converted range, center rule"),
    ("Concentration by species", 23, "Per-species concentration of converted range across agglomerations for every species of concern with converted range, center rule"),
    ("Concentration by boundary rule", 5, "Concentration under the three boundary rules (Extended Data Fig. 4d)"),
    ("Most exposed species", 6, "Converted share of the twenty most exposed threatened species with at least 50 km2 of mapped range under the three boundary rules (Extended Data Fig. 3h)"),
    ("Spatial cross-validation", 8, "Held-out R2 by specification, block size and buffer (Extended Data Fig. 5d)"),
    ("Exponent by realm and biome", 9, "Size exponent within each realm and biome (Extended Data Table 2)"),
    ("Exponent by quantile", 10, "Size exponent by quantile regression (Extended Data Table 2)"),
    ("Delineation sensitivity", 11, "Size exponent under each delineation configuration (Extended Data Table 2)"),
    ("Size variable definitions", 12, "Size exponent under alternative size variables and minimum sizes (Extended Data Table 2)"),
    ("Setting from one taxon", 13, "Setting measured on one taxonomic group (Extended Data Fig. 5e and Extended Data Table 2)"),
    ("Exponent by species set", 14, "Size exponent for each species set, unweighted and rarity weighted (Fig. 2g)"),
    ("Two-part coefficients", 15, "Coefficients of the two-part model (Extended Data Fig. 7e)"),
    ("Two-part fold changes", 16, "Fold changes predicted by the two-part model (Extended Data Fig. 7d)"),
    ("Lattice and boundary rule", 17, "Size exponent by lattice resolution and boundary rule (Extended Data Table 2)"),
    ("Species ledger", 18, "Species between the range assembly and the analysis pool"),
    ("Land cover agreement subsets", 19, "Estimates where the two land-cover products agree (Extended Data Table 2)"),
    ("Rarity decomposition", 20, "Unweighted, rarity-weighted and mean-range exponents (Extended Data Fig. 6e)"),
    ("Two-part validation", 21, "Validation of the two-part model on held-out blocks (Extended Data Fig. 7f)"),
]

EXTRA_SHEETS = [
    ("Narrow-range share by region", lambda: region_overlap(), "New impervious surface over narrow-range species by region (Extended Data Fig. 6f)"),
    ("Reallocation one country out", lambda: left_out(), "Reallocation with each country left out (Fig. 3e)"),
    ("Reallocation pooled", lambda: pooled(), "Reallocation with land pooled within country, realm and the tropics (Fig. 3f)"),
    ("Reallocation capped", lambda: capped(), "Reallocation with capped allocations (Supplementary Note 2)"),
    ("Check setting buffer", lambda: pd.read_csv(TABLE_DIR / "setting_radius.csv"), "Setting measured within 25, 50 and 100 km (Supplementary Note 2)"),
    ("Check residual Moran", lambda: pd.read_csv(TABLE_DIR / "residual_moran.csv"), "Moran's I of the model residuals (Supplementary Note 2)"),
]


def region_overlap():
    d = json.loads((TABLE_DIR / "region_window_overlap.json").read_text())
    rows = [dict(region=k, lon_min=v.get("box", [np.nan] * 4)[0], lon_max=v.get("box", [np.nan] * 4)[1],
                 lat_min=v.get("box", [np.nan] * 4)[2], lat_max=v.get("box", [np.nan] * 4)[3],
                 added_km2=v["added_km2"], share_over_narrow_ranges_pct=v["share_pct"])
            for k, v in d.items() if isinstance(v, dict)]
    out = pd.DataFrame(rows)
    out["display_narrow_threshold_km2"] = d["narrow_threshold_km2"]
    return out


def band_totals():
    b = json.loads((TABLE_DIR / "recency_summary.json").read_text())["band"]
    rows = [dict(quantity="impervious surface in the tropical band, 2000 (km2)", value=b["impervious_2000_km2"]),
            dict(quantity="impervious surface in the tropical band, 2020 (km2)", value=b["impervious_2020_km2"]),
            dict(quantity="impervious surface added, 2000 to 2020 (km2)", value=b["added_2000_2020_km2"]),
            dict(quantity="share of the addition inside delineated agglomerations (%)",
                 value=b["added_inside_agglomerations_pct"]),
            dict(quantity="impervious surface added outside delineated agglomerations, 2000 to 2020 (km2)",
                 value=b["added_outside_agglomerations_km2"])]
    labels = {"below_0.05": "below 0.05", "0.05_to_0.10": "0.05 to 0.10", "0.10_to_0.20": "0.10 to 0.20",
              "0.20_or_more": "0.20 or more (clusters below 5 km2)"}
    for k, lab in labels.items():
        v = b["added_outside_by_2020_builtup_fraction"][k]
        rows.append(dict(quantity=f"added outside agglomerations in cells with 2020 built-up fraction {lab} (km2)",
                         value=v["km2"]))
        rows.append(dict(quantity=f"added outside agglomerations in cells with 2020 built-up fraction {lab} (% of outside)",
                         value=v["pct_of_outside"]))
    return pd.DataFrame(rows)


def outside_tables():
    o = json.loads((TABLE_DIR / "outside_agglomerations.json").read_text())
    groups = [("species of concern", o["species_of_concern"]), ("threatened", o["threatened"]),
              ("narrow-range", o["narrow_range"])]
    groups += [(f"threatened {k.lower()}", v) for k, v in o["threatened_by_taxon"].items()]
    names = {"species": "species with range in the tropical band", "reached_inside": "reached inside agglomerations",
             "reached_outside": "reached outside agglomerations", "reached_either": "reached inside or outside",
             "reached_outside_only": "reached only outside agglomerations",
             "above_1pct_inside": "at least 1 percent of range converted inside agglomerations",
             "above_1pct_either": "at least 1 percent of range converted inside and outside together",
             "converted_inside_km2": "range converted inside agglomerations (km2)",
             "converted_outside_km2": "range converted outside agglomerations (km2)",
             "median_outside_km2_outside_only": "median range converted per species reached only outside agglomerations (km2)"}
    sp = pd.DataFrame([dict(group=g, **{names[k]: v[k] for k in names}) for g, v in groups])
    bu = pd.DataFrame([dict(built_up_fraction_2020=c, land_added_km2=o["land_added_outside_by_class_km2"][c],
                            range_converted_km2=o["converted_outside_by_class_km2"][c],
                            range_converted_per_km2_added=o["intensity_outside_by_class"][c])
                       for c in o["land_added_outside_by_class_km2"]])
    bu = pd.concat([bu, pd.DataFrame([dict(built_up_fraction_2020="all outside agglomerations",
                                           land_added_km2=o["land_added_outside_km2"],
                                           range_converted_km2=o["species_of_concern"]["converted_outside_km2"],
                                           range_converted_per_km2_added=o["intensity_outside"]),
                                      dict(built_up_fraction_2020="inside agglomerations",
                                           land_added_km2=o["land_added_inside_km2"],
                                           range_converted_km2=o["species_of_concern"]["converted_inside_km2"],
                                           range_converted_per_km2_added=o["intensity_inside"])])], ignore_index=True)
    return sp, bu


def write_book(path, sheets, readme):
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        pd.DataFrame(readme, columns=["sheet", "content"]).to_excel(w, sheet_name="README", index=False)
        w.sheets["README"].set_column(0, 0, 32)
        w.sheets["README"].set_column(1, 1, 110)
        for name, d in sheets:
            d.to_excel(w, sheet_name=name, index=False)
            w.sheets[name].freeze_panes(1, 0)


def build(out=OUT):
    import extended_data as ed
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    temporal = [(f.split(".")[0], pd.read_csv(T / f)) for f in TEMPORAL_FILES]
    temporal.append(("band_totals", band_totals()))
    sp_out, bu_out = outside_tables()
    temporal += [("outside_species", sp_out), ("outside_builtup", bu_out)]
    temporal.insert(1, ("temporal_fits", temporal_fits()))
    write_book(out / "Supplementary_Data_3.xlsx", temporal, [(n, TEMPORAL_TEXT[n]) for n, _ in temporal])
    d = tn.table()[CITY_COLUMNS].sort_values("city_id")
    cols = pd.DataFrame({"column": CITY_COLUMNS, "definition": [CITY_DEFINITIONS[c] for c in CITY_COLUMNS]})
    write_book(out / "Supplementary_Data_1.xlsx", [("Agglomerations", d), ("Columns", cols)],
               [("Agglomerations", "Per-agglomeration analysis table, 7,229 rows"),
                ("Columns", "Definition of each column")])
    sheets = [("ED Table 1", TABLES[7]()), ("ED Table 2", ed.robustness_table())]
    readme = [("ED Table 1", "Primary estimates under each response and specification (Extended Data Table 1)"),
              ("ED Table 2", "Robustness of the size exponent and setting coefficient (Extended Data Table 2)")]
    for name, k, text in SHEETS:
        t = TABLES[k]()
        sheets.append((name, t))
        readme.append((name, text))
    for name, f, text in EXTRA_SHEETS:
        sheets.append((name, f()))
        readme.append((name, text))
    write_book(out / "Supplementary_Data_2.xlsx", sheets, readme)
    return dict(agglomerations=d.shape, sheets_sd2=len(sheets), out=str(out))


if __name__ == "__main__":
    print(build(sys.argv[1]) if len(sys.argv) > 1 else build())
