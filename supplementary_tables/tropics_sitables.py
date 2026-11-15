"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd

from tropics_config import TABLE_DIR
import tropics_numbers as tn
import tropics_refit as rf
from tropics_scaling import fit_scaling

OUT = TABLE_DIR / "si"
THREAT = ("CR", "EN", "VU")


def _plan():
    return json.loads((TABLE_DIR / "plan4.json").read_text())


def _w(name, df):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    return df


ADDITIVE = ("urban_km2_base", "urban_km2_final", "urban_added_km2",
            "cost_km2", "cost_narrow_km2", "rarity_converted")


def s1_city_summary():
    d = tn.analysis_sample()
    cols = ["urban_km2_base", "urban_km2_final", "urban_added_km2",
            "cost_km2", "cost_narrow_km2", "rarity_converted",
            "setting_footprint", "setting_richness", "cost_per_km2",
            "rarity_per_km2", "form_infill_share", "form_edge_share",
            "form_leapfrog_share"]
    cols = [c for c in cols if c in d.columns]
    r = []
    for c in cols:
        v = pd.to_numeric(d[c], errors="coerce").dropna()
        r.append(dict(variable=c, n=int(v.size), mean=float(v.mean()),
                      sd=float(v.std()), min=float(v.min()),
                      p25=float(v.quantile(.25)), median=float(v.median()),
                      p75=float(v.quantile(.75)), max=float(v.max()),
                      total=float(v.sum()) if c in ADDITIVE else np.nan))
    return _w("tableS1_city_summary.csv", pd.DataFrame(r))


RESPONSES = [
    ("threatened and narrow range, per km2 added", "cost_per_km2"),
    ("range-rarity weighted, per km2 added", "rarity_per_km2"),
]


def s2_response_definitions():
    d = tn.analysis_sample()
    rows = []
    for lab, resp in RESPONSES:
        for key in ("A", "B", "C"):
            note, kw = rf.SPECS[key]
            if kw["response"] != "cost_per_km2":
                continue
            r = fit_scaling(d, spatial=True, **{**kw, "response": resp})
            rows.append(dict(response=lab, specification=note.split("   ", 1)[-1],
                             size_variable=kw["size"], setting_variable=kw["setting"],
                             n=int(r["n"]), beta=float(r["beta"]),
                             ci_lower=float(r["ci_lower"]), ci_upper=float(r["ci_upper"]),
                             p_value=float(r["p_value"]),
                             gamma_setting=float(r.get("gamma_setting", np.nan)),
                             R2=float(r["R2"])))
    for v in _plan()["rarity_variants"]:
        rows.append(dict(response=v["label"], specification="size 2000, setting footprint",
                         size_variable="urban_km2_base",
                         setting_variable="setting_footprint", n=int(v["n"]),
                         beta=float(v["beta"]), ci_lower=float(v["ci"][0]),
                         ci_upper=float(v["ci"][1]), p_value=float(v["p"]),
                         gamma_setting=np.nan, R2=float(v["R2"])))
    return _w("tableS2_response_definitions.csv", pd.DataFrame(rows))


def s3_regions():
    a = pd.read_csv(TABLE_DIR / "regime_fits_realm.csv").assign(level="realm")
    b = pd.read_csv(TABLE_DIR / "regime_fits_biome.csv").assign(level="biome")
    f = pd.concat([a, b], ignore_index=True)
    f["excludes_zero"] = f.ci_lower * f.ci_upper > 0
    return _w("tableS3_realm_biome.csv", f)


def s4_delineation():
    v3 = TABLE_DIR / "sensitivity_delineation_v3.csv"
    src = v3 if v3.exists() else TABLE_DIR / "sensitivity_delineation_v2.csv"
    print(f"  S4 delineation from {src.name}")
    return _w("tableS4_delineation.csv", pd.read_csv(src))


def s5_cross_taxon():
    import tropics_models as p4
    import tropics_si_figures as si
    x = si._taxon_setting_frame()
    cols = [c for c in ("set_birds", "set_mammals", "set_herps") if c in x.columns]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rows.append(dict(block="rank correlation between settings",
                             setting_from=a.replace("set_", ""),
                             exposure_of=b.replace("set_", ""),
                             n=int(len(x)),
                             statistic=float(x[a].corr(x[b], method="spearman")),
                             se=np.nan, p_value=np.nan, beta_size=np.nan))
    for r in _plan()["cross_taxon"]:
        rows.append(dict(block="setting predicts the excluded groups",
                         setting_from=r["setting_from"],
                         exposure_of=r["exposure_of"], n=int(r["n"]),
                         statistic=float(r["gamma"]), se=float(r["gamma_se"]),
                         p_value=float(r["p"]), beta_size=float(r["beta_size"])))
    return _w("tableS5_cross_taxon.csv", pd.DataFrame(rows))


def s6_countries(n=25):
    d = tn.table()
    g = (d.groupby("country")
           .agg(n_cities=("city_id", "size"),
                urban_added_km2=("urban_added_km2", "sum"),
                cost_km2=("cost_km2", "sum"),
                rarity_converted=("rarity_converted", "sum"))
           .reset_index())
    g["exposure_per_km2"] = g.cost_km2 / g.urban_added_km2
    g["share_of_converted_range_pct"] = 100 * g.cost_km2 / g.cost_km2.sum()
    g["share_of_land_added_pct"] = (100 * g.urban_added_km2
                                    / g.urban_added_km2.sum())
    g = g.sort_values("cost_km2", ascending=False).head(n)
    return _w("tableS6_countries.csv", g)


def s7_size_variables():
    return _w("tableS7_size_variables.csv",
              pd.read_csv(TABLE_DIR / "size_variables.csv"))


def s8_counterfactual():
    r = json.loads((TABLE_DIR / "cf_robust.json").read_text())
    alphas = r["alphas"]
    rows = []
    for block, key in (("response", "response"), ("realm", "realm")):
        for name, series in r[key].items():
            for a, v in zip(alphas, series["pct"]):
                rows.append(dict(block=block, group=name, alpha=float(a),
                                 change_pct=float(v)))
    f = pd.DataFrame(rows)
    keep = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    f = f[np.isclose(f.alpha.to_numpy()[:, None], keep).any(1)]
    return _w("tableS8_counterfactual.csv",
              f.pivot_table(index=["block", "group"], columns="alpha",
                            values="change_pct").reset_index())


def _read(name):
    return pd.read_csv(TABLE_DIR / name)


def s9_spatial_cv():
    f = _read("_spatial_cv_buffered.csv")
    f = f[["block_deg", "buffer_km", "n_blocks", "model", "n_scored", "R2_oos"]]
    ladder = ["size only", "setting only", "size + setting",
              "size + setting + biome", "+ country"]
    f = f.assign(_o=f.model.map({m: i for i, m in enumerate(ladder)}))
    assert f._o.notna().all(), sorted(set(f.model) - set(ladder))
    return _w("tableS9_spatial_cv.csv",
              f.sort_values(["block_deg", "buffer_km", "_o"]).drop(columns="_o"))


def s10_variance_components():
    f = _read("r2_components.csv")
    cols = ["label", "n", "var_fixed", "var_country", "var_block",
            "var_residual", "R2_marginal", "R2_conditional", "share_country",
            "share_block", "share_residual", "icc_country"]
    return _w("tableS10_variance_components.csv", f[cols])


def s11_species_ledger():
    led = _read("species_ledger.csv").assign(block="ledger")
    comp = _read("species_pool_composition.csv").assign(block="composition")
    return _w("tableS11_species_ledger.csv",
              pd.concat([led, comp], ignore_index=True))


def s12_weighting_vs_sample():
    return _w("tableS12_weighting_vs_sample.csv",
              _read("weighting_vs_sample.csv"))


def s13_allocation_influence():
    return _w("tableS13_allocation_influence.csv", _read("cf_influence.csv"))


def s14_landcover_disagreement():
    geo = _read("landcover_disagreement_geography.csv").assign(block="geography")
    fit = _read("landcover_disagreement_fits.csv").assign(block="estimates")
    return _w("tableS14_landcover_disagreement.csv",
              pd.concat([geo, fit], ignore_index=True))


def s15_objective_decomposition():
    a = _read("objective_decomposition.csv").assign(block="responses")
    b = _read("driver_groups.csv").assign(block="regions")
    return _w("tableS15_objective_decomposition.csv",
              pd.concat([a, b], ignore_index=True))


def s16_delineation_reconciliation():
    return _w("tableS16_delineation_reconciliation.csv",
              _read("delineation_reconciliation.csv"))


JOBS = dict(s1=s1_city_summary, s2=s2_response_definitions, s3=s3_regions,
            s4=s4_delineation, s5=s5_cross_taxon, s6=s6_countries,
            s7=s7_size_variables, s8=s8_counterfactual,
            s9=s9_spatial_cv, s10=s10_variance_components,
            s11=s11_species_ledger, s12=s12_weighting_vs_sample,
            s13=s13_allocation_influence, s14=s14_landcover_disagreement,
            s15=s15_objective_decomposition,
            s16=s16_delineation_reconciliation)

if __name__ == "__main__":
    import sys
    for k in (sys.argv[1:] or list(JOBS)):
        f = JOBS[k]()
        print(f"{k}  {f.shape[0]:>4} rows x {f.shape[1]} cols", flush=True)
