"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tropics_config import SUPP_TABLE_DIR, TABLE_DIR

SRC = TABLE_DIR / "si"
OUT = SUPP_TABLE_DIR


def n0(v):
    return "" if pd.isna(v) else f"{int(round(float(v))):,}"


def f(v, d=3, sign=False):
    if pd.isna(v):
        return ""
    s = f"{float(v):+.{d}f}" if sign else f"{float(v):.{d}f}"
    return s


PVAL_FLOOR = 1e-300


def pval(v):
    if pd.isna(v):
        return ""
    v = float(v)
    if v >= 0.001:
        return f"{v:.3f}"
    if v < PVAL_FLOOR:
        return f"< {PVAL_FLOOR:.0e}"
    return f"{v:.1e}"


def ci(lo, hi, d=4):
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"{float(lo):.{d}f} to {float(hi):.{d}f}"


def fk(v, d=1):
    return "" if pd.isna(v) else f"{float(v):,.{d}f}"


def pct(v, d=1):
    return "" if pd.isna(v) else f"{float(v)*100:.{d}f}%"


def read(stem):
    hits = sorted(SRC.glob(f"{stem}_*.csv"))
    if not hits:
        raise FileNotFoundError(f"no source for {stem} under {SRC}")
    return pd.read_csv(hits[0])


DOC_NUMBER = {1: 1, 14: 2, 2: 3, 10: None, 9: 4, 3: 5, 6: 6, 7: None, 13: None,
              8: None, 12: None, 15: 7, 19: 8, 17: 11, 18: None, 4: 12, 11: 13,
              5: 15}


def write(num, df):
    a = num
    num = DOC_NUMBER[num] if isinstance(num, int) else num
    OUT.mkdir(parents=True, exist_ok=True)
    if num is None:
        (OUT / "_unprinted").mkdir(exist_ok=True)
        p = OUT / "_unprinted" / f"analysis_table{a}.csv"
        df.to_csv(p, index=False)
        print(f"analysis table {a} -> _unprinted/{p.name}", flush=True)
        return df
    p = OUT / f"tableS{num}.csv"
    df.to_csv(p, index=False)
    print(f"S{num}  {df.shape[0]:>3} rows x {df.shape[1]} cols  -> {p.name}",
          flush=True)
    return df


LABELS = {
    "urban_km2_base": "2000 extent (km²)",
    "urban_km2_final": "2020 extent (km²)",
    "urban_added_km2": "Urban land added (km²)",
    "cost_km2": "Converted range, species of concern (km²)",
    "cost_threat_km2": "Converted range, threatened (km²)",
    "cost_narrow_km2": "Converted range, narrow range (km²)",
    "rarity_converted": "Rarity-weighted converted range (10⁻³)",
    "cost_per_km2": "Exposure intensity (km² km⁻²)",
    "rarity_per_km2": "Rarity-weighted exposure intensity (10⁻⁵ km⁻²)",
    "setting_footprint": "Setting richness (threatened species)",
    "form_infill_share": "Infill share",
    "form_edge_share": "Edge share",
    "form_leapfrog_share": "Leapfrog share",
}


S1_SCALE = {"rarity_converted": 1e3, "rarity_per_km2": 1e5}


def s1():
    d = read("tableS1")
    keep = [c for c in d.variable if c in LABELS]
    d = d[d.variable.isin(keep)].copy()
    sc = np.array([S1_SCALE.get(v, 1.0) for v in d.variable])
    dec = [3 if S1_SCALE.get(v) else 1 for v in d.variable]
    def col(vals):
        return [f(v * k, n) for v, k, n in zip(vals, sc, dec)]
    out = pd.DataFrame({
        "Variable": [LABELS[v] for v in d.variable],
        "n": [n0(v) for v in d.n],
        "Min": col(d["min"]),
        "25th": col(d.p25),
        "Median": col(d["median"]),
        "75th": col(d.p75),
        "Max": [n0(v) if k == 1.0 else fk(v * k)
                for v, k in zip(d["max"], sc)],
        "Total": ["" if pd.isna(v) else (n0(v) if k == 1.0 else fk(v * k))
                  for v, k in zip(d.total, sc)],
    })
    return write(1, out)


def s2():
    d = read("tableS2")
    out = pd.DataFrame({
        "Response": d.response,
        "Specification": [str(v).replace(" (as drafted)", "")
                          for v in d.specification],
        "n": [n0(v) for v in d.n],
        "Exponent": [f(v, 4, sign=True) for v in d.beta],
        "95% CI": [ci(a, b) for a, b in zip(d.ci_lower, d.ci_upper)],
        "P": [pval(v) for v in d.p_value],
        "Setting coef.": [f(v, 3, sign=True) for v in d.gamma_setting],
        "R² (cond. resid.)": [f(v, 3) for v in d.R2],
    })
    return write(2, out)


def s3():
    d = read("tableS3")
    out = pd.DataFrame({
        "Level": d.level,
        "Region": d.group,
        "Cities": [n0(v) for v in d.n],
        "Exponent": [f(v, 4, sign=True) for v in d.beta],
        "95% CI": [ci(a, b) for a, b in zip(d.ci_lower, d.ci_upper)],
        "Setting coef.": [f(v, 3, sign=True) for v in d.gamma_setting],
        "R2": [f(v, 3) for v in d.R2],
        "Converted range (km²)": [n0(v) for v in d.total_cost],
        "Excludes zero": ["yes" if v else "no" for v in d.excludes_zero],
    })
    return write(3, out)


def s4():
    d = read("tableS4")
    out = pd.DataFrame({
        "Configuration": d.label,
        "Cities": [n0(v) for v in d.n_cities],
        "Converted range (km²)": [n0(v) for v in d.cost_km2],
        "Mean setting": [f(v, 1) for v in d.mean_setting],
        "Narrow threshold (km²)": [n0(v) for v in d.narrow_threshold_km2],
        "Exponent": [f(v, 4, sign=True) for v in d.beta],
        "95% CI": [ci(a, b) for a, b in zip(d.ci_lower, d.ci_upper)],
        "Excludes zero": ["yes" if v else "no" for v in d.excludes_zero],
    })
    return write(4, out)


def s5():
    d = read("tableS7")
    cols = {c.lower(): c for c in d.columns}
    lab = cols.get("label", d.columns[0])
    out = pd.DataFrame({
        "Size variable or restriction": d[lab],
        "n": [n0(v) for v in d[cols.get("n", "n")]],
        "Exponent": [f(v, 4, sign=True) for v in d[cols.get("beta", "beta")]],
        "95% CI": [ci(a, b) for a, b in zip(d[cols.get("ci_lower", "ci_lower")],
                                            d[cols.get("ci_upper", "ci_upper")])],
        "P": [pval(v) for v in d[cols.get("p_value", "p_value")]],
        "Setting coef.": [f(v, 3, sign=True)
                          for v in d[cols.get("gamma_setting", "gamma_setting")]],
    })
    return write(5, out)


S6_COLS = {"block": "Block", "setting_from": "Setting measured on",
           "exposure_of": "Predicting", "n": "n", "statistic": "Statistic",
           "se": "SE", "p_value": "P", "beta_size": "Size exponent"}


def s6():
    d = read("tableS5")
    cor = d[d.block == "rank correlation between settings"]
    pred = d[d.block == "setting predicts the excluded groups"]
    groups = ["birds", "mammals", "herps"]
    rho = {}
    for r in cor.itertuples():
        rho[(r.setting_from, r.exposure_of)] = rho[(r.exposure_of, r.setting_from)] = r.statistic
    rows = []
    for r in pred.itertuples():
        row = {"Setting measured on": r.setting_from, "Predicting": r.exposure_of}
        for g in groups:
            row[f"Spearman with {g}"] = "" if g == r.setting_from else f(rho[(r.setting_from, g)], 3, sign=True)
        row.update({"Setting coefficient": f(r.statistic, 3, sign=True), "SE": f(r.se, 3),
                    "P": pval(r.p_value), "Size exponent": f(r.beta_size, 4, sign=True)})
        rows.append(row)
    return write(6, pd.DataFrame(rows))


def s7():
    d = read("tableS6")
    out = pd.DataFrame({
        "Country": d.country,
        "Agglomerations": [n0(v) for v in d.n_cities],
        "Urban land added (km²)": [n0(v) for v in d.urban_added_km2],
        "Converted range (km²)": [n0(v) for v in d.cost_km2],
        "Rarity-weighted converted range": [f(v, 2) for v in d.rarity_converted],
        "Exposure intensity (km² km⁻²)": [f(v, 1) for v in d.exposure_per_km2],
        "Share of converted range": [f(v, 1) + "%" for v in
                                     d.share_of_converted_range_pct],
        "Share of land added": [f(v, 1) + "%" for v in d.share_of_land_added_pct],
    })
    return write(7, out)


def s8():
    d = read("tableS8")
    out = d.copy()
    ren = {c: (f"alpha = {float(c):+.1f}" if _isnum(c) else c.capitalize())
           for c in out.columns}
    for c in out.columns:
        if _isnum(c):
            out[c] = [f(v, 1, sign=True) for v in out[c]]
    return write(8, out.rename(columns=ren))


def _isnum(c):
    try:
        float(c); return True
    except ValueError:
        return False


def s9():
    d = read("tableS9")
    out = pd.DataFrame({
        "Block size": [f"{float(v):.0f}°" for v in d.block_deg],
        "Buffer (km)": [n0(v) for v in d.buffer_km],
        "Blocks": [n0(v) for v in d.n_blocks],
        "Specification": d.model,
        "Agglomerations scored": [n0(v) for v in d.n_scored],
        "Out-of-sample R\u00b2": [f(v, 4, sign=True) for v in d.R2_oos],
    })
    return write(9, out)


def s10():
    d = read("tableS10")
    out = pd.DataFrame({
        "Response": d.label,
        "n": [n0(v) for v in d.n],
        "Var(fixed)": [f(v, 4) for v in d.var_fixed],
        "Var(country)": [f(v, 4) for v in d.var_country],
        "Var(2° block)": [f(v, 4) for v in d.var_block],
        "Var(residual)": [f(v, 4) for v in d.var_residual],
        "Marginal R2": [f(v, 3) for v in d.R2_marginal],
        "Conditional R2": [f(v, 3) for v in d.R2_conditional],
        "Country share": [f(v, 3) for v in d.share_country],
        "Block share": [f(v, 3) for v in d.share_block],
    })
    return write(10, out)


NOTE_TEXT = {"Taxa the study does not analyse; empty in the current configuration":
             "Taxa the study does not analyze, none in the current configuration",
             "Species with a non-zero converted area; the rest are in the pool through the setting covariate only":
             "Species with a non-zero converted area. The rest are in the pool through the setting covariate only"}


def s11():
    d = read("tableS11")
    led = d[d.block == "ledger"]
    comp = d[d.block == "composition"]
    a = pd.DataFrame({"Step": led.step, "Removed": [n0(v) for v in led.n_removed],
                      "Remaining": [n0(v) for v in led.n_remaining],
                      "Note": [NOTE_TEXT.get(v, v) for v in led.note]})
    b = pd.DataFrame({"Class": [str(v).capitalize() for v in comp.taxon],
                      "Species": [n0(v) for v in comp.n],
                      "Threatened": [n0(v) for v in comp.threatened],
                      "Narrow range": [n0(v) for v in comp.narrow_range],
                      "Median global range (km²)": [n0(v) for v in comp.median_range_km2]})
    write("13", a)
    return write("14", b)


def s12():
    d = read("tableS12")
    out = pd.DataFrame({
        "Response family": d.family,
        "Weighting": d.weighting,
        "Sample": d["sample"],
        "n": [n0(v) for v in d.n],
        "Exponent": [f(v, 4, sign=True) for v in d.beta],
        "95% CI": [ci(a, b) for a, b in zip(d.lo, d.hi)],
        "P": [pval(v) for v in d.p],
        "R2": [f(v, 3) for v in d.R2],
    })
    return write(12, out)


def s13():
    d = read("tableS13")
    out = pd.DataFrame({
        "Country dropped": d.dropped,
        "n kept": [n0(v) for v in d.n_cities],
        "n dropped": [n0(v) for v in d.n_dropped],
        "Measured, dispersed (%)": [f(v, 2, sign=True) for v in d.measured_dispersed],
        "Measured, concentrated (%)": [f(v, 2, sign=True) for v in d.measured_concentrated],
        "Predicted, dispersed (%)": [f(v, 2, sign=True) for v in d.predicted_dispersed],
        "Predicted, concentrated (%)": [f(v, 2, sign=True) for v in d.predicted_concentrated],
        "Paired, dispersed (pts)": [f(v, 2, sign=True) for v in d.paired_dispersed],
        "Paired, concentrated (pts)": [f(v, 2, sign=True) for v in d.paired_concentrated],
    })
    return write(13, out)


def s14():
    d = read("tableS14")
    geo = d[d.block == "geography"]
    fit = d[d.block == "estimates"]
    a = pd.DataFrame({
        "Grouping": [str(v).replace("_", " ") for v in geo.grouping],
        "Group": geo.group,
        "n": [n0(v) for v in geo.n],
        "Land added (km²)": [n0(v) for v in geo.land_added_km2],
        "Mean": [pct(v) for v in geo.disagree_mean],
        "Median": [pct(v) for v in geo.disagree_median],
        "90th": [pct(v) for v in geo.disagree_p90],
    })
    b = pd.DataFrame({
        "Subset": fit.subset,
        "Response": fit.response,
        "n": [n0(v) for v in fit.n],
        "Exponent": [f(v, 4, sign=True) for v in fit.beta],
        "95% CI": [ci(x, y) for x, y in zip(fit.lo, fit.hi)],
        "Setting coef.": [f(v, 3, sign=True) for v in fit.gamma],
        "P": [pval(v) for v in fit.p],
    })
    write(14, a)
    write("16", b)
    return a


def s15():
    d = read("tableS15")
    out = pd.DataFrame({
        "Grouping": d.block.map({"responses": "species subset", "regions": "region"}).fillna(d.block),
        "Subset": d.subset,
        "n": [n0(v) for v in d.n],
        "Unweighted exponent": [f(v, 4, sign=True) for v in d.beta_unweighted],
        "Rarity-weighted exponent": [f(v, 4, sign=True) for v in d.beta_rarity],
        "Mean range exponent": [f(v, 4, sign=True) for v in d.beta_mean_range],
        "Reconciliation residual": [f(v, 4, sign=True) for v in d.identity_residual],
        "Median mean range (km²)": [n0(v) for v in d.median_mean_range_km2],
    })
    return write(15, out)


def s16():
    d = read("tableS16")
    out = pd.DataFrame({
        "Configuration": d.label,
        "Built-up fraction": [f(v, 2) for v in d.fraction],
        "Link (cells)": [n0(v) for v in d.link],
        "Floor (km²)": [f(v, 0) for v in d.floor],
        "Cities": [n0(v) for v in d.n_cities],
        "Cities vs headline": [pct(v) for v in d.n_vs_headline],
        "Narrow threshold (km²)": [n0(v) for v in d.narrow_threshold_km2],
        "Threshold vs headline": [pct(v) for v in d.threshold_vs_headline],
        "Exponent": [f(v, 4, sign=True) for v in d.beta],
        "95% CI": [ci(a, b) for a, b in zip(d.ci_lower, d.ci_upper)],
        "Exponent vs headline": [f(v, 4, sign=True) for v in d.beta_minus_headline],
    })
    return write(16, out)


def s17():
    d = pd.read_csv(TABLE_DIR / "resolution_test_v2.csv")
    order = {"centroid": 0, "touches": 1, "within": 2}
    d = (d.assign(_l=d.lattice_arcsec, _r=[order.get(v, 9) for v in d["rule"]])
           .sort_values(["_l", "_r"]).drop(columns=["_l", "_r"])
           .reset_index(drop=True))
    rule_name = {"centroid": "cell centre inside the range",
                 "touches": "range touches the cell",
                 "within": "cell wholly inside the range"}
    out = pd.DataFrame({
        "Lattice": [f"{int(v)} arcsec" for v in d.lattice_arcsec],
        "Boundary rule": [rule_name.get(v, v) for v in d["rule"]],
        "Cells": [n0(v) for v in d.n_cells],
        "Converted range (km²)": [n0(v) for v in d.converted_km2],
        "vs reference": [pct(v, 2) for v in d.converted_vs_headline],
        "n": [n0(v) for v in d.n],
        "Exponent": [f(v, 4, sign=True) for v in d.beta],
        "95% CI": [ci(a, b) for a, b in zip(d.lo, d.hi)],
        "P": [pval(v) for v in d.p],
        "Setting coef.": [f(v, 3, sign=True) for v in d.gamma],
        "Rarity exponent": [f(v, 4, sign=True) for v in d.beta_rarity],
    })
    return write(17, out)


def s18():
    d = pd.read_csv(TABLE_DIR / "threshold_isolation.csv")
    out = pd.DataFrame({
        "Threshold taken from": d.configuration,
        "Narrow threshold (km²)": [n0(v) for v in d.narrow_threshold_km2],
        "n": [n0(v) for v in d.n],
        "Converted range (km²)": [n0(v) for v in d.converted_km2],
        "Exponent": [f(v, 5, sign=True) for v in d.beta],
        "95% CI": [ci(a, b, 5) for a, b in zip(d.ci_lower, d.ci_upper)],
        "vs published threshold": [f(v, 5, sign=True)
                                   for v in d.beta_minus_headline],
        "Setting coef.": [f(v, 3, sign=True) for v in d.gamma_setting],
    })
    return write(18, out)


def s19():
    d = pd.read_csv(TABLE_DIR / "species_boundary_shares.csv")
    d = d.sort_values("share_published", ascending=False).reset_index(drop=True)
    out = pd.DataFrame({
        "Species": d.sci,
        "Category": d.category,
        "Global range (km²)": [fk(v) for v in d.range_km2],
        "Published share": [pct(v, 1) for v in d.share_published],
        "Cell wholly inside": [pct(v, 1) for v in d.share_within],
        "Cell centre inside": [pct(v, 1) for v in d.share_centroid],
        "Range touches cell": [pct(v, 1) for v in d.share_touches],
        "Span (pp)": [f(v, 1) for v in d.share_span_pp],
    })
    return write(19, out)


JOBS = {(f"S{DOC_NUMBER[i]}" if DOC_NUMBER[i] else f"unprinted_{i}"): globals()[f"s{i}"]
        for i in DOC_NUMBER}


def _selftest():
    assert n0(7207.0) == "7,207"
    assert n0(np.nan) == ""
    assert f(-0.00898, 4, sign=True) == "-0.0090"
    assert f(0.0919, 4, sign=True) == "+0.0919"
    assert pval(4.1765e-3) == "0.004"
    assert pval(2.36e-13) == "2.4e-13"
    assert ci(-0.0121, -0.0059) == "-0.0121 to -0.0059"
    assert pct(0.2944) == "29.4%"
    assert fk(30337.43) == "30,337.4"
    assert pval(0.0) == "< 1e-300"
    assert pval(1e-40) == "1.0e-40"
    assert pval(1.3e-27) == "1.3e-27"
    assert pval(2.4e-13) == "2.4e-13"
    print("ok  formatters round-trip")


if __name__ == "__main__":
    _selftest()
    for k in (sys.argv[1:] or [f"S{i}" for i in sorted(v for v in DOC_NUMBER.values() if v)]):
        JOBS[k]()
