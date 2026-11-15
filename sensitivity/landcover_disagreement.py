"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tropics_numbers as tn
import tropics_refit as trf
from tropics_config import TABLE_DIR

OUT_GEO = TABLE_DIR / "landcover_disagreement_geography.csv"
OUT_FIT = TABLE_DIR / "landcover_disagreement_fits.csv"


def frame():
    d = tn.analysis_sample()
    d = d[d.hab_sampled_km2.notna() & (d.hab_sampled_km2 > 0)].copy()
    d["disagree"] = (d.hab_impervious_km2 / d.hab_sampled_km2).clip(0, 1)
    d["size_class"] = pd.cut(d.urban_km2_base,
                             [0, 10, 50, 200, 1e9],
                             labels=["under 10", "10 to 50", "50 to 200",
                                     "over 200"])
    return d


def geography(d):
    def block(g, key):
        return (g.groupby(key, observed=True)
                 .apply(lambda x: pd.Series(dict(
                     n=len(x),
                     land_added_km2=x.urban_added_km2.sum(),
                     disagree_mean=np.average(
                         x.disagree, weights=x.hab_sampled_km2),
                     disagree_median=x.disagree.median(),
                     disagree_p90=x.disagree.quantile(0.90))),
                     include_groups=False)
                 .reset_index().rename(columns={key: "group"})
                 .assign(grouping=key))
    parts = [block(d, "realm"), block(d, "size_class")]
    if "continent" in d.columns:
        parts.append(block(d, "continent"))
    out = pd.concat(parts, ignore_index=True)
    return out[["grouping", "group", "n", "land_added_km2", "disagree_mean",
                "disagree_median", "disagree_p90"]]


def fits(d):
    rows = []
    x = d.copy()
    x["threat_per_km2"] = x.cost_threat_km2 / x.urban_added_km2
    specs = [("Exposure intensity", "cost_per_km2"),
             ("Rarity weighted", "rarity_per_km2")]
    cuts = [("all cities with land cover", 1.01),
            ("disagreement under 50%", 0.50),
            ("disagreement under 25%", 0.25),
            ("disagreement under 10%", 0.10)]
    for label, thr in cuts:
        sub = x[x.disagree < thr]
        for rlabel, col in specs:
            r = trf.fit_scaling(sub, response=col, size="urban_km2_base",
                                setting="setting_footprint", spatial=True)
            rows.append(dict(subset=label, cut=thr, response=rlabel,
                             n=int(r["n"]), beta=float(r["beta"]),
                             lo=float(r["ci_lower"]), hi=float(r["ci_upper"]),
                             gamma=float(r.get("gamma_setting", np.nan)),
                             p=float(r["p_value"])))
    return pd.DataFrame(rows)


def _selftest():
    d = frame()
    assert d.disagree.between(0, 1).all()
    assert d.size_class.notna().all(), "a city fell outside every size class"
    assert int(d.size_class.value_counts().sum()) == len(d)
    w = np.average(d.disagree, weights=d.hab_sampled_km2)
    assert 0.0 <= w <= 1.0
    print(f"ok  n={len(d)}  area-weighted disagreement {w:.4f}")


if __name__ == "__main__":
    _selftest()
    d = frame()
    g = geography(d)
    g.to_csv(OUT_GEO, index=False)
    pd.set_option("display.width", 200)
    print(g.to_string(index=False))
    f = fits(d)
    f.to_csv(OUT_FIT, index=False)
    print()
    print(f.to_string(index=False))
