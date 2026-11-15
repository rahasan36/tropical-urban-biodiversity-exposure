"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json
from pathlib import Path

import tropics_scaling as ts
from tropics_config import TABLE_DIR, THREAT_CATS
from tropics_cost import PAIRS_PATH

MOIST_DRY = ["Tropical & Subtropical Moist Broadleaf Forests",
             "Tropical & Subtropical Dry Broadleaf Forests"]


def _fit(df, response, label, spatial=True, **kw):
    r = ts.fit_scaling(df, response=response, spatial=spatial, **kw)
    d = r.pop("_d"); resid = r.pop("_resid")
    coords = ts.project_equal_area(d.lon.to_numpy(), d.lat.to_numpy())
    I, z, p = ts.morans_i(coords, resid)
    r.update(label=label, response=response, moran_I=I, moran_z=z, moran_p=p)
    return r


def rebuild_costs_excluding(taxa=(), only_threat=False, only_narrow=False):
    import tropics_assemble as ta
    sp, pairs, setting = ta.load_pairs()
    tropical = set(pairs.sp_i.unique()).union(set(setting.sp_i.unique()))
    sp, thr = ta.apply_flags(sp, tropical)
    keep = ~sp.taxon.isin(list(taxa))
    sel = sp[keep].copy()
    if only_threat:
        sel = sel[sel.is_threat]
    elif only_narrow:
        sel = sel[sel.is_narrow]
    else:
        sel = sel[sel.is_threat | sel.is_narrow]
    p = pairs[pairs.sp_i.isin(set(sel.sp_i))]
    p = p.merge(sel[["sp_i", "range_km2"]], on="sp_i", how="left")
    cost = p.groupby("city_id").conv_km2.sum().rename("cost_km2")
    rar = (p.conv_km2 / p.range_km2.clip(lower=1e-6)).groupby(p.city_id).sum() \
            .rename("rarity_converted")
    return cost, rar


def with_costs(df, cost, rar):
    d = df.drop(columns=["cost_km2", "rarity_converted", "cost_per_km2",
                         "rarity_per_km2"], errors="ignore").merge(
        pd.concat([cost, rar], axis=1).reset_index(), on="city_id", how="left")
    d[["cost_km2", "rarity_converted"]] = d[["cost_km2", "rarity_converted"]].fillna(0.0)
    d["cost_per_km2"] = np.where(d.urban_added_km2 > 0,
                                 d.cost_km2 / d.urban_added_km2, np.nan)
    d["rarity_per_km2"] = np.where(d.urban_added_km2 > 0,
                                   d.rarity_converted / d.urban_added_km2, np.nan)
    return d


def run_all(csv=None):
    csv = csv or TABLE_DIR / "cities_cost.csv"
    df = pd.read_csv(csv)
    rows = []

    for sp_flag, tag in ((False, "no spatial term"), (True, "spatial block RE")):
        for resp in ("cost_per_km2", "rarity_per_km2"):
            rows.append(_fit(df, resp, f"headline, {tag}", spatial=sp_flag))

    rows.append(_fit(df, "cost_per_km2", "no setting covariate",
                     setting="__none__"))

    for floor in (10.0, 25.0, 50.0):
        rows.append(_fit(df[df.urban_km2_final >= floor], "cost_per_km2",
                         f"size floor {floor:.0f} km2"))

    rows.append(_fit(df[df.biome.isin(MOIST_DRY)], "cost_per_km2",
                     "moist + dry broadleaf biomes only"))

    for cont, sub in df.groupby("continent"):
        if len(sub) >= 200 and sub.country.nunique() > 2:
            rows.append(_fit(sub, "cost_per_km2", f"continent: {cont}"))

    variants = [("threatened species only", dict(only_threat=True)),
                ("narrow-range species only", dict(only_narrow=True)),
                ("excluding birds", dict(taxa=("AVES",))),
                ("excluding birds and mammals", dict(taxa=("AVES", "MAMMALIA")))]
    for label, kw in variants:
        cost, rar = rebuild_costs_excluding(**kw)
        d = with_costs(df, cost, rar)
        rows.append(_fit(d, "cost_per_km2", label))
        rows.append(_fit(d, "rarity_per_km2", label))

    res = pd.DataFrame(rows)
    cols = ["label", "response", "model", "n", "beta", "se", "ci_lower",
            "ci_upper", "beta_std", "p_value", "R2", "gamma_setting",
            "vs_one", "vs_zero", "verdict", "moran_I", "moran_p", "resid_kind"]
    res = res[[c for c in cols if c in res.columns]]
    res.to_csv(TABLE_DIR / "scaling_results.csv", index=False)

    conc = [ts.concentration(df, top_pct=p) for p in (0.1, 0.5, 1.0, 5.0, 10.0)]
    pd.DataFrame(conc).to_csv(TABLE_DIR / "concentration.csv", index=False)
    return res, pd.DataFrame(conc)


if __name__ == "__main__":
    res, conc = run_all()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(conc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
