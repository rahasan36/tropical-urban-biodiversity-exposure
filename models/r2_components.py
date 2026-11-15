"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tropics_config import TABLE_DIR

OUT = TABLE_DIR / "r2_components.csv"


def _blocks(lon, lat, deg=2.0):
    return (np.floor(np.asarray(lat) / deg).astype(int) * 1000
            + np.floor(np.asarray(lon) / deg).astype(int))


def components(d, response, size="urban_km2_base",
               setting="setting_footprint", spatial=True, block_deg=2.0):
    x = d.copy()
    x = x[(x[response] > 0) & (x[size] > 0)].dropna(subset=[response, size])
    x["y"] = np.log10(x[response])
    x["logS"] = np.log10(x[size])
    rhs = ["logS"]
    if setting in x.columns and (x[setting] > 0).any():
        x["logSet"] = np.log10(x[setting].clip(lower=0.5))
        rhs.append("logSet")
    if "biome" in x.columns and x["biome"].nunique() > 1:
        x["_biome"] = x["biome"].astype(str)
        rhs.append("C(_biome)")
    formula = "y ~ " + " + ".join(rhs)
    x["_country"] = x["country"].astype(str)
    vc = None
    if spatial and {"lon", "lat"} <= set(x.columns):
        x["_block"] = _blocks(x.lon.to_numpy(), x.lat.to_numpy(), block_deg)
        vc = {"block": "0 + C(_block)"}
    md = (smf.mixedlm(formula, x, groups=x["_country"], vc_formula=vc,
                      re_formula="1")
          if vc else smf.mixedlm(formula, x, groups=x["_country"]))
    f = md.fit(reml=True, method="bfgs")

    fe = np.asarray(f.model.exog, float) @ np.asarray(f.fe_params, float)
    var_f = float(np.var(fe, ddof=0))
    cov_re = np.atleast_2d(np.asarray(f.cov_re, float))
    var_country = float(cov_re[0, 0]) if cov_re.size else 0.0
    var_block = 0.0
    if f.vcomp is not None and len(np.atleast_1d(f.vcomp)):
        var_block = float(np.atleast_1d(f.vcomp)[0])
    var_resid = float(f.scale)
    total = var_f + var_country + var_block + var_resid
    return dict(
        response=response, n=int(len(x)), formula=formula,
        var_fixed=var_f, var_country=var_country, var_block=var_block,
        var_residual=var_resid,
        R2_marginal=var_f / total,
        R2_conditional=(var_f + var_country + var_block) / total,
        share_country=var_country / total, share_block=var_block / total,
        share_residual=var_resid / total,
        icc_country=var_country / (var_country + var_block + var_resid))


def _selftest():
    rng = np.random.default_rng(3)
    n_c, per = 60, 60
    n = n_c * per
    country = np.repeat(np.arange(n_c), per)
    logS = rng.normal(0, 1, n)
    a = rng.normal(0, np.sqrt(0.25), n_c)[country]
    e = rng.normal(0, np.sqrt(0.25), n)
    y = np.sqrt(0.5) * logS + a + e
    d = pd.DataFrame(dict(
        cost=10.0 ** y, urban_km2_base=10.0 ** logS,
        setting_footprint=1.0, biome="b",
        country=country.astype(str),
        lon=rng.uniform(-60, 120, n), lat=rng.uniform(-23, 23, n)))
    r = components(d, "cost", setting="none", spatial=False)
    assert abs(r["R2_marginal"] - 0.5) < 0.06, r["R2_marginal"]
    assert abs(r["R2_conditional"] - 0.75) < 0.06, r["R2_conditional"]
    assert abs(r["var_country"] - 0.25) < 0.06, r["var_country"]
    assert abs(r["var_residual"] - 0.25) < 0.06, r["var_residual"]
    print(f"ok  marginal {r['R2_marginal']:.3f} against 0.500, "
          f"conditional {r['R2_conditional']:.3f} against 0.750, "
          f"country variance {r['var_country']:.3f} against 0.250")


def run():
    import tropics_numbers as tn
    import tropics_models as p4
    d = p4.table()
    x = d.copy()
    x["threat_per_km2"] = x.cost_threat_km2 / x.urban_added_km2
    x["narrow_per_km2"] = x.cost_narrow_km2 / x.urban_added_km2
    rows = []
    for label, col in (("Exposure intensity", "cost_per_km2"),
                       ("Rarity weighted", "rarity_per_km2"),
                       ("Threatened", "threat_per_km2"),
                       ("Narrow range", "narrow_per_km2")):
        r = components(x, col)
        r["label"] = label
        rows.append(r)
        print(f"{label:<20} n={r['n']:<6} marginal {r['R2_marginal']:.3f}  "
              f"conditional {r['R2_conditional']:.3f}  "
              f"country {r['share_country']:.3f}  block {r['share_block']:.3f}",
              flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    _selftest()
    t = run()
    t.to_csv(OUT, index=False)
    print("\nwritten", OUT)
