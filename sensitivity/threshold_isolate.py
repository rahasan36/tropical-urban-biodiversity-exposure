"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tropics_flags as tfl
import tropics_scaling as ts
from tropics_config import (INTERIM_DIR, TABLE_DIR, THREAT_CATS, EXCLUDED_TAXA,
                            NARROW_RANGE_PCTL)

PAIRS = INTERIM_DIR / "species_city_pairs.npz"
_V3 = TABLE_DIR / "sensitivity_delineation_v3.csv"
SENS = _V3 if _V3.exists() else TABLE_DIR / "sensitivity_delineation_v2.csv"
OUT = TABLE_DIR / "threshold_isolation.csv"


def _species():
    z = np.load(PAIRS, allow_pickle=True)
    cat, tax, _sci, area = tfl.flags(PAIRS)
    cat = np.array([str(c).upper() for c in cat])
    tax = np.array([str(t).upper() for t in tax])
    area = np.asarray(area, float)
    keep = ~np.isin(tax, [t.upper() for t in EXCLUDED_TAXA])
    pool = np.zeros(len(cat), bool)
    pool[np.union1d(np.unique(z["pair_sp"]), np.unique(z["set_sp"]))] = True
    pool &= (area > 0) & keep
    is_threat = np.isin(cat, [c.upper() for c in THREAT_CATS]) & keep
    return z, is_threat, pool, area


def run(verbose=True):
    z, is_threat, pool, area = _species()
    ps, pc, pv = z["pair_sp"], z["pair_city"], z["pair_conv"]
    import tropics_numbers as tn
    d = tn.table()
    n_city = int(d.city_id.max()) + 1
    published_thr = float(np.percentile(area[pool], NARROW_RANGE_PCTL))

    def cost_at(thr):
        concern = is_threat | (pool & (area <= thr))
        m = concern[ps] & (pv > 0)
        c = np.bincount(pc[m], weights=pv[m], minlength=n_city)
        return pd.Series(c).reindex(d.city_id).to_numpy()

    def fit_at(thr):
        t = d.copy()
        t["cost_km2"] = cost_at(thr)
        t["cost_per_km2"] = np.where(t.urban_added_km2 > 0,
                                     t.cost_km2 / t.urban_added_km2, np.nan)
        t = t[t.urban_km2_base > 0]
        r = ts.fit_scaling(t, response="cost_per_km2", size="urban_km2_base",
                           setting="setting_footprint", spatial=True)
        return r, float(np.nansum(t.cost_km2))

    base_r, base_total = fit_at(published_thr)
    pub_total = float(d[d.urban_km2_base > 0].cost_km2.sum())
    resid = base_total / pub_total - 1.0

    thresholds = pd.read_csv(SENS)[["label", "narrow_threshold_km2"]]
    thresholds = pd.concat([
        pd.DataFrame([dict(label="published analysis",
                           narrow_threshold_km2=published_thr)]),
        thresholds], ignore_index=True)
    rows = []
    for lab, thr in thresholds.itertuples(index=False):
        r, tot = fit_at(float(thr))
        rows.append(dict(configuration=lab, narrow_threshold_km2=float(thr),
                         n=int(r["n"]), beta=float(r["beta"]),
                         ci_lower=float(r["ci_lower"]), ci_upper=float(r["ci_upper"]),
                         gamma_setting=float(r.get("gamma_setting", np.nan)),
                         converted_km2=tot))
        if verbose:
            print(f"  threshold {thr:9,.0f} km2   beta {r['beta']:+.5f}   "
                  f"converted {tot:,.0f}", flush=True)
    f = pd.DataFrame(rows)
    head = f.iloc[0]
    f["beta_minus_headline"] = f.beta - head.beta
    f.attrs["reproduction_residual"] = resid
    if verbose:
        print(f"\n  rebuild reproduces the published converted total to "
              f"{resid * 1e6:+.1f} parts per million")
    return f, resid


def _selftest():
    f, resid = run(verbose=False)
    g = f.sort_values("narrow_threshold_km2")
    assert (g.converted_km2.diff().dropna() >= -1e-6).all(), \
        g[["narrow_threshold_km2", "converted_km2"]].to_dict("records")
    assert abs(resid) < 1e-4, resid
    span = f.beta.max() - f.beta.min()
    print(f"ok  converted total rises with the threshold; rebuild within "
          f"{resid * 1e6:+.1f} ppm of the published total; the threshold alone "
          f"moves the exponent by {span:.5f}")
    return f


if __name__ == "__main__":
    f = _selftest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print()
    print(f[["configuration", "narrow_threshold_km2", "beta",
             "beta_minus_headline", "converted_km2"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    print(f"\nwritten to {OUT}")
