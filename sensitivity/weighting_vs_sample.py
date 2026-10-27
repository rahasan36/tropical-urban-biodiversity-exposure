"""A third of the world's threatened land vertebrates exposed to tropical urbanization

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
import tropics_numbers as tn
import tropics_refit as trf
from tropics_config import TABLE_DIR, EXCLUDED_TAXA
from tropics_cost import PAIRS_PATH

OUT = TABLE_DIR / "weighting_vs_sample.csv"
THREAT = ("CR", "EN", "VU")


def responses():
    d = tn.analysis_sample()
    z = np.load(PAIRS_PATH, allow_pickle=False)
    cat, tax, _sci, ar = tfl.flags(PAIRS_PATH)
    thr = float(json.loads((TABLE_DIR / "cities_cost_meta.json").read_text())
                ["narrow_threshold_km2"])
    keep = ~np.isin(tax, [t.upper() for t in EXCLUDED_TAXA])
    narrow = (ar > 0) & (ar <= thr) & keep
    threat = np.isin(cat, list(THREAT)) & keep

    ps, pc, pv = z["pair_sp"], z["pair_city"], z["pair_conv"]
    w = np.where(ar[ps] > 0, pv / np.maximum(ar[ps], 1e-6), 0.0)

    def agg(mask, val, name):
        s = (pd.DataFrame(dict(city_id=pc[mask], v=val[mask]))
             .groupby("city_id").v.sum().rename(name))
        return s

    concern = threat | narrow

    x = d.copy()
    for name, mask, val in (
            ("unw_pool", np.ones(ps.size, bool), pv),
            ("rar_pool", np.ones(ps.size, bool), w),
            ("unw_all", concern[ps], pv),
            ("rar_all", concern[ps], w),
            ("unw_threat", threat[ps], pv),
            ("unw_narrow", narrow[ps], pv),
            ("rar_threat", threat[ps], w),
            ("rar_narrow", narrow[ps], w)):
        x = x.join(agg(mask, val, name), on="city_id")
        x[name] = x[name].fillna(0.0) / x.urban_added_km2
    return x


def _fit(y, col):
    r = trf.fit_scaling(y, response=col, size="urban_km2_base",
                        setting="setting_footprint", spatial=True)
    return dict(n=int(r["n"]), beta=float(r["beta"]),
                lo=float(r["ci_lower"]), hi=float(r["ci_upper"]),
                p=float(r["p_value"]), R2=float(r["R2"]))


def run():
    x = responses()
    pairs = [("Whole analysis pool", "unw_pool", "rar_pool"),
             ("Species of concern", "unw_all", "rar_all"),
             ("Threatened", "unw_threat", "rar_threat"),
             ("Narrow range", "unw_narrow", "rar_narrow")]
    rows = []
    for label, cu, cr in pairs:
        both = x[(x[cu] > 0) & (x[cr] > 0)]
        for weighting, col in (("unweighted", cu), ("rarity weighted", cr)):
            rows.append(dict(family=label, weighting=weighting,
                             sample="own positive sample",
                             **_fit(x[x[col] > 0], col)))
            rows.append(dict(family=label, weighting=weighting,
                             sample="common positive sample",
                             **_fit(both, col)))
    return pd.DataFrame(rows)


def _selftest():
    rng = np.random.default_rng(11)
    n = 600
    size = 10.0 ** rng.uniform(0.0, 3.0, n)
    setting = 10.0 ** rng.uniform(1.0, 3.0, n)
    added = 10.0 ** rng.uniform(-1.0, 1.5, n)
    y = size ** 0.3 * setting ** 0.5 * 10.0 ** rng.normal(0, 0.15, n)
    d = pd.DataFrame(dict(city_id=np.arange(n), urban_km2_base=size,
                          setting_footprint=setting, urban_added_km2=added,
                          a=y, b=y * 7.0,
                          lon=rng.uniform(-60, 120, n),
                          lat=rng.uniform(-23, 23, n),
                          country=rng.integers(0, 20, n).astype(str),
                          biome=rng.integers(0, 5, n).astype(str)))
    ra, rb = _fit(d, "a"), _fit(d, "b")
    assert abs(ra["beta"] - rb["beta"]) < 1e-6, (ra["beta"], rb["beta"])
    assert abs(ra["beta"] - 0.3) < 0.05, ra["beta"]
    sub = d[d.a > np.median(d.a)]
    rs = _fit(sub, "a")
    assert rs["n"] == len(sub)
    print("ok  proportional rescaling leaves the exponent fixed; "
          f"recovered beta {ra['beta']:.3f} against 0.300")


if __name__ == "__main__":
    _selftest()
    d = run()
    d.to_csv(OUT, index=False)
    print(d.to_string(index=False))
