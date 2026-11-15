"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np, pandas as pd

from tropics_config import TABLE_DIR
from tropics_refit import table
from tropics_scaling import fit_scaling

FLOOR_KM2 = 5.0

SPECS = [
    ("2000 impervious extent (primary)", "urban_km2_base", None, False),
    ("2020 impervious extent", "urban_km2_final", None, False),
    ("urban land added, 2000-2020", "urban_added_km2", None, False),
    ("2020 impervious extent, common sample", "urban_km2_final", None, True),
    ("urban land added, common sample", "urban_added_km2", None, True),
    ("2000 extent, emergent agglomerations excluded", "urban_km2_base", FLOOR_KM2, False),
    ("2000 extent, at least 10 km2 in 2000", "urban_km2_base", 10.0, False),
    ("2000 extent, at least 25 km2 in 2000", "urban_km2_base", 25.0, False),
    ("2000 extent, at least 50 km2 in 2000", "urban_km2_base", 50.0, False),
]


def _one(d, label, size, floor, common, response, spatial):
    x = d if floor is None else d[d.urban_km2_base >= floor]
    if common:
        x = x[x.urban_km2_base > 0]
    r = fit_scaling(x, response=response, size=size,
                    setting="setting_footprint", spatial=spatial)
    return dict(label=label, size_variable=size,
                min_2000_extent_km2=(np.nan if floor is None else floor),
                common_sample=bool(common),
                response=response, n=int(r["n"]), beta=float(r["beta"]),
                se=float(r["se"]) if "se" in r else np.nan,
                ci_lower=float(r["ci_lower"]), ci_upper=float(r["ci_upper"]),
                p_value=float(r["p_value"]),
                gamma_setting=float(r.get("gamma_setting", np.nan)),
                R2=float(r["R2"]),
                excludes_zero=bool(r["ci_lower"] * r["ci_upper"] > 0))


def run(d=None, response="cost_per_km2", spatial=True, out="size_variables.csv"):
    d = table() if d is None else d
    rows = [_one(d, *s, response=response, spatial=spatial) for s in SPECS]
    f = pd.DataFrame(rows)
    if out:
        f.to_csv(TABLE_DIR / out, index=False)
    return f


def _selftest(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    size = 10 ** rng.uniform(0.0, 2.5, n)
    setting = np.clip(rng.poisson(30, n), 1, None)
    country = rng.integers(0, 25, n).astype(str)
    biome = rng.integers(0, 4, n).astype(str)
    for true_b in (-0.30, 0.0, +0.45):
        y = 10 ** (1.0 + true_b * np.log10(size)
                   + 0.20 * np.log10(setting)
                   + rng.normal(0, 0.05, n))
        d = pd.DataFrame(dict(cost_per_km2=y, urban_km2_base=size,
                              setting_footprint=setting, country=country,
                              biome=biome,
                              lon=rng.uniform(-80, 120, n),
                              lat=rng.uniform(-20, 20, n)))
        r = fit_scaling(d, response="cost_per_km2", size="urban_km2_base",
                        setting="setting_footprint", spatial=False)
        assert abs(r["beta"] - true_b) < 0.01, (true_b, r["beta"])
        assert r["ci_lower"] <= true_b <= r["ci_upper"], (true_b, r)
    print("size exponent recovered on synthetic data   ok")

    d = pd.DataFrame(dict(cost_per_km2=10 ** rng.normal(1, .3, n),
                          urban_km2_base=size, urban_km2_final=size * 1.4,
                          urban_added_km2=size * 0.4,
                          setting_footprint=setting, country=country,
                          biome=biome, lon=rng.uniform(-80, 120, n),
                          lat=rng.uniform(-20, 20, n)))
    a = _one(d, "x", "urban_km2_base", None, False, "cost_per_km2", False)
    b = _one(d, "x", "urban_km2_base", 0.0, False, "cost_per_km2", False)
    assert a["n"] == b["n"] == n and abs(a["beta"] - b["beta"]) < 1e-12
    c = _one(d, "x", "urban_km2_base", 50.0, False, "cost_per_km2", False)
    assert c["n"] == int((d.urban_km2_base >= 50).sum()) < n
    print("size floor subsets and does not reweight    ok")

    a2 = _one(d, "x", "urban_km2_final", None, True, "cost_per_km2", False)
    assert a2["n"] == n and abs(a2["beta"] - _one(
        d, "x", "urban_km2_final", None, False, "cost_per_km2",
        False)["beta"]) < 1e-12
    e = d.copy()
    e.loc[e.index[:37], "urban_km2_base"] = 0.0
    f_ = _one(e, "x", "urban_km2_final", None, True, "cost_per_km2", False)
    g = _one(e, "x", "urban_km2_final", None, False, "cost_per_km2", False)
    assert g["n"] == n and f_["n"] == n - 37, (g["n"], f_["n"])
    print("common sample drops only the zero-extent rows ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        f = run()
        with pd.option_context("display.width", 200):
            print(f[["label", "n", "beta", "ci_lower", "ci_upper", "p_value",
                     "gamma_setting", "R2"]].to_string(index=False))
