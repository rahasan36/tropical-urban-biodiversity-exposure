"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cf_fix as C
import cf_release as R
from tropics_config import TABLE_DIR

OUT = TABLE_DIR / "cf_influence.csv"
MIN_CITIES = 20
A = C.ALPHAS


def _point(d, alpha_hat):
    m, p, _s, _n = R.curves(d, alpha_hat)
    return m, p


def run(budget_s=1800):
    import tropics_counterfactual as cf
    d = R.sample()
    ah = float(cf.implied_alpha(d))
    m0, p0 = _point(d, ah)
    i_lo, i_hi = 0, len(A) - 1

    rows = [dict(dropped="none (all countries)", n_cities=len(d),
                 n_dropped=0,
                 measured_dispersed=m0[i_lo], measured_concentrated=m0[i_hi],
                 predicted_dispersed=p0[i_lo], predicted_concentrated=p0[i_hi],
                 paired_concentrated=p0[i_hi] - m0[i_hi],
                 paired_dispersed=p0[i_lo] - m0[i_lo])]

    counts = d.country.value_counts()
    big = [k for k in counts.index if counts[k] >= MIN_CITIES]
    small = [k for k in counts.index if counts[k] < MIN_CITIES]
    t0 = time.time()
    for k in big:
        if time.time() - t0 > budget_s:
            print("budget reached", flush=True)
            break
        sub = d[d.country != k]
        m, p = _point(sub, ah)
        rows.append(dict(dropped=str(k), n_cities=len(sub),
                         n_dropped=int(counts[k]),
                         measured_dispersed=m[i_lo], measured_concentrated=m[i_hi],
                         predicted_dispersed=p[i_lo], predicted_concentrated=p[i_hi],
                         paired_concentrated=p[i_hi] - m[i_hi],
                         paired_dispersed=p[i_lo] - m[i_lo]))
    if small:
        sub = d[~d.country.isin(small)]
        m, p = _point(sub, ah)
        rows.append(dict(dropped=f"all countries with under {MIN_CITIES} cities",
                         n_cities=len(sub),
                         n_dropped=int(len(d) - len(sub)),
                         measured_dispersed=m[i_lo], measured_concentrated=m[i_hi],
                         predicted_dispersed=p[i_lo], predicted_concentrated=p[i_hi],
                         paired_concentrated=p[i_hi] - m[i_hi],
                         paired_dispersed=p[i_lo] - m[i_lo]))
    return pd.DataFrame(rows), ah


def _selftest():
    rng = np.random.default_rng(5)
    n = 400
    d = pd.DataFrame(dict(
        city_id=np.arange(n),
        urban_km2_base=10.0 ** rng.uniform(0, 3, n),
        urban_added_km2=10.0 ** rng.uniform(-1, 1.5, n),
        cost_per_km2=10.0 ** rng.uniform(0, 2, n),
        setting_footprint=10.0 ** rng.uniform(1, 3, n),
        biome=rng.integers(0, 4, n).astype(str),
        country=np.where(np.arange(n) < 300, "BIG", "small"),
        lon=rng.uniform(-60, 120, n), lat=rng.uniform(-23, 23, n)))
    t_all = C.totals(d.assign(_p=d.cost_per_km2), "_p", alphas=A)
    t_wo = C.totals(d[d.country != "BIG"].assign(_p=d[d.country != "BIG"].cost_per_km2),
                    "_p", alphas=A)
    assert np.any(np.abs(np.asarray(t_all) - np.asarray(t_wo)) > 0)
    print("ok  removing a country moves the totals the experiment is built on")


if __name__ == "__main__":
    _selftest()
    t, ah = run(budget_s=float(sys.argv[1]) if len(sys.argv) > 1 else 1800)
    t.to_csv(OUT, index=False)
    base = t.iloc[0]
    d = t.iloc[1:]
    pd.set_option("display.width", 220)
    print(f"\nalpha_hat {ah:.4f}   baseline paired difference "
          f"concentrated {base.paired_concentrated:+.2f}  "
          f"dispersed {base.paired_dispersed:+.2f}")
    print(f"leave-one-out range, concentrated: "
          f"{d.paired_concentrated.min():+.2f} to {d.paired_concentrated.max():+.2f}")
    print(f"leave-one-out range, dispersed:    "
          f"{d.paired_dispersed.min():+.2f} to {d.paired_dispersed.max():+.2f}")
    worst = d.reindex(d.paired_concentrated.sub(base.paired_concentrated)
                      .abs().sort_values(ascending=False).index).head(6)
    print("\nmost influential exclusions (concentrated endpoint):")
    print(worst[["dropped", "n_dropped", "paired_concentrated",
                 "paired_dispersed"]].to_string(index=False))
