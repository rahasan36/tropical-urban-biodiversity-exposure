"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np, pandas as pd, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import statsmodels.formula.api as smf


def blocks(lon, lat, deg=10.0):
    return (np.floor(lat / deg).astype(int) * 1000
            + np.floor(lon / deg).astype(int))


def block_cv(d, specs, deg=10.0, k=10, seed=3):
    d = d.copy()
    d["_blk"] = blocks(d.lon.to_numpy(), d.lat.to_numpy(), deg)
    u = np.unique(d._blk)
    rng = np.random.default_rng(seed)
    fold = dict(zip(u, rng.integers(0, k, len(u))))
    d["_fold"] = d._blk.map(fold)
    out = []
    for name, formula in specs.items():
        num = den = 0.0
        scored = 0
        for f in range(k):
            tr, te = d[d._fold != f], d[d._fold == f]
            if len(te) < 20 or len(tr) < 200:
                continue
            for c in ("biome", "country"):
                if f"C({c})" in formula:
                    te = te[te[c].isin(set(tr[c]))]
            if len(te) < 20:
                continue
            m = smf.ols(formula, tr).fit()
            p = m.predict(te)
            ok = p.notna() & te.log_cost.notna()
            num += float(((te.log_cost[ok] - p[ok]) ** 2).sum())
            den += float(((te.log_cost[ok] - tr.log_cost.mean()) ** 2).sum())
            scored += int(ok.sum())
        out.append(dict(model=name, n_blocks=len(u), n_scored=scored,
                        R2_oos=1 - num / den))
    return pd.DataFrame(out)


def _selftest():
    rng = np.random.default_rng(0)
    n = 3000
    d = pd.DataFrame(dict(lon=rng.uniform(-180, 180, n), lat=rng.uniform(-25, 25, n)))
    d["log_size"] = rng.normal(0, 1, n)
    d["log_setting"] = rng.normal(0, 1, n)
    d["log_cost"] = 0.8 * d.log_setting + rng.normal(0, 0.4, n)
    r = block_cv(d, {"size": "log_cost ~ log_size",
                     "setting": "log_cost ~ log_setting"}).set_index("model")
    assert r.loc["setting", "R2_oos"] > 0.6, r
    assert abs(r.loc["size", "R2_oos"]) < 0.05, r
    print("selftest ok:  setting %.3f   size %.3f"
          % (r.loc["setting", "R2_oos"], r.loc["size", "R2_oos"]))


if __name__ == "__main__":
    _selftest()
