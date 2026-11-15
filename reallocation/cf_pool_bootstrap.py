"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import json
import sys
import numpy as np
import pandas as pd
import tropics_counterfactual as cf
import tropics_cfrobust as cr
import tropics_models as p4
from tropics_config import TABLE_DIR

OUT = TABLE_DIR / "cf_pool_bootstrap.json"
A = cr.ALPHAS
POOLS = {"country": "country", "realm": "realm", "tropics": "_all"}


def arrays(d):
    return dict(size=d.urban_km2_base.to_numpy(float), land=d.urban_added_km2.to_numpy(float),
                cost=d.m_hat.to_numpy(float))


def curve(v, groups, alpha_hat):
    tot = np.zeros(len(A))
    for g in np.unique(groups):
        m = groups == g
        s, L, c = v["size"][m], float(v["land"][m].sum()), v["cost"][m]
        tot += np.array([float((c * cf.allocate(s, L, a)).sum()) for a in A])
    return cr.rebase(A, tot, alpha_hat)[0]


def replicate(d, v, members, keys, take, alpha_hat):
    idx = np.concatenate([members[k] for k in take])
    w = {k: x[idx] for k, x in v.items()}
    tag = np.concatenate([np.full(len(members[k]), i) for i, k in enumerate(take)])
    grp = {"country": tag, "realm": d.realm.to_numpy()[idx], "tropics": np.zeros(len(idx), int)}
    return {p: curve(w, grp[p], alpha_hat) for p in POOLS}


def run(d, n_boot=1000, seed=7, alpha_hat=None):
    d = d.dropna(subset=["urban_km2_base", "urban_added_km2", "m_hat", "country", "realm"]).reset_index(drop=True)
    alpha_hat = float(cf.implied_alpha(d)) if alpha_hat is None else alpha_hat
    v = arrays(d)
    keys = d.country.unique()
    members = {k: np.flatnonzero(d.country.to_numpy() == k) for k in keys}
    point = replicate(d, v, members, keys, keys, alpha_hat)
    rng = np.random.default_rng(seed)
    reps = {p: np.empty((n_boot, len(A))) for p in POOLS}
    for b in range(n_boot):
        r = replicate(d, v, members, keys, rng.choice(keys, size=len(keys), replace=True), alpha_hat)
        for p in POOLS:
            reps[p][b] = r[p]
    out = dict(alpha_hat=alpha_hat, alphas=[float(a) for a in A], n=len(d), n_boot=n_boot, seed=seed)
    for p in POOLS:
        lo, hi = np.percentile(reps[p], [2.5, 97.5], axis=0)
        out[p] = dict(pct=point[p].tolist(), ci_lo=lo.tolist(), ci_hi=hi.tolist())
    for p in ("realm", "tropics"):
        diff = reps[p] - reps["country"]
        lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
        out[p]["minus_country"] = dict(pct=(point[p] - point["country"]).tolist(), ci_lo=lo.tolist(),
                                       ci_hi=hi.tolist())
    return out


def selftest():
    rng = np.random.default_rng(0)
    n = 600
    t = pd.DataFrame(dict(country=rng.integers(0, 30, n).astype(str), realm=rng.choice(["x", "y", "z"], n),
                          urban_km2_base=10 ** rng.uniform(-1, 3, n), urban_added_km2=10 ** rng.uniform(-1, 2, n)))
    t["m_hat"] = 12.0
    o = run(t, n_boot=40, alpha_hat=0.7)
    for p in POOLS:
        assert np.allclose(o[p]["pct"], 0, atol=1e-9) and np.allclose(o[p]["ci_lo"], 0, atol=1e-9) \
            and np.allclose(o[p]["ci_hi"], 0, atol=1e-9), p
    print("constant intensity gives no change under any pool     ok")
    t["m_hat"] = 10 ** rng.uniform(0.5, 1.5, n)
    o = run(t, n_boot=5, alpha_hat=0.7)
    for p, g in POOLS.items():
        z = t.assign(_all="tropics")
        ref = cr.rebase(A, cr.curve_generic(z, A, group=g), 0.7)[0]
        assert np.allclose(o[p]["pct"], ref, atol=1e-8), p
    print("point curves match the existing reallocation routine  ok")
    t["m_hat"] = t.urban_km2_base ** 0.3
    o = run(t, n_boot=200, alpha_hat=0.7)
    assert o["tropics"]["pct"][-1] > 0 and o["tropics"]["ci_lo"][-1] > 0
    assert o["tropics"]["pct"][0] < 0 and o["tropics"]["ci_hi"][0] < 0
    print("intensity rising with size is detected with its sign  ok")


if __name__ == "__main__":
    selftest()
    if len(sys.argv) > 1:
        d = p4.table()
        d["m_hat"] = d.cost_per_km2
        o = run(d, n_boot=int(sys.argv[1]))
        OUT.write_text(json.dumps(o, indent=1))
        for p in POOLS:
            r = o[p]
            print(f"{p:8s} dispersed {r['pct'][0]:+.1f} [{r['ci_lo'][0]:+.1f}, {r['ci_hi'][0]:+.1f}]   "
                  f"concentrated {r['pct'][-1]:+.1f} [{r['ci_lo'][-1]:+.1f}, {r['ci_hi'][-1]:+.1f}]")
