"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np, pandas as pd

import tropics_counterfactual as cf
from tropics_counterfactual import allocate, SIZE_FLOOR_KM2, ALPHAS


def allocate_capped(size, total, alpha, cap, floor=SIZE_FLOOR_KM2, iters=64):
    s = np.maximum(np.asarray(size, float), floor)
    c = np.asarray(cap, float)
    if c.sum() <= total:
        return c.copy()
    w = s ** float(alpha)
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        w = np.ones_like(s)
    out = np.zeros_like(s)
    free = np.ones(s.shape, bool)
    left = float(total)
    for _ in range(iters):
        wf = w[free]
        if left <= 1e-12 or wf.sum() <= 0:
            break
        add = left * wf / wf.sum()
        prop = out[free] + add
        over = prop > c[free]
        if not over.any():
            out[free] = prop
            left = 0.0
            break
        idx = np.flatnonzero(free)
        out[idx[over]] = c[idx[over]]
        left -= float((c[idx[over]] - (prop[over] - add[over])).sum())
        free[idx[over]] = False
    return out


def curve_generic(d, alphas=ALPHAS, size="urban_km2_base",
                  land="urban_added_km2", cost="m_hat", group="country",
                  cap_mult=None):
    d = d.dropna(subset=[size, land, cost, group])
    caps = None if cap_mult is None else np.maximum(
        cap_mult * d[land].to_numpy(), d[land].to_numpy())
    d = d.assign(_cap=caps) if caps is not None else d
    out = []
    for a in alphas:
        tot = 0.0
        for _, g in d.groupby(group, sort=False):
            if cap_mult is None:
                L = allocate(g[size].to_numpy(), float(g[land].sum()), a)
            else:
                L = allocate_capped(g[size].to_numpy(), float(g[land].sum()), a,
                                    g["_cap"].to_numpy())
            tot += float((g[cost].to_numpy() * L).sum())
        out.append(tot)
    return np.asarray(out)


def rebase(alphas, totals, alpha_hat):
    ref = float(np.interp(alpha_hat, alphas, totals))
    return 100.0 * (np.asarray(totals) / ref - 1.0), ref


def bootstrap(d, n_boot=400, seed=0, alphas=ALPHAS, alpha_hat=None, **kw):
    grp = kw.get("group", "country")
    alpha_hat = cf.implied_alpha(d) if alpha_hat is None else alpha_hat
    keys = d[grp].dropna().unique()
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, len(alphas)))
    for b in range(n_boot):
        take = rng.choice(keys, size=len(keys), replace=True)
        parts = [d[d[grp] == k].assign(**{grp: f"{k}__{i}"})
                 for i, k in enumerate(take)]
        s = pd.concat(parts, ignore_index=True)
        reps[b] = rebase(alphas, curve_generic(s, alphas, **kw), alpha_hat)[0]
    return (np.percentile(reps, 2.5, axis=0), np.percentile(reps, 97.5, axis=0))


def bootstrap_modeled(d, n_boot=300, seed=1, alphas=ALPHAS, alpha_hat=None):
    alpha_hat = cf.implied_alpha(d) if alpha_hat is None else alpha_hat
    keys = d["country"].dropna().unique()
    rng = np.random.default_rng(seed)
    reps, kept = np.empty((n_boot, len(alphas))), 0
    for b in range(n_boot):
        take = rng.choice(keys, size=len(keys), replace=True)
        parts = [d[d.country == k].assign(country=f"{k}__{i}")
                 for i, k in enumerate(take)]
        s = pd.concat(parts, ignore_index=True)
        s["city_id"] = np.arange(len(s), dtype=np.int64)
        try:
            z = cf.m_hat_model(s)
        except Exception:
            continue
        z = z[z.m_hat_model.notna()].copy()
        if z.empty:
            continue
        z["m_hat"] = z.m_hat_model
        reps[kept] = rebase(alphas, curve_generic(z, alphas), alpha_hat)[0]
        kept += 1
    reps = reps[:kept]
    return (np.percentile(reps, 2.5, axis=0), np.percentile(reps, 97.5, axis=0),
            int(kept))


def _selftest():
    s = np.array([1.0, 4.0, 9.0, 16.0])
    for a in (-1.0, 0.0, 0.7, 2.0):
        u = allocate(s, 100.0, a)
        c = allocate_capped(s, 100.0, a, np.full(4, 1e6))
        assert np.allclose(u, c), (a, u, c)
    print("uncapped allocation reproduced          ok")

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(2, 12)
        s = 10 ** rng.uniform(-1, 2, n)
        cap = np.maximum(rng.uniform(0.5, 30.0, n), 0.5)
        tot = float(min(cap.sum() * rng.uniform(0.2, 0.95), 1e6))
        L = allocate_capped(s, tot, rng.uniform(-1, 2), cap)
        assert abs(L.sum() - tot) < 1e-6 * max(tot, 1.0), (L.sum(), tot)
        assert (L <= cap + 1e-9).all()
    print("capped allocation conserves the total   ok")

    L = allocate_capped(np.array([100.0, 1.0]), 50.0, 2.0,
                        np.array([10.0, 1e6]))
    assert abs(L[0] - 10.0) < 1e-9 and abs(L[1] - 40.0) < 1e-9, L
    print("capped land is redistributed            ok")

    al = np.array([0.0, 0.5, 1.0]); tot = np.array([10.0, 12.0, 16.0])
    pct, ref = rebase(al, tot, 0.5)
    assert abs(pct[1]) < 1e-12 and abs(ref - 12.0) < 1e-12
    pct, ref = rebase(al, tot, 0.25)
    assert abs(ref - 11.0) < 1e-12 and abs(pct[0] - (100 * (10 / 11 - 1))) < 1e-9
    print("re-basing puts zero at the fitted alpha ok")


if __name__ == "__main__":
    _selftest()
