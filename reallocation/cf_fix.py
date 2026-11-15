"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np, pandas as pd

SIZE_FLOOR_KM2 = 0.09
ALPHAS = np.round(np.arange(-1.0, 2.01, 0.25), 2)


def predict_price(d, setting="setting_footprint", response="cost_per_km2",
                  use_ranef=True, smearing=True):
    import statsmodels.formula.api as smf
    x = d[d[response] > 0].copy()
    x["y"] = np.log10(x[response])
    x["logSet"] = np.log10(x[setting].clip(lower=0.5))
    x["_biome"] = x["biome"].astype(str)
    x["_country"] = x["country"].astype(str)
    f = smf.mixedlm("y ~ logSet + C(_biome)", x, groups=x["_country"]).fit(
        reml=True, method="lbfgs")
    fe = np.asarray(f.model.exog, float) @ np.asarray(f.fe_params, float)
    pred = fe.copy()
    if use_ranef:
        re = {k: float(np.asarray(v).ravel()[0]) for k, v in f.random_effects.items()}
        pred = pred + x["_country"].map(re).fillna(0.0).to_numpy()
    resid = x["y"].to_numpy() - pred
    s = float(np.mean(10.0 ** resid)) if smearing else 1.0
    x["m_hat_model"] = (10.0 ** pred) * s
    return d.merge(x[["city_id", "m_hat_model"]], on="city_id", how="left"), s


def allocate(size, total, alpha, floor=SIZE_FLOOR_KM2):
    s = np.maximum(np.asarray(size, float), floor)
    w = s ** float(alpha)
    t = w.sum()
    if not np.isfinite(t) or t <= 0:
        return np.full(s.shape, total / s.size)
    return total * w / t


def totals(df, price, size="urban_km2_base", land="urban_added_km2",
           group="country", alphas=ALPHAS):
    d = df.dropna(subset=[size, land, price, group])
    sz, ld, pr = d[size].to_numpy(), d[land].to_numpy(), d[price].to_numpy()
    keys, inv = np.unique(d[group].to_numpy(), return_inverse=True)
    out = np.empty(len(alphas))
    for a_i, a in enumerate(alphas):
        tot = 0.0
        for g in range(len(keys)):
            m = inv == g
            tot += float((pr[m] * allocate(sz[m], float(ld[m].sum()), a)).sum())
        out[a_i] = tot
    return out


def rel_to_fitted(curve, alphas, alpha_hat):
    ref = float(np.interp(alpha_hat, alphas, curve))
    return (curve / ref - 1.0) * 100.0


def _selftest():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 0.4, 200_000)
    s = np.mean(10.0 ** e)
    assert abs(s - np.mean(10.0 ** e)) < 1e-12
    assert s > 1.0, s
    L = allocate(np.array([1.0, 9.0]), 10.0, 0.0)
    assert np.allclose(L, [5.0, 5.0]), L
    L = allocate(np.array([1.0, 9.0]), 10.0, 1.0)
    assert np.allclose(L, [1.0, 9.0]), L
    d = pd.DataFrame(dict(city_id=[1, 2], urban_km2_base=[1.0, 9.0],
                          urban_added_km2=[5.0, 5.0], country=["A", "A"],
                          p=[2.0, 1.0]))
    t = totals(d, "p", alphas=np.array([1.0]))
    assert np.isclose(t[0], 11.0), t
    d2 = pd.concat([d, d.assign(city_id=[3, 4], country=["B", "B"])])
    t2 = totals(d2, "p", alphas=np.array([1.0]))
    assert np.isclose(t2[0], 22.0), t2
    print("selftest ok: smearing factor %.4f on sigma=0.4" % s)


if __name__ == "__main__":
    _selftest()
