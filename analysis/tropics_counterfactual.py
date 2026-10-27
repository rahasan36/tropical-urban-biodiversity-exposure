"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np, pandas as pd

SIZE_FLOOR_KM2 = 0.09
ALPHAS = np.round(np.arange(-1.0, 2.01, 0.25), 2)


def allocate(size, total, alpha, floor=SIZE_FLOOR_KM2):
    s = np.maximum(np.asarray(size, float), floor)
    w = s ** float(alpha)
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.full(s.shape, total / s.size)
    return total * w / tot


def curve(df, size="urban_km2_base", land="urban_added_km2", cost="m_hat",
          group="country", alphas=ALPHAS):
    d = df.dropna(subset=[size, land, cost, group])
    out = []
    for a in alphas:
        tot = 0.0
        for _, g in d.groupby(group, sort=False):
            L = allocate(g[size].to_numpy(), float(g[land].sum()), a)
            tot += float((g[cost].to_numpy() * L).sum())
        out.append(tot)
    obs = float((d[cost] * d[land]).sum())
    return pd.DataFrame(dict(alpha=alphas, cost=out, ratio=np.array(out) / obs)), obs


def implied_alpha(df, size="urban_km2_base", land="urban_added_km2",
                  group="country"):
    d = df[(df[size] > 0) & (df[land] > 0)]
    xs, ys = [], []
    for _, g in d.groupby(group, sort=False):
        if len(g) < 2:
            continue
        x = np.log(np.maximum(g[size].to_numpy(), SIZE_FLOOR_KM2))
        y = np.log(g[land].to_numpy() / g[land].sum())
        xs.append(x - x.mean()); ys.append(y - y.mean())
    x = np.concatenate(xs); y = np.concatenate(ys)
    return float((x * y).sum() / (x * x).sum())


def m_hat_model(d, setting="setting_footprint", response="cost_per_km2"):
    import statsmodels.formula.api as smf
    x = d[(d[response] > 0)].copy()
    x["y"] = np.log10(x[response])
    x["logSet"] = np.log10(x[setting].clip(lower=0.5))
    x["_biome"] = x["biome"].astype(str)
    f = smf.mixedlm("y ~ logSet + C(_biome)", x,
                    groups=x["country"].astype(str)).fit(reml=True, method="lbfgs")
    pred = np.asarray(f.model.exog, float) @ np.asarray(f.fe_params, float)
    x["m_hat_model"] = 10.0 ** pred
    return d.merge(x[["city_id", "m_hat_model"]], on="city_id", how="left")


def _selftest():
    d = pd.DataFrame(dict(country=["A", "A"], urban_km2_base=[1.0, 9.0],
                          urban_added_km2=[2.0, 8.0], m_hat=[10.0, 1.0]))
    c, obs = curve(d, alphas=np.array([0.0, 1.0]))
    assert abs(obs - (10 * 2 + 1 * 8)) < 1e-9, obs
    assert abs(c.cost[0] - 55.0) < 1e-9, c.cost[0]
    assert abs(c.cost[1] - 19.0) < 1e-9, c.cost[1]
    d1 = pd.DataFrame(dict(country=["B"], urban_km2_base=[4.0],
                           urban_added_km2=[3.0], m_hat=[2.0]))
    c1, obs1 = curve(d1, alphas=np.array([-1.0, 0.0, 2.0]))
    assert np.allclose(c1.cost.to_numpy(), obs1), c1
    d2 = pd.DataFrame(dict(country=["C"] * 4, urban_km2_base=[1.0, 2.0, 4.0, 8.0],
                           urban_added_km2=[1.0, 2.0, 4.0, 8.0]))
    assert abs(implied_alpha(d2) - 1.0) < 1e-9, implied_alpha(d2)
    d3 = d2.copy(); d3["urban_added_km2"] = [1.0, 1.0, 1.0, 1.0]
    assert abs(implied_alpha(d3) - 0.0) < 1e-9, implied_alpha(d3)
    print("selftest ok: allocation, single city country, implied alpha")


if __name__ == "__main__":
    _selftest()
