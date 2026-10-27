"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False


def project_equal_area(lon, lat):
    R = 6371.0
    phi0 = np.radians(23.4394)
    x = R * np.radians(np.asarray(lon, float)) * np.cos(phi0)
    y = R * np.sin(np.radians(np.asarray(lat, float))) / np.cos(phi0)
    return np.column_stack([x, y])


def morans_i(coords, values, k=8, n_perm=999, seed=0):
    x = np.asarray(values, float)
    x = x - x.mean()
    n = x.size
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    idx = idx[:, 1:]
    denom = float((x ** 2).sum())

    def stat(v):
        return float((v * v[idx].mean(axis=1)).sum() / denom)

    I = stat(x)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for b in range(n_perm):
        perm[b] = stat(rng.permutation(x))
    z = (I - perm.mean()) / (perm.std() + 1e-12)
    p = (1 + int((np.abs(perm) >= abs(I)).sum())) / (n_perm + 1)
    return float(I), float(z), float(p)


def spatial_lag(coords, values, k=8):
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    return np.asarray(values, float)[idx[:, 1:]].mean(axis=1)


def spatial_block(lon, lat, deg=2.0):
    bx = np.floor(np.asarray(lon, float) / deg).astype(int)
    by = np.floor(np.asarray(lat, float) / deg).astype(int)
    return np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))


def fit_scaling(df, response="cost_per_km2", size="urban_km2_final",
                setting="setting_richness", country="country", biome="biome",
                spatial=False, block_deg=2.0, seed=0):
    d = df.copy()
    d = d[(d[response] > 0) & (d[size] > 0)].dropna(subset=[response, size])
    d["y"] = np.log10(d[response])
    d["logS"] = np.log10(d[size])
    rhs = ["logS"]
    if setting in d.columns and (d[setting] > 0).any():
        d["logSet"] = np.log10(d[setting].clip(lower=0.5))
        rhs.append("logSet")
    if biome in d.columns and d[biome].nunique() > 1:
        d["_biome"] = d[biome].astype(str)
        rhs.append("C(_biome)")
    formula = "y ~ " + " + ".join(rhs)
    vc = None
    if spatial:
        d["_block"] = spatial_block(d.lon.to_numpy(), d.lat.to_numpy(), block_deg)
        vc = {"block": "0 + C(_block)"}

    out = dict(n=int(len(d)), formula=formula)
    fitted = None
    if HAVE_SM and country in d.columns and d[country].nunique() > 1:
        try:
            md = smf.mixedlm(formula, d, groups=d[country].astype(str),
                             vc_formula=vc, re_formula="1") if vc else \
                 smf.mixedlm(formula, d, groups=d[country].astype(str))
            fitted = md.fit(reml=True, method="lbfgs")
            out["model"] = "MixedLM (country random intercept)"
        except Exception as exc:
            out["mixed_error"] = f"{type(exc).__name__}: {exc}"
            fitted = None
    if fitted is None:
        fitted = smf.ols(formula, d).fit(cov_type="cluster",
                                         cov_kwds={"groups": d[country].astype(str)})
        out["model"] = "OLS (country-clustered SE)"

    def residuals(f):
        try:
            return np.asarray(f.resid, float), "conditional"
        except Exception:
            return (np.asarray(f.model.endog, float)
                    - np.asarray(f.model.exog, float) @ np.asarray(f.fe_params, float),
                    "marginal")

    beta = float(fitted.params["logS"]); se = float(fitted.bse["logS"])
    out.update(beta=beta, se=se, ci_lower=beta - 1.96 * se, ci_upper=beta + 1.96 * se,
               p_value=float(fitted.pvalues["logS"]))
    if "logSet" in fitted.params.index:
        out["gamma_setting"] = float(fitted.params["logSet"])
        out["gamma_se"] = float(fitted.bse["logSet"])
    resid, resid_kind = residuals(fitted)
    out["resid_kind"] = resid_kind
    ss_tot = float(((d["y"] - d["y"].mean()) ** 2).sum())
    out["R2"] = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot else np.nan
    out["beta_std"] = beta * float(d["logS"].std()) / float(d["y"].std())
    out["vs_one"] = ("sublinear" if out["ci_upper"] < 1 else
                     "superlinear" if out["ci_lower"] > 1 else
                     "not distinguishable from proportional")
    out["vs_zero"] = ("no detectable size dependence"
                      if out["ci_lower"] <= 0 <= out["ci_upper"] else
                      "size dependence detected")
    out["verdict"] = ("scale invariant" if out["vs_zero"].startswith("no")
                      else out["vs_one"])
    out["_resid"] = resid
    out["_d"] = d
    return out


def report(df, response="cost_per_km2", **kw):
    r = fit_scaling(df, response=response, **kw)
    print(f"[{response}] {r['model']}")
    print(f"   beta = {r['beta']:.4f}  95% CI [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
          f"  standardized {r['beta_std']:.3f}  R2 = {r['R2']:.3f}  n = {r['n']:,}")
    if "gamma_setting" in r:
        print(f"   setting richness gamma = {r['gamma_setting']:.4f} "
              f"(SE {r['gamma_se']:.4f})")
    print(f"   -> vs proportional: {r['vs_one']};  vs zero: {r['vs_zero']}"
          f"  =>  {r['verdict']}")
    d = r["_d"]
    if {"lon", "lat"} <= set(d.columns):
        coords = project_equal_area(d.lon.to_numpy(), d.lat.to_numpy())
        I, z, p = morans_i(coords, r["_resid"])
        r["moran_I"], r["moran_z"], r["moran_p"] = I, z, p
        flag = "spatial term required" if p < 0.05 else "no spatial term needed"
        print(f"   residual Moran's I = {I:.4f} (z = {z:.1f}, p = {p:.3f}) -> {flag}")
    return r


def concentration(df, size="urban_km2_final", cost="cost_km2", top_pct=1.0):
    d = df.dropna(subset=[size, cost]).sort_values(size, ascending=False)
    k = max(1, int(round(len(d) * top_pct / 100.0)))
    return dict(top_pct=top_pct, n_cities=k,
                share_cost=float(d[cost].head(k).sum() / d[cost].sum()),
                share_urban_added=float(d.urban_added_km2.head(k).sum()
                                        / d.urban_added_km2.sum()))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 3000
    size = 10 ** rng.uniform(0.7, 3.5, n)
    setting = rng.integers(5, 120, n).astype(float)
    country = rng.integers(0, 25, n)
    biome = rng.integers(0, 6, n)
    lon = rng.uniform(-180, 180, n); lat = rng.uniform(-23, 23, n)

    BETA, GAMMA = 0.65, 0.40
    y = (0.3 + BETA * np.log10(size) + GAMMA * np.log10(setting)
         + rng.normal(0, 0.25, 25)[country] + rng.normal(0, 0.15, n))
    df = pd.DataFrame(dict(cost_per_km2=10 ** y, urban_km2_final=size,
                           urban_added_km2=size * 0.3, cost_km2=10 ** y * size * 0.3,
                           setting_richness=setting, lon=lon, lat=lat,
                           country=[f"C{c}" for c in country],
                           biome=[f"B{b}" for b in biome]))
    r = fit_scaling(df)
    assert abs(r["beta"] - BETA) < 0.02, r["beta"]
    assert r["ci_lower"] < BETA < r["ci_upper"]
    assert r["verdict"] == "sublinear"
    print(f"planted exponent 0.650   recovered {r['beta']:.4f} "
          f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]  ok")
    assert abs(r["gamma_setting"] - GAMMA) < 0.03, r["gamma_setting"]
    print(f"planted covariate 0.400  recovered {r['gamma_setting']:.4f}  ok")

    y2 = 0.3 + 1.35 * np.log10(size) + rng.normal(0, 0.15, n)
    df2 = df.assign(cost_per_km2=10 ** y2)
    r2 = fit_scaling(df2)
    assert r2["verdict"] == "superlinear" and abs(r2["beta"] - 1.35) < 0.02, r2["verdict"]
    print(f"planted exponent 1.350   recovered {r2['beta']:.4f}  verdict "
          f"{r2['verdict']}  ok")

    y3 = 0.3 + 1.00 * np.log10(size) + rng.normal(0, 0.30, n)
    r3 = fit_scaling(df.assign(cost_per_km2=10 ** y3))
    assert r3["vs_one"] == "not distinguishable from proportional", r3["vs_one"]
    print(f"planted exponent 1.000   recovered {r3['beta']:.4f}  verdict "
          f"{r3['vs_one']}  ok")

    y4 = 0.3 + 0.0 * np.log10(size) + rng.normal(0, 0.20, n)
    r4 = fit_scaling(df.assign(cost_per_km2=10 ** y4))
    assert r4["verdict"] == "scale invariant", (r4["beta"], r4["verdict"])
    print(f"planted exponent 0.000   recovered {r4['beta']:+.4f}  verdict "
          f"{r4['verdict']}  ok")

    r5 = fit_scaling(df, spatial=True)
    assert abs(r5["beta"] - BETA) < 0.03, r5["beta"]
    print(f"spatial block refit      ok  (beta {r5['beta']:.4f} with a "
          f"{len(set(spatial_block(lon, lat)))}-block variance component)")

    coords = project_equal_area(lon, lat)
    I0, z0, p0 = morans_i(coords, rng.normal(size=n), n_perm=199)
    assert p0 > 0.05, (I0, p0)
    clustered = coords[:, 0] / 1000.0 + rng.normal(0, 0.1, n)
    I1, z1, p1 = morans_i(coords, clustered, n_perm=199)
    assert p1 < 0.05 and I1 > 0.5, (I1, p1)
    print(f"Moran's I                ok  (independent {I0:+.3f} p={p0:.2f}; "
          f"clustered {I1:+.3f} p={p1:.3f})")

    c = concentration(df, top_pct=1.0)
    assert 0 < c["share_cost"] < 1
    print(f"concentration statistic  ok  (top 1% of {n} cities carry "
          f"{c['share_cost']*100:.1f}% of cost)")
    print("ALL SCALING TESTS PASS")
