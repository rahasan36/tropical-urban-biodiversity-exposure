"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from tropics_config import TABLE_DIR, INTERIM_DIR
import tropics_scaling as ts

RESP = "cost_per_km2"


def load_full():
    df = pd.read_csv(TABLE_DIR / "cities_cost.csv")
    add = pd.read_csv(INTERIM_DIR / "city_form.csv")
    add = add[[c for c in add.columns if c == "city_id" or c not in df.columns]]
    df = df.merge(add, on="city_id", how="left")
    form_cols = [c for c in ("form_infill_km2", "form_edge_km2", "form_leapfrog_km2")
                 if c in df.columns]
    if form_cols:
        tot = df[form_cols].sum(axis=1).replace(0, np.nan)
        for c in form_cols:
            df[c.replace("_km2", "_share")] = df[c] / tot
    hab_cols = [c for c in df.columns if c.startswith("hab_")]
    if hab_cols:
        tot = df[hab_cols].sum(axis=1).replace(0, np.nan)
        for c in hab_cols:
            df[c.replace("_km2", "_share")] = df[c] / tot
        nat = [c for c in hab_cols if c.split("_")[1] in
               ("forest", "shrub", "grass", "sparse", "wetland")]
        df["natural_share"] = df[nat].sum(axis=1) / tot
        df["habitat_measured"] = tot.notna()
    df["growth_ratio"] = df.urban_km2_final / df.urban_km2_base.replace(0, np.nan)
    df["growth_pct"] = (df.growth_ratio - 1.0) * 100.0
    df["narrow_cost_share"] = (df.cost_narrow_km2 / df.cost_km2.replace(0, np.nan))
    df["threat_cost_share"] = (df.cost_threat_km2 / df.cost_km2.replace(0, np.nan))
    df["log_cost"] = np.log10(df[RESP].replace(0, np.nan))
    df["log_size"] = np.log10(df.urban_km2_final.replace(0, np.nan))
    df["log_added"] = np.log10(df.urban_added_km2.replace(0, np.nan))
    df["log_setting"] = np.log10(df.setting_richness.clip(lower=0.5))
    return df


def variance_decomposition(df, response="log_cost"):
    d = df.dropna(subset=[response, "log_size", "log_setting", "biome",
                          "country", "realm"]).copy()
    steps = [("size", ["log_size"]),
             ("+ setting richness", ["log_size", "log_setting"]),
             ("+ biome", ["log_size", "log_setting", "C(biome)"]),
             ("+ realm", ["log_size", "log_setting", "C(biome)", "C(realm)"]),
             ("+ country", ["log_size", "log_setting", "C(biome)", "C(realm)",
                            "C(country)"])]
    rows, prev = [], 0.0
    for label, terms in steps:
        f = smf.ols(f"{response} ~ " + " + ".join(terms), d).fit()
        rows.append(dict(model=label, k=int(f.df_model), R2=float(f.rsquared),
                         increment=float(f.rsquared) - prev,
                         adj_R2=float(f.rsquared_adj), aic=float(f.aic)))
        prev = float(f.rsquared)
    f_no_size = smf.ols(f"{response} ~ log_setting + C(biome) + C(realm) + C(country)",
                        d).fit()
    f_size = smf.ols(f"{response} ~ log_size + log_setting + C(biome) + C(realm) "
                     f"+ C(country)", d).fit()
    rows.append(dict(model="everything except size", k=int(f_no_size.df_model),
                     R2=float(f_no_size.rsquared), increment=np.nan,
                     adj_R2=float(f_no_size.rsquared_adj), aic=float(f_no_size.aic)))
    rows.append(dict(model="size added last", k=int(f_size.df_model),
                     R2=float(f_size.rsquared),
                     increment=float(f_size.rsquared - f_no_size.rsquared),
                     adj_R2=float(f_size.rsquared_adj), aic=float(f_size.aic)))
    lr = 2 * (f_size.llf - f_no_size.llf)
    from scipy import stats as sstats
    p = float(sstats.chi2.sf(lr, 1))
    return pd.DataFrame(rows), dict(lr_stat=float(lr), lr_p=p, n=int(len(d)))


def quantile_scaling(df, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    d = df.dropna(subset=["log_cost", "log_size", "log_setting"])
    rows = []
    for q in qs:
        f = smf.quantreg("log_cost ~ log_size + log_setting", d).fit(q=q)
        rows.append(dict(quantile=q, beta=float(f.params["log_size"]),
                         se=float(f.bse["log_size"]),
                         ci_lower=float(f.conf_int().loc["log_size", 0]),
                         ci_upper=float(f.conf_int().loc["log_size", 1]),
                         gamma=float(f.params["log_setting"]),
                         n=int(f.nobs)))
    return pd.DataFrame(rows)


def regime_fits(df, by="realm", min_n=150):
    rows = []
    for key, sub in df.groupby(by):
        if len(sub) < min_n or sub.country.nunique() < 2:
            continue
        r = ts.fit_scaling(sub, response=RESP, spatial=True)
        rows.append(dict(group=str(key), n=r["n"], beta=r["beta"], se=r["se"],
                         ci_lower=r["ci_lower"], ci_upper=r["ci_upper"],
                         gamma_setting=r.get("gamma_setting", np.nan),
                         R2=r["R2"], verdict=r["verdict"],
                         median_cost=float(sub[RESP].median()),
                         total_cost=float(sub.cost_km2.sum())))
    return pd.DataFrame(rows).sort_values("total_cost", ascending=False)


def driver_model(df):
    cand = ["log_size", "log_setting", "growth_pct", "form_leapfrog_share",
            "form_infill_share", "narrow_cost_share", "abs_lat"]
    d = df.copy()
    d["abs_lat"] = d.lat.abs()
    cand = [c for c in cand if c in d.columns and d[c].notna().sum() > 0.6 * len(d)]
    d = d.dropna(subset=["log_cost", "biome", "country"] + cand)
    z = d.copy()
    for c in cand:
        s = z[c].std()
        z[c] = (z[c] - z[c].mean()) / (s if s else 1.0)
    f = smf.mixedlm("log_cost ~ " + " + ".join(cand) + " + C(biome)", z,
                    groups=z.country.astype(str)).fit(reml=True, method="lbfgs")
    rows = []
    for c in cand:
        rows.append(dict(predictor=c, beta_std=float(f.params[c]),
                         se=float(f.bse[c]), p=float(f.pvalues[c])))
    out = pd.DataFrame(rows).reindex(
        pd.Series(np.argsort(-np.abs([r["beta_std"] for r in rows]))).values)
    return out.reset_index(drop=True), dict(n=int(len(z)))


def run_all():
    df = load_full()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    vd, lr = variance_decomposition(df)
    vd.to_csv(TABLE_DIR / "variance_decomposition.csv", index=False)
    qr = quantile_scaling(df)
    qr.to_csv(TABLE_DIR / "quantile_scaling.csv", index=False)
    rf_realm = regime_fits(df, "realm")
    rf_realm.to_csv(TABLE_DIR / "regime_fits_realm.csv", index=False)
    rf_biome = regime_fits(df, "biome")
    rf_biome.to_csv(TABLE_DIR / "regime_fits_biome.csv", index=False)
    dm, dmeta = driver_model(df)
    dm.to_csv(TABLE_DIR / "driver_model.csv", index=False)
    summary = dict(likelihood_ratio=lr, driver_n=dmeta["n"])
    (TABLE_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    return df, vd, lr, qr, rf_realm, rf_biome, dm


if __name__ == "__main__":
    df, vd, lr, qr, rf_realm, rf_biome, dm = run_all()
    pd.set_option("display.width", 200)
    print("=== variance decomposition (response: log marginal cost) ===")
    print(vd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nlikelihood ratio for adding size last: {lr['lr_stat']:.2f}, "
          f"p = {lr['lr_p']:.3f}, n = {lr['n']:,}")
    print("\n=== quantile regression ===")
    print(qr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== by realm ===")
    print(rf_realm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== standardised drivers ===")
    print(dm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
