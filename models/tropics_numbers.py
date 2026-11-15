"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd

from tropics_config import TABLE_DIR, INTERIM_DIR
import tropics_analysis as ta
import tropics_refit as tr
import tropics_counterfactual as cf

OUT = TABLE_DIR / "numbers.json"
THREAT = ("CR", "EN", "VU")


def table():
    d = ta.load_full()
    d = d.merge(tr.footprint_setting(), on="city_id", how="left")
    d["setting_footprint"] = d.setting_footprint.fillna(0.0)
    hab = pd.read_csv(INTERIM_DIR / "city_habitat.csv")
    d = d.drop(columns=[c for c in d.columns
                        if c.startswith("hab_") or c.startswith("nat_")],
               errors="ignore")
    keep = [c for c in hab.columns if c == "city_id" or c not in d.columns]
    d = d.merge(hab[keep], on="city_id", how="left")
    d["log_size"] = np.log10(d.urban_km2_base.replace(0, np.nan))
    d["log_setting"] = np.log10(d.setting_footprint.clip(lower=0.5))
    return d


def analysis_sample(d=None):
    d = table() if d is None else d
    return d[d.urban_km2_base > 0].copy()


def extent(d):
    return dict(
        n_cities=int(len(d)),
        n_countries=int(d.country.nunique()),
        n_biomes=int(d.biome.nunique()),
        urban_2000_km2=float(d.urban_km2_base.sum()),
        urban_2020_km2=float(d.urban_km2_final.sum()),
        added_km2=float(d.urban_added_km2.sum()),
        growth_pct=float(100 * d.urban_added_km2.sum() / d.urban_km2_base.sum()),
        post2000_share=float(d.urban_added_km2.sum() / d.urban_km2_final.sum()),
        median_size_2000_km2=float(d.urban_km2_base.median()),
        median_size_2020_km2=float(d.urban_km2_final.median()),
        largest_2020_km2=float(d.urban_km2_final.max()),
        cities_without_2000_footprint=int((d.setting_footprint == 0).sum()))


def landcover(d):
    m = d[d.hab_sampled_km2.notna() & (d.hab_sampled_km2 > 0)]
    sc = [c for c in m.columns if c.startswith("hab_") and c.endswith("_km2_scaled")]
    tot = float(m[sc].sum().sum())
    raw = {c.replace("hab_", "").replace("_km2_scaled", ""): float(m[c].sum()) / tot
           for c in sc}
    imp = raw.get("impervious", 0.0)
    rest = {k: v / (1.0 - imp) for k, v in raw.items() if k != "impervious"}
    nat = [f"nat_{f}_km2_scaled" for f in ("infill", "edge", "leapfrog")]
    return dict(
        cities_measured=int(len(m)),
        area_share_covered=float(m.urban_added_km2.sum() / d.urban_added_km2.sum()),
        cities_unmeasured=int(len(d) - len(m)),
        already_impervious_share=imp,
        composition_all={k: v for k, v in sorted(raw.items(), key=lambda x: -x[1])},
        composition_of_remainder={k: v for k, v in
                                  sorted(rest.items(), key=lambda x: -x[1])},
        natural_share=float(sum(raw[k] for k in
                                ("forest", "shrub", "grass", "sparse", "wetland")
                                if k in raw)),
        natural_by_form_km2={c.split("_")[1]: float(m[c].sum())
                             for c in nat if c in m.columns})


def cost(d):
    q = d.cost_per_km2.dropna()
    return dict(
        converted_range_km2=float(d.cost_km2.sum()),
        threat_km2=float(d.cost_threat_km2.sum()),
        narrow_km2=float(d.cost_narrow_km2.sum()),
        multiplier=float(d.cost_km2.sum() / d.urban_added_km2.sum()),
        median_intensity=float(q.median()),
        iqr=[float(q.quantile(0.25)), float(q.quantile(0.75))],
        p5_p95=[float(q.quantile(0.05)), float(q.quantile(0.95))],
        spread_ratio=float(q.quantile(0.95) / q.quantile(0.05)))


def models(d):
    out = {}
    for key in ("A", "B", "C", "D"):
        label, kw = tr.SPECS[key]
        r = tr.fit_scaling(d, spatial=True, **kw)
        out[key] = dict(label=label, n=r["n"], beta=r["beta"],
                        ci=[r["ci_lower"], r["ci_upper"]], p=r["p_value"],
                        beta_std=r["beta_std"], R2=r["R2"],
                        gamma=r.get("gamma_setting"), gamma_se=r.get("gamma_se"))
    return out


def between_block(d, deg=2.0):
    from tropics_scaling import spatial_block
    x = d[d.cost_per_km2 > 0].copy()
    x["blk"] = spatial_block(x.lon.to_numpy(), x.lat.to_numpy(), deg)
    out = {}
    for name, col, logged in (("intensity", "cost_per_km2", True),
                              ("setting_footprint", "setting_footprint", True),
                              ("setting_final_extent", "setting_richness", True),
                              ("size_2000", "urban_km2_base", True)):
        v = np.log10(x[col].clip(lower=0.5)) if logged else x[col]
        g = v.groupby(x.blk)
        between = float((((g.mean() - v.mean()) ** 2) * g.size()).sum() / len(v))
        within = float(((v - g.transform("mean")) ** 2).sum() / len(v))
        out[name] = between / (between + within)
    out["n_blocks"] = int(x.blk.nunique())
    return out


def concentration(d):
    out = {}
    s = d.sort_values("urban_km2_base", ascending=False)
    for pct in (0.1, 1.0, 10.0):
        k = max(1, int(round(len(s) * pct / 100)))
        top = s.head(k)
        cs = float(top.cost_km2.sum() / d.cost_km2.sum())
        ls = float(top.urban_added_km2.sum() / d.urban_added_km2.sum())
        out[f"top_{pct}"] = dict(n=k, cost_share=cs, land_share=ls, ratio=cs / ls)
    return out


def counterfactual(d):
    import numpy as np_
    rel = json.loads((TABLE_DIR / "cf_release.json").read_text())
    a = np_.asarray(rel["alphas"], float)
    m = np_.asarray(rel["reps_m"], float)
    p_ = np_.asarray(rel["reps_p"], float)
    ok = ~np_.isnan(p_).any(axis=1)
    pm = np_.asarray(rel["point"]["measured"], float)
    pp = np_.asarray(rel["point"]["predicted"], float)
    diff = p_[ok] - m[ok]
    m_lo, m_hi = np_.percentile(m, [2.5, 97.5], axis=0)
    p_lo, p_hi = np_.percentile(p_[ok], [2.5, 97.5], axis=0)
    d_lo, d_hi = np_.percentile(diff, [2.5, 97.5], axis=0)

    def at(i):
        return dict(alpha=float(a[i]),
                    measured_pct=float(pm[i]),
                    measured_ci=[float(m_lo[i]), float(m_hi[i])],
                    predicted_pct=float(pp[i]),
                    predicted_ci=[float(p_lo[i]), float(p_hi[i])],
                    paired_pct=float(pp[i] - pm[i]),
                    paired_ci=[float(d_lo[i]), float(d_hi[i])])

    x = d.copy()
    x["m_hat"] = x.cost_per_km2.fillna(0.0)
    _c, obs = cf.curve(x)
    return dict(implied_alpha=float(rel["alpha_hat"]),
                n=int(rel["n"]), n_priced=int(rel["n_priced"]),
                n_replicates=int(ok.sum()),
                smearing_factor=float(rel["smearing"]),
                observed_total_km2=float(obs),
                alphas=[float(v) for v in a],
                measured_pct=[float(v) for v in pm],
                predicted_pct=[float(v) for v in pp],
                predicted_fe_only_pct=[float(v) for v
                                       in rel["point"]["predicted_fe_only"]],
                dispersed=at(0), concentrated=at(len(a) - 1))


def rarity_and_narrow(d):
    out = {}
    for key, resp, sub in (("rarity", "rarity_per_km2", None),
                           ("narrow", "cost_narrow_per_km2", "narrow"),
                           ("threat", "cost_threat_per_km2", None)):
        x = d.copy()
        if key == "narrow":
            x["cost_narrow_per_km2"] = x.cost_narrow_km2 / x.urban_added_km2
            x = x[x.cost_narrow_km2 > 0]
        if key == "threat":
            x["cost_threat_per_km2"] = x.cost_threat_km2 / x.urban_added_km2
        r = tr.fit_scaling(x, response=resp, size="urban_km2_base",
                           setting="setting_footprint", spatial=True)
        out[key] = dict(n=r["n"], beta=r["beta"],
                        ci=[r["ci_lower"], r["ci_upper"]], p=r["p_value"],
                        beta_std=r["beta_std"], R2=r["R2"])
    return out


def structure(d):
    tab, lr = ta.variance_decomposition(d)
    qr = ta.quantile_scaling(d)
    rf = []
    for key, sub in d.groupby("realm"):
        if len(sub) < 150 or sub.country.nunique() < 2:
            continue
        r = tr.fit_scaling(sub, response="cost_per_km2", size="urban_km2_base",
                           setting="setting_footprint", spatial=True)
        rf.append(dict(realm=str(key), n=r["n"], beta=r["beta"],
                       ci=[r["ci_lower"], r["ci_upper"]], p=r["p_value"],
                       gamma=r.get("gamma_setting"), R2=r["R2"],
                       median_intensity=float(sub.cost_per_km2.median()),
                       total_km2=float(sub.cost_km2.sum())))
    rf = sorted(rf, key=lambda x: -x["total_km2"])
    bf = []
    for key, sub in d.groupby("biome"):
        if len(sub) < 150 or sub.country.nunique() < 2:
            continue
        r = tr.fit_scaling(sub, response="cost_per_km2", size="urban_km2_base",
                           setting="setting_footprint", spatial=True)
        bf.append(dict(biome=str(key), n=r["n"], beta=r["beta"],
                       ci=[r["ci_lower"], r["ci_upper"]], p=r["p_value"],
                       gamma=r.get("gamma_setting"), R2=r["R2"],
                       total_km2=float(sub.cost_km2.sum())))
    bf = sorted(bf, key=lambda x: -x["total_km2"])
    return dict(decomposition=tab.to_dict("records"), lr=lr,
                quantile=qr.to_dict("records") if hasattr(qr, "to_dict") else qr,
                realms=rf, biomes=bf)


def drivers(d):
    x = d.copy()
    x["narrow_cost_share"] = np.nan
    tab, meta = ta.driver_model(x)
    return dict(rows=tab.to_dict("records"), **meta)


def export_fig_tables():
    import shutil
    n = json.loads(OUT.read_text())
    for f in ("scaling_spatial.json", "quantile_scaling.csv",
              "variance_decomposition.csv", "driver_model.csv",
              "regime_fits_realm.csv", "regime_fits_biome.csv"):
        src = TABLE_DIR / f
        bak = TABLE_DIR / f.replace(".", "_asdrafted.")
        if src.exists() and not bak.exists():
            shutil.copy(src, bak)
    a = n["models"]["A"]
    (TABLE_DIR / "scaling_spatial.json").write_text(json.dumps(
        {"cost_per_km2": dict(beta=a["beta"], ci_lower=a["ci"][0],
                              ci_upper=a["ci"][1], p_value=a["p"], n=a["n"],
                              R2=a["R2"], gamma_setting=a["gamma"],
                              gamma_se=a["gamma_se"])}, indent=1))
    q = pd.DataFrame(n["structure"]["quantile"])
    q.to_csv(TABLE_DIR / "quantile_scaling.csv", index=False)
    v = pd.DataFrame(n["structure"]["decomposition"])
    v.to_csv(TABLE_DIR / "variance_decomposition.csv", index=False)
    d = pd.DataFrame(n["drivers"]["rows"])
    d.to_csv(TABLE_DIR / "driver_model.csv", index=False)
    r = pd.DataFrame([dict(group=x["realm"], n=x["n"], beta=x["beta"],
                           ci_lower=x["ci"][0], ci_upper=x["ci"][1],
                           gamma_setting=x["gamma"], R2=x["R2"],
                           total_cost=x["total_km2"])
                      for x in n["structure"]["realms"]])
    r.to_csv(TABLE_DIR / "regime_fits_realm.csv", index=False)
    b = pd.DataFrame([dict(group=x["biome"], n=x["n"], beta=x["beta"],
                           ci_lower=x["ci"][0], ci_upper=x["ci"][1],
                           gamma_setting=x["gamma"], R2=x["R2"],
                           total_cost=x["total_km2"])
                      for x in n["structure"]["biomes"]])
    b.to_csv(TABLE_DIR / "regime_fits_biome.csv", index=False)
    return sorted(x.name for x in TABLE_DIR.glob("*asdrafted*"))


def run(parts=None):
    d = analysis_sample()
    todo = dict(extent=extent, landcover=landcover, cost=cost, models=models,
                between_block=between_block, concentration=concentration,
                counterfactual=counterfactual, rarity_and_narrow=rarity_and_narrow,
                structure=structure, drivers=drivers)
    have = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, fn in todo.items():
        if parts and name not in parts:
            continue
        have[name] = fn(d)
        OUT.write_text(json.dumps(have, indent=1))
        print(f"  {name} ok", flush=True)
    return have


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
