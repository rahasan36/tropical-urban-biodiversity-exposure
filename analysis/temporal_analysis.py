"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import tropics_temporal as t
from tropics_config import TABLE_DIR
from tropics_measure import MEASURE_PATH

OUT = TABLE_DIR / "temporal"
LAB = ["2000–2005", "2005–2010", "2010–2015", "2015–2020"]


def flags():
    import tropics_rangecache as rc
    import resolution_test as rt
    sp, _ = rc.load_index()
    fl, thr = rt.concern_flags(sp)
    n = int(sp.sp_i.max()) + 1
    th = np.zeros(n, bool); na = np.zeros(n, bool)
    rg = np.zeros(n)
    th[fl.sp_i.to_numpy()] = fl.is_threat.to_numpy()
    na[fl.sp_i.to_numpy()] = fl.is_narrow.to_numpy()
    rg[fl.sp_i.to_numpy()] = fl.range_km2.to_numpy()
    return th, na, rg


def city_table():
    import tropics_numbers as tn
    d = tn.analysis_sample(tn.table())
    d = d.dropna(subset=["log_cost", "log_size", "log_setting", "biome", "realm",
                         "country", "form_infill_share", "form_leapfrog_share"]).copy()
    return d.set_index("city_id")


def year_links():
    rows = []
    for y in t.YEARS:
        with np.load(t.DELIN_DIR / f"agglomerations_{y}.npz") as q:
            link, area = q["link2020"], q["area"]
        s = pd.DataFrame(dict(city_id=link, area=area)).groupby("city_id")
        r = pd.DataFrame(dict(delin_km2=s.area.sum(), pieces=s.area.size()))
        r["year"] = y
        rows.append(r.reset_index())
    return pd.concat(rows, ignore_index=True)


def per_city_periods(a, b, V, con, th, na, rar, n_city):
    out = {}
    for j in range(4):
        for k, m in (("concern", con), ("threat", th), ("narrow", na)):
            out[(j, k)] = np.bincount(b[m], weights=V[m, j], minlength=n_city)
        out[(j, "rarity")] = np.bincount(b[con], weights=V[con, j] * rar[con], minlength=n_city)
    return out


def reach_table(a, V, th_s, na_s):
    th, na = th_s[a], na_s[a]
    seen = {k: np.zeros(th_s.size, bool) for k in ("threat", "narrow", "concern")}
    rec = []
    for j in range(4):
        p = V[:, j] > 0
        r = {}
        for k, m in (("threat", th), ("narrow", na), ("concern", th | na)):
            u = np.unique(a[p & m])
            r[f"{k}_reached"] = int(u.size)
            r[f"{k}_new"] = int((~seen[k][u]).sum())
            seen[k][u] = True
            r[f"{k}_cum"] = int(seen[k].sum())
        r["period"] = LAB[j]
        rec.append(r)
    return pd.DataFrame(rec)


def shapley_period(d, y_col):
    import tropics_models as p4
    e = d.rename(columns={y_col: "log_cost"}).copy()
    return p4.shapley(e)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(MEASURE_PATH, allow_pickle=False)
    ce = z["city_epoch"]
    n_city = ce.shape[0]
    th_s, na_s, rg_s = flags()
    d = city_table()
    ids = d.index.to_numpy()
    a, b, V = t.load_pairs()
    reach = reach_table(a, V, th_s, na_s)
    keep = np.isin(b, ids)
    a, b, V = a[keep], b[keep], V[keep]
    th, na = th_s[a], na_s[a]
    con = th | na
    rar = np.where(rg_s[a] > 0, 1.0 / np.maximum(rg_s[a], 1e-6), 0.0)
    pos = V > 0

    cum_th, cum_na, cum_co = set(), set(), set()
    seen_th = np.zeros(th_s.size, bool); seen_na = seen_th.copy(); seen_co = seen_th.copy()
    rec = []
    for j, (y0, y1) in enumerate(t.PERIODS):
        added = float(ce[ids, t.YEAR_CODE[y1]].sum())
        conv_co = float(V[con, j].sum()); conv_th = float(V[th, j].sum())
        conv_na = float(V[na, j].sum()); conv_all = float(V[:, j].sum())
        rar_co = float((V[con, j] * rar[con]).sum())
        rec.append(dict(period=LAB[j], added_km2=added, conv_concern_km2=conv_co,
                        conv_threat_km2=conv_th, conv_narrow_km2=conv_na,
                        conv_all_km2=conv_all, rarity_concern=rar_co,
                        rarity_per_km2=rar_co / added,
                        intensity_concern=conv_co / added,
                        intensity_threat=conv_th / added,
                        intensity_narrow=conv_na / added))
    tot = pd.DataFrame(rec).merge(reach, on="period")
    tot.to_csv(OUT / "temporal_periods.csv", index=False)

    yl = year_links()
    inv = []
    for y in t.YEARS:
        with np.load(t.DELIN_DIR / f"agglomerations_{y}.npz") as q:
            area = q["area"]; link = q["link2020"]
        imperv = float(ce[ids, 1:t.YEAR_CODE[y] + 1].sum())
        inv.append(dict(year=y, agglomerations=int(area.size),
                        delineated_km2=float(area.sum()),
                        impervious_km2_in_2020_footprints=imperv,
                        later_2020_agglomerations=int(np.unique(link).size)))
    inv = pd.DataFrame(inv)
    inv.to_csv(OUT / "temporal_agglomerations.csv", index=False)

    cc = per_city_periods(a, b, V, con, th, na, rar, n_city)
    rows = []
    for j, (y0, y1) in enumerate(t.PERIODS):
        start = ce[ids, 1:t.YEAR_CODE[y0] + 1].sum(1)
        add = ce[ids, t.YEAR_CODE[y1]]
        ex = yl[yl.year == y0].set_index("city_id").reindex(ids)
        rows.append(pd.DataFrame(dict(
            city_id=ids, period=LAB[j], j=j, start_km2=start, added_km2=add,
            conv_concern=cc[(j, "concern")][ids], conv_threat=cc[(j, "threat")][ids],
            conv_narrow=cc[(j, "narrow")][ids], rarity=cc[(j, "rarity")][ids],
            lon=d.lon.to_numpy(), lat=d.lat.to_numpy(),
            start_delin_km2=ex.delin_km2.fillna(0).to_numpy(),
            existed=ex.delin_km2.notna().to_numpy(),
            pieces=ex.pieces.fillna(0).to_numpy().astype(int),
            setting=d.setting_footprint.to_numpy(), log_setting=d.log_setting.to_numpy(),
            realm=d.realm.to_numpy(), biome=d.biome.to_numpy(), country=d.country.to_numpy(),
            form_infill_share=d.form_infill_share.to_numpy(),
            form_leapfrog_share=d.form_leapfrog_share.to_numpy())))
    cp = pd.concat(rows, ignore_index=True)
    cp.to_csv(OUT / "temporal_city_periods.csv.gz", index=False, compression="gzip")
    return tot, inv, cp


def _main():
    tot, inv, cp = run()
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(tot.round(2).to_string())
    print(inv.round(1).to_string())
    print(cp.groupby("period")[["added_km2", "conv_concern"]].sum())


SIZE_EDGES = [0, 1, 5, 25, 100, np.inf]
SIZE_LAB = ["Under 1 km²", "1 to 5 km²", "5 to 25 km²", "25 to 100 km²", "100 km² or more"]


def load_cp():
    cp = pd.read_csv(OUT / "temporal_city_periods.csv.gz")
    tot = (cp.groupby("city_id")
             .agg(added_km2=("added_km2", "sum"), conv_concern=("conv_concern", "sum"),
                  conv_threat=("conv_threat", "sum"), conv_narrow=("conv_narrow", "sum"),
                  rarity=("rarity", "sum")))
    first = cp[cp.j == 0].set_index("city_id")
    keep = [c for c in first.columns if c not in tot.columns and c not in ("period", "j")]
    full = first[keep].join(tot).reset_index()
    full["period"] = "2000–2020"; full["j"] = 4
    full["existed"] = full.start_delin_km2 > 0
    return pd.concat([cp, full[cp.columns]], ignore_index=True)


def prep(x):
    x = x[(x.added_km2 > 0) & (x.start_km2 > 0)].copy()
    x["cost_per_km2"] = x.conv_concern / x.added_km2
    x["rarity_per_km2"] = x.rarity / x.added_km2
    x["log_cost"] = np.log10(x.cost_per_km2.where(x.cost_per_km2 > 0))
    x["log_size"] = np.log10(x.start_km2)
    return x.dropna(subset=["log_cost"])


def fits(j):
    from tropics_scaling import fit_scaling
    import tropics_models as p4
    cp = load_cp()
    x = prep(cp[cp.j == j])
    out = dict(period=x.period.iloc[0], n=int(len(x)))
    for resp in ("cost_per_km2", "rarity_per_km2"):
        r = fit_scaling(x, response=resp, size="start_km2", setting="setting", spatial=True)
        out[resp] = {k: r[k] for k in ("n", "beta", "ci_lower", "ci_upper", "p_value", "R2",
                                       "gamma_setting", "model") if k in r}
    sh = p4.shapley(x)
    out["shapley"] = sh["shapley"]; out["shapley_R2"] = sh["full_R2"]
    out["rho_size"] = float(spearmanr(x.log_size, x.log_cost).correlation)
    out["rho_setting"] = float(spearmanr(x.setting, x.log_cost).correlation)
    out["rho_size_rarity"] = float(spearmanr(x.log_size, x.rarity_per_km2).correlation)
    f = OUT / f"temporal_fit_{j}.json"
    f.write_text(json.dumps(out, indent=1))
    return out


def classes():
    cp = load_cp()
    base = prep(cp[cp.j == 4])
    _, se = pd.qcut(base.setting, 5, retbins=True)
    se[0], se[-1] = -np.inf, np.inf
    rows = []
    for j in range(5):
        x = cp[(cp.j == j) & (cp.added_km2 > 0)].copy()
        x["size_c"] = np.where(~x.existed, "Not yet an agglomeration",
                               pd.cut(x.start_delin_km2, [0, 5, 25, 100, 500, np.inf],
                                      right=False, labels=["Under 5 km²", "5 to 25 km²",
                                                           "25 to 100 km²", "100 to 500 km²",
                                                           "500 km² or more"]).astype(str))
        x["setting_c"] = pd.cut(x.setting, se, labels=False)
        for col in ("size_c", "setting_c"):
            g = x.groupby(col)
            r = pd.DataFrame(dict(added_km2=g.added_km2.sum(), conv_concern=g.conv_concern.sum(),
                                  rarity=g.rarity.sum(), cities=g.size()))
            r["share_added"] = r.added_km2 / r.added_km2.sum()
            r["intensity"] = r.conv_concern / r.added_km2
            r["rarity_per_km2"] = r.rarity / r.added_km2
            r["classed_by"] = col; r["period"] = x.period.iloc[0]
            rows.append(r.reset_index().rename(columns={col: "class"}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "temporal_classes.csv", index=False)
    return out


def shift_share(cl, j0=0, j1=3, by="setting_c", val="intensity"):
    a = cl[(cl.classed_by == by) & (cl.period == LAB[j0])].set_index("class")
    b = cl[(cl.classed_by == by) & (cl.period == LAB[j1])].set_index("class")
    a, b = a.align(b, join="inner", axis=0)
    tot0 = (a.share_added * a[val]).sum(); tot1 = (b.share_added * b[val]).sum()
    comp = ((b.share_added - a.share_added) * (a[val] + b[val]) / 2).sum()
    within = ((b[val] - a[val]) * (a.share_added + b.share_added) / 2).sum()
    return dict(by=by, value=val, start=float(tot0), end=float(tot1), change=float(tot1 - tot0),
                composition=float(comp), within=float(within))


def persistence():
    cp = load_cp()
    x = cp[cp.j < 4].copy()
    x = x[x.added_km2 > 0.05]
    x["y"] = np.log10((x.conv_concern / x.added_km2).where(x.conv_concern > 0))
    w = x.pivot(index="city_id", columns="j", values="y")
    out = {}
    for j in range(3):
        m = w[[j, j + 1]].dropna()
        out[f"{LAB[j]} vs {LAB[j + 1]}"] = dict(n=int(len(m)),
                                               rho=float(spearmanr(m[j], m[j + 1]).correlation))
    m = w[[0, 3]].dropna()
    out[f"{LAB[0]} vs {LAB[3]}"] = dict(n=int(len(m)), rho=float(spearmanr(m[0], m[3]).correlation))
    return out


def emergence():
    cp = load_cp()
    rows = []
    for j in range(4):
        x = cp[(cp.j == j) & (cp.added_km2 > 0)]
        for k, g in x.groupby("existed"):
            rows.append(dict(period=LAB[j], existed=bool(k), cities=int(len(g)),
                             added_km2=float(g.added_km2.sum()),
                             intensity=float(g.conv_concern.sum() / g.added_km2.sum()),
                             rarity_per_km2=float(g.rarity.sum() / g.added_km2.sum())))
    e = pd.DataFrame(rows)
    e["share_added"] = e.added_km2 / e.groupby("period").added_km2.transform("sum")
    e.to_csv(OUT / "temporal_emergence.csv", index=False)
    return e


def summarize():
    cl = classes()
    ss = [shift_share(cl, by=b, val=v) for b in ("setting_c", "size_c")
          for v in ("intensity", "rarity_per_km2")]
    out = dict(shift_share=ss, persistence=persistence(),
               emergence=emergence().to_dict("records"))
    (OUT / "temporal_summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if not a or a[0] == "run":
        _main()
    elif a[0] == "fit":
        print(json.dumps(fits(int(a[1])), indent=1))
    elif a[0] == "summary":
        print(json.dumps(summarize(), indent=1))


def realms():
    cp = load_cp()
    x = cp[cp.j < 4]
    g = x.groupby(["realm", "period"])
    r = pd.DataFrame(dict(added_km2=g.added_km2.sum(), conv_concern=g.conv_concern.sum(),
                          rarity=g.rarity.sum())).reset_index()
    r["intensity"] = r.conv_concern / r.added_km2
    r["rarity_per_km2"] = r.rarity / r.added_km2
    r.to_csv(OUT / "temporal_realms.csv", index=False)
    return r


def rarity_persistence():
    cp = load_cp()
    x = cp[(cp.j < 4) & (cp.added_km2 > 0.05)].copy()
    x["y"] = np.log10((x.rarity / x.added_km2).where(x.rarity > 0))
    w = x.pivot(index="city_id", columns="j", values="y")
    m = w[[0, 3]].dropna()
    return dict(n=int(len(m)), rho=float(spearmanr(m[0], m[3]).correlation))
