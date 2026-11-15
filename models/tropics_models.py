"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json
from itertools import combinations

import numpy as np, pandas as pd
import statsmodels.formula.api as smf

from tropics_config import TABLE_DIR, EXCLUDED_TAXA
import tropics_numbers as tn
import tropics_counterfactual as cf

OUT = TABLE_DIR / "plan4.json"
THREAT = ("CR", "EN", "VU")

GROUPS = {
    "size": ["log_size"],
    "setting": ["log_setting"],
    "biome": ["C(biome)"],
    "realm": ["C(realm)"],
    "country": ["C(country)"],
    "form": ["form_infill_share", "form_leapfrog_share"],
}


def table():
    d = tn.analysis_sample()
    d = d.dropna(subset=["log_cost", "log_size", "log_setting", "biome",
                         "realm", "country", "form_infill_share",
                         "form_leapfrog_share"]).copy()
    return d


def _r2(d, terms):
    if not terms:
        return 0.0
    return float(smf.ols("log_cost ~ " + " + ".join(terms), d).fit().rsquared)


def shapley(d=None, groups=GROUPS):
    d = table() if d is None else d
    names = list(groups)
    k = len(names)
    r2 = {}
    for r in range(k + 1):
        for sub in combinations(names, r):
            terms = [t for n in sub for t in groups[n]]
            r2[frozenset(sub)] = _r2(d, terms)
    from math import factorial
    share = {}
    for i, name in enumerate(names):
        rest = [n for n in names if n != name]
        tot = 0.0
        for r in range(len(rest) + 1):
            w = factorial(r) * factorial(k - r - 1) / factorial(k)
            for sub in combinations(rest, r):
                s = frozenset(sub)
                tot += w * (r2[s | {name}] - r2[s])
        share[name] = tot
    full = r2[frozenset(names)]
    return dict(shapley=share, full_R2=full, n=int(len(d)),
                sum_of_parts=float(sum(share.values())))


def _taxon_exposure(d):
    from tropics_cost import PAIRS_PATH
    z = np.load(PAIRS_PATH, allow_pickle=False)
    idx = json.loads(str(z["sp_index"]))
    meta = json.loads(str(z["sp_meta"]))
    taxon = {}
    for sp, i in idx.items():
        m = meta.get(sp) or ["", "", ""]
        taxon[i] = m[2]
    tx = np.array([taxon.get(i, "") for i in range(max(taxon) + 1)], dtype=object)
    ps, pc, pv = z["pair_sp"], z["pair_city"], z["pair_conv"]
    t = tx[ps]
    out = {}
    for name, keep in (("birds", ("AVES",)), ("mammals", ("MAMMALIA",)),
                       ("herps", ("AMPHIBIA", "REPTILIA"))):
        m = np.isin(t, keep)
        s = (pd.DataFrame(dict(city_id=pc[m], v=pv[m]))
               .groupby("city_id").v.sum().rename(f"cost_{name}_km2"))
        out[name] = s
    return out


def _taxon_setting(d):
    f = pd.read_csv(TABLE_DIR / "setting_by_taxon_v2.csv").set_index("city_id")
    return {c.replace("set_", ""): f[c]
            for c in ("set_birds", "set_mammals", "set_herps")}


def cross_taxon(d=None):
    d = table() if d is None else d
    ex = _taxon_exposure(d)
    se = _taxon_setting(d)
    x = d.copy()
    for s in list(ex.values()) + list(se.values()):
        x = x.join(s, on="city_id")
    for c in [c for c in x.columns if c.startswith(("cost_birds", "cost_mammals",
                                                    "cost_herps", "set_"))]:
        x[c] = x[c].fillna(0.0)
    rows = []
    pairs = [("birds", ("mammals", "herps")),
             ("mammals", ("birds", "herps")),
             ("herps", ("birds", "mammals"))]
    for src, tgt in pairs:
        y = sum(x[f"cost_{t}_km2"] for t in tgt) / x.urban_added_km2
        z = x.assign(_y=y.replace(0, np.nan))
        z = z.dropna(subset=["_y"])
        z["ly"] = np.log10(z["_y"])
        z["lx"] = np.log10(z[f"set_{src}"].clip(lower=0.5))
        f = smf.mixedlm("ly ~ lx + log_size + C(biome)", z,
                        groups=z.country.astype(str)).fit(reml=True, method="lbfgs")
        rows.append(dict(setting_from=src, exposure_of=" + ".join(tgt),
                         n=int(len(z)), gamma=float(f.params["lx"]),
                         gamma_se=float(f.bse["lx"]), p=float(f.pvalues["lx"]),
                         beta_size=float(f.params["log_size"]),
                         beta_size_se=float(f.bse["log_size"])))
    return rows


def rarity_variants(d=None):
    import tropics_refit as trf
    from tropics_cost import PAIRS_PATH
    d = tn.analysis_sample() if d is None else d
    import tropics_flags as tfl
    z = np.load(PAIRS_PATH, allow_pickle=False)
    cat, tax, _sci, ar = tfl.flags(PAIRS_PATH)
    n_sp = cat.size
    meta_path = TABLE_DIR / "cities_cost_meta.json"
    thr_km2 = float(json.loads(meta_path.read_text())["narrow_threshold_km2"])
    keep = ~np.isin(tax, [t.upper() for t in EXCLUDED_TAXA])
    narrow = (ar > 0) & (ar <= thr_km2) & keep
    threat = np.isin(cat, list(THREAT)) & keep
    ps, pc, pv = z["pair_sp"], z["pair_city"], z["pair_conv"]
    w = np.where(ar[ps] > 0, pv / np.maximum(ar[ps], 1e-6), 0.0)
    x = d.copy()
    for name, mask, val in (("rarity_all", np.ones(ps.size, bool), w),
                            ("rarity_threat", threat[ps], w),
                            ("rarity_narrow", narrow[ps], w)):
        s_ = (pd.DataFrame(dict(city_id=pc[mask], v=val[mask]))
                .groupby("city_id").v.sum().rename(name))
        x = x.join(s_, on="city_id")
        x[name] = x[name].fillna(0.0) / x.urban_added_km2
    x["threat_per_km2"] = x.cost_threat_km2 / x.urban_added_km2
    x["narrow_per_km2"] = x.cost_narrow_km2 / x.urban_added_km2
    rows = []
    specs = [("Threatened, unweighted", "threat_per_km2", None),
             ("Narrow range, unweighted", "narrow_per_km2", "narrow"),
             ("Species of concern, rarity weighted", "rarity_per_km2", None),
             ("Threatened, rarity weighted", "rarity_threat", None),
             ("Narrow range, rarity weighted", "rarity_narrow", "narrow")]
    for label, col, sub in specs:
        y = x[x[col] > 0] if sub else x
        r = trf.fit_scaling(y, response=col, size="urban_km2_base",
                            setting="setting_footprint", spatial=True)
        rows.append(dict(label=label, n=r["n"], beta=r["beta"],
                         ci=[r["ci_lower"], r["ci_upper"]], p=r["p_value"],
                         R2=r["R2"]))
    return rows


def counterfactual_by_realm(d=None):
    d = tn.analysis_sample()
    x = d.copy()
    x["m_hat"] = x.cost_per_km2.fillna(0.0)
    out = {}
    for realm, sub in x.groupby("realm"):
        if len(sub) < 150:
            continue
        c, obs = cf.curve(sub)
        out[str(realm)] = dict(n=int(len(sub)), observed=float(obs),
                               alpha=[float(a) for a in c.alpha],
                               ratio=[float(r) for r in c.ratio])
    return out


def counterfactual_by_response(d=None):
    d = tn.analysis_sample()
    x = d.copy()
    x["threat_per_km2"] = x.cost_threat_km2 / x.urban_added_km2
    x["narrow_per_km2"] = x.cost_narrow_km2 / x.urban_added_km2
    out = {}
    for name, col in (("all species of concern", "cost_per_km2"),
                      ("threatened", "threat_per_km2"),
                      ("narrow range", "narrow_per_km2"),
                      ("rarity weighted", "rarity_per_km2")):
        y = x.assign(m_hat=x[col].fillna(0.0))
        c, obs = cf.curve(y)
        out[name] = dict(alpha=[float(a) for a in c.alpha],
                         ratio=[float(r) for r in c.ratio], observed=float(obs))
    return out


def admissible_gaps(d=None, tol=0.10, named=True):
    from tropics_panels import city_names
    d = table() if d is None else d
    x = d[(d.urban_km2_base > 1.0) & (d.urban_added_km2 > 5.0)
          & d.cost_per_km2.notna()].copy()
    if named:
        x = city_names(x)
        x = x[x.name.astype(str).str.strip() != ""].copy()
    x = x.sort_values("log_size").reset_index(drop=True)
    ls = x.log_size.to_numpy(); lc = np.log10(x.cost_per_km2.to_numpy())
    gaps = []
    for i in range(len(x)):
        j0 = np.searchsorted(ls, ls[i] - tol)
        j1 = np.searchsorted(ls, ls[i] + tol)
        for j in range(j0, j1):
            if j > i:
                gaps.append(abs(lc[j] - lc[i]))
    return np.asarray(gaps), int(len(x))


def matched_pairs(d=None, tol=0.10, n_pairs=3, min_km=1500.0, named=True):
    from tropics_panels import city_names
    d = table() if d is None else d
    x = d[(d.urban_km2_base > 1.0) & (d.urban_added_km2 > 5.0)
          & d.cost_per_km2.notna()].copy()
    if named:
        x = city_names(x)
        x = x[x.name.astype(str).str.strip() != ""].copy()
    x = x.sort_values("log_size").reset_index(drop=True)
    ls = x.log_size.to_numpy(); lc = np.log10(x.cost_per_km2.to_numpy())
    cand = []
    for i in range(len(x)):
        j0 = np.searchsorted(ls, ls[i] - tol)
        j1 = np.searchsorted(ls, ls[i] + tol)
        for j in range(j0, j1):
            if j <= i:
                continue
            cand.append((abs(lc[j] - lc[i]), i, j))
    cand.sort(reverse=True)
    lat = np.radians(x.lat.to_numpy()); lon = np.radians(x.lon.to_numpy())

    def far(a, b):
        dl = lon[a] - lon[b]
        c = np.sin(lat[a]) * np.sin(lat[b]) + np.cos(lat[a]) * np.cos(lat[b]) * np.cos(dl)
        return 6371.0 * np.arccos(np.clip(c, -1, 1))

    taken, out = [], []
    for gap, i, j in cand:
        if any(min(far(i, a), far(i, b), far(j, a), far(j, b)) < min_km
               for a, b in taken):
            continue
        taken.append((i, j))
        lo, hi = (i, j) if lc[i] < lc[j] else (j, i)
        out.append(dict(gap_log10=float(gap),
                        low=x.iloc[lo].to_dict(), high=x.iloc[hi].to_dict()))
        if len(out) == n_pairs:
            break
    keep = ["city_id", "lon", "lat", "country", "urban_km2_base",
            "urban_km2_final", "urban_added_km2", "cost_per_km2",
            "setting_footprint"] + (["name"] if named else [])
    for p in out:
        for side in ("low", "high"):
            p[side] = {k: p[side][k] for k in keep}
    return out


CHOSEN_PAIRS = [(391, 668),
                (1201, 1433),
                (618, 747)]


def chosen_pairs(d=None, ids=None, tol=0.13):
    from tropics_panels import city_names
    d = table() if d is None else d
    x = city_names(d).set_index("city_id")
    keep = ["lon", "lat", "country", "urban_km2_base", "urban_km2_final",
            "urban_added_km2", "cost_per_km2", "setting_footprint", "name"]
    out = []
    for a, b in (ids or CHOSEN_PAIRS):
        ra, rb = x.loc[a], x.loc[b]
        d_size = abs(np.log10(ra.urban_km2_base) - np.log10(rb.urban_km2_base))
        if d_size > tol:
            raise ValueError(f"pair {a}/{b} differs by {d_size:.3f} in log size, "
                             f"above the {tol} the row is allowed")
        lo, hi = (ra, rb) if ra.cost_per_km2 < rb.cost_per_km2 else (rb, ra)
        rec = dict(gap_log10=float(abs(np.log10(ra.cost_per_km2)
                                       - np.log10(rb.cost_per_km2))),
                   size_gap_log10=float(d_size))
        for side, r in (("low", lo), ("high", hi)):
            rec[side] = {k: r[k] for k in keep}
            rec[side]["city_id"] = int(r.name)
        out.append(rec)
    return out


def _selftest():
    d = table().sample(1200, random_state=0)
    s = shapley(d, {k: GROUPS[k] for k in ("size", "setting", "biome")})
    assert abs(s["sum_of_parts"] - s["full_R2"]) < 1e-9, s
    print("shapley parts sum to the full R2  ok")
    p = matched_pairs(n_pairs=2)
    for q in p:
        a, b = q["low"], q["high"]
        assert abs(np.log10(a["urban_km2_base"]) - np.log10(b["urban_km2_base"])) <= 0.101
    print("matched pairs hold size fixed      ok")
    for q in chosen_pairs():
        a, b = q["low"], q["high"]
        assert abs(np.log10(a["urban_km2_base"])
                   - np.log10(b["urban_km2_base"])) <= 0.13
        assert a["cost_per_km2"] < b["cost_per_km2"]
        assert str(a["name"]).strip() and str(b["name"]).strip()
    print("drawn pairs named and size matched ok")


def run(parts=None):
    d = table()
    todo = dict(shapley=lambda x: shapley(x), cross_taxon=cross_taxon,
                rarity_variants=rarity_variants,
                cf_realm=counterfactual_by_realm,
                cf_response=counterfactual_by_response,
                matched=chosen_pairs,
                matched_ranked=matched_pairs)
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
    a = sys.argv[1:]
    if a and a[0] == "test":
        _selftest()
    else:
        run(a or None)
