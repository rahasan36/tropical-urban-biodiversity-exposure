"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd

from tropics_config import INTERIM_DIR, TABLE_DIR
PAIRS = INTERIM_DIR / "species_city_pairs.npz"
CITIES = TABLE_DIR / "cities_cost.csv"
OUT = TABLE_DIR

THREAT_CATS = {"CR", "EN", "VU"}
NARROW_PCTL = 25


def concentration(sp, city, conv):
    sp = np.asarray(sp); conv = np.asarray(conv, dtype=float)
    order = np.lexsort((-conv, sp))
    sp, conv = sp[order], conv[order]
    start = np.r_[0, np.flatnonzero(np.diff(sp)) + 1]
    stop = np.r_[start[1:], len(sp)]
    ids = sp[start]
    total = np.add.reduceat(conv, start)
    n_city = stop - start
    top1 = conv[start]
    top3 = np.array([conv[a:min(a + 3, b)].sum() for a, b in zip(start, stop)])
    n90 = np.empty(len(ids), dtype=np.int32)
    for i, (a, b) in enumerate(zip(start, stop)):
        c = np.cumsum(conv[a:b])
        n90[i] = int(np.searchsorted(c, 0.9 * c[-1]) + 1)
    return pd.DataFrame(dict(sp_ix=ids, converted_km2=total, n_city=n_city,
                             top1_share=top1 / total, top3_share=top3 / total,
                             n_city_90=n90))


def duan_smearing(resid_log10):
    return float(np.mean(10.0 ** np.asarray(resid_log10)))


def combined_expectation(p, mu_log10, smear):
    return np.asarray(p) * (10.0 ** np.asarray(mu_log10)) * smear


def selftest():
    sp = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    city = np.arange(len(sp))
    conv = np.array([7.0,
                     1.0, 1.0, 1.0, 1.0,
                     10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    c = concentration(sp, city, conv).set_index("sp_ix")
    assert c.loc[0, "n_city"] == 1 and abs(c.loc[0, "top1_share"] - 1.0) < 1e-12
    assert c.loc[0, "n_city_90"] == 1
    assert c.loc[1, "n_city"] == 4 and abs(c.loc[1, "top1_share"] - 0.25) < 1e-12
    assert abs(c.loc[1, "top3_share"] - 0.75) < 1e-12
    assert c.loc[1, "n_city_90"] == 4, c.loc[1, "n_city_90"]
    assert c.loc[2, "n_city"] == 10 and abs(c.loc[2, "top1_share"] - 10 / 19) < 1e-12
    assert c.loc[2, "n_city_90"] == 9, c.loc[2, "n_city_90"]
    assert abs(c.loc[2, "converted_km2"] - 19.0) < 1e-12
    print("part 1  concentration statistics          ok")

    rng = np.random.default_rng(0)
    q = rng.permutation(len(sp))
    c2 = concentration(sp[q], city[q], conv[q]).set_index("sp_ix")
    assert np.allclose(c.sort_index().values, c2.sort_index().values)
    print("part 1  invariant to input order          ok")

    import statsmodels.formula.api as smf
    rng = np.random.default_rng(7)
    n = 40000
    x = rng.normal(0, 1, n)
    b_occ, b_int = 0.80, -0.25
    p = 1.0 / (1.0 + np.exp(-(-0.5 + b_occ * x)))
    y_pos = rng.binomial(1, p)
    mu = 1.0 + b_int * x
    y = np.where(y_pos == 1, 10.0 ** (mu + rng.normal(0, 0.3, n)), 0.0)
    d = pd.DataFrame(dict(x=x, y=y, any_y=(y > 0).astype(int)))
    mo = smf.logit("any_y ~ x", data=d).fit(disp=0)
    assert abs(mo.params["x"] - b_occ) < 0.05, mo.params["x"]
    pos = d[d.any_y == 1].copy(); pos["ly"] = np.log10(pos.y)
    mi = smf.ols("ly ~ x", data=pos).fit()
    assert abs(mi.params["x"] - b_int) < 0.02, mi.params["x"]
    print(f"part 2  occurrence slope {mo.params['x']:+.3f} (true {b_occ:+.2f}), "
          f"intensity {mi.params['x']:+.3f} (true {b_int:+.2f})  ok")

    smear = duan_smearing(mi.resid)
    grid = pd.DataFrame(dict(x=x))
    pred = combined_expectation(mo.predict(grid), mi.predict(grid), smear)
    rel = abs(pred.mean() - y.mean()) / y.mean()
    assert rel < 0.02, rel
    print(f"part 2  recombined mean within {rel*100:.2f}% of the simulated mean  ok")

    s = 0.3
    e = rng.normal(0, s, 400000)
    want = float(np.exp((s * np.log(10)) ** 2 / 2))
    got = duan_smearing(e)
    assert abs(got - want) / want < 0.01, (got, want)
    print(f"part 2  Duan smearing {got:.4f} against analytic {want:.4f}  ok")
    print("ALL SELF-TESTS PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()


RESTEST = INTERIM_DIR / "restest"
RULES = {"within": "30_within", "centroid": "30_centroid", "touches": "30_touches"}


def load_rule_pairs(rule):
    import resolution_resume as rr
    import tropics_rangecache as rc
    pairs, meta = rr.load_slices(RULES[rule])
    idx, _parts = rc.load_index()
    pairs = pairs.assign(sp_id=pairs.sp_i.map(dict(zip(idx.sp_i.astype(int),
                                                       idx.sp_id))))
    assert pairs.sp_id.notna().all(), "a boundary-pass species is not in the cache index"
    return pairs, meta


def species_table():
    import tropics_flags as tfl
    from tropics_cost import PAIRS_PATH
    cat, tax, sci, area = tfl.flags(PAIRS_PATH)
    return pd.DataFrame(dict(sp_ix=np.arange(len(cat)), sci_name=sci,
                             category=np.char.upper(cat.astype(str)),
                             taxon=tax, range_km2=area))


def published_pool():
    from tropics_cost import PAIRS_PATH
    with np.load(PAIRS_PATH, allow_pickle=False) as z:
        return np.union1d(np.unique(z["pair_sp"]), np.unique(z["set_sp"]))


def analysis_pool(sp_tab, pool_ix):
    d = sp_tab[sp_tab.sp_ix.isin(pool_ix)].copy()
    thr = float(np.percentile(d.range_km2, NARROW_PCTL))
    d["is_threat"] = d.category.isin(THREAT_CATS)
    d["is_narrow"] = d.range_km2 <= thr
    d["in_pool"] = d.is_threat | d.is_narrow
    return d, thr


def part1():
    import tropics_rangecache as rc
    from tropics_cost import PAIRS_PATH
    sp_tab = species_table()
    pool, thr = analysis_pool(sp_tab, published_pool())
    assert len(pool) == 26528, len(pool)
    assert abs(thr - 5730.255) < 0.01, thr
    assert int(pool.is_threat.sum()) == 4835 and int(pool.is_narrow.sum()) == 6632
    with np.load(PAIRS_PATH, allow_pickle=True) as z:
        rev = {v: k for k, v in json.loads(str(z["sp_index"])).items()}
    pool["sp_id"] = [rev.get(int(i)) for i in pool.sp_ix]
    assert pool.sp_id.notna().all()
    concern = pool[pool.in_pool].set_index("sp_id")
    print(f"analysed pool {len(pool):,}   narrow-range threshold {thr:,.1f} km2   "
          f"species of concern {len(concern):,}")

    per_rule, meta = {}, {}
    for rule in RULES:
        pairs, _m = load_rule_pairs(rule)
        p = pairs[pairs.sp_id.isin(concern.index) & (pairs.conv_km2 > 0)]
        codes, uniq = pd.factorize(p.sp_id.to_numpy())
        c = concentration(codes, p.city_id.to_numpy(), p.conv_km2.to_numpy())
        c["sp_id"] = uniq[c.pop("sp_ix").to_numpy()]
        c = c.join(
            concern[["sci_name", "category", "taxon", "range_km2",
                     "is_threat", "is_narrow"]], on="sp_id")
        c["frac_converted"] = c.converted_km2 / c.range_km2
        c["rule"] = rule
        per_rule[rule] = c
        meta[rule] = dict(species_positive=int(len(c)),
                          agglomerations=int(p.city_id.nunique()),
                          converted_km2=round(float(c.converted_km2.sum()), 1))
        print(f"  {rule:9s} species with converted range {len(c):,}   "
              f"converted {meta[rule]['converted_km2']:,.0f} km2")
    meta["pool"] = dict(analysed=int(len(pool)), of_concern=int(len(concern)),
                        narrow_threshold_km2=round(thr, 3),
                        published_converted_km2=1154469.7)
    per_rule["centroid"].to_csv(OUT / "species_concentration.csv", index=False)
    pd.concat(per_rule.values()).to_csv(OUT / "species_concentration_rules.csv",
                                        index=False)
    (OUT / "species_concentration_meta.json").write_text(json.dumps(meta, indent=1))
    return per_rule


def part1_strata(per_rule, n_strata=5):
    from scipy.stats import spearmanr
    c = per_rule["centroid"].copy()
    c["log_range"] = np.log10(c.range_km2)
    c["stratum"] = pd.qcut(c.log_range, n_strata, labels=False)
    lines = []

    print("\n=== concentration by range-size quintile, centroid rule ===")
    print(f"{'quintile':>8} {'range (km2)':>22} {'n':>6} {'n_contrib':>10} "
          f"{'top1 %':>7} {'top3 %':>7}  {'rho(frac,top1)':>15} {'P':>9}")
    for s, g in c.groupby("stratum"):
        lo, hi = g.range_km2.min(), g.range_km2.max()
        rho, p = spearmanr(g.frac_converted, g.top1_share)
        print(f"{int(s)+1:>8} {lo:>10,.0f} to {hi:<9,.0f} {len(g):>6} "
              f"{g.n_city.median():>10,.0f} {g.top1_share.median()*100:>7.1f} "
              f"{g.top3_share.median()*100:>7.1f}  {rho:>15.3f} {p:>9.1e}")
        lines.append(dict(stratum=int(s) + 1, range_lo=float(lo), range_hi=float(hi),
                          n=int(len(g)), n_contrib_median=float(g.n_city.median()),
                          top1_median=float(g.top1_share.median()),
                          top3_median=float(g.top3_share.median()),
                          rho=float(rho), p=float(p)))
    rho_all, p_all = spearmanr(c.frac_converted, c.top1_share)
    print(f"\nunstratified rho(proportional exposure, top-1 share) "
          f"{rho_all:+.3f} (P {p_all:.1e})")
    print("within-quintile rho is the controlled version of that number")

    print("\n=== by class, centroid rule ===")
    taxa = []
    for t, g in c.groupby("taxon"):
        print(f"  {t:<9} n {len(g):>5}  median range {g.range_km2.median():>9,.0f} km2  "
              f"n_contrib {g.n_city.median():>5,.0f}  top1 {g.top1_share.median()*100:>5.1f}%")
        taxa.append(dict(taxon=str(t), n=int(len(g)),
                         range_median=float(g.range_km2.median()),
                         n_contrib_median=float(g.n_city.median()),
                         top1_median=float(g.top1_share.median()),
                         top3_median=float(g.top3_share.median())))

    print("\n=== proportional-exposure bands, centroid rule ===")
    bands = [("all", 0.0), (">=0.1%", 0.001), (">=1%", 0.01), (">=5%", 0.05)]
    band_rows = []
    for lab, t in bands:
        g = c[c.frac_converted >= t]
        if len(g) < 5:
            continue
        print(f"  {lab:>7} n {len(g):>5}  n_contrib median {g.n_city.median():>4,.0f} "
              f"(IQR {g.n_city.quantile(.25):.0f} to {g.n_city.quantile(.75):.0f})  "
              f"top1 {g.top1_share.median()*100:>5.1f}%  top3 {g.top3_share.median()*100:>5.1f}%  "
              f"share with top1>50% {100*(g.top1_share > .5).mean():>5.1f}%  "
              f"median range {g.range_km2.median():>8,.0f} km2")
        band_rows.append(dict(band=lab, n=int(len(g)),
                              n_contrib_median=float(g.n_city.median()),
                              top1_median=float(g.top1_share.median()),
                              top3_median=float(g.top3_share.median()),
                              share_top1_gt50=float((g.top1_share > .5).mean()),
                              range_median=float(g.range_km2.median())))

    print("\n=== boundary-rule sensitivity of threshold membership ===")
    sets = {}
    for rule, d in per_rule.items():
        for lab, t in [(">=1%", 0.01), (">=5%", 0.05)]:
            sets[(rule, lab)] = set(d.loc[d.frac_converted >= t, "sp_id"])
    sens = []
    for lab in (">=1%", ">=5%"):
        cen = sets[("centroid", lab)]
        wit, tou = sets[("within", lab)], sets[("touches", lab)]
        stable = cen & wit & tou
        anyr = cen | wit | tou
        print(f"  {lab:>5}  centroid {len(cen):>5}  within {len(wit):>5}  "
              f"touches {len(tou):>5}  in all three {len(stable):>5}  "
              f"in any {len(anyr):>5}  stable share of centroid "
              f"{100*len(stable)/max(len(cen),1):.1f}%")
        g = per_rule["centroid"][per_rule["centroid"].sp_id.isin(stable)]
        print(f"         on the {len(g)} species all rules agree on: "
              f"n_contrib median {g.n_city.median():.0f}, top1 "
              f"{g.top1_share.median()*100:.1f}%, top1>50% in "
              f"{100*(g.top1_share > .5).mean():.1f}%")
        sens.append(dict(band=lab, centroid=len(cen), within=len(wit),
                         touches=len(tou), all_three=len(stable), any_rule=len(anyr),
                         stable_n_contrib_median=float(g.n_city.median()),
                         stable_top1_median=float(g.top1_share.median()),
                         stable_share_top1_gt50=float((g.top1_share > .5).mean())))
    json.dump(dict(strata=lines, bands=band_rows, taxa=taxa, rule_sensitivity=sens,
                   rho_unstratified=[float(rho_all), float(p_all)]),
              open(OUT / "species_concentration_strata.json", "w"), indent=1)
    return c


SPECS = {"A: size, setting, land added": "ls + lset + ladd",
         "B: size, setting (published intensity spec)": "ls + lset"}
FE = " + C(biome)"


def city_frame():
    d = pd.read_csv(CITIES)
    d = d[d.urban_km2_base > 0].copy()
    d["y"] = d.cost_narrow_km2
    d["any_y"] = (d.y > 0).astype(int)
    d["ls"] = np.log10(d.urban_km2_base)
    d["lset"] = np.log10(d.setting_richness + 1.0)
    d["ladd"] = np.log10(d.urban_added_km2.clip(lower=1e-3))
    d["intensity"] = d.y / d.urban_added_km2
    return d


def fit_two_part(d, covs):
    import statsmodels.formula.api as smf
    occ = smf.logit(f"any_y ~ {covs}{FE}", data=d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d.country})
    pos = d[d.any_y == 1].copy()
    pos["li"] = np.log10(pos.intensity)
    inten = smf.ols(f"li ~ {covs}{FE}", data=pos).fit(
        cov_type="cluster", cov_kwds={"groups": pos.country})
    return occ, inten, duan_smearing(inten.resid)


def size_grid(d, n=25):
    g = pd.DataFrame(dict(ls=np.linspace(d.ls.quantile(.01), d.ls.quantile(.99), n)))
    g["lset"] = d.lset.median()
    g["ladd"] = d.ladd.median()
    g["biome"] = d.biome.mode().iloc[0]
    return g


def part2(n_boot=200, seed=11):
    d = city_frame()
    print(f"estimation sample {len(d):,}   with positive narrow-range exposure "
          f"{int(d.any_y.sum()):,} ({100*d.any_y.mean():.1f}%)")
    g = size_grid(d)
    countries = d.country.dropna().unique()
    store = {}
    for label, covs in SPECS.items():
        rng = np.random.default_rng(seed)
        occ, inten, smear = fit_two_part(d, covs)
        print(f"\n=== {label} ===   Duan smearing {smear:.4f}")
        for v in covs.replace(" ", "").split("+"):
            print(f"  {v:>5}  occurrence {occ.params[v]:>+9.4f} "
                  f"(se {occ.bse[v]:.4f}, P {occ.pvalues[v]:.1e})   "
                  f"intensity | positive {inten.params[v]:>+9.4f} "
                  f"(se {inten.bse[v]:.4f}, P {inten.pvalues[v]:.1e})")
        p = occ.predict(g).to_numpy(); mu = inten.predict(g).to_numpy()
        comb = combined_expectation(p, mu, smear)
        P, M, C, ok, conv = [], [], [], 0, 0
        for _ in range(n_boot):
            pick = rng.choice(countries, size=len(countries), replace=True)
            b = pd.concat([d[d.country == c] for c in pick], ignore_index=True)
            if b.any_y.sum() < 50:
                continue
            try:
                o2, i2, s2 = fit_two_part(b, covs)
            except Exception:
                continue
            conv += int(bool(getattr(o2, "mle_retvals", {}).get("converged", True)))
            P.append(o2.predict(g).to_numpy()); M.append(i2.predict(g).to_numpy())
            C.append(combined_expectation(o2.predict(g), i2.predict(g), s2)); ok += 1
        P, M, C = map(np.array, (P, M, C))
        q = lambda a: (np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0))
        out = pd.DataFrame(dict(
            urban_km2_base=10.0 ** g.ls, log10_size=g.ls,
            p_occurrence=p, p_lo=q(P)[0], p_hi=q(P)[1],
            intensity_pos=10.0 ** mu * smear,
            intensity_pos_lo=10.0 ** q(M)[0] * smear,
            intensity_pos_hi=10.0 ** q(M)[1] * smear,
            combined=comb, combined_lo=q(C)[0], combined_hi=q(C)[1]))
        ratio = C[:, -1] / C[:, 0]
        f, l = out.iloc[0], out.iloc[-1]
        print(f"  bootstrap {ok}/{n_boot} refitted, {conv} with a converged logit")
        print(f"  P(exposure)  {f.p_occurrence:.4f} -> {l.p_occurrence:.4f} "
              f"[{l.p_lo:.3f}, {l.p_hi:.3f}]   ratio {l.p_occurrence/f.p_occurrence:.2f}")
        print(f"  intensity|+  {f.intensity_pos:.3f} -> {l.intensity_pos:.3f} "
              f"[{l.intensity_pos_lo:.3f}, {l.intensity_pos_hi:.3f}]   "
              f"ratio {l.intensity_pos/f.intensity_pos:.2f}")
        print(f"  COMBINED     {f.combined:.4f} -> {l.combined:.4f} "
              f"[{l.combined_lo:.3f}, {l.combined_hi:.3f}]")
        print(f"  combined ratio largest/smallest {l.combined/f.combined:.2f} "
              f"[{np.percentile(ratio,2.5):.2f}, {np.percentile(ratio,97.5):.2f}]   "
              f"rising in {100*np.mean(ratio>1):.1f}% of replicates, monotone in "
              f"{100*np.mean(np.all(np.diff(C,axis=1)>0,axis=1)):.1f}%")
        tag = label.split(":")[0]
        out.to_csv(OUT / f"narrow_two_part_{tag}.csv", index=False)
        store[label] = dict(
            covariates=covs, smearing=smear, boot_ok=ok, boot_converged=conv,
            occurrence={v: [float(occ.params[v]), float(occ.bse[v]), float(occ.pvalues[v])]
                        for v in covs.replace(" ", "").split("+")},
            intensity={v: [float(inten.params[v]), float(inten.bse[v]), float(inten.pvalues[v])]
                       for v in covs.replace(" ", "").split("+")},
            p_lo_hi=[float(f.p_occurrence), float(l.p_occurrence)],
            intensity_lo_hi=[float(f.intensity_pos), float(l.intensity_pos)],
            combined_lo_hi=[float(f.combined), float(l.combined)],
            combined_ratio=[float(l.combined / f.combined),
                            float(np.percentile(ratio, 2.5)),
                            float(np.percentile(ratio, 97.5))],
            share_rising=float(np.mean(ratio > 1)),
            share_monotone=float(np.mean(np.all(np.diff(C, axis=1) > 0, axis=1))))
    store["sample"] = dict(n=int(len(d)), n_positive=int(d.any_y.sum()),
                           size_grid_km2=[float(10 ** g.ls.iloc[0]),
                                          float(10 ** g.ls.iloc[-1])])
    json.dump(store, open(OUT / "narrow_two_part_meta.json", "w"), indent=1)
    return store


def part2_validate(deg=10.0, k=10, buffer_km=250.0, seed=3, spec="B"):
    import statsmodels.formula.api as smf
    import spatial_cv_blocks as cv2
    from sklearn.metrics import roc_auc_score, brier_score_loss
    covs = [v for kk, v in SPECS.items() if kk.startswith(spec)][0]
    d = city_frame().reset_index(drop=True)
    d["_blk"] = cv2.blocks(d.lon.to_numpy(), d.lat.to_numpy(), deg)
    u = np.unique(d._blk)
    rng = np.random.default_rng(seed)
    d["_fold"] = d._blk.map(dict(zip(u, rng.integers(0, k, len(u)))))
    xyz = cv2._unit_xyz(d.lon.to_numpy(), d.lat.to_numpy())
    rows = []
    for f in range(k):
        tr_i = np.flatnonzero((d._fold != f).to_numpy())
        te_i = np.flatnonzero((d._fold == f).to_numpy())
        if len(te_i) < 20 or len(tr_i) < 200:
            continue
        if buffer_km > 0:
            tr_i = tr_i[~cv2._near_any(xyz[tr_i], xyz[te_i], buffer_km)]
            if len(tr_i) < 200:
                continue
        tr, te = d.iloc[tr_i].copy(), d.iloc[te_i].copy()
        if tr.any_y.sum() < 50:
            continue
        keep = te.biome.isin(tr.biome.unique())
        te = te[keep]
        if len(te) < 20:
            continue
        try:
            o, i2, sm = fit_two_part(tr, covs)
        except Exception:
            continue
        p = np.asarray(o.predict(te), float)
        mu = np.asarray(i2.predict(te), float)
        rows.append(pd.DataFrame(dict(fold=f, any_y=te.any_y.to_numpy(),
                                      p=p, mu=mu, smear=sm,
                                      intensity=te.intensity.to_numpy())))
    r = pd.concat(rows, ignore_index=True).dropna()
    r["comb"] = r.p * (10.0 ** r.mu) * r.smear
    auc = roc_auc_score(r.any_y, r.p)
    brier = brier_score_loss(r.any_y, r.p)
    cal = smf.logit("any_y ~ lp", data=r.assign(
        lp=np.log(r.p.clip(1e-6, 1 - 1e-6) / (1 - r.p.clip(1e-6, 1 - 1e-6))))).fit(disp=0)
    pos = r[r.any_y == 1]
    ss = 1 - ((np.log10(pos.intensity) - pos.mu) ** 2).sum() / \
             ((np.log10(pos.intensity) - np.log10(pos.intensity).mean()) ** 2).sum()
    obs = r.intensity.to_numpy()
    r2c = 1 - ((obs - r.comb) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    print(f"\n=== spatially separated validation, spec {spec}, {deg:.0f} degree "
          f"blocks, {buffer_km:.0f} km buffer ===")
    print(f"  scored agglomerations {len(r):,} in {r.fold.nunique()} folds "
          f"({int(r.any_y.sum()):,} positive)")
    print(f"  occurrence  AUC {auc:.3f}   Brier {brier:.4f}   "
          f"calibration slope {cal.params['lp']:.3f} "
          f"(1.0 is perfect), intercept {cal.params['Intercept']:+.3f}")
    print(f"  mean predicted {r.p.mean():.3f} against observed rate {r.any_y.mean():.3f}")
    print(f"  intensity among held-out positives, R2 on log10 {ss:+.3f}")
    print(f"  combined prediction against observed intensity, R2 {r2c:+.3f}")
    json.dump(dict(spec=spec, covariates=covs, deg=deg, buffer_km=buffer_km,
                   n_scored=int(len(r)), folds=int(r.fold.nunique()),
                   n_positive=int(r.any_y.sum()), auc=float(auc),
                   brier=float(brier), cal_slope=float(cal.params["lp"]),
                   cal_intercept=float(cal.params["Intercept"]),
                   mean_pred=float(r.p.mean()), obs_rate=float(r.any_y.mean()),
                   intensity_r2_log10=float(ss), combined_r2=float(r2c)),
              open(OUT / f"narrow_two_part_validation_{spec}.json", "w"), indent=1)
    return r


def part2_contrasts(spec="B", n_boot=200, seed=11):
    covs = [v for k, v in SPECS.items() if k.startswith(spec)][0]
    d = city_frame()
    base = dict(ls=d.ls.median(), lset=d.lset.median(), ladd=d.ladd.median(),
                biome=d.biome.mode().iloc[0])

    def curve(o, i2, sm, var):
        lo, hi = d[var].quantile(.01), d[var].quantile(.99)
        g = pd.DataFrame([dict(base, **{var: lo}), dict(base, **{var: hi})])
        p = np.asarray(o.predict(g), float); mu = np.asarray(i2.predict(g), float)
        return p, 10.0 ** mu * sm, combined_expectation(p, mu, sm)

    o, i2, sm = fit_two_part(d, covs)
    point = {v: curve(o, i2, sm, v) for v in ("ls", "lset")}
    rng = np.random.default_rng(seed)
    cs = d.country.dropna().unique()
    boot = {v: [] for v in point}
    for _ in range(n_boot):
        pick = rng.choice(cs, size=len(cs), replace=True)
        b = pd.concat([d[d.country == c] for c in pick], ignore_index=True)
        if b.any_y.sum() < 50:
            continue
        try:
            o2, i3, s2 = fit_two_part(b, covs)
        except Exception:
            continue
        for v in point:
            boot[v].append([x[1] / x[0] for x in curve(o2, i3, s2, v)])
    print(f"\n=== adjusted 1st-to-99th-percentile contrasts, spec {spec} ===")
    print(f"{'covariate':>10} {'P(exposure)':>22} {'intensity | positive':>24} "
          f"{'combined':>24}")
    rec = {}
    for v, lab in (("ls", "year 2000 size"), ("lset", "setting richness")):
        a = np.array(boot[v])
        names = ("p", "intensity_pos", "combined")
        cells, store = [], {}
        for j, nm in enumerate(names):
            r = point[v][j][1] / point[v][j][0]
            lo, hi = np.percentile(a[:, j], [2.5, 97.5])
            cells.append(f"{r:>7.2f}x [{lo:.2f}, {hi:.2f}]")
            store[nm] = [float(r), float(lo), float(hi)]
        rec[lab] = store
        print(f"{lab:>10} {cells[0]:>22} {cells[1]:>24} {cells[2]:>24}")
    print(f"  (fold change between the 1st and 99th percentile of the covariate, "
          f"{len(np.array(boot['ls']))} bootstrap replicates)")
    json.dump(rec, open(OUT / f"narrow_two_part_contrasts_{spec}.json", "w"), indent=1)
    return rec
