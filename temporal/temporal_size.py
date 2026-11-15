"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import tropics_temporal as t
from tropics_config import TABLE_DIR

OUT = TABLE_DIR / "temporal"
LAB = ["2000–2005", "2005–2010", "2010–2015", "2015–2020"]
DCLS = [5, 25, 100, 500, np.inf]
DLAB = ["5–25 km²", "25–100 km²", "100–500 km²", "≥500 km²"]
ICLS = [0, 1, 5, 25, 100, np.inf]
ILAB = ["Under 1 km²", "1–5 km²", "5–25 km²", "25–100 km²", "≥100 km²"]


def year_agg(y):
    with np.load(t.DELIN_DIR / f"agglomerations_{y}.npz") as q:
        return {k: q[k] for k in ("rows", "cols", "ids", "area", "link2020")}


def pareto_gi(a, xmin):
    s = np.sort(a[a >= xmin])[::-1]
    r = np.arange(1, s.size + 1) - 0.5
    b = np.polyfit(np.log(s), np.log(r), 1)[0]
    alpha = -b
    return float(alpha), float(alpha * np.sqrt(2.0 / s.size)), int(s.size)


def distribution():
    rows = []
    for y in t.YEARS:
        a = year_agg(y)["area"]
        cls = pd.cut(a, DCLS, right=False, labels=DLAB)
        r = dict(year=y, n=int(a.size), total_km2=float(a.sum()),
                 median_km2=float(np.median(a)), p90_km2=float(np.percentile(a, 90)),
                 largest_km2=float(a.max()),
                 top10_share=float(np.sort(a)[::-1][:10].sum() / a.sum()),
                 top1pct_share=float(np.sort(a)[::-1][:max(1, a.size // 100)].sum() / a.sum()))
        for lab in DLAB:
            m = cls == lab
            r[f"n {lab}"] = int(m.sum()); r[f"area {lab}"] = float(a[m].sum())
        for xmin in (25, 50, 100):
            al, se, n = pareto_gi(a, xmin)
            r[f"zipf_alpha_{xmin}"] = al; r[f"zipf_se_{xmin}"] = se; r[f"zipf_n_{xmin}"] = n
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "temporal_size_distribution.csv", index=False)
    return d


def transitions():
    rows, mats = [], {}
    for y0, y1 in t.PERIODS:
        A, B = year_agg(y0), year_agg(y1)
        link, miss, split = t.link_to_final(A["rows"], A["cols"], A["ids"],
                                            B["rows"], B["cols"], B["ids"])
        tgt = link[1:]
        n1 = B["area"].size
        pieces = np.bincount(tgt, minlength=n1 + 1)[1:]
        src_area = np.bincount(tgt, weights=A["area"], minlength=n1 + 1)[1:]
        rows.append(dict(period=f"{y0}–{y1}", start=int(A["area"].size), end=int(n1),
                         new=int((pieces == 0).sum()), continued=int((pieces == 1).sum()),
                         merged=int((pieces > 1).sum()),
                         pieces_absorbed=int(pieces[pieces > 1].sum()),
                         cells_unlinked=miss, split=split,
                         new_area_km2=float(B["area"][pieces == 0].sum()),
                         end_area_km2=float(B["area"].sum())))
        c0 = pd.cut(A["area"], DCLS, right=False, labels=DLAB).astype(str)
        c1 = pd.cut(B["area"][tgt - 1], DCLS, right=False, labels=DLAB).astype(str)
        mats[f"{y0}–{y1}"] = pd.crosstab(pd.Series(c0, name="start"), pd.Series(c1, name="end"))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "temporal_transitions.csv", index=False)
    m = pd.concat(mats, names=["period"])
    m.to_csv(OUT / "temporal_class_transitions.csv")
    return d, m


def growth_by_size():
    import temporal_analysis as ta
    cp = ta.load_cp()
    cp = cp[cp.j < 4].copy()
    cp["cls"] = pd.cut(cp.start_km2, ICLS, right=False, labels=ILAB).astype(str)
    g = cp.groupby(["period", "cls"])
    s = pd.DataFrame(dict(cities=g.size(), start_km2=g.start_km2.sum(), added_km2=g.added_km2.sum(),
                          conv=g.conv_concern.sum(), conv_threat=g.conv_threat.sum(),
                          conv_narrow=g.conv_narrow.sum(), rarity=g.rarity.sum())).reset_index()
    for c in ("added_km2", "conv", "conv_threat", "conv_narrow", "rarity", "start_km2"):
        s[f"share_{c}"] = s[c] / s.groupby("period")[c].transform("sum")
    s["growth_pct"] = 100 * s.added_km2 / s.start_km2
    s["exposure_ratio"] = s.share_conv / s.share_added_km2
    s["rarity_ratio"] = s.share_rarity / s.share_added_km2
    s["narrow_ratio"] = s.share_conv_narrow / s.share_added_km2
    s.to_csv(OUT / "temporal_size_classes.csv", index=False)

    gib = []
    for j, p in enumerate(LAB):
        x = cp[(cp.j == j) & (cp.added_km2 > 0)].copy()
        x["lg"] = np.log10(x.added_km2 / x.start_km2)
        x["ls"] = np.log10(x.start_km2)
        f = smf.ols("lg ~ ls", x).fit(cov_type="cluster", cov_kwds={"groups": x.country.astype(str)})
        top = x.sort_values("start_km2", ascending=False)
        gib.append(dict(period=p, n=int(len(x)), gibrat_slope=float(f.params.ls),
                        ci_lower=float(f.conf_int().loc["ls", 0]),
                        ci_upper=float(f.conf_int().loc["ls", 1]),
                        top10_share_added=float(top.added_km2.iloc[:10].sum() / x.added_km2.sum()),
                        top1pct_share_added=float(top.added_km2.iloc[:len(x) // 100].sum() / x.added_km2.sum()),
                        top10_share_start=float(top.start_km2.iloc[:10].sum() / x.start_km2.sum())))
    gib = pd.DataFrame(gib)
    gib.to_csv(OUT / "temporal_gibrat.csv", index=False)
    return s, gib


def _selftest():
    rng = np.random.default_rng(3)
    a = (rng.pareto(1.0, 200000) + 1) * 5
    al, se, n = pareto_gi(a, 50)
    assert abs(al - 1.0) < 0.05, al
    print("pareto_gi ok", round(al, 3))


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)
    a = sys.argv[1:]
    if a and a[0] == "test":
        _selftest()
    else:
        _selftest()
        print(distribution().round(3).T.to_string())
        d, m = transitions(); print(d.to_string()); print(m.to_string())
        s, gib = growth_by_size()
        print(s.round(3).to_string()); print(gib.round(4).to_string())
