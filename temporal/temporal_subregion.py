"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import tropics_temporal as t
import temporal_analysis as ta

OUT = ta.OUT
NP = 4


def subregion_of(iso):
    import pyogrio
    import tropics_robinson as rb
    ne = pyogrio.read_dataframe(rb.NE_PATH, columns=["ADM0_A3", "SUBREGION"], read_geometry=False)
    m = dict(zip(ne.ADM0_A3, ne.SUBREGION))
    return pd.Series(iso).map(m).to_numpy()


def accumulate(sp, sub, V, rg, grp):
    df = pd.DataFrame({"sp": sp, "sub": sub})
    for j in range(NP):
        df[f"v{j}"] = V[:, j]
    g = df.groupby(["sub", "sp"], sort=True)[[f"v{j}" for j in range(NP)]].sum()
    v = g.to_numpy()
    s_sub = g.index.get_level_values(0).to_numpy()
    s_sp = g.index.get_level_values(1).to_numpy()
    hit = v > 0
    npr = hit.sum(1)
    cum = np.cumsum(v, 1)
    frac = 100 * cum / rg[s_sp][:, None]
    rep = []
    cml = []
    for name, gmask in grp.items():
        sel = gmask[s_sp] & (npr > 0)
        for u in np.unique(s_sub):
            k = sel & (s_sub == u)
            rep.append(dict(sub=u, group=name, reached=int(k.sum()),
                            **{f"n{p}": int((k & (npr == p)).sum()) for p in range(1, NP + 1)},
                            first=int((k & hit[:, 0]).sum())))
            for j in range(NP):
                kj = k & (cum[:, j] > 0)
                cml.append(dict(sub=u, group=name, j=j, reached=int(kj.sum()),
                                overlap_km2=float(cum[k, j].sum()),
                                range_equiv=float((cum[k, j] / rg[s_sp[k]]).sum()),
                                med_pct=float(np.median(frac[k, j])) if k.any() else np.nan,
                                ge1=int((frac[k, j] >= 1).sum()), ge01=int((frac[k, j] >= 0.1).sum())))
    return pd.DataFrame(rep), pd.DataFrame(cml)


def test():
    sp = np.array([0, 0, 0, 1, 1, 2, 2])
    sub = np.array(["A", "A", "B", "A", "A", "B", "B"])
    V = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 2, 0],
                  [0, 2, 0, 0], [0, 0, 0, 2], [5, 5, 5, 5], [0, 0, 0, 0]], float)
    rg = np.array([100.0, 10.0, 50.0])
    grp = {"all": np.array([True, True, True])}
    rep, cml = accumulate(sp, sub, V, rg, grp)
    r = rep.set_index("sub")
    assert r.loc["A", "reached"] == 2 and r.loc["A", "n4"] == 1 and r.loc["A", "n2"] == 1
    assert r.loc["A", "first"] == 1
    assert r.loc["B", "reached"] == 2 and r.loc["B", "n1"] == 1 and r.loc["B", "n4"] == 1
    c = cml.set_index(["sub", "j"])
    assert c.loc[("A", 0), "overlap_km2"] == 2 and c.loc[("A", 3), "overlap_km2"] == 9
    assert np.isclose(c.loc[("A", 3), "range_equiv"], 5 / 100 + 4 / 10)
    assert c.loc[("A", 0), "reached"] == 1 and c.loc[("A", 1), "reached"] == 2
    assert c.loc[("B", 3), "overlap_km2"] == 22 and c.loc[("B", 1), "reached"] == 1
    assert np.isclose(c.loc[("A", 3), "med_pct"], np.median([5.0, 40.0]))
    assert c.loc[("A", 3), "ge1"] == 2 and c.loc[("B", 0), "ge1"] == 1
    print("synthetic checks passed")


def run():
    th, na, rg = ta.flags()
    import tropics_numbers as tn
    d = tn.table().set_index("city_id")
    a, b, V = t.load_pairs()
    keep = th[a] | na[a]
    a, b, V = a[keep], b[keep], V[keep, :NP]
    city_sub = pd.Series(subregion_of(d.country.to_numpy()), index=d.index)
    fra = d.country.isin(["FRA", "NLD"]).to_numpy()
    city_sub[fra] = np.where(d.lon[fra] > 0, "Eastern Africa", np.where(d.lat[fra] > 10, "Caribbean", "South America"))
    print(city_sub.value_counts().to_string())
    sub = city_sub.loc[b].to_numpy()
    ok = pd.notna(sub)
    print("pairs", int(keep.sum()), "unmapped", int((~ok).sum()))
    grp = {"Threatened": th, "Narrow-ranged": na}
    rep, cml = accumulate(a[ok], sub[ok], V[ok], rg, grp)
    rep_all, cml_all = accumulate(a[ok], np.full(ok.sum(), "All tropics", object), V[ok], rg, grp)
    rep = pd.concat([rep, rep_all], ignore_index=True)
    cml = pd.concat([cml, cml_all], ignore_index=True)
    rep.to_csv(OUT / "temporal_subregion_repeat.csv", index=False)
    cml.to_csv(OUT / "temporal_subregion_cumulative.csv", index=False)
    print(rep.to_string())


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
