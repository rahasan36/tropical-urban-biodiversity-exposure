"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import tropics_temporal as t
import temporal_analysis as ta
import temporal_subregion as tsr

OUT = ta.OUT
NP = 4
BIG_KM2 = 100.0


def species_tables():
    th, na, _rg = ta.flags()
    a, _b, V = t.load_pairs()
    V = V[:, :NP]
    S = np.zeros((th.size, NP))
    for j in range(NP):
        S[:, j] = np.bincount(a, weights=V[:, j], minlength=th.size)
    rows, q = [], []
    for grp, m in (("threat", th), ("narrow", na)):
        k = m & (S.sum(1) > 0)
        hit = S[k] > 0
        rows.append(pd.DataFrame(dict(sp=np.flatnonzero(k), group=grp,
                                      n_periods=hit.sum(1), first=hit.argmax(1))))
        c = 100 * np.cumsum(S[k], 1) / S[k].sum(1)[:, None]
        for j in range(NP):
            q.append(dict(group=grp, j=j, **{f"q{p}": float(np.percentile(c[:, j], p))
                                             for p in (10, 25, 50, 75, 90)}))
    periods = pd.concat(rows, ignore_index=True)
    cumulative = pd.DataFrame(q)
    periods.to_csv(OUT / "temporal_species_periods.csv", index=False)
    cumulative.to_csv(OUT / "temporal_species_cumulative.csv", index=False)
    return periods, cumulative


def subregion_table():
    import tropics_numbers as tn
    cp = pd.read_csv(OUT / "temporal_city_periods.csv.gz")
    d = tn.table().set_index("city_id")
    sub = pd.Series(tsr.subregion_of(d.country.to_numpy()), index=d.index)
    cp["sub"] = sub.reindex(cp.city_id).to_numpy()
    x = cp[cp.j < NP]
    g = x.groupby(["sub", "j"])[["added_km2", "conv_concern", "rarity"]].sum()
    add = g.added_km2.unstack()
    inten = (g.conv_concern / g.added_km2).unstack()
    rint = (g.rarity / g.added_km2).unstack()
    big = (x[x.start_delin_km2 >= BIG_KM2].groupby(["sub", "j"]).added_km2.sum()
           .unstack().reindex(index=add.index, columns=range(NP)).fillna(0) / add)
    o = pd.DataFrame(dict(added0=add[0], added3=add[NP - 1],
                          d_int=100 * (inten[NP - 1] / inten[0] - 1),
                          d_rint=100 * (rint[NP - 1] / rint[0] - 1),
                          d_big=100 * (big[NP - 1] - big[0]), int0=inten[0],
                          n=x[x.j == 0].groupby("sub").size()))
    o = o.sort_values("added0", ascending=False)
    o.index.name = "sub"
    o.to_csv(OUT / "temporal_subregions.csv")
    return o


if __name__ == "__main__":
    species_tables()
    print(subregion_table().round(2).to_string())
