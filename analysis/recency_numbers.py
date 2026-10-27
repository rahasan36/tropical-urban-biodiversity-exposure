"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import fig_profile_map as pm
import fig_trivariate as tv
import tropics_panels as tp
from tropics_config import TABLE_DIR, MOSAIC_LAT_TOP, DELIN_DEG
from tropics_measure import MEASURE_PATH


def band():
    lon, lat, P = tv.lattice_parts()
    T = P.sum(0)
    z = np.load(MEASURE_PATH, allow_pickle=False)
    key_l = (np.round((MOSAIC_LAT_TOP - lat) / DELIN_DEG - 0.5).astype(np.int64) * 10 ** 6
             + np.round((lon + 180) / DELIN_DEG - 0.5).astype(np.int64))
    key_m = z["cell_row"].astype(np.int64) * 10 ** 6 + z["cell_col"].astype(np.int64)
    inside = np.isin(key_l, key_m)
    windows = {}
    for nm, lo0, lo1, la0, la1 in tv.WIN4:
        m = (lon >= lo0) & (lon < lo1) & (lat >= la0) & (lat < la1)
        s = P[m].sum(0)
        windows[nm] = round(100 * s[1:].sum() / s.sum(), 2)
    return {
        "impervious_2000_km2": round(float(T[0])),
        "impervious_2020_km2": round(float(T.sum())),
        "added_2000_2020_km2": round(float(T[1:].sum())),
        "split_by_2000_2000_2010_2010_2020_pct": [round(float(v), 1) for v in 100 * T / T.sum()],
        "added_inside_agglomerations_pct": round(float(100 * P[inside, 1:].sum() / T[1:].sum()), 1),
        "added_after_2000_by_window_pct": windows,
    }


def agglomerations():
    d = pm.city_table()
    top = tp.city_names(d.nlargest(25, "tot"))
    top = top.assign(present_2000_pct=(100 * top.s0 / top.tot).round(1))
    c = d.groupby("country")[["s0", "tot"]].sum().nlargest(25, "tot")
    c = c.assign(present_2000_pct=(100 * c.s0 / c.tot).round(1))
    return {
        "largest_25_present_2000_pct": dict(zip(top["name"].fillna("Unnamed"), top.present_2000_pct)),
        "countries_25_present_2000_pct": c.present_2000_pct.to_dict(),
    }


def recency():
    d = pm.city_table()
    d = d[d.city_id.isin(set(tp.cities(sample=True).city_id))].copy()
    d["recent"] = 1 - d.s0 / d.tot
    t = pd.qcut(d.recent, 3, labels=False)
    g = d.groupby("country").agg(n=("city_id", "size"), s0=("s0", "sum"), tot=("tot", "sum"),
                                 added=("urban_added_km2", "sum"), rar=("rarity_converted", "sum"))
    g = g[g.n >= 10]
    g["recent"] = 1 - g.s0 / g.tot
    rho = stats.spearmanr(g.recent, g.rar / g.added)
    return {
        "n": int(len(d)),
        "tercile_median_exposure": d.groupby(t).cost_per_km2.median().round(2).tolist(),
        "tercile_median_rarity_1e5": (1e5 * d.groupby(t).rarity_per_km2.median()).round(2).tolist(),
        "countries": int(len(g)),
        "country_spearman_rho": round(float(rho.statistic), 3),
        "country_spearman_p": float(rho.pvalue),
    }


def main():
    out = {"band": band(), "agglomerations": agglomerations(), "recency": recency()}
    (TABLE_DIR / "recency_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
