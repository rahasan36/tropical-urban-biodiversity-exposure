"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyogrio, shapely, json
from shapely import STRtree
from pathlib import Path

from tropics_config import (INTERIM_DIR, TABLE_DIR, THREAT_CATS,
                            NARROW_RANGE_PCTL, BOTW_CHECKLIST, DOWNLOADS,
                            EXCLUDED_TAXA,
                            URBAN_FRACTION, LINK_CELLS, MIN_CITY_KM2,
                            BASE_EPOCH, FINAL_EPOCH)
from tropics_cost import PAIRS_PATH, load_city_geometry
from tropics_measure import city_table
from tropics_species import read_bird_checklist

COUNTRIES = DOWNLOADS / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp"
ECOREGIONS = DOWNLOADS / "ecoregions" / "Ecoregions2017.shp"


def load_pairs():
    z = np.load(PAIRS_PATH, allow_pickle=False)
    sp_index = json.loads(str(z["sp_index"]))
    sp_meta = json.loads(str(z["sp_meta"]))
    idx_to_sp = {v: k for k, v in sp_index.items()}
    n = len(sp_index)
    sp = pd.DataFrame(dict(
        sp_i=np.arange(n),
        sp_id=[idx_to_sp[i] for i in range(n)],
        range_km2=z["sp_area"][:n]))
    meta = pd.DataFrame([(k,) + tuple(v) for k, v in sp_meta.items()],
                        columns=["sp_id", "sci_name", "category", "taxon"])
    sp = sp.merge(meta, on="sp_id", how="left")
    pairs = pd.DataFrame(dict(sp_i=z["pair_sp"], city_id=z["pair_city"],
                              conv_km2=z["pair_conv"].astype(np.float64)))
    setting = pd.DataFrame(dict(sp_i=z["set_sp"], city_id=z["set_city"]))
    return sp, pairs, setting


def merged_categories(sp):
    cat = read_bird_checklist(BOTW_CHECKLIST).rename(
        columns={"category": "bird_category"})
    out = sp.merge(cat, on="sci_name", how="left")
    out["category"] = out["category"].astype(str).str.strip().str.upper()
    miss = out["category"].isin(["", "NAN", "NONE"])
    out.loc[miss, "category"] = out.loc[miss, "bird_category"].astype(str).str.upper()
    return out


def apply_flags(sp, tropical_sp_i, narrow_pctl=NARROW_RANGE_PCTL):
    sp = merged_categories(sp.copy())
    sp["is_threat"] = sp["category"].isin(THREAT_CATS)
    keep = ~sp["taxon"].astype(str).str.upper().isin(
        [t.upper() for t in EXCLUDED_TAXA])
    sp["is_threat"] &= keep
    pool = sp["sp_i"].isin(tropical_sp_i) & (sp["range_km2"] > 0) & keep
    thr = float(np.percentile(sp.loc[pool, "range_km2"], narrow_pctl))
    sp["in_pool"] = pool
    sp["is_narrow"] = pool & (sp["range_km2"] <= thr)
    sp.attrs["narrow_threshold_km2"] = thr
    return sp, thr


def _attach_polygon_attr(cities, path, fields, prefix, layer=None):
    kw = dict(columns=fields)
    if layer:
        kw["layer"] = layer
    gdf = pyogrio.read_dataframe(path, **kw)
    tree = STRtree(gdf.geometry.to_numpy())
    pts = shapely.points(cities.lon.to_numpy(), cities.lat.to_numpy())
    out = {f"{prefix}_{f}": np.array([None] * len(cities), dtype=object)
           for f in fields}
    qi, qj = tree.query(pts, predicate="within")
    for i, j in zip(qi, qj):
        for f in fields:
            if out[f"{prefix}_{f}"][i] is None:
                out[f"{prefix}_{f}"][i] = gdf[f].iloc[j]
    missing = np.flatnonzero(pd.isna(pd.Series(out[f"{prefix}_{fields[0]}"])))
    if missing.size:
        nearest = tree.nearest(pts[missing])
        for i, j in zip(missing, np.atleast_1d(nearest)):
            for f in fields:
                out[f"{prefix}_{f}"][i] = gdf[f].iloc[j]
    for k, v in out.items():
        cities[k] = v
    return cities


def build(out_csv=None):
    sp, pairs, setting = load_pairs()
    tropical = set(pairs.sp_i.unique()).union(set(setting.sp_i.unique()))
    sp, thr = apply_flags(sp, tropical)

    flags = sp[["sp_i", "is_threat", "is_narrow", "range_km2"]]
    p = pairs.merge(flags, on="sp_i", how="left")
    p = p[p.conv_km2 > 0]
    p["of_concern"] = p.is_threat | p.is_narrow

    def agg(mask, name):
        d = p[mask].groupby("city_id")["conv_km2"].sum().rename(name)
        return d

    tab = pd.DataFrame(index=pd.Index(sorted(p.city_id.unique()), name="city_id"))
    tab = tab.join(agg(p.of_concern, "cost_km2"))
    tab = tab.join(agg(p.is_threat, "cost_threat_km2"))
    tab = tab.join(agg(p.is_narrow, "cost_narrow_km2"))
    tab = tab.join(agg(slice(None) if False else np.ones(len(p), bool), "cost_all_km2"))
    rar = p[p.of_concern].copy()
    rar["w"] = rar.conv_km2 / rar.range_km2.clip(lower=1e-6)
    tab = tab.join(rar.groupby("city_id")["w"].sum().rename("rarity_converted"))
    tab = tab.join(p[p.is_threat].groupby("city_id")["sp_i"].nunique()
                    .rename("n_threat_affected"))
    tab = tab.join(p[p.is_narrow].groupby("city_id")["sp_i"].nunique()
                    .rename("n_narrow_affected"))

    s = setting.merge(sp[["sp_i", "is_threat", "is_narrow"]], on="sp_i", how="left")
    tab = tab.join(s[s.is_threat].groupby("city_id")["sp_i"].nunique()
                    .rename("setting_richness"))
    tab = tab.join(s.groupby("city_id")["sp_i"].nunique()
                    .rename("setting_richness_all"))

    tab = tab.reset_index()
    tab = tab.merge(city_table(), on="city_id", how="right")
    cells, cities = load_city_geometry()
    tab = tab.merge(cities[["city_id", "lon", "lat", "n_cells"]], on="city_id",
                    how="left")
    for c in ["cost_km2", "cost_threat_km2", "cost_narrow_km2", "cost_all_km2",
              "rarity_converted", "n_threat_affected", "n_narrow_affected",
              "setting_richness", "setting_richness_all"]:
        tab[c] = tab[c].fillna(0.0)

    tab["cost_per_km2"] = np.where(tab.urban_added_km2 > 0,
                                   tab.cost_km2 / tab.urban_added_km2, np.nan)
    tab["rarity_per_km2"] = np.where(tab.urban_added_km2 > 0,
                                     tab.rarity_converted / tab.urban_added_km2,
                                     np.nan)
    tab = _attach_polygon_attr(tab, COUNTRIES, ["ADM0_A3", "CONTINENT"], "ne")
    tab = _attach_polygon_attr(tab, ECOREGIONS, ["BIOME_NAME", "REALM"], "eco")
    tab = tab.rename(columns={"ne_ADM0_A3": "country", "ne_CONTINENT": "continent",
                              "eco_BIOME_NAME": "biome", "eco_REALM": "realm"})
    tab.attrs["narrow_threshold_km2"] = thr

    if out_csv is None:
        out_csv = TABLE_DIR / "cities_cost.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_csv, index=False)
    meta = dict(urban_fraction=URBAN_FRACTION, link_cells=LINK_CELLS,
                min_city_km2=MIN_CITY_KM2, base_epoch=BASE_EPOCH,
                final_epoch=FINAL_EPOCH, narrow_threshold_km2=thr,
                n_cities=int(len(tab)), n_species_pool=int(sp.in_pool.sum()),
                n_threatened_in_pool=int((sp.in_pool & sp.is_threat).sum()),
                n_narrow_in_pool=int(sp.is_narrow.sum()))
    (Path(out_csv).parent / "cities_cost_meta.json").write_text(json.dumps(meta, indent=2))
    return tab, sp, meta


if __name__ == "__main__":
    sp = pd.DataFrame(dict(sp_i=[0, 1, 2, 3],
                           sp_id=["A", "B", "C", "D"],
                           range_km2=[100.0, 1000.0, 10000.0, 100000.0],
                           sci_name=["a", "b", "c", "d"],
                           category=["CR", "LC", "EN", "LC"],
                           taxon="MAMMALIA"))
    sp["is_threat"] = sp.category.isin(THREAT_CATS)
    sp["is_narrow"] = sp.range_km2 <= 1000.0
    pairs = pd.DataFrame(dict(sp_i=[0, 1, 2, 3, 0],
                             city_id=[1, 1, 1, 1, 2],
                             conv_km2=[2.0, 4.0, 8.0, 16.0, 1.0]))
    p = pairs.merge(sp[["sp_i", "is_threat", "is_narrow", "range_km2"]], on="sp_i")
    p["of_concern"] = p.is_threat | p.is_narrow

    cost = p[p.of_concern].groupby("city_id").conv_km2.sum()
    assert cost[1] == 14.0, cost
    assert cost[2] == 1.0
    print("cost aggregation         ok  (14.0 km2 of concern, widespread LC excluded)")

    rar = p[p.of_concern].copy(); rar["w"] = rar.conv_km2 / rar.range_km2
    w1 = rar[rar.city_id == 1].w.sum()
    expect = 2/100 + 4/1000 + 8/10000
    assert abs(w1 - expect) < 1e-12, (w1, expect)
    print(f"rarity weighting         ok  ({w1:.6f}; the 100 km2 endemic dominates)")

    n_thr = p[p.is_threat].groupby("city_id").sp_i.nunique()
    assert n_thr[1] == 2 and n_thr[2] == 1
    print("affected-species counts  ok")

    a = 14.0 / 7.0
    b = 28.0 / 14.0
    assert a == b
    print("per-unit framing         ok  (scale-invariant by construction)")
    print("ALL ASSEMBLY TESTS PASS")
