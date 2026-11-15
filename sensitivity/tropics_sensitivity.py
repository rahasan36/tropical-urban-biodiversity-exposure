"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import shapely, json, time
from shapely import STRtree

from tropics_config import (INTERIM_DIR, TABLE_DIR, DELIN_DEG, MOSAIC_LAT_TOP,
                            MOSAIC_LAT_BOTTOM, THREAT_CATS, NARROW_RANGE_PCTL,
                            SETTING_BUFFER_KM, TROPIC_LAT,
                            EXCLUDED_TAXA)
from tropics_grid import cell_area_km2
from tropics_urban import load_counts, urban_mask, MOSAIC_ROWS, MOSAIC_COLS
from tropics_cities import delineate, tropical_band_mask
import tropics_rangecache as rc
import tropics_scaling as ts

MASTER = INTERIM_DIR / "city_measurements_f10.npz"
MASTER_FRACTION = 0.10


def master_cells():
    z = np.load(MASTER, allow_pickle=False)
    return pd.DataFrame(dict(row=z["cell_row"].astype(np.int64),
                             col=z["cell_col"].astype(np.int64),
                             conv=z["cell_conv"], base=z["cell_base"],
                             final=z["cell_final"]))


def config_cities(fraction, link_cells, min_km2, cells=None):
    cells = master_cells() if cells is None else cells
    counts = load_counts(which=("count_final",))
    urban = urban_mask(counts, fraction) & tropical_band_mask()[:, None]
    row_area = cell_area_km2(MOSAIC_LAT_TOP, MOSAIC_ROWS, DELIN_DEG)
    rr, cc, ids, sizes = delineate(urban, link_cells=link_cells,
                                   min_km2=min_km2, row_area=row_area)
    del urban, counts
    d = pd.DataFrame(dict(row=rr.astype(np.int64), col=cc.astype(np.int64),
                          city_id=ids.astype(np.int64)))
    d = d.merge(cells, on=["row", "col"], how="left")
    d[["conv", "base", "final"]] = d[["conv", "base", "final"]].fillna(0.0)
    d["lat"] = MOSAIC_LAT_TOP - (d.row + 0.5) * DELIN_DEG
    d["lon"] = -180.0 + (d.col + 0.5) * DELIN_DEG
    cities = (d.groupby("city_id")
                .agg(lon=("lon", "mean"), lat=("lat", "mean"),
                     n_cells=("lon", "size"), urban_km2_base=("base", "sum"),
                     urban_km2_final=("final", "sum"),
                     urban_added_km2=("conv", "sum")).reset_index())
    return d, cities, sizes


def cost_from_cache(cells, cities, sp=None, part_sp=None,
                    buffer_km=SETTING_BUFFER_KM, taxa=None,
                    setting_on="footprint", with_pool=False):
    if sp is None:
        sp, part_sp = rc.load_index()
    keep_sp = (np.ones(len(sp), bool) if taxa is None
               else sp.taxon.isin(taxa).to_numpy())
    cell_pts = shapely.points(cells.lon.to_numpy(), cells.lat.to_numpy())
    tree_cell = STRtree(cell_pts)
    cell_city = cells.city_id.to_numpy()
    cell_conv = cells.conv.to_numpy()
    city_ids = cities.city_id.to_numpy()
    n_city = int(city_ids.max())
    buf = buffer_km / 111.32
    on_foot = setting_on == "footprint"
    if on_foot:
        base_flag = cells.base.to_numpy() > 0
        tree_set, set_owner = tree_cell, None
    else:
        base_flag = None
        tree_set = STRtree(shapely.points(cities.lon.to_numpy(),
                                          cities.lat.to_numpy()))
        set_owner = city_ids

    tree_pt = STRtree(shapely.points(cities.lon.to_numpy(), cities.lat.to_numpy()))
    pool = set()
    hits = {}
    near_hits = {}
    for sp_batch, geom_batch in rc.iter_parts(part_sp):
        for i, g in zip(sp_batch, geom_batch):
            i = int(i)
            if not keep_sp[i] or g is None or g.is_empty:
                continue
            idx = tree_cell.query(g, predicate="intersects")
            if idx.size:
                hits.setdefault(i, []).append(idx.astype(np.int32))
            if with_pool and i not in pool and tree_pt.query(
                    g, predicate="dwithin", distance=buf).size:
                pool.add(i)
            near = tree_set.query(g, predicate="dwithin", distance=buf)
            if near.size:
                if on_foot:
                    near = near[base_flag[near]]
                    if not near.size:
                        continue
                    near = cell_city[near]
                else:
                    near = set_owner[near]
                near_hits.setdefault(i, []).append(
                    np.unique(near).astype(np.int32))

    pair_sp, pair_city, pair_conv = [], [], []
    for i, parts in hits.items():
        u = np.unique(np.concatenate(parts))
        agg = np.bincount(cell_city[u], weights=cell_conv[u], minlength=n_city + 1)
        nz = np.flatnonzero(agg)
        if nz.size:
            pair_sp.append(np.full(nz.size, i, np.int32))
            pair_city.append(nz.astype(np.int32))
            pair_conv.append(agg[nz])
    set_sp, set_city = [], []
    for i, parts in near_hits.items():
        u = np.unique(np.concatenate(parts))
        set_sp.append(np.full(u.size, i, np.int32))
        set_city.append(u.astype(np.int32))

    def cat(parts, dt):
        return np.concatenate(parts) if parts else np.zeros(0, dt)
    pairs = pd.DataFrame(dict(sp_i=cat(pair_sp, np.int32),
                              city_id=cat(pair_city, np.int32),
                              conv_km2=cat(pair_conv, np.float64)))
    setting = pd.DataFrame(dict(sp_i=cat(set_sp, np.int32),
                                city_id=cat(set_city, np.int32))).drop_duplicates()
    if with_pool:
        pool = np.asarray(sorted(pool | set(pairs.sp_i.unique())), np.int32)
        return sp, pairs, setting, pool
    return sp, pairs, setting


def assemble(cities, sp, pairs, setting, threshold=None, pool=None):
    import tropics_assemble as ta
    s = ta.merged_categories(sp.copy())
    keep = ~s["taxon"].astype(str).str.upper().isin(
        [t.upper() for t in EXCLUDED_TAXA])
    s["is_threat"] = s.category.isin(THREAT_CATS) & keep
    if pool is None:
        pool = set(pairs.sp_i.unique()) | set(setting.sp_i.unique())
    inpool = s.sp_i.isin(set(int(v) for v in pool)) & (s.range_km2 > 0) & keep
    own = float(np.percentile(s.loc[inpool, "range_km2"], NARROW_RANGE_PCTL))
    thr = own if threshold is None else float(threshold)
    s["is_narrow"] = inpool & (s.range_km2 <= thr)
    p = pairs.merge(s[["sp_i", "is_threat", "is_narrow", "range_km2"]], on="sp_i")
    p = p[(p.is_threat | p.is_narrow) & (p.conv_km2 > 0)]
    t = cities.set_index("city_id")
    t["cost_km2"] = p.groupby("city_id").conv_km2.sum()
    t["rarity_converted"] = (p.conv_km2 / p.range_km2.clip(lower=1e-6)) \
        .groupby(p.city_id).sum()
    ss = setting.merge(s[["sp_i", "is_threat"]], on="sp_i")
    t["setting_richness"] = ss[ss.is_threat].groupby("city_id").sp_i.nunique()
    t = t.fillna({"cost_km2": 0.0, "rarity_converted": 0.0, "setting_richness": 0.0})
    t["cost_per_km2"] = np.where(t.urban_added_km2 > 0,
                                 t.cost_km2 / t.urban_added_km2, np.nan)
    t["rarity_per_km2"] = np.where(t.urban_added_km2 > 0,
                                   t.rarity_converted / t.urban_added_km2, np.nan)
    return t.reset_index(), thr, own


def attach_groups(t):
    base = pd.read_csv(TABLE_DIR / "cities_cost.csv")[
        ["lon", "lat", "country", "biome", "realm", "continent"]]
    from scipy.spatial import cKDTree
    tree = cKDTree(base[["lon", "lat"]].to_numpy())
    _, j = tree.query(t[["lon", "lat"]].to_numpy(), k=1)
    for c in ("country", "biome", "realm", "continent"):
        t[c] = base[c].to_numpy()[j]
    return t


CONFIGS = [
    dict(label="headline (fraction 0.20, link 2, floor 5)", fraction=0.20, link=2, floor=5.0),
    dict(label="built-up fraction 0.10", fraction=0.10, link=2, floor=5.0),
    dict(label="built-up fraction 0.30", fraction=0.30, link=2, floor=5.0),
    dict(label="linking distance 1 cell", fraction=0.20, link=1, floor=5.0),
    dict(label="linking distance 4 cells", fraction=0.20, link=4, floor=5.0),
    dict(label="size floor 15 km2", fraction=0.20, link=2, floor=15.0),
]


def run(configs=None, out_name="sensitivity_delineation.csv", taxa=None):
    configs = configs or CONFIGS
    sp, part_sp = rc.load_index()
    cells0 = master_cells()
    rows = []
    for cfg in configs:
        t0 = time.time()
        cells, cities, sizes = config_cities(cfg["fraction"], cfg["link"],
                                             cfg["floor"], cells=cells0)
        sp2, pairs, setting = cost_from_cache(cells, cities, sp, part_sp,
                                              taxa=taxa)
        t, thr, _own = assemble(cities, sp2, pairs, setting)
        t = attach_groups(t)
        r = ts.fit_scaling(t, response="cost_per_km2", size="urban_km2_base",
                           setting="setting_richness", spatial=True)
        rows.append(dict(label=cfg["label"], taxa=("all" if taxa is None
                                                   else "+".join(taxa)), fraction=cfg["fraction"],
                         link=cfg["link"], floor=cfg["floor"],
                         n_cities=int(len(t)), n_pairs=int(len(pairs)),
                         urban_added_km2=float(t.urban_added_km2.sum()),
                         cost_km2=float(t.cost_km2.sum()),
                         narrow_threshold_km2=thr, beta=r["beta"],
                         ci_lower=r["ci_lower"], ci_upper=r["ci_upper"],
                         gamma_setting=r.get("gamma_setting", np.nan),
                         R2=r["R2"], verdict=r["verdict"],
                         seconds=round(time.time() - t0, 1)))
        print(json.dumps(rows[-1], default=str), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / out_name, index=False)
    return out


if __name__ == "__main__":
    import sys
    sel = None
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        sel = [CONFIGS[int(i)] for i in sys.argv[1].split(",")]
    taxa = sys.argv[2].split("+") if len(sys.argv) > 2 else None
    tag = "" if taxa is None else "_" + "".join(t[:4].lower() for t in taxa)
    out = run(sel, taxa=taxa,
              out_name=("sensitivity_delineation%s.csv" % tag if sel is None
                        else f"sensitivity_part_{sys.argv[1].replace(',', '_')}{tag}.csv"))
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
