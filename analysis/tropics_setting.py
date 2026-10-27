"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.geometry import Point

from tropics_config import (INTERIM_DIR, TABLE_DIR, DELIN_DEG, MOSAIC_LAT_TOP,
                            SETTING_BUFFER_KM, URBAN_FRACTION, THREAT_CATS)

BUF_DEG = SETTING_BUFFER_KM / 111.32
STATE_PATH = INTERIM_DIR / "setting_footprint.npz"


def footprint_2000():
    from tropics_cities import load_city_cells
    from tropics_urban import load_counts, CELLS_PER_DELIN
    rr, cc, ids = load_city_cells()
    count_final, count_base = load_counts()
    base = count_base[rr, cc]
    keep = base >= URBAN_FRACTION * CELLS_PER_DELIN
    return rr[keep], cc[keep], ids[keep]


def city_neighbourhoods(rr, cc, ids, buf_deg=BUF_DEG, simplify=0.01):
    lat = MOSAIC_LAT_TOP - (rr.astype(np.float64) + 0.5) * DELIN_DEG
    lon = -180.0 + (cc.astype(np.float64) + 0.5) * DELIN_DEG
    order = np.argsort(ids, kind="stable")
    ids_s, lon_s, lat_s = ids[order], lon[order], lat[order]
    bounds = np.flatnonzero(np.diff(ids_s)) + 1
    starts = np.concatenate([[0], bounds])
    stops = np.concatenate([bounds, [len(ids_s)]])
    out_ids, geoms = [], []
    for a, b in zip(starts, stops):
        pts = shapely.points(lon_s[a:b], lat_s[a:b])
        g = shapely.union_all(shapely.buffer(pts, buf_deg, quad_segs=4))
        if simplify:
            g = g.simplify(simplify, preserve_topology=True)
        out_ids.append(int(ids_s[a])); geoms.append(g)
    return np.asarray(out_ids, np.int32), np.asarray(geoms, dtype=object)


def _selftest():
    buf = 0.5
    cells = [(1, 0.0), (2, 1.0), (2, 2.0), (2, 3.0), (3, 10.0)]
    ids = np.array([c for c, _ in cells], np.int32)
    lon = np.array([x for _, x in cells])
    lat = np.zeros_like(lon)
    order = np.argsort(ids, kind="stable")
    out_ids, geoms = [], []
    for c in (1, 2, 3):
        m = ids == c
        pts = shapely.points(lon[m], lat[m])
        geoms.append(shapely.union_all(shapely.buffer(pts, buf, quad_segs=8)))
        out_ids.append(c)
    tree = STRtree(np.asarray(geoms, dtype=object))
    sp = shapely.box(3.3, -0.05, 3.5, 0.05)
    hit = {out_ids[i] for i in tree.query(sp, predicate="intersects")}
    assert hit == {2}, hit
    cent = STRtree(np.asarray([Point(0, 0), Point(2, 0), Point(10, 0)], dtype=object))
    hit_c = {(1, 2, 3)[i] for i in cent.query(sp, predicate="dwithin", distance=buf)}
    assert hit_c == set(), hit_c
    print("  footprint buffer reaches the elongated city   ok")
    print("  centroid buffer misses it, as expected        ok")

    far = shapely.box(6.0, -0.05, 6.2, 0.05)
    assert {out_ids[i] for i in tree.query(far, predicate="intersects")} == set()
    print("  a species between two cities reaches neither  ok")


if __name__ == "__main__":
    _selftest()


NEIGH_PATH = INTERIM_DIR / "setting_neighbourhoods.npz"
PAIRS_PATH = INTERIM_DIR / "setting_pairs.npz"


def build_neighbourhoods():
    cid, geoms = city_neighbourhoods(*footprint_2000())
    np.savez(NEIGH_PATH, cid=cid, wkb=np.array([shapely.to_wkb(g) for g in geoms], dtype=object))
    return cid.size


def load_neighbourhoods():
    z = np.load(NEIGH_PATH, allow_pickle=True)
    return z["cid"], np.array([shapely.from_wkb(b) for b in z["wkb"]], dtype=object)


def run_pass(budget_s=150, batch=400, verbose=True):
    import time
    import tropics_rangecache as rc
    t0 = time.time()
    cid, geoms = load_neighbourhoods()
    tree = STRtree(geoms)
    sp, parts = rc.load_index()
    part_sp, off, ln = parts
    n_parts = len(part_sp)

    done = 0
    pair_sp, pair_city = [], []
    if PAIRS_PATH.exists():
        z = np.load(PAIRS_PATH, allow_pickle=False)
        done = int(z["done"])
        pair_sp, pair_city = [z["pair_sp"]], [z["pair_city"]]

    acc_sp, acc_city = [], []
    i = done
    while i < n_parts and time.time() - t0 < budget_s:
        j = min(i + batch, n_parts)
        sel = (part_sp[i:j], off[i:j], ln[i:j])
        for si, g in rc.iter_parts(sel, batch=batch):
            for k, geom in zip(si, g):
                if geom is None or geom.is_empty:
                    continue
                hit = tree.query(geom, predicate="intersects")
                if hit.size:
                    acc_sp.append(np.full(hit.size, k, np.int32))
                    acc_city.append(cid[hit].astype(np.int32))
        i = j

    if acc_sp:
        pair_sp.append(np.concatenate(acc_sp))
        pair_city.append(np.concatenate(acc_city))
    ps = np.concatenate(pair_sp) if pair_sp else np.zeros(0, np.int32)
    pc = np.concatenate(pair_city) if pair_city else np.zeros(0, np.int32)
    if ps.size:
        key = ps.astype(np.int64) * 100000 + pc.astype(np.int64)
        uniq = np.unique(key)
        ps, pc = (uniq // 100000).astype(np.int32), (uniq % 100000).astype(np.int32)
    tmp = PAIRS_PATH.with_suffix(".tmp.npz")
    np.savez(tmp, pair_sp=ps, pair_city=pc, done=i)
    import os
    os.replace(tmp, PAIRS_PATH)
    if verbose:
        print(f"  parts {i:,}/{n_parts:,} ({i/n_parts*100:.1f}%)  "
              f"pairs {ps.size:,}  {time.time()-t0:.0f}s")
    return i, n_parts, ps.size


def write_table():
    import tropics_rangecache as rc
    z = np.load(PAIRS_PATH, allow_pickle=False)
    d = pd.DataFrame(dict(sp_i=z["pair_sp"], city_id=z["pair_city"])).drop_duplicates()
    sp, _ = rc.load_index()
    thr = sp.category.isin(THREAT_CATS).to_numpy()
    cc = pd.read_csv(TABLE_DIR / "cities_cost.csv")
    out = cc[["city_id", "setting_richness", "urban_km2_final", "cost_per_km2"]].copy()
    fp = d[thr[d.sp_i.to_numpy()]].groupby("city_id").sp_i.nunique()
    fa = d.groupby("city_id").sp_i.nunique()
    out["setting_fp"] = fp.reindex(out.city_id).to_numpy(float)
    out["setting_fp_all"] = fa.reindex(out.city_id).to_numpy(float)
    out.to_csv(TABLE_DIR / "setting_footprint.csv", index=False)
    return out
