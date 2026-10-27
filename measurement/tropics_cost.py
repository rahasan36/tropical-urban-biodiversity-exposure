"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyogrio, shapely
from shapely import STRtree
from pyproj import Geod
from pathlib import Path
import json, time

from tropics_config import (IUCN_SOURCES, BOTW_GPKG, BOTW_LAYER, TROPIC_LAT,
                            INTERIM_DIR, DELIN_DEG, MOSAIC_LAT_TOP,
                            SETTING_BUFFER_KM, PRESENCE_KEEP, ORIGIN_KEEP,
                            SEASONAL_KEEP, EXCLUDED_TAXA)
from tropics_measure import MEASURE_PATH

GEOD = Geod(ellps="WGS84")
from tropics_config import SCRATCH
PAIRS_LIVE = SCRATCH / "species_city_pairs.npz"
PAIRS_FINAL = INTERIM_DIR / "species_city_pairs.npz"


def _pairs_path():
    if PAIRS_LIVE.exists():
        return PAIRS_LIVE
    if PAIRS_FINAL.exists():
        return PAIRS_FINAL
    return PAIRS_LIVE


PAIRS_PATH = _pairs_path()
PAIRS_TAGGED = None
CHUNK = 300


def load_city_geometry():
    z = np.load(MEASURE_PATH, allow_pickle=False)
    lat = MOSAIC_LAT_TOP - (z["cell_row"].astype(np.float64) + 0.5) * DELIN_DEG
    lon = -180.0 + (z["cell_col"].astype(np.float64) + 0.5) * DELIN_DEG
    cells = pd.DataFrame(dict(city_id=z["cell_city"], lon=lon, lat=lat,
                              conv_km2=z["cell_conv"]))
    cities = (cells.groupby("city_id")
                   .agg(lon=("lon", "mean"), lat=("lat", "mean"),
                        conv_km2=("conv_km2", "sum"), n_cells=("lon", "size"))
                   .reset_index())
    return cells, cities


def union_converted(cell_idx_parts, cell_city, cell_conv, n_city):
    if not cell_idx_parts:
        return np.zeros(0, np.int32), np.zeros(0)
    u = np.unique(np.concatenate(cell_idx_parts))
    agg = np.bincount(cell_city[u], weights=cell_conv[u], minlength=n_city + 1)
    nz = np.flatnonzero(agg)
    return nz.astype(np.int32), agg[nz]


def build_trees(cells, cities):
    cell_pts = shapely.points(cells.lon.to_numpy(), cells.lat.to_numpy())
    city_pts = shapely.points(cities.lon.to_numpy(), cities.lat.to_numpy())
    return STRtree(cell_pts), STRtree(city_pts)


def _sources():
    excl = {t.upper() for t in EXCLUDED_TAXA}
    src = []
    for taxon, files in IUCN_SOURCES.items():
        for f in files:
            src.append(dict(taxon=taxon, path=str(f), layer=None,
                            id_field="id_no", kind="iucn"))
    src.append(dict(taxon="AVES", path=str(BOTW_GPKG), layer=BOTW_LAYER,
                    id_field="sisid", kind="botw"))
    return [s for s in src if s["taxon"].upper() not in excl]


def _read_chunk(s, skip, n):
    cols = [s["id_field"], "sci_name", "presence", "origin", "seasonal"]
    if s["kind"] == "iucn":
        cols.append("category")
    kw = dict(columns=cols, skip_features=skip, max_features=n)
    if s["layer"]:
        kw["layer"] = s["layer"]
    return pyogrio.read_dataframe(s["path"], **kw)


def _keep(df):
    def i(c):
        return pd.to_numeric(df[c], errors="coerce")
    return (i("presence").isin(PRESENCE_KEEP) & i("origin").isin(ORIGIN_KEEP)
            & i("seasonal").isin(SEASONAL_KEEP)).to_numpy()


def run_pass(budget_s=150, verbose=True):
    cells, cities = load_city_geometry()
    cell_tree, city_tree = build_trees(cells, cities)
    cell_city = cells.city_id.to_numpy()
    cell_conv = cells.conv_km2.to_numpy()
    city_ids_arr = cities.city_id.to_numpy()
    n_city = int(cities.city_id.max())
    buf_deg = SETTING_BUFFER_KM / 111.32

    state, sp_index, sp_meta = {}, {}, {}
    sp_area = []
    pair_sp, pair_city, pair_conv = [], [], []
    set_sp, set_city = [], []
    if PAIRS_PATH.exists():
        z = np.load(PAIRS_PATH, allow_pickle=False)
        state = json.loads(str(z["state"]))
        sp_index = json.loads(str(z["sp_index"]))
        sp_meta = json.loads(str(z["sp_meta"]))
        sp_area = list(z["sp_area"])
        pair_sp = [z["pair_sp"]]; pair_city = [z["pair_city"]]
        pair_conv = [z["pair_conv"]]
        set_sp = [z["set_sp"]]; set_city = [z["set_city"]]

    def sp_slot(sp):
        i = sp_index.get(sp)
        if i is None:
            i = len(sp_index)
            sp_index[sp] = i
            sp_area.append(0.0)
        return i

    open_ = dict(sp=None, i=-1, start=0, area=0.0, cells=[], near=[])

    def flush_open():
        if open_["sp"] is None:
            return
        sp_area[open_["i"]] += open_["area"]
        nz, vals = union_converted(open_["cells"], cell_city, cell_conv, n_city)
        if nz.size:
            pair_sp.append(np.full(nz.size, open_["i"], np.int32))
            pair_city.append(nz)
            pair_conv.append(vals.astype(np.float32))
        if open_["near"]:
            u = np.unique(np.concatenate(open_["near"]))
            set_sp.append(np.full(u.size, open_["i"], np.int32))
            set_city.append(u.astype(np.int32))
        open_.update(sp=None, i=-1, area=0.0, cells=[], near=[])

    t0 = time.time()
    stop = False
    for s in _sources():
        if stop:
            break
        lk = {"layer": s["layer"]} if s["layer"] else {}
        total = pyogrio.read_info(s["path"], **lk)["features"]
        key = f"{Path(s['path']).name}:{s['layer'] or ''}"
        chunk_start = state.get(key, 0)
        while chunk_start < total:
            df = _read_chunk(s, chunk_start, CHUNK)
            n_read = len(df)
            if n_read == 0:
                state[key] = total
                break
            m = _keep(df)
            pos = np.flatnonzero(m)
            df = df[m]
            ids = df[s["id_field"]].astype("int64").astype(str)
            sp_ids = (s["taxon"][:4] + "_" + ids).tolist()
            cats = (df["category"].astype(str).tolist() if s["kind"] == "iucn"
                    else [""] * len(df))
            names = df["sci_name"].astype(str).tolist()
            geoms = df.geometry.to_numpy()
            for j, (sp, cat, nm, g) in enumerate(zip(sp_ids, cats, names, geoms)):
                if sp != open_["sp"]:
                    flush_open()
                    open_["sp"] = sp
                    open_["i"] = sp_slot(sp)
                    open_["start"] = chunk_start + int(pos[j])
                if sp not in sp_meta:
                    sp_meta[sp] = [nm, cat, s["taxon"]]
                elif sp_meta[sp][1] in ("", "nan", "None") and cat not in ("", "nan"):
                    sp_meta[sp][1] = cat
                if g is None or g.is_empty:
                    continue
                open_["area"] += abs(GEOD.geometry_area_perimeter(g)[0]) / 1.0e6
                minx, miny, maxx, maxy = shapely.bounds(g)
                if maxy < -TROPIC_LAT - buf_deg or miny > TROPIC_LAT + buf_deg:
                    continue
                idx = cell_tree.query(g, predicate="intersects")
                if idx.size:
                    open_["cells"].append(idx.astype(np.int32))
                near = city_tree.query(g, predicate="dwithin", distance=buf_deg)
                if near.size:
                    open_["near"].append(city_ids_arr[near].astype(np.int32))
            chunk_start += n_read
            state[key] = (open_["start"] if open_["sp"] is not None else chunk_start)
            if verbose and (chunk_start // CHUNK) % 10 == 0:
                print(f"  {key} {chunk_start}/{total}  {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > budget_s:
                stop = True
                break
        if not stop:
            flush_open()
            state[key] = total
    if stop:
        open_.update(sp=None, i=-1, area=0.0, cells=[], near=[])

    def cat_(parts, dt):
        return np.concatenate(parts) if parts else np.zeros(0, dt)

    ps = cat_(pair_sp, np.int32); pc = cat_(pair_city, np.int32)
    pv = cat_(pair_conv, np.float32)
    ss = cat_(set_sp, np.int32); sc = cat_(set_city, np.int32)
    if ps.size:
        k = ps.astype(np.int64) * (n_city + 1) + pc
        o = np.argsort(k, kind="stable"); k, ps, pc, pv = k[o], ps[o], pc[o], pv[o]
        first = np.r_[True, k[1:] != k[:-1]]
        grp = np.cumsum(first) - 1
        out = np.zeros(int(grp[-1]) + 1, np.float64)
        np.maximum.at(out, grp, pv.astype(np.float64))
        pv = out.astype(np.float32)
        ps, pc = ps[first], pc[first]
    if ss.size:
        k = ss.astype(np.int64) * (n_city + 1) + sc
        u = np.unique(k)
        ss = (u // (n_city + 1)).astype(np.int32)
        sc = (u % (n_city + 1)).astype(np.int32)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp_path = PAIRS_LIVE.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path, pair_sp=ps, pair_city=pc, pair_conv=pv,
        set_sp=ss, set_city=sc, sp_area=np.asarray(sp_area, np.float64),
        state=json.dumps(state), sp_index=json.dumps(sp_index),
        sp_meta=json.dumps(sp_meta))
    import os
    os.replace(tmp_path, PAIRS_LIVE)

    remaining = 0
    for s in _sources():
        lk = {"layer": s["layer"]} if s["layer"] else {}
        info = pyogrio.read_info(s["path"], **lk)
        key = f"{Path(s['path']).name}:{s['layer'] or ''}"
        remaining += max(0, info["features"] - state.get(key, 0))
    if remaining == 0:
        import shutil
        PAIRS_FINAL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PAIRS_LIVE, PAIRS_FINAL)
    return dict(features_remaining=int(remaining), species_seen=len(sp_index),
                pairs=int(pv.size), setting_pairs=int(ss.size))


if __name__ == "__main__":
    from shapely.geometry import box, Point

    cells = pd.DataFrame(dict(city_id=[1, 1, 2, 2, 3],
                              lon=[0.0, 0.01, 10.0, 10.01, 50.0],
                              lat=[0.0, 0.0, 0.0, 0.0, 0.0],
                              conv_km2=[1.0, 2.0, 4.0, 8.0, 16.0]))
    cities = (cells.groupby("city_id")
                   .agg(lon=("lon", "mean"), lat=("lat", "mean"),
                        conv_km2=("conv_km2", "sum"), n_cells=("lon", "size"))
                   .reset_index())
    cell_tree, city_tree = build_trees(cells, cities)
    cell_city = cells.city_id.to_numpy(); cell_conv = cells.conv_km2.to_numpy()

    g = box(-0.5, -0.5, 0.5, 0.5)
    idx = cell_tree.query(g, predicate="intersects")
    agg = np.bincount(cell_city[idx], weights=cell_conv[idx], minlength=4)
    assert agg[1] == 3.0 and agg[2] == 0.0 and agg[3] == 0.0, agg
    print("range over one city      ok  (3.0 km2 charged to city 1 only)")

    g2 = box(9.995, -0.5, 10.005, 0.5)
    idx = cell_tree.query(g2, predicate="intersects")
    agg = np.bincount(cell_city[idx], weights=cell_conv[idx], minlength=4)
    assert agg[2] == 4.0, agg
    print("partial overlap          ok  (only the covered cell charged)")

    tot = 0.0
    for _ in range(2):
        idx = cell_tree.query(g, predicate="intersects")
        tot += cell_conv[idx].sum()
    assert tot == 6.0
    print("species-weighted sum     ok  (same pixels charged once per species)")

    assert set(np.unique(cell_city[cell_tree.query(g, predicate='intersects')])) == {1}
    print("no cross-city leakage    ok")

    near = city_tree.query(box(0.2, -0.05, 0.3, 0.05), predicate="dwithin",
                           distance=50.0 / 111.32)
    assert set(cities.city_id.to_numpy()[near]) == {1}, near
    far = city_tree.query(box(0.9, -0.05, 1.0, 0.05), predicate="dwithin",
                          distance=50.0 / 111.32)
    assert far.size == 0, far
    print("setting-richness query   ok  (inside 50 km found, beyond it not)")

    parts = [np.array([0, 1]), np.array([1, 2]), np.array([3])]
    ids, vals = union_converted(parts, cell_city, cell_conv, 3)
    assert ids.tolist() == [1, 2], ids
    assert vals.tolist() == [3.0, 12.0], vals
    sum_naive = cell_conv[np.concatenate(parts)].sum()
    assert sum_naive == 17.0 and vals.sum() == 15.0
    print("overlapping polygons     ok  (union 15.0 km2, naive sum would be 17.0)")

    a = abs(GEOD.geometry_area_perimeter(box(0, 0, 1, 1))[0]) / 1e6
    assert 12_300 < a < 12_400, a
    print(f"range area (1 deg box)   ok  ({a:,.0f} km2)")
    print("ALL COST-ATTRIBUTION TESTS PASS")
