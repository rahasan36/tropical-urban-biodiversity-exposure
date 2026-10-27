"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyogrio
import shapely
from pyproj import Geod
from shapely import STRtree

from tropics_config import (BOTW_GPKG, BOTW_LAYER, TABLE_DIR, TROPIC_LAT,
                            SETTING_BUFFER_KM, PRESENCE_KEEP, ORIGIN_KEEP,
                            SEASONAL_KEEP)
import tropics_cost as tc

GEOD = Geod(ellps="WGS84")
STATE = TABLE_DIR / "birdlife_union.json"
OUT = TABLE_DIR / "birdlife_union_summary.csv"
OUT_PAIRS = TABLE_DIR / "birdlife_union_pairs.npz"
BATCH_SPECIES = 200


def geodesic_area_km2(g):
    if g is None or g.is_empty:
        return 0.0
    return abs(GEOD.geometry_area_perimeter(g)[0]) / 1.0e6


def union_area_km2(geoms):
    gs = [g for g in geoms if g is not None and not g.is_empty]
    if not gs:
        return 0.0, None
    try:
        u = shapely.union_all(gs)
    except Exception:
        gs = [g if g.is_valid else g.buffer(0) for g in gs]
        u = shapely.union_all(gs)
    return geodesic_area_km2(u), u


def species_feature_map(verbose=True):
    cols = ["sisid", "sci_name", "presence", "origin", "seasonal"]
    t0 = time.time()
    df = pyogrio.read_dataframe(BOTW_GPKG, layer=BOTW_LAYER, columns=cols,
                                read_geometry=False)
    if verbose:
        print(f"  attributes read: {len(df):,} features "
              f"[{time.time() - t0:.0f}s]", flush=True)

    def num(c):
        return pd.to_numeric(df[c], errors="coerce")

    keep = (num("presence").isin(PRESENCE_KEEP)
            & num("origin").isin(ORIGIN_KEEP)
            & num("seasonal").isin(SEASONAL_KEEP)).to_numpy()
    df = df.loc[keep].copy()
    df["_row"] = np.flatnonzero(keep)
    df["sp_id"] = "AVES_" + df["sisid"].astype("int64").astype(str)
    if verbose:
        print(f"  after presence, origin and seasonal filters: "
              f"{len(df):,} features, {df.sp_id.nunique():,} species",
              flush=True)
    return df


def read_rows(rows):
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    out = []
    start = 0
    while start < rows.size:
        end = start
        while (end + 1 < rows.size
               and rows[end + 1] - rows[start] < 20000):
            end += 1
        lo, hi = int(rows[start]), int(rows[end])
        df = pyogrio.read_dataframe(BOTW_GPKG, layer=BOTW_LAYER,
                                    columns=["sisid", "sci_name"],
                                    skip_features=lo,
                                    max_features=hi - lo + 1)
        want = rows[start:end + 1] - lo
        out.append(df.iloc[want])
        start = end + 1
    return pd.concat(out, ignore_index=True) if out else None


def run(budget_s=3600, verbose=True):
    cells, cities = tc.load_city_geometry()
    cell_tree, city_tree = tc.build_trees(cells, cities)
    cell_city = cells.city_id.to_numpy()
    cell_conv = cells.conv_km2.to_numpy()
    n_city = int(cities.city_id.max())
    buf_deg = SETTING_BUFFER_KM / 111.32

    fmap = species_feature_map(verbose=verbose)
    groups = fmap.groupby("sp_id")["_row"].apply(list).to_dict()
    names = fmap.drop_duplicates("sp_id").set_index("sp_id")["sci_name"].to_dict()

    st = json.loads(STATE.read_text()) if STATE.exists() else {"done": {}}
    done = st["done"]
    todo = [s for s in groups if s not in done]
    if verbose:
        print(f"  {len(done):,} species already measured, {len(todo):,} to go",
              flush=True)

    t0 = time.time()
    processed = 0
    for sp in todo:
        if time.time() - t0 > budget_s:
            if verbose:
                print("  budget reached; state saved, rerun to continue",
                      flush=True)
            break
        rows = groups[sp]
        df = read_rows(rows)
        if df is None or not len(df):
            done[sp] = dict(area_union=0.0, area_sum=0.0, n_features=0,
                            cities=[], conv=[])
            continue
        geoms = [g for g in df.geometry.to_numpy() if g is not None]
        area_sum = float(sum(geodesic_area_km2(g) for g in geoms))
        area_union, u = union_area_km2(geoms)

        city_ids, conv = np.zeros(0, np.int32), np.zeros(0)
        near = np.zeros(0, np.int32)
        if u is not None:
            b = shapely.bounds(u)
            if not (b[3] < -TROPIC_LAT - buf_deg or b[1] > TROPIC_LAT + buf_deg):
                idx = cell_tree.query(u, predicate="intersects")
                if idx.size:
                    city_ids, conv = tc.union_converted(
                        [idx.astype(np.int32)], cell_city, cell_conv, n_city)
                nb = city_tree.query(u, predicate="dwithin", distance=buf_deg)
                if nb.size:
                    near = cities.city_id.to_numpy()[nb].astype(np.int32)
        done[sp] = dict(area_union=area_union, area_sum=area_sum,
                        n_features=len(geoms),
                        cities=city_ids.tolist(), conv=[float(v) for v in conv],
                        near=near.tolist())
        processed += 1
        if processed % BATCH_SPECIES == 0:
            st["done"] = done
            STATE.write_text(json.dumps(st))
            if verbose:
                print(f"  {len(done):,}/{len(groups):,} species "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    st["done"] = done
    STATE.write_text(json.dumps(st))
    return done, names, len(groups)


def summarize(done, names):
    rows = []
    ps, pc, pv = [], [], []
    for sp, d in done.items():
        rows.append(dict(sp_id=sp, sci_name=names.get(sp, ""),
                         n_features=d["n_features"],
                         area_union_km2=d["area_union"],
                         area_sum_km2=d["area_sum"],
                         inflation=(d["area_sum"] / d["area_union"] - 1.0
                                    if d["area_union"] > 0 else np.nan),
                         converted_km2=float(sum(d["conv"]))))
        for c, v in zip(d["cities"], d["conv"]):
            ps.append(sp); pc.append(int(c)); pv.append(float(v))
    t = pd.DataFrame(rows)
    pairs = pd.DataFrame(dict(sp_id=ps, city_id=pc, conv_km2=pv))
    return t, pairs


def _selftest():
    from shapely.geometry import box

    a = box(0.0, 0.0, 1.0, 1.0)
    b = box(0.5, 0.0, 1.5, 1.0)
    s = geodesic_area_km2(a) + geodesic_area_km2(b)
    u, _ = union_area_km2([a, b])
    overlap = geodesic_area_km2(box(0.5, 0.0, 1.0, 1.0))
    assert abs((s - u) - overlap) / overlap < 1e-3, (s, u, overlap)
    assert u < s

    cell_city = np.array([0, 1, 1, 1, 2], dtype=np.int64)
    cell_conv = np.array([0.0, 2.0, 3.0, 4.0, 5.0])
    f1 = np.array([1, 2], np.int32)
    f2 = np.array([2, 3], np.int32)
    ids, vals = tc.union_converted([f1, f2], cell_city, cell_conv, 2)
    assert ids.tolist() == [1], ids
    assert abs(vals[0] - 9.0) < 1e-9, vals
    m1 = tc.union_converted([f1], cell_city, cell_conv, 2)[1][0]
    m2 = tc.union_converted([f2], cell_city, cell_conv, 2)[1][0]
    assert abs(max(m1, m2) - 7.0) < 1e-9, (m1, m2)
    assert max(m1, m2) < 9.0, "the maximum rule must under-count here"

    sums = geodesic_area_km2(box(0, 0, 1, 1)) * 2
    dis, _ = union_area_km2([box(0, 0, 1, 1), box(10, 10, 11, 11)])
    assert abs(dis - sums) / sums < 0.15, (dis, sums)

    print("ok  union beats sum by exactly the overlap; "
          "the maximum rule under-counts a split city by 2.0 of 9.0 here")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--budget", type=float, default=3600.0)
    args = ap.parse_args()
    _selftest()
    if args.selftest:
        raise SystemExit(0)
    done, names, n_total = run(budget_s=args.budget)
    t, pairs = summarize(done, names)
    t.to_csv(OUT, index=False)
    np.savez_compressed(OUT_PAIRS,
                        sp_id=pairs.sp_id.to_numpy().astype("U24"),
                        city_id=pairs.city_id.to_numpy().astype(np.int32),
                        conv_km2=pairs.conv_km2.to_numpy())
    ok = t[t.area_union_km2 > 0]
    print(f"\nspecies measured {len(t):,} of {n_total:,}")
    print(f"median area inflation from summing features: "
          f"{ok.inflation.median():+.4%}")
    print(f"share of species whose summed area exceeds the union by over 5%: "
          f"{(ok.inflation > 0.05).mean():.3%}")
    print(f"total converted range, birds, union rule: "
          f"{t.converted_km2.sum():,.0f} km2")
    print(f"written {OUT}")
