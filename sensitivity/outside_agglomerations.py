"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fig_trivariate as tv
import resolution_resume as rr
import resolution_test as rt
import tropics_rangecache as rc
from tropics_config import EXCLUDED_TAXA, INTERIM_DIR, TABLE_DIR
from tropics_measure import MEASURE_PATH

PART_DIR = INTERIM_DIR / "outside"
OUT = TABLE_DIR / "outside_agglomerations.json"
CSV = TABLE_DIR / "outside_agglomerations_species.csv"
N_SLICES = 24
CLASSES = [("below 0.05", 0.0, 0.05), ("0.05 to 0.10", 0.05, 0.10),
           ("0.10 to 0.20", 0.10, 0.20), ("0.20 or more", 0.20, np.inf)]


def outside_cells():
    lon, lat, P = tv.lattice_parts()
    added = P[:, 1:].sum(1)
    frac = _counts_2020() / tv.CELLS_PER_DELIN
    assert frac.size == lon.size
    m = np.load(MEASURE_PATH, allow_pickle=False)
    key_l = (np.round((tv.MOSAIC_LAT_TOP - lat) / tv.DELIN_DEG - 0.5).astype(np.int64) * 10 ** 6
             + np.round((lon + 180) / tv.DELIN_DEG - 0.5).astype(np.int64))
    key_m = m["cell_row"].astype(np.int64) * 10 ** 6 + m["cell_col"].astype(np.int64)
    k = ~np.isin(key_l, key_m) & (added > 0)
    cls = np.zeros(k.sum(), np.int8)
    f = frac[k]
    for j, (_n, lo, hi) in enumerate(CLASSES):
        cls[(f >= lo) & (f < hi)] = j
    o = np.argsort(lon[k], kind="stable")
    return lon[k][o], lat[k][o], added[k][o], cls[o]


def _counts_2020():
    z = np.load(tv.t.OUT_DIR / "lattice_cum_counts.npz")
    key, cum = z["key"], z["cum"].astype(np.int64)
    r = key // tv.MOSAIC_COLS
    ok = r < tv.MOSAIC_ROWS
    lat = tv.MOSAIC_LAT_TOP - (r[ok] + 0.5) * tv.DELIN_DEG
    return cum[ok][np.abs(lat) <= tv.rb.TROPIC, 4]


def hits(g, lon, lat):
    x0, y0, x1, y1 = g.bounds
    a = int(np.searchsorted(lon, x0, side="left"))
    b = int(np.searchsorted(lon, x1, side="right"))
    if b <= a:
        return np.zeros(0, np.int64)
    sub = np.arange(a, b)
    sub = sub[(lat[a:b] >= y0) & (lat[a:b] <= y1)]
    if not sub.size:
        return sub
    shapely.prepare(g)
    return sub[shapely.intersects_xy(g, lon[sub], lat[sub])]


def concern():
    sp, parts = rc.load_index()
    flags, thr = rt.concern_flags(sp)
    keep = ~sp.taxon.astype(str).str.upper().isin([t.upper() for t in EXCLUDED_TAXA]).to_numpy()
    rng = flags.range_km2.to_numpy()
    narrow = keep & (rng > 0) & (rng <= thr)
    out = pd.DataFrame(dict(sp_i=sp.sp_i, sci_name=sp.sci_name, taxon=sp.taxon,
                            category=sp.category, range_km2=rng,
                            is_threat=flags.is_threat.to_numpy(), is_narrow=narrow))
    return out[out.is_threat | out.is_narrow].reset_index(drop=True), parts, thr


def slices(n=N_SLICES):
    sp, parts, _thr = concern()
    want = sp.sp_i.to_numpy()
    m = np.isin(parts[0], want)
    uniq, counts = np.unique(parts[0][m], return_counts=True)
    edges = np.searchsorted(np.cumsum(counts), np.linspace(0, counts.sum(), n + 1)[1:])
    out, lo = [], 0
    for e in edges:
        e = int(min(max(e + 1, lo + 1), uniq.size))
        out.append(uniq[lo:e])
        lo = e
    out = [g for g in out if g.size]
    assert sum(g.size for g in out) == uniq.size
    return out, parts


def slice_path(k, n=N_SLICES):
    return PART_DIR / f"slice_{k:02d}_of_{n:02d}.npz"


def run_slice(k, n=N_SLICES):
    p = slice_path(k, n)
    if p.exists():
        print(f"slice {k} already measured", flush=True)
        return
    t0 = time.time()
    groups, parts = slices(n)
    g = groups[k]
    m = np.isin(parts[0], g)
    sub = (parts[0][m], parts[1][m], parts[2][m])
    lon, lat, added, cls = outside_cells()
    found = {}
    for sp_batch, geom_batch in rc.iter_parts(sub):
        for i, geom in zip(sp_batch, geom_batch):
            if geom is None or geom.is_empty:
                continue
            h = hits(geom, lon, lat)
            if h.size:
                found.setdefault(int(i), []).append(h)
    ids = np.array(sorted(found), np.int64)
    conv = np.zeros((ids.size, len(CLASSES)))
    cells = np.zeros(ids.size, np.int64)
    for j, i in enumerate(ids):
        u = np.unique(np.concatenate(found[int(i)]))
        conv[j] = np.bincount(cls[u], weights=added[u], minlength=len(CLASSES))
        cells[j] = u.size
    PART_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(p, sp_i=ids, conv=conv, cells=cells, assigned=g)
    print(f"slice {k}/{len(groups)}: {g.size:,} species, {ids.size:,} reached, "
          f"{conv.sum():,.0f} km2 [{time.time() - t0:.0f}s]", flush=True)


def status(n=N_SLICES):
    done = [k for k in range(n) if slice_path(k, n).exists()]
    print(f"{len(done)}/{n} slices measured; missing {[k for k in range(n) if k not in done]}")
    return len(done) == n


def finish(n=N_SLICES):
    sp, _parts, thr = concern()
    got, assigned = [], []
    for k in range(n):
        with np.load(slice_path(k, n)) as z:
            got.append(pd.DataFrame(dict(sp_i=z["sp_i"], **{f"out_{c[0]}": z["conv"][:, j]
                                                             for j, c in enumerate(CLASSES)})))
            assigned.append(z["assigned"])
    a = np.concatenate(assigned)
    assert np.unique(a).size == a.size, "a species was measured twice"
    with_parts = np.unique(_parts[0][np.isin(_parts[0], sp.sp_i.to_numpy())])
    assert np.setdiff1d(with_parts, a).size == 0, "species not measured"
    out = pd.concat(got, ignore_index=True)
    out["outside_km2"] = out[[f"out_{c[0]}" for c in CLASSES]].sum(1)
    pairs, _meta = rr.load_slices("30:centroid")
    inside = pairs.groupby("sp_i").conv_km2.sum().rename("inside_km2")
    d = sp.merge(out, on="sp_i", how="left").merge(inside, on="sp_i", how="left").fillna(
        {"inside_km2": 0.0, "outside_km2": 0.0, **{f"out_{c[0]}": 0.0 for c in CLASSES}})
    d["share_inside_pct"] = 100 * d.inside_km2 / d.range_km2
    d["share_all_pct"] = 100 * (d.inside_km2 + d.outside_km2) / d.range_km2
    lon, lat, added, cls = outside_cells()
    added_cls = np.bincount(cls, weights=added, minlength=len(CLASSES))
    land_in = float(np.load(MEASURE_PATH)["cell_conv"].sum())

    def group(m):
        x = d[m]
        ri, ro = x.inside_km2 > 0, x.outside_km2 > 0
        return dict(species=int(len(x)), reached_inside=int(ri.sum()), reached_outside=int(ro.sum()),
                    reached_either=int((ri | ro).sum()), reached_outside_only=int((ro & ~ri).sum()),
                    above_1pct_inside=int((x.share_inside_pct >= 1).sum()),
                    above_1pct_either=int((x.share_all_pct >= 1).sum()),
                    converted_inside_km2=round(float(x.inside_km2.sum()), 1),
                    converted_outside_km2=round(float(x.outside_km2.sum()), 1),
                    median_outside_km2_outside_only=round(float(x.outside_km2[ro & ~ri].median()), 3))

    res = dict(
        narrow_threshold_km2=thr,
        land_added_inside_km2=round(land_in, 1),
        land_added_outside_km2=round(float(added.sum()), 1),
        land_added_outside_by_class_km2={c[0]: round(float(v), 1) for c, v in zip(CLASSES, added_cls)},
        species_of_concern=group(np.ones(len(d), bool)),
        threatened=group(d.is_threat.to_numpy()),
        narrow_range=group(d.is_narrow.to_numpy()),
        threatened_by_taxon={t: group((d.is_threat & (d.taxon == t)).to_numpy())
                             for t in sorted(d[d.is_threat].taxon.unique())},
        converted_outside_by_class_km2={c[0]: round(float(d[f"out_{c[0]}"].sum()), 1) for c in CLASSES},
    )
    res["intensity_inside"] = round(res["species_of_concern"]["converted_inside_km2"] / land_in, 2)
    res["intensity_outside"] = round(res["species_of_concern"]["converted_outside_km2"] / added.sum(), 2)
    res["intensity_outside_by_class"] = {c[0]: round(float(d[f"out_{c[0]}"].sum() / v), 2)
                                         for c, v in zip(CLASSES, added_cls)}
    OUT.write_text(json.dumps(res, indent=1))
    d.to_csv(CSV, index=False)
    print(json.dumps(res, indent=1))
    return res


def run(n=N_SLICES):
    for k in range(n):
        run_slice(k, n)
    return finish(n)


def _selftest():
    xs, ys = np.meshgrid(np.arange(0.5, 10), np.arange(0.5, 10))
    lon, lat = xs.ravel(), ys.ravel()
    o = np.argsort(lon, kind="stable")
    lon, lat = lon[o], lat[o]
    sq = shapely.box(2, 2, 5, 5)
    assert hits(sq, lon, lat).size == 9
    tri = shapely.Polygon([(0, 0), (10.2, 0), (0, 10.2)])
    assert hits(tri, lon, lat).size == 55
    assert hits(shapely.box(20, 20, 21, 21), lon, lat).size == 0
    two = [hits(shapely.box(0, 0, 4, 4), lon, lat), hits(shapely.box(2, 2, 6, 6), lon, lat)]
    assert np.unique(np.concatenate(two)).size == 16 + 16 - 4
    print("selftest passed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "selftest":
        _selftest()
    elif cmd == "slice":
        for k in sys.argv[2:]:
            run_slice(int(k))
    elif cmd == "run":
        _selftest()
        run()
    elif cmd == "finish":
        finish()
    else:
        status()
