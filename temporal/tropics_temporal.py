"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

from tropics_config import INTERIM_DIR, TABLE_DIR, EXCLUDED_TAXA
from tropics_grid import row_area_km2
from tropics_urban import (list_tiles, mosaic_index, tile_corner, MOSAIC_COLS,
                           ROW_CHUNK, TILE_DELIN, CELLS_PER_DELIN)
from tropics_measure import MEASURE_PATH

OUT_DIR = INTERIM_DIR / "temporal"
TILE_DIR_OUT = OUT_DIR / "tiles"
HIT_DIR = OUT_DIR / "hits"
YEARS = (2000, 2005, 2010, 2015, 2020)
YEAR_CODE = {2000: 4, 2005: 5, 2010: 6, 2015: 7, 2020: 8}
PERIODS = ((2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020))
N_CODE = 9


def city_cell_keys():
    z = np.load(MEASURE_PATH, allow_pickle=False)
    key = z["cell_row"].astype(np.int64) * MOSAIC_COLS + z["cell_col"].astype(np.int64)
    o = np.argsort(key, kind="stable")
    return key, o


def fold_tile(path, ckey_sorted, corder, row_chunk=ROW_CHUNK):
    lon_w, lat_n = tile_corner(path)
    mr0, mc0 = mosaic_index(np.array([lat_n]), np.array([lon_w]))
    mr0, mc0 = int(mr0[0]), int(mc0[0])
    nloc = TILE_DELIN * TILE_DELIN
    cnt = np.zeros((nloc, N_CODE), np.int64)
    n_cell = ckey_sorted.size
    cell_area = {}
    with rasterio.open(path) as src:
        T = src.transform
        H, W = src.height, src.width
        for r0 in range(0, H, row_chunk):
            h = min(row_chunk, H - r0)
            arr = src.read(1, window=Window(0, r0, W, h))
            hit = (arr >= 1) & (arr <= 8)
            if not hit.any():
                continue
            rr, cc = np.nonzero(hit)
            vals = arr[rr, cc].astype(np.int64)
            lat_top = T.f + (r0 + rr) * T.e
            lat = lat_top + 0.5 * T.e
            lon = T.c + (cc + 0.5) * T.a
            mr, mc = mosaic_index(lat, lon)
            lr, lc = mr - mr0, mc - mc0
            ok = (lr >= 0) & (lr < TILE_DELIN) & (lc >= 0) & (lc < TILE_DELIN)
            flat = (lr[ok] * TILE_DELIN + lc[ok]).astype(np.int64)
            cnt += np.bincount(flat * N_CODE + vals[ok], minlength=nloc * N_CODE
                               ).reshape(nloc, N_CODE)
            area = row_area_km2(lat_top, lat_top + T.e, T.a)
            k = mr.astype(np.int64) * MOSAIC_COLS + mc
            pos = np.searchsorted(ckey_sorted, k)
            np.minimum(pos, n_cell - 1, out=pos)
            inside = ckey_sorted[pos] == k
            if inside.any():
                ci = corder[pos[inside]]
                a = np.bincount(ci * N_CODE + vals[inside], weights=area[inside])
                nz = np.flatnonzero(a)
                for f, v in zip(nz, a[nz]):
                    cell_area[int(f)] = cell_area.get(int(f), 0.0) + float(v)
    cum = np.cumsum(cnt, axis=1)[:, [YEAR_CODE[y] for y in YEARS]]
    nzl = np.flatnonzero(cum[:, -1])
    lkey = ((nzl // TILE_DELIN + mr0).astype(np.int64) * MOSAIC_COLS
            + (nzl % TILE_DELIN + mc0))
    fk = np.array(sorted(cell_area), np.int64)
    fv = np.array([cell_area[f] for f in fk], np.float64)
    return dict(lat_key=lkey, lat_cum=cum[nzl].astype(np.uint16),
                cell_idx=(fk // N_CODE).astype(np.int32),
                cell_code=(fk % N_CODE).astype(np.int8), cell_area=fv)


def run_tiles(budget_s=150.0, tile_dir=None, verbose=True):
    TILE_DIR_OUT.mkdir(parents=True, exist_ok=True)
    key, o = city_cell_keys()
    ks = key[o]
    t0 = time.time()
    tiles = list_tiles(*([tile_dir] if tile_dir else []))
    todo = [p for p in tiles if not (TILE_DIR_OUT / (p.stem + ".npz")).exists()]
    n = 0
    for p in todo:
        r = fold_tile(p, ks, o.astype(np.int64))
        tmp = TILE_DIR_OUT / (p.stem + ".part.npz")
        np.savez_compressed(tmp, **r)
        tmp.rename(TILE_DIR_OUT / (p.stem + ".npz"))
        n += 1
        if time.time() - t0 > budget_s:
            break
    left = len(todo) - n
    if verbose:
        print(json.dumps(dict(done_now=n, remaining=left, tiles=len(tiles),
                              seconds=round(time.time() - t0))), flush=True)
    return left


def assemble():
    files = sorted(TILE_DIR_OUT.glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".part.npz")]
    K, C, I, E, A = [], [], [], [], []
    for f in files:
        with np.load(f) as z:
            K.append(z["lat_key"]); C.append(z["lat_cum"].astype(np.int64))
            I.append(z["cell_idx"]); E.append(z["cell_code"]); A.append(z["cell_area"])
    K = np.concatenate(K); C = np.concatenate(C)
    u, inv = np.unique(K, return_inverse=True)
    cum = np.zeros((u.size, len(YEARS)), np.int64)
    np.add.at(cum, inv, C)
    z = np.load(MEASURE_PATH, allow_pickle=False)
    n_cell = z["cell_conv"].size
    cell = np.zeros((n_cell, N_CODE))
    np.add.at(cell, (np.concatenate(I), np.concatenate(E).astype(np.int64)),
              np.concatenate(A))
    np.savez_compressed(OUT_DIR / "lattice_cum_counts.npz", key=u,
                        cum=cum.astype(np.uint16), years=np.array(YEARS))
    np.savez_compressed(OUT_DIR / "cell_epoch_area.npz", area=cell)
    conv = cell[:, 5:9].sum(axis=1)
    chk = dict(files=len(files), lattice_cells=int(u.size),
               cell_conv_maxabs=float(np.abs(conv - z["cell_conv"]).max()),
               cell_conv_total=float(conv.sum()),
               cell_conv_total_orig=float(z["cell_conv"].sum()))
    m = np.load(INTERIM_DIR / "urban_counts_30arcsec.npz")
    r, c = u // MOSAIC_COLS, u % MOSAIC_COLS
    from tropics_config import MOSAIC_LAT_TOP
    inb = r < m["count_final"].shape[0]
    chk["lattice_2020_mismatch"] = int((m["count_final"][r[inb], c[inb]] != cum[inb, 4]).sum())
    chk["lattice_2000_mismatch"] = int((m["count_base"][r[inb], c[inb]] != cum[inb, 0]).sum())
    (OUT_DIR / "assemble_check.json").write_text(json.dumps(chk, indent=1))
    return chk


def period_weights():
    a = np.load(OUT_DIR / "cell_epoch_area.npz")["area"]
    return np.stack([a[:, YEAR_CODE[b]] for _, b in PERIODS], axis=1)


def _tree():
    import shapely
    from shapely import STRtree
    from tropics_config import MOSAIC_LAT_TOP, DELIN_DEG
    z = np.load(MEASURE_PATH, allow_pickle=False)
    lat = MOSAIC_LAT_TOP - (z["cell_row"].astype(np.float64) + 0.5) * DELIN_DEG
    lon = -180.0 + (z["cell_col"].astype(np.float64) + 0.5) * DELIN_DEG
    return STRtree(shapely.points(lon, lat))


def hits_for(species, tree=None):
    import tropics_rangecache as rc
    tree = tree or _tree()
    sp, parts = rc.load_index()
    keep = ~sp.taxon.astype(str).str.upper().isin([t.upper() for t in EXCLUDED_TAXA]).to_numpy()
    want = np.asarray(sorted({int(v) for v in species}), np.int64)
    m = np.isin(parts[0], want)
    parts = (parts[0][m], parts[1][m], parts[2][m])
    hits = {}
    for sp_batch, geom_batch in rc.iter_parts(parts):
        for i, g in zip(sp_batch, geom_batch):
            i = int(i)
            if not keep[i] or g is None or g.is_empty:
                continue
            idx = tree.query(g, predicate="intersects")
            if idx.size:
                hits.setdefault(i, []).append(idx.astype(np.int32))
    S, Cc = [], []
    for i, hs in hits.items():
        u = np.unique(np.concatenate(hs))
        S.append(np.full(u.size, i, np.int32)); Cc.append(u)
    cat = lambda a, t: np.concatenate(a) if a else np.zeros(0, t)
    return cat(S, np.int32), cat(Cc, np.int32)


def all_species():
    import tropics_rangecache as rc
    _sp, parts = rc.load_index()
    return np.unique(np.asarray(parts[0], np.int64)), parts


def hit_ranges(max_parts=1200):
    uniq, parts = all_species()
    _, counts = np.unique(np.asarray(parts[0], np.int64), return_counts=True)
    out, lo, acc = [], 0, 0
    for j, c in enumerate(counts):
        if acc and acc + c > max_parts:
            out.append((lo, j)); lo, acc = j, 0
        acc += c
    out.append((lo, len(counts)))
    return out


def run_hits(budget_s=150.0, max_parts=1200, verbose=True):
    HIT_DIR.mkdir(parents=True, exist_ok=True)
    uniq, _ = all_species()
    tree = _tree()
    t0, n = time.time(), 0
    rngs = hit_ranges(max_parts)
    todo = [(lo, hi) for lo, hi in rngs
            if not (HIT_DIR / f"hits_{lo:06d}_{hi:06d}.npz").exists()]
    for lo, hi in todo:
        s, c = hits_for(uniq[lo:hi], tree)
        tmp = HIT_DIR / f"hits_{lo:06d}_{hi:06d}.part.npz"
        np.savez_compressed(tmp, sp_i=s, cell_i=c, assigned=uniq[lo:hi])
        tmp.rename(HIT_DIR / f"hits_{lo:06d}_{hi:06d}.npz")
        n += 1
        if time.time() - t0 > budget_s:
            break
    left = len(todo) - n
    if verbose:
        print(json.dumps(dict(done_now=n, remaining=left, ranges=len(rngs),
                              seconds=round(time.time() - t0))), flush=True)
    return left


def load_hits():
    S, C = [], []
    for f in sorted(HIT_DIR.glob("hits_*.npz")):
        if f.name.endswith(".part.npz"):
            continue
        with np.load(f) as z:
            S.append(z["sp_i"]); C.append(z["cell_i"])
    return np.concatenate(S), np.concatenate(C)


def pairs_from_hits(sp_i, cell_i, weight, cell_city):
    key = sp_i.astype(np.int64) * (int(cell_city.max()) + 1) + cell_city[cell_i]
    w = weight[cell_i]
    m = w > 0
    u, inv = np.unique(key[m], return_inverse=True)
    v = np.bincount(inv, weights=w[m])
    n = int(cell_city.max()) + 1
    return (u // n).astype(np.int32), (u % n).astype(np.int32), v


PAIR_DIR = OUT_DIR / "pairs"


def period_pairs(sp_i, cell_i, W, cell_city, chunk=4_000_000):
    n = int(cell_city.max()) + 1
    acc = {}
    for s in range(0, sp_i.size, chunk):
        si, ci = sp_i[s:s + chunk], cell_i[s:s + chunk]
        w = W[ci]
        m = w.sum(1) > 0
        key = si[m].astype(np.int64) * n + cell_city[ci[m]]
        u, inv = np.unique(key, return_inverse=True)
        v = np.stack([np.bincount(inv, weights=w[m, j], minlength=u.size)
                      for j in range(W.shape[1])], 1)
        acc[s] = (u, v)
    u = np.concatenate([x[0] for x in acc.values()])
    v = np.concatenate([x[1] for x in acc.values()])
    uu, inv = np.unique(u, return_inverse=True)
    vv = np.stack([np.bincount(inv, weights=v[:, j], minlength=uu.size)
                   for j in range(v.shape[1])], 1)
    return (uu // n).astype(np.int32), (uu % n).astype(np.int32), vv


def run_pairs(budget_s=150.0, verbose=True):
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    z = np.load(MEASURE_PATH, allow_pickle=False)
    cell_city = z["cell_city"].astype(np.int64)
    W = np.column_stack([period_weights(), z["cell_conv"].astype(np.float64)])
    t0, n = time.time(), 0
    files = [f for f in sorted(HIT_DIR.glob("hits_*.npz")) if not f.name.endswith(".part.npz")]
    todo = [f for f in files if not (PAIR_DIR / f.name.replace("hits_", "pairs_")).exists()]
    for f in todo:
        with np.load(f) as q:
            s, c = q["sp_i"], q["cell_i"]
        a, b, v = period_pairs(s, c, W, cell_city)
        out = PAIR_DIR / f.name.replace("hits_", "pairs_")
        tmp = out.with_name(out.stem + ".part.npz")
        np.savez_compressed(tmp, sp_i=a, city=b, conv=v.astype(np.float64))
        tmp.rename(out)
        n += 1
        if time.time() - t0 > budget_s:
            break
    left = len(todo) - n
    if verbose:
        print(json.dumps(dict(done_now=n, remaining=left, files=len(files),
                              seconds=round(time.time() - t0))), flush=True)
    return left


def load_pairs():
    A, B, V = [], [], []
    for f in sorted(PAIR_DIR.glob("pairs_*.npz")):
        if f.name.endswith(".part.npz"):
            continue
        with np.load(f) as q:
            A.append(q["sp_i"]); B.append(q["city"]); V.append(q["conv"])
    return np.concatenate(A), np.concatenate(B), np.concatenate(V)


DELIN_DIR = OUT_DIR / "delineation"


def delineate_year(year, lattice=None):
    import tropics_cities as tc
    from tropics_urban import CELLS_PER_DELIN, MOSAIC_ROWS
    from tropics_config import URBAN_FRACTION, MOSAIC_LAT_TOP, DELIN_DEG
    from tropics_grid import cell_area_km2
    lat = lattice or np.load(OUT_DIR / "lattice_cum_counts.npz")
    key, cum = lat["key"], lat["cum"]
    j = YEARS.index(int(year))
    thr = int(np.ceil(URBAN_FRACTION * CELLS_PER_DELIN))
    k = key[cum[:, j] >= thr]
    r, c = k // MOSAIC_COLS, k % MOSAIC_COLS
    band = tc.tropical_band_mask()
    m = (r < MOSAIC_ROWS) & band[np.minimum(r, MOSAIC_ROWS - 1)]
    urban = np.zeros((MOSAIC_ROWS, MOSAIC_COLS), bool)
    urban[r[m], c[m]] = True
    row_area = cell_area_km2(MOSAIC_LAT_TOP, MOSAIC_ROWS, DELIN_DEG)
    rr, cc, ids, sizes = tc.delineate(urban, row_area=row_area)
    return rr.astype(np.int32), cc.astype(np.int32), ids.astype(np.int32), sizes


def link_to_final(rr, cc, ids, fr, fc, fids):
    fk = fr.astype(np.int64) * MOSAIC_COLS + fc
    o = np.argsort(fk); fk, fids = fk[o], fids[o]
    k = rr.astype(np.int64) * MOSAIC_COLS + cc
    pos = np.minimum(np.searchsorted(fk, k), fk.size - 1)
    hit = fk[pos] == k
    tgt = np.where(hit, fids[pos], 0)
    n = int(ids.max()) + 1 if ids.size else 1
    link = np.zeros(n, np.int32)
    link[ids] = tgt
    split = np.zeros(n, bool)
    np.logical_or.at(split, ids, tgt != link[ids])
    return link, int((~hit).sum()), int(split.sum())


def run_delineation_years(years=None):
    import tropics_cities as tc
    DELIN_DIR.mkdir(parents=True, exist_ok=True)
    fr, fc, fids = tc.load_city_cells()
    lat = dict(np.load(OUT_DIR / "lattice_cum_counts.npz"))
    out = {}
    for y in (years or YEARS):
        f = DELIN_DIR / f"agglomerations_{y}.npz"
        if f.exists():
            continue
        rr, cc, ids, sizes = delineate_year(y, lat)
        link, miss, split = link_to_final(rr, cc, ids, fr, fc, fids)
        np.savez_compressed(f.with_name(f.stem + ".part.npz"), rows=rr, cols=cc, ids=ids,
                            area=sizes.delin_area_km2.to_numpy(), link2020=link[1:])
        f.with_name(f.stem + ".part.npz").rename(f)
        rec = dict(year=int(y), agglomerations=int(len(sizes)),
                   area_km2=float(sizes.delin_area_km2.sum()),
                   cells_outside_2020=miss, split_links=split)
        if int(y) == 2020:
            k1 = np.sort(rr.astype(np.int64) * MOSAIC_COLS + cc)
            k2 = np.sort(fr.astype(np.int64) * MOSAIC_COLS + fc)
            rec["cells_equal_2020"] = bool(np.array_equal(k1, k2))
            rec["ids_equal_2020"] = bool((link[1:] == np.arange(1, len(sizes) + 1)).all())
        out[y] = rec
        print(json.dumps(rec), flush=True)
    return out


def _selftest():
    import tempfile
    from rasterio.transform import from_origin
    tmp = Path(tempfile.mkdtemp())
    res = 0.0002694945852358564
    p = tmp / "GISD30_1985-2020_E10N5.tif"
    arr = np.zeros((120, 120), np.uint8)
    arr[:10, :10] = 4
    arr[:10, 10:20] = 6
    arr[50:60, 50:60] = 8
    arr[0, 0] = 2
    with rasterio.open(p, "w", driver="GTiff", height=120, width=120, count=1,
                       dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(10.0, 5.0, res, res)) as dst:
        dst.write(arr, 1)
    mr, mc = mosaic_index(np.array([5.0 - 0.5 * res]), np.array([10.0 + 0.5 * res]))
    k0 = int(mr[0]) * MOSAIC_COLS + int(mc[0])
    ks = np.array([k0], np.int64)
    r = fold_tile(p, ks, np.array([0], np.int64))
    lk = list(r["lat_key"])
    assert k0 in lk
    row = r["lat_cum"][lk.index(k0)]
    assert row.tolist() == [100, 100, 200, 200, 200], row.tolist()
    codes = dict(zip(r["cell_code"].tolist(), r["cell_area"].tolist()))
    assert set(codes) == {2, 4, 6}, codes
    px = row_area_km2(np.array([5.0]), np.array([5.0 - res]), res)[0]
    assert abs(codes[6] - 100 * px) / codes[6] < 1e-3
    assert abs(codes[4] + codes[2] - 100 * px) / codes[4] < 1e-3
    s = np.array([0, 0, 1, 1, 1], np.int32)
    c = np.array([0, 1, 1, 2, 3], np.int32)
    city = np.array([1, 1, 2, 2], np.int32)
    w = np.array([1.0, 2.0, 4.0, 8.0])
    ps, pc, pv = pairs_from_hits(s, c, w, city)
    got = {(int(a), int(b)): float(v) for a, b, v in zip(ps, pc, pv)}
    assert got == {(0, 1): 3.0, (1, 1): 2.0, (1, 2): 12.0}, got
    print("selftest ok")


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if not a or a[0] == "test":
        _selftest()
    elif a[0] == "tiles":
        run_tiles(float(a[1]) if len(a) > 1 else 150.0)
    elif a[0] == "assemble":
        print(json.dumps(assemble(), indent=1))
    elif a[0] == "hits":
        run_hits(float(a[1]) if len(a) > 1 else 150.0)
    elif a[0] == "delin":
        run_delineation_years([int(v) for v in a[1:]] or None)
    elif a[0] == "pairs":
        run_pairs(float(a[1]) if len(a) > 1 else 150.0)
