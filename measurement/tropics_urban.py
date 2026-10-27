"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import re, time, sys

from tropics_config import (GISD30_TILE_DIR, INTERIM_DIR, DELIN_DEG,
                            MOSAIC_LAT_TOP, MOSAIC_LAT_BOTTOM,
                            GISD30_EPOCHS, BASE_EPOCH, FINAL_EPOCH)

TILE_RE = re.compile(r'_([EW])(\d+)([NS])(\d+)\.tif$')
ROW_CHUNK = 3000

MOSAIC_ROWS = int(round((MOSAIC_LAT_TOP - MOSAIC_LAT_BOTTOM) / DELIN_DEG))
MOSAIC_COLS = int(round(360.0 / DELIN_DEG))
MOSAIC_PATH = INTERIM_DIR / "urban_counts_30arcsec.npz"
CELLS_PER_DELIN = int(round((DELIN_DEG / 0.0002694945852358564) ** 2))

EPOCH_OF = {v: k for k, v in GISD30_EPOCHS.items()}
BASE_CODE  = EPOCH_OF[BASE_EPOCH]
FINAL_CODE = EPOCH_OF[FINAL_EPOCH]


def tile_corner(path):
    m = TILE_RE.search(str(path))
    if not m:
        raise ValueError(f"not a GISD30 tile name: {path}")
    ew, lon, ns, lat = m.groups()
    lon = int(lon) * (1 if ew == "E" else -1)
    lat = int(lat) * (1 if ns == "N" else -1)
    return lon, lat


def list_tiles(tile_dir=GISD30_TILE_DIR):
    return sorted(Path(tile_dir).glob("*.tif"))


def mosaic_index(lat, lon):
    row = np.floor((MOSAIC_LAT_TOP - lat) / DELIN_DEG).astype(np.int32)
    col = np.floor((lon + 180.0) / DELIN_DEG).astype(np.int32)
    return row, col


def mosaic_transform():
    from rasterio.transform import from_origin
    return from_origin(-180.0, MOSAIC_LAT_TOP, DELIN_DEG, DELIN_DEG)


def check_tile(path):
    lon_w, lat_n = tile_corner(path)
    try:
        src = rasterio.open(path)
    except Exception as exc:
        return False, f"unreadable: {type(exc).__name__}"
    with src:
        T = src.transform
        if T.a == 1.0 and T.e == 1.0:
            return False, "no geotransform"
        if abs(T.c - lon_w) > 0.01 or abs(T.f - lat_n) > 0.01:
            return False, f"transform {T.c:.4f},{T.f:.4f} != name {lon_w},{lat_n}"
        if abs(T.a - 0.000269494585) > 1e-9:
            return False, f"unexpected resolution {T.a}"
    return True, "ok"


TILE_DELIN = int(round(5.0 / DELIN_DEG)) + 2


def accumulate_tile(count_final, count_base, path, row_chunk=ROW_CHUNK):
    lon_w, lat_n = tile_corner(path)
    mr0, mc0 = mosaic_index(np.array([lat_n]), np.array([lon_w]))
    mr0, mc0 = int(mr0[0]), int(mc0[0])
    loc_f = np.zeros(TILE_DELIN * TILE_DELIN, np.int32)
    loc_b = np.zeros(TILE_DELIN * TILE_DELIN, np.int32)
    with rasterio.open(path) as src:
        T = src.transform
        H, W = src.height, src.width
        for r0 in range(0, H, row_chunk):
            h = min(row_chunk, H - r0)
            arr = src.read(1, window=Window(0, r0, W, h))
            hit = arr >= 1
            if not hit.any():
                continue
            rr, cc = np.nonzero(hit)
            vals = arr[rr, cc]
            lat = T.f + (r0 + rr + 0.5) * T.e
            lon = T.c + (cc + 0.5) * T.a
            mr, mc = mosaic_index(lat, lon)
            lr, lc = mr - mr0, mc - mc0
            ok = (lr >= 0) & (lr < TILE_DELIN) & (lc >= 0) & (lc < TILE_DELIN)
            flat = (lr[ok] * TILE_DELIN + lc[ok]).astype(np.int64)
            v = vals[ok]
            loc_f += np.bincount(flat[v <= FINAL_CODE], minlength=loc_f.size).astype(np.int32)
            loc_b += np.bincount(flat[v <= BASE_CODE], minlength=loc_b.size).astype(np.int32)
    nz = np.flatnonzero(loc_f)
    if nz.size == 0:
        return
    gr = mr0 + nz // TILE_DELIN
    gc = mc0 + nz % TILE_DELIN
    keep = (gr >= 0) & (gr < MOSAIC_ROWS) & (gc >= 0) & (gc < MOSAIC_COLS)
    gr, gc, nz = gr[keep], gc[keep], nz[keep]
    np.add.at(count_final, (gr, gc), loc_f[nz].astype(np.uint16))
    np.add.at(count_base, (gr, gc), loc_b[nz].astype(np.uint16))


def build_mosaic(tile_dir=GISD30_TILE_DIR, out_path=MOSAIC_PATH, budget_s=None,
                 verbose=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        z = np.load(out_path, allow_pickle=False)
        count_final = z["count_final"]
        count_base = z["count_base"]
        done = set(z["done"].tolist())
    else:
        count_final = np.zeros((MOSAIC_ROWS, MOSAIC_COLS), np.uint16)
        count_base = np.zeros((MOSAIC_ROWS, MOSAIC_COLS), np.uint16)
        done = set()

    tiles = [p for p in list_tiles(tile_dir) if p.name not in done]
    t0 = time.time()
    bad = []
    for i, p in enumerate(tiles):
        ok, why = check_tile(p)
        if not ok:
            bad.append((p.name, why))
            continue
        try:
            accumulate_tile(count_final, count_base, p)
        except Exception as exc:
            bad.append((p.name, f"read failed: {type(exc).__name__}"))
            continue
        done.add(p.name)
        if verbose and (i % 10 == 0 or i == len(tiles) - 1):
            print(f"  {len(done)} tiles folded, {time.time()-t0:.0f}s", flush=True)
        if budget_s is not None and time.time() - t0 > budget_s:
            break
    np.savez_compressed(out_path, count_final=count_final, count_base=count_base,
                        done=np.array(sorted(done)))
    remaining = len(list_tiles(tile_dir)) - len(done)
    return dict(done=len(done), remaining=remaining, bad=bad,
                impervious_cells_final=int(count_final.sum(dtype=np.int64)),
                impervious_cells_base=int(count_base.sum(dtype=np.int64)))


def load_counts(path=MOSAIC_PATH, which=("count_final", "count_base")):
    z = np.load(path, allow_pickle=False)
    out = tuple(z[k] for k in which)
    return out[0] if len(out) == 1 else out


def urban_mask(counts, fraction):
    return counts >= int(np.ceil(fraction * CELLS_PER_DELIN))


if __name__ == "__main__":
    import numpy.testing as npt

    assert tile_corner("GISD30_1985-2020_E100N0.tif") == (100, 0)
    assert tile_corner("GISD30_1985-2020_W65S15.tif") == (-65, -15)
    print("tile name parsing        ok")

    r, c = mosaic_index(np.array([25.0, 0.0, -25.0 + 1e-9]),
                        np.array([-180.0, 0.0, 179.999]))
    npt.assert_array_equal(r, [0, 3000, 5999])
    npt.assert_array_equal(c, [0, 21600, 43199])
    print("mosaic indexing          ok  "
          f"(grid {MOSAIC_ROWS} x {MOSAIC_COLS})")

    import tempfile, os
    from rasterio.transform import from_origin
    tmp = Path(tempfile.mkdtemp())
    prof = dict(driver="GTiff", height=200, width=200, count=1, dtype="uint8",
                crs="EPSG:4326", transform=from_origin(10.0, 5.0, 0.0002694945852358564,
                                                       0.0002694945852358564))
    a = np.zeros((200, 200), np.uint8)
    a[10, 10] = 8
    a[150, 150] = 3
    p = tmp / "GISD30_1985-2020_E10N5.tif"
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(a, 1)
    uf = np.zeros((MOSAIC_ROWS, MOSAIC_COLS), np.uint16)
    ub = np.zeros((MOSAIC_ROWS, MOSAIC_COLS), np.uint16)
    accumulate_tile(uf, ub, p)
    assert uf.sum() == 2, uf.sum()
    assert ub.sum() == 1, ub.sum()
    assert (uf > 0).sum() == 2 and (ub > 0).sum() == 1
    rr, cc = np.nonzero(uf)
    lat = MOSAIC_LAT_TOP - (rr + 0.5) * DELIN_DEG
    lon = -180.0 + (cc + 0.5) * DELIN_DEG
    assert np.all((lat > 4.9) & (lat < 5.0)) and np.all((lon > 10.0) & (lon < 10.1)), (lat, lon)
    assert ub.sum() == 1 and uf[np.nonzero(ub)].all()
    print("synthetic tile fold      ok  (2 cells final, 1 cell base, correct place)")

    a2 = np.zeros((200, 200), np.uint8); a2[0:31, 0:31] = 2
    p2 = tmp / "GISD30_1985-2020_E20N5.tif"
    with rasterio.open(p2, "w", **{**prof, "transform": from_origin(
            20.0, 5.0, 0.0002694945852358564, 0.0002694945852358564)}) as dst:
        dst.write(a2, 1)
    cf = np.zeros((MOSAIC_ROWS, MOSAIC_COLS), np.uint16)
    cb = np.zeros((MOSAIC_ROWS, MOSAIC_COLS), np.uint16)
    accumulate_tile(cf, cb, p2)
    assert cf.sum() == 31 * 31, cf.sum()
    assert cf.max() >= 900, cf.max()
    assert urban_mask(cf, 0.5).sum() == 1, sorted(cf[cf > 0].tolist())[-4:]
    assert urban_mask(cf, 1.01).sum() == 0
    print(f"count fold + fraction    ok  ({CELLS_PER_DELIN} native cells per "
          f"30 arcsec cell, max count {int(cf.max())})")
    print("ALL URBAN-MODULE TESTS PASS")
