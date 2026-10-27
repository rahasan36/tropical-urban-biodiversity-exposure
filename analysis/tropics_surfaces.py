"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import json, time
from pathlib import Path

from tropics_config import (INTERIM_DIR, DELIN_DEG, MOSAIC_LAT_TOP,
                            MOSAIC_LAT_BOTTOM, TROPIC_LAT, THREAT_CATS,
                            EXCLUDED_TAXA,
                            NARROW_RANGE_PCTL)
from tropics_grid import cell_area_km2, row_area_km2
from tropics_urban import (load_counts, MOSAIC_ROWS, MOSAIC_COLS,
                           CELLS_PER_DELIN)

SURF_DIR = INTERIM_DIR / "surfaces"
DISP_DEG = 0.05
RICH_DEG = 0.10
LAT_TOP, LAT_BOT = 25.0, -25.0


def _grid(deg):
    rows = int(round((LAT_TOP - LAT_BOT) / deg))
    cols = int(round(360.0 / deg))
    return rows, cols


def urban_area_surface(deg=DISP_DEG):
    cf, cb = load_counts(which=("count_final", "count_base"))
    native_area = cell_area_km2(MOSAIC_LAT_TOP, MOSAIC_ROWS, DELIN_DEG) / CELLS_PER_DELIN
    rows, cols = _grid(deg)
    k = int(round(deg / DELIN_DEG))
    r0 = int(round((MOSAIC_LAT_TOP - LAT_TOP) / DELIN_DEG))
    r1 = r0 + rows * k
    out = {}
    for name, arr in (("final", cf), ("base", cb)):
        a = arr[r0:r1, :].astype(np.float64)
        a *= native_area[r0:r1, None]
        a = a[: (a.shape[0] // k) * k, : (a.shape[1] // k) * k]
        a = a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).sum(axis=(1, 3))
        out[name] = a.astype(np.float32)
        del a
    del cf, cb
    out["added"] = out["final"] - out["base"]
    return out


def richness_surfaces(deg=RICH_DEG, budget_s=None, verbose=True):
    import rasterio.features as rfeat
    from rasterio.transform import from_origin
    import tropics_rangecache as rc

    SURF_DIR.mkdir(parents=True, exist_ok=True)
    path = SURF_DIR / f"richness_{int(deg*100):03d}.npz"
    rows, cols = _grid(deg)
    sp, parts = rc.load_index()
    sp["category"] = sp.category.astype(str).str.upper()
    keep = ~sp.taxon.astype(str).str.upper().isin(
        [t.upper() for t in EXCLUDED_TAXA]).to_numpy()
    is_threat = sp.category.isin(THREAT_CATS).to_numpy() & keep
    pool = (sp.range_km2.to_numpy() > 0) & keep
    thr = float(np.percentile(sp.range_km2.to_numpy()[pool], NARROW_RANGE_PCTL))
    is_narrow = pool & (sp.range_km2.to_numpy() <= thr)

    if path.exists() and path.stat().st_size:
        z = np.load(path, allow_pickle=False)
        acc = {k: z[k] for k in ("all", "threat", "narrow")}
        acc["rarity"] = z["rarity"] if "rarity" in z.files else \
            np.zeros((rows, cols), np.float32)
        done = int(z["done"])
    else:
        acc = {k: np.zeros((rows, cols), np.int16) for k in ("all", "threat", "narrow")}
        acc["rarity"] = np.zeros((rows, cols), np.float32)
        done = 0

    part_sp, part_off, part_len = parts
    t0 = time.time()
    seen = set()
    n_new = 0
    for start in range(done, part_sp.size, 1500):
        stop = min(start + 1500, part_sp.size)
        sub = (part_sp[start:stop], part_off[start:stop], part_len[start:stop])
        for i, g in zip(*_read_batch(sub)):
            if g is None or g.is_empty:
                continue
            b = g.bounds
            if b[3] < LAT_BOT or b[1] > LAT_TOP:
                continue
            c0 = int(np.floor((max(b[0], -180.0) + 180.0) / deg))
            c1 = int(np.ceil((min(b[2], 180.0) + 180.0) / deg))
            r0 = int(np.floor((LAT_TOP - min(b[3], LAT_TOP)) / deg))
            r1 = int(np.ceil((LAT_TOP - max(b[1], LAT_BOT)) / deg))
            r0, r1 = max(r0, 0), min(max(r1, r0 + 1), rows)
            c0, c1 = max(c0, 0), min(max(c1, c0 + 1), cols)
            if r1 <= r0 or c1 <= c0:
                continue
            tr = from_origin(-180.0 + c0 * deg, LAT_TOP - r0 * deg, deg, deg)
            try:
                m = rfeat.geometry_mask([g], out_shape=(r1 - r0, c1 - c0),
                                        transform=tr, invert=True, all_touched=False)
            except Exception:
                continue
            if not m.any():
                continue
            key = int(i)
            if not keep[key]:
                continue
            acc["all"][r0:r1, c0:c1] += m
            rng = float(sp.range_km2.to_numpy()[key])
            if rng > 0:
                acc["rarity"][r0:r1, c0:c1] += (m / rng).astype(np.float32)
            if is_threat[key]:
                acc["threat"][r0:r1, c0:c1] += m
            if is_narrow[key]:
                acc["narrow"][r0:r1, c0:c1] += m
            n_new += 1
        done = stop
        if verbose:
            print(f"  {done}/{part_sp.size} parts  {time.time()-t0:.0f}s", flush=True)
        if budget_s is not None and time.time() - t0 > budget_s:
            break
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, done=done, narrow_threshold=thr, **acc)
    import os
    os.replace(tmp, path)
    return dict(parts_done=int(done), parts_total=int(part_sp.size),
                species_added_this_run=n_new,
                max_threat=int(acc["threat"].max()),
                max_narrow=int(acc["narrow"].max()),
                max_rarity=float(acc["rarity"].max()),
                narrow_threshold_km2=thr)


def _read_batch(sub):
    import shapely
    import tropics_rangecache as rc
    part_sp, part_off, part_len = sub
    idx, geoms = [], []
    with open(rc.BLOB, "rb") as fh:
        for i, off, ln in zip(part_sp, part_off, part_len):
            fh.seek(int(off))
            wb = fh.read(int(ln))
            try:
                geoms.append(shapely.from_wkb(wb))
                idx.append(i)
            except Exception:
                continue
    return idx, geoms


def land_mask(deg=RICH_DEG):
    import rasterio.features as rfeat
    from rasterio.transform import from_origin
    import tropics_basemap as bm
    path = SURF_DIR / f"landmask_{int(deg*100):03d}.npy"
    if path.exists():
        return np.load(path)
    rows, cols = _grid(deg)
    geoms, _ = bm._load_countries()
    tr = from_origin(-180.0, LAT_TOP, deg, deg)
    m = rfeat.geometry_mask(geoms, out_shape=(rows, cols), transform=tr,
                            invert=True, all_touched=True)
    SURF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, m)
    return m


def load_richness(deg=RICH_DEG, mask_ocean=True):
    z = np.load(SURF_DIR / f"richness_{int(deg*100):03d}.npz", allow_pickle=False)
    keys = ("all", "threat", "narrow") + (("rarity",) if "rarity" in z.files else ())
    out = {k: z[k].astype(np.float32) for k in keys}
    if mask_ocean:
        m = land_mask(deg)
        for k in out:
            out[k][~m] = 0
    return out, float(z["narrow_threshold"])


def save_urban(deg=DISP_DEG):
    SURF_DIR.mkdir(parents=True, exist_ok=True)
    u = urban_area_surface(deg)
    np.savez_compressed(SURF_DIR / f"urban_{int(deg*1000):03d}.npz", **u)
    return {k: (float(v.sum()), int((v > 0).sum())) for k, v in u.items()}


def load_urban(deg=DISP_DEG):
    z = np.load(SURF_DIR / f"urban_{int(deg*1000):03d}.npz", allow_pickle=False)
    return {k: z[k] for k in ("final", "base", "added")}


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "urban":
        r = save_urban()
        print(json.dumps({k: [round(a, 1), n] for k, (a, n) in r.items()}))
    else:
        print(json.dumps(richness_surfaces(budget_s=float(sys.argv[2]) if len(sys.argv) > 2 else 140,
                                           verbose=False)))
