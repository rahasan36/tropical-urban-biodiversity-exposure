"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import label as cc_label, binary_dilation
from pathlib import Path
import time, json

from tropics_config import (tagged, TROPIC_LAT, DELIN_DEG, LINK_CELLS, MIN_CITY_KM2,
                            URBAN_FRACTION,
                            INTERIM_DIR, GISD30_EPOCHS, BASE_EPOCH, FINAL_EPOCH,
                            MOSAIC_LAT_TOP)
from tropics_grid import cell_area_km2, row_area_km2
from tropics_urban import (load_counts, urban_mask, list_tiles, MOSAIC_ROWS,
                           MOSAIC_COLS, mosaic_index, ROW_CHUNK, BASE_CODE,
                           FINAL_CODE, CELLS_PER_DELIN)

CITY_ID_PATH   = tagged(INTERIM_DIR / "city_id_30arcsec.npz")
CITY_CELL_PATH = tagged(INTERIM_DIR / "city_cells.npz")


def tropical_band_mask(rows=MOSAIC_ROWS, lat_top=MOSAIC_LAT_TOP,
                       cell=DELIN_DEG, tropic_lat=TROPIC_LAT):
    centres = lat_top - (np.arange(rows) + 0.5) * cell
    return np.abs(centres) <= tropic_lat


def delineate(urban, link_cells=LINK_CELLS, min_km2=MIN_CITY_KM2, row_area=None):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_rows, n_cols = urban.shape
    rr, cc = np.nonzero(urban)
    n = rr.size
    if n == 0:
        return rr, cc, np.zeros(0, np.int32), pd.DataFrame(
            columns=["city_id", "n_cells", "delin_area_km2"])
    key = rr.astype(np.int64) * n_cols + cc
    order = np.argsort(key, kind="stable")
    rr, cc, key = rr[order], cc[order], key[order]

    d = int(link_cells) + 1

    comp = np.arange(n, dtype=np.int32)
    ncomp = n
    for dr in range(0, d + 1):
        for dc in range(-d, d + 1):
            if dr == 0 and dc <= 0:
                continue
            r2, c2 = rr + dr, cc + dc
            ok = (r2 < n_rows) & (c2 >= 0) & (c2 < n_cols)
            k2 = r2[ok].astype(np.int64) * n_cols + c2[ok]
            pos = np.searchsorted(key, k2)
            np.minimum(pos, n - 1, out=pos)
            hit = key[pos] == k2
            if not hit.any():
                continue
            a = comp[np.flatnonzero(ok)[hit]]
            b = comp[pos[hit]]
            m = a != b
            if not m.any():
                continue
            a, b = a[m], b[m]
            g = coo_matrix((np.ones(a.size, np.uint8), (a, b)),
                           shape=(ncomp, ncomp))
            k, lab = connected_components(g, directed=False)
            comp = lab[comp].astype(np.int32)
            ncomp = k
            del g, lab, a, b

    if row_area is None:
        row_area = np.ones(n_rows)
    areas = np.bincount(comp, weights=row_area[rr], minlength=ncomp)
    counts = np.bincount(comp, minlength=ncomp)
    keep = np.flatnonzero(areas >= min_km2)
    order2 = keep[np.argsort(-areas[keep])]
    remap = np.zeros(ncomp, np.int32)
    remap[order2] = np.arange(1, len(order2) + 1, dtype=np.int32)
    city_id = remap[comp]
    sizes = pd.DataFrame(dict(city_id=np.arange(1, len(order2) + 1),
                              n_cells=counts[order2],
                              delin_area_km2=areas[order2]))
    m = city_id > 0
    return rr[m], cc[m], city_id[m], sizes


def run_delineation(save=True, link_cells=LINK_CELLS, min_km2=MIN_CITY_KM2,
                    fraction=URBAN_FRACTION, counts=None):
    counts = load_counts(which=("count_final",)) if counts is None else counts
    urban_final = urban_mask(counts, fraction)
    urban_final &= tropical_band_mask()[:, None]
    row_area = cell_area_km2(MOSAIC_LAT_TOP, MOSAIC_ROWS, DELIN_DEG)
    rr, cc, ids, sizes = delineate(urban_final, link_cells=link_cells,
                                   min_km2=min_km2, row_area=row_area)
    del urban_final
    if save:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CITY_ID_PATH, rows=rr.astype(np.int32),
                            cols=cc.astype(np.int32), ids=ids.astype(np.int32))
        sizes.to_csv(tagged(INTERIM_DIR / "city_sizes.csv"), index=False)
    return rr, cc, ids, sizes


def load_city_cells():
    z = np.load(CITY_ID_PATH)
    return z["rows"], z["cols"], z["ids"]


if __name__ == "__main__":
    u = np.zeros((40, 60), bool)
    u[5:9, 5:9] = True
    u[5:9, 12:16] = True
    u[25:27, 30:50] = True
    area = np.ones(40)

    rr, cc, ids, sizes = delineate(u, link_cells=2, min_km2=1, row_area=area)
    assert len(sizes) == 3, sizes
    assert sizes.n_cells.tolist() == [40, 16, 16], sizes
    print("delineation, gap > link  ok  (3 cities, largest first)")

    rr, cc, ids, sizes = delineate(u, link_cells=3, min_km2=1, row_area=area)
    assert len(sizes) == 2, sizes
    assert sizes.n_cells.tolist() == [40, 32], sizes
    print("delineation, gap <= link ok  (A and B merge at the wider link)")

    assert ids.size == int(u.sum()) == 72
    assert np.all(u[rr, cc]) and len(set(zip(rr.tolist(), cc.tolist()))) == 72
    print("cities are disjoint      ok  (one id per urban cell, none elsewhere)")

    area2 = np.ones(40); area2[25:27] = 0.01
    rr, cc, ids, sizes = delineate(u, link_cells=2, min_km2=1, row_area=area2)
    assert sizes.n_cells.tolist() == [16, 16], sizes
    print("area-based size floor    ok  (area-poor strip dropped, cell-rich as it is)")

    v = np.zeros((20, 20), bool); v[5, 5] = True; v[7, 7] = True
    _, _, _, sz = delineate(v, link_cells=1, min_km2=0.5, row_area=np.ones(20))
    assert len(sz) == 1, sz
    _, _, _, sz = delineate(v, link_cells=0, min_km2=0.5, row_area=np.ones(20))
    assert len(sz) == 2, sz
    print("diagonal linking         ok")

    band = tropical_band_mask()
    lat_hi = MOSAIC_LAT_TOP - (np.flatnonzero(band)[0] + 0.5) * DELIN_DEG
    lat_lo = MOSAIC_LAT_TOP - (np.flatnonzero(band)[-1] + 0.5) * DELIN_DEG
    assert lat_hi <= TROPIC_LAT and lat_lo >= -TROPIC_LAT
    assert abs(lat_hi - TROPIC_LAT) < DELIN_DEG and abs(lat_lo + TROPIC_LAT) < DELIN_DEG
    print(f"tropical band mask       ok  ({band.sum()} rows, "
          f"{lat_lo:.4f} to {lat_hi:.4f} deg)")
    print("ALL DELINEATION TESTS PASS")
