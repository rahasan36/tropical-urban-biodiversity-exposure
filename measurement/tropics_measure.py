"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
import time, json

from tropics_config import (tagged, INTERIM_DIR, DELIN_DEG, MOSAIC_LAT_TOP,
                            GISD30_EPOCHS, BASE_EPOCH, FINAL_EPOCH)
from tropics_grid import row_area_km2
from tropics_urban import (list_tiles, mosaic_index, MOSAIC_COLS, MOSAIC_ROWS,
                           ROW_CHUNK, BASE_CODE, FINAL_CODE, tile_corner)
from tropics_cities import load_city_cells

N_EPOCH = max(GISD30_EPOCHS) + 1
MEASURE_PATH = tagged(INTERIM_DIR / "city_measurements.npz")


def build_cell_lookup():
    rr, cc, ids = load_city_cells()
    key = rr.astype(np.int64) * MOSAIC_COLS + cc
    o = np.argsort(key, kind="stable")
    return key[o], ids[o].astype(np.int32), rr[o].astype(np.int32), cc[o].astype(np.int32)


def measure(tile_dir=None, budget_s=None, verbose=True):
    key, cell_city, cell_row, cell_col = build_cell_lookup()
    n_cell = key.size
    n_city = int(cell_city.max())

    if MEASURE_PATH.exists() and MEASURE_PATH.stat().st_size > 0:
        z = np.load(MEASURE_PATH, allow_pickle=False)
        city_epoch = z["city_epoch"]
        cell_conv = z["cell_conv"]
        cell_base = z["cell_base"] if "cell_base" in z else np.zeros(n_cell)
        cell_final = z["cell_final"] if "cell_final" in z else np.zeros(n_cell)
        cell_epoch_px = z["cell_epoch_px"]
        done = set(z["done"].tolist())
    else:
        city_epoch = np.zeros((n_city + 1, N_EPOCH), np.float64)
        cell_conv = np.zeros(n_cell, np.float64)
        cell_base = np.zeros(n_cell, np.float64)
        cell_final = np.zeros(n_cell, np.float64)
        cell_epoch_px = np.zeros((n_city + 1, N_EPOCH), np.int64)
        done = set()

    tiles = [p for p in list_tiles(*( [tile_dir] if tile_dir else [] ))
             if p.name not in done]
    t0 = time.time()
    for i, p in enumerate(tiles):
        _measure_tile(p, key, cell_city, cell_conv, city_epoch, cell_epoch_px,
                      cell_base=cell_base, cell_final=cell_final)
        done.add(p.name)
        if verbose and (i % 25 == 0):
            print(f"  {len(done)} tiles measured, {time.time()-t0:.0f}s", flush=True)
        if budget_s is not None and time.time() - t0 > budget_s:
            break
    np.savez_compressed(MEASURE_PATH, city_epoch=city_epoch, cell_conv=cell_conv,
                        cell_base=cell_base, cell_final=cell_final,
                        cell_epoch_px=cell_epoch_px, cell_city=cell_city,
                        cell_row=cell_row, cell_col=cell_col,
                        done=np.array(sorted(done)))
    return dict(done=len(done),
                remaining=len(list_tiles()) - len(done),
                urban_km2_final=float(city_epoch[:, :FINAL_CODE + 1].sum()),
                urban_km2_base=float(city_epoch[:, :BASE_CODE + 1].sum()),
                converted_km2=float(cell_conv.sum()))


def _measure_tile(path, key, cell_city, cell_conv, city_epoch, cell_epoch_px,
                  cell_base=None, cell_final=None, row_chunk=ROW_CHUNK):
    n_cell = key.size
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
            vals = arr[rr, cc].astype(np.int64)
            lat_top = T.f + (r0 + rr) * T.e
            lat_bot = lat_top + T.e
            area = row_area_km2(lat_top, lat_bot, T.a)
            lat = lat_top + 0.5 * T.e
            lon = T.c + (cc + 0.5) * T.a
            mr, mc = mosaic_index(lat, lon)
            k = mr.astype(np.int64) * MOSAIC_COLS + mc
            pos = np.searchsorted(key, k)
            np.minimum(pos, n_cell - 1, out=pos)
            inside = key[pos] == k
            if not inside.any():
                continue
            pos, vals, area = pos[inside], vals[inside], area[inside]
            cid = cell_city[pos].astype(np.int64)
            flat = cid * N_EPOCH + vals
            city_epoch.ravel()[:] += np.bincount(
                flat, weights=area, minlength=city_epoch.size)
            cell_epoch_px.ravel()[:] += np.bincount(
                flat, minlength=cell_epoch_px.size)
            new = (vals > BASE_CODE) & (vals <= FINAL_CODE)
            if new.any():
                cell_conv += np.bincount(pos[new], weights=area[new],
                                         minlength=n_cell)
            if cell_base is not None:
                b = vals <= BASE_CODE
                if b.any():
                    cell_base += np.bincount(pos[b], weights=area[b],
                                             minlength=n_cell)
                f = vals <= FINAL_CODE
                if f.any():
                    cell_final += np.bincount(pos[f], weights=area[f],
                                              minlength=n_cell)


def city_table():
    z = np.load(MEASURE_PATH, allow_pickle=False)
    ce = z["city_epoch"]
    ids = np.arange(ce.shape[0])
    base = ce[:, 1:BASE_CODE + 1].sum(axis=1)
    final = ce[:, 1:FINAL_CODE + 1].sum(axis=1)
    df = pd.DataFrame(dict(city_id=ids, urban_km2_base=base,
                           urban_km2_final=final,
                           urban_added_km2=final - base))
    return df[df.city_id > 0].reset_index(drop=True)


if __name__ == "__main__":
    import tempfile
    from rasterio.transform import from_origin

    tmp = Path(tempfile.mkdtemp())
    res = 0.0002694945852358564
    prof = dict(driver="GTiff", height=200, width=200, count=1, dtype="uint8",
                crs="EPSG:4326", transform=from_origin(10.0, 5.0, res, res))
    a = np.zeros((200, 200), np.uint8)
    a[0:10, 0:10] = 3
    a[100:110, 100:110] = 5
    p = tmp / "GISD30_1985-2020_E10N5.tif"
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(a, 1)

    mr, mc = mosaic_index(np.array([5.0 - 5 * res, 5.0 - 105 * res]),
                          np.array([10.0 + 5 * res, 10.0 + 105 * res]))
    key = (mr.astype(np.int64) * MOSAIC_COLS + mc)
    o = np.argsort(key)
    key = key[o]
    cell_city = np.ones(2, np.int32)
    city_epoch = np.zeros((2, N_EPOCH))
    cell_conv = np.zeros(2)
    cell_px = np.zeros((2, N_EPOCH), np.int64)
    _measure_tile(p, key, cell_city, cell_conv, city_epoch, cell_px)

    exp_1995 = float(row_area_km2(5.0, 5.0 - 10 * res, res)) * 10
    exp_2005 = float(row_area_km2(5.0 - 100 * res, 5.0 - 110 * res, res)) * 10
    assert abs(city_epoch[1, 3] - exp_1995) / exp_1995 < 1e-9, (city_epoch[1, 3], exp_1995)
    assert abs(city_epoch[1, 5] - exp_2005) / exp_2005 < 1e-9
    assert cell_px[1, 3] == 100 and cell_px[1, 5] == 100
    print(f"epoch areas              ok  (1995 block {exp_1995*1e6:,.0f} m2, "
          f"2005 block {exp_2005*1e6:,.0f} m2)")

    assert abs(cell_conv.sum() - exp_2005) / exp_2005 < 1e-9
    assert (cell_conv > 0).sum() == 1
    print("converted area           ok  (pre-base area excluded)")

    _measure_tile(p, key, cell_city, cell_conv, city_epoch, cell_px)
    assert abs(city_epoch[1, 3] - 2 * exp_1995) / exp_1995 < 1e-9
    print("additive accumulation    ok")

    b = np.zeros((200, 200), np.uint8); b[50, 50] = 8
    p2 = tmp / "GISD30_1985-2020_E15N5.tif"
    with rasterio.open(p2, "w", **{**prof, "transform": from_origin(15.0, 5.0, res, res)}) as dst:
        dst.write(b, 1)
    before = city_epoch.copy()
    _measure_tile(p2, key, cell_city, cell_conv, city_epoch, cell_px)
    assert np.array_equal(before, city_epoch)
    print("outside-city pixels      ok  (ignored, not charged to any city)")
    print("ALL MEASUREMENT TESTS PASS")
