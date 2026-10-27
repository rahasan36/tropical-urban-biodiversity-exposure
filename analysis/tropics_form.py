"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import os as _os

_os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
_os.environ.setdefault("VSI_CACHE", "TRUE")
_os.environ.setdefault("VSI_CACHE_SIZE", "50000000")
_os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "209715200")
_os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
_os.environ.setdefault("GDAL_HTTP_VERSION", "2")
_os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
_os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
_os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
import os
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.zip")
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "200000000")
os.environ.setdefault("GDAL_INGESTED_BYTES_AT_OPEN", "33554432")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "16777216")

import sys, json, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import label as cc_label, binary_dilation, find_objects
from pathlib import Path

from tropics_config import (INTERIM_DIR, DELIN_DEG, MOSAIC_LAT_TOP, GLC_NATURAL,
                            GLC_FOREST, GLC_SHRUB, GLC_GRASS, GLC_SPARSE,
                            GLC_WETLAND, GLC_CROPLAND, GLC_IMPERVIOUS, GLC_BARE,
                            GLC_WATER, DOWNLOADS)
from tropics_grid import row_area_km2
from tropics_urban import list_tiles, BASE_CODE, FINAL_CODE, tile_corner
from tropics_cities import load_city_cells

FORM_DIR = INTERIM_DIR / "form"
LEI_BUFFER_PX = 4
MAX_PX = 30_000_000
GLC_LOCAL = DOWNLOADS / "GLC_FCS30D_19852022maps_E90-E95"
GLC_ZENODO = "https://zenodo.org/records/8239305/files"

HABITAT_GROUPS = {
    "forest": GLC_FOREST, "shrub": GLC_SHRUB, "grass": GLC_GRASS,
    "sparse": GLC_SPARSE, "wetland": GLC_WETLAND, "cropland": GLC_CROPLAND,
    "impervious": GLC_IMPERVIOUS, "bare": GLC_BARE, "water": GLC_WATER,
}
FORMS = ["infill", "edge", "leapfrog"]


def glc_band_zip(lon_west):
    if lon_west >= 0:
        b = (int(lon_west) // 10) * 10
        return f"E{b}-E{b+5}"
    a = abs(int(lon_west))
    b = ((a + 9) // 10) * 10
    return f"W{b-5}-W{b}"


def glc_path(tile_name, lon_west):
    fn = f"GLC_FCS30D_20002022_{tile_name}_Annual.tif"
    local = GLC_LOCAL / fn
    if local.exists():
        return str(local)
    return f"/vsizip//vsicurl/{GLC_ZENODO}/GLC_FCS30D_19852022maps_{glc_band_zip(lon_west)}.zip/{fn}"


def classify_form(old, new, buffer_px=LEI_BUFFER_PX):
    out = np.zeros(new.shape, np.uint8)
    if not new.any():
        return out
    lab, n = cc_label(new, output=np.int32)
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        r0 = max(sl[0].start - buffer_px, 0); r1 = min(sl[0].stop + buffer_px, new.shape[0])
        c0 = max(sl[1].start - buffer_px, 0); c1 = min(sl[1].stop + buffer_px, new.shape[1])
        patch = lab[r0:r1, c0:c1] == i
        ring = binary_dilation(patch, iterations=buffer_px) & ~patch
        if not ring.any():
            continue
        share = float((old[r0:r1, c0:c1] & ring).sum()) / float(ring.sum())
        out[r0:r1, c0:c1][patch] = 1 if share > 0.5 else (2 if share > 0 else 3)
    return out


def _tile_cities(rr, cc, ids, lon_w, lat_n, transform, shape):
    h, w = shape
    r_top = int(np.floor((MOSAIC_LAT_TOP - lat_n) / DELIN_DEG))
    c_left = int(np.floor((lon_w + 180.0) / DELIN_DEG))
    span = int(round(5.0 / DELIN_DEG))
    m = (rr >= r_top) & (rr < r_top + span) & (cc >= c_left) & (cc < c_left + span)
    if not m.any():
        return {}
    lat = MOSAIC_LAT_TOP - (rr[m] + 0.5) * DELIN_DEG
    lon = -180.0 + (cc[m] + 0.5) * DELIN_DEG
    px = ((lon - transform.c) / transform.a).astype(int)
    py = ((lat - transform.f) / transform.e).astype(int)
    half = int(round(DELIN_DEG / abs(transform.a) / 2)) + 1
    sub_ids, sub_rr, sub_cc = ids[m], rr[m], cc[m]
    out = {}
    for cid in np.unique(sub_ids):
        s = sub_ids == cid
        r0 = max(int(py[s].min()) - half, 0); r1 = min(int(py[s].max()) + half + 1, h)
        c0 = max(int(px[s].min()) - half, 0); c1 = min(int(px[s].max()) + half + 1, w)
        if r1 > r0 and c1 > c0:
            out[int(cid)] = dict(r0=r0, c0=c0, h=r1 - r0, w=c1 - c0,
                                 mrow=sub_rr[s], mcol=sub_cc[s])
    return out


def _cell_membership(mrow, mcol, r0, c0, h, w, transform):
    rows = np.arange(r0, r0 + h)
    cols = np.arange(c0, c0 + w)
    lat = transform.f + (rows + 0.5) * transform.e
    lon = transform.c + (cols + 0.5) * transform.a
    mr = np.floor((MOSAIC_LAT_TOP - lat) / DELIN_DEG).astype(np.int64)
    mc = np.floor((lon + 180.0) / DELIN_DEG).astype(np.int64)
    r_min, r_max = int(mrow.min()), int(mrow.max())
    c_min, c_max = int(mcol.min()), int(mcol.max())
    look = np.zeros((r_max - r_min + 1, c_max - c_min + 1), bool)
    look[mrow - r_min, mcol - c_min] = True
    ri = np.clip(mr - r_min, 0, look.shape[0] - 1)
    ci = np.clip(mc - c_min, 0, look.shape[1] - 1)
    mask = look[ri[:, None], ci[None, :]]
    mask &= ((mr >= r_min) & (mr <= r_max))[:, None]
    mask &= ((mc >= c_min) & (mc <= c_max))[None, :]
    return mask


HAB_WIN = int(os.environ.get("TROPICS_HAB_WIN", 1024))
HAB_MAX_WINDOWS = int(os.environ.get("TROPICS_HAB_WINDOWS", 3))
HAB_MAX_CITIES = int(os.environ.get("TROPICS_HAB_CITIES", 0))


def _habitat_sample(cid, box, src, glc, T, rng):
    r0, c0, h, w = box["r0"], box["c0"], box["h"], box["w"]
    mrow, mcol = box["mrow"], box["mcol"]
    half = int(round(DELIN_DEG / abs(T.a) / 2))
    idx = rng.permutation(len(mrow))[: HAB_MAX_WINDOWS * 4]
    rec = dict(city_id=int(cid), hab_sampled_km2=0.0, hab_windows=0)
    for name in HABITAT_GROUPS:
        rec[f"hab_{name}_km2"] = 0.0
    for code in FORMS:
        rec[f"nat_{code}_km2"] = 0.0
    used = 0
    seen_windows = set()
    for k in idx:
        if used >= HAB_MAX_WINDOWS:
            break
        lat = MOSAIC_LAT_TOP - (float(mrow[k]) + 0.5) * DELIN_DEG
        lon = -180.0 + (float(mcol[k]) + 0.5) * DELIN_DEG
        pr = int((lat - T.f) / T.e); pc = int((lon - T.c) / T.a)
        rr0 = max(min(pr - HAB_WIN // 2, src.height - HAB_WIN), 0)
        cc0 = max(min(pc - HAB_WIN // 2, src.width - HAB_WIN), 0)
        hh = min(HAB_WIN, src.height - rr0); ww = min(HAB_WIN, src.width - cc0)
        if (rr0, cc0) in seen_windows:
            continue
        seen_windows.add((rr0, cc0))
        win = Window(cc0, rr0, ww, hh)
        try:
            a = src.read(1, window=win)
            g = glc.read(1, window=win)
        except Exception:
            continue
        mine = _cell_membership(mrow, mcol, rr0, cc0, hh, ww, T)
        old = (a >= 1) & (a <= BASE_CODE)
        new = (a > BASE_CODE) & (a <= FINAL_CODE) & mine
        if not new.any():
            continue
        form = classify_form(old, new)
        rows = np.arange(hh)
        lat_top = T.f + rr0 * T.e
        area = np.repeat(row_area_km2(lat_top + rows * T.e,
                                      lat_top + (rows + 1) * T.e, T.a)[:, None],
                         ww, axis=1)
        hh2, ww2 = min(g.shape[0], hh), min(g.shape[1], ww)
        gg, nn = g[:hh2, :ww2], new[:hh2, :ww2]
        ff, aa = form[:hh2, :ww2], area[:hh2, :ww2]
        rec["hab_sampled_km2"] += float(aa[nn].sum())
        for name, codes in HABITAT_GROUPS.items():
            rec[f"hab_{name}_km2"] += float(aa[np.isin(gg, codes) & nn].sum())
        for j, code in enumerate(FORMS, start=1):
            rec[f"nat_{code}_km2"] += float(aa[np.isin(gg, GLC_NATURAL)
                                              & (ff == j) & nn].sum())
        used += 1
    rec["hab_windows"] = used
    return rec if used else None


def _one_city(cid, box, src, glc, T, tname):
    r0, c0, h, w = box["r0"], box["c0"], box["h"], box["w"]
    step = max(1, MAX_PX // max(w, 1))
    recs = []
    for rs in range(0, h, step):
        hh = min(step, h - rs)
        win = Window(c0, r0 + rs, w, hh)
        a = src.read(1, window=win)
        mine = _cell_membership(box["mrow"], box["mcol"], r0 + rs, c0, hh, w, T)
        old = (a >= 1) & (a <= BASE_CODE)
        new = (a > BASE_CODE) & (a <= FINAL_CODE) & mine
        if not new.any():
            continue
        form = classify_form(old, new)
        lat_top = T.f + (r0 + rs) * T.e
        rows = np.arange(hh)
        area = np.repeat(row_area_km2(lat_top + rows * T.e,
                                      lat_top + (rows + 1) * T.e, T.a)[:, None],
                         w, axis=1)
        rec = dict(city_id=int(cid), tile=tname)
        for k, code in enumerate(FORMS, start=1):
            rec[f"form_{code}_km2"] = float(area[form == k].sum())
        if glc is not None:
            g = None
            for attempt in range(3):
                try:
                    g = glc.read(1, window=win)
                    break
                except Exception as exc:
                    rec["glc_read_error"] = type(exc).__name__
                    time.sleep(2 * (attempt + 1))
            if g is not None:
                hh2, ww2 = min(g.shape[0], hh), min(g.shape[1], w)
                gg, nn = g[:hh2, :ww2], new[:hh2, :ww2]
                ff, aa = form[:hh2, :ww2], area[:hh2, :ww2]
                for name, codes in HABITAT_GROUPS.items():
                    rec[f"hab_{name}_km2"] = float(aa[np.isin(gg, codes) & nn].sum())
                for k, code in enumerate(FORMS, start=1):
                    sel = np.isin(gg, GLC_NATURAL) & (ff == k) & nn
                    rec[f"nat_{code}_km2"] = float(aa[sel].sum())
        recs.append(rec)
    return recs


def process_tile(path, rr, cc, ids, want_habitat=True):
    lon_w, lat_n = tile_corner(path)
    tname = Path(path).stem.split("_")[-1]
    recs = []
    with rasterio.open(path) as src:
        cities = _tile_cities(rr, cc, ids, lon_w, lat_n, src.transform, src.shape)
        if not cities:
            return recs
        glc, err = None, ""
        if want_habitat:
            for attempt in range(6):
                try:
                    glc = rasterio.open(glc_path(tname, lon_w))
                    break
                except Exception as exc:
                    err = f"{type(exc).__name__}: {str(exc)[-80:]}"
                    time.sleep(3 * (attempt + 1))
            if glc is None:
                recs.append(dict(city_id=-1, tile=tname, error=f"glc open: {err}"))
        rng = np.random.default_rng(20260829)
        if HAB_MAX_CITIES and glc is not None and len(cities) > HAB_MAX_CITIES:
            keep = sorted(cities, key=lambda c: -len(cities[c]["mrow"]))[:HAB_MAX_CITIES]
            cities = {c: cities[c] for c in keep}
        try:
            for cid, box in cities.items():
                try:
                    if glc is None:
                        recs.extend(_one_city(cid, box, src, None,
                                              src.transform, tname))
                    else:
                        r = _habitat_sample(cid, box, src, glc, src.transform, rng)
                        if r is not None:
                            r["tile"] = tname
                            recs.append(r)
                except Exception as exc:
                    recs.append(dict(city_id=-1, tile=tname,
                                     error=f"city {cid}: {type(exc).__name__}: "
                                           f"{str(exc)[:80]}"))
        finally:
            if glc is not None:
                glc.close()
    return recs


def failed_habitat_tiles():
    got, failed = set(), set()
    for f in FORM_DIR.glob("recs*_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r.get("city_id", -1) > 0 and r.get("hab_sampled_km2", 0) > 0:
                got.add(r.get("tile"))
            elif r.get("city_id", -1) == -1 and "glc open" in str(r.get("error", "")):
                failed.add(r.get("tile"))
    order = _tile_priority()
    want = (failed - got)
    return [p for p in order if p.stem.split("_")[-1] in want]


def run(shard=0, nshard=1, budget_s=150, want_habitat=True, tag="", verbose=False,
        tiles=None):
    FORM_DIR.mkdir(parents=True, exist_ok=True)
    done_path = FORM_DIR / f"done{tag}_{shard}_{nshard}.json"
    out_path = FORM_DIR / f"recs{tag}_{shard}_{nshard}.jsonl"
    done = (set(json.loads(done_path.read_text()))
            if done_path.exists() and done_path.stat().st_size else set())
    rr, cc, ids = load_city_cells()
    order = _tile_priority()
    mine = tiles if tiles is not None else [p for i, p in enumerate(order)
                                            if i % nshard == shard]
    todo = [p for p in mine if p.name not in done]
    t0 = time.time()
    n_rec = 0
    with open(out_path, "a") as fh:
        for p in todo:
            if time.time() - t0 > budget_s:
                break
            try:
                recs = process_tile(p, rr, cc, ids, want_habitat=want_habitat)
            except Exception as exc:
                recs = [dict(city_id=-1, tile=p.stem.split("_")[-1],
                             error=f"tile: {type(exc).__name__}: {str(exc)[:80]}")]
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            n_rec += sum(1 for r in recs if r.get("city_id", -1) > 0)
            done.add(p.name)
            fh.flush()
            done_path.write_text(json.dumps(sorted(done)))
            if verbose:
                print(f"  {p.name} {len(recs)}  {time.time()-t0:.0f}s", flush=True)
    return dict(shard=shard, done=len(done), total=len(mine), records=n_rec)


def _tile_priority():
    import numpy as np
    from tropics_measure import MEASURE_PATH
    z = np.load(MEASURE_PATH, allow_pickle=False)
    lat = MOSAIC_LAT_TOP - (z["cell_row"].astype(float) + 0.5) * DELIN_DEG
    lon = -180.0 + (z["cell_col"].astype(float) + 0.5) * DELIN_DEG
    conv = z["cell_conv"]
    key_lat = np.floor(lat / 5.0) * 5.0 + 5.0
    key_lon = np.floor(lon / 5.0) * 5.0
    weight = {}
    for la, lo, v in zip(key_lat, key_lon, conv):
        weight[(float(lo), float(la))] = weight.get((float(lo), float(la)), 0.0) + float(v)
    tiles = list_tiles()
    def w(p):
        lo, la = tile_corner(p)
        return weight.get((float(lo), float(la)), 0.0)
    return sorted(tiles, key=w, reverse=True)


def coverage(pattern="recs_*.jsonl"):
    import pandas as pd
    from tropics_config import TABLE_DIR
    d = collect(pattern)
    hab_cols = [c for c in d.columns if c.startswith("hab_")]
    if not hab_cols:
        return dict(cities=0, share=0.0)
    df = pd.read_csv(TABLE_DIR / "cities_cost.csv")
    got = d[d[hab_cols].sum(axis=1) > 0].city_id
    return dict(cities=int(got.size),
                share=float(df[df.city_id.isin(set(got))].urban_added_km2.sum()
                            / df.urban_added_km2.sum()))


def collect(pattern="recs*_*.jsonl"):
    rows = []
    for f in sorted(FORM_DIR.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            if r.get("city_id", -1) > 0:
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop(columns=["tile", "glc_read_error"], errors="ignore")
    return df.groupby("city_id").sum(numeric_only=True).reset_index()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        old = np.zeros((60, 60), bool); new = np.zeros((60, 60), bool)
        old[10:30, 10:30] = True
        old[18:22, 18:22] = False; new[18:22, 18:22] = True
        new[10:30, 30:34] = True
        new[50:54, 50:54] = True
        f = classify_form(old, new)
        assert f[19, 19] == 1 and f[20, 31] == 2 and f[52, 52] == 3, f[19, 19]
        print("expansion form           ok  (infill / edge / leapfrog recovered)")
        assert glc_band_zip(90) == "E90-E95" and glc_band_zip(95) == "E90-E95"
        assert glc_band_zip(-45) == "W45-W50" and glc_band_zip(-50) == "W45-W50"
        assert glc_band_zip(0) == "E0-E5" and glc_band_zip(-5) == "W5-W10"
        print("GLC archive naming       ok")
        print("ALL FORM TESTS PASS")
    elif sys.argv[1] == "retry":
        budget = float(sys.argv[2]) if len(sys.argv) > 2 else 150
        tl = failed_habitat_tiles()
        print(json.dumps(dict(to_retry=len(tl))), flush=True)
        print(json.dumps(run(0, 1, budget, want_habitat=True, tag="R", tiles=tl)))
    else:
        shard, nshard = int(sys.argv[1]), int(sys.argv[2])
        budget = float(sys.argv[3]) if len(sys.argv) > 3 else 150
        hab = (len(sys.argv) <= 4 or sys.argv[4] != "noglc")
        tag = "" if hab else "F"
        print(json.dumps(run(shard, nshard, budget, want_habitat=hab, tag=tag)))


def collect_habitat(pattern="recs_*_4.jsonl"):
    import pandas as pd
    from tropics_config import TABLE_DIR
    rows = []
    for f in sorted(FORM_DIR.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            if r.get("city_id", -1) > 0 and r.get("hab_sampled_km2", 0) > 0:
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    d = (pd.DataFrame(rows).drop(columns=["tile"], errors="ignore")
           .groupby("city_id").sum(numeric_only=True).reset_index())
    df = pd.read_csv(TABLE_DIR / "cities_cost.csv")[["city_id", "urban_added_km2"]]
    d = d.merge(df, on="city_id", how="left")
    hab = [c for c in d.columns if c.startswith("hab_")
           and c not in ("hab_sampled_km2", "hab_windows")]
    scale = d.urban_added_km2 / d.hab_sampled_km2.replace(0, np.nan)
    for c in hab:
        d[c.replace("_km2", "_share")] = d[c] / d.hab_sampled_km2
        d[c + "_scaled"] = d[c] * scale
    for c in [f"nat_{f}_km2" for f in FORMS]:
        if c in d.columns:
            d[c + "_scaled"] = d[c] * scale
    d["hab_sample_frac"] = d.hab_sampled_km2 / d.urban_added_km2
    return d.drop(columns=["urban_added_km2"])


def write_city_tables():
    f = collect("recsF_*.jsonl")
    f = f[["city_id"] + [c for c in f.columns if c.startswith("form_")]]
    f.to_csv(INTERIM_DIR / "city_form.csv", index=False)
    h = collect_habitat("recs*_*.jsonl")
    h.to_csv(INTERIM_DIR / "city_habitat.csv", index=False)
    return dict(form_cities=int(len(f)), habitat_cities=int(len(h)))
