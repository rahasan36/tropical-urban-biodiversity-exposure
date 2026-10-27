"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyogrio, shapely, json, time, os, warnings
from shapely import wkb as shp_wkb
from shapely.geometry import box
from pyproj import Geod
from pathlib import Path

from tropics_config import (TROPIC_LAT, SETTING_BUFFER_KM, PROJECT)
from tropics_cost import _sources, _read_chunk, _keep, CHUNK

GEOD = Geod(ellps="WGS84")
CACHE_DIR = PROJECT / "data" / "interim" / "rangecache"
BLOB = CACHE_DIR / "ranges.bin"
INDEX = CACHE_DIR / "ranges_idx.jsonl"
STATE = CACHE_DIR / "state.json"

SIMPLIFY_DEG = 0.002
BAND_MARGIN = SETTING_BUFFER_KM / 111.32
CLIP_BOX = box(-180.0, -TROPIC_LAT - BAND_MARGIN, 180.0, TROPIC_LAT + BAND_MARGIN)
STATE_EVERY = 40


MAX_COORDS = 20_000


def _prepare(g):
    b = shapely.bounds(g)
    if b[3] < CLIP_BOX.bounds[1] or b[1] > CLIP_BOX.bounds[3]:
        return None
    if shapely.get_num_coordinates(g) > MAX_COORDS:
        g = g.simplify(SIMPLIFY_DEG, preserve_topology=False)
    gi = g.intersection(CLIP_BOX)
    if gi.is_empty:
        return None
    gi = gi.simplify(SIMPLIFY_DEG, preserve_topology=True)
    return None if gi.is_empty else gi


def build(budget_s=140, verbose=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    state = (json.loads(STATE.read_text())
             if STATE.exists() and STATE.stat().st_size else {})
    tot_path = CACHE_DIR / "totals.json"
    totals = json.loads(tot_path.read_text()) if tot_path.exists() else {}
    blob = open(BLOB, "ab")
    index = open(INDEX, "a")
    offset = BLOB.stat().st_size if BLOB.exists() else 0

    t0 = time.time()
    stop = False
    written = 0
    for s in _sources():
        if stop:
            break
        key = f"{Path(s['path']).name}:{s['layer'] or ''}"
        if key not in totals:
            lk = {"layer": s["layer"]} if s["layer"] else {}
            totals[key] = int(pyogrio.read_info(s["path"], **lk)["features"])
            tot_path.write_text(json.dumps(totals))
        total = totals[key]
        off = state.get(key, 0)
        while off < total:
            df = _read_chunk(s, off, CHUNK)
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
                fi = off + int(pos[j])
                if g is None or g.is_empty:
                    continue
                b = shapely.bounds(g)
                if b[3] < CLIP_BOX.bounds[1] or b[1] > CLIP_BOX.bounds[3]:
                    continue
                area = 0.0
                gi = _prepare(g)
                n = 0
                if gi is not None:
                    wb = shapely.to_wkb(gi)
                    blob.write(wb)
                    n = len(wb)
                index.write(json.dumps(dict(sp=sp, src=key, fi=fi, off=offset,
                                            len=n, area=area,
                                            meta=[nm, cat, s["taxon"]])) + "\n")
                offset += n
                written += 1
            off += n_read
            blob.flush(); index.flush()
            state[key] = off
            STATE.write_text(json.dumps(state))
            if verbose:
                print(f"  {key} {off}/{total} {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > budget_s:
                stop = True
                break
    blob.close(); index.close()

    remaining = sum(max(0, totals.get(f"{Path(s['path']).name}:{s['layer'] or ''}", 0)
                        - state.get(f"{Path(s['path']).name}:{s['layer'] or ''}", 0))
                    for s in _sources())
    return dict(features_remaining=int(remaining), features_written=written,
                blob_mb=round(BLOB.stat().st_size / 1e6, 1))


def global_areas():
    import numpy as _np
    from tropics_cost import PAIRS_PATH
    z = _np.load(PAIRS_PATH, allow_pickle=False)
    idx = json.loads(str(z["sp_index"]))
    area = z["sp_area"]
    return {sp: float(area[i]) for sp, i in idx.items() if i < area.size}


def load_index():
    rows = [json.loads(l) for l in open(INDEX)]

    seen = set()
    sp_index, sp_area, sp_meta = {}, [], []
    for r in rows:
        fid = (r["src"], r["fi"])
        if fid in seen:
            continue
        seen.add(fid)
        i = sp_index.get(r["sp"])
        if i is None:
            i = len(sp_index); sp_index[r["sp"]] = i
            sp_area.append(0.0); sp_meta.append(r["meta"] or ["", "", ""])
        sp_area[i] += r["area"]

    repaired = {(r["src"], r["fi"]) for r in rows if r.get("repair")}
    parts_rows = [r for r in rows
                  if r.get("repair") or (r["src"], r["fi"]) not in repaired]
    seen = set()
    part_sp, part_off, part_len = [], [], []
    for r in parts_rows:
        fid = (r["src"], r["fi"])
        if fid in seen:
            continue
        seen.add(fid)
        if r["len"]:
            part_sp.append(sp_index[r["sp"]])
            part_off.append(r["off"]); part_len.append(r["len"])
    ga = global_areas()
    ids = list(sp_index.keys())
    sp = pd.DataFrame(dict(
        sp_i=np.arange(len(ids)), sp_id=ids,
        range_km2=[ga.get(k, a) for k, a in zip(ids, sp_area)],
        sci_name=[m[0] for m in sp_meta], category=[m[1] for m in sp_meta],
        taxon=[m[2] for m in sp_meta]))
    return sp, (np.asarray(part_sp, np.int32), np.asarray(part_off, np.int64),
                np.asarray(part_len, np.int64))


def fingerprint():
    return (int(INDEX.stat().st_size) if INDEX.exists() else -1,
            int(BLOB.stat().st_size) if BLOB.exists() else -1)


def iter_parts(parts, batch=1500):
    iter_parts.bad = getattr(iter_parts, "bad", 0)
    start_bad = iter_parts.bad
    part_sp, part_off, part_len = parts
    with open(BLOB, "rb") as fh:
        for i in range(0, part_sp.size, batch):
            j = min(i + batch, part_sp.size)
            wkbs = []
            for k in range(i, j):
                fh.seek(int(part_off[k]))
                wkbs.append(fh.read(int(part_len[k])))
            keep_i, geoms = [], []
            for k, wb in zip(range(i, j), wkbs):
                try:
                    g = shapely.from_wkb(wb)
                except Exception:
                    iter_parts.bad += 1
                    continue
                keep_i.append(part_sp[k]); geoms.append(g)
            if geoms:
                yield np.asarray(keep_i, np.int32), np.asarray(geoms, dtype=object)
    lost = iter_parts.bad - start_bad
    if lost:
        warnings.warn(
            f"{lost:,} of {part_sp.size:,} cached range parts would not parse "
            "and were skipped. Every species whose only parts are among them "
            "contributes no converted area to this pass, silently. Run "
            "_cache_parse_check.py to see which species and what they are "
            "worth before using the result.",
            RuntimeWarning, stacklevel=2)


iter_parts_bad = 0


def load_parts():
    rows = [json.loads(l) for l in open(INDEX)]
    blob = open(BLOB, "rb").read()
    seen = set()
    sp_index, sp_area, sp_meta = {}, [], []
    part_sp, part_wkb = [], []
    for r in rows:
        fid = (r["src"], r["fi"])
        if fid in seen:
            continue
        seen.add(fid)
        i = sp_index.get(r["sp"])
        if i is None:
            i = len(sp_index); sp_index[r["sp"]] = i
            sp_area.append(0.0); sp_meta.append(r["meta"] or ["", "", ""])
        sp_area[i] += r["area"]
        if r["len"]:
            part_sp.append(i)
            part_wkb.append(blob[r["off"]:r["off"] + r["len"]])
    geoms = shapely.from_wkb(part_wkb) if part_wkb else np.array([], dtype=object)
    ga = global_areas()
    ids = list(sp_index.keys())
    sp = pd.DataFrame(dict(
        sp_i=np.arange(len(ids)), sp_id=ids,
        range_km2=[ga.get(k, a) for k, a in zip(ids, sp_area)],
        sci_name=[m[0] for m in sp_meta], category=[m[1] for m in sp_meta],
        taxon=[m[2] for m in sp_meta]))
    return sp, np.asarray(part_sp, np.int32), geoms


def load(with_geometry=True):
    rows = [json.loads(l) for l in open(INDEX)]
    blob = open(BLOB, "rb").read() if with_geometry else b""
    seen = set()
    by_sp = {}
    for r in rows:
        fid = (r["src"], r["fi"])
        if fid in seen:
            continue
        seen.add(fid)
        d = by_sp.setdefault(r["sp"], dict(area=0.0, parts=[], meta=r["meta"]))
        d["area"] += r["area"]
        if with_geometry and r["len"]:
            d["parts"].append(shp_wkb.loads(blob[r["off"]:r["off"] + r["len"]]))
    sp_ids, areas, names, cats, taxa, geoms = [], [], [], [], [], []
    for sp, d in by_sp.items():
        sp_ids.append(sp); areas.append(d["area"])
        m = d["meta"] or ["", "", ""]
        names.append(m[0]); cats.append(m[1]); taxa.append(m[2])
        if with_geometry:
            geoms.append(shapely.union_all(d["parts"]) if d["parts"] else None)
    ga = global_areas()
    areas = [ga.get(s_, a) for s_, a in zip(sp_ids, areas)]
    sp = pd.DataFrame(dict(sp_id=sp_ids, range_km2=areas, sci_name=names,
                           category=cats, taxon=taxa))
    sp["sp_i"] = np.arange(len(sp))
    return sp, (np.array(geoms, dtype=object) if with_geometry else None)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sp, geoms = load()
        n = int(sum(g is not None for g in geoms))
        print(f"species {len(sp):,}  with tropical geometry {n:,}")
        print(sp.taxon.value_counts().to_string())
        print(f"median global range {sp.range_km2.median():,.0f} km2")
    else:
        print(json.dumps(build(budget_s=float(sys.argv[1]) if len(sys.argv) > 1 else 140)))
