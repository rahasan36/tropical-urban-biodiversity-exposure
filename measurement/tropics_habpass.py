"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json, time
from pathlib import Path

import tropics_form as tf


def _tile_state():
    hab_tiles, cities = set(), set()
    for f in sorted(tf.FORM_DIR.glob("recs*_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("city_id", -1) > 0 and r.get("hab_sampled_km2", 0) > 0:
                hab_tiles.add(r.get("tile"))
                cities.add(int(r["city_id"]))
    return hab_tiles, cities


def outstanding():
    hab_tiles, _ = _tile_state()
    return [p for p in tf._tile_priority()
            if p.stem.split("_")[-1] not in hab_tiles]


def status():
    hab_tiles, cities = _tile_state()
    return dict(tiles_done=len(hab_tiles), tiles_left=len(outstanding()),
                cities=len(cities))


def run(budget_s=150, tag="H2", verbose=True, worker=0, nworker=1):
    tf.FORM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = tf.FORM_DIR / f"recs{tag}w{worker}_0_1.jsonl"
    fail_path = tf.FORM_DIR / f"fail{tag}w{worker}.json"
    fails = json.loads(fail_path.read_text()) if fail_path.exists() else {}
    rr, cc, ids = tf.load_city_cells()
    todo = [p for p in outstanding() if fails.get(p.stem.split("_")[-1], 0) < 3]
    todo = todo[worker::nworker]
    t0 = time.time()
    done, n_rec, n_err = [], 0, 0
    with open(out_path, "a") as fh:
        for p in todo:
            if time.time() - t0 > budget_s:
                break
            tname = p.stem.split("_")[-1]
            try:
                recs = tf.process_tile(p, rr, cc, ids, want_habitat=True)
            except Exception as exc:
                recs = [dict(city_id=-1, tile=tname,
                             error=f"tile: {type(exc).__name__}: {str(exc)[:80]}")]
            good = sum(1 for r in recs if r.get("city_id", -1) > 0
                       and r.get("hab_sampled_km2", 0) > 0)
            if good == 0:
                fails[tname] = fails.get(tname, 0) + 1
                n_err += 1
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            fail_path.write_text(json.dumps(fails))
            n_rec += good
            done.append(tname)
            if verbose:
                print(f"  {tname:8s} {good:4d} cities  {time.time()-t0:6.0f}s",
                      flush=True)
    return dict(worker=worker, tiles=len(done), cities_measured=n_rec,
                tiles_empty=n_err, elapsed=round(time.time() - t0, 1))


def _selftest():
    hab, _ = _tile_state()
    out = {p.stem.split("_")[-1] for p in outstanding()}
    assert not (hab & out), "a measured tile is listed as outstanding"
    assert len(hab) + len(out) == len(tf.list_tiles()), "tiles unaccounted for"
    print("selftest ok:", len(hab), "measured,", len(out), "outstanding")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        budget = float(sys.argv[2]); w = int(sys.argv[3]); n = int(sys.argv[4])
        print(run(budget_s=budget, worker=w, nworker=n))
    else:
        _selftest()


def measured_cities():
    import os as _o
    p = _o.environ.get("TROPICS_WANTED")
    if p:
        want = set(json.load(open(p)))
        rr, cc, ids = tf.load_city_cells()
        import numpy as _np
        return set(_np.unique(ids).tolist()) - want
    got = set()
    for f in sorted(tf.FORM_DIR.glob("recs*_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("city_id", -1) > 0 and r.get("hab_sampled_km2", 0) > 0:
                got.add(int(r["city_id"]))
    return got


def wanted_cells():
    import numpy as np
    rr, cc, ids = tf.load_city_cells()
    have = measured_cities()
    keep = ~np.isin(ids, np.fromiter(have, ids.dtype, len(have)))
    return rr[keep], cc[keep], ids[keep]


def _tile_key(rr, cc):
    import numpy as np
    from tropics_config import DELIN_DEG, MOSAIC_LAT_TOP
    lat = MOSAIC_LAT_TOP - (rr.astype(float) + 0.5) * DELIN_DEG
    lon = -180.0 + (cc.astype(float) + 0.5) * DELIN_DEG
    return (np.floor(lon / 5.0) * 5.0, np.floor(lat / 5.0) * 5.0 + 5.0)


def outstanding_cities():
    rr, cc, _ = wanted_cells()
    lo, la = _tile_key(rr, cc)
    need = set(zip(lo.tolist(), la.tolist()))
    return [p for p in tf._tile_priority() if tf.tile_corner(p) in need]


def _process_for(path, rr, cc, ids):
    import numpy as np, rasterio, time as _t
    lon_w, lat_n = tf.tile_corner(path)
    tname = path.stem.split("_")[-1]
    recs = []
    with rasterio.open(path) as src:
        cities = tf._tile_cities(rr, cc, ids, lon_w, lat_n,
                                 src.transform, src.shape)
        if not cities:
            return recs
        glc, err = None, ""
        for attempt in range(4):
            try:
                glc = rasterio.open(tf.glc_path(tname, lon_w))
                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {str(exc)[-60:]}"
                _t.sleep(3 * (attempt + 1))
        if glc is None:
            return [dict(city_id=-1, tile=tname, error=f"glc open: {err}")]
        rng = np.random.default_rng(20260902)
        try:
            for cid, box in cities.items():
                try:
                    r = tf._habitat_sample(cid, box, src, glc, src.transform, rng)
                    if r is not None:
                        r["tile"] = tname
                        recs.append(r)
                except Exception as exc:
                    recs.append(dict(city_id=-1, tile=tname,
                                     error=f"city {cid}: {type(exc).__name__}: "
                                           f"{str(exc)[:60]}"))
        finally:
            glc.close()
    return recs


def run_cities(budget_s=7200, tag="H3", worker=0, nworker=1, verbose=True):
    tf.FORM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = tf.FORM_DIR / f"recs{tag}w{worker}_0_1.jsonl"
    rr, cc, ids = wanted_cells()
    todo = outstanding_cities()[worker::nworker]
    t0 = time.time()
    n_rec, n_tile = 0, 0
    with open(out_path, "a") as fh:
        for p in todo:
            if time.time() - t0 > budget_s:
                break
            try:
                recs = _process_for(p, rr, cc, ids)
            except Exception as exc:
                recs = [dict(city_id=-1, tile=p.stem.split("_")[-1],
                             error=f"tile: {type(exc).__name__}: {str(exc)[:60]}")]
            good = sum(1 for r in recs if r.get("city_id", -1) > 0)
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            n_rec += good
            n_tile += 1
            if verbose:
                print(f"  {p.stem.split('_')[-1]:8s} {good:4d} cities  "
                      f"{time.time()-t0:6.0f}s", flush=True)
    return dict(worker=worker, tiles=n_tile, cities_measured=n_rec,
                elapsed=round(time.time() - t0, 1))


def _selftest_cities():
    import numpy as np
    have = measured_cities()
    rr, cc, ids = wanted_cells()
    assert not (set(np.unique(ids).tolist()) & have), "a measured city is wanted"
    lo, la = _tile_key(rr, cc)
    need = set(zip(lo.tolist(), la.tolist()))
    tiles = {tf.tile_corner(p) for p in outstanding_cities()}
    assert need <= tiles | (need - {tf.tile_corner(p) for p in tf.list_tiles()}), \
        "a wanted cell has no tile in the outstanding list"
    print(f"selftest ok: {len(have):,} measured, {len(np.unique(ids)):,} wanted, "
          f"{len(tiles)} tiles to open")
