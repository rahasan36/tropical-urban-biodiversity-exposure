"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resolution_test as rt
import tropics_numbers as tn
import tropics_rangecache as rc
from tropics_config import INTERIM_DIR, TABLE_DIR

PART_DIR = INTERIM_DIR / "restest"
N_SLICES = 8
KEYS = [f"{int(rt.DELIN_DEG * f * 3600)}:{r}" for f, r in rt.CONFIGS]


def _config(key):
    for f, r in rt.CONFIGS:
        if f"{int(rt.DELIN_DEG * f * 3600)}:{r}" == key:
            return f, r
    raise SystemExit(f"unknown configuration {key}; expected one of {KEYS}")


def species_slices(n=N_SLICES):
    _sp, parts = rc.load_index()
    sp_i = np.asarray(parts[0], np.int64)
    uniq, counts = np.unique(sp_i, return_counts=True)
    edges = np.searchsorted(np.cumsum(counts),
                            np.linspace(0, counts.sum(), n + 1)[1:])
    out, lo = [], 0
    for e in edges:
        e = int(min(max(e + 1, lo + 1), uniq.size))
        out.append(uniq[lo:e])
        lo = e
    out = [g for g in out if g.size]
    assert sum(g.size for g in out) == uniq.size, "partition lost species"
    assert len({int(v) for g in out for v in g}) == uniq.size, "groups overlap"
    return out


def slice_path(key, k, n):
    return PART_DIR / key.replace(":", "_") / f"slice_{k:02d}_of_{n:02d}.npz"


def all_species():
    _sp, parts = rc.load_index()
    return np.unique(np.asarray(parts[0], np.int64))


def covered(key):
    got = []
    for p in sorted((PART_DIR / key.replace(":", "_")).glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            got.append(z["assigned"])
    return np.unique(np.concatenate(got)) if got else np.zeros(0, np.int64)


def run_range(key, lo, hi):
    factor, rule = _config(key)
    uniq = all_species()
    lo, hi = int(lo), int(min(hi, uniq.size))
    g = uniq[lo:hi]
    if not g.size:
        raise SystemExit(f"empty range {lo}:{hi}")
    p = PART_DIR / key.replace(":", "_") / f"range_{lo:06d}_{hi:06d}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs, _sp, meta = rt.converted(factor, rule, verbose=False, only_sp=g)
    np.savez(p, sp_i=pairs.sp_i.to_numpy(), city_id=pairs.city_id.to_numpy(),
             conv_km2=pairs.conv_km2.to_numpy(), assigned=g,
             meta=np.array(json.dumps(meta)))
    print(f"{key} range {lo}:{hi}: {g.size:,} species, {len(pairs):,} pairs, "
          f"{pairs.conv_km2.sum():,.0f} km2 [{time.time() - t0:.0f}s] -> {p.name}",
          flush=True)


def run_slice(key, k, n=N_SLICES):
    factor, rule = _config(key)
    groups = species_slices(n)
    if not 0 <= k < len(groups):
        raise SystemExit(f"slice {k} out of range for {len(groups)} groups")
    g = groups[k]
    p = slice_path(key, k, len(groups))
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs, _sp, meta = rt.converted(factor, rule, verbose=False, only_sp=g)
    np.savez(p, sp_i=pairs.sp_i.to_numpy(), city_id=pairs.city_id.to_numpy(),
             conv_km2=pairs.conv_km2.to_numpy(), assigned=g,
             meta=np.array(json.dumps(meta)))
    print(f"{key} slice {k}/{len(groups)}: {g.size:,} species, "
          f"{len(pairs):,} pairs, {pairs.conv_km2.sum():,.0f} km2 "
          f"[{time.time() - t0:.0f}s] -> {p.name}", flush=True)


def load_slices(key, n=None):
    frames, assigned, meta = [], [], None
    for p in sorted((PART_DIR / key.replace(":", "_")).glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            frames.append(pd.DataFrame(dict(sp_i=z["sp_i"], city_id=z["city_id"],
                                            conv_km2=z["conv_km2"])))
            assigned.append(z["assigned"])
            meta = json.loads(str(z["meta"]))
    if not frames:
        raise SystemExit(f"{key}: nothing measured yet")
    want = all_species()
    got = np.concatenate(assigned)
    if np.unique(got).size != got.size:
        raise SystemExit(f"{key}: a species was measured twice; the pieces "
                         "overlap and the total would be too large")
    missing = np.setdiff1d(want, got)
    if missing.size:
        raise SystemExit(f"{key}: {missing.size:,} species not measured yet "
                         f"(first at position "
                         f"{int(np.searchsorted(want, missing[0]))})")
    extra = np.setdiff1d(got, want)
    if extra.size:
        raise SystemExit(f"{key}: {extra.size:,} measured species are not in "
                         "the cache index")
    pairs = pd.concat(frames, ignore_index=True)
    d = pairs.duplicated(["sp_i", "city_id"])
    assert not d.any(), f"{int(d.sum())} duplicated species-city rows"
    return pairs, meta


def finish(key, n=N_SLICES):
    pairs, meta = load_slices(key, n)
    sp, _parts = rc.load_index()
    resp, thr = rt.responses(pairs, sp)
    base = tn.analysis_sample().set_index("city_id")
    sig = rt.setting_signature(base)
    rec = rt.fit(base, resp)
    factor, rule = _config(key)
    fp = list(rc.fingerprint())
    store = json.loads(rt.OUT.read_text()) if rt.OUT.exists() else {}
    prev = store.get(key, {})
    if "converted_km2" in prev:
        rel = abs(rec["converted_km2"] / prev["converted_km2"] - 1.0)
        print(f"  converted total {rec['converted_km2']:,.1f} km2 against the "
              f"stored {prev['converted_km2']:,.1f}, {rel:.2e} relative")
        if rel > 1e-9:
            raise SystemExit(
                f"{key}: slicing changed the converted total by {rel:.2e}; the "
                "geometry does not depend on the covariate, so the partition "
                "is wrong and the fit must not be stored")
    rec.update(meta, narrow_threshold_km2=thr, factor=factor, rule=rule,
               cache_fp=fp, setting_sig=sig, seconds=None, resumed=True)
    store[key] = rec
    rt.OUT.write_text(json.dumps(store, indent=1))
    d = rt.report(store)
    print(f"{key}  beta {rec['beta']:+.6f} [{rec['lo']:+.4f}, {rec['hi']:+.4f}]"
          f"  gamma {rec['gamma']:+.4f}  "
          f"converted {rec['converted_km2']:,.0f} km2", flush=True)
    return d


def advance(key, budget_s=80.0, chunk=400, floor=50):
    t0 = time.time()
    want = all_species()
    while True:
        have = covered(key)
        left = np.setdiff1d(want, have)
        if not left.size:
            print(f"{key}: all {want.size:,} species measured", flush=True)
            return True
        lo = int(np.searchsorted(want, left[0]))
        spent = time.time() - t0
        if spent > budget_s:
            print(f"{key}: {have.size:,}/{want.size:,} measured, "
                  f"next position {lo:,}", flush=True)
            return False
        n = int(min(chunk, want.size - lo))
        t1 = time.time()
        run_range(key, lo, lo + n)
        took = time.time() - t1
        if took > 0.45 * budget_s:
            chunk = max(floor, chunk // 2)
        elif took < 0.12 * budget_s:
            chunk = int(chunk * 1.8)


def status(n=None):
    want = all_species()
    store = json.loads(rt.OUT.read_text()) if rt.OUT.exists() else {}
    sig = rt.setting_signature(tn.analysis_sample().set_index("city_id"))
    for key in KEYS:
        have = covered(key)
        nxt = np.setdiff1d(want, have)
        pos = int(np.searchsorted(want, nxt[0])) if nxt.size else want.size
        s = store.get(key, {})
        vin = ("current" if list(s.get("setting_sig", [])) == sig
               else ("stale" if s else "absent"))
        print(f"  {key:14s} {have.size:>6,}/{want.size:,} species measured, "
              f"next position {pos:>6,}   stored fit: {vin}")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "status"
    if cmd == "plan":
        n = int(a[1]) if len(a) > 1 else N_SLICES
        for i, g in enumerate(species_slices(n)):
            print(f"  slice {i}: {g.size:,} species, indices "
                  f"{int(g[0])}..{int(g[-1])}")
    elif cmd == "slice":
        run_slice(a[1], int(a[2]), int(a[3]) if len(a) > 3 else N_SLICES)
    elif cmd == "range":
        run_range(a[1], int(a[2]), int(a[3]))
    elif cmd == "advance":
        advance(a[1], float(a[2]) if len(a) > 2 else 80.0,
                int(a[3]) if len(a) > 3 else 400)
    elif cmd == "finish":
        finish(a[1], int(a[2]) if len(a) > 2 else N_SLICES)
    elif cmd == "status":
        status(int(a[1]) if len(a) > 1 else N_SLICES)
    else:
        raise SystemExit(__doc__)
