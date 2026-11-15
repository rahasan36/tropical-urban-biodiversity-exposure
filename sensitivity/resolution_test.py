"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tropics_numbers as tn
import tropics_rangecache as rc
import tropics_refit as trf
from tropics_config import (DELIN_DEG, INTERIM_DIR, MOSAIC_LAT_TOP, TABLE_DIR,
                            THREAT_CATS, NARROW_RANGE_PCTL, EXCLUDED_TAXA)
from tropics_measure import MEASURE_PATH

OUT = TABLE_DIR / "resolution_test_v2.json"
CSV = TABLE_DIR / "resolution_test_v2.csv"

CONFIGS = [(1, "centroid"), (1, "touches"), (1, "within"),
           (2, "centroid"), (4, "centroid"), (2, "touches")]


def cells(factor=1):
    z = np.load(MEASURE_PATH, allow_pickle=False)
    d = pd.DataFrame(dict(row=z["cell_row"].astype(np.int64),
                          col=z["cell_col"].astype(np.int64),
                          city=z["cell_city"].astype(np.int64),
                          conv=z["cell_conv"].astype(float)))
    if factor > 1:
        d["row"] = d.row // factor
        d["col"] = d.col // factor
        owner = (d.groupby(["row", "col", "city"], sort=False).conv.sum()
                  .reset_index()
                  .sort_values("conv", ascending=False)
                  .drop_duplicates(["row", "col"]))
        tot = d.groupby(["row", "col"], sort=False).conv.sum().rename("conv")
        merged = int((d.groupby(["row", "col"], sort=False).city.nunique() > 1).sum())
        d = owner.drop(columns="conv").merge(tot, on=["row", "col"])
    else:
        merged = 0
    deg = DELIN_DEG * factor
    d["lat"] = MOSAIC_LAT_TOP - (d.row + 0.5) * deg
    d["lon"] = -180.0 + (d.col + 0.5) * deg
    return d, deg, merged


def cell_boxes(d, deg):
    half = deg / 2.0
    return shapely.box(d.lon.to_numpy() - half, d.lat.to_numpy() - half,
                       d.lon.to_numpy() + half, d.lat.to_numpy() + half)


def converted(factor, rule, budget_s=None, verbose=True, only_sp=None):
    d, deg, merged = cells(factor)
    if rule == "centroid":
        geom = shapely.points(d.lon.to_numpy(), d.lat.to_numpy())
        predicate = "intersects"
    elif rule == "touches":
        geom = cell_boxes(d, deg)
        predicate = "intersects"
    elif rule == "within":
        geom = cell_boxes(d, deg)
        predicate = "contains"
    else:
        raise ValueError(rule)
    tree = STRtree(geom)
    city = d.city.to_numpy()
    conv = d.conv.to_numpy()
    n_city = int(city.max())

    sp, parts = rc.load_index()
    keep = ~sp.taxon.astype(str).str.upper().isin(
        [t.upper() for t in EXCLUDED_TAXA]).to_numpy()
    if only_sp is not None:
        want = np.asarray(sorted({int(v) for v in only_sp}), np.int64)
        m = np.isin(parts[0], want)
        if not m.any():
            raise ValueError("no cached parts for any of the requested species")
        missing = np.setdiff1d(want, np.unique(parts[0][m]))
        if missing.size:
            raise ValueError(
                f"{missing.size} requested species have no cached part at all "
                f"(first indices {missing[:5].tolist()}); a pass restricted to "
                "them would report zero converted area for a species that has "
                "geometry, which is indistinguishable from a real zero")
        parts = (parts[0][m], parts[1][m], parts[2][m])
        if verbose:
            print(f"    restricted to {want.size} species, "
                  f"{int(m.sum()):,} of {m.size:,} parts", flush=True)

    hits = {}
    t0 = time.time()
    n_seen = 0
    for sp_batch, geom_batch in rc.iter_parts(parts):
        for i, g in zip(sp_batch, geom_batch):
            i = int(i)
            n_seen += 1
            if not keep[i] or g is None or g.is_empty:
                continue
            idx = tree.query(g, predicate=predicate)
            if idx.size:
                hits.setdefault(i, []).append(idx.astype(np.int32))
        if verbose and n_seen % 30000 < 1500:
            print(f"    {n_seen:,} parts  [{time.time()-t0:.0f}s]", flush=True)
        if budget_s and time.time() - t0 > budget_s:
            raise TimeoutError("budget reached before the pass finished")

    ps, pc, pv = [], [], []
    for i, hs in hits.items():
        u = np.unique(np.concatenate(hs))
        agg = np.bincount(city[u], weights=conv[u], minlength=n_city + 1)
        nz = np.flatnonzero(agg)
        if nz.size:
            ps.append(np.full(nz.size, i, np.int32))
            pc.append(nz.astype(np.int32))
            pv.append(agg[nz])
    cat = lambda a, t: np.concatenate(a) if a else np.zeros(0, t)
    pairs = pd.DataFrame(dict(sp_i=cat(ps, np.int32), city_id=cat(pc, np.int32),
                              conv_km2=cat(pv, float)))
    return pairs, sp, dict(deg=deg, n_cells=int(len(d)),
                           cells_merged_across_cities=merged,
                           land_added_km2=float(conv.sum()))


def concern_flags(sp):
    import tropics_flags as tfl
    from tropics_cost import PAIRS_PATH
    cat, tax, _sci, area = tfl.flags(PAIRS_PATH)
    thr = float(json.loads((TABLE_DIR / "cities_cost_meta.json").read_text())
                ["narrow_threshold_km2"])
    z = np.load(PAIRS_PATH, allow_pickle=False)
    order = json.loads(str(z["sp_index"]))
    pool = np.zeros(cat.size, bool)
    pool[np.unique(z["set_sp"])] = True
    pool[np.unique(z["pair_sp"])] = True
    keep = ~np.isin(tax, [t.upper() for t in EXCLUDED_TAXA])
    threat = np.isin(cat, sorted(THREAT_CATS)) & keep
    narrow = pool & keep & (area > 0) & (area <= thr)

    j = np.array([order.get(str(v), -1) for v in sp.sp_id], dtype=np.int64)
    ok = j >= 0
    take = np.where(ok, j, 0)
    out = pd.DataFrame(dict(
        sp_i=sp.sp_i.to_numpy(),
        is_threat=np.where(ok, threat[take], False),
        is_narrow=np.where(ok, narrow[take], False),
        range_km2=np.where(ok, area[take], 0.0)))
    out.attrs["unmatched"] = int((~ok).sum())
    out.attrs["n_cache"] = int(len(sp))
    return out, thr


def preflight(verbose=True):
    sp, _parts = rc.load_index()
    flags, thr = concern_flags(sp)
    n_cache = flags.attrs["n_cache"]
    unmatched = flags.attrs["unmatched"]
    concern = int((flags.is_threat | flags.is_narrow).sum())
    if verbose:
        print(f"  cache species          {n_cache:,}")
        print(f"  unmatched on sp_id     {unmatched:,}")
        print(f"  narrow-range threshold {thr:,.1f} km2")
        print(f"  species of concern     {concern:,} "
              f"(published pool has 10,333 over the full index)", flush=True)
    if unmatched > 0.01 * n_cache:
        raise SystemExit(
            f"{unmatched:,} of {n_cache:,} cached species do not match the pair "
            f"file on sp_id; the flags would be attached to the wrong species.")
    if concern < 8000:
        raise SystemExit(
            f"only {concern:,} species of concern reachable from the cache; "
            f"expected most of the published 10,333.")
    return True


def responses(pairs, sp):
    s, thr = concern_flags(sp)
    assert s.attrs["unmatched"] <= 0.01 * s.attrs["n_cache"]
    p = pairs.merge(s[["sp_i", "is_threat", "is_narrow", "range_km2"]], on="sp_i")
    p = p[(p.is_threat | p.is_narrow) & (p.conv_km2 > 0)]
    out = pd.DataFrame({
        "cost_km2": p.groupby("city_id").conv_km2.sum(),
        "rarity_converted": (p.conv_km2 / p.range_km2.clip(lower=1e-6))
                            .groupby(p.city_id).sum(),
    })
    return out, thr


HEADLINE_BETA = -0.008984


def fit(base, resp):
    x = base.join(resp, on="city_id", rsuffix="_cfg")
    x["cost_km2_cfg"] = x.cost_km2_cfg.fillna(0.0)
    x["rarity_converted_cfg"] = x.rarity_converted_cfg.fillna(0.0)
    x["cost_per_km2"] = x.cost_km2_cfg / x.urban_added_km2
    x["rarity_per_km2"] = x.rarity_converted_cfg / x.urban_added_km2
    r = trf.fit_scaling(x, response="cost_per_km2", size="urban_km2_base",
                        setting="setting_footprint", spatial=True)
    q = trf.fit_scaling(x, response="rarity_per_km2", size="urban_km2_base",
                        setting="setting_footprint", spatial=True)
    return dict(n=int(r["n"]), beta=float(r["beta"]),
                lo=float(r["ci_lower"]), hi=float(r["ci_upper"]),
                p=float(r["p_value"]),
                gamma=float(r.get("gamma_setting", np.nan)),
                beta_rarity=float(q["beta"]),
                rarity_lo=float(q["ci_lower"]), rarity_hi=float(q["ci_upper"]),
                converted_km2=float(x.cost_km2_cfg.sum()),
                land_added_km2=float(x.urban_added_km2.sum()),
                median_intensity=float(x.cost_per_km2.median()))


def _selftest():
    rng = np.random.default_rng(4)
    n = 4000
    d = pd.DataFrame(dict(row=rng.integers(0, 200, n).astype(np.int64),
                          col=rng.integers(0, 200, n).astype(np.int64),
                          city=rng.integers(1, 40, n).astype(np.int64),
                          conv=rng.random(n)))
    d = d.drop_duplicates(["row", "col"])
    fine = d.conv.sum()
    coarse = (d.assign(row=d.row // 2, col=d.col // 2)
               .groupby(["row", "col"]).conv.sum().sum())
    assert abs(fine - coarse) < 1e-9, (fine, coarse)

    deg = 0.01
    lat = 10.0 + np.arange(60) * deg
    lon = 20.0 + np.arange(60) * deg
    LO, LA = np.meshgrid(lon, lat)
    g = pd.DataFrame(dict(lon=LO.ravel(), lat=LA.ravel()))
    boxes = cell_boxes(g, deg)
    pts = shapely.points(g.lon.to_numpy(), g.lat.to_numpy())
    t_box, t_pt = STRtree(boxes), STRtree(pts)
    for r_ in (0.03, 0.12, 0.30):
        poly = shapely.buffer(shapely.points([20.3], [10.3])[0], r_)
        n_within = t_box.query(poly, predicate="contains").size
        assert n_within == int(np.sum(shapely.within(boxes, poly))), (
            "the tree predicate is evaluated query-first; see `converted`")
        n_centre = t_pt.query(poly, predicate="intersects").size
        n_touch = t_box.query(poly, predicate="intersects").size
        assert n_within <= n_centre <= n_touch, (r_, n_within, n_centre, n_touch)
        if r_ == 0.30:
            assert (n_touch - n_within) / max(n_centre, 1) < 0.25, (
                n_within, n_centre, n_touch)
    cache = pd.DataFrame(dict(sp_i=[0, 1, 2],
                              sp_id=["AVES_3", "MAMM_1", "REPT_9"]))
    order = {"MAMM_1": 0, "AVES_3": 1, "REPT_9": 2, "AMPH_7": 3}
    truth = np.array([True, False, False, True])
    j = np.array([order[v] for v in cache.sp_id])
    assert list(truth[j]) == [False, True, False], truth[j]
    assert list(truth[cache.sp_i.to_numpy()]) != [False, True, False], (
        "the positional join must differ here, or this case tests nothing")
    print("ok  aggregation conserves area; the three rules are ordered and "
          "converge as the polygon grows; the species join is by identifier")


def setting_signature(base):
    v = base["setting_footprint"].to_numpy(float)
    return [int(v.size), round(float(v.sum()), 6)]


def run(only=None, verbose=True):
    base = tn.analysis_sample().set_index("city_id")
    sig = setting_signature(base)
    store = json.loads(OUT.read_text()) if OUT.exists() else {}

    import tropics_rangecache as rc
    fp = list(rc.fingerprint())
    stale = [k for k, v in store.items()
             if not isinstance(v, dict) or list(v.get("cache_fp", [])) != fp]
    if stale:
        print(f"discarding {len(stale)} stored configuration(s) measured "
              f"against a different range cache: {', '.join(sorted(stale))}",
              flush=True)
        store = {k: v for k, v in store.items() if k not in stale}

    for factor, rule in CONFIGS:
        key = f"{int(DELIN_DEG * factor * 3600)}:{rule}"
        if only and key != only:
            continue
        if key in store:
            was = list(store[key].get("setting_sig", []))
            if was and was != sig:
                print(f"{key}  already stored, but fitted against a different "
                      f"setting covariate ({was} against {sig}); the rows stay "
                      f"comparable with each other and not with the primary "
                      f"fit", flush=True)
            elif not was:
                print(f"{key}  already stored, covariate vintage unrecorded; "
                      f"re-run to fit it against the current covariate",
                      flush=True)
            else:
                print(f"{key}  already stored", flush=True)
            continue
        t0 = time.time()
        print(f"{key}  running", flush=True)
        pairs, sp, meta = converted(factor, rule, verbose=verbose)
        resp, thr = responses(pairs, sp)
        rec = fit(base, resp)
        rec.update(meta, narrow_threshold_km2=thr, factor=factor, rule=rule,
                   cache_fp=fp, setting_sig=sig,
                   seconds=round(time.time() - t0, 1))
        if key == "30:centroid":
            gap = abs(rec["beta"] - HEADLINE_BETA)
            print(f"    reference check: beta {rec['beta']:+.5f} against "
                  f"published {HEADLINE_BETA:+.5f}, gap {gap:.5f}", flush=True)
            if gap > 0.0015:
                raise SystemExit(
                    f"30:centroid does not reproduce the published exponent "
                    f"(gap {gap:.5f}); the configurations would not be "
                    f"comparable, so nothing is written.")
        store[key] = rec
        OUT.write_text(json.dumps(store, indent=1))
        print(f"{key}  beta {rec['beta']:+.4f} [{rec['lo']:+.4f}, {rec['hi']:+.4f}]"
              f"  gamma {rec['gamma']:+.3f}  converted {rec['converted_km2']:,.0f} km2"
              f"  [{rec['seconds']:.0f}s]", flush=True)
    return store


def report(store):
    rows = []
    for key, r in store.items():
        rows.append(dict(configuration=key, lattice_arcsec=key.split(":")[0],
                         rule=r["rule"], n_cells=r["n_cells"],
                         cells_merged=r["cells_merged_across_cities"],
                         converted_km2=r["converted_km2"],
                         median_intensity=r["median_intensity"],
                         narrow_threshold_km2=r["narrow_threshold_km2"],
                         n=r["n"], beta=r["beta"], lo=r["lo"], hi=r["hi"],
                         p=r["p"], gamma=r["gamma"],
                         beta_rarity=r["beta_rarity"],
                         setting_sig=";".join(str(v) for v
                                              in r.get("setting_sig", []))))
    d = pd.DataFrame(rows)
    if len(d):
        ref = d[d.configuration == "30:centroid"]
        if len(ref):
            d["beta_minus_headline"] = d.beta - float(ref.beta.iloc[0])
            d["converted_vs_headline"] = (d.converted_km2
                                          / float(ref.converted_km2.iloc[0]) - 1.0)
    d.to_csv(CSV, index=False)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", default=None,
                    help='one configuration, e.g. "60:centroid"')
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="check the index join and stop; seconds, no geometry")
    a = ap.parse_args()
    _selftest()
    if a.selftest:
        raise SystemExit(0)
    print("preflight:")
    preflight()
    if a.preflight:
        raise SystemExit(0)
    store = run(only=a.only, verbose=not a.quiet)
    pd.set_option("display.width", 220)
    print()
    print(report(store).to_string(index=False))
    print(f"\nwritten {CSV}")
