"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

R_EARTH_KM = 6371.0088


def blocks(lon, lat, deg=10.0):
    return (np.floor(lat / deg).astype(int) * 1000
            + np.floor(lon / deg).astype(int))


def _unit_xyz(lon, lat):
    la, lo = np.radians(lat), np.radians(lon)
    c = np.cos(la)
    return np.column_stack([c * np.cos(lo), c * np.sin(lo), np.sin(la)])


def _chord(km):
    return 2.0 * np.sin(min(km / R_EARTH_KM, np.pi) / 2.0)


def _near_any(train_xyz, test_xyz, buffer_km):
    if buffer_km <= 0 or not len(test_xyz) or not len(train_xyz):
        return np.zeros(len(train_xyz), bool)
    tree = cKDTree(test_xyz)
    hit = tree.query_ball_point(train_xyz, r=_chord(buffer_km),
                                return_length=True)
    return np.asarray(hit) > 0


def block_cv(d, specs, deg=10.0, k=10, seed=3, buffer_km=0.0,
             common_sample=True, response="log_cost"):
    d = d.copy()
    d["_blk"] = blocks(d.lon.to_numpy(), d.lat.to_numpy(), deg)
    u = np.unique(d._blk)
    rng = np.random.default_rng(seed)
    fold = dict(zip(u, rng.integers(0, k, len(u))))
    d["_fold"] = d._blk.map(fold)
    xyz = _unit_xyz(d.lon.to_numpy(), d.lat.to_numpy())

    acc = {name: [0.0, 0.0, 0] for name in specs}
    dropped_by_buffer = 0
    for f in range(k):
        tr_i = np.flatnonzero((d._fold != f).to_numpy())
        te_i = np.flatnonzero((d._fold == f).to_numpy())
        if len(te_i) < 20 or len(tr_i) < 200:
            continue
        if buffer_km > 0:
            near = _near_any(xyz[tr_i], xyz[te_i], buffer_km)
            dropped_by_buffer += int(near.sum())
            tr_i = tr_i[~near]
            if len(tr_i) < 200:
                continue
        tr, te = d.iloc[tr_i], d.iloc[te_i]

        keep = {}
        for name, formula in specs.items():
            m = np.array(te[response].notna().to_numpy(), dtype=bool)
            for c in ("biome", "country"):
                if f"C({c})" in formula:
                    m &= te[c].isin(set(tr[c])).to_numpy()
            keep[name] = m
        common = np.logical_and.reduce(list(keep.values())) if common_sample \
            else None
        if common_sample and common.sum() < 20:
            continue

        for name, formula in specs.items():
            sel = common if common_sample else keep[name]
            if sel.sum() < 20:
                continue
            t2 = te[sel]
            fit = smf.ols(formula, tr).fit()
            p = np.asarray(fit.predict(t2), float)
            ok = np.isfinite(p) & t2[response].notna().to_numpy()
            y = t2[response].to_numpy()[ok]
            acc[name][0] += float(((y - p[ok]) ** 2).sum())
            acc[name][1] += float(((y - tr[response].mean()) ** 2).sum())
            acc[name][2] += int(ok.sum())

    rows = []
    for name in specs:
        num, den, n = acc[name]
        rows.append(dict(model=name, block_deg=deg, buffer_km=buffer_km,
                         n_blocks=len(u), n_scored=n,
                         R2_oos=(1 - num / den) if den > 0 else np.nan,
                         train_rows_dropped_by_buffer=dropped_by_buffer))
    return pd.DataFrame(rows)


def _synthetic(n=6000, seed=0, patch_deg=3.0):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-170, 170, n)
    lat = rng.uniform(-23, 23, n)
    tile = (np.floor(lat / patch_deg).astype(int) * 10000
            + np.floor(lon / patch_deg).astype(int))
    off = {t: rng.normal(0, 1.4) for t in np.unique(tile)}
    real = rng.normal(0, 1, n)
    y = 1.4 * real + np.array([off[t] for t in tile]) + rng.normal(0, 0.3, n)
    return pd.DataFrame(dict(lon=lon, lat=lat, log_cost=y, real=real,
                             biome="b", country=[f"t{t}" for t in tile]))


def _selftest():
    d = _synthetic()
    specs = {"real": "log_cost ~ real", "patch": "log_cost ~ C(country)"}
    a = block_cv(d, specs, deg=10.0, buffer_km=0.0, common_sample=False)
    b = block_cv(d, specs, deg=10.0, buffer_km=600.0, common_sample=False)
    a_i, b_i = a.set_index("model"), b.set_index("model")

    want = 1.4 ** 2 / (1.4 ** 2 + 1.4 ** 2 + 0.3 ** 2)
    assert abs(a_i.R2_oos["real"] - want) < 0.05, (a_i.R2_oos["real"], want)
    assert abs(b_i.R2_oos["real"] - a_i.R2_oos["real"]) < 0.05, \
        (a_i.R2_oos["real"], b_i.R2_oos["real"])

    assert b_i.n_scored["patch"] < 0.5 * a_i.n_scored["patch"], \
        (a_i.n_scored["patch"], b_i.n_scored["patch"])
    assert b_i.n_scored["real"] == a_i.n_scored["real"], \
        (a_i.n_scored["real"], b_i.n_scored["real"])

    n1 = int(block_cv(d, specs, deg=10.0,
                      buffer_km=300.0).train_rows_dropped_by_buffer.iloc[0])
    n2 = int(block_cv(d, specs, deg=10.0,
                      buffer_km=600.0).train_rows_dropped_by_buffer.iloc[0])
    assert 0 < n1 < n2, (n1, n2)

    c = block_cv(d, specs, deg=10.0, buffer_km=0.0, common_sample=True)
    assert c.n_scored.nunique() == 1, c[["model", "n_scored"]].to_dict("records")

    p = _unit_xyz(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert _near_any(p[:1], p[1:], 120.0)[0]
    assert not _near_any(p[:1], p[1:], 100.0)[0]

    print(f"ok  global covariate {a_i.R2_oos['real']:.3f} -> "
          f"{b_i.R2_oos['real']:.3f} under a 600 km buffer, unchanged; "
          f"local covariate scored on {a_i.n_scored['patch']:,} rows -> "
          f"{b_i.n_scored['patch']:,}; {n1:,} then {n2:,} training rows removed")


if __name__ == "__main__":
    _selftest()
