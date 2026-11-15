"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json, time, shutil
import numpy as np, pandas as pd
import shapely
from shapely import STRtree

from tropics_config import INTERIM_DIR, TABLE_DIR, SETTING_BUFFER_KM
import tropics_rangecache as rc
import tropics_sensitivity as sv
import tropics_scaling as ts

ROOT = INTERIM_DIR / "delin"
CONFIGS = sv.CONFIGS


def _dir(k):
    d = ROOT / f"cfg{k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_cells(k):
    d = _dir(k)
    if (d / "cells.npz").exists():
        return
    cfg = CONFIGS[k]
    cells, cities, _ = sv.config_cities(cfg["fraction"], cfg["link"], cfg["floor"])
    np.savez(d / "cells.npz", lon=cells.lon.to_numpy(), lat=cells.lat.to_numpy(),
             city_id=cells.city_id.to_numpy().astype(np.int32),
             conv=cells.conv.to_numpy(), base=cells.base.to_numpy())
    cities.to_csv(d / "cities.csv", index=False)


def _load(k):
    d = _dir(k)
    z = np.load(d / "cells.npz")
    cells = pd.DataFrame({c: z[c] for c in z.files})
    cities = pd.read_csv(d / "cities.csv")
    return cells, cities


def species_order(parts):
    part_sp, part_off, part_len = parts
    o = np.argsort(part_sp, kind="stable")
    return part_sp[o], part_off[o], part_len[o]


def _species_bounds(part_sp, start, target=1200):
    end = min(start + target, part_sp.size)
    while end < part_sp.size and part_sp[end] == part_sp[end - 1]:
        end += 1
    return end


def step(k, budget_s=150.0, target=1200):
    t0 = time.time()
    stage_cells(k)
    d = _dir(k)
    cells, cities = _load(k)
    sp, part_sp_all = rc.load_index()
    part_sp, part_off, part_len = species_order(part_sp_all)
    n_parts = int(part_sp.size)
    state = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() \
        else dict(next=0, n_parts=n_parts)
    if state["next"] >= n_parts:
        return True, state["next"], n_parts

    lon, lat = cells.lon.to_numpy(), cells.lat.to_numpy()
    tree = STRtree(shapely.points(lon, lat))
    tree_pt = STRtree(shapely.points(cities.lon.to_numpy(), cities.lat.to_numpy()))
    cell_city = cells.city_id.to_numpy()
    cell_conv = cells.conv.to_numpy()
    base_flag = cells.base.to_numpy() > 0
    n_city = int(cities.city_id.max())
    buf = SETTING_BUFFER_KM / 111.32

    while state["next"] < n_parts and time.time() - t0 < budget_s:
        a = state["next"]
        b = _species_bounds(part_sp, a, target)
        hits, near, pool = {}, {}, set()
        for sp_batch, geom_batch in rc.iter_parts(
                (part_sp[a:b], part_off[a:b], part_len[a:b]), batch=400):
            for i, g in zip(sp_batch, geom_batch):
                i = int(i)
                if g is None or g.is_empty:
                    continue
                ix = tree.query(g, predicate="intersects")
                if ix.size:
                    hits.setdefault(i, []).append(ix.astype(np.int32))
                if i not in pool and tree_pt.query(
                        g, predicate="dwithin", distance=buf).size:
                    pool.add(i)
                nr = tree.query(g, predicate="dwithin", distance=buf)
                if nr.size:
                    nr = nr[base_flag[nr]]
                    if nr.size:
                        near.setdefault(i, []).append(
                            np.unique(cell_city[nr]).astype(np.int32))
        ps, pc, pv = [], [], []
        for i, parts in hits.items():
            u = np.unique(np.concatenate(parts))
            agg = np.bincount(cell_city[u], weights=cell_conv[u],
                              minlength=n_city + 1)
            nz = np.flatnonzero(agg)
            if nz.size:
                ps.append(np.full(nz.size, i, np.int32))
                pc.append(nz.astype(np.int32)); pv.append(agg[nz])
        ss, sc = [], []
        for i, parts in near.items():
            u = np.unique(np.concatenate(parts))
            ss.append(np.full(u.size, i, np.int32)); sc.append(u)

        def cat(x, dt):
            return np.concatenate(x) if x else np.zeros(0, dt)
        np.savez(d / f"chunk_{a:07d}.npz",
                 pair_sp=cat(ps, np.int32), pair_city=cat(pc, np.int32),
                 pair_conv=cat(pv, np.float64),
                 set_sp=cat(ss, np.int32), set_city=cat(sc, np.int32),
                 pool_sp=np.asarray(sorted(pool), np.int32))
        state["next"] = int(b)
        (d / "state.json").write_text(json.dumps(state))
        print(f"cfg{k}  parts {b:,}/{n_parts:,}  "
              f"{100*b/n_parts:5.1f}%  {time.time()-t0:5.0f}s", flush=True)
    return state["next"] >= n_parts, state["next"], n_parts


def finish(k):
    d = _dir(k)
    cells, cities = _load(k)
    P, S, Q = [], [], []
    for f in sorted(d.glob("chunk_*.npz")):
        z = np.load(f)
        Q.append(z["pool_sp"])
        P.append(pd.DataFrame(dict(sp_i=z["pair_sp"], city_id=z["pair_city"],
                                   conv_km2=z["pair_conv"])))
        S.append(pd.DataFrame(dict(sp_i=z["set_sp"], city_id=z["set_city"])))
    pairs = pd.concat(P, ignore_index=True)
    setting = pd.concat(S, ignore_index=True).drop_duplicates()
    sp, _ = rc.load_index()
    pool = np.union1d(np.concatenate(Q), pairs.sp_i.unique())
    t, thr, _own = sv.assemble(cities, sp, pairs, setting, pool=pool)
    t = sv.attach_groups(t)
    r = ts.fit_scaling(t, response="cost_per_km2", size="urban_km2_base",
                       setting="setting_richness", spatial=True)
    cfg = CONFIGS[k]
    return dict(label=cfg["label"], fraction=cfg["fraction"], link=cfg["link"],
                floor=cfg["floor"], n_cities=int(len(t)), n_pairs=int(len(pairs)),
                urban_added_km2=float(t.urban_added_km2.sum()),
                cost_km2=float(t.cost_km2.sum()),
                mean_setting=float(t.setting_richness.mean()),
                narrow_threshold_km2=float(thr), beta=float(r["beta"]),
                ci_lower=float(r["ci_lower"]), ci_upper=float(r["ci_upper"]),
                gamma_setting=float(r.get("gamma_setting", np.nan)),
                R2=float(r["R2"]),
                excludes_zero=bool(r["ci_lower"] * r["ci_upper"] > 0)), t


def status():
    out = []
    for k, cfg in enumerate(CONFIGS):
        f = _dir(k) / "state.json"
        s = json.loads(f.read_text()) if f.exists() else dict(next=0, n_parts=0)
        out.append((k, cfg["label"], s["next"], s.get("n_parts", 0)))
    return out


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if not a or a[0] == "status":
        for k, lab, n, tot in status():
            print(f"cfg{k}  {n:>8,}/{tot:<8,}  {lab}")
    elif a[0] == "step":
        k = int(a[1]); budget = float(a[2]) if len(a) > 2 else 150.0
        done, n, tot = step(k, budget)
        print(json.dumps(dict(cfg=k, done=done, next=n, n_parts=tot)))
    elif a[0] == "finish":
        rows = []
        for k in ([int(x) for x in a[1].split(",")] if len(a) > 1
                  else range(len(CONFIGS))):
            r, _ = finish(k)
            rows.append(r); print(json.dumps(r), flush=True)
        pd.DataFrame(rows).to_csv(TABLE_DIR / "sensitivity_delineation_v2.csv",
                                  index=False)
