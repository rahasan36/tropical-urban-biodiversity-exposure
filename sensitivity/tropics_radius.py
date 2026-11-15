"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json, time
import numpy as np, pandas as pd
import shapely
from shapely import STRtree

from tropics_config import INTERIM_DIR, TABLE_DIR, EXCLUDED_TAXA
import tropics_rangecache as rc
import tropics_delin as dl
import tropics_numbers as tn
from tropics_scaling import fit_scaling

RADII_KM = (25.0, 50.0, 100.0)
ROOT = INTERIM_DIR / "radius"
THREAT = ("CR", "EN", "VU")


def _dir():
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT


def step(budget_s=90.0, target=1200):
    t0 = time.time()
    dl.stage_cells(0)
    d = _dir()
    cells, cities = dl._load(0)
    sp, parts = rc.load_index()
    part_sp, part_off, part_len = parts
    n_parts = int(part_sp.size)
    f = d / "state.json"
    state = json.loads(f.read_text()) if f.exists() else dict(next=0, n_parts=n_parts)
    if state["next"] >= n_parts:
        return True, state["next"], n_parts

    tree = STRtree(shapely.points(cells.lon.to_numpy(), cells.lat.to_numpy()))
    cell_city = cells.city_id.to_numpy()
    base_flag = cells.base.to_numpy() > 0
    bufs = [r / 111.32 for r in RADII_KM]

    while state["next"] < n_parts and time.time() - t0 < budget_s:
        a = state["next"]
        b = dl._species_bounds(part_sp, a, target)
        near = {r: {} for r in RADII_KM}
        for sp_batch, geom_batch in rc.iter_parts(
                (part_sp[a:b], part_off[a:b], part_len[a:b]), batch=400):
            for i, g in zip(sp_batch, geom_batch):
                i = int(i)
                if g is None or g.is_empty:
                    continue
                for r, bf in zip(RADII_KM, bufs):
                    nr = tree.query(g, predicate="dwithin", distance=bf)
                    if not nr.size:
                        continue
                    nr = nr[base_flag[nr]]
                    if nr.size:
                        near[r].setdefault(i, []).append(
                            np.unique(cell_city[nr]).astype(np.int32))
        pack = {}
        for r in RADII_KM:
            ss, sc = [], []
            for i, ps in near[r].items():
                u = np.unique(np.concatenate(ps))
                ss.append(np.full(u.size, i, np.int32)); sc.append(u)
            pack[f"sp_{r:g}"] = np.concatenate(ss) if ss else np.zeros(0, np.int32)
            pack[f"city_{r:g}"] = np.concatenate(sc) if sc else np.zeros(0, np.int32)
        np.savez(d / f"chunk_{a:07d}.npz", **pack)
        state["next"] = int(b)
        f.write_text(json.dumps(state))
        print(f"radius  parts {b:,}/{n_parts:,}  {100*b/n_parts:5.1f}%  "
              f"{time.time()-t0:5.0f}s", flush=True)
    return state["next"] >= n_parts, state["next"], n_parts


def settings():
    import tropics_assemble as ta
    sp = ta.merged_categories(rc.load_index()[0])
    excl = [t.upper() for t in EXCLUDED_TAXA]
    keep = set(sp.sp_i[sp.category.isin(THREAT)
                       & ~sp.taxon.astype(str).str.upper().isin(excl)].to_numpy())
    keep = np.fromiter(keep, np.int32)
    out = {}
    for r in RADII_KM:
        S = []
        for f in sorted(_dir().glob("chunk_*.npz")):
            z = np.load(f)
            s, c = z[f"sp_{r:g}"], z[f"city_{r:g}"]
            m = np.isin(s, keep)
            S.append(pd.DataFrame(dict(sp_i=s[m], city_id=c[m])))
        t = pd.concat(S, ignore_index=True).drop_duplicates()
        out[r] = t.groupby("city_id").sp_i.size().rename(f"setting_{r:g}km")
    return out


def fits():
    from scipy.spatial import cKDTree
    d = tn.table()
    cells, cities = dl._load(0)
    se = settings()
    x = cities[["city_id", "lon", "lat"]].copy()
    for s in se.values():
        x = x.join(s, on="city_id")
    x = x.fillna(0.0)
    tree = cKDTree(x[["lon", "lat"]].to_numpy())
    dist, j = tree.query(d[["lon", "lat"]].to_numpy(), k=1)
    z = d.copy()
    for r in RADII_KM:
        col = f"setting_{r:g}km"
        z[col] = np.where(dist < 0.05, x[col].to_numpy()[j], np.nan)
    rows = []
    for r in RADII_KM:
        col = f"setting_{r:g}km"
        f = fit_scaling(z.dropna(subset=[col]), response="cost_per_km2",
                        size="urban_km2_base", setting=col, spatial=True)
        rows.append(dict(radius_km=r, n=int(f["n"]), beta=float(f["beta"]),
                         ci_lower=float(f["ci_lower"]), ci_upper=float(f["ci_upper"]),
                         gamma_setting=float(f.get("gamma_setting", np.nan)),
                         R2=float(f["R2"]),
                         mean_setting=float(z[col].mean()),
                         corr_with_50km=float(z[col].corr(z["setting_50km"],
                                                          method="spearman"))))
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "setting_radius.csv", index=False)
    z[["city_id"] + [f"setting_{r:g}km" for r in RADII_KM]].to_csv(
        TABLE_DIR / "setting_radius_cities.csv", index=False)
    return out


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if a and a[0] == "fits":
        print(fits().to_string(index=False))
    else:
        done, n, tot = step(float(a[0]) if a else 90.0)
        print(json.dumps(dict(done=done, next=n, n_parts=tot)))


def taxon_settings(radius=50.0):
    sp, _ = rc.load_index()
    groups = {"birds": ["AVES"], "mammals": ["MAMMALIA"],
              "herps": ["AMPHIBIA", "REPTILIA"]}
    out = {}
    for name, taxa in groups.items():
        keep = np.fromiter(set(sp.sp_i[sp.taxon.isin(taxa)].to_numpy()), np.int32)
        S = []
        for f in sorted(_dir().glob("chunk_*.npz")):
            z = np.load(f)
            s, c = z[f"sp_{radius:g}"], z[f"city_{radius:g}"]
            m = np.isin(s, keep)
            S.append(pd.DataFrame(dict(sp_i=s[m], city_id=c[m])))
        t = pd.concat(S, ignore_index=True).drop_duplicates()
        out[name] = t.groupby("city_id").sp_i.size().rename(f"set_{name}")
    return out


def write_taxon_settings(out="setting_by_taxon_v2.csv"):
    from scipy.spatial import cKDTree
    import tropics_numbers as tn
    cells, cities = dl._load(0)
    se = taxon_settings()
    x = cities[["city_id", "lon", "lat"]].copy()
    for s in se.values():
        x = x.join(s, on="city_id")
    x = x.fillna(0.0)
    d = tn.table()
    tree = cKDTree(x[["lon", "lat"]].to_numpy())
    dist, j = tree.query(d[["lon", "lat"]].to_numpy(), k=1)
    cols = {c: x[c].to_numpy()[j]
            for c in ("set_birds", "set_mammals", "set_herps")}
    f = pd.DataFrame(dict(city_id=d.city_id.to_numpy(), **cols))
    f.to_csv(TABLE_DIR / out, index=False)
    return f
