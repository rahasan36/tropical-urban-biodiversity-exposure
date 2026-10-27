"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
import region_panels as rp
import tropics_surfaces as sf

from tropics_config import TABLE_DIR
OUT = TABLE_DIR / "region_window_search_v1.csv"
TROPIC = 23.4394
ZONES = {"Africa": (-20.0, 52.0, -TROPIC, TROPIC),
         "Indonesia": (94.0, 142.0, -11.5, 6.5)}


def make_sum(added, rich, R, A):
    ny, nx = added.shape
    lat_c = sf.LAT_TOP - (np.arange(ny) + 0.5) * A
    lon_c = -180 + (np.arange(nx) + 0.5) * A
    ri = np.clip(((sf.LAT_TOP - lat_c) // R).astype(int), 0, rich.shape[0] - 1)
    ci = np.clip(((lon_c + 180) // R).astype(int), 0, rich.shape[1] - 1)
    a = np.nan_to_num(added)
    on = rich[np.ix_(ri, ci)] > 0
    S = np.zeros((ny + 1, nx + 1)); S[1:, 1:] = a.cumsum(0).cumsum(1)
    O = np.zeros((ny + 1, nx + 1)); O[1:, 1:] = (a * on).cumsum(0).cumsum(1)
    return lat_c, lon_c, S, O


def window(lat_c, lon_c, S, O, lon0, lon1, lat0, lat1):
    rs = np.where((lat_c >= lat0) & (lat_c < lat1))[0]
    cs = np.where((lon_c >= lon0) & (lon_c < lon1))[0]
    if rs.size == 0 or cs.size == 0:
        return 0.0, 0.0
    r0, r1, c0, c1 = rs[0], rs[-1] + 1, cs[0], cs[-1] + 1
    f = lambda M: M[r1, c1] - M[r0, c1] - M[r1, c0] + M[r0, c0]
    return float(f(S)), float(f(O))


def _selftest():
    rng = np.random.default_rng(1)
    added = rng.random((40, 60)); rich = (rng.random((20, 30)) > 0.5).astype(float)
    lat_top, R, A = sf.LAT_TOP, 2 * 0.1, 0.1
    old = sf.LAT_TOP
    lat_c, lon_c, S, O = make_sum(added, rich, R, A)
    lon0, lon1 = lon_c[5] - 0.05, lon_c[33] + 0.05
    lat0, lat1 = lat_c[30] - 0.05, lat_c[7] + 0.05
    t, o = window(lat_c, lon_c, S, O, lon0, lon1, lat0, lat1)
    rs = (lat_c >= lat0) & (lat_c < lat1); cs = (lon_c >= lon0) & (lon_c < lon1)
    ri = np.clip(((sf.LAT_TOP - lat_c) // R).astype(int), 0, rich.shape[0] - 1)
    ci = np.clip(((lon_c + 180) // R).astype(int), 0, rich.shape[1] - 1)
    a = added[np.ix_(rs, cs)]; on = rich[np.ix_(ri[rs], ci[cs])] > 0
    assert abs(t - a.sum()) < 1e-6 and abs(o - a[on].sum()) < 1e-6
    print("selftest ok")


def run():
    r, thr = sf.load_richness()
    rich = r["narrow"].astype(float)
    added = sf.load_urban()["added"]
    lat_c, lon_c, S, O = make_sum(added, rich, sf.RICH_DEG, sf.DISP_DEG)
    rows = []
    for zone, (x0, x1, y0, y1) in ZONES.items():
        for w in (8.0, 11.0, 14.0, 17.0):
            h = 0.86 * w
            for lon0 in np.arange(x0, x1 - w + 0.01, 1.0):
                for lat0 in np.arange(y0, y1 - h + 0.01, 1.0):
                    t, o = window(lat_c, lon_c, S, O, lon0, lon0 + w, lat0, lat0 + h)
                    if t <= 0:
                        continue
                    rows.append(dict(zone=zone, w=w, lon0=lon0, lon1=lon0 + w,
                                     lat0=round(lat0, 2), lat1=round(lat0 + h, 2),
                                     added_km2=t, overlap_km2=o, share_pct=o / t * 100))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    return d


if __name__ == "__main__":
    _selftest()
    d = run()
    for zone in ZONES:
        z = d[d.zone == zone]
        print(f"\n== {zone}: top by overlap km2 (share >= 10%)")
        print(z[z.share_pct >= 10].sort_values("overlap_km2", ascending=False)
              .head(12).round(1).to_string(index=False))
        print(f"== {zone}: top by share (added >= 300 km2)")
        print(z[z.added_km2 >= 300].sort_values("share_pct", ascending=False)
              .head(12).round(1).to_string(index=False))
