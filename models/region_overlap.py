"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import json
import numpy as np
from pathlib import Path
import region_panels as rp
import tropics_surfaces as sf

from tropics_config import TABLE_DIR
OUT = TABLE_DIR / "region_window_overlap.json"
TROPIC = 23.4394

def build():
    r, thr = sf.load_richness()
    rich = r["narrow"].astype(float)
    added = sf.load_urban()["added"]
    R, A = sf.RICH_DEG, sf.DISP_DEG
    ny, nx = added.shape
    lat_c = sf.LAT_TOP - (np.arange(ny) + 0.5) * A
    lon_c = -180 + (np.arange(nx) + 0.5) * A
    ri = np.clip(((sf.LAT_TOP - lat_c) // R).astype(int), 0, rich.shape[0] - 1)
    ci = np.clip(((lon_c + 180) // R).astype(int), 0, rich.shape[1] - 1)

    def share(lon0, lon1, lat0, lat1):
        rs = (lat_c >= lat0) & (lat_c < lat1)
        cs = (lon_c >= lon0) & (lon_c < lon1)
        a = np.nan_to_num(added[np.ix_(rs, cs)])
        on = rich[np.ix_(ri[rs], ci[cs])] > 0
        return float(a.sum()), float(a[on].sum() / a.sum() * 100)

    out = {"narrow_threshold_km2": thr}
    t, s = share(-180, 180, -TROPIC, TROPIC)
    out["tropics"] = {"added_km2": t, "share_pct": s}
    for b in rp.NAMED_REGIONS + rp.RULE_REGIONS_4 + [rp.EASTERN_ARC, rp.SUMATRA]:
        f = rp._fit_box(b, 0.86)
        t, s = share(*f[1:5])
        out[b[0]] = {"box": list(f[1:5]), "added_km2": t, "share_pct": s}
    return out

def _selftest():
    added = np.ones((4, 4)); rich = np.zeros((2, 2)); rich[0, :] = 1
    ri = np.array([0, 0, 1, 1]); ci = np.array([0, 0, 1, 1])
    on = rich[np.ix_(ri, ci)] > 0
    assert abs(added[on].sum() / added.sum() * 100 - 50.0) < 1e-12
    print("selftest ok")

if __name__ == "__main__":
    _selftest()
    d = build()
    OUT.write_text(json.dumps(d, indent=1))
    for k, v in d.items():
        print(k, v if not isinstance(v, dict) else f"{v['share_pct']:.1f}%  of {v['added_km2']:,.0f} km2")
