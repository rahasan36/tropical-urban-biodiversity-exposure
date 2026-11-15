"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np, pandas as pd

from tropics_config import TABLE_DIR
import tropics_setting as ts
import tropics_rangecache as rc
from tropics_scaling import fit_scaling

THREAT = ("CR", "EN", "VU")


_V3 = TABLE_DIR / "setting_footprint_v3.csv"
SETTING_PATH = _V3 if _V3.exists() else TABLE_DIR / "setting_footprint_v2.csv"


def footprint_setting():
    return (pd.read_csv(SETTING_PATH).set_index("city_id")
              .setting_footprint.rename("setting_footprint"))


def table():
    d = pd.read_csv(TABLE_DIR / "cities_cost.csv")
    return d.join(footprint_setting(), on="city_id")


def _fmt(r):
    g = ""
    if "gamma_setting" in r:
        gm, gs = r["gamma_setting"], r["gamma_se"]
        g = (f"   setting  g {gm:+.4f}  [{gm-1.96*gs:+.4f}, {gm+1.96*gs:+.4f}]")
    return (f"  n {r['n']:,}  {r['model']}\n"
            f"   size     b {r['beta']:+.4f}  "
            f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]  P {r['p_value']:.2g}"
            f"   R2 {r['R2']:.3f}\n{g}")


SPECS = {
    "A": ("cost_per_km2   size 2000, setting footprint",
          dict(response="cost_per_km2", size="urban_km2_base",
               setting="setting_footprint")),
    "B": ("cost_per_km2   size 2000, setting 2020 extent",
          dict(response="cost_per_km2", size="urban_km2_base",
               setting="setting_richness")),
    "C": ("cost_per_km2   size 2020, setting 2020 extent (as drafted)",
          dict(response="cost_per_km2", size="urban_km2_final",
               setting="setting_richness")),
    "D": ("rarity_per_km2 size 2000, setting footprint",
          dict(response="rarity_per_km2", size="urban_km2_base",
               setting="setting_footprint")),
}


def report(keys=("A", "B", "C", "D"), spatial=True, d=None):
    d = table() if d is None else d
    out = {}
    for k in keys:
        label, kw = SPECS[k]
        r = fit_scaling(d, spatial=spatial, **kw)
        print(f"\n{k}  {label}")
        print(_fmt(r))
        out[k] = r
    return out


if __name__ == "__main__":
    import sys
    report(tuple(sys.argv[1]) if len(sys.argv) > 1 else ("A", "B", "C", "D"),
           spatial=("--nospatial" not in sys.argv))
