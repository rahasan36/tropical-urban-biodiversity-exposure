"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tropics_config import TABLE_DIR, INTERIM_DIR, THREAT_CATS, EXCLUDED_TAXA

STORED = TABLE_DIR / "setting_footprint_v2.csv"
OUT = TABLE_DIR / "setting_footprint_v3.csv"
CFG0 = INTERIM_DIR / "sens_cfg0.npz"
FIXED = TABLE_DIR / "delineation_fixed_threshold.csv"


def threat_mask(sp):
    import tropics_assemble as ta
    s = ta.merged_categories(sp.copy())
    keep = ~s.taxon.astype(str).str.upper().isin(
        [t.upper() for t in EXCLUDED_TAXA]).to_numpy()
    return s.category.isin(THREAT_CATS).to_numpy() & keep


def setting_from(pairs_npz):
    import tropics_rangecache as rc
    z = np.load(pairs_npz, allow_pickle=False)
    sp, _ = rc.load_index()
    thr = threat_mask(sp)
    d = pd.DataFrame(dict(sp_i=z["set_sp"],
                          city_id=z["set_city"])).drop_duplicates()
    d = d[thr[d.sp_i.to_numpy()]]
    return d.groupby("city_id").sp_i.nunique().rename("setting_footprint")


def cities():
    return pd.read_csv(TABLE_DIR / "cities_cost.csv").city_id


def _selftest():
    assert CFG0.exists(), f"{CFG0.name} is not on disk; run delineation_fixed.py"
    s = setting_from(CFG0).reindex(cities()).fillna(0.0)
    got = float(s.mean())
    f = pd.read_csv(FIXED)
    want = float(f[f.threshold_rule == "own quartile"].iloc[0].mean_setting)
    assert abs(got - want) < 1e-9, (got, want)
    print(f"ok  rule reproduces the delineation pass's mean setting "
          f"{got:.6f} over {len(s):,} agglomerations")


def main():
    _selftest()
    if "--selftest" in sys.argv[1:]:
        return
    import tropics_rangecache as rc
    s = setting_from(CFG0).reindex(cities()).fillna(0.0)
    d = s.rename("setting_footprint").rename_axis("city_id").reset_index()
    d.to_csv(OUT, index=False)
    print(f"written to {OUT.name}   cache fingerprint {rc.fingerprint()}   "
          f"mean setting {s.mean():.4f} over {len(s):,} agglomerations")
    if STORED.exists():
        stored = pd.read_csv(STORED).set_index("city_id").setting_footprint
        dif = s.reindex(stored.index) - stored
        print(f"  {int((dif != 0).sum()):,} of {len(dif):,} agglomerations differ from "
              f"{STORED.name}, by a mean of {dif.mean():+.4f}")


if __name__ == "__main__":
    main()
