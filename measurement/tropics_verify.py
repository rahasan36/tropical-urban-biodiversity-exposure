"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np, pandas as pd, json
from tropics_config import TABLE_DIR, INTERIM_DIR
from tropics_measure import MEASURE_PATH, city_table
from tropics_cost import PAIRS_PATH
import tropics_assemble as ta

def main():
    ok = []
    df = pd.read_csv(TABLE_DIR / "cities_cost.csv")
    z = np.load(MEASURE_PATH, allow_pickle=False)

    tot_conv_cells = float(z["cell_conv"].sum())
    tot_added = float(df.urban_added_km2.sum())
    ok.append(("converted area, cells vs cities",
               abs(tot_conv_cells - tot_added) / tot_added < 1e-9,
               f"{tot_conv_cells:,.1f} vs {tot_added:,.1f} km2"))

    ok.append(("urban area monotone 2000->2020",
               bool((df.urban_km2_final >= df.urban_km2_base - 1e-9).all()),
               f"min delta {float((df.urban_km2_final-df.urban_km2_base).min()):.3g} km2"))

    n_conc = df.n_threat_affected + df.n_narrow_affected
    bound = n_conc * df.urban_added_km2
    ratio = (df.cost_km2 / bound.clip(lower=1e-9)).replace([np.inf], 0)
    ok.append(("cost bounded by species x area", bool((ratio <= 1 + 1e-5).all()),
               f"max ratio {float(ratio.max()):.6f}"))

    rr, cc = z["cell_row"], z["cell_col"]
    keys = rr.astype(np.int64) * 43200 + cc
    ok.append(("one city per cell", int(np.unique(keys).size) == keys.size,
               f"{keys.size:,} cells, {np.unique(keys).size:,} unique"))

    zz = np.load(PAIRS_PATH, allow_pickle=False)
    pv = pd.DataFrame(dict(city_id=zz["pair_city"], conv=zz["pair_conv"].astype(float)))
    mx = pv.groupby("city_id").conv.max()
    cmp_ = df.set_index("city_id").urban_added_km2.reindex(mx.index)
    ok.append(("per-species converted <= city converted",
               bool((mx <= cmp_ * (1 + 1e-5) + 1e-3).all()),
               f"max excess {float((mx-cmp_).max()):.3g} km2"))

    meta = json.loads((TABLE_DIR / "cities_cost_meta.json").read_text())
    sp, pairs, setting = ta.load_pairs()
    trop = set(pairs.sp_i.unique()) | set(setting.sp_i.unique())
    sp2, thr = ta.apply_flags(sp, trop)
    ok.append(("narrow-range threshold reproducible",
               abs(thr - meta["narrow_threshold_km2"]) < 1e-6,
               f"{thr:,.1f} km2 = 25th percentile of {int(sp2.in_pool.sum()):,} species"))

    ok.append(("city sets agree",
               set(df.city_id) == set(city_table().city_id),
               f"{len(df):,} cities"))

    width = max(len(n) for n, _, _ in ok)
    for name, passed, detail in ok:
        print(f"{name:<{width}}  {'PASS' if passed else 'FAIL'}   {detail}")
    return all(p for _, p, _ in ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
