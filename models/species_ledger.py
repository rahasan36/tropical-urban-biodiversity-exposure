"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tropics_flags as tfl
from tropics_config import TABLE_DIR, EXCLUDED_TAXA
from tropics_cost import PAIRS_PATH

OUT = TABLE_DIR / "species_ledger.csv"


def ledger():
    z = np.load(PAIRS_PATH, allow_pickle=False)
    cat, tax, sci, area = tfl.flags(PAIRS_PATH)
    n_index = cat.size

    set_sp = np.unique(z["set_sp"])
    pair_sp = np.unique(z["pair_sp"])

    in_tropics = np.zeros(n_index, bool)
    in_tropics[set_sp] = True
    in_tropics[pair_sp] = True

    excluded_taxon = np.isin(tax, [t.upper() for t in EXCLUDED_TAXA])

    rows = []
    remaining = np.ones(n_index, bool)

    rows.append(dict(step="Range polygons read from the two providers",
                     n_removed=0, n_remaining=n_index,
                     note="Species with at least one mapped range part in the "
                          "IUCN Red List spatial data or the BirdLife and HBW "
                          "distribution maps"))
    import tropics_rangecache as rc
    n_band = int(len(rc.load_index()[0]))
    outside = n_index - n_band
    unreached = int((~in_tropics).sum()) - outside
    assert outside >= 0 and unreached >= 0 and int(in_tropics.sum()) <= n_band
    rows.append(dict(step="No range part in the tropical band or its 50 km margin",
                     n_removed=outside, n_remaining=n_band,
                     note="The mapped range lies wholly outside the tropical "
                          "band and its 50 km margin"))
    rows.append(dict(step="Range reaches no delineated agglomeration or its setting",
                     n_removed=unreached, n_remaining=n_band - unreached,
                     note="The species has mapped area in the band and none of "
                          "it falls in a setting buffer or a converted cell, so "
                          "it enters neither the covariate nor the response"))
    remaining = remaining & in_tropics & ~excluded_taxon
    assert int(remaining.sum()) == n_band - unreached

    pool = remaining
    rows.append(dict(step="Analysis pool", n_removed=0,
                     n_remaining=int(pool.sum()),
                     note="Species that can contribute to the covariate or the "
                          "response for at least one agglomeration"))

    conv = np.zeros(n_index, bool)
    conv[pair_sp] = True
    rows.append(dict(step="Of the pool, range converted",
                     n_removed=int((pool & ~conv).sum()),
                     n_remaining=int((pool & conv).sum()),
                     note="Species with a non-zero converted area. The rest "
                          "are in the pool through the setting covariate only"))

    d = pd.DataFrame(rows)

    thr = float(json.loads((TABLE_DIR / "cities_cost_meta.json").read_text())
                ["narrow_threshold_km2"])
    threat = np.isin(cat, ["CR", "EN", "VU"]) & pool
    narrow = pool & (area > 0) & (area <= thr)
    comp = (pd.DataFrame(dict(taxon=tax[pool], cat=cat[pool],
                              area=area[pool],
                              threat=threat[pool], narrow=narrow[pool]))
            .groupby("taxon")
            .agg(n=("cat", "size"), threatened=("threat", "sum"),
                 narrow_range=("narrow", "sum"),
                 median_range_km2=("area", "median"))
            .reset_index().sort_values("n", ascending=False))
    return d, comp, dict(pool=int(pool.sum()), threatened=int(threat.sum()),
                         narrow=int(narrow.sum()),
                         narrow_threshold_km2=thr,
                         converted=int((pool & conv).sum()))


def _selftest():
    d, comp, s = ledger()
    start = int(d.iloc[0].n_remaining)
    removed = int(d.n_removed[1:3].sum())
    assert start - removed == s["pool"], (start, removed, s["pool"])
    assert int(comp.n.sum()) == s["pool"]
    print("ok  ledger lines sum to the pool")


if __name__ == "__main__":
    _selftest()
    d, comp, s = ledger()
    d.to_csv(OUT, index=False)
    comp.to_csv(TABLE_DIR / "species_pool_composition.csv", index=False)
    print(d.to_string(index=False))
    print()
    print(comp.to_string(index=False))
    print()
    print(json.dumps(s, indent=1))
