"""A third of the world's threatened land vertebrates exposed to tropical urbanization

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
    no_area = ~(area > 0)

    rows = []
    remaining = np.ones(n_index, bool)

    def drop(label, mask, note):
        nonlocal remaining
        m = remaining & mask
        rows.append(dict(step=label, n_removed=int(m.sum()),
                         n_remaining=int((remaining & ~m).sum()), note=note))
        remaining = remaining & ~m

    rows.append(dict(step="Range polygons read from the two providers",
                     n_removed=0, n_remaining=n_index,
                     note="Species with at least one mapped range part in the "
                          "IUCN Red List spatial data or the BirdLife and HBW "
                          "distribution maps"))
    drop("No mapped area after clipping to the tropical band", no_area,
         "The range polygon exists but resolves to zero area on the 30 arcsec "
         "lattice inside the study extent")
    drop("Excluded higher taxon", excluded_taxon,
         "Taxa the study does not analyse; empty in the current configuration")
    drop("Range does not reach any delineated agglomeration or its setting",
         ~in_tropics,
         "The species has mapped tropical area but none of it falls in a city "
         "setting or a converted cell, so it can enter neither the covariate "
         "nor the response")

    pool = remaining
    rows.append(dict(step="Analysis pool", n_removed=0,
                     n_remaining=int(pool.sum()),
                     note="Species that can contribute to the covariate or the "
                          "response for at least one agglomeration"))

    conv = np.zeros(n_index, bool)
    conv[pair_sp] = True
    rows.append(dict(step="Of the pool, range actually converted",
                     n_removed=int((pool & ~conv).sum()),
                     n_remaining=int((pool & conv).sum()),
                     note="Species with a non-zero converted area; the rest "
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
    removed = int(d.n_removed[1:4].sum())
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
