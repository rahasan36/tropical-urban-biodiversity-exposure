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
import tropics_numbers as tn
import tropics_refit as trf
from tropics_config import TABLE_DIR, NARROW_RANGE_PCTL, EXCLUDED_TAXA, THREAT_CATS
from tropics_cost import PAIRS_PATH

UNION_PAIRS = TABLE_DIR / "birdlife_union_pairs.npz"
UNION_SUMMARY = TABLE_DIR / "birdlife_union_summary.csv"
OUT = TABLE_DIR / "bird_union_effect.csv"


def load():
    z = np.load(PAIRS_PATH, allow_pickle=False)
    idx = json.loads(str(z["sp_index"]))
    cat, tax, sci, area = tfl.flags(PAIRS_PATH)
    pairs = pd.DataFrame(dict(sp_i=z["pair_sp"].astype(np.int64),
                              city_id=z["pair_city"].astype(np.int64),
                              conv_km2=z["pair_conv"].astype(float)))
    pool = np.zeros(cat.size, bool)
    pool[np.unique(z["set_sp"])] = True
    pool[np.unique(z["pair_sp"])] = True
    return idx, cat, tax, area, pairs, pool


def substitute(idx, tax, area, pairs):
    u = np.load(UNION_PAIRS, allow_pickle=False)
    m = {k: i for k, i in idx.items()}
    sp_i = np.array([m.get(str(s), -1) for s in u["sp_id"]], dtype=np.int64)
    keep = sp_i >= 0
    bird_pairs = pd.DataFrame(dict(sp_i=sp_i[keep],
                                   city_id=u["city_id"][keep].astype(np.int64),
                                   conv_km2=u["conv_km2"][keep].astype(float)))
    aves = tax == "AVES"
    kept = pairs[~aves[pairs.sp_i.to_numpy()]]
    new_pairs = pd.concat([kept, bird_pairs], ignore_index=True)

    t = pd.read_csv(UNION_SUMMARY)
    t["sp_i"] = [m.get(str(s), -1) for s in t.sp_id]
    t = t[t.sp_i >= 0]
    new_area = area.copy()
    good = t.area_union_km2.to_numpy() > 0
    new_area[t.sp_i.to_numpy()[good]] = t.area_union_km2.to_numpy()[good]
    return new_pairs, new_area, int(keep.size - keep.sum()), int((~good).sum())


def response(pairs, area, cat, tax, pool, thr=None):
    keep = ~np.isin(tax, [t.upper() for t in EXCLUDED_TAXA])
    if thr is None:
        a = area[pool & keep & (area > 0)]
        thr = float(np.percentile(a, NARROW_RANGE_PCTL))
    threat = np.isin(cat, sorted(THREAT_CATS)) & keep
    narrow = pool & keep & (area > 0) & (area <= thr)
    concern = threat | narrow
    ps = pairs.sp_i.to_numpy()
    sel = concern[ps]
    p = pairs[sel]
    g = p.groupby("city_id")
    out = pd.DataFrame({
        "cost_km2": g.conv_km2.sum(),
        "rarity_converted": (p.conv_km2 / np.maximum(area[p.sp_i.to_numpy()], 1e-6))
                            .groupby(p.city_id).sum(),
    })
    return out, thr, int(concern.sum())


def fit(base, resp, label, thr, n_concern):
    x = base.join(resp, on="city_id", rsuffix="_c")
    x["cost_km2_c"] = x.cost_km2_c.fillna(0.0)
    x["rarity_converted_c"] = x.rarity_converted_c.fillna(0.0)
    x["cost_per_km2"] = x.cost_km2_c / x.urban_added_km2
    x["rarity_per_km2"] = x.rarity_converted_c / x.urban_added_km2
    r = trf.fit_scaling(x, response="cost_per_km2", size="urban_km2_base",
                        setting="setting_footprint", spatial=True)
    q = trf.fit_scaling(x, response="rarity_per_km2", size="urban_km2_base",
                        setting="setting_footprint", spatial=True)
    return dict(variant=label, n=int(r["n"]),
                converted_km2=float(x.cost_km2_c.sum()),
                narrow_threshold_km2=thr, n_species_of_concern=n_concern,
                beta=float(r["beta"]), lo=float(r["ci_lower"]),
                hi=float(r["ci_upper"]), p=float(r["p_value"]),
                gamma=float(r.get("gamma_setting", np.nan)),
                beta_rarity=float(q["beta"]),
                rarity_lo=float(q["ci_lower"]), rarity_hi=float(q["ci_upper"]))


def _selftest():
    idx, cat, tax, area, pairs, pool = load()
    new_pairs, new_area, unmatched, no_union = substitute(idx, tax, area, pairs)
    aves = tax == "AVES"
    a0 = pairs[~aves[pairs.sp_i.to_numpy()]].conv_km2.sum()
    a1 = new_pairs[~aves[new_pairs.sp_i.to_numpy()]].conv_km2.sum()
    assert abs(a0 - a1) < 1e-6, (a0, a1)
    assert np.allclose(area[~aves], new_area[~aves])
    print(f"ok  non-bird pairs and ranges untouched; {unmatched:,} union rows "
          f"matched no species in the index, {no_union:,} species had no union area")


if __name__ == "__main__":
    _selftest()
    base = tn.analysis_sample().set_index("city_id")
    idx, cat, tax, area, pairs, pool = load()

    rows = []
    r0, thr0, n0 = response(pairs, area, cat, tax, pool)
    rows.append(fit(base, r0, "as published (max rule, summed areas)", thr0, n0))
    new_pairs, new_area, _, _ = substitute(idx, tax, area, pairs)
    r1, thr1, n1 = response(new_pairs, new_area, cat, tax, pool)
    rows.append(fit(base, r1, "bird spatial unions", thr1, n1))
    r2, thr2, n2 = response(new_pairs, area, cat, tax, pool)
    rows.append(fit(base, r2, "union converted area only", thr2, n2))
    r3, thr3, n3 = response(pairs, new_area, cat, tax, pool)
    rows.append(fit(base, r3, "union global range only", thr3, n3))

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    pd.set_option("display.width", 240)
    print()
    print(d.to_string(index=False))
