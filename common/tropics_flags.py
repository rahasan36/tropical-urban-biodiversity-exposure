"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tropics_config import INTERIM_DIR, THREAT_CATS, EXCLUDED_TAXA

CACHE = INTERIM_DIR / "sp_flags_merged.npz"


def normalize_category(c):
    s = np.asarray(c, dtype=object)
    out = np.array([str(v).strip().upper().split(" ")[0] if v is not None else ""
                    for v in s.ravel()], dtype="<U8")
    return out.reshape(s.shape)


def _rebuild(pairs_path):
    import pandas as pd
    import tropics_assemble as ta

    z = np.load(pairs_path, allow_pickle=False)
    idx = json.loads(str(z["sp_index"]))
    meta = json.loads(str(z["sp_meta"]))
    n = int(max(idx.values())) + 1
    inv = {v: k for k, v in idx.items()}
    sci = np.array([(meta.get(inv[i]) or ["", "", ""])[0] for i in range(n)],
                   dtype="<U80")
    cat = np.array([(meta.get(inv[i]) or ["", "", ""])[1] for i in range(n)],
                   dtype=object)
    tax = np.array([(meta.get(inv[i]) or ["", "", ""])[2] for i in range(n)],
                   dtype="<U12")
    area = np.zeros(n, float)
    a = z["sp_area"]
    area[:min(a.size, n)] = a[:min(a.size, n)]

    book = ta.read_bird_checklist(ta.BOTW_CHECKLIST)
    lut = dict(zip(book["sci_name"].astype(str), book["category"].astype(str)))
    cat = normalize_category(cat)
    miss = np.isin(cat, ["", "NAN", "NONE"])
    fill = np.array([lut.get(s, "") for s in sci], dtype=object)
    cat[miss] = normalize_category(fill[miss])
    np.savez_compressed(CACHE, cat=cat, tax=np.char.upper(tax), sci=sci,
                        area=area)
    return cat, np.char.upper(tax), sci, area


def flags(pairs_path=None):
    from tropics_cost import PAIRS_PATH
    pairs_path = Path(pairs_path or PAIRS_PATH)
    z = np.load(pairs_path, allow_pickle=False)
    n = int(max(json.loads(str(z["sp_index"])).values())) + 1
    if CACHE.exists():
        m = np.load(CACHE, allow_pickle=False)
        if m["cat"].size == n:
            return (normalize_category(m["cat"]), np.char.upper(m["tax"]),
                    m["sci"], m["area"])
    return _rebuild(pairs_path)


def masks(pairs_path=None, narrow_threshold_km2=None):
    cat, tax, _, area = flags(pairs_path)
    keep = ~np.isin(tax, [t.upper() for t in EXCLUDED_TAXA])
    threat = np.isin(cat, sorted(THREAT_CATS)) & keep
    narrow = None
    if narrow_threshold_km2 is not None:
        narrow = (area > 0) & (area <= float(narrow_threshold_km2)) & keep
    return keep, threat, narrow, area


def _selftest():
    assert normalize_category(np.array(["CR (PEW)"]))[0] == "CR"
    assert normalize_category(np.array([" en "]))[0] == "EN"
    assert normalize_category(np.array([None], dtype=object))[0] == ""
    cat, tax, sci, area = flags()
    assert cat.size == tax.size == sci.size == area.size
    aves = tax == "AVES"
    assert aves.sum() > 10000, aves.sum()
    blank = np.isin(cat[aves], ["", "NAN", "NONE"]).sum()
    assert blank == 0, f"{blank} birds still carry no category"
    keep, threat, narrow, _ = masks(narrow_threshold_km2=1e4)
    assert threat[aves].sum() > 0, "no threatened birds after the merge"
    print(f"ok  n={cat.size}  birds={int(aves.sum())}  "
          f"threatened={int(threat.sum())}  threatened birds={int(threat[aves].sum())}")


if __name__ == "__main__":
    _selftest()
