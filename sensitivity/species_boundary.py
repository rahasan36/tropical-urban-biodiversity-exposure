"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resolution_test as RT
import tropics_speciesmaps as sm
from tropics_config import TABLE_DIR

OUT = TABLE_DIR / "species_boundary_shares.csv"
RULES = ("centroid", "touches", "within")
RULE_LABEL = {"centroid": "cell centre inside the range",
              "touches": "range touches the cell",
              "within": "cell wholly inside the range"}


def published():
    w = sm.worst(n=20, cats=("CR", "EN", "VU"))
    return w[["sp_id", "sci", "category", "converted_km2", "range_km2",
              "share"]].rename(columns={"converted_km2": "converted_published",
                                        "share": "share_published"})


def per_species(rule, budget_s=None, verbose=True, only_sp=None):
    pairs, sp, meta = RT.converted(1, rule, budget_s=budget_s, verbose=verbose,
                                   only_sp=only_sp)
    g = pairs.groupby("sp_i", as_index=False).conv_km2.sum()
    g["sp_id"] = sp.set_index("sp_i").sp_id.reindex(g.sp_i).to_numpy()
    return g[["sp_id", "conv_km2"]], meta


def run(verbose=True, full=False):
    base = published()
    out = base.copy()
    only = None
    if not full:
        sp_all, _ = __import__("tropics_rangecache").load_index()
        pos = sp_all.set_index("sp_id").sp_i
        miss = [s for s in base.sp_id if s not in pos.index]
        if miss:
            raise SystemExit(f"{len(miss)} atlas species are not in the range "
                             f"cache index: {miss[:5]}")
        only = [int(pos[s]) for s in base.sp_id]
    for rule in RULES:
        t0 = time.time()
        g, meta = per_species(rule, verbose=verbose, only_sp=only)
        m = base[["sp_id"]].merge(g, on="sp_id", how="left")
        assert len(m) == len(base)
        miss = int(m.conv_km2.isna().sum())
        if miss:
            raise SystemExit(f"{miss} of the twenty species carry no measurement "
                             f"under {rule}; the join is wrong, not the rule")
        out[f"conv_{rule}"] = m.conv_km2.to_numpy()
        out[f"share_{rule}"] = m.conv_km2.to_numpy() / base.range_km2.to_numpy()
        if verbose:
            print(f"  {rule:9s} {meta['n_cells']:,} cells  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    sh = out[[f"share_{r}" for r in RULES]].to_numpy()
    out["share_min"] = sh.min(axis=1)
    out["share_max"] = sh.max(axis=1)
    out["share_span_pp"] = (out.share_max - out.share_min) * 100.0
    return out.sort_values("share_published", ascending=False).reset_index(drop=True)


def _selftest():
    d = pd.read_csv(OUT) if OUT.exists() else run()
    bad = d[~((d.conv_within <= d.conv_centroid + 1e-9)
              & (d.conv_centroid <= d.conv_touches + 1e-9))]
    if len(bad):
        raise SystemExit("the three rules are not nested for "
                         + ", ".join(bad.sci.astype(str)))
    print(f"ok  nested for all {len(d)} species; share span "
          f"{d.share_span_pp.min():.2f} to {d.share_span_pp.max():.2f} "
          "percentage points")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _selftest()
    else:
        d = run()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT, index=False)
        pd.set_option("display.width", 220)
        print(d[["sci", "category", "share_published", "share_within",
                 "share_centroid", "share_touches", "share_span_pp"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nwritten to {OUT}")
        _selftest()
