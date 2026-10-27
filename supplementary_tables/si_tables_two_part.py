"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
from tropics_config import SUPP_TABLE_DIR, TABLE_DIR
T = TABLE_DIR
OUT = SUPP_TABLE_DIR


def s19():
    m = json.load(open(T / "narrow_two_part_meta.json"))
    con = json.load(open(T / "narrow_two_part_contrasts_B.json"))
    v = json.load(open(T / "narrow_two_part_validation_B.json"))
    names = {"ls": "log10 year 2000 extent", "lset": "log10 setting richness + 1",
             "ladd": "log10 urban land added"}
    rows = []
    for spec, d in m.items():
        if spec == "sample":
            continue
        for part in ("occurrence", "intensity"):
            for k, (b, se, p) in d[part].items():
                rows.append({"Specification": spec, "Part": part,
                             "Term": names[k], "Coefficient": round(b, 4),
                             "s.e.": round(se, 4), "P": f"{p:.1e}"})
        rows.append({"Specification": spec, "Part": "combined",
                     "Term": "fold change, 1st to 99th percentile of size",
                     "Coefficient": round(d["combined_ratio"][0], 3),
                     "s.e.": "", "P": f"[{d['combined_ratio'][1]:.2f}, "
                                      f"{d['combined_ratio'][2]:.2f}]"})
        rows.append({"Specification": spec, "Part": "combined",
                     "Term": "replicates with a higher value at the largest size",
                     "Coefficient": round(d["share_rising"], 3), "s.e.": "",
                     "P": f"monotone in {d['share_monotone']:.3f}"})
    for cov, q in [(c, q) for c in con for q in con[c]]:
        r, lo, hi = con[cov][q]
        rows.append({"Specification": "B: adjusted contrast", "Part": q,
                     "Term": cov, "Coefficient": round(r, 3), "s.e.": "",
                     "P": f"[{lo:.2f}, {hi:.2f}]"})
    for spec in ("B", "A"):
        f = T / f"narrow_two_part_validation_{spec}.json"
        if not f.exists():
            continue
        v = json.load(open(f))
        for k, lab, n in [
            ("auc", "occurrence, area under the ROC curve", v["n_scored"]),
            ("cal_slope", "occurrence, calibration slope", v["n_scored"]),
            ("brier", "occurrence, Brier score", v["n_scored"]),
            ("intensity_r2_log10", "intensity among held-out positives, R2 on log10",
             v["n_positive"]),
            ("combined_r2", "combined prediction, R2 on observed intensity",
             v["n_scored"])]:
            rows.append({"Specification": f"{spec}: spatially separated validation",
                         "Part": f"{v['deg']:.0f} degree blocks, "
                                 f"{v['buffer_km']:.0f} km buffer",
                         "Term": lab, "Coefficient": round(v[k], 3),
                         "s.e.": "", "P": f"n = {n:,}"})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tableS9.csv", index=False)
    return d


def s20():
    s = json.load(open(T / "species_concentration_strata.json"))
    rows = []
    for r in s["strata"]:
        rows.append({"Block": "range-size quintile",
                     "Group": f"{r['range_lo']:,.0f} to {r['range_hi']:,.0f} km2",
                     "Species": r["n"],
                     "Contributing agglomerations, median": round(r["n_contrib_median"]),
                     "Largest-agglomeration share, median": f"{r['top1_median']*100:.1f}%",
                     "Largest-three share, median": f"{r['top3_median']*100:.1f}%",
                     "Spearman with converted share of range": round(r["rho"], 3),
                     "P": f"{r['p']:.1e}"})
    for b in s["bands"]:
        rows.append({"Block": "converted share of mapped range",
                     "Group": b["band"], "Species": b["n"],
                     "Contributing agglomerations, median": round(b["n_contrib_median"]),
                     "Largest-agglomeration share, median": f"{b['top1_median']*100:.1f}%",
                     "Largest-three share, median": f"{b['top3_median']*100:.1f}%",
                     "Spearman with converted share of range":
                         f"median range {b['range_median']:,.0f} km2",
                     "P": f"top-1 above 50% in {b['share_top1_gt50']*100:.1f}%"})
    for r in s.get("taxa", []):
        rows.append({"Block": "taxonomic class", "Group": r["taxon"],
                     "Species": r["n"],
                     "Contributing agglomerations, median": round(r["n_contrib_median"]),
                     "Largest-agglomeration share, median": f"{r['top1_median']*100:.1f}%",
                     "Largest-three share, median": f"{r['top3_median']*100:.1f}%",
                     "Spearman with converted share of range":
                         f"median range {r['range_median']:,.0f} km2",
                     "P": ""})
    for r in s["rule_sensitivity"]:
        rows.append({"Block": "boundary rule", "Group": r["band"],
                     "Species": f"{r['within']} / {r['centroid']} / {r['touches']}",
                     "Contributing agglomerations, median":
                         round(r["stable_n_contrib_median"]),
                     "Largest-agglomeration share, median":
                         f"{r['stable_top1_median']*100:.1f}%",
                     "Largest-three share, median": "",
                     "Spearman with converted share of range":
                         f"{r['all_three']} species in all three rules",
                     "P": f"top-1 above 50% in {r['stable_share_top1_gt50']*100:.1f}%"})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tableS10.csv", index=False)
    return d


if __name__ == "__main__":
    a, b = s19(), s20()
    print(f"tableS9.csv {a.shape}   tableS10.csv {b.shape}")
