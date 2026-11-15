"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
from tropics_config import SUPP_TABLE_DIR, TABLE_DIR
T = TABLE_DIR
OUT = SUPP_TABLE_DIR


TERMS = {"ls": "log10 year 2000 extent", "lset": "log10 setting richness + 1",
         "ladd": "log10 urban land added"}
SPEC = {"A: size, setting, land added": "A", "B: size, setting (published intensity spec)": "B"}
PARTS = {"p": "occurrence", "intensity_pos": "conditional intensity", "combined": "combined"}


def interval(lo, hi, d=2):
    return f"{lo:.{d}f} to {hi:.{d}f}"


def s19():
    m = json.load(open(T / "narrow_two_part_meta.json"))
    con = json.load(open(T / "narrow_two_part_contrasts_B.json"))
    rows = []
    for spec, d in m.items():
        if spec == "sample":
            continue
        for part in ("occurrence", "intensity"):
            for k, (b, se, p) in d[part].items():
                rows.append({"Specification": SPEC[spec],
                             "Part": "occurrence" if part == "occurrence" else "conditional intensity",
                             "Quantity": f"coefficient, {TERMS[k]}", "Estimate": f"{b:+.4f}",
                             "s.e.": f"{se:.4f}", "P": f"{p:.1e}", "95% interval": ""})
        if SPEC[spec] == "A":
            r, lo, hi = d["combined_ratio"]
            rows.append({"Specification": "A", "Part": "combined",
                         "Quantity": "fold change, 1st to 99th percentile of year 2000 size",
                         "Estimate": f"{r:.3f}", "s.e.": "", "P": "", "95% interval": interval(lo, hi)})
        rows.append({"Specification": SPEC[spec], "Part": "combined",
                     "Quantity": "Duan smearing factor", "Estimate": f"{d['smearing']:.2f}",
                     "s.e.": "", "P": "", "95% interval": ""})
        rows.append({"Specification": SPEC[spec], "Part": "combined",
                     "Quantity": "share of replicates higher at the largest size than the smallest",
                     "Estimate": f"{d['share_rising']:.3f}", "s.e.": "", "P": "", "95% interval": ""})
        rows.append({"Specification": SPEC[spec], "Part": "combined",
                     "Quantity": "share of replicates rising across the whole size grid",
                     "Estimate": f"{d['share_monotone']:.3f}", "s.e.": "", "P": "", "95% interval": ""})
    for cov in con:
        for q, (r, lo, hi) in con[cov].items():
            rows.append({"Specification": "B", "Part": PARTS[q],
                         "Quantity": f"fold change, 1st to 99th percentile of {cov}",
                         "Estimate": f"{r:.3f}", "s.e.": "", "P": "", "95% interval": interval(lo, hi)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tableS9.csv", index=False)
    val = []
    labels = {"auc": "occurrence, area under the ROC curve", "cal_slope": "occurrence, calibration slope",
              "brier": "occurrence, Brier score",
              "intensity_r2_log10": "conditional intensity, R² on log10 among held-out positives",
              "combined_r2": "combined prediction, R² on observed intensity"}
    for spec in ("A", "B"):
        f = T / f"narrow_two_part_validation_{spec}.json"
        if not f.exists():
            continue
        v = json.load(open(f))
        for k, lab in labels.items():
            val.append({"Specification": spec, "Metric": lab, "Value": f"{v[k]:.3f}",
                        "Agglomerations scored": f"{(v['n_positive'] if k == 'intensity_r2_log10' else v['n_scored']):,}"})
    pd.DataFrame(val).to_csv(OUT / "tableS17.csv", index=False)
    return d


def s20():
    s = json.load(open(T / "species_concentration_strata.json"))
    rows = []
    def row(grouping, group, r, rho="", p=""):
        return {"Grouping": grouping, "Group": group, "Species": f"{r['n']:,}",
                "Median global range (km²)": f"{r['range_median']:,.0f}" if "range_median" in r else "",
                "Contributing agglomerations, median": f"{round(r['n_contrib_median'])}",
                "Largest-agglomeration share, median": f"{r['top1_median']*100:.1f}%",
                "Largest-three share, median": f"{r['top3_median']*100:.1f}%",
                "Largest agglomeration above 50%": f"{r['share_top1_gt50']*100:.1f}%" if "share_top1_gt50" in r else "",
                "Spearman, converted share and largest share": rho, "P": p}
    for b in s["bands"]:
        rho = p = ""
        if b["band"] == "all" and "rho_unstratified" in s:
            u = s["rho_unstratified"]
            rho, p = f"{u[0]:+.3f}", f"{u[1]:.2f}"
        rows.append(row("converted share of mapped range", b["band"], b, rho, p))
    for r in s["strata"]:
        rows.append(row("range-size quintile", f"{r['range_lo']:,.0f} to {r['range_hi']:,.0f} km²", r,
                        f"{r['rho']:+.3f}", f"{r['p']:.1e}"))
    for r in s.get("taxa", []):
        rows.append(row("taxonomic class", r["taxon"].capitalize(), r))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tableS10.csv", index=False)
    br = [{"Converted share of mapped range": r["band"], "Wholly inside": f"{r['within']:,}",
           "Center inside": f"{r['centroid']:,}", "Touches": f"{r['touches']:,}",
           "In all three rules": f"{r['all_three']:,}",
           "Contributing agglomerations, median": f"{round(r['stable_n_contrib_median'])}",
           "Largest-agglomeration share, median": f"{r['stable_top1_median']*100:.1f}%",
           "Largest agglomeration above 50%": f"{r['stable_share_top1_gt50']*100:.1f}%"}
          for r in s["rule_sensitivity"]]
    pd.DataFrame(br).to_csv(OUT / "tableS18.csv", index=False)
    return d


if __name__ == "__main__":
    a, b = s19(), s20()
    print(f"tableS9.csv {a.shape}   tableS10.csv {b.shape}")
