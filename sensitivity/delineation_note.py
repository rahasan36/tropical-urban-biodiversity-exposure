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

from tropics_config import TABLE_DIR

_V3 = TABLE_DIR / "sensitivity_delineation_v3.csv"
SENS = _V3 if _V3.exists() else TABLE_DIR / "sensitivity_delineation_v2.csv"
CITIES = TABLE_DIR / "cities_cost.csv"
OUT = TABLE_DIR / "delineation_reconciliation.csv"


def run():
    s = pd.read_csv(SENS)
    head = s.iloc[0]
    meta = json.loads((TABLE_DIR / "cities_cost_meta.json").read_text())

    d = pd.read_csv(CITIES)
    main_conc = float(d.cost_km2.sum())
    main_added = float(d.urban_added_km2.sum())
    assert len(d) == int(meta["n_cities"]), (len(d), meta["n_cities"])

    rec = pd.DataFrame([
        dict(quantity="Agglomerations delineated",
             main_pipeline=meta["n_cities"], sensitivity_headline=head.n_cities,
             relative_difference=0.0),
        dict(quantity="Urban land added (km2)",
             main_pipeline=main_added,
             sensitivity_headline=head.urban_added_km2,
             relative_difference=head.urban_added_km2 / main_added - 1.0),
        dict(quantity="Converted range, species of concern (km2)",
             main_pipeline=main_conc, sensitivity_headline=head.cost_km2,
             relative_difference=head.cost_km2 / main_conc - 1.0),
        dict(quantity="Narrow-range threshold (km2)",
             main_pipeline=meta["narrow_threshold_km2"],
             sensitivity_headline=head.narrow_threshold_km2,
             relative_difference=(head.narrow_threshold_km2
                                  / meta["narrow_threshold_km2"] - 1.0)),
    ])

    w = s.copy()
    w["threshold_vs_headline"] = w.narrow_threshold_km2 / head.narrow_threshold_km2 - 1.0
    w["beta_minus_headline"] = w.beta - head.beta
    w["n_vs_headline"] = w.n_cities / head.n_cities - 1.0
    w = w[["label", "fraction", "link", "floor", "n_cities", "n_vs_headline",
           "narrow_threshold_km2", "threshold_vs_headline", "beta",
           "ci_lower", "ci_upper", "beta_minus_headline", "gamma_setting",
           "excludes_zero"]]
    return rec, w


def _selftest():
    rec, w = run()
    g = rec.set_index("quantity").relative_difference
    assert abs(float(g["Agglomerations delineated"])) < 1e-12
    assert abs(float(g["Urban land added (km2)"])) < 1e-6, \
        "the two pipelines do not agree on the land added; the rows below are "\
        "then not a reconciliation"
    conc = float(g["Converted range, species of concern (km2)"])
    assert -0.02 < conc < 0.0, conc
    assert bool(w.excludes_zero.all()), "a configuration lost the size effect"
    assert abs(float(w.iloc[0].beta_minus_headline)) < 1e-12
    print(f"ok  identical delineation and land added; the two measurement paths "
          f"differ by {conc * 100:+.2f}% on the species-of-concern total; "
          f"every configuration keeps the size effect")


RECON = TABLE_DIR / "delineation_reconciliation_pipelines.csv"

if __name__ == "__main__":
    _selftest()
    rec, w = run()
    w.to_csv(OUT, index=False)
    rec.to_csv(RECON, index=False)
    pd.set_option("display.width", 200)
    print(rec.to_string(index=False))
    print()
    print(w.to_string(index=False))
