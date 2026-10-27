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
from tropics_config import TABLE_DIR, EXCLUDED_TAXA
from tropics_cost import PAIRS_PATH

OUT_DECOMP = TABLE_DIR / "objective_decomposition.csv"
OUT_GROUPS = TABLE_DIR / "driver_groups.csv"
THREAT = ("CR", "EN", "VU")


def build(subset=None):
    d = tn.analysis_sample()
    z = np.load(PAIRS_PATH, allow_pickle=False)
    cat, tax, _sci, ar = tfl.flags(PAIRS_PATH)
    thr = float(json.loads((TABLE_DIR / "cities_cost_meta.json").read_text())
                ["narrow_threshold_km2"])
    keep = ~np.isin(tax, [t.upper() for t in EXCLUDED_TAXA])
    threat = np.isin(cat, list(THREAT)) & keep
    narrow = (ar > 0) & (ar <= thr) & keep
    concern = threat | narrow

    sel = concern.copy()
    if subset in ("AVES", "MAMMALIA", "AMPHIBIA", "REPTILIA"):
        sel &= (tax == subset)
    elif subset == "herpetofauna":
        sel &= np.isin(tax, ["AMPHIBIA", "REPTILIA"])
    elif subset == "narrow":
        sel = narrow
    elif subset == "threatened":
        sel = threat

    ps, pc, pv = z["pair_sp"], z["pair_city"], z["pair_conv"]
    m = sel[ps]
    conv = (pd.DataFrame(dict(city_id=pc[m], v=pv[m]))
            .groupby("city_id").v.sum().rename("conv_km2"))
    w = np.where(ar[ps] > 0, pv / np.maximum(ar[ps], 1e-6), 0.0)
    rar = (pd.DataFrame(dict(city_id=pc[m], v=w[m]))
           .groupby("city_id").v.sum().rename("rar"))
    x = d.join(conv, on="city_id").join(rar, on="city_id")
    x["conv_km2"] = x.conv_km2.fillna(0.0)
    x["rar"] = x.rar.fillna(0.0)
    x["unw_per_km2"] = x.conv_km2 / x.urban_added_km2
    x["rar_per_km2"] = x.rar / x.urban_added_km2
    x["mean_range_km2"] = np.where(x.rar > 0, x.conv_km2 / x.rar, np.nan)
    return x


def _fit(x, col):
    r = trf.fit_scaling(x, response=col, size="urban_km2_base",
                        setting="setting_footprint", spatial=True)
    return dict(n=int(r["n"]), beta=float(r["beta"]),
                lo=float(r["ci_lower"]), hi=float(r["ci_upper"]),
                gamma=float(r.get("gamma_setting", np.nan)))


def decompose(x, label):
    y = x[(x.unw_per_km2 > 0) & (x.rar_per_km2 > 0) & x.mean_range_km2.notna()]
    u = _fit(y, "unw_per_km2")
    r = _fit(y, "rar_per_km2")
    m = _fit(y, "mean_range_km2")
    return dict(subset=label, n=u["n"],
                beta_unweighted=u["beta"], beta_rarity=r["beta"],
                beta_mean_range=m["beta"],
                identity_residual=r["beta"] - (u["beta"] - m["beta"]),
                median_mean_range_km2=float(y.mean_range_km2.median()),
                gamma_unweighted=u["gamma"], gamma_rarity=r["gamma"])


def _selftest():
    x = build()
    y = x[(x.unw_per_km2 > 0) & (x.rar_per_km2 > 0)]
    lhs = np.log10(y.rar_per_km2)
    rhs = np.log10(y.unw_per_km2) - np.log10(y.mean_range_km2)
    assert float(np.abs(lhs - rhs).max()) < 1e-9, float(np.abs(lhs - rhs).max())
    print(f"ok  identity holds city by city on n={len(y)} "
          f"(max deviation {float(np.abs(lhs - rhs).max()):.2e})")


def run():
    rows = [decompose(build(), "All species of concern")]
    for s, label in (("threatened", "Threatened only"),
                     ("narrow", "Narrow range only"),
                     ("AVES", "Birds"), ("MAMMALIA", "Mammals"),
                     ("herpetofauna", "Amphibians and reptiles")):
        try:
            rows.append(decompose(build(s), label))
        except Exception as exc:
            rows.append(dict(subset=label, n=0,
                             identity_residual=np.nan,
                             median_mean_range_km2=np.nan,
                             beta_unweighted=np.nan, beta_rarity=np.nan,
                             beta_mean_range=np.nan,
                             gamma_unweighted=np.nan, gamma_rarity=np.nan))
            print(f"  {label}: {type(exc).__name__}: {exc}", flush=True)
        print(f"  done {label}", flush=True)
    dec = pd.DataFrame(rows)

    x = build()
    grp = []
    for key in ("realm", "continent"):
        if key not in x.columns:
            continue
        for g, sub in x.groupby(key):
            if len(sub) < 150:
                continue
            try:
                r = decompose(sub, f"{key}: {g}")
            except Exception:
                continue
            r["grouping"] = key
            r["land_added_km2"] = float(sub.urban_added_km2.sum())
            r["median_setting"] = float(sub.setting_footprint.median())
            grp.append(r)
            print(f"  done {key} {g}", flush=True)
    return dec, pd.DataFrame(grp)


if __name__ == "__main__":
    _selftest()
    dec, grp = run()
    dec.to_csv(OUT_DECOMP, index=False)
    grp.to_csv(OUT_GROUPS, index=False)
    pd.set_option("display.width", 230)
    print()
    print(dec.to_string(index=False))
    print()
    print(grp.to_string(index=False))
