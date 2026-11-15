"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tropics_models as p4
from tropics_config import TABLE_DIR
from tropics_scaling import fit_scaling

OUT = TABLE_DIR / "setting_definition_compare.csv"

BUILTUP_FILE = TABLE_DIR / "setting_footprint.csv"
BUILTUP_COL = "setting_fp"
DEFS = [("built-up fraction 0.20", BUILTUP_COL),
        ("setting on the 2020 extent", "setting_richness"),
        ("year 2000 footprint (published)", "setting_footprint")]


def one(d, col):
    r = fit_scaling(d, response="cost_per_km2", size="urban_km2_base",
                    setting=col, spatial=True)
    x = d.copy()
    x["log_setting"] = np.log10(x[col].clip(lower=0.5))
    sh = p4.shapley(x)
    return dict(n=int(r["n"]), n_zero=int((d[col] <= 0).sum()),
                beta=float(r["beta"]), ci_lower=float(r["ci_lower"]),
                ci_upper=float(r["ci_upper"]),
                gamma_setting=float(r["gamma_setting"]),
                shapley_setting=float(sh["shapley"]["setting"]),
                shapley_size=float(sh["shapley"]["size"]),
                full_R2=float(sh["full_R2"]))


def frame():
    d = p4.table()
    f = pd.read_csv(BUILTUP_FILE)
    if BUILTUP_COL not in f.columns:
        raise SystemExit(f"{BUILTUP_FILE.name} has no {BUILTUP_COL} column")
    n0 = len(d)
    d = d.merge(f[["city_id", BUILTUP_COL]], on="city_id", how="left")
    if len(d) != n0:
        raise SystemExit("the built-up-fraction join changed the sample size")
    d[BUILTUP_COL] = d[BUILTUP_COL].fillna(0.0)
    return d


def zeroed(d):
    x = d.copy()
    x["setting_zeroed"] = np.where(x[BUILTUP_COL] > 0, x.setting_footprint, 0.0)
    return x


def missing_profile(d):
    m = d[d[BUILTUP_COL] <= 0]
    biome = m.biome.astype(str).value_counts()
    country = m.country.astype(str).value_counts()
    return dict(n=int(len(m)),
                top_biome=str(biome.index[0]), top_biome_n=int(biome.iloc[0]),
                countries=[(str(k), int(v)) for k, v in country.head(3).items()],
                median_intensity=float(m.cost_per_km2.median()),
                study_median=float(d.cost_per_km2.median()))


def agreement(d):
    both = d[(d[BUILTUP_COL] > 0) & (d.setting_footprint > 0)]
    rho = float(both[BUILTUP_COL].corr(both.setting_footprint, method="spearman"))
    same = float((both[BUILTUP_COL] == both.setting_footprint).mean())
    return dict(definition="agreement where both are defined",
                column=f"{BUILTUP_COL} vs setting_footprint", n=int(len(both)),
                n_zero=int((d[BUILTUP_COL] <= 0).sum()),
                beta=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                gamma_setting=np.nan, shapley_setting=np.nan,
                shapley_size=np.nan, full_R2=np.nan,
                spearman=rho, identical_share=same)


def run():
    d = frame()
    rows = []
    for label, col in DEFS:
        rows.append(dict(definition=label, column=col, **one(d, col)))
        print(f"  {label:32s} beta {rows[-1]['beta']:+.4f}  "
              f"gamma {rows[-1]['gamma_setting']:+.3f}  "
              f"setting share {rows[-1]['shapley_setting'] * 100:5.1f}%  "
              f"R2 {rows[-1]['full_R2']:.3f}", flush=True)
    x = zeroed(d)
    rows.append(dict(definition="year 2000 footprint, zeroed where the "
                                "built-up rule returns no geometry",
                     column="setting_zeroed", **one(x, "setting_zeroed")))
    r = rows[-1]
    print(f"  {'published, zeroed on those rows':32s} beta {r['beta']:+.4f}  "
          f"gamma {r['gamma_setting']:+.3f}  "
          f"setting share {r['shapley_setting'] * 100:5.1f}%  "
          f"R2 {r['full_R2']:.3f}")
    mp = missing_profile(d)
    print(f"  without built-up geometry: {mp['n']} agglomerations, "
          f"{mp['top_biome_n']} in {mp['top_biome']}, "
          f"{mp['countries']}, median intensity {mp['median_intensity']:.1f} "
          f"against the study median {mp['study_median']:.1f}")
    ag = agreement(d)
    ag.update(top_biome=mp["top_biome"], top_biome_n=mp["top_biome_n"],
              median_intensity=mp["median_intensity"],
              study_median=mp["study_median"],
              countries="; ".join(f"{k} {v}" for k, v in mp["countries"]))
    print(f"  both defined on {ag['n']:,} agglomerations, Spearman "
          f"{ag['spearman']:.3f}, identical for {ag['identical_share'] * 100:.0f}%")
    rows.append(ag)
    f = pd.DataFrame(rows)
    f.to_csv(OUT, index=False)
    print(f"written to {OUT.name}")
    return f


def _selftest(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    size = 10 ** rng.uniform(0.0, 2.5, n)
    setting = np.clip(rng.poisson(30, n), 1, None).astype(float)
    d = pd.DataFrame(dict(
        urban_km2_base=size, setting_real=setting,
        setting_noise=np.clip(rng.poisson(30, n), 1, None).astype(float),
        country=rng.integers(0, 12, n).astype(str),
        realm=rng.integers(0, 4, n).astype(str),
        biome=rng.integers(0, 5, n).astype(str),
        form_infill_share=rng.uniform(0, 1, n),
        form_leapfrog_share=rng.uniform(0, 1, n),
        lon=rng.uniform(-80, 120, n), lat=rng.uniform(-20, 20, n)))
    d["cost_per_km2"] = 10 ** (1.0 + 0.9 * np.log10(d.setting_real)
                               + rng.normal(0, 0.05, n))
    d["log_cost"] = np.log10(d.cost_per_km2)
    d["log_size"] = np.log10(d.urban_km2_base)
    got = {}
    for col in ("setting_real", "setting_noise"):
        x = d.copy()
        x["log_setting"] = np.log10(x[col].clip(lower=0.5))
        sh = p4.shapley(x)
        got[col] = sh["shapley"]["setting"]
        print(f"  setting share with {col:13s} {got[col] * 100:6.2f}%")
    assert got["setting_real"] > 0.5, got
    assert got["setting_noise"] < 0.02, got
    assert got["setting_real"] > 20 * max(got["setting_noise"], 1e-6)
    print("the setting share answers to the covariate given   ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        run()
