"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tropics_config import TABLE_DIR, INTERIM_DIR
import tropics_sensitivity as tsen
import tropics_scaling as ts

PUBLISHED = TABLE_DIR / "sensitivity_delineation_v2.csv"
STATE = TABLE_DIR / "delineation_fixed.json"
OUT = TABLE_DIR / "delineation_fixed_threshold.csv"
DELIN_OUT = TABLE_DIR / "sensitivity_delineation_v3.csv"
CACHE = INTERIM_DIR / "sens_cfg{}.npz"

TOL_THRESHOLD = 1e-6
TOL_RESPONSE_PPM = 50.0
TOL_BETA_SE = 0.10


def repaired_cache():
    import tropics_rangecache as rc
    try:
        with open(rc.INDEX) as fh:
            return any('"repair": true' in line for line in fh)
    except OSError:
        return False


def config_tables(i, cells0=None):
    cfg = tsen.CONFIGS[i]
    cells, cities, _ = tsen.config_cities(cfg["fraction"], cfg["link"],
                                          cfg["floor"], cells=cells0)
    return cfg, cells, cities


def pairs_for(i, cells, cities, sp, part_sp):
    import tropics_rangecache as rc
    fp = np.asarray(rc.fingerprint(), dtype=np.int64)
    p = Path(str(CACHE).format(i))
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            same_cache = ("cache_fp" in z
                          and bool((z["cache_fp"] == fp).all()))
            match = (same_cache and z["city_ids"].size == len(cities)
                     and bool((z["city_ids"] == cities.city_id.to_numpy()).all()))
            if not same_cache:
                print(f"  stored pairs for config {i} were built against a "
                      "different range cache; recomputing", flush=True)
            if match:
                pairs = pd.DataFrame(dict(sp_i=z["pair_sp"],
                                          city_id=z["pair_city"],
                                          conv_km2=z["pair_conv"]))
                setting = pd.DataFrame(dict(sp_i=z["set_sp"],
                                            city_id=z["set_city"]))
                match = match and "pool_sp" in z.files
                pool = z["pool_sp"] if "pool_sp" in z.files else None
        if match:
            return pairs, setting, pool, "cache"
        print(f"  stored pairs for config {i} do not match this delineation; "
              "recomputing", flush=True)
    sp2, pairs, setting, pool = tsen.cost_from_cache(cells, cities, sp, part_sp,
                                                     with_pool=True)
    np.savez_compressed(
        p, cache_fp=fp, city_ids=cities.city_id.to_numpy(),
        pair_sp=pairs.sp_i.to_numpy(), pair_city=pairs.city_id.to_numpy(),
        pair_conv=pairs.conv_km2.to_numpy(),
        set_sp=setting.sp_i.to_numpy(), set_city=setting.city_id.to_numpy(),
        pool_sp=pool)
    return pairs, setting, pool, "computed"


def fit_at(cities, sp, pairs, setting, threshold, pool=None):
    t, thr, own = tsen.assemble(cities, sp, pairs, setting, threshold=threshold,
                                pool=pool)
    t = tsen.attach_groups(t)
    r = ts.fit_scaling(t, response="cost_per_km2", size="urban_km2_base",
                       setting="setting_richness", spatial=True)
    return t, thr, own, r


def run_config(i, fixed_threshold, sp, part_sp, cells0):
    t0 = time.time()
    cfg, cells, cities = config_tables(i, cells0)
    pairs, setting, pool, src = pairs_for(i, cells, cities, sp, part_sp)
    rows = []
    passes = [("own quartile", None)]
    if fixed_threshold is not None:
        passes.append(("fixed", fixed_threshold))
    for how, thr_in in passes:
        t, thr, own, r = fit_at(cities, sp, pairs, setting, thr_in, pool)
        rows.append(dict(
            config=cfg["label"], threshold_rule=how, fraction=cfg["fraction"],
            link=cfg["link"], floor=cfg["floor"], n_cities=int(len(t)),
            n_pairs=int(len(pairs)),
            urban_added_km2=float(t.urban_added_km2.sum()),
            mean_setting=float(t.setting_richness.mean()),
            own_threshold_km2=own,
            threshold_km2=thr, converted_km2=float(t.cost_km2.sum()),
            beta=r["beta"], ci_lower=r["ci_lower"], ci_upper=r["ci_upper"],
            gamma_setting=r.get("gamma_setting", np.nan), R2=r["R2"],
            pairs_from=src, seconds=round(time.time() - t0, 1)))
        print(json.dumps(rows[-1], default=str), flush=True)
    return rows


def check_against_published(rows):
    if not PUBLISHED.exists():
        print("  no published sensitivity on disk; not placed against it")
        return []
    pub = pd.read_csv(PUBLISHED).set_index("label")
    post_repair = repaired_cache()
    if post_repair:
        print(f"  {PUBLISHED.name} was measured before the range cache was "
              "repaired, so it is a reference and not a target")
    bad = []
    for r in rows:
        if r["threshold_rule"] != "own quartile" or r["config"] not in pub.index:
            continue
        p = pub.loc[r["config"]]
        d_thr = abs(float(p.narrow_threshold_km2) - r["own_threshold_km2"])
        d_cost = r["converted_km2"] - float(p.cost_km2)
        ppm = 1e6 * d_cost / float(p.cost_km2)
        d_pairs = r["n_pairs"] - int(p.n_pairs)
        se = (float(p.ci_upper) - float(p.ci_lower)) / (2 * 1.96)
        d_beta = r["beta"] - float(p.beta)
        in_se = d_beta / se if se > 0 else float("inf")

        same_cfg = int(p.n_cities) == r["n_cities"]
        ok_resp = abs(ppm) < TOL_RESPONSE_PPM
        ok_beta = abs(in_se) < TOL_BETA_SE
        if post_repair:
            verdict = ("DIFFERENT DELINEATION" if not same_cfg else
                       "superseded by the repair")
        else:
            same_cfg = same_cfg and d_thr < TOL_THRESHOLD
            verdict = ("DIFFERENT CONFIGURATION" if not same_cfg else
                       "response moved" if not ok_resp else
                       "exponent moved" if not ok_beta else "re-measured")
        print(f"  {verdict:<24} {r['config']}")
        print(f"     cities {r['n_cities']} vs {int(p.n_cities)}   "
              f"threshold delta {d_thr:.2e} km2")
        print(f"     converted {d_cost:+,.1f} km2 ({ppm:+.1f} ppm), "
              f"pairs {d_pairs:+,}")
        print(f"     exponent {d_beta:+.2e} ({in_se:+.3f} of its SE "
              f"{se:.2e})")
        if verdict not in ("re-measured", "superseded by the repair"):
            bad.append((r["config"], verdict))
    return bad


def summarize(d):
    own = d[d.threshold_rule == "own quartile"].set_index("config")
    fix = d[d.threshold_rule == "fixed"].set_index("config")
    both = own.index.intersection(fix.index)
    if len(both) < 2:
        return
    s_own = own.loc[both, "beta"]
    s_fix = fix.loc[both, "beta"]
    print(f"\nexponent span across {len(both)} configurations")
    print(f"  own quartile per configuration  {s_own.max() - s_own.min():.5f}")
    print(f"  one threshold throughout        {s_fix.max() - s_fix.min():.5f}")
    print(f"  difference, the classification  "
          f"{(s_own.max() - s_own.min()) - (s_fix.max() - s_fix.min()):+.5f}")


def _selftest():
    from tropics_config import NARROW_RANGE_PCTL

    rng = np.random.default_rng(0)
    n_city, n_sp = 40, 200
    cities = pd.DataFrame(dict(
        city_id=np.arange(1, n_city + 1),
        lon=rng.uniform(-60, -40, n_city), lat=rng.uniform(-10, 10, n_city),
        n_cells=rng.integers(5, 50, n_city),
        urban_km2_base=rng.uniform(5, 500, n_city),
        urban_km2_final=rng.uniform(5, 600, n_city),
        urban_added_km2=rng.uniform(1, 60, n_city)))
    sp = pd.DataFrame(dict(
        sp_i=np.arange(n_sp),
        sp_id=[f"SP_{i}" for i in range(n_sp)],
        range_km2=np.exp(rng.normal(8, 2, n_sp)),
        sci_name=[f"Genus species{i}" for i in range(n_sp)],
        category=rng.choice(["LC", "VU", "EN", "CR"], n_sp),
        taxon=np.array(["MAMMALIA"] * n_sp)))
    k = 600
    pairs = pd.DataFrame(dict(
        sp_i=rng.integers(0, n_sp, k),
        city_id=rng.integers(1, n_city + 1, k),
        conv_km2=rng.uniform(0.01, 5.0, k))).drop_duplicates(["sp_i", "city_id"])
    setting = pairs[["sp_i", "city_id"]].copy()

    import tropics_assemble as ta
    real = ta.merged_categories
    ta.merged_categories = lambda s: s
    try:
        pool = set(pairs.sp_i) | set(setting.sp_i)
        inpool = sp.sp_i.isin(pool) & (sp.range_km2 > 0)
        own_expected = float(np.percentile(sp.loc[inpool, "range_km2"],
                                           NARROW_RANGE_PCTL))

        t_own, thr_own, own = tsen.assemble(cities, sp, pairs, setting)
        assert abs(own - own_expected) < 1e-9, (own, own_expected)
        assert thr_own == own

        t_same, thr_same, own2 = tsen.assemble(cities, sp, pairs, setting,
                                               threshold=own_expected)
        assert thr_same == own_expected and abs(own2 - own_expected) < 1e-9
        pd.testing.assert_frame_equal(t_own, t_same)

        t_lo, thr_lo, _ = tsen.assemble(cities, sp, pairs, setting,
                                        threshold=0.0)
        t_hi, thr_hi, _ = tsen.assemble(cities, sp, pairs, setting,
                                        threshold=1e12)
        assert thr_lo == 0.0 and thr_hi == 1e12
        lo, mid, hi = (t_lo.cost_km2.sum(), t_own.cost_km2.sum(),
                       t_hi.cost_km2.sum())
        assert lo <= mid <= hi, (lo, mid, hi)
        assert lo < hi, "the threshold changed nothing, so nothing is being tested"

        for c in ("urban_km2_base", "urban_added_km2", "setting_richness"):
            assert np.allclose(t_lo[c], t_hi[c]), c
    finally:
        ta.merged_categories = real

    import tempfile

    def save(path, pr, st):
        np.savez_compressed(path, city_ids=cities.city_id.to_numpy(),
                            pair_sp=pr.sp_i.to_numpy(),
                            pair_city=pr.city_id.to_numpy(),
                            pair_conv=pr.conv_km2.to_numpy(),
                            set_sp=st.sp_i.to_numpy(),
                            set_city=st.city_id.to_numpy())

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "sens_selftest.npz"
        save(p, pairs, setting)
        with np.load(p, allow_pickle=False) as z:
            back = pd.DataFrame(dict(sp_i=z["pair_sp"], city_id=z["pair_city"],
                                     conv_km2=z["pair_conv"]))
        pd.testing.assert_frame_equal(pairs.reset_index(drop=True),
                                      back.reset_index(drop=True),
                                      check_dtype=False)
        save(p, pairs.iloc[:5], setting.iloc[:5])
        with np.load(p, allow_pickle=False) as z:
            assert z["pair_sp"].size == 5
        p.unlink()
        assert not p.exists()
    print("ok  fixed threshold changes the classification and nothing else; "
          "pairs round-trip, replace and delete")


def main():
    args = sys.argv[1:]
    _selftest()
    if "--selftest" in args:
        return
    only = None
    if "--only" in args:
        only = [int(v) for v in args[args.index("--only") + 1].split(",")]

    done = json.loads(STATE.read_text()) if STATE.exists() else {}
    import tropics_rangecache as rc
    print(f"cache {'carries' if repaired_cache() else 'carries no'} repair "
          "records")
    sp, part_sp = rc.load_index()
    cells0 = tsen.master_cells()

    if "0" not in done:
        print("configuration 0 first: it sets the threshold the others hold")
        done["0"] = run_config(0, None, sp, part_sp, cells0)
        STATE.write_text(json.dumps(done, indent=1, default=str))
    fixed = float([r for r in done["0"]
                   if r["threshold_rule"] == "own quartile"][0]
                  ["own_threshold_km2"])
    print(f"threshold held at {fixed:,.3f} km2, the headline configuration's "
          "own quartile in this run")

    if len(done["0"]) == 1:
        done["0"].append(dict(done["0"][0], threshold_rule="fixed",
                              threshold_km2=fixed))
        STATE.write_text(json.dumps(done, indent=1, default=str))

    for i in range(1, len(tsen.CONFIGS)):
        if only is not None and i not in only:
            continue
        if str(i) in done:
            print(f"config {i} already done; skipping")
            continue
        rows = run_config(i, fixed, sp, part_sp, cells0)
        done[str(i)] = rows
        STATE.write_text(json.dumps(done, indent=1, default=str))

    rows = [r for v in done.values() for r in v]
    if not rows:
        return
    d = pd.DataFrame(rows)
    order = {c["label"]: i for i, c in enumerate(tsen.CONFIGS)}
    d = (d.assign(_o=[order.get(v, 99) for v in d.config],
                  _t=[0 if v == "own quartile" else 1 for v in d.threshold_rule])
           .sort_values(["_o", "_t"]).drop(columns=["_o", "_t"])
           .reset_index(drop=True))
    own = d[d.threshold_rule == "own quartile"].copy()
    if len(own) == len(tsen.CONFIGS):
        v3 = pd.DataFrame(dict(
            label=own.config, fraction=own.fraction, link=own.link,
            floor=own.floor, n_cities=own.n_cities, n_pairs=own.n_pairs,
            urban_added_km2=own.urban_added_km2, cost_km2=own.converted_km2,
            mean_setting=own.mean_setting,
            narrow_threshold_km2=own.own_threshold_km2, beta=own.beta,
            ci_lower=own.ci_lower, ci_upper=own.ci_upper,
            gamma_setting=own.gamma_setting, R2=own.R2,
            excludes_zero=own.ci_lower * own.ci_upper > 0))
        v3.to_csv(DELIN_OUT, index=False)
        print(f"delineation table reissued to {DELIN_OUT.name}; point "
              "tropics_sitables.s4_delineation at it")

    print("\nagainst the stored delineation table, own-quartile rows only:")
    bad = check_against_published(rows)
    d.to_csv(OUT, index=False)
    summarize(d)
    print(f"\nwritten to {OUT}")
    if bad:
        print("\nnot a clean re-measurement:")
        for label, why in bad:
            print(f"  {why}: {label}")
        print("The spans above are still a valid decomposition as long as "
              "every configuration in them came from this run; they are not "
              "comparable with the stored table row by row.")


if __name__ == "__main__":
    main()
