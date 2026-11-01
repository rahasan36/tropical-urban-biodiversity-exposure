"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
CODE_DIRS = ["common", "measurement", "sensitivity", "temporal", "models", "reallocation",
             "supplementary_tables", "figures/plotting"]
PYTHONPATH = os.pathsep.join(str(REPO / d) for d in CODE_DIRS)

RESOLUTION_KEYS = ["30:centroid", "30:touches", "30:within", "60:centroid", "120:centroid", "60:touches"]

MEASUREMENT = [
    ("place_names", []),
    ("tropics_urban", "m.build_mosaic()"),
    ("tropics_cities", "m.run_delineation()"),
    ("tropics_measure", "m.measure()"),
    ("tropics_rangecache", "m.build(budget_s=1e9)"),
    ("tropics_cost", "m.run_pass(budget_s=1e9)"),
    ("tropics_assemble", "m.build()"),
    ("tropics_setting", "m.build_neighbourhoods()"),
    ("tropics_setting", "m.run_pass(budget_s=1e9)"),
    ("tropics_setting", "m.write_table()"),
    ("tropics_delin", "[m.step(k, 1e9) for k in range(len(m.CONFIGS))]"),
    ("tropics_delin", ["finish"]),
    ("delineation_fixed", "m.main()"),
    ("setting_rebuild", "m.main()"),
    ("birdlife_union", ["--budget", "1e9"]),
    ("tropics_form", "m.run(budget_s=1e9, want_habitat=False, tag='F')"),
    ("tropics_form", "m.run(budget_s=1e9)"),
    ("tropics_habpass", "m.run_cities(budget_s=1e9)"),
    ("tropics_form", "m.write_city_tables()"),
    ("tropics_surfaces", "m.save_urban(); m.richness_surfaces()"),
    ("tropics_radius", "m.step(budget_s=1e9)"),
    ("tropics_radius", "m.write_taxon_settings()"),
    *[("resolution_resume", ["advance", k, "1e9"]) for k in RESOLUTION_KEYS],
    *[("resolution_resume", ["finish", k]) for k in RESOLUTION_KEYS],
    ("resolution_test", []),
    ("species_boundary", []),
    ("species_concentration", "m.part1_strata(m.part1())"),
    ("tropics_results", []),
    ("tropics_crosscheck", []),
]

TEMPORAL = [
    ("tropics_temporal", ["tiles", "1e9"]),
    ("tropics_temporal", ["assemble"]),
    ("tropics_temporal", ["hits", "1e9"]),
    ("tropics_temporal", ["delin"]),
    ("tropics_temporal", ["pairs", "1e9"]),
    ("temporal_analysis", ["run"]),
    ("temporal_analysis", "[m.fits(j) for j in range(5)]; m.summarize(); m.realms()"),
    ("temporal_size", []),
    ("temporal_subregion", ["run"]),
    ("temporal_species", []),
]

ANALYSIS = [
    ("tropics_refit", []),
    ("tropics_analysis", []),
    ("tropics_cluster", []),
    ("tropics_models", []),
    ("tropics_sizevars", []),
    ("cf_release", ["300", "1e9"]),
    ("cf_run", []),
    ("cf_bands", ["response"]),
    ("cf_bands", ["realm"]),
    ("cf_influence", ["1e9"]),
    ("cf_boundary_check", ["chunk", "0", "100"]),
    ("cf_boundary_check", ["chunk", "1", "100"]),
    ("cf_boundary_check", ["chunk", "2", "100"]),
    ("cf_boundary_check", ["aggregate"]),
    ("r2_components", []),
    ("landcover_disagreement", []),
    ("drivers", []),
    ("spatial_cv_run", []),
    ("delineation_note", []),
    ("threshold_isolate", []),
    ("setting_definition_compare", []),
    ("species_ledger", []),
    ("weighting_vs_sample", []),
    ("bird_union_effect", []),
    ("bird_sensitivity", []),
    ("species_concentration", "m.part2(); m.part2_validate(spec='A'); m.part2_validate(spec='B'); m.part2_contrasts(spec='B')"),
    ("region_overlap", []),
    ("region_search", []),
    ("tropics_numbers", "m.run(); m.export_fig_tables()"),
    ("recency_numbers", []),
]

TABLES = [
    ("tropics_sitables", []),
    ("si_print", []),
    ("si_tables_two_part", []),
    ("table_s5_quantiles", []),
    ("supplementary_data", []),
]

VERIFY = [
    ("tropics_verify", []),
    ("temporal_verify", []),
]


def call(mod, spec):
    env = dict(os.environ, PYTHONPATH=PYTHONPATH, MPLBACKEND="Agg")
    if isinstance(spec, str):
        prog = f"import {mod} as m\n{spec}\n"
    else:
        prog = (f"import runpy, sys\nsys.argv = [{mod!r}] + {list(spec)!r}\n"
                f"runpy.run_module({mod!r}, run_name='__main__')\n")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", prog], cwd=REPO, env=env)
    print(f"{mod:<28} {'ok' if r.returncode == 0 else 'failed'}  {time.time() - t0:,.0f} s", flush=True)
    return r.returncode == 0


def figures():
    ok = True
    for s in sorted((REPO / "figures").glob("*/make_*.py")):
        t0 = time.time()
        r = subprocess.run([sys.executable, str(s), "--pipeline"], cwd=REPO, env=dict(os.environ, MPLBACKEND="Agg"))
        print(f"{s.parent.name:<28} {'ok' if r.returncode == 0 else 'failed'}  {time.time() - t0:,.0f} s", flush=True)
        ok = ok and r.returncode == 0
    return ok


def main(tiers):
    plan = {"measurement": MEASUREMENT, "temporal": TEMPORAL, "analysis": ANALYSIS, "tables": TABLES,
            "verify": VERIFY}
    failed = []
    for t in tiers:
        if t == "figures":
            if not figures():
                failed.append("figures")
            continue
        for mod, spec in plan[t]:
            if not call(mod, spec):
                failed.append(mod)
                break
    print(f"failed: {', '.join(failed)}" if failed else "all stages ok")
    return 1 if failed else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "module":
        sys.exit(0 if call(a[1], a[2:]) else 1)
    tiers = a or ["tables", "figures"]
    if tiers == ["all"]:
        tiers = ["measurement", "temporal", "analysis", "tables", "figures", "verify"]
    sys.exit(main(tiers))
