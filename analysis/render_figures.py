"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import sys
import matplotlib; matplotlib.use("Agg")
import tropics_panels as tp, tropics_bandmaps as m2, tropics_citymaps as cm
import tropics_si_figures as si


def _spatial_cv(name):
    import si_spatialcv as scv
    return scv.figure(name)

def _conc_fig(name):
    import two_part_figs as tpf
    return tpf.si_figure(name)


JOBS = {
    "S01": (cm.map_inventory,        "figS01_inventory_v5",
            "Delineated agglomerations", "surfaces"),
    "S02": (lambda name: tp.fig_form_habitat(name, landcover=False), "figS02_form_v6",
            "Expansion form and land cover converted", "tables only"),
    "S03": (si.fig_quantile,         "figS06_quantile_v6",
            "The size exponent across the conditional distribution", "tables only"),
    "S04": (si.fig_pairs_selection,  "figS13_pair_selection_v1",
            "Where the drawn pairs sit among all admissible pairs", "tables only"),
    "S05": (_conc_fig,               "figS14_concentration_v3",
            "Species-level concentration of converted range", "tables only"),
}

MAIN = {
    "1": ("figure1_expansion_robinson_v9", "Tropical urban expansion", "source rasters"),
    "2": ("figure2_exposure_robinson_v10", "Exposure intensity and its spread", "surfaces"),
    "3": ("figure3_distributions_v7",  "Size, setting and geography", "tables only"),
    "4": ("figure4_concentration_v16",  "Reallocating the same urban land", "tables only"),
    "5": ("figure5_rarity_regions_robinson_v5", "Range rarity, the species and three windows", "surfaces"),
    "6": ("figure6_matched_cities_v11", "Matched agglomeration pairs", "source rasters"),
    "7": ("figure7_two_part_v4",        "Occurrence and intensity of narrow-range conversion",
          "tables only"),
    "8": ("figure8_narrow_regions_4row_robinson_sumatra_v1", "Range-restricted species and new urban land",
          "surfaces"),
}

def _robin(fn, **kw):
    def g(name):
        import tropics_main_figures as f4
        old = f4.PROJECTION
        f4.PROJECTION = "robinson"
        try:
            return fn(name, **kw)
        finally:
            f4.PROJECTION = old
    return g


def _main_fns():
    import tropics_main_figures as f4
    import two_part_figs as tpf
    import region_panels as rp
    return {"1": _robin(f4.figure1), "2": _robin(f4.figure2), "3": f4.figure3_shift,
            "4": f4.figure4, "5": _robin(f4.figure5, bottom="regions"),
            "6": f4.figure6, "7": tpf.main_figure,
            "8": _robin(rp.narrow_range_figure, regions=rp.RULE_REGIONS_4S, cols=4)}


def render(keys=None):
    mf = _main_fns()
    if not keys:
        keys = sorted(JOBS) + sorted(MAIN)
    out = []
    for k in keys:
        if k == "si":
            out += render(sorted(JOBS)); continue
        if k == "main":
            out += render(sorted(MAIN)); continue
        fn, name = (JOBS[k][0], JOBS[k][1]) if k in JOBS else (mf[k], MAIN[k][0])
        p = fn(name)
        print(k, p, flush=True)
        out.append(p)
    return out


if __name__ == "__main__":
    render(sys.argv[1:])
