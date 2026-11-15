"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tropics_style as st
import tropics_panels as tp
from tropics_config import TABLE_DIR, FIG_DIR


def _save(fig, name):
    if st.held(fig):
        return fig
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    return FIG_DIR / f"{name}.png"


def main_figure(name="figure7_two_part_v4", spec="B"):
    d = pd.read_csv(TABLE_DIR / f"narrow_two_part_{spec}.csv")
    con = json.load(open(TABLE_DIR / f"narrow_two_part_contrasts_{spec}.json"))
    st.apply_rc()
    W = tp.PAGE_W
    fig = plt.figure(figsize=(W, 5.00))
    x = d.urban_km2_base
    L, R, BOT, TOP = 0.108, 0.605, 0.575, 0.095
    wd, ht = 0.350, 0.345
    pan = [
        (L, BOT, d.p_occurrence, d.p_lo, d.p_hi, st.SET_COLOR["narrow"][0],
         "probability of any\nnarrow-range conversion", None, "a"),
        (R, BOT, d.intensity_pos, d.intensity_pos_lo, d.intensity_pos_hi, st.SET_COLOR["narrow"][0],
         "intensity given conversion\n(km$^2$ per km$^2$ added)", "log", "b"),
        (L, TOP, d.combined, d.combined_lo, d.combined_hi, st.SET_COLOR["narrow"][0],
         "combined expectation\n(km$^2$ per km$^2$ added)", None, "c"),
    ]
    for left, bot, y, lo, hi, c, ylab, ysc, letter in pan:
        ax = fig.add_axes([left, bot, wd, ht])
        ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(x, y, color=c, lw=1.6)
        ax.set_xscale("log")
        if ysc:
            ax.set_yscale(ysc)
        ax.set_xlabel("year 2000 impervious extent (km$^2$)")
        ax.set_ylabel(ylab)
        st.despine(ax)
        ax.text(-0.235, 1.07, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", color=st.INK, va="top")
        if letter == "c":
            ax.axhline(d.combined.iloc[0], color=st.MUTED, lw=0.8, ls=(0, (3, 2)))

    ax = fig.add_axes([R, TOP, wd, ht])
    quant = [("p", "P(any)"), ("intensity_pos", "intensity | any"),
             ("combined", "combined")]
    for j, (q, lab) in enumerate(quant):
        for k, (v, c, mk) in enumerate([("year 2000 size", st.BLUE, "o"),
                                        ("setting richness", st.CRIMSON, "s")]):
            yy = j + (0.17 if k == 0 else -0.17)
            r, lo, hi = con[v][q]
            ax.plot([lo, hi], [yy, yy], color=c, lw=1.4, solid_capstyle="butt")
            ax.plot([r], [yy], mk, color=c, ms=4.4, mec="white", mew=0.6)
    ax.axvline(1.0, color=st.MUTED, lw=0.9, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_yticks(range(len(quant)))
    ax.set_yticklabels([lab for _, lab in quant])
    ax.set_xlabel("fold change, 1st to 99th percentile")
    ax.set_ylim(-0.60, len(quant) - 0.25)
    st.despine(ax)
    ax.text(-0.235, 1.07, "d", transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=st.INK, va="top")
    ax.plot([], [], "o", color=st.BLUE, ms=4.4, label="size")
    ax.plot([], [], "s", color=st.CRIMSON, ms=4.4, label="setting")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.25)
    return _save(fig, name)


def si_figure(name="figS14_concentration_v3"):
    c = pd.read_csv(TABLE_DIR / "species_concentration.csv")
    s = json.load(open(TABLE_DIR / "species_concentration_strata.json"))
    st.apply_rc()
    W = tp.PAGE_W
    fig = plt.figure(figsize=(W, 2.28))

    ax = fig.add_axes([0.108, 0.258, 0.335, 0.682])
    ax.scatter(c.range_km2, c.top1_share * 100, s=2.2, c=st.FAINT, lw=0,
               rasterized=True)
    q = pd.qcut(np.log10(c.range_km2), 20, labels=False)
    m = c.groupby(q).agg(r=("range_km2", "median"), t=("top1_share", "median"))
    ax.plot(m.r, m.t * 100, color=st.CRIMSON, lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("global range of the species (km$^2$)")
    ax.set_ylabel("converted range in the\nlargest agglomeration (%)")
    st.despine(ax)
    ax.text(-0.245, 1.07, "a", transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=st.INK, va="top")

    ax = fig.add_axes([0.655, 0.258, 0.325, 0.682])
    rows = s["strata"]
    yy = np.arange(len(rows))
    ax.axvline(0, color=st.MUTED, lw=0.9, ls=(0, (3, 2)))
    ax.barh(yy, [r["rho"] for r in rows], height=0.62, color=st.BLUE, lw=0)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r['range_lo']:,.0f}" for r in rows])
    ax.set_xlabel("Spearman correlation ($\\rho$) within quintile")
    ax.set_ylabel("lower edge of the range-size\nquintile (km$^2$)")
    st.despine(ax)
    ax.text(-0.245, 1.07, "b", transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=st.INK, va="top")
    return _save(fig, name)


if __name__ == "__main__":
    print(main_figure())
    print(si_figure())
