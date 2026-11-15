"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

import tropics_style as st
import tropics_frame as tf
from tropics_panels import cities, head, save as psave

CITY_RAMP = ["#FBDDB8", "#F8C58C", "#F09B57", "#E0702C", "#C24A10", "#8F3202",
             "#5C1D00"]


def _class_index(v, breaks):
    return np.clip(np.digitize(np.asarray(v, float), breaks, right=True), 0,
                   len(breaks))


def _size(area, lo=1.2, hi=14.0):
    a = np.sqrt(np.clip(area, 0, None))
    return lo + (hi - lo) * (a - a.min()) / (a.max() - a.min())


def _points(ax, d, col, breaks, ramp, size_by="urban_km2_final", zorder=6):
    idx = _class_index(d[col].values, breaks)
    colors = np.array([to_rgb(c) for c in ramp])[idx]
    import tropics_main_figures as _f4
    x, y = _f4._lonlat(d.lon.values, d.lat.values)
    ax.scatter(x, y, s=_size(d[size_by].values),
               c=colors, linewidths=0.0, zorder=zorder, rasterized=True)


def city_map(col, breaks, title, legend_title, name, ramp=CITY_RAMP,
             color_by=True, flat_color="#C24A10", sample=True, degrees=False):
    fig, ax, insets = tf.band_figure(title=title, panel_letter="a",
                                     legend_in_map=True, land="white",
                                     degrees=degrees,
                                     col_gap=0.17 if degrees else None)
    d = cities(sample=sample)
    for a in [ax] + [i for i, _ in insets]:
        a._outline_on_top()
        if color_by:
            _points(a, d, col, breaks, ramp)
        else:
            a.scatter(d.lon.values, d.lat.values,
                      s=_size(d.urban_km2_final.values), c=flat_color,
                      linewidths=0.0, zorder=6, rasterized=True)
    if color_by:
        x0, y0, w, h = tf.legend_rect()
        tf.map_legend(ax, [(legend_title, breaks, ramp, None)],
                      rect=(x0, y0 + h * 0.46, w, h * 0.54))
    from matplotlib.patches import Rectangle as _Rect
    lx, ly, lw_, lh_ = tf.legend_rect("atlantic", scale=0.92)
    ax.add_patch(_Rect((lx, ly), lw_, lh_, transform=ax.transAxes,
                       facecolor="white", edgecolor="#D0CDC8", lw=0.5,
                       zorder=10, clip_on=False))
    ax.text(lx + 0.006, ly + lh_ - 0.02, "City urban extent (km$^2$)",
            transform=ax.transAxes, fontsize=5.6, fontweight="normal", ha="left",
            va="top", color=st.INK, zorder=11)
    keys = [5.0, 100.0, 2000.0]
    lo, hi = d.urban_km2_final.min(), d.urban_km2_final.max()
    for i, a in enumerate(keys):
        r = float(_size(np.array([lo, a, hi]))[1])
        fx = lx + lw_ * (0.20 + 0.30 * i)
        ax.plot([fx], [ly + lh_ * 0.46], marker="o", ms=np.sqrt(r) * 1.15,
                mfc=flat_color if not color_by else "#8F3202", mec="none",
                ls="none", transform=ax.transAxes, zorder=11, clip_on=False)
        ax.text(fx, ly + lh_ * 0.20, f"{a:,.0f}", transform=ax.transAxes,
                fontsize=5.2, ha="center", va="center", color=st.INK, zorder=11)
    return tf.save(fig, name)


def map_inventory(name="fig_map_inventory_v2"):
    breaks = [10, 25, 50, 150, 500, 1500]
    return city_map("urban_km2_final", breaks,
                    "Spatial inventory of the 7,229 delineated agglomerations",
                    "City urban extent, 2020 (km$^2$)", name, color_by=False,
                    sample=False, degrees=True)


def map_cost(name="fig_map_city_cost_v2"):
    breaks = [8, 12, 17, 24, 35, 55]
    return city_map("cost_per_km2", breaks,
                    "Marginal biodiversity cost of each agglomeration",
                    "Converted range per km$^2$ added", name)


def map_rarity(name="fig_map_city_rarity_v2"):
    d = cities()
    v = d.rarity_per_km2.values
    breaks = [float(f"{x:.2g}") for x in
              np.percentile(v[v > 0], [40, 60, 75, 87, 95, 99])]
    return city_map("rarity_per_km2", breaks,
                    "Range-rarity-weighted marginal cost of each agglomeration",
                    "Inverse range size per km$^2$ added", name)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "inv"):
        print(map_inventory().name)
    if which in ("all", "cost"):
        print(map_cost().name)
    if which in ("all", "rar"):
        print(map_rarity().name)


def map_bivariate(name="fig_map_bivariate_v2"):
    d = cities()
    grid = st.bivariate_colors(3, c_x=st.ORANGE, c_y=st.TEAL)
    cx, cy = st.bivariate_class(d.urban_added_km2.values, d.cost_per_km2.values, 3)
    cols = np.array([grid[j, i] for i, j in zip(cx, cy)])
    fig, ax, insets = tf.band_figure(
        title="Urban land added against the cost of each unit added",
        panel_letter="a", footer_in=0.72)
    for a in [ax] + [i for i, _ in insets]:
        a._outline_on_top()
        a.scatter(d.lon.values, d.lat.values, s=_size(d.urban_km2_final.values),
                  c=cols, linewidths=0.0, zorder=6, rasterized=True)
    g = fig._frame
    ks = 0.44
    kax = fig.add_axes([(g["left"] * g["w"] + 0.10) / g["w"], 0.16 / g["h"],
                        ks / g["w"], ks / g["h"]])
    kax.imshow(grid, origin="lower", interpolation="nearest")
    kax.set_xticks([]); kax.set_yticks([])
    for s_ in kax.spines.values():
        s_.set_color("#B9B6B1"); s_.set_linewidth(0.5)
    kax.set_xlabel("urban land added →", fontsize=5.8, labelpad=1.4,
                   color=st.INK)
    kax.set_ylabel("exposure intensity →", fontsize=5.8, labelpad=1.4,
                   color=st.INK)
    nhh = int(((cx == 2) & (cy == 2)).sum())
    fig.text((g["left"] * g["w"] + 0.10 + ks + 0.16) / g["w"], 0.34 / g["h"],
             f"terciles of each quantity, one point per city\n"
             f"{nhh:,} cities fall in the top tercile of both",
             fontsize=6.4, ha="left", va="bottom", color=st.INK, linespacing=1.4)
    return tf.save(fig, name)
