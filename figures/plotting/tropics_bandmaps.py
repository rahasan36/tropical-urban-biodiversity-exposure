"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

import tropics_style as st
import tropics_frame as tf
import tropics_surfaces as sf
from tropics_config import TROPIC_LAT

EXTENT = (-180.0, 180.0, sf.LAT_BOT, sf.LAT_TOP)

RAMP_URBAN = ["#FBDDB8", "#F8C58C", "#F5A863", "#E87E3C", "#CF561F", "#A63603", "#6B2203"]
RAMP_RICH  = ["#D5E9E0", "#B3DCCC", "#93CEB8", "#5DB395", "#2E9375", "#127055", "#084735"]
RAMP_RARE  = ["#E1D8F0", "#CBBBE3", "#B8A1D6", "#9878C0", "#7A52A6", "#5C3385", "#3D1D5C"]


RAMP_AGE = ["#6E6A64", "#9C9892", "#C7C3BC", "#EFDCC6", "#E4A264", "#CE6A1E",
            "#8A3703"]


def _classify_raster(a, breaks, ramp, floor=None):
    v = np.where(np.isfinite(a), a, np.nan)
    idx = np.digitize(v, breaks, right=True)
    rgba = np.zeros(v.shape + (4,), np.uint8)
    cols = (np.array([matplotlib.colors.to_rgb(c) for c in ramp]) * 255).astype(np.uint8)
    lo = breaks[0] if floor is None else floor
    ok = np.isfinite(v) & (v >= lo)
    rgba[ok, :3] = cols[np.clip(idx[ok], 0, len(ramp) - 1)]
    rgba[ok, 3] = 255
    return rgba


def _draw(ax, rgba, extent=EXTENT, tropics_only=False):
    im = ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest",
                   zorder=3, aspect="auto")
    if tropics_only:
        from matplotlib.patches import Rectangle as _Rect
        clip = _Rect((tf.LON0, -TROPIC_LAT), tf.LON1 - tf.LON0, 2 * TROPIC_LAT,
                     transform=ax.transData)
        im.set_clip_path(clip)
    return im


def map_urban(name="fig_map_urban_v2", inset_cols=6, legend="indian",
              legend_scale=0.86):
    u = sf.load_urban()
    add, base = u["added"], u["base"]
    breaks = [0.05, 0.15, 0.4, 1.0, 2.5, 6.0]
    grey_breaks = [0.05, 0.4, 2.0]
    grey_ramp = ["#D8D6D2", "#BAB7B2", "#98948E", "#76726C"]
    fig, ax, insets = tf.band_figure(
        title="Tropical urban land added between 2000 and 2020",
        panel_letter=None, land="white", inset_cols=inset_cols,
        footer_in=1.20)
    grey = _classify_raster(base, grey_breaks, grey_ramp)
    rgba = _classify_raster(add, breaks, RAMP_URBAN)
    _draw(ax, grey, tropics_only=True); _draw(ax, rgba, tropics_only=True)
    ax._outline_on_top()
    for a in [i for i, _ in insets]:
        _draw(a, grey); _draw(a, rgba); a._outline_on_top()
    tf.footer_legend(fig, [
        ("Urban land added, 2000\u20132020 (km$^2$ per 3 arcmin cell)",
         breaks, RAMP_URBAN, None),
        ("Already impervious in 2000", grey_breaks, grey_ramp, None),
    ])
    return tf.save(fig, name)


def map_age(name="fig_map_age_v2", inset_cols=6):
    u = sf.load_urban()
    add, base = u["added"], u["base"]
    total = base + add
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, add / total, np.nan) * 100.0
    share = np.where(total >= 0.05, share, np.nan)
    breaks = [20, 35, 50, 65, 80, 90]
    fig, ax, insets = tf.band_figure(
        title="How much of the 2020 tropical urban footprint arrived after 2000",
        panel_letter=None, land="white", inset_cols=inset_cols, footer_in=0.80)
    rgba = _classify_raster(share, breaks, RAMP_AGE, floor=-1.0)
    _draw(ax, rgba, tropics_only=True); ax._outline_on_top()
    for a in [i for i, _ in insets]:
        _draw(a, rgba); a._outline_on_top()
    tf.footer_legend(fig, [("Share of the 2020 impervious area added after "
                            "2000 (%)", breaks, RAMP_AGE, 0)])
    return tf.save(fig, name)


def map_richness(which="threat", name="fig_map_richness_v2"):
    r, thr = sf.load_richness()
    a = r[which].astype(float)
    if which == "threat":
        breaks = [3, 6, 10, 15, 22, 32]
        ramp, ttl = RAMP_RICH, "Threatened species richness across the tropics"
        leg = "Threatened species per cell"
        left = None
    elif which == "narrow":
        breaks = [2, 4, 7, 12, 20]
        cls_breaks = [1.5, 3.5, 6.5, 11.5, 19.5]
        ramp, ttl = RAMP_RARE[1:], "Narrow-range species richness"
        leg = "Narrow-range species per cell"
        left = 1
    else:
        pos = a[a > 0]
        breaks = list(np.round(np.nanpercentile(pos, [50, 70, 85, 93, 97, 99.3]), 3))
        ramp, ttl = RAMP_RARE, "Range rarity across the tropics"
        leg = "Summed inverse range size"
        left = None
    fig, ax, insets = tf.band_figure(title=ttl, panel_letter="a",
                                     footer_in=0.78)
    rgba = _classify_raster(a, locals().get("cls_breaks") or breaks, ramp)
    _draw(ax, rgba, tropics_only=True); ax._outline_on_top()
    for x in [i for i, _ in insets]:
        _draw(x, rgba); x._outline_on_top()
    x0, y0, w, h = tf.legend_rect()
    tf.map_legend(ax, [(leg, breaks, ramp, left)],
                  rect=(x0, y0 + h * 0.42, w, h * 0.58))
    return tf.save(fig, name)


SURFACE_SPEC = {
    "threat": dict(ramp=RAMP_RICH, title="Threatened species richness",
                   legend="Threatened species per cell", left=None),
    "narrow": dict(ramp=RAMP_RARE[2:], title="Narrow-range species richness",
                   legend="Narrow-range species per cell", left=1, dilate=3),
    "rarity": dict(ramp=RAMP_RARE[1:], title="Range rarity",
                   legend="Summed inverse range size ($10^{-3}$ km$^{-2}$)",
                   left=None),
}


def _surface_breaks(which, a):
    if which == "threat":
        return [3, 6, 10, 15, 22, 32], None
    if which == "narrow":
        return [2, 4, 7, 12, 20], [1.5, 3.5, 6.5, 11.5, 19.5]
    pos = a[a > 0]
    b = [float(f"{v:.1f}") for v in
         np.nanpercentile(pos, [55, 75, 90, 96, 99.3]) * 1e3]
    return b, [v / 1e3 for v in b]


def map_surfaces(name="figS_surfaces_v1",
                 which=("threat", "narrow", "rarity")):
    st.apply_rc()
    r, thr = sf.load_richness()
    left, right = 0.030, 0.012
    W = tf.LIVE_W
    map_w = W * (1 - left - right)
    map_h = map_w * (tf.LAT1 - tf.LAT0) / (tf.LON1 - tf.LON0)
    head, gap, foot = 0.30, 0.30, 0.22
    fig_h = head + len(which) * map_h + (len(which) - 1) * gap + foot
    fig = plt.figure(figsize=(W, fig_h))
    for i, w in enumerate(which):
        spec = SURFACE_SPEC[w]
        a = r[w].astype(float)
        breaks, cls = _surface_breaks(w, a)
        k = spec.get("dilate")
        if k:
            from scipy.ndimage import maximum_filter
            a = maximum_filter(np.nan_to_num(a, nan=0.0), size=k)
            a = np.where(np.isfinite(r[w].astype(float)) | (a > 0), a, np.nan)
        y = fig_h - head - (i + 1) * map_h - i * gap
        ax = fig.add_axes([left, y / fig_h, map_w / W, map_h / fig_h])
        ax.set_xlim(tf.LON0, tf.LON1); ax.set_ylim(tf.LAT0, tf.LAT1)
        ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#B9B6B1"); sp.set_linewidth(0.5)
        tf._graticule(ax); tf._draw_land(ax, context=True)
        _draw(ax, _classify_raster(a, cls or breaks, spec["ramp"]),
              tropics_only=True)
        tf._outline(ax, context=True)
        x0, y0, ww, hh = tf.legend_rect()
        tf.map_legend(ax, [(spec["legend"], breaks, spec["ramp"], spec["left"])],
                      rect=(x0, y0 + hh * 0.42, ww, hh * 0.58))
        fig.text(left, (y + map_h + 0.055) / fig_h, "abc"[i], fontsize=9.5,
                 fontweight="bold", va="bottom", ha="left", color=st.INK)
    return tf.save(fig, name)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "urban"):
        print(map_urban().name)
        print(map_urban("fig_map_urban_onerow_v2", inset_cols=6,
                        legend="atlantic", legend_scale=1.0).name)
        print(map_age().name)
        print(map_age("fig_map_age_onerow_v2", inset_cols=6).name)
    if which in ("all", "rich"):
        print(map_richness("threat").name)
        print(map_richness("narrow", name="fig_map_narrow_v2").name)
        print(map_richness("rarity", name="fig_map_rarity_v2").name)
