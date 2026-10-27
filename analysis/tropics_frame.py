"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
import matplotlib.patheffects as pe

HALO = [pe.withStroke(linewidth=1.9, foreground="white")]
HALO_T = [pe.withStroke(linewidth=1.6, foreground="white")]

import tropics_style as st
import tropics_basemap as bm
from tropics_config import TROPIC_LAT

PAGE_W_IN, PAGE_H_IN = 8.5, 11.0
MARGIN_IN = 0.75
LIVE_W = PAGE_W_IN - 2 * MARGIN_IN
LIVE_H = PAGE_H_IN - 2 * MARGIN_IN

LON0, LON1 = -125.0, 180.0

LAT0, LAT1 = -60.0, 60.0
TROP0, TROP1 = -25.5, 25.5

LAND = "#FFFFFF"
LAND_EDGE = "#C9C7C3"
CTX_LAND = "#EAE8E4"
CTX_EDGE = "#D6D4D0"
GRAT = "#DEDCD8"
NODATA = "#DCDAD6"

INSET_KM = 1800.0
KM_PER_DEG_LAT = 110.57

FRAME_COLORS = ["#2B62C4", "#C05A0F", "#0F8A6E", "#B0184E", "#7A3EAF", "#6E7B12"]

REGION_CENTERS = [
    ("Mesoamerica",     -93.5,  15.0, FRAME_COLORS[0]),
    ("Atlantic Forest", -44.0, -17.0, FRAME_COLORS[1]),
    ("Gulf of Guinea",    3.0,   8.5, FRAME_COLORS[2]),
    ("East Africa",      35.0,   2.5, FRAME_COLORS[3]),
    ("Western Ghats",    77.0,  13.5, FRAME_COLORS[4]),
    ("Sunda",           107.0,  -1.0, FRAME_COLORS[5]),
]


def region_bounds(lon_c, lat_c, km=INSET_KM):
    dlat = km / KM_PER_DEG_LAT
    dlon = km / (111.32 * np.cos(np.radians(lat_c)))
    return (lon_c - dlon / 2, lon_c + dlon / 2, lat_c - dlat / 2, lat_c + dlat / 2)


REGIONS = [(n,) + region_bounds(x, y) + (c,) for n, x, y, c in REGION_CENTERS]


def classed(values, breaks, colors, nodata=NODATA):
    v = np.asarray(values, float)
    idx = np.digitize(v, breaks, right=True)
    out = np.array([to_rgb(c) for c in colors])[np.clip(idx, 0, len(colors) - 1)]
    bad = ~np.isfinite(v)
    if bad.any():
        out[bad] = to_rgb(nodata)
    return out


def class_legend(ax, breaks, colors, labels=None, title="", loc=(0.012, 0.94),
                 sw=0.0135, sh=0.052, fs=6.6):
    x0, y0 = loc
    ax.text(x0, y0 + 0.035, title, transform=ax.transAxes, fontsize=fs + 0.9,
            fontweight="bold", va="bottom", ha="left", color=st.INK, zorder=9)
    if labels is None:
        labels = ([f"< {breaks[0]:g}"] +
                  [f"{breaks[i]:g}–{breaks[i+1]:g}" for i in range(len(breaks) - 1)] +
                  [f"≥ {breaks[-1]:g}"])
    for i, (c, lab) in enumerate(zip(colors, labels)):
        y = y0 - i * sh
        ax.add_patch(Rectangle((x0, y - sh * 0.72), sw, sh * 0.66,
                               transform=ax.transAxes, facecolor=c,
                               edgecolor="white", lw=0.4, zorder=9, clip_on=False))
        ax.text(x0 + sw * 1.5, y - sh * 0.40, lab, transform=ax.transAxes,
                fontsize=fs, va="center", ha="left", color=st.INK, zorder=9)


def class_legend_h(fig, breaks, colors, labels=None, title="", y_in=None,
                   x_in=None, sw_in=0.20, sh_in=0.085, fs=6.3, left_label=None):
    g = fig._frame
    if labels is None:
        labels = ([f"< {breaks[0]:g}"] +
                  [f"{breaks[i]:g}\u2013{breaks[i+1]:g}" for i in range(len(breaks) - 1)] +
                  [f"\u2265 {breaks[-1]:g}"])
    x0 = g["left"] * g["w"] if x_in is None else x_in
    y = (g["h"] - g["head"] - g["map_h"] - 0.32) if y_in is None else y_in
    fig.text(x0 / g["w"], (y + sh_in * 1.7) / g["h"], title, fontsize=fs + 0.9,
             fontweight="bold", ha="left", va="bottom", color=st.INK)
    x = x0
    for c in colors:
        fig.patches.append(Rectangle((x / g["w"], y / g["h"]),
                                     sw_in / g["w"], sh_in / g["h"],
                                     transform=fig.transFigure, facecolor=c,
                                     edgecolor="white", lw=0.5, zorder=9))
        x += sw_in
    if left_label is not None:
        fig.text(x0 / g["w"], (y - 0.03) / g["h"], str(left_label),
                 fontsize=fs, ha="center", va="top", color=st.INK)
    for i, b in enumerate(breaks):
        fig.text((x0 + (i + 1) * sw_in) / g["w"], (y - 0.03) / g["h"], f"{b:g}",
                 fontsize=fs, ha="center", va="top", color=st.INK)
    return x


LEGEND_LON = (52.0, 97.0)
LEGEND_LAT = (-47.5, -26.0)

LEGEND_LON_ATL = (-33.0, 8.0)
LEGEND_LAT_ATL = (-45.0, -26.0)

LEGEND_BOXES = {"indian": (LEGEND_LON, LEGEND_LAT),
                "atlantic": (LEGEND_LON_ATL, LEGEND_LAT_ATL)}


def legend_rect(where="indian", scale=1.0, lon0=None, lon1=None, lat0=None,
                lat1=None):
    (a0, a1), (b0, b1) = LEGEND_BOXES[where]
    lon0 = a0 if lon0 is None else lon0
    lon1 = a1 if lon1 is None else lon1
    lat0 = b0 if lat0 is None else lat0
    lat1 = b1 if lat1 is None else lat1
    if scale != 1.0:
        cx, cy = (lon0 + lon1) / 2, (lat0 + lat1) / 2
        lon0, lon1 = cx + (lon0 - cx) * scale, cx + (lon1 - cx) * scale
        lat0, lat1 = cy + (lat0 - cy) * scale, cy + (lat1 - cy) * scale
    x0 = (lon0 - LON0) / (LON1 - LON0)
    x1 = (lon1 - LON0) / (LON1 - LON0)
    y0 = (lat0 - LAT0) / (LAT1 - LAT0)
    y1 = (lat1 - LAT0) / (LAT1 - LAT0)
    return x0, y0, x1 - x0, y1 - y0


def map_legend(ax, blocks, rect=None, fs=None, pad_in=0.030, title_fs=None,
               arrow_end=True):
    fig = ax.figure
    fs = fs if fs is not None else st.MIN_PT
    title_fs = title_fs if title_fs is not None else fs
    x0, y0, w, h = rect or legend_rect()
    ex = ax.get_window_extent()
    px, py = pad_in / (ex.width / fig.dpi), pad_in / (ex.height / fig.dpi)
    inner = w - 2 * px
    avail_px = inner * ex.width
    top_inside = (y0 + h) <= 1.0 and y0 >= 0.0

    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def width_px(txt, size, bold=False):
        t = ax.text(0, 0, txt, transform=ax.transAxes, fontsize=size,
                    fontweight="bold" if bold else "normal")
        wpx = t.get_window_extent(r).width
        t.remove()
        return wpx

    def wrap(title, size):
        words, lines, cur = title.split(" "), [], ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if cur and width_px(trial, size, bold=True) > avail_px:
                lines.append(cur)
                cur = wd
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    line_h = 1.18 * (title_fs / 72.0) / (ex.height / fig.dpi)
    lab_h = 1.14 * (fs / 72.0) / (ex.height / fig.dpi)
    sh = 0.62 * (fs / 72.0) / (ex.height / fig.dpi) * 1.15
    plan = []
    total = 0.0
    for title, breaks, colors, left in blocks:
        lines = wrap(title, title_fs)
        n_sq = len(colors) - 1 if arrow_end else len(colors)
        sw = inner / (n_sq + (0.72 if arrow_end else 0))
        labels = [str(left)] if left is not None else []
        labels += [f"{b:g}" for b in breaks]
        widest = max((width_px(t, fs) for t in labels), default=0)
        stagger = widest > sw * ex.width * 0.88
        rows = 2 if stagger else 1
        bh = len(lines) * line_h + 0.12 * line_h + sh + rows * lab_h + 0.22 * lab_h
        plan.append((lines, breaks, colors, left, n_sq, sw, stagger, bh))
        total += bh

    h_fit = total + 2 * py
    if h_fit > h:
        y0 = y0 + h - h_fit
        if y0 >= 0.0 or top_inside:
            y0 = max(y0, 0.012)
        h = h_fit
    ax.add_patch(Rectangle((x0, y0), w, h, transform=ax.transAxes,
                           facecolor="white", edgecolor="#D0CDC8", lw=0.5,
                           zorder=10, clip_on=False))

    top = y0 + h - py
    for lines, breaks, colors, left, n_sq, sw, stagger, bh in plan:
        for ln in lines:
            ax.text(x0 + px, top, ln, transform=ax.transAxes, fontsize=title_fs,
                    fontweight="bold", ha="left", va="top", color=st.INK,
                    zorder=11)
            top -= line_h
        top -= 0.12 * line_h
        ys = top - sh
        for i, c in enumerate(colors[:n_sq]):
            ax.add_patch(Rectangle((x0 + px + i * sw, ys), sw, sh,
                                   transform=ax.transAxes, facecolor=c,
                                   edgecolor="white", lw=0.4, zorder=11,
                                   clip_on=False))
        if arrow_end:
            xa = x0 + px + n_sq * sw
            ax.add_patch(MplPolygon([[xa, ys], [xa, ys + sh],
                                     [xa + sw * 0.72, ys + sh / 2]],
                                    closed=True, transform=ax.transAxes,
                                    facecolor=colors[-1], edgecolor="white",
                                    lw=0.4, zorder=11, clip_on=False))
        marks = []
        if left is not None:
            marks.append((x0 + px, str(left), 0))
        for i, b in enumerate(breaks):
            marks.append((x0 + px + (i + 1) * sw, f"{b:g}", (i + 1) % 2 if stagger else 0))
        for xpos, txt, row in marks:
            ax.text(xpos, ys - 0.10 * lab_h - row * lab_h, txt,
                    transform=ax.transAxes, fontsize=fs, ha="center", va="top",
                    color=st.INK, zorder=11)
        top = ys - 0.10 * lab_h - (2 if stagger else 1) * lab_h - 0.22 * lab_h
    return ax


def _draw_land(ax, lw=0.25, color=None, context=False):
    if context:
        wp = []
        for g in bm._load_world():
            for q in bm._polys(g):
                wp.append(MplPolygon(np.asarray(q.exterior.coords), closed=True))
        ax.add_collection(PatchCollection(wp, facecolors=CTX_LAND,
                                          edgecolors=CTX_EDGE, linewidths=lw,
                                          zorder=0.5))
    geoms = bm._load_tropic() if context else bm._load_countries()[0]
    patches = []
    for g in geoms:
        for p in bm._polys(g):
            patches.append(MplPolygon(np.asarray(p.exterior.coords), closed=True))
    ax.add_collection(PatchCollection(patches, facecolors=color or LAND,
                                      edgecolors=LAND_EDGE, linewidths=lw, zorder=1))


def _outline(ax, lw=0.22, color="#9C9995", context=False):
    geoms = bm._load_tropic() if context else bm._load_countries()[0]
    patches = []
    for g in geoms:
        for p in bm._polys(g):
            patches.append(MplPolygon(np.asarray(p.exterior.coords), closed=True))
    ax.add_collection(PatchCollection(patches, facecolors="none", edgecolors=color,
                                      linewidths=lw, zorder=5))


def _graticule(ax, lon_step=30, lat_step=20):
    for x in range(int(LON0 // lon_step) * lon_step, int(LON1) + 1, lon_step):
        ax.axvline(x, color=GRAT, lw=0.4, zorder=0)
    y0 = int(np.ceil(LAT0 / lat_step) * lat_step)
    for y in range(y0, int(LAT1) + 1, lat_step):
        ax.axhline(y, color=GRAT, lw=0.4, zorder=0)
    for y, lab in ((TROPIC_LAT, "23.4°N"), (-TROPIC_LAT, "23.4°S")):
        ax.axhline(y, color="#8E8B86", lw=0.5, ls=(0, (5, 4)), zorder=6)
        ax.text(LON1 - 1.2, y + 0.6, lab, fontsize=5.6, ha="right", va="bottom",
                color="#8E8B86", zorder=7)


def scale_bar(ax, lon0, lon1, lat0, lat1, km=None, frac=0.30, x_frac=0.055):
    latm = 0.5 * (lat0 + lat1)
    km_per_deg = 111.32 * np.cos(np.radians(latm))
    span_km = (lon1 - lon0) * km_per_deg
    if km is None:
        target = span_km * frac
        km = min([25, 50, 100, 200, 250, 500, 1000, 2000],
                 key=lambda v: abs(v - target))
    dx = km / km_per_deg
    x = lon0 + (lon1 - lon0) * x_frac
    y = lat0 + (lat1 - lat0) * 0.075
    ax.plot([x, x + dx], [y, y], color=st.INK, lw=1.5, solid_capstyle="butt",
            zorder=9, path_effects=HALO)
    ax.text(x + dx / 2, y + (lat1 - lat0) * 0.030, f"{km:,g} km", fontsize=5.6,
            ha="center", va="bottom", color=st.INK, zorder=9, path_effects=HALO_T)


def north_arrow(ax, lon0, lon1, lat0, lat1, x_frac=0.935, y_frac=0.075):
    x = lon0 + (lon1 - lon0) * x_frac
    y0 = lat0 + (lat1 - lat0) * y_frac
    dy = (lat1 - lat0) * 0.13
    ax.annotate("", xy=(x, y0 + dy), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=st.INK, lw=0.9,
                                mutation_scale=6,
                                path_effects=HALO), zorder=9)
    ax.text(x, y0 + dy * 1.12, "N", fontsize=5.8, ha="center", va="bottom",
            color=st.INK, zorder=9, path_effects=HALO_T)


def shared_scale_bar(fig, x_in, y_in, km, in_per_km, label=None, fs=6.2):
    g = fig._frame
    dx = km * in_per_km
    fig.add_artist(Line2D([x_in / g["w"], (x_in + dx) / g["w"]],
                          [y_in / g["h"], y_in / g["h"]], color=st.INK, lw=1.4,
                          solid_capstyle="butt", transform=fig.transFigure))
    fig.text((x_in + dx / 2) / g["w"], (y_in + 0.035) / g["h"],
             label or f"{km:,g} km", fontsize=fs, ha="center", va="bottom",
             color=st.INK)


LEGEND_IN_MAP = (0.573, 0.035)


def band_figure(width_in=LIVE_W, title=None, panel_letter=None, n_inset=6,
                inset_cols=6, footer_in=0.80, subtitle=None, scale_y_in=None,
                panel_scales=True, legend_in_map=False, land=None,
                inset_in=1.15, regions=None):
    st.apply_rc()
    left, right = (0.014, 0.012) if legend_in_map else (0.030, 0.012)
    if legend_in_map and footer_in > 0.30:
        footer_in = 0.14
    map_w = width_in * (1 - left - right)
    map_h = map_w * (LAT1 - LAT0) / (LON1 - LON0)
    head = 0.16 if (title or panel_letter) else 0.10
    gap = 0.34
    col_gap, row_gap, label_h = 0.085, 0.20, 0.150

    rows = int(np.ceil(n_inset / inset_cols)) if n_inset else 0
    pw = min((map_w - (inset_cols - 1) * col_gap) / inset_cols, inset_in) if n_inset else 0
    ph = pw
    grid_w = inset_cols * pw + (inset_cols - 1) * col_gap if n_inset else 0
    x_grid = left * width_in + (map_w - grid_w) / 2
    grid_h = (rows * (ph + label_h) + (rows - 1) * row_gap) if rows else 0
    fig_h = head + map_h + (gap + grid_h if rows else 0) + footer_in
    fig = plt.figure(figsize=(width_in, fig_h))

    ax = fig.add_axes([left, (fig_h - head - map_h) / fig_h,
                       map_w / width_in, map_h / fig_h])
    ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
    ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#B9B6B1"); sp.set_linewidth(0.5)
    _graticule(ax); _draw_land(ax, color=land, context=True)
    ax._outline_on_top = lambda: _outline(ax, context=True)

    if panel_letter:
        fig.text(left, 1 - 0.10 / fig_h, panel_letter, fontsize=9.5,
                 fontweight="bold", va="top", ha="left", color=st.INK)

    insets = []
    if rows:
        y_top = fig_h - head - map_h - gap
        use = REGIONS if regions is None else [REGIONS[i] for i in regions]
        for i, (name, lo, hi, la, lb, col) in enumerate(use[:n_inset]):
            r, c = divmod(i, inset_cols)
            x = x_grid + c * (pw + col_gap)
            y = y_top - (r + 1) * (ph + label_h) - r * row_gap
            iax = fig.add_axes([x / width_in, y / fig_h, pw / width_in,
                                ph / fig_h])
            iax.set_xlim(lo, hi); iax.set_ylim(la, lb)
            iax.set_xticks([]); iax.set_yticks([]); iax.set_facecolor("white")
            for sp in iax.spines.values():
                sp.set_color(col); sp.set_linewidth(1.1)
            _draw_land(iax, lw=0.3, color=land)
            iax._outline_on_top = (lambda a=iax: _outline(a, lw=0.3))
            north_arrow(iax, lo, hi, la, lb)
            if panel_scales:
                scale_bar(iax, lo, hi, la, lb, km=500)
            iax.text(0.0, 1.030, name, transform=iax.transAxes, fontsize=7.0,
                     fontweight="bold", color=col, ha="left", va="bottom")
            ax.add_patch(Rectangle((lo, la), hi - lo, lb - la, fill=False,
                                   edgecolor=col, lw=0.9, zorder=8))
            bx, by = lo + (hi - lo) * 0.5, lb + (lb - la) * 0.10
            ax.plot([bx], [by], marker="o", ms=7.4, mfc="white", mec=col,
                    mew=0.9, ls="none", zorder=9, clip_on=False)
            ax.text(bx, by, str(i + 1), fontsize=6.0, fontweight="bold",
                    color=col, ha="center", va="center", zorder=10,
                    clip_on=False)
            iax.text(0.975, 0.965, str(i + 1), transform=iax.transAxes,
                     fontsize=7.0, fontweight="bold", color=col, ha="right",
                     va="top", zorder=9,
                     bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
            insets.append((iax, (lo, hi, la, lb)))
        fig._frame = dict(w=width_in, h=fig_h, left=left, map_h=map_h, head=head)
        if not panel_scales:
            shared_scale_bar(fig, (left + 0.74) * width_in,
                             footer_in - 0.17 if scale_y_in is None else scale_y_in,
                             500.0, pw / INSET_KM,
                             label=f"500 km, every inset panel "
                                   f"{INSET_KM:,.0f} \u00d7 {INSET_KM:,.0f} km")
    fig._frame = dict(w=width_in, h=fig_h, left=left, map_h=map_h, head=head)
    return fig, ax, insets


def footer_legend(fig, blocks, x_in=None, y_in=0.30, sw_in=0.285, sh_in=0.082,
                  fs=6.1, row_in=0.44):
    g = fig._frame
    x0 = (g["left"] * g["w"]) if x_in is None else x_in
    y = y_in + row_in * (len(blocks) - 1)
    for title, breaks, colors, left in blocks:
        x = x0
        fig.text(x / g["w"], (y + sh_in * 1.85) / g["h"], title, fontsize=fs + 0.5,
                 fontweight="bold", ha="left", va="bottom", color=st.INK)
        for i, c in enumerate(colors):
            fig.patches.append(Rectangle(((x + i * sw_in) / g["w"], y / g["h"]),
                                         sw_in / g["w"], sh_in / g["h"],
                                         transform=fig.transFigure, facecolor=c,
                                         edgecolor="white", lw=0.5, zorder=9))
        if left is not None:
            fig.text(x / g["w"], (y - 0.028) / g["h"], str(left), fontsize=fs,
                     ha="center", va="top", color=st.INK)
        for i, b in enumerate(breaks):
            fig.text((x + (i + 1) * sw_in) / g["w"], (y - 0.028) / g["h"],
                     f"{b:g}", fontsize=fs, ha="center", va="top", color=st.INK)
        y -= row_in
    return x0


def save(fig, name, fig_dir=None):
    from tropics_config import FIG_DIR
    d = fig_dir or FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    return d / f"{name}.png"
