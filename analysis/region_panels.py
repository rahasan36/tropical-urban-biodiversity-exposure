"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

import tropics_style as st
import tropics_surfaces as sf
import tropics_frame as tf
import tropics_basemap as bm

RICH_RAMP = ["#EDF1EC", "#CFE0D2", "#A8CCB5", "#7DB296", "#52947C", "#2E6F5B"]
URBAN = "#C24A10"
URBAN_GATE = 1.0
COAST = "#A8A49F"

RULE_REGIONS = [
    ("Northern South America", -78.0, -60.0, 0.0, 12.0),
    ("Mesoamerica", -106.0, -86.0, 12.5, 23.0),
    ("Mainland Southeast Asia", 92.0, 108.0, 5.0, 22.0),
]

NIGERIA = ("Nigeria and West Africa", -1.0, 15.0, 3.0, 13.0)
RULE_REGIONS_4 = RULE_REGIONS + [NIGERIA]

EASTERN_ARC = ("Eastern Arc and coastal Tanzania", 34.0, 41.5, -11.0, -2.5, 250)
RULE_REGIONS_4E = RULE_REGIONS + [EASTERN_ARC]
SUMATRA = ("Sumatra and the Malay Peninsula", 95.0, 106.0, -6.0, 6.0)
RULE_REGIONS_4S = RULE_REGIONS + [SUMATRA]

NAMED_REGIONS = [
    ("Amazonian Brazil", -70.0, -52.0, -10.0, 2.0, 500),
    ("Java", 105.0, 116.0, -9.0, -5.5, 300),
    ("Gulf of Guinea", -8.0, 14.0, 3.0, 13.0, 200),
]


def _fit_box(box, aspect):
    name, lon0, lon1, lat0, lat1, *rest = box
    dx, dy = lon1 - lon0, lat1 - lat0
    if dy / dx < aspect:
        dy2 = dx * aspect
        cy = 0.5 * (lat0 + lat1)
        lat0, lat1 = cy - dy2 / 2, cy + dy2 / 2
    else:
        dx2 = dy / aspect
        cx = 0.5 * (lon0 + lon1)
        lon0, lon1 = cx - dx2 / 2, cx + dx2 / 2
    return (name, lon0, lon1, lat0, lat1, *rest)


def _wrap_name(name, limit):
    if len(name) <= limit:
        return name
    words = name.split()
    best, cut = None, None
    for i in range(1, len(words)):
        a = " ".join(words[:i])
        d = abs(len(a) - len(name) / 2)
        if best is None or d < best:
            best, cut = d, i
    return " ".join(words[:cut]) + "\n" + " ".join(words[cut:])


def _rich_breaks(a):
    v = a[np.isfinite(a) & (a > 0)]
    if v.size == 0:
        return [1, 2, 3, 4, 5]
    q = np.percentile(v, [50, 70, 85, 94, 98])
    out = []
    for x in q:
        x = float(np.round(x))
        if not out or x > out[-1]:
            out.append(x)
    while len(out) < 5:
        out.append(out[-1] + 1)
    return out[:5]


RICH_VMIN, RICH_VMAX = 1.0, 20.0
RICH_TICKS = [1, 2, 5, 10, 20]


def _rich_cmap_norm():
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    return (LinearSegmentedColormap.from_list("rich", RICH_RAMP[1:]),
            LogNorm(RICH_VMIN, RICH_VMAX))


def _shade(a):
    cmap, norm = _rich_cmap_norm()
    v = np.where(np.isfinite(a), a, 0.0)
    rgba = cmap(norm(np.clip(v, RICH_VMIN, RICH_VMAX)))
    rgba[..., 3] = np.where(v > 0, 1.0, 0.0)
    return rgba


def _classify(a, breaks, ramp):
    idx = np.digitize(a, breaks, right=False)
    rgba = np.zeros(a.shape + (4,), dtype=float)
    cols = np.array([to_rgb(c) for c in ramp])
    rgba[..., :3] = cols[np.clip(idx, 0, len(ramp) - 1)]
    rgba[..., 3] = np.where(np.isfinite(a) & (a > 0), 1.0, 0.0)
    return rgba


def _window(arr, deg, lon0, lon1, lat0, lat1):
    r0 = int(round((sf.LAT_TOP - lat1) / deg))
    r1 = int(round((sf.LAT_TOP - lat0) / deg))
    c0 = int(round((lon0 + 180.0) / deg))
    c1 = int(round((lon1 + 180.0) / deg))
    r0, r1 = max(r0, 0), min(r1, arr.shape[0])
    c0, c1 = max(c0, 0), min(c1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def region_panel(ax, box, rich, added, breaks, show_scale=True):
    name, lon0, lon1, lat0, lat1, *rest = box
    km = rest[0] if rest else 500
    ext = (lon0, lon1, lat0, lat1)
    rw = _window(rich, sf.RICH_DEG, *ext)
    aw = _window(added, sf.DISP_DEG, *ext)

    ax.imshow(_shade(rw), extent=(lon0, lon1, lat0, lat1),
              origin="upper", interpolation="nearest", zorder=1, aspect="auto")
    m = np.isfinite(aw) & (aw >= URBAN_GATE)
    u = np.zeros(aw.shape + (4,), dtype=float)
    u[..., :3] = to_rgb(URBAN)
    u[..., 3] = np.where(m, 1.0, 0.0)
    ax.imshow(u, extent=(lon0, lon1, lat0, lat1), origin="upper",
              interpolation="nearest", zorder=3, aspect="auto")

    for g in bm._load_world():
        for p in bm._polys(g):
            x, y = p.exterior.xy
            ax.plot(x, y, color=COAST, lw=0.3, zorder=2)
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#B9B6B1"); sp.set_linewidth(0.5)
    if show_scale:
        tf.scale_bar(ax, lon0, lon1, lat0, lat1, km=km, x_frac=0.56)
        tf.north_arrow(ax, lon0, lon1, lat0, lat1, x_frac=0.94, y_frac=0.075)
    return ax


def load_layers():
    r, _ = sf.load_richness()
    u = sf.load_urban()
    return r["narrow"].astype(float), u["added"]


def _conc_axes(ax, c):
    import pandas as pd
    ax.scatter(c.range_km2, c.top1_share * 100, s=2.0, c=st.FAINT, lw=0,
               rasterized=True)
    q = pd.qcut(np.log10(c.range_km2), 20, labels=False)
    m = c.groupby(q).agg(r=("range_km2", "median"), t=("top1_share", "median"))
    ax.plot(m.r, m.t * 100, color=st.CRIMSON, lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Global range of the species (km$^2$)", fontsize=st.MIN_PT)
    ax.set_ylabel("Converted range in the\nlargest agglomeration (%)",
                  fontsize=st.MIN_PT)
    st.despine(ax)
    return ax


def _key(fig, rect, breaks, title):
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.ticker import NullLocator
    F = fig.get_size_inches()[1]
    Wd = fig.get_size_inches()[0]
    x0, y0, w, h = rect
    thick = 0.085
    by = y0 + 0.13 / F
    cax = fig.add_axes([x0, by, w * 0.62, thick / F])
    cmap, norm = _rich_cmap_norm()
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal",
                      ticks=RICH_TICKS, extend="neither")
    cax.xaxis.set_minor_locator(NullLocator())
    cb.set_ticklabels([f"{t:g}" for t in RICH_TICKS])
    cax.tick_params(labelsize=st.MIN_PT, length=2, width=0.5, colors=st.INK,
                    pad=1.5)
    cb.outline.set_linewidth(0.4)
    cax.annotate(title, xy=(0.0, 1.0), xycoords="axes fraction",
                 xytext=(0, 2.0), textcoords="offset points", ha="left",
                 va="bottom", fontsize=st.MIN_PT, fontweight="normal",
                 color=st.INK)
    sx = x0 + w * 0.62 + 0.40 / Wd
    sw = fig.add_axes([sx, by, 0.24 / Wd, thick / F])
    sw.add_patch(Rectangle((0, 0), 1, 1, facecolor=URBAN, edgecolor="none"))
    sw.set_xlim(0, 1); sw.set_ylim(0, 1); sw.axis("off")
    fig.text(sx + 0.30 / Wd, by + 0.5 * thick / F, "Urban land added",
             ha="left", va="center", fontsize=st.MIN_PT, color=st.INK)
    return cax


def _band_panel(ax, rich, added, breaks):
    import tropics_main_figures as f4
    import tropics_bandmaps as m2
    f4._draw_raster(ax, _shade(rich))
    u = np.zeros(added.shape + (4,), dtype=float)
    u[..., :3] = to_rgb(URBAN)
    u[..., 3] = np.where(np.isfinite(added) & (added >= URBAN_GATE), 1.0, 0.0)
    f4._draw_raster(ax, u)
    if f4._is_robin():
        f4._tropic_lines(ax)
    else:
        tf._outline(ax, context=True)
    return ax


def narrow_range_figure(name="figure8_narrow_regions_v3", regions=None,
                        cols=None):
    import tropics_main_figures as f4
    import tropics_panels as tp
    st.apply_rc()
    regions = regions or RULE_REGIONS
    regions = sorted(regions, key=lambda b: 0.5 * (b[1] + b[2]))
    rich, added = load_layers()
    breaks = _rich_breaks(rich)

    W = tp.PAGE_W
    ML, MR = 0.08, 0.08
    gap = 0.14
    live = W - ML - MR
    ncol = cols or (2 if len(regions) == 4 else len(regions))
    nrow = int(np.ceil(len(regions) / ncol))
    if ncol >= 4:
        gap = 0.10
    pw = (live - (ncol - 1) * gap) / ncol
    ph = pw * (0.70 if ncol == 2 else 0.86)
    row_gap = 0.34
    title_lim = int(pw * 11)
    head_lines = 2 if any(len(b[0]) > title_lim for b in regions) else 1
    band_ML, band_MR = 0.08, 0.08
    band_w = W - band_ML - band_MR
    BAND_LAT = 33.0
    if f4._is_robin():
        import tropics_robinson as rb
        band_h = band_w * rb.band_aspect(f4.ROBIN_COVER)
    else:
        band_h = band_w * (2 * BAND_LAT) / (tf.LON1 - tf.LON0)

    mid0 = 0.40 if f4._is_robin() else 0.52
    head, mid, key_h, foot = 0.30, mid0 + 0.14 * (head_lines - 1), 0.56, 0.14
    row_gap = row_gap + 0.14 * (head_lines - 1)
    rows_h = nrow * ph + (nrow - 1) * row_gap
    fig_h = head + band_h + mid + rows_h + key_h + foot
    fig = plt.figure(figsize=(W, fig_h))

    band = f4._band_axes(fig, W, fig_h, fig_h - head, ML=band_ML, MR=band_MR)
    ax = band["ax"] if isinstance(band, dict) else band
    if not f4._is_robin():
        ax.set_ylim(-BAND_LAT, BAND_LAT)
        ax.set_position([band_ML / W, (fig_h - head - band_h) / fig_h,
                         band_w / W, band_h / fig_h])
    _band_panel(ax, rich, added, breaks)
    fig.text(band_ML / W, 1 - 0.055 / fig_h, "a", fontsize=st.MIN_PT + 1,
             fontweight="bold", va="top", ha="left", color=st.INK)

    top = key_h + foot + rows_h
    boxes = [_fit_box(b, ph / pw) for b in regions]
    for i, box in enumerate(boxes):
        r, c = divmod(i, ncol)
        x = ML + c * (pw + gap)
        y = top - (r + 1) * ph - r * row_gap
        a = fig.add_axes([x / W, y / fig_h, pw / W, ph / fig_h])
        region_panel(a, box, rich, added, breaks)
        lab = _wrap_name(box[0], title_lim)
        if head_lines == 2 and "\n" not in lab:
            lab = lab + "\n"
        a.text(0.0, 1.035, f"{'bcdefg'[i]}  {lab}",
               transform=a.transAxes, fontsize=st.MIN_PT, fontweight="bold",
               ha="left", va="bottom", color=st.INK, linespacing=1.25)
        f4._region_outline(ax, box, "bcdefg"[i])

    _key(fig, [ML / W, (foot + 0.06) / fig_h, (live * 0.62) / W,
               (key_h - 0.16) / fig_h], breaks,
         "Narrow-range species with a mapped range in the cell")
    return tp.save(fig, name)
