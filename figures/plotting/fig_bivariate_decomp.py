"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import tropics_style as st
import tropics_panels as tp
import tropics_robinson as rb
import tropics_frame as tf
import fig_bivariate as fb
from tropics_config import MOSAIC_LAT_TOP, DELIN_DEG

BLUES = ["#E6F2FA", "#BFDDF3", "#8FC6EC", "#56B4E9", "#2B88C2"]
ORANGES = ["#F7E1EE", "#EDB3D3", "#DD72AE", "#C51B7D", "#8E0F5A"]
SECTORS = [("Americas", -180, -30), ("Africa and Arabia", -30, 60), ("South Asia", 60, 92),
           ("Southeast Asia and Oceania", 92, 180)]


def frame(ax, lat_view, xlim=None):
    (x0, x1), _ = rb.band_limits(0.797)
    if xlim is not None:
        x0, x1 = xlim
    yv = float(rb.lat_to_y(lat_view))
    ax.set_xlim(x0, x1); ax.set_ylim(-yv, yv); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#B9B6B1"); s.set_linewidth(0.6)
    for la in (rb.TROPIC, -rb.TROPIC):
        ll = np.linspace(rb.LON_LO + 1e-6, rb.LON_HI - 1e-6, 240)
        xx, yy = rb.project(ll, np.full_like(ll, la))
        ax.plot(xx, yy, color="#8C8884", lw=0.5, ls=(0, (4, 3)), zorder=4)


def draw(ax, rgba, B, f, lat_view, borders=True, xlim=None):
    rb.draw_globe(ax, zorder=0, top_face="white", borders=borders)
    ext_src = (-180.0, 180.0, MOSAIC_LAT_TOP - B.shape[0] * f * DELIN_DEG, MOSAIC_LAT_TOP)
    w, ext = rb.warp_raster(rgba, ext_src, out_w=4200, lat_clip=(-rb.TROPIC, rb.TROPIC))
    ax.imshow(w, extent=ext, origin="upper", interpolation="nearest", zorder=0.55, aspect="auto")
    frame(ax, lat_view, xlim)


def single(V, ok, ramp):
    b = np.percentile(V[ok & (V > 0)], [20, 40, 60, 80])
    k = np.digitize(V, b)
    rgba = np.zeros(V.shape + (4,), np.uint8)
    lut = np.array([to_rgba(c) for c in ramp])
    m = ok & (V > 0)
    rgba[m] = (lut[k[m]] * 255).astype(np.uint8)
    return rgba, b


def strip_key(fig, rect, ramp, breaks, label):
    k = fig.add_axes(rect)
    for i, c in enumerate(ramp):
        k.add_patch(Rectangle((i, 0), 1, 1, facecolor=c, edgecolor="white", lw=0.5))
    k.set_xlim(0, 5); k.set_ylim(0, 1); k.set_yticks([])
    k.set_xticks(range(1, 5), [f"{v:.2f}" if v < 1 else f"{v:.1f}" for v in breaks])
    k.tick_params(length=2, pad=1)
    for s in k.spines.values():
        s.set_visible(False)
    k.set_xlabel(label, fontsize=8, labelpad=2)


SPLIT_LON = 57.0


def lat_for_y(y):
    lo, hi = 0.0, 80.0
    for _ in range(60):
        m = (lo + hi) / 2
        if float(rb.lat_to_y(m)) < y:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def figure(name="figure_bivariate_decomp_v3", f=30, lat_view=27.0, mode=None):
    st.apply_rc(min_pt=6)
    r, c, base, add = fb.lattice()
    B, A = fb.grid(r, c, base, add, f)
    floor = 0.5 * (f / 60) ** 2
    rgba, bb, ab, ok = fb.classify(B, A, floor)
    W = tp.PAGE_W
    (x0, x1), _ = rb.band_limits(0.797)
    yv = float(rb.lat_to_y(lat_view))
    KW = 1.05
    map_w = W - KW
    map_h = map_w * 2 * yv / (x1 - x0)
    gapc = 0.25
    labw = 0.95
    sw = (W - 0.10 - 0.15 - 2 * gapc - labw) / 3
    sh0 = sw * 2 * yv / (x1 - x0)
    low_h0 = sh0 + 0.10
    xm = float(rb.project([SPLIT_LON], [0.0])[0][0])
    if mode == "split":
        spans = [(x0, xm), (xm, x1)]
        sx_w = max(b_ - a_ for a_, b_ in spans)
        strip_h = sw * 2 * yv / sx_w
        sh = 2 * strip_h + 0.04
    elif mode == "tall":
        lat_small = lat_for_y(2 * yv)
        sh = 2 * sh0
    else:
        sh = sh0
    low_h = sh + 0.10
    H = map_h + 0.42 + low_h + 0.55 + 0.12
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([KW / W, 1 - map_h / H, map_w / W, map_h / H])
    draw(ax, rgba, B, f, lat_view)
    tf.robinson_degree_labels(ax, lat_side="right", lat_inside=True)
    fig.text(0.02 / W, 1 - 0.02 / H, "a", fontsize=st.MIN_PT + 1, fontweight="bold", va="top")
    fig.text(0.12 / W, 1 - 0.16 / H, "Impervious surface\nper 0.25° cell", fontsize=6.5,
             va="top", linespacing=1.1)
    fb.key_square(fig, [0.40 / W, 1 - 1.12 / H, 0.56 / W, 0.56 / H], bb, ab)

    yb = 0.74
    for i, (V, ramp, lab, letter) in enumerate(((B, BLUES, "Impervious by 2000 (km² per cell)", "b"),
                                                (A, ORANGES, "Added 2000–2020 (km² per cell)", "c"))):
        rg, br = single(V, ok, ramp)
        sx = 0.10 + i * (sw + gapc)
        if mode == "split":
            for q, (a_, b_) in enumerate(spans):
                wq = sw * (b_ - a_) / sx_w
                sa = fig.add_axes([sx / W, (yb + low_h - (q + 1) * strip_h - q * 0.04) / H, wq / W, strip_h / H])
                draw(sa, rg, V, f, lat_view, borders=False, xlim=(a_, b_))
        elif mode == "tall":
            sa = fig.add_axes([sx / W, (yb + low_h - sh) / H, sw / W, sh / H])
            draw(sa, rg, V, f, lat_small, borders=False)
            tf.robinson_degree_labels(sa, lons=[-60, 0, 60, 120], lat_side="right",
                                      lat_inside=True)
        else:
            sa = fig.add_axes([sx / W, (yb + low_h - sh) / H, sw / W, sh / H])
            draw(sa, rg, V, f, lat_view, borders=False)
        fig.text(sx / W, (yb + low_h + 0.06) / H, letter, fontsize=st.MIN_PT + 1, fontweight="bold", va="bottom")
        fig.text((sx + 0.15) / W, (yb + low_h + 0.06) / H, lab.split(" (")[0], fontsize=8, fontweight="bold",
                 va="bottom")
        strip_key(fig, [(sx + 0.15) / W, (yb + low_h - sh - 0.32) / H, (sw - 0.30) / W, 0.09 / H], ramp, br,
                  "km² per 0.25° cell")

    lon = -180.0 + (np.arange(B.shape[1]) + 0.5) * f * DELIN_DEG
    LON = np.broadcast_to(lon, B.shape)
    bx = np.digitize(B, bb); ay = np.digitize(A, ab)
    cls = ay * 3 + bx
    dx0 = 0.10 + 2 * (sw + gapc)
    dtop = yb + low_h
    da = fig.add_axes([(dx0 + labw) / W, (dtop - low_h0 - 0.25 + 0.0) / H, sw / W, (low_h0 + 0.25) / H])
    names = []
    for k_, (nm, lo, hi) in enumerate(SECTORS):
        m = ok & (LON >= lo) & (LON < hi)
        cnt = np.bincount(cls[m], minlength=9) / m.sum() * 100
        left = 0.0
        order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for q in order:
            j, i = divmod(q, 3)
            da.barh(k_, cnt[q], left=left, height=0.66, color=fb.PAL[j, i], edgecolor="white", lw=0.3)
            left += cnt[q]
        names.append(nm)
    short = {"Americas": "Americas", "Africa and Arabia": "Africa and\nArabia", "South Asia": "South Asia",
             "Southeast Asia and Oceania": "Southeast Asia\nand Oceania"}
    da.set_yticks(range(len(SECTORS)), [short[n] for n in names], fontsize=8)
    da.invert_yaxis(); da.set_xlim(0, 100)
    da.set_xlabel("Share of cells (%)")
    for s in ("top", "right"):
        da.spines[s].set_visible(False)
    da.tick_params(axis="y", length=0)
    fig.text(dx0 / W, (yb + low_h + 0.06) / H, "d", fontsize=st.MIN_PT + 1, fontweight="bold",
             va="bottom")
    fig.text((dx0 + 0.15) / W, (yb + low_h + 0.06) / H, "Classes of a by region", fontsize=8,
             fontweight="bold", va="bottom")
    return tp.save(fig, name)


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else None
    nm = {"split": "figure_bivariate_decomp_v4_split", "tall": "figure_bivariate_decomp_v4_tall"}.get(m)
    print(figure(nm, mode=m) if nm else figure())
