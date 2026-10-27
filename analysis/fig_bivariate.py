"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
import tropics_style as st
import tropics_panels as tp
import tropics_robinson as rb
import tropics_temporal as t
from tropics_urban import MOSAIC_ROWS, MOSAIC_COLS, CELLS_PER_DELIN
from tropics_config import MOSAIC_LAT_TOP, DELIN_DEG
from tropics_grid import cell_area_km2

PAL = np.array([["#EDEDE8", "#BBD7EE", "#6FA6D8"],
                ["#F9D35B", "#B8BC9A", "#8AA5BC"],
                ["#EE7D1E", "#C27443", "#7E6866"]])
WINDOWS = [("Gulf of Guinea", -3.5, 9.0, 4.0, 10.25),
           ("Java", 105.2, 115.2, -9.3, -4.3),
           ("Southeastern Brazil", -50.5, -40.0, -24.2, -19.0)]
COVER = 0.797


def lattice():
    z = np.load(t.OUT_DIR / "lattice_cum_counts.npz")
    key, cum = z["key"], z["cum"].astype(np.int64)
    r, c = key // MOSAIC_COLS, key % MOSAIC_COLS
    ok = r < MOSAIC_ROWS
    r, c, cum = r[ok], c[ok], cum[ok]
    pa = cell_area_km2(MOSAIC_LAT_TOP, MOSAIC_ROWS, DELIN_DEG) / CELLS_PER_DELIN
    base = cum[:, 0] * pa[r]
    add = (cum[:, 4] - cum[:, 0]) * pa[r]
    lat = MOSAIC_LAT_TOP - (r + 0.5) * DELIN_DEG
    band = np.abs(lat) <= rb.TROPIC
    return r[band], c[band], base[band], add[band]


def grid(r, c, base, add, f, rows=None, cols=None):
    gr, gc = r // f, c // f
    R = int(np.ceil(MOSAIC_ROWS / f)); C = MOSAIC_COLS // f
    B = np.zeros((R, C)); A = np.zeros((R, C))
    np.add.at(B, (gr, gc), base); np.add.at(A, (gr, gc), add)
    return B, A


def classify(B, A, floor, bb=None, ab=None):
    ok = (A + B) >= floor
    if bb is None:
        bb = np.percentile(B[ok], [100 / 3, 200 / 3])
        ab = np.percentile(A[ok], [100 / 3, 200 / 3])
    bx = np.digitize(B, bb); ay = np.digitize(A, ab)
    rgba = np.zeros(B.shape + (4,), np.uint8)
    from matplotlib.colors import to_rgba
    lut = np.array([[to_rgba(PAL[j, i]) for i in range(3)] for j in range(3)])
    rgba[ok] = (lut[ay[ok], bx[ok]] * 255).astype(np.uint8)
    rgba[~ok, 3] = 0
    return rgba, bb, ab, ok


def key_square(fig, rect, bb, ab, unit="km²", parent=None):
    k = parent.inset_axes(rect, transform=parent.transData) if parent is not None else fig.add_axes(rect)
    for j in range(3):
        for i in range(3):
            k.add_patch(Rectangle((i, j), 1, 1, facecolor=PAL[j, i], edgecolor="white", lw=0.6))
    k.set_xlim(0, 3); k.set_ylim(0, 3); k.set_aspect("equal")
    k.set_xticks([1, 2], [f"{v:.0f}" if v >= 10 else f"{v:.1f}" for v in bb])
    k.set_yticks([1, 2], [f"{v:.0f}" if v >= 10 else f"{v:.1f}" for v in ab])
    k.tick_params(length=2, pad=1.5)
    for sp in k.spines.values():
        sp.set_visible(False)
    k.set_xlabel(f"By 2000 ({unit}) →", fontsize=st.MIN_PT, labelpad=1)
    k.set_ylabel(f"Added ({unit}) →", fontsize=st.MIN_PT, labelpad=1)
    return k


def zoom_grid(r, c, base, add, win, f=6):
    name, lo0, lo1, la0, la1 = win
    r0 = int((MOSAIC_LAT_TOP - la1) / DELIN_DEG); r1 = int(np.ceil((MOSAIC_LAT_TOP - la0) / DELIN_DEG))
    c0 = int((lo0 + 180) / DELIN_DEG); c1 = int(np.ceil((lo1 + 180) / DELIN_DEG))
    m = (r >= r0) & (r < r1) & (c >= c0) & (c < c1)
    B, A = grid(r[m] - r0, c[m] - c0, base[m], add[m], f)
    nr, nc = (r1 - r0) // f + 1, (c1 - c0) // f + 1
    return B[:nr, :nc], A[:nr, :nc]


def zoom(ax, B, A, win, bb, ab, f=6):
    name, lo0, lo1, la0, la1 = win
    geoms, land = rb._load()
    from shapely.geometry import box
    clip = land.intersection(box(lo0 - 1, la0 - 1, lo1 + 1, la1 + 1))
    for g in getattr(clip, "geoms", [clip]):
        if g.is_empty or g.geom_type != "Polygon":
            continue
        ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), closed=True, facecolor="white",
                                edgecolor="#B9B6B1", lw=0.4, zorder=0))
    for gg in geoms:
        gi = gg.intersection(box(lo0 - 1, la0 - 1, lo1 + 1, la1 + 1))
        if gi.is_empty:
            continue
        for p in getattr(gi, "geoms", [gi]):
            if p.geom_type == "Polygon":
                xy = np.asarray(p.exterior.coords)
                ax.plot(xy[:, 0], xy[:, 1], color=rb.BORDER, lw=0.35, zorder=0.5)
    rgba, _, _, ok = classify(B, A, 0.05, bb, ab)
    ext = (lo0, lo0 + B.shape[1] * f * DELIN_DEG, la1 - B.shape[0] * f * DELIN_DEG, la1)
    ax.imshow(rgba, extent=ext, origin="upper", interpolation="nearest", zorder=1)
    ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1)
    ax.set_aspect(1 / np.cos(np.deg2rad((la0 + la1) / 2)), adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(st.FAINT); sp.set_linewidth(0.6)
    ax.set_facecolor("#F4F7FA")


def figure(name="figure_bivariate_v9", f_main=30, lat_view=38.0):
    st.apply_rc()
    r, c, base, add = lattice()
    B, A = grid(r, c, base, add, f_main)
    floor = 0.5 * (f_main / 60) ** 2
    rgba, bb, ab, ok = classify(B, A, floor)
    W = tp.PAGE_W
    yv = float(rb.lat_to_y(lat_view))
    xpad = rb.X_MAX * 0.012
    asp = (2 * yv + 0.02 * yv) / (2 * rb.X_MAX + 2 * xpad)
    map_h = W * asp
    gap = 0.22
    zw = (W - 0.10 - 2 * gap) / 3
    zh = zw * 0.52
    H = map_h + 0.40 + zh + 0.62
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 1 - map_h / H, 1, map_h / H])
    rb.draw_globe(ax, zorder=0, top_face="white")
    ext_src = (-180.0, 180.0, MOSAIC_LAT_TOP - B.shape[0] * f_main * DELIN_DEG, MOSAIC_LAT_TOP)
    warped, ext = rb.warp_raster(rgba, ext_src, out_w=4200, lat_clip=(-rb.TROPIC, rb.TROPIC))
    ax.imshow(warped, extent=ext, origin="upper", interpolation="nearest", zorder=0.55, aspect="auto")
    lats = np.linspace(-lat_view, lat_view, 200)
    lx, ly = rb.project(np.full_like(lats, rb.LON_LO + 1e-6), lats)
    rx, ry = rb.project(np.full_like(lats, rb.LON_HI - 1e-6), lats)
    outline = np.vstack([np.column_stack([lx, ly]), np.column_stack([rx, ry])[::-1]])
    from matplotlib.patches import Polygon as Pg
    clip = Pg(outline, closed=True, facecolor="none", edgecolor="#B9B6B1", lw=0.6, zorder=6)
    ax.add_patch(clip)
    for art in list(ax.images) + list(ax.collections) + list(ax.lines):
        art.set_clip_path(clip)
    ax.set_xlim(-rb.X_MAX - xpad, rb.X_MAX + xpad); ax.set_ylim(-yv * 1.01, yv * 1.01)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for la in (rb.TROPIC, -rb.TROPIC):
        lons = np.linspace(rb.LON_LO + 1e-6, rb.LON_HI - 1e-6, 240)
        xx, yy = rb.project(lons, np.full_like(lons, la))
        ax.plot(xx, yy, color="#8C8884", lw=0.6, ls=(0, (4, 3)), zorder=4)
    for i, w in enumerate(WINDOWS):
        _, lo0, lo1, la0, la1 = w
        bx = np.array([lo0, lo1, lo1, lo0, lo0]); by = np.array([la0, la0, la1, la1, la0])
        bxx = np.concatenate([np.linspace(bx[k], bx[k + 1], 20) for k in range(4)])
        byy = np.concatenate([np.linspace(by[k], by[k + 1], 20) for k in range(4)])
        px, py = rb.project(bxx, byy)
        ax.plot(px, py, color=st.INK, lw=0.7, zorder=5)
        ax.text(px.max(), py.max(), "bcd"[i], fontsize=st.MIN_PT + 1, fontweight="bold",
                ha="left", va="bottom", zorder=6)
    ax.text(0.0, 1.0, "a", transform=ax.transAxes, fontsize=st.MIN_PT + 1, fontweight="bold",
            va="top", ha="left")

    kx0, ky0 = rb.project([-127.0], [-22.0]); kx1, ky1 = rb.project([-95.0], [10.0])
    side = min(kx1[0] - kx0[0], ky1[0] - ky0[0])
    key_square(fig, [kx0[0], ky0[0], side, side], bb, ab, parent=ax)
    tx, ty = rb.project([-146.0], [11.0])
    ax.text(tx[0], ty[0], "Impervious surface", fontsize=st.MIN_PT, fontweight="bold",
            ha="left", va="bottom", zorder=7)

    Z = [zoom_grid(r, c, base, add, w) for w in WINDOWS]
    Bp = np.concatenate([z[0].ravel() for z in Z]); Ap = np.concatenate([z[1].ravel() for z in Z])
    okp = (Bp + Ap) >= 0.05
    bbz = np.percentile(Bp[okp], [100 / 3, 200 / 3]); abz = np.percentile(Ap[okp], [100 / 3, 200 / 3])

    y_row = 0.62
    for i, (w, (Bz, Az)) in enumerate(zip(WINDOWS, Z)):
        zx = 0.05 + i * (zw + gap)
        za = fig.add_axes([zx / W, y_row / H, zw / W, zh / H])
        zoom(za, Bz, Az, w, bbz, abz)
        za.text(0.0, 1.03, "bcd"[i], transform=za.transAxes, fontsize=st.MIN_PT + 1,
                fontweight="bold", va="bottom", ha="left")
        za.text(0.06, 1.03, w[0], transform=za.transAxes, fontsize=st.MIN_PT, fontweight="bold",
                va="bottom", ha="left")
    fig.text(0.05 / W, 0.08 / H,
             f"Panels b–d share one set of classes per 0.05° cell. Impervious by 2000: breaks at "
             f"{bbz[0]:.2f} and {bbz[1]:.2f} km².\nAdded 2000–2020: breaks at {abz[0]:.2f} and "
             f"{abz[1]:.2f} km². Panel a uses 0.25° cells. Classes are terciles of cells holding at least\n"
             f"0.05 km² (b–d) or 0.125 km² (a) of impervious surface by 2020.",
             fontsize=st.MIN_PT, ha="left", va="bottom", color=st.MUTED, linespacing=1.15)
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
