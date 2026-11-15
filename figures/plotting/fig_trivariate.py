"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon as Pg
from matplotlib.colors import to_rgb
import tropics_style as st
import tropics_panels as tp
import tropics_robinson as rb
import tropics_frame as tf
import tropics_temporal as t
from tropics_urban import MOSAIC_ROWS, MOSAIC_COLS, CELLS_PER_DELIN
from tropics_config import MOSAIC_LAT_TOP, DELIN_DEG
from tropics_grid import cell_area_km2
import fig_bivariate as fb

PRIM = ["#3F86D1", "#E69F00", "#D63A6E"]
PLAB = ["By 2000", "Added 2000–2010", "Added 2010–2020"]
WIN4 = [("São Paulo and Rio", -48.0, -42.8, -24.1, -20.4),
        ("Lagos and Ibadan", 1.6, 5.6, 5.9, 8.9),
        ("Chennai and Bengaluru", 76.9, 81.1, 11.6, 14.6),
        ("Jakarta and Bandung", 105.5, 109.5, -8.0, -5.0)]


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gam(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WP = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb):
    xyz = _lin(rgb) @ _M.T / _WP
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116; fx = fy + lab[..., 1] / 500; fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WP
    return _gam(xyz @ np.linalg.inv(_M).T)


PLABS = rgb2lab(np.array([to_rgb(c) for c in PRIM]))


NEUTRAL = "#D8D8D8"


def ternary_color(w, center, power=0.45):
    w = np.asarray(w, float)
    w = w / np.maximum(w.sum(-1, keepdims=True), 1e-12)
    v = w / center
    v = v / v.sum(-1, keepdims=True)
    mix = lab2rgb(v @ PLABS)
    dev = np.clip(np.abs(v - 1 / 3).sum(-1) / (4 / 3), 0, 1) ** power
    n = np.array(to_rgb(NEUTRAL))
    return n * (1 - dev[..., None]) + mix * dev[..., None]


def lattice_parts():
    z = np.load(t.OUT_DIR / "lattice_cum_counts.npz")
    key, cum = z["key"], z["cum"].astype(np.int64)
    r, c = key // MOSAIC_COLS, key % MOSAIC_COLS
    ok = r < MOSAIC_ROWS
    r, c, cum = r[ok], c[ok], cum[ok]
    lat = MOSAIC_LAT_TOP - (r + 0.5) * DELIN_DEG
    band = np.abs(lat) <= rb.TROPIC
    r, c, cum, lat = r[band], c[band], cum[band], lat[band]
    pa = cell_area_km2(MOSAIC_LAT_TOP, MOSAIC_ROWS, DELIN_DEG)[r] / CELLS_PER_DELIN
    P = np.column_stack([cum[:, 0], cum[:, 2] - cum[:, 0], cum[:, 4] - cum[:, 2]]) * pa[:, None]
    lon = -180.0 + (c + 0.5) * DELIN_DEG
    return lon, lat, P


def stagger(x, y, s):
    h = s * np.sqrt(3) / 2
    j = np.round(y / h).astype(np.int64)
    off = (j % 2) * s / 2
    i = np.round((x - off) / s).astype(np.int64)
    return i, j, i * s + (j % 2) * s / 2, j * h


def dots(ax, x, y, P, s, center, cap=None, rmin=0.16, floor=0.0, pct=90, rmax=1.0, edge=False):
    i, j, _, _ = stagger(x, y, s)
    k = i * 1_000_003 + j
    u, inv = np.unique(k, return_inverse=True)
    S = np.column_stack([np.bincount(inv, weights=P[:, q]) for q in range(3)])
    ui = np.empty(u.size, np.int64); uj = np.empty(u.size, np.int64)
    ui[inv] = i; uj[inv] = j
    cx = ui * s + (uj % 2) * s / 2; cy = uj * s * np.sqrt(3) / 2
    tot = S.sum(1)
    keep = tot > floor
    cap = cap or float(np.percentile(tot[keep], pct))
    rad = 0.5 * s * 0.96 * rmax * np.clip(np.sqrt(tot / cap), rmin / rmax, 1.0)
    col = ternary_color(S, center)
    o = np.argsort(-tot)
    pc = PatchCollection([Circle((cx[q], cy[q]), rad[q]) for q in o if keep[q]],
                         facecolors=col[o][keep[o]], edgecolors="white" if edge else "none",
                         linewidths=0.25 if edge else 0, zorder=3)
    ax.add_collection(pc)
    return cap


def tri_key(ax, rect, center, n=24):
    k = ax.inset_axes(rect, transform=ax.transData)
    k.set_aspect("equal"); k.axis("off")
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    tris, cols = [], []
    for a in range(n):
        for b in range(n - a):
            for up in (True, False):
                if up:
                    pts = [(a, b), (a + 1, b), (a, b + 1)]
                else:
                    if a + b + 1 >= n:
                        continue
                    pts = [(a + 1, b), (a + 1, b + 1), (a, b + 1)]
                bc = np.array([[n - p - q, p, q] for p, q in pts], float) / n
                xy = bc @ V
                w = bc.mean(0)
                tris.append(Pg(xy, closed=True)); cols.append(ternary_color(w[None], center)[0])
    k.add_collection(PatchCollection(tris, facecolors=cols, edgecolors=cols, lw=0.2))
    m = np.asarray(center) @ V
    k.plot(*m, "o", ms=3, mfc="white", mec=st.INK, mew=0.6)
    k.text(V[0, 0], V[0, 1] - 0.06, "By\n2000", ha="center", va="top", fontsize=6,
           linespacing=1.0)
    k.text(V[1, 0], V[1, 1] - 0.06, "Added\n2000–2010", ha="center", va="top", fontsize=6,
           linespacing=1.0)
    k.text(V[2, 0], V[2, 1] + 0.05, "Added 2010–2020", ha="center", va="bottom", fontsize=6)
    k.set_xlim(-0.30, 1.30); k.set_ylim(-0.40, 1.08)
    return k


def figure(name="figure_trivariate_v9", lat_view=38.0, s_main=2.4e5, s_zoom=0.055,
           rmax_main=1.0, pct_main=97, name_v=None):
    st.apply_rc(min_pt=6)
    lon, lat, P = lattice_parts()
    center = P.sum(0) / P.sum()
    W = tp.PAGE_W
    yv = float(rb.lat_to_y(lat_view))
    xpad = rb.X_MAX * 0.012
    asp = (2 * yv * 1.01) / (2 * rb.X_MAX + 2 * xpad)
    map_h = W * asp
    gap, zl, zb = 0.17, 0.17, 0.18
    zw = (W - zl - 0.03 - 3 * gap) / 4
    zh = zw * 0.80
    H = map_h + 0.36 + zh + zb
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 1 - map_h / H, 1, map_h / H])
    rb.draw_globe(ax, zorder=0, top_face="white")
    x, y = rb.project(lon, lat)
    cap = dots(ax, np.asarray(x), np.asarray(y), P, s_main, center, floor=0.05, pct=pct_main,
               rmax=rmax_main, edge=True)
    lats = np.linspace(-lat_view, lat_view, 200)
    lx, ly = rb.project(np.full_like(lats, rb.LON_LO + 1e-6), lats)
    rx, ry = rb.project(np.full_like(lats, rb.LON_HI - 1e-6), lats)
    outline = np.vstack([np.column_stack([lx, ly]), np.column_stack([rx, ry])[::-1]])
    clip = Pg(outline, closed=True, facecolor="none", edgecolor="#B9B6B1", lw=0.6, zorder=6)
    ax.add_patch(clip)
    for art in list(ax.collections) + list(ax.lines) + list(ax.patches):
        if art is not clip:
            art.set_clip_path(clip)
    ax.set_xlim(-rb.X_MAX - xpad, rb.X_MAX + xpad); ax.set_ylim(-yv * 1.01, yv * 1.01)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    tf.robinson_degree_labels(ax, lat_side="right", lat_inside=True, x_margin=0.06)
    for t_ in ax.texts:
        t_.set_clip_on(False)
    for la in (rb.TROPIC, -rb.TROPIC):
        ll = np.linspace(rb.LON_LO + 1e-6, rb.LON_HI - 1e-6, 240)
        xx, yy = rb.project(ll, np.full_like(ll, la))
        ax.plot(xx, yy, color="#8C8884", lw=0.6, ls=(0, (4, 3)), zorder=4)
    for i, w in enumerate(WIN4):
        _, lo0, lo1, la0, la1 = w
        bx = np.array([lo0, lo1, lo1, lo0, lo0]); by = np.array([la0, la0, la1, la1, la0])
        bxx = np.concatenate([np.linspace(bx[k], bx[k + 1], 20) for k in range(4)])
        byy = np.concatenate([np.linspace(by[k], by[k + 1], 20) for k in range(4)])
        px, py = rb.project(bxx, byy)
        ax.plot(px, py, color=st.INK, lw=0.7, zorder=5)
        ax.text(px.max(), py.max(), "bcde"[i], fontsize=7, fontweight="bold",
                ha="left", va="bottom", zorder=6)
    ax.text(0.0, 1.0, "a", transform=ax.transAxes, fontsize=st.MIN_PT + 1, fontweight="bold",
            va="top", ha="left")
    kx0, ky0 = rb.project([-143.0], [-33.0]); kx1, _ = rb.project([-91.0], [-33.0])
    wdt = kx1[0] - kx0[0]
    tri_key(ax, [kx0[0], ky0[0], wdt, wdt * 0.92], center)
    tx, ty = rb.project([-143.0], [11.0])
    ax.text(tx[0], ty[0], "Share of 2020\nimpervious surface", fontsize=6, fontweight="normal",
            ha="left", va="bottom", zorder=7, linespacing=1.05)
    sx, sy = rb.project([-42.0], [-33.5])
    vals = (20, 200, int(round(cap, -2)))
    for q, v in enumerate(vals):
        rr = 0.5 * s_main * 0.96 * rmax_main * np.clip(np.sqrt(v / cap), 0.16 / rmax_main, 1.0)
        xx = sx[0] + q * 8.0 * s_main
        ax.add_patch(Circle((xx, sy[0]), rr, facecolor="#9A9A9A", edgecolor="none", zorder=7))
        ax.text(xx, sy[0] - 0.7 * s_main, ("≥" if q == 2 else "") + f"{v:,}", ha="center", va="top",
                fontsize=6, zorder=7)
    ax.text(sx[0] - 0.6 * s_main, sy[0] + 0.9 * s_main, "Impervious km² per dot", ha="left",
            va="bottom", fontsize=6, zorder=7)

    for i, w in enumerate(WIN4):
        name_, lo0, lo1, la0, la1 = w
        zx = zl + i * (zw + gap)
        za = fig.add_axes([zx / W, zb / H, zw / W, zh / H])
        m = (lon >= lo0 - 0.5) & (lon <= lo1 + 0.5) & (lat >= la0 - 0.5) & (lat <= la1 + 0.5)
        sc = np.cos(np.deg2rad((la0 + la1) / 2))
        geoms, land = rb._load()
        from shapely.geometry import box
        cl = land.intersection(box(lo0 - 1, la0 - 1, lo1 + 1, la1 + 1))
        for g in getattr(cl, "geoms", [cl]):
            if g.is_empty or g.geom_type != "Polygon":
                continue
            xy = np.asarray(g.exterior.coords)
            za.add_patch(Pg(np.column_stack([xy[:, 0] * sc, xy[:, 1]]), closed=True, facecolor="white",
                            edgecolor="#B9B6B1", lw=0.4, zorder=0))
        dots(za, lon[m] * sc, lat[m], P[m], s_zoom, center, cap=None, floor=0.01, pct=85, rmin=0.28)
        za.set_xlim(lo0 * sc, lo1 * sc); za.set_ylim(la0, la1)
        za.set_aspect("equal", adjustable="datalim")
        za.set_xticks([]); za.set_yticks([])
        for sp in za.spines.values():
            sp.set_color(st.FAINT); sp.set_linewidth(0.6)
        za.set_facecolor("#F4F7FA")
        za.apply_aspect()
        x0z, x1z = za.get_xlim(); y0z, y1z = za.get_ylim()
        tf.corner_labels(za, x0z / sc, x1z / sc, y0z, y1z)
        km = 50.0
        L = km / 111.32
        pos = {1: "center", 3: "left"}.get(i, "right")
        if pos == "right":
            bx1 = x1z - 0.05 * (x1z - x0z); bx0 = bx1 - L
        elif pos == "left":
            bx0 = x0z + 0.05 * (x1z - x0z); bx1 = bx0 + L
        else:
            bx0 = (x0z + x1z) / 2 - L / 2; bx1 = bx0 + L
        by = y0z + 0.07 * (y1z - y0z)
        za.add_patch(plt.Rectangle((bx0 - 0.03 * (x1z - x0z), by - 0.035 * (y1z - y0z)),
                                   L + 0.06 * (x1z - x0z), 0.14 * (y1z - y0z),
                                   facecolor="white", edgecolor="none", alpha=0.85, zorder=8))
        za.plot([bx0, bx1], [by, by], color=st.INK, lw=1.0, solid_capstyle="butt", zorder=9)
        for xx in (bx0, bx1):
            za.plot([xx, xx], [by, by + 0.02 * (y1z - y0z)], color=st.INK, lw=0.6, zorder=9)
        za.text((bx0 + bx1) / 2, by + 0.025 * (y1z - y0z), f"{km:.0f} km", ha="center", va="bottom",
                fontsize=6, color=st.INK, zorder=9)
        if i == 3:
            nx, ny = 0.93, 0.80
            za.annotate("", xy=(nx, ny + 0.13), xytext=(nx, ny), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=st.INK, lw=0.8, mutation_scale=7),
                        zorder=9)
            za.text(nx, ny + 0.145, "N", transform=za.transAxes, ha="center", va="bottom",
                    fontsize=6, fontweight="normal", color=st.INK, zorder=9)
        za.text(0.0, 1.03, "bcde"[i], transform=za.transAxes, fontsize=st.MIN_PT + 1,
                fontweight="bold", va="bottom", ha="left")
        za.text(0.085, 1.03, name_, transform=za.transAxes, fontsize=st.MIN_PT, fontweight="bold",
                va="bottom", ha="left")
    return tp.save(fig, name)


def _selftest():
    c = np.array([0.5, 0.25, 0.25])
    out = ternary_color(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], c]), c)
    assert np.all((out >= 0) & (out <= 1))
    lab = rgb2lab(np.array([to_rgb("#5B9BD5")]))
    back = lab2rgb(lab)
    assert np.allclose(back, np.array([to_rgb("#5B9BD5")]), atol=1e-6)
    i, j, gx, gy = stagger(np.array([0.0, 1.0, 0.5]), np.array([0.0, 0.0, np.sqrt(3) / 2]), 1.0)
    assert list(i) == [0, 1, 0] and list(j) == [0, 0, 1] and np.isclose(gx[2], 0.5)
    print("trivariate selftest ok")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["test"]:
        _selftest()
    else:
        _selftest(); print(figure())
