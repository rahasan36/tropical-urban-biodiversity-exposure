"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle, Polygon
import tropics_style as st
import matplotlib.patheffects as pe
import tropics_panels as tp
import tropics_robinson as rb
from tropics_measure import MEASURE_PATH

COVER = 0.797
POW = 0.5
SEG_COL = ["#5F7391", "#F6D6B8", "#EDB07C", "#DC8640", "#A84C0C"]
SEG_LAB = ["Impervious by 2000", "Added 2000–2005", "Added 2005–2010", "Added 2010–2015",
           "Added 2015–2020"]
NAMES = {"Mexico City": "right", "Lima": "left", "Rio de Janeiro": "left", "Lagos": "left",
         "Khartoum": "left", "Chennai": "left", "Bangkok": "right", "Hanoi": "right",
         "Guangzhou": "left", "Manila": "left", "Jakarta": "left"}


def data():
    d = tp.cities(sample=False)
    z = np.load(MEASURE_PATH, allow_pickle=False)
    ce = z["city_epoch"]
    seg = np.column_stack([ce[d.city_id, 1:5].sum(1)] + [ce[d.city_id, k] for k in (5, 6, 7, 8)])
    x, y = rb.project(d.lon.to_numpy(), d.lat.to_numpy())
    return d.assign(x=x, y=y, tot=seg.sum(1)), seg


def height(v, hmax, vmax):
    return hmax * (np.asarray(v, float) / vmax) ** POW


def spikes(ax, d, seg, hmax, w0, vmax):
    order = np.argsort(-d.y.to_numpy())
    verts, cols = [], []
    for i in order:
        tot = d.tot.iat[i]
        if tot <= 0:
            continue
        H = float(height(tot, hmax, vmax))
        x, y = d.x.iat[i], d.y.iat[i]
        f = np.clip(np.concatenate([[0], np.cumsum(seg[i])]) / tot, 0, 1)
        cum = H * (1 - np.sqrt(1 - f))
        for k in range(5):
            a, b = cum[k], cum[k + 1]
            if b - a <= 0:
                continue
            wa, wb = w0 * (1 - a / H) / 2, w0 * (1 - b / H) / 2
            verts.append([(x - wa, y + a), (x + wa, y + a), (x + wb, y + b), (x - wb, y + b)])
            cols.append(SEG_COL[k])
    ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors="none", zorder=3))


def repel(xs, widths, lo, hi, gap):
    o = np.argsort(xs)
    p = np.asarray(xs, float)[o].copy(); w = np.asarray(widths, float)[o]
    for _ in range(200):
        moved = False
        for i in range(1, len(p)):
            need = (w[i - 1] + w[i]) / 2 + gap
            if p[i] - p[i - 1] < need:
                s = (need - (p[i] - p[i - 1])) / 2
                p[i - 1] -= s; p[i] += s; moved = True
        p = np.clip(p, lo + w / 2, hi - w / 2)
        if not moved:
            break
    out = np.empty_like(p); out[o] = p
    return out


def skyline(ax, d, seg, x0, x1, nbin=150):
    e = np.linspace(x0, x1, nbin + 1)
    b = np.clip(np.digitize(d.x.to_numpy(), e) - 1, 0, nbin - 1)
    bot = np.zeros(nbin)
    w = (e[1] - e[0]) * 0.86
    for k in range(1, 5):
        v = np.bincount(b, weights=seg[:, k], minlength=nbin)
        ax.bar(e[:-1] + (e[1] - e[0]) / 2, v, w, bottom=bot, color=SEG_COL[k], linewidth=0)
        bot += v
    ax.set_xlim(x0, x1)
    return bot


def figure(name="figure_size_map_v6"):
    st.apply_rc()
    d, seg = data()
    W = tp.PAGE_W
    (x0, x1), (y0, y1) = rb.band_limits(COVER)
    span = x1 - x0
    yt = 2.45e6 + 1.75e6 + 0.30e6
    head = yt - y1
    ML, MR = 0.56, 0.05
    map_w = W - ML - MR
    map_h = map_w * (yt - y0) / span
    sky_h, gap, bot_m = 0.95, 0.12, 0.62
    H = map_h + gap + sky_h + bot_m + 0.12
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([ML / W, (bot_m + sky_h + gap) / H, map_w / W, map_h / H])
    rb.draw_globe(ax, zorder=0)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, yt); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(Rectangle((x0 - span, y1), 3 * span, 2 * head, facecolor="white",
                           edgecolor="none", zorder=1))

    vmax = d.tot.max()
    y_g = float(d.loc[d.tot.idxmax(), "y"])
    hmax = 1.75e6
    w0 = span * 0.0030
    spikes(ax, d, seg, hmax, w0, vmax)

    nm = tp.city_names(d.nlargest(30, "tot"))
    nm = nm[nm["name"].isin(list(NAMES))].drop_duplicates("name")
    pad = 0.006 * span
    for _, r in nm.iterrows():
        tip = r.y + float(height(r.tot, hmax, vmax))
        side = NAMES[r["name"]]
        xx = r.x + (pad if side == "left" else -pad)
        ax.text(xx, tip, r["name"], ha=side, va="center", fontsize=st.MIN_PT,
                color=st.INK, zorder=5,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    kx = float(rb.project([60.0], [-21.0])[0][0])
    ky = float(rb.lat_to_y(-21.0))
    ax.text(kx - w0, ky + float(height(5000, hmax, vmax)) + 0.12e6, "Impervious\narea, 2020", fontsize=st.MIN_PT, ha="left",
            va="bottom", color=st.INK, zorder=5)
    for j, v in enumerate((100, 1000, 5000)):
        xx = kx + j * 0.068 * span
        hh = float(height(v, hmax, vmax))
        ax.add_patch(Polygon([(xx - w0 / 2, ky), (xx + w0 / 2, ky), (xx, ky + hh)],
                             facecolor=SEG_COL[0], edgecolor="none", zorder=5))
        ax.text(xx, ky - 0.08e6, f"{v:,}" + (" km²" if v == 5000 else ""), fontsize=st.MIN_PT,
                ha="center", va="top", color=st.INK, zorder=5)
    ax.text(-ML / map_w, 1.0, "a", transform=ax.transAxes, fontsize=st.MIN_PT + 1,
            fontweight="bold", va="top", ha="left")
    ax.text(0.0, 1.0, "Agglomeration size in 2020 and the period its land was added",
            transform=ax.transAxes, fontsize=st.MIN_PT, fontweight="bold", va="top", ha="left")

    bx = fig.add_axes([ML / W, bot_m / H, map_w / W, sky_h / H])
    tot = skyline(bx, d, seg, x0, x1)
    bx.set_ylabel("Land added (km²)")
    bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    lon_t = [-90, -60, -30, 0, 30, 60, 90, 120, 150]
    bx.set_xticks(rb.project(np.array(lon_t, float), np.zeros(len(lon_t)))[0],
                  [f"{abs(v)}°{'W' if v < 0 else ('E' if v > 0 else '')}" for v in lon_t])
    bx.text(-ML / map_w, 1.08, "b", transform=bx.transAxes, fontsize=st.MIN_PT + 1,
            fontweight="bold", va="bottom", ha="left")
    bx.text(0.0, 1.08, "Urban land added 2000–2020 by longitude", transform=bx.transAxes,
            fontsize=st.MIN_PT, fontweight="bold", va="bottom", ha="left")
    h = [Rectangle((0, 0), 1, 1, color=c) for c in SEG_COL]
    fig.legend(h, SEG_LAB, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=False,
               handlelength=1.1, handletextpad=0.4, columnspacing=1.0, fontsize=st.MIN_PT)
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
