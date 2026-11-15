"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext
from scipy import stats
import tropics_style as st
import tropics_panels as tp
import tropics_robinson as rb
import tropics_frame as tf
import fig_country_violin as cv
import fig_recency_rarity as rr


def panel_country_scatter(ax, c):
    area = c.tot.to_numpy()
    size = 8 + 140 * np.sqrt(area / area.max())
    for i, r in enumerate(rr.REALMS):
        m = (c.realm == r).to_numpy()
        ax.scatter(c.recent[m], c.rpk[m] * 1e5, s=size[m], color=st.CAT[i], marker=st.CAT_MARKERS[i],
                   alpha=0.75, edgecolor="white", lw=0.5, zorder=3)
    other = ~c.realm.isin(rr.REALMS).to_numpy()
    if other.any():
        ax.scatter(c.recent[other], c.rpk[other] * 1e5, s=size[other], color=st.FAINT, edgecolor="white",
                   lw=0.5, zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(5, 88)
    ax.set_ylim(0.001, 3000)
    ax.set_xticks([20, 40, 60])
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    for code, ty in rr.LABELS:
        x0, y0 = c.recent[code], c.rpk[code] * 1e5
        ax.plot([x0, 64.5], [y0, ty], color=st.MUTED, lw=0.5, zorder=1)
        ax.text(65.2, ty, rr.NAMES[code], fontsize=8, color=st.INK, va="center", ha="left", zorder=4)
    rho = stats.spearmanr(c.recent, c.rpk)
    ax.text(0.98, 0.98, "$\\rho$ = −%.2f\n$P$ = %.1f × 10$^{-5}$" % (abs(rho.statistic), rho.pvalue * 1e5),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=st.INK)
    ax.set_xlabel("Share of 2020 surface\nadded after 2000 (%)")
    ax.set_ylabel("Rarity-weighted intensity\n(10$^{-5}$ km$^{-2}$)")
    st.despine(ax)
    hs = [Line2D([], [], marker=st.CAT_MARKERS[i], ls="", color=st.CAT[i], ms=4.5) for i in range(4)]
    ax.legend(hs, rr.REALMS, loc="lower left", ncol=1, fontsize=7, handletextpad=0.1, borderaxespad=0.1,
              labelspacing=0.15)


def figure(name="figure_country_recency_v2"):
    st.apply_rc()
    W = tp.PAGE_W
    letter = dict(fontsize=st.MIN_PT + 1, fontweight="bold", color=st.INK)

    (x0, x1), _ = rb.band_limits(0.797)
    yv = float(rb.lat_to_y(30.0))
    KW, low_h = 0.95, 1.75
    map_h = (W - KW) * (2 * yv) / (x1 - x0)
    H1 = map_h + 0.44 + low_h + 0.62
    H2, gap = 2.75, -0.12
    H = H1 + gap + H2
    fig = plt.figure(figsize=(W, H))
    off = H2 + gap

    d = cv.table()
    ax = fig.add_axes([KW / W, (H - map_h) / H, (W - KW) / W, map_h / H])
    c = cv.panel_map(ax, d)
    tf.robinson_degree_labels(ax, lat_side="right", lat_inside=True, lat_va="outward")
    kax = fig.add_axes([0.10 / W, (H - map_h) / H, (KW - 0.12) / W, map_h / H])
    kax.axis("off")
    cv.key(kax, [0.0, 0.02, 1.0, 0.96])
    fig.text(0.005, 1 - 0.02 / H, "a", va="top", **letter)
    bx = fig.add_axes([0.55 / W, (off + 0.62) / H, 2.85 / W, low_h / H])
    cv.panel_violins(bx, d, c)
    yl = (off + 0.62 + low_h + 0.12) / H
    fig.text(0.005, yl, "b", va="bottom", **letter)
    cw = (W - 3.95 - 0.10 - 2 * 0.30) / 3
    axs = [fig.add_axes([(3.95 + i * (cw + 0.30)) / W, (off + 0.62) / H, cw / W, (low_h - 0.12) / H])
           for i in range(3)]
    cv.panel_existing(axs, d)
    fig.text(3.55 / W, yl, "c", va="bottom", **letter)

    r = rr.table()
    cc = rr.countries(r)
    ml, g2, mr, bot, top = 0.70, 0.80, 0.04, 0.62, 0.30
    pw = (W - ml - mr - 2 * g2) / 3
    ph = H2 - bot - top
    ws = [pw - 0.05, pw - 0.05, pw + 0.10]
    xs = [ml, ml + ws[0] + g2, ml + ws[0] + ws[1] + 2 * g2]
    lo = [fig.add_axes([xs[i] / W, bot / H, ws[i] / W, ph / H]) for i in range(3)]
    rr.panel_hex(lo[0], r, "cost_per_km2", 1.0, "Exposure intensity\n(km² per km² added)", (1, 2000))
    rr.panel_hex(lo[1], r, "rarity_per_km2", 1e5, "Rarity-weighted intensity\n(10$^{-5}$ km$^{-2}$)",
                 (0.01, 100000))
    panel_country_scatter(lo[2], cc)
    for a, l in zip(lo, "def"):
        fig.text((xs["def".index(l)] - 0.62) / W, (bot + ph + 0.10) / H, l, va="bottom", **letter)
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
