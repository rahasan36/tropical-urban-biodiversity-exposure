"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from scipy import stats
import tropics_style as st
import tropics_panels as tp
import fig_profile_map as pm

REALMS = ["Indomalayan", "Neotropic", "Afrotropic", "Australasia"]
NAMES = dict(NGA="Nigeria", IND="India", BRA="Brazil", IDN="Indonesia", MEX="Mexico", CHN="China")
LABELS = [("MEX", 150), ("IDN", 70), ("BRA", 33), ("IND", 15), ("NGA", 7), ("CHN", 3.2)]
RED = "#C8102E"
HEX = LinearSegmentedColormap.from_list("hex", mpl.colormaps["Blues"](np.linspace(0.22, 1.0, 256)))


def table():
    d = pm.city_table()
    keep = set(tp.cities(sample=True).city_id)
    d = d[d.city_id.isin(keep)].copy()
    d["recent"] = 100 * (1 - d.s0 / d.tot)
    return d


def countries(d, min_n=10):
    g = d.groupby("country").agg(n=("city_id", "size"), s0=("s0", "sum"), tot=("tot", "sum"),
                                 added=("urban_added_km2", "sum"), rar=("rarity_converted", "sum"),
                                 realm=("realm", lambda x: x.value_counts().index[0]))
    g = g[g.n >= min_n].copy()
    g["recent"] = 100 * (1 - g.s0 / g.tot)
    g["rpk"] = g.rar / g.added
    return g


def binned(x, y, k=10):
    q = pd.qcut(x, k, labels=False)
    b = pd.DataFrame(dict(x=x, y=y, q=q)).groupby("q")
    return b.x.median().to_numpy(), b.y.quantile(0.25).to_numpy(), b.y.median().to_numpy(), b.y.quantile(0.75).to_numpy()


def summary(d, c):
    t = pd.qcut(d.recent, 3, labels=False)
    out = dict(
        n=len(d),
        tercile_exposure_median=d.groupby(t).cost_per_km2.median().round(2).tolist(),
        tercile_exposure_geomean=d.groupby(t).cost_per_km2.apply(lambda v: float(np.exp(np.log(v).mean()))).round(2).tolist(),
        tercile_rarity_median=d.groupby(t).rarity_per_km2.median().tolist(),
        rho_city_exposure=stats.spearmanr(d.recent, d.cost_per_km2),
        rho_city_rarity=stats.spearmanr(d.recent, d.rarity_per_km2),
        countries=len(c), rho_country_rarity=stats.spearmanr(c.recent, c.rpk),
    )
    return out


def panel_binned(ax, d, col, scale, ylabel, color, marker, ylim):
    xm, lo, md, hi = binned(d.recent.to_numpy(), d[col].to_numpy() * scale)
    ax.vlines(xm, lo, hi, color=color, lw=1.2, alpha=0.55, zorder=2)
    ax.plot(xm, md, "-", color=color, lw=1.0, zorder=3)
    ax.scatter(xm, md, s=26, marker=marker, color=color, edgecolor="white", lw=0.6, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.3f}".rstrip("0").rstrip(".")))
    ax.set_xlim(5, 90)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_xlabel("Share of 2020 surface\nadded after 2000 (%)")
    ax.set_ylabel(ylabel)
    r = stats.spearmanr(d.recent, d[col])
    p = "P < 10$^{-64}$" if r.pvalue < 1e-64 else f"P = {r.pvalue:.3f}"
    sign = "\u2212" if r.statistic < 0 else ""
    ax.text(0.98, 0.98, f"ρ = {sign}{abs(r.statistic):.2f}\n{p}", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color=st.INK)
    st.despine(ax)


TLAB = ["Below 34", "34 to 51", "Above 51"]


def panel_violin(ax, d, col, scale, ylabel, ylim):
    t = pd.qcut(d.recent, 3, labels=False).to_numpy()
    v = np.log10(d[col].to_numpy() * scale)
    grid = np.linspace(np.log10(ylim[0]), np.log10(ylim[1]), 300)
    for i in range(3):
        x = v[t == i]
        lo_, hi_ = np.percentile(x, [0.5, 99.5])
        g = grid[(grid >= lo_) & (grid <= hi_)]
        k = stats.gaussian_kde(x)(g)
        k = 0.36 * k / k.max()
        ax.fill_betweenx(10 ** g, i - k, i, color=st.FAINT, lw=0, zorder=1)
        ax.plot(np.r_[i, i - k, i], 10 ** np.r_[g[0], g, g[-1]], color=st.MUTED, lw=0.6, zorder=2)
        q5, q25, q50, q75, q95 = np.percentile(x, [5, 25, 50, 75, 95])
        ax.vlines(i + 0.05, 10 ** q5, 10 ** q95, color=st.INK, lw=0.7, zorder=3)
        ax.add_patch(plt.Rectangle((i + 0.01, 10 ** q25), 0.08, 10 ** q75 - 10 ** q25, facecolor="white",
                                   edgecolor=st.INK, lw=0.7, zorder=4))
        ax.hlines(10 ** q50, i + 0.01, i + 0.09, color=st.INK, lw=1.4, zorder=5)
    for j, r in enumerate(REALMS):
        m = d.realm.to_numpy() == r
        md = [np.median(d[col].to_numpy()[m & (t == i)]) * scale for i in range(3)]
        xs = np.arange(3) + 0.17 + 0.055 * j
        ax.plot(xs, md, color=st.CAT[j], lw=0.8, zorder=5)
        ax.scatter(xs, md, s=18, marker=st.CAT_MARKERS[j], color=st.CAT[j], edgecolor="white", lw=0.4, zorder=6)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v_, _: f"{v_:,.3f}".rstrip("0").rstrip(".")))
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(range(3), TLAB)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Share of 2020 surface\nadded after 2000 (%)")
    ax.set_ylabel(ylabel)
    r = stats.spearmanr(d.recent, d[col])
    pv = "P < 10$^{-64}$" if r.pvalue < 1e-64 else f"P = {r.pvalue:.3f}"
    sign = "\u2212" if r.statistic < 0 else ""
    ax.text(0.98, 0.02, f"ρ = {sign}{abs(r.statistic):.2f}\n{pv}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, color=st.INK)
    st.despine(ax)


def panel_hex(ax, d, col, scale, ylabel, ylim):
    x = d.recent.to_numpy()
    y = d[col].to_numpy() * scale
    ax.hexbin(x, y, yscale="log", gridsize=(22, 14), extent=(0, 100, np.log10(y.min()), np.log10(y.max())),
              cmap=HEX, mincnt=1, bins="log", lw=0.1, zorder=1)
    xm, lo, md, hi = binned(x, y)
    ax.plot(xm, md, "-", color=RED, lw=1.4, zorder=3)
    ax.scatter(xm, md, s=20, color=RED, edgecolor="white", lw=0.6, zorder=4)
    ax.plot(xm, lo, "--", color=RED, lw=0.8, zorder=3)
    ax.plot(xm, hi, "--", color=RED, lw=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of 2020 surface\nadded after 2000 (%)")
    ax.set_ylabel(ylabel)
    r = stats.spearmanr(d.recent, d[col])
    pv = "P < 10$^{-64}$" if r.pvalue < 1e-64 else f"P = {r.pvalue:.3f}"
    sign = "\u2212" if r.statistic < 0 else ""
    ax.text(0.98, 0.98, f"ρ = {sign}{abs(r.statistic):.2f}\n{pv}", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color=st.INK, bbox=None)
    st.despine(ax)


def figure(name="figure_recency_rarity_v2", style="line"):
    st.apply_rc()
    d = table()
    c = countries(d)
    W, H = tp.PAGE_W, 2.75
    ml, gap, mr, bot, top = 0.70, 0.80, 0.04, 0.62, 0.30
    pw = (W - ml - mr - 2 * gap) / 3
    ph = H - bot - top
    fig = plt.figure(figsize=(W, H))
    ws = [pw - 0.05, pw - 0.05, pw + 0.10]
    xs = [ml, ml + ws[0] + gap, ml + ws[0] + ws[1] + 2 * gap]
    axs = [fig.add_axes([xs[i] / W, bot / H, ws[i] / W, ph / H]) for i in range(3)]

    la = ("Exposure intensity\n(km² per km² added)", "Rarity-weighted intensity\n(10$^{-5}$ km$^{-2}$)")
    if style == "line":
        panel_binned(axs[0], d, "cost_per_km2", 1.0, la[0], st.INK, "o", (0.5, 500))
        panel_binned(axs[1], d, "rarity_per_km2", 1e5, la[1], RED, "o", (0.1, 100))
    elif style == "violin":
        panel_violin(axs[0], d, "cost_per_km2", 1.0, la[0], (1, 2000))
        panel_violin(axs[1], d, "rarity_per_km2", 1e5, la[1], (0.002, 5000))
    else:
        panel_hex(axs[0], d, "cost_per_km2", 1.0, la[0], (1, 2000))
        panel_hex(axs[1], d, "rarity_per_km2", 1e5, la[1], (0.01, 100000))

    ax = axs[2]
    area = c.tot.to_numpy()
    size = 8 + 140 * np.sqrt(area / area.max())
    for i, r in enumerate(REALMS):
        m = (c.realm == r).to_numpy()
        ax.scatter(c.recent[m], c.rpk[m] * 1e5, s=size[m], color=st.CAT[i], marker=st.CAT_MARKERS[i],
                   alpha=0.75, edgecolor="white", lw=0.5, zorder=3)
    other = ~c.realm.isin(REALMS).to_numpy()
    if other.any():
        ax.scatter(c.recent[other], c.rpk[other] * 1e5, s=size[other], color=st.FAINT, edgecolor="white",
                   lw=0.5, zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(5, 88)
    ax.set_ylim(0.001, 3000)
    ax.set_xticks([20, 40, 60])
    if style == "hex":
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.3f}".rstrip("0").rstrip(".")))
    for code, ty in LABELS:
        x0, y0 = c.recent[code], c.rpk[code] * 1e5
        ax.plot([x0, 64.5], [y0, ty], color=st.MUTED, lw=0.5, zorder=1)
        ax.text(65.2, ty, NAMES[code], fontsize=8, color=st.INK, va="center", ha="left", zorder=4)
    rho = stats.spearmanr(c.recent, c.rpk)
    ax.text(0.98, 0.98, "ρ = \u2212%.2f\nP = %.1f × 10$^{-5}$" % (abs(rho.statistic), rho.pvalue * 1e5),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=st.INK)
    ax.set_xlabel("Share of 2020 surface\nadded after 2000 (%)")
    ax.set_ylabel("Rarity-weighted intensity\n(10$^{-5}$ km$^{-2}$)")
    st.despine(ax)
    hs = [Line2D([], [], marker=st.CAT_MARKERS[i], ls="", color=st.CAT[i], ms=4.5) for i in range(4)]
    ax.legend(hs, REALMS, loc="lower left", ncol=1, fontsize=7, handletextpad=0.1, borderaxespad=0.1,
              labelspacing=0.15)
    for a, l in zip(axs, "abc"):
        a.text(-0.30, 1.04, l, transform=a.transAxes, fontsize=14, fontweight="bold", va="bottom",
               ha="left", color=st.INK)
    out = tp.save(fig, name)
    return out, summary(d, c)


if __name__ == "__main__":
    import sys
    style = sys.argv[1] if len(sys.argv) > 1 else "line"
    nm = dict(line="figure_recency_rarity_v2", violin="figure_recency_rarity_v3", hex="figure_recency_rarity_v5")[style]
    p, s = figure(nm, style)
    print(p)
    for k, v in s.items():
        print(k, v)
