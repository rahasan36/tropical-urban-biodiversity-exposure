"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tropics_style as st
import tropics_panels as tp
import tropics_temporal as t
from tropics_config import TABLE_DIR

T = TABLE_DIR / "temporal"
PER = ["2000–2005", "2005–2010", "2010–2015", "2015–2020"]
PLAB = ["2000–\n2005", "2005–\n2010", "2010–\n2015", "2015–\n2020"]
YCOL = ["#C9CED8", "#A3ADBD", "#7D8BA2", "#5F7391", "#2B3A52"]
DLAB = ["5–25 km²", "25–100 km²", "100–500 km²", "≥500 km²"]
DCOL = ["#C9CED8", "#94A1B5", "#5F7391", "#2B3A52"]
ILAB = ["Under 5 km²", "5–25 km²", "25–100 km²", "≥100 km²"]
ICOL = ["#E0A36E", "#94A1B5", "#5F7391", "#2B3A52"]


def title(ax, letter, text, x=-0.27, dx=0.075):
    ax.text(x, 1.05, letter, transform=ax.transAxes, fontsize=st.MIN_PT + 1,
            fontweight="bold", ha="left", va="bottom", color=st.INK)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def classes4():
    s = pd.read_csv(T / "temporal_size_classes.csv")
    s["c4"] = s.cls.replace({"Under 1 km²": "Under 5 km²", "1–5 km²": "Under 5 km²",
                             "≥100 km²": "≥100 km²"})
    g = s.groupby(["period", "c4"])[["added_km2", "conv", "rarity", "conv_narrow"]].sum().reset_index()
    for c in ("added_km2", "conv", "rarity", "conv_narrow"):
        g[f"sh_{c}"] = g[c] / g.groupby("period")[c].transform("sum")
    g["ratio"] = g.sh_conv / g.sh_added_km2
    g["ratio_rarity"] = g.sh_rarity / g.sh_added_km2
    return g


def panel_ranksize(ax):
    d = pd.read_csv(T / "temporal_size_distribution.csv")
    h, l = [], []
    for y, c in zip(t.YEARS, YCOL):
        with np.load(t.DELIN_DIR / f"agglomerations_{y}.npz") as q:
            a = np.sort(q["area"])[::-1]
        ax.loglog(np.arange(1, a.size + 1), a, color=c, lw=1.2)
        al = float(d.loc[d.year == y, "zipf_alpha_50"].iloc[0])
        h.append(Line2D([], [], color=c, lw=1.4)); l.append(f"{y}   {al:.2f}")
    ax.set_xlabel("Rank"); ax.set_ylabel("Agglomeration area (km²)")
    ax.set_xlim(0.8, 1e4); ax.set_ylim(1, 3e4)
    clean(ax)
    return h, l


def panel_area_share(ax):
    d = pd.read_csv(T / "temporal_size_distribution.csv")
    x = np.arange(5); bot = np.zeros(5)
    for lab, c in zip(DLAB, DCOL):
        v = 100 * d[f"area {lab}"] / d.total_km2
        ax.bar(x, v, 0.62, bottom=bot, color=c, edgecolor="white", linewidth=0.4)
        bot += v.to_numpy()
    top = 100 * d["area ≥500 km²"] / d.total_km2
    n5 = d["n ≥500 km²"]
    ax.set_xticks(x, [str(y) for y in d.year]); ax.set_ylim(0, 100)
    ax.set_ylabel("Share of agglomerated area (%)")
    clean(ax)
    labs = DLAB[:-1] + [f"≥500 km² ({n5.iloc[0]} to {n5.iloc[-1]} agglomerations)"]
    return [plt.Rectangle((0, 0), 1, 1, color=c) for c in DCOL], labs


def panel_events(ax):
    d = pd.read_csv(T / "temporal_transitions.csv")
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2, d.new, w, color=st.ORANGE, edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, d.merged, w, color=YCOL[3], edgecolor="white", linewidth=0.4)
    for i in range(4):
        ax.text(i + w / 2, d.merged.iloc[i] + 25, f"{d.pieces_absorbed.iloc[i]}", ha="center",
                va="bottom", fontsize=st.MIN_PT, color=st.INK)
    ax.text(3.45, 1590, "Numbers above bars:\nagglomerations\nabsorbed by mergers", ha="right",
            va="top", fontsize=st.MIN_PT, color=st.INK, linespacing=1.1)
    ax.set_xticks(x, PLAB); ax.set_ylim(0, 1600)
    ax.set_ylabel("Agglomerations")
    clean(ax)
    return ([plt.Rectangle((0, 0), 1, 1, color=st.ORANGE), plt.Rectangle((0, 0), 1, 1, color=YCOL[3])],
            ["Newly formed", "Formed by merger"])


def panel_land_share(ax, g):
    x = np.arange(4); bot = np.zeros(4)
    for lab, c in zip(ILAB, ICOL):
        v = 100 * g[g.c4 == lab].set_index("period").loc[PER, "sh_added_km2"].to_numpy()
        ax.bar(x, v, 0.62, bottom=bot, color=c, edgecolor="white", linewidth=0.4)
        if lab == "≥100 km²":
            for i in range(4):
                ax.text(i, bot[i] + v[i] / 2, f"{v[i]:.0f}%", ha="center", va="center",
                        fontsize=st.MIN_PT, color="white")
        bot += v
    ax.set_xticks(x, PLAB); ax.set_ylim(0, 100)
    ax.set_ylabel("Share of urban land added (%)")
    clean(ax)
    return [plt.Rectangle((0, 0), 1, 1, color=c) for c in ICOL], ILAB


def panel_gibrat(ax):
    d = pd.read_csv(T / "temporal_gibrat.csv")
    x = np.arange(4)
    ax.errorbar(x, d.gibrat_slope, yerr=[d.gibrat_slope - d.ci_lower, d.ci_upper - d.gibrat_slope],
                fmt="o", color=YCOL[3], ms=4, lw=1.0, capsize=0, mec="white", mew=0.4)
    ax.axhline(0, color=st.MUTED, lw=0.6)
    ax.text(3.35, 0.012, "Growth proportional to size", ha="right", va="bottom",
            fontsize=st.MIN_PT, color=st.MUTED)
    ax.set_xticks(x, PLAB); ax.set_xlim(-0.4, 3.4); ax.set_ylim(-0.32, 0.08)
    ax.set_ylabel("Slope of growth rate on size")
    clean(ax)


def panel_ratio(ax, g):
    x = np.arange(4)
    for lab, c in zip(ILAB, ICOL):
        v = g[g.c4 == lab].set_index("period").loc[PER]
        ax.plot(x, v.ratio, "-o", color=c, lw=1.2, ms=3.6, mec="white", mew=0.4)
    ax.axhline(1, color=st.MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks(x, PLAB); ax.set_xlim(-0.3, 3.3); ax.set_ylim(0.8, 1.15)
    ax.set_ylabel("Share of range converted ÷\nshare of land added")
    clean(ax)
    return [Line2D([], [], color=c, marker="o", ms=3.6, lw=1.2, mec="white", mew=0.4) for c in ICOL], ILAB


def figure(name="figure_temporal_size_v1", land_share=True):
    st.apply_rc()
    g = classes4()
    W, H = tp.PAGE_W, 6.4
    fig = plt.figure(figsize=(W, H))
    L, R, B, Tp = 0.095, 0.985, 0.085, 0.955
    hg, vg = 0.105, 0.235
    cw = (R - L - 2 * hg) / 3
    ch = (Tp - B - vg) / 2
    ax = [[fig.add_axes([L + j * (cw + hg), B + (1 - i) * (ch + vg), cw, ch]) for j in range(3)]
          for i in range(2)]
    key_y = B + ch + vg - 0.080
    k = 1.0
    if not land_share:
        for a in ax[1]:
            a.remove()
        bw = (R - L - hg) / 2
        ax[1] = [None] + [fig.add_axes([L + j * (bw + hg), B, bw, ch]) for j in range(2)]
        k = cw / bw

    def key(a, hl, ncol, x_off=-0.06):
        bb = a.get_position()
        fig.legend(*hl, loc="upper left", bbox_to_anchor=(bb.x0 + x_off, key_y), ncol=ncol,
                   frameon=False, handlelength=1.1, handletextpad=0.4, columnspacing=0.8,
                   borderaxespad=0, fontsize=st.MIN_PT)

    hl = panel_ranksize(ax[0][0])
    ax[0][0].legend(*hl, loc="lower left", frameon=False, handlelength=1.2, fontsize=st.MIN_PT,
                    title="Year   $\\alpha$ (≥50 km²)", title_fontsize=st.MIN_PT,
                    alignment="left", borderaxespad=0.1)
    title(ax[0][0], "a", "Rank-size distribution")
    key(ax[0][1], panel_area_share(ax[0][1]), 1)
    title(ax[0][1], "b", "Agglomerated area by size class")
    key(ax[0][2], panel_events(ax[0][2]), 1)
    title(ax[0][2], "c", "Emergence and coalescence")
    if land_share:
        hl = panel_land_share(ax[1][0], g)
        bb = ax[0][0].get_position()
        fig.legend(*hl, loc="upper left", bbox_to_anchor=(bb.x0 - 0.06, key_y), ncol=2, frameon=False,
                   handlelength=1.1, handletextpad=0.4, columnspacing=0.8, fontsize=st.MIN_PT,
                   borderaxespad=0, title="Impervious area at period start (d, f)",
                   title_fontsize=st.MIN_PT, alignment="left")
        title(ax[1][0], "d", "Land added by size at period start")
    panel_gibrat(ax[1][1])
    title(ax[1][1], "e" if land_share else "d", "Growth rate and size", x=-0.27 * k, dx=0.075 * k)
    hl = panel_ratio(ax[1][2], g)
    title(ax[1][2], "f" if land_share else "e", "Exposure relative to land added", x=-0.27 * k, dx=0.075 * k)
    if not land_share:
        ax[1][2].legend(*hl, loc="lower left", ncol=2, frameon=False, handlelength=1.4, handletextpad=0.4,
                        columnspacing=0.8, fontsize=st.MIN_PT, borderaxespad=0.2,
                        title="Impervious area at period start", title_fontsize=st.MIN_PT, alignment="left")
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
