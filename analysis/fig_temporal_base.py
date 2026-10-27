"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tropics_style as st
import tropics_panels as tp
from tropics_config import TABLE_DIR

T = TABLE_DIR / "temporal"
LAB = ["2000–\n2005", "2005–\n2010", "2010–\n2015", "2015–\n2020"]
PER = ["2000–2005", "2005–2010", "2010–2015", "2015–2020"]
REALMS = ["Indomalayan", "Neotropic", "Afrotropic", "Australasia"]
SLATE = "#5F7391"
GREENS = ["#A8CCB5", "#7DB296", "#52947C", "#2E6F5B", "#1C4A3C"]
SLATES = ["#C9CED8", "#A3ADBD", "#7D8BA2", "#5F7391", "#3B4A61"]


def title(ax, letter, text, x=-0.27, dx=0.075):
    ax.text(x, 1.05, letter, transform=ax.transAxes, fontsize=st.MIN_PT + 1,
            fontweight="bold", ha="left", va="bottom", color=st.INK)
    ax.text(x + dx, 1.05, text, transform=ax.transAxes, fontsize=st.MIN_PT,
            fontweight="bold", ha="left", va="bottom", color=st.INK)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_land(ax):
    r = pd.read_csv(T / "temporal_realms.csv")
    r["grp"] = np.where(r.realm.isin(REALMS), r.realm, "Other")
    p = r.pivot_table(index="period", columns="grp", values="added_km2", aggfunc="sum").loc[PER]
    x = np.arange(4); bot = np.zeros(4)
    cols = dict(zip(REALMS, st.CAT)); cols["Other"] = st.FAINT
    for g in REALMS + ["Other"]:
        v = p[g].to_numpy() / 1000
        ax.bar(x, v, 0.62, bottom=bot, color=cols[g], edgecolor="white", linewidth=0.4, label=g)
        bot += v
    for i, v in enumerate(bot):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=st.MIN_PT, color=st.INK)
    ax.set_xticks(x, LAB); ax.set_ylim(0, 22); ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("Urban land added (thousand km²)")
    clean(ax)
    return [plt.Rectangle((0, 0), 1, 1, color=cols[g]) for g in REALMS + ["Other"]], REALMS + ["Other realms"]


def panel_aggl(ax):
    inv = pd.read_csv(T / "temporal_agglomerations.csv")
    first = int(inv.later_2020_agglomerations.iloc[0])
    lin = inv.later_2020_agglomerations.to_numpy()
    x = np.arange(5)
    ax.bar(x, np.full(5, first) / 1000, 0.62, color=SLATE, edgecolor="white", linewidth=0.4)
    ax.bar(x, (lin - first) / 1000, 0.62, bottom=first / 1000, color=st.ORANGE,
           edgecolor="white", linewidth=0.4)
    for i, v in enumerate(lin):
        ax.text(i, v / 1000 + 0.12, f"{v:,}", ha="center", va="bottom", fontsize=st.MIN_PT)
    ax.set_xticks(x, [str(y) for y in inv.year]); ax.set_ylim(0, 8.6); ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_ylabel("Agglomerations (thousand)")
    clean(ax)
    return ([plt.Rectangle((0, 0), 1, 1, color=SLATE), plt.Rectangle((0, 0), 1, 1, color=st.ORANGE)],
            ["Present by 2000", "Emerged after 2000"])


def panel_reach(ax):
    p = pd.read_csv(T / "temporal_periods.csv")
    x = np.arange(4); w = 0.34
    th, na = p.threat_reached.to_numpy(), p.narrow_reached.to_numpy()
    ax.bar(x - w / 2, th, w, color=st.CRIMSON, edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, na, w, color=GREENS[3], edgecolor="white", linewidth=0.4)
    tc, nc = int(p.threat_cum.iloc[-1]), int(p.narrow_cum.iloc[-1])
    ax.axhline(tc, color=st.INK, lw=0.7, ls=(0, (2, 2)))
    ax.text(3.45, 3330, f"Reached over 2000–2020:\n{tc:,} threatened\n{nc:,} narrow-ranged",
            ha="right", va="top", fontsize=st.MIN_PT, color=st.INK, linespacing=1.15)
    ax.set_xticks(x, LAB); ax.set_ylim(0, 3400)
    ax.set_ylabel("Species reached in the period")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    clean(ax)
    return ([plt.Rectangle((0, 0), 1, 1, color=st.CRIMSON), plt.Rectangle((0, 0), 1, 1, color=GREENS[3])],
            ["Threatened", "Narrow-ranged"])


def _lines(ax, cl, by, order, colors, labels):
    x = np.arange(4)
    for k, c in zip(order, colors):
        v = cl[(cl.classed_by == by) & (cl["class"].astype(str) == str(k))].set_index("period").loc[PER]
        ax.plot(x, v.intensity, "-o", color=c, ms=3.6, lw=1.2, mec="white", mew=0.4)
    p = pd.read_csv(T / "temporal_periods.csv")
    ax.plot(x, p.intensity_concern, "--", color=st.INK, lw=0.9)
    ax.set_xticks(x, LAB); ax.set_xlim(-0.3, 3.3)
    ax.set_ylabel("Range converted per km² added (km²)")
    clean(ax)


def panel_setting(ax, cl):
    _lines(ax, cl, "setting_c", [0, 1, 2, 3, 4], GREENS, None)
    ax.set_ylim(0, 40)
    h = [Line2D([], [], color=c, marker="o", ms=3.6, lw=1.2, mec="white", mew=0.4) for c in GREENS]
    h.append(Line2D([], [], color=st.INK, lw=0.9, ls="--"))
    return h, ["Lowest", "Second", "Third", "Fourth", "Highest", "All growth"]


def panel_size(ax, cl):
    order = ["Not yet an agglomeration", "5 to 25 km²", "25 to 100 km²",
             "100 to 500 km²", "500 km² or more"]
    cols = ["#E0A36E"] + SLATES[1:]
    _lines(ax, cl, "size_c", order, cols, None)
    ax.set_ylim(0, 40)
    h = [Line2D([], [], color=c, marker="o", ms=3.6, lw=1.2, mec="white", mew=0.4) for c in cols]
    return h, ["Not yet formed", "5–25 km²", "25–100 km²", "100–500 km²", "≥500 km²"]


def panel_beta(ax):
    fits = [json.loads((T / f"temporal_fit_{j}.json").read_text()) for j in range(5)]
    x = np.arange(5)
    for off, key, c, lab in ((-0.12, "cost_per_km2", SLATE, "Unweighted"),
                             (0.12, "rarity_per_km2", st.CRIMSON, "Rarity weighted")):
        b = np.array([f[key]["beta"] for f in fits])
        lo = np.array([f[key]["ci_lower"] for f in fits]); hi = np.array([f[key]["ci_upper"] for f in fits])
        ax.errorbar(x + off, b, yerr=[b - lo, hi - b], fmt="o", color=c, ms=3.8, lw=1.0,
                    capsize=0, mec="white", mew=0.4)
    ax.axhline(0, color=st.MUTED, lw=0.6)
    ax.axvline(3.5, color=st.HAIR, lw=0.8)
    ax.set_xticks(x, LAB + ["2000–\n2020"]); ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.04, 0.22)
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_ylabel("Size exponent of exposure intensity")
    clean(ax)
    h = [Line2D([], [], color=SLATE, marker="o", ls="", ms=3.8),
         Line2D([], [], color=st.CRIMSON, marker="o", ls="", ms=3.8)]
    return h, ["Unweighted", "Rarity weighted"]


def figure(name="figure_temporal_v1"):
    st.apply_rc()
    cl = pd.read_csv(T / "temporal_classes.csv")
    W, H = tp.PAGE_W, 6.3
    fig = plt.figure(figsize=(W, H))
    L, R, B, Tp = 0.090, 0.985, 0.085, 0.955
    hg, vg = 0.098, 0.225
    cw = (R - L - 2 * hg) / 3
    ch = (Tp - B - vg) / 2
    ax = [[fig.add_axes([L + j * (cw + hg), B + (1 - i) * (ch + vg), cw, ch]) for j in range(3)]
          for i in range(2)]
    key_y = B + ch + vg - 0.080
    def key(a, hl, ncol, y=key_y, dy=0.0):
        h, l = hl
        bb = a.get_position()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(bb.x0 - 0.05, y + dy), ncol=ncol,
                   frameon=False, handlelength=1.1, handletextpad=0.4, columnspacing=0.8,
                   borderaxespad=0, fontsize=st.MIN_PT)
    key(ax[0][0], panel_land(ax[0][0]), 2)
    title(ax[0][0], "a", "Urban land added by period")
    key(ax[0][1], panel_aggl(ax[0][1]), 1)
    title(ax[0][1], "b", "Agglomerations by first appearance")
    key(ax[0][2], panel_reach(ax[0][2]), 2)
    title(ax[0][2], "c", "Species reached in each period")
    hs = panel_setting(ax[1][0], cl)
    ax[1][0].legend(hs[0], hs[1], loc="upper left", ncol=2, frameon=False, handlelength=1.3,
                    handletextpad=0.4, columnspacing=0.8, fontsize=st.MIN_PT,
                    title="Threatened species within 50 km", title_fontsize=st.MIN_PT,
                    alignment="left", borderaxespad=0.2)
    title(ax[1][0], "d", "Exposure intensity by setting")
    hz = panel_size(ax[1][1], cl)
    ax[1][1].legend(hz[0], hz[1], loc="lower left", ncol=2, frameon=False, handlelength=1.1,
                    handletextpad=0.35, columnspacing=0.6, fontsize=st.MIN_PT, title="Agglomeration area at period start",
                    title_fontsize=st.MIN_PT, alignment="left", borderaxespad=0.2)
    title(ax[1][1], "e", "Exposure intensity by size")
    hb = panel_beta(ax[1][2])
    ax[1][2].legend(hb[0], hb[1], loc="upper left", frameon=False, handletextpad=0.3,
                    fontsize=st.MIN_PT, borderaxespad=0.2)
    title(ax[1][2], "f", "Size dependence by period")
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
