"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import tropics_style as st
import tropics_panels as tp
import fig_temporal_base as f1
from tropics_config import TABLE_DIR

T = TABLE_DIR / "temporal"
PER = ["2000–2005", "2005–2010", "2010–2015", "2015–2020"]
PLAB = ["2000–\n2005", "2005–\n2010", "2010–\n2015", "2015–\n2020"]
NCOL = {4: "#2B3A52", 3: "#7D8BA2", 2: "#E9A35A", 1: "#C05A0F"}
THREAT, NARROW = "#B0184E", "#2E6F5B"
SIZE_ORDER = ["Not yet an agglomeration", "5 to 25 km²", "25 to 100 km²", "100 to 500 km²", "500 km² or more"]
SIZE_LAB = ["Not yet formed", "5–25 km²", "25–100 km²", "100–500 km²", "≥500 km²"]
SIZE_COL = ["#E7C9A9", "#C9CED8", "#94A1B5", "#5F7391", "#2B3A52"]
REALMS = ["Australasia", "Neotropic", "Indomalayan", "Afrotropic"]


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_waffle(ax):
    d = pd.read_csv(T / "temporal_species_periods.csv")
    side = 50
    gap = 10
    for g, (grp, lab) in enumerate((("threat", "Threatened"), ("narrow", "Narrow-ranged"))):
        n = np.sort(d[d.group == grp].n_periods.to_numpy())[::-1]
        x0 = g * (side + gap)
        for i, v in enumerate(n):
            r, c = divmod(i, side)
            ax.add_patch(Rectangle((x0 + c, side - 1 - r), 0.86, 0.86, facecolor=NCOL[int(v)], lw=0))
        k4 = int((n == 4).sum())
        ax.text(x0 + side / 2, -3, f"{lab}\n{k4:,} of {n.size:,}", ha="center", va="top",
                fontsize=st.MIN_PT, linespacing=1.15)
    ax.set_xlim(-1, 2 * side + gap + 1); ax.set_ylim(-16, side + 1); ax.set_aspect("equal"); ax.axis("off")
    return ([Rectangle((0, 0), 1, 1, color=NCOL[k]) for k in (4, 3, 2, 1)],
            ["All four periods", "Three", "Two", "One"])


def panel_accrual(ax):
    q = pd.read_csv(T / "temporal_species_cumulative.csv")
    p = pd.read_csv(T / "temporal_periods.csv")
    yrs = np.array([2000, 2005, 2010, 2015, 2020])
    land = np.concatenate([[0], 100 * p.added_km2.cumsum() / p.added_km2.sum()])
    for grp, c in (("threat", THREAT), ("narrow", NARROW)):
        g = q[q.group == grp].sort_values("j")
        z = lambda col: np.concatenate([[0], g[col].to_numpy()])
        ax.fill_between(yrs, z("q25"), z("q75"), color=c, alpha=0.18, lw=0)
        ax.plot(yrs, z("q50"), "-o", color=c, ms=3.4, lw=1.3, mec="white", mew=0.4)
    ax.plot(yrs, land, color=st.INK, lw=1.0, ls=(0, (3, 2)))
    ax.set_xlim(1999, 2021); ax.set_ylim(0, 102); ax.set_xticks(yrs)
    ax.set_ylabel("Cumulative share of 2000–2020 total (%)")
    clean(ax)
    return ([Line2D([], [], color=THREAT, marker="o", ms=3.4, lw=1.3), Line2D([], [], color=NARROW, marker="o", ms=3.4, lw=1.3),
             Line2D([], [], color=st.INK, lw=1.0, ls=(0, (3, 2)))],
            ["Threatened", "Narrow-ranged", "Land added"])


def panel_shift(ax):
    cl = pd.read_csv(T / "temporal_classes.csv")
    cl = cl[cl.classed_by == "size_c"]
    p = pd.read_csv(T / "temporal_periods.csv")
    x = np.arange(4); bot = np.zeros(4)
    for k, col in zip(SIZE_ORDER, SIZE_COL):
        v = 100 * cl[cl["class"] == k].set_index("period").loc[PER, "share_added"].to_numpy()
        ax.bar(x, v, 0.62, bottom=bot, color=col, edgecolor="white", lw=0.4)
        if k == "500 km² or more":
            for i in range(4):
                ax.text(i, bot[i] + v[i] / 2, f"{v[i]:.0f}%", ha="center", va="center", fontsize=st.MIN_PT, color="white")
        bot += v
    for i, val in enumerate(p.intensity_concern):
        ax.text(i, 102, f"{val:.1f}", ha="center", va="bottom", fontsize=st.MIN_PT, color=st.INK)
    ax.text(-0.42, 110, "km² of range per km² added", ha="left", va="bottom", fontsize=st.MIN_PT, color=st.MUTED)
    ax.set_xticks(x, PLAB); ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.set_ylabel("Share of land added (%)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return [Rectangle((0, 0), 1, 1, color=c) for c in SIZE_COL], SIZE_LAB


def panel_rarity(ax, gap=11):
    r = pd.read_csv(T / "temporal_realms.csv")
    p = pd.read_csv(T / "temporal_periods.csv")
    x = np.arange(4)
    cols = dict(zip(["Indomalayan", "Neotropic", "Afrotropic", "Australasia"], st.CAT))
    series = [(rl, r[r.realm == rl].set_index("period").loc[PER, "rarity_per_km2"].to_numpy(), cols[rl], 1.2)
              for rl in REALMS]
    series.append(("All tropics", p.rarity_per_km2.to_numpy(), st.INK, 2.0))
    ends = []
    for nm, v, c, lw in series:
        idx = 100 * v / v[0]
        ax.plot(x, idx, "-o", color=c, lw=lw, ms=3.2, mec="white", mew=0.4, zorder=3 if nm == "All tropics" else 2)
        ends.append([idx[-1], nm, c])
    ends.sort()
    ys = [e[0] for e in ends]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    for (y0, nm, c), y in zip(ends, ys):
        ax.text(3.15, y, f"{nm}\n{y0 - 100:+.0f}%".replace("-", "\u2212"), color=c, fontsize=st.MIN_PT, va="center", ha="left",
                linespacing=1.0)
    ax.axhline(100, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(x, ["2005", "2010", "2015", "2020"]); ax.set_xlim(-0.2, 3.1); ax.set_ylim(50, 130)
    ax.set_xlabel("End of five-year period")
    ax.set_ylabel("Index, 2000–2005 = 100")
    clean(ax)

SHORT = {"South-Eastern Asia": "SE Asia", "South America": "S America", "Southern Asia": "S Asia",
         "Western Africa": "W Africa", "Central America": "C America", "Eastern Asia": "E Asia",
         "Eastern Africa": "E Africa", "Middle Africa": "Middle Africa", "Caribbean": "Caribbean",
         "Western Asia": "W Asia", "Northern Africa": "N Africa"}
MIN_ADDED = 150.0


def subregions():
    o = pd.read_csv(T / "temporal_subregions.csv", index_col=0)
    return o[o.added0 >= MIN_ADDED].copy()


def panel_sub_change(ax):
    o = subregions().sort_values("d_rint")
    y = np.arange(len(o)); h = 0.38
    ax.barh(y - h / 2, o.d_int, h, color=SIZE_COL[3], lw=0)
    ax.barh(y + h / 2, o.d_rint, h, color=THREAT, lw=0)
    ax.axvline(0, color=st.MUTED, lw=0.6)
    ax.set_yticks(y, [SHORT[k] for k in o.index]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-62, 32); ax.set_xticks([-60, -40, -20, 0, 20])
    ax.set_xlabel("Change from 2000–2005\nto 2015–2020 (%)")
    clean(ax)
    return ([Rectangle((0, 0), 1, 1, color=SIZE_COL[3]), Rectangle((0, 0), 1, 1, color=THREAT)],
            ["Exposure intensity", "Rarity-weighted intensity"])


def panel_sub_scatter(ax):
    from scipy.stats import spearmanr
    o = subregions()
    ax.axhline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.axvline(0, color=st.MUTED, lw=0.6, ls=(0, (3, 2)))
    sz = 12 + 60 * np.sqrt(o.added0 / o.added0.max())
    ax.scatter(o.d_big, o.d_rint, s=sz, color=THREAT, edgecolor="white", lw=0.5, zorder=3)
    off = {"Caribbean": (4, 5), "E Asia": (0, -9), "S Asia": (5, 0), "W Africa": (5, 0), "W Asia": (5, 0),
           "N Africa": (5, 0), "S America": (5, 2)}
    lead = {"SE Asia": (-14, -31), "C America": (-14, -39), "E Africa": (-14, -47), "Middle Africa": (-14, -55)}
    for k, r_ in o.iterrows():
        nm = SHORT[k]
        if nm in lead:
            ax.annotate(nm, (r_.d_big, r_.d_rint), xytext=lead[nm], textcoords="data", ha="left", va="center",
                        fontsize=st.MIN_PT, color=st.INK, bbox=dict(fc="white", ec="none", pad=0.6),
                        arrowprops=dict(arrowstyle="-", color=st.MUTED, lw=0.5, shrinkA=1.5, shrinkB=3))
            continue
        dx, dy = off.get(nm, (4, 0))
        ax.annotate(nm, (r_.d_big, r_.d_rint), xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"), va="center",
                    fontsize=st.MIN_PT, color=st.INK, bbox=dict(fc="white", ec="none", pad=0.6))
    rho, pv = spearmanr(o.d_big, o.d_rint)
    ax.text(0.97, 0.80, f"Spearman ρ = {rho:.2f}\nP = {pv:.3f}, n = {len(o)}".replace("-", "\u2212"),
            transform=ax.transAxes, ha="right", va="top", fontsize=st.MIN_PT, linespacing=1.15)
    ax.set_xlim(-15, 38); ax.set_ylim(-64, 35); ax.set_xticks([-10, 0, 10, 20, 30])
    ax.set_xlabel("Change in share of land added to\nagglomerations of 100 km² or more (points)")
    ax.set_ylabel("Change in rarity-weighted intensity (%)")
    clean(ax)
    return rho, pv


def figure(name="figure_temporal_v7"):
    st.apply_rc()
    W, H = tp.PAGE_W, 6.7
    fig = plt.figure(figsize=(W, H))
    L, R, B, Tp = 0.090, 0.985, 0.150, 0.958
    hg, vg = 0.098, 0.225
    cw = (R - L - 2 * hg) / 3
    ch = (Tp - B - vg) / 2
    pos = lambda i, j: [L + j * (cw + hg), B + (1 - i) * (ch + vg), cw, ch]
    key_y = B + ch + vg - 0.080

    def key(a, hl, ncol, dx=-0.05):
        bb = a.get_position()
        fig.legend(*hl, loc="upper left", bbox_to_anchor=(bb.x0 + dx, key_y), ncol=ncol, frameon=False,
                   handlelength=1.1, handletextpad=0.4, columnspacing=0.8, borderaxespad=0, fontsize=st.MIN_PT)

    a = fig.add_axes(pos(0, 0)); key(a, f1.panel_land(a), 2); f1.title(a, "a", "Urban land added by period")
    b = fig.add_axes(pos(0, 1)); key(b, f1.panel_aggl(b), 1); f1.title(b, "b", "Agglomerations by first appearance")
    c = fig.add_axes([pos(0, 2)[0] + 0.06, pos(0, 2)[1], cw - 0.06, ch])
    key(c, panel_sub_change(c), 1, dx=-0.06)
    f1.title(c, "c", "Exposure change by sub-region", x=-0.586, dx=0.088)
    d = fig.add_axes(pos(1, 0)); rho, pv = panel_sub_scatter(d)
    f1.title(d, "d", "Concentration and rarity, by sub-region")
    print("rho", rho, pv)
    e = fig.add_axes(pos(1, 1)); hl = panel_shift(e)
    bb = e.get_position()
    fig.legend(*hl, loc="upper left", bbox_to_anchor=(bb.x0 - 0.05, B - 0.075), ncol=3, frameon=False,
               fontsize=st.MIN_PT, handlelength=1.0, handletextpad=0.4, columnspacing=0.8, borderaxespad=0,
               title="Agglomeration area at period start", title_fontsize=st.MIN_PT, alignment="left")
    f1.title(e, "e", "Growth shifted to larger agglomerations")
    f = fig.add_axes([pos(1, 2)[0], pos(1, 2)[1], cw - 0.095, ch])
    panel_rarity(f)
    f1.title(f, "f", "Rarity-weighted exposure intensity", x=-0.30, dx=0.085)
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
