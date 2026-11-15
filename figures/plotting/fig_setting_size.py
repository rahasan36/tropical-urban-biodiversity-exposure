"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, spearmanr
import tropics_style as st
import tropics_panels as tp
import tropics_main_figures as f4
import fig_distributions as f3

PT = 7.0
SHORT = [["<1", "1–5", "5–25", "25–100", "≥100"], None, None]
CLASSES = ["Urban extent in 2000 (km²)", "Urban growth, 2000 to 2020 (%)", "Threatened species within 50 km"]
LEFT, RIGHT = 0.02, 6.96
PW, WD, WE, WF = 1.05, 0.86, 1.36, 1.96
XLIM = (1.9, 80)
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]
REALMS = ["Indomalayan", "Neotropic", "Afrotropic", "Australasia"]
YLIM = (2.3, 85)
OTHER_C = "#A9A9A9"
SHAP = {"country": "Country", "setting": "Threatened species\nwithin 50 km", "realm": "Realm", "biome": "Biome",
        "form": "Expansion form", "size": "Urban extent\nin 2000"}
NAMES = {"MYS": ("Malaysia", -8, -11, "right"), "IDN": ("Indonesia", 8, -9, "left"), "IND": ("India", -6, 8, "right"),
         "BRA": ("Brazil", -20, 11, "right"), "MEX": ("Mexico", 2, -10, "left"), "NGA": ("Nigeria", -15, 9, "right"),
         "YEM": ("Yemen", 6, -6, "left")}


def letter(fig, x, y, ch):
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, ch, fontsize=PT + 2, fontweight="bold", ha="left", va="bottom", color=st.INK)


def ylabel(fig, ax, x, text):
    W, H = fig.get_size_inches()
    p = ax.get_position()
    return fig.text(x / W, p.y0 + p.height / 2, text, fontsize=PT, rotation=90, ha="center", va="center",
                    color=st.INK)


def short_labels(labs):
    out = []
    for l in labs:
        l = l.replace(" species", "").replace("%", "").replace(" to ", "–")
        out.append(l.replace("Under ", "<").replace("Over ", ">"))
    return out


def raincloud(ax, d, key, labs, col, ref):
    rng = np.random.default_rng(7)
    grid = np.linspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 300)
    vals = [np.log10(d.loc[d[key] == row, "cost_per_km2"].to_numpy()) for row in range(5)]
    left = 10 ** min(np.percentile(v, 0.5) for v in vals) / 1.10
    for row, v in enumerate(vals):
        a, b = np.percentile(v, [0.5, 99.5])
        g = grid[(grid >= a) & (grid <= b)]
        kd = gaussian_kde(v)(g)
        kd = 0.44 * kd / kd.max()
        ax.fill_between(10 ** g, row - kd, row, color=col, alpha=0.35, lw=0, zorder=2)
        ax.plot(10 ** np.r_[g[0], g, g[-1]], np.r_[row, row - kd, row], color=col, lw=0.6, zorder=2)
        ax.scatter(10 ** v, row + rng.uniform(0.10, 0.42, v.size), s=0.7, color=col, alpha=0.25, lw=0, zorder=1,
                   rasterized=True)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        p5, p95 = np.percentile(v, [5, 95])
        ax.add_patch(Rectangle((10 ** q1, row - 0.06), 10 ** q3 - 10 ** q1, 0.12, facecolor="white",
                               edgecolor=st.INK, lw=0.6, zorder=4))
        ax.plot([10 ** q2] * 2, [row - 0.06, row + 0.06], color=st.INK, lw=1.0, zorder=5)
        ax.plot([10 ** p5, 10 ** q1], [row, row], color=st.INK, lw=0.6, zorder=3)
        ax.plot([10 ** q3, 10 ** p95], [row, row], color=st.INK, lw=0.6, zorder=3)
        ax.plot(10 ** v.mean(), row, "o", ms=3.6, color=f3.RED, mec="white", mew=0.5, zorder=6)
        ax.text(10 ** b * 1.07, row - 0.17, f"{10 ** v.mean():.1f}", ha="left", va="center", fontsize=PT,
                color=st.INK, clip_on=False)
        ax.text(left, row - 0.17, labs[row], ha="right", va="center", fontsize=PT, color=st.MUTED, clip_on=False)
    ax.axvline(ref, color=st.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_ylim(4.55, -0.62)
    ax.tick_params(axis="x", labelsize=PT)
    ax.set_xlabel("Exposure intensity\n(km² km$^{-2}$)", fontsize=PT, linespacing=1.1)
    st.despine(ax, keep=("bottom",))


def panel_shapley(ax):
    sh = f4._plan()["shapley"]["shapley"]
    items = sorted(sh.items(), key=lambda kv: -kv[1])
    for i, (k, v) in enumerate(items):
        col = {"size": f3.SIZE_C, "setting": f3.SET_C}.get(k, OTHER_C)
        ax.barh(i, max(100 * v, 0.35), color=col, height=0.62, edgecolor="none")
        ax.text(-1.0, i, SHAP[k], va="center", ha="right", fontsize=PT, color=st.INK, linespacing=1.0,
                clip_on=False)
        ax.text(100 * v + 0.9, i, f"{100 * v:.1f}", va="center", ha="left", fontsize=PT, color=st.INK)
    ax.set_xlim(0, 43)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([])
    ax.set_ylim(len(items) - 0.45, -0.62)
    ax.set_xlabel("Variance explained (%)", fontsize=PT, x=1.0, ha="right")
    ax.tick_params(labelsize=PT)
    st.despine(ax, keep=("bottom",))


def panel_realm(ax, d):
    rng = np.random.default_rng(3)
    med = d[d.realm.isin(REALMS)].groupby("realm").cost_per_km2.median().sort_values(ascending=False)
    for i, r in enumerate(med.index):
        col = st.CAT[REALMS.index(r)]
        v = np.log10(d.loc[d.realm == r, "cost_per_km2"].to_numpy())
        a, b = np.percentile(v, [0.5, 99.5])
        yy = np.linspace(a, b, 200)
        w = gaussian_kde(v)(yy)
        w = 0.40 * w / w.max()
        ax.fill_betweenx(10 ** yy, i + 0.07, i + 0.07 + w, color=col, alpha=0.55, lw=0, zorder=2)
        ax.scatter(i - 0.22 + rng.uniform(-0.11, 0.11, v.size), 10 ** v, s=0.9, color=col, alpha=0.30, lw=0,
                   rasterized=True, zorder=1)
        q1, q2, q3 = 10 ** np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((i - 0.055, q1), 0.11, q3 - q1, facecolor="white", edgecolor=st.INK, lw=0.6,
                               zorder=3))
        ax.plot([i - 0.055, i + 0.055], [q2, q2], color=st.INK, lw=1.1, zorder=4)
        ax.text(i + 0.05, 70, f"{q2:.1f}", fontsize=PT, ha="center", va="bottom", color=st.INK)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_yticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    ax.set_yticks([], minor=True)
    ax.set_xlim(-0.45, len(med) - 0.42)
    ax.set_xticks(range(len(med)), med.index, rotation=25, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, labelsize=PT, pad=3)
    ax.tick_params(axis="y", labelsize=PT)
    st.despine(ax)


def panel_country(ax, d):
    c = d.groupby("country").agg(n=("city_id", "size"), y=("cost_per_km2", "median"),
                                 x=("setting_footprint", "median"), tot=("urban_km2_final", "sum"),
                                 realm=("realm", lambda v: v.value_counts().index[0]))
    c = c[c.n >= 10]
    area = lambda km2: 5 + 95 * np.sqrt(km2 / c.tot.max())
    size = area(c.tot)
    for r in c.realm.unique():
        m = c.realm == r
        k = REALMS.index(r) if r in REALMS else None
        ax.scatter(c.x[m], c.y[m], s=size[m], color=st.FAINT if k is None else st.CAT[k],
                   marker="o" if k is None else st.CAT_MARKERS[k], alpha=0.78, edgecolor="white", lw=0.5,
                   zorder=2 if k is None else 3)
    for code, (nm, dx, dy, ha) in NAMES.items():
        ax.annotate(nm, xy=(c.x[code], c.y[code]), xytext=(dx, dy), textcoords="offset points", fontsize=PT,
                    ha=ha, va="center", color=st.INK, zorder=5, path_effects=HALO,
                    arrowprops=dict(arrowstyle="-", color=st.MUTED, lw=0.5, shrinkA=0, shrinkB=2))
    rho = spearmanr(c.x, c.y)
    e = int(np.floor(np.log10(rho.pvalue)))
    ax.text(0.03, 1.0, f"Spearman $\\rho$ = {rho.statistic:.2f}, $P$ = {rho.pvalue / 10 ** e:.1f} × 10$^{{{e}}}$".replace(
        "-", "−"), transform=ax.transAxes, ha="left", va="top", fontsize=PT, color=st.INK)
    ax.text(0.03, 0.895, "Urban area in 2020 (km²)", transform=ax.transAxes, ha="left", va="top", fontsize=PT,
            color=st.INK)
    for xa, km2 in ((0.06, 1000), (0.31, 10000)):
        ax.scatter([xa], [0.785], s=area(km2), color=OTHER_C, edgecolor="white", lw=0.5, transform=ax.transAxes,
                   zorder=3)
        ax.text(xa + 0.04, 0.785, f"{km2:,}", transform=ax.transAxes, ha="left", va="center", fontsize=PT,
                color=st.INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(3.6, 62)
    ax.set_xlim(4.5, 125)
    ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    ax.set_xticks([], minor=True)
    ax.set_yticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    ax.set_yticks([], minor=True)
    ax.tick_params(labelsize=PT)
    ax.set_xlabel("Median threatened species within 50 km", fontsize=PT)
    st.despine(ax)
    hs = [plt.Line2D([], [], marker=st.CAT_MARKERS[i], ls="", color=st.CAT[i], ms=4.2, mec="white", mew=0.4)
          for i in range(4)]
    ax.legend(hs, REALMS, loc="lower right", fontsize=PT, handletextpad=0.1, borderaxespad=0.0, borderpad=0.1,
              labelspacing=0.2, frameon=False)


def panel_exponent(ax):
    rv = {r["label"]: r for r in f4._plan()["rarity_variants"]}
    s0 = json.loads((f4.TABLE_DIR / "scaling_spatial.json").read_text())["cost_per_km2"]
    sets = [("Species of concern", dict(beta=s0["beta"], ci=[s0["ci_lower"], s0["ci_upper"]], n=s0["n"]),
             rv["Species of concern, rarity weighted"]),
            ("Threatened", rv["Threatened, unweighted"], rv["Threatened, rarity weighted"]),
            ("Narrow range", rv["Narrow range, unweighted"], rv["Narrow range, rarity weighted"])]
    xs = np.linspace(-0.32, 0.18, 1500)
    ax.axvline(0, color=st.FAINT, lw=0.7, zorder=0)
    for i, (lab, u, r) in enumerate(sets):
        y = -1.5 * i
        light, dark = "#C9D1DD", f3.SIZE_C
        ax.plot([xs[0], xs[-1]], [y, y], color=light, lw=0.6, zorder=1)
        for e, c in ((u, light), (r, dark)):
            se = (e["ci"][1] - e["ci"][0]) / (2 * 1.959964)
            dens = 0.85 * np.exp(-0.5 * ((xs - e["beta"]) / se) ** 2)
            ax.fill_between(xs, y, y + dens, color=c, alpha=0.50, lw=0, zorder=2)
            ax.plot(xs, y + dens, color=dark, lw=0.6, zorder=3)
            ax.plot(e["ci"], [y, y], color=dark, lw=1.6, solid_capstyle="butt", zorder=4)
            ax.plot(e["beta"], y, "o", ms=4.0, mfc=c, mec=dark, mew=0.8, zorder=5)
        ax.annotate("", xy=(r["beta"], y + 0.36), xytext=(u["beta"], y + 0.36), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=st.INK, lw=0.7, mutation_scale=6, shrinkA=0,
                                    shrinkB=0))
        ax.text(-0.318, y + 0.98, f"{lab}, $n$ = {u['n']:,}", fontsize=PT, ha="left", va="bottom", color=st.INK)
    top = sets[0]
    ax.text(top[1]["beta"] - 0.006, 0.62, "Unweighted", fontsize=PT, ha="right", va="center", color=st.MUTED)
    ax.text(top[2]["beta"], 0.92, "Rarity weighted", fontsize=PT, ha="center", va="bottom", color=st.INK)
    ax.set_xlim(-0.32, 0.18)
    ax.set_ylim(-3.25, 1.35)
    ax.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1])
    ax.set_yticks([])
    ax.tick_params(labelsize=PT)
    ax.set_xlabel("Size exponent $\\beta$ with 95% interval", fontsize=PT)
    st.despine(ax, keep=("bottom",))


def top_row(fig, d, groups, y1, ph, xs):
    W, H = fig.get_size_inches()
    ref = 10 ** np.log10(d.cost_per_km2).mean()
    panels = []
    for i, ((ttl, key, xcol, labs, col), ch) in enumerate(zip(groups, "abc")):
        ax = fig.add_axes([xs[i] / W, y1 / H, PW / W, ph / H])
        raincloud(ax, d, key, SHORT[i] or short_labels(labs), col, ref)
        panels.append((ax, [ylabel(fig, ax, xs[i] - 0.33, CLASSES[i])]))
        letter(fig, xs[i] - 0.20, y1 + ph + 0.05, ch)
    ax = fig.add_axes([xs[3] / W, y1 / H, WD / W, ph / H])
    panel_shapley(ax)
    panels.append((ax, []))
    letter(fig, xs[3] - 0.22, y1 + ph + 0.05, "d")
    hs = [Line2D([], [], marker="o", ls="", color=f3.RED, mec="white", mew=0.5, ms=4.2),
          Patch(facecolor="white", edgecolor=st.INK, lw=0.6), Line2D([], [], color=st.INK, lw=0.6),
          Line2D([], [], color=st.MUTED, lw=0.7, ls=(0, (3, 2)))]
    fig.legend(hs, ["Geometric mean, with its value", "Median and quartiles", "5th to 95th percentile",
                    "Mean of all agglomerations"], ncol=4, fontsize=PT, loc="center",
               bbox_to_anchor=(0.5, (y1 - 0.72) / H), handlelength=1.3, handleheight=0.7, columnspacing=1.1,
               handletextpad=0.4, frameon=False, borderpad=0)
    return panels


def bottom_row(fig, d, bot, ph, xs, wg):
    W, H = fig.get_size_inches()
    ax = fig.add_axes([xs[0] / W, bot / H, WE / W, ph / H])
    panel_realm(ax, d)
    panels = [(ax, [ylabel(fig, ax, xs[0] - 0.39, "Exposure intensity (km² km$^{-2}$)")])]
    letter(fig, xs[0] - 0.22, bot + ph + 0.05, "e")
    ax = fig.add_axes([xs[1] / W, bot / H, WF / W, ph / H])
    panel_country(ax, d)
    panels.append((ax, [ylabel(fig, ax, xs[1] - 0.36, "Country median intensity (km² km$^{-2}$)")]))
    letter(fig, xs[1] - 0.22, bot + ph + 0.05, "f")
    ax = fig.add_axes([xs[2] / W, bot / H, wg / W, ph / H])
    panel_exponent(ax)
    panels.append((ax, []))
    letter(fig, xs[2], bot + ph + 0.05, "g")
    return panels


def overhang(fig, panels):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax, texts in panels:
        boxes = [ax.get_tightbbox(r)] + [t.get_window_extent(r) for t in texts]
        p = ax.get_window_extent(r)
        out.append(((p.x0 - min(b.x0 for b in boxes)) / fig.dpi, (max(b.x1 for b in boxes) - p.x1) / fig.dpi))
    out[-1] = (out[-1][0], 0.0)
    return out


def place(widths, hang, gap):
    xs, x = [], LEFT
    for w, (l, r) in zip(widths, hang):
        xs.append(x + l)
        x += l + w + r + gap
    return xs


def compose(d, groups, xt, xb, wg):
    ph, bot = 1.92, 0.66
    y1 = bot + ph + 0.26 + 0.26 + 0.56
    fig = plt.figure(figsize=(tp.PAGE_W, y1 + ph + 0.26))
    return fig, top_row(fig, d, groups, y1, ph, xt), bottom_row(fig, d, bot, ph, xb, wg)


def figure(name="figure2_setting_size_v7"):
    st.apply_rc(min_pt=7)
    d, groups = f3.data()
    span, wt = RIGHT - LEFT, [PW, PW, PW, WD]
    xt, xb, wg = [0.42, 2.14, 3.86, 6.10], [0.48, 2.44, 4.74], 2.22
    for _ in range(2):
        fig, top, bottom = compose(d, groups, xt, xb, wg)
        ht, hb = overhang(fig, top), overhang(fig, bottom)
        plt.close(fig)
        gap = (span - sum(w + l + r for w, (l, r) in zip(wt, ht))) / 3
        wg = span - sum(w + l + r for w, (l, r) in zip([WE, WF], hb)) - hb[2][0] - 2 * gap
        xt, xb = place(wt, ht, gap), place([WE, WF, wg], hb, gap)
    fig, top, bottom = compose(d, groups, xt, xb, wg)
    return tp.save(fig, name)


if __name__ == "__main__":
    import sys
    for fn in (sys.argv[1:] or ["figure"]):
        print(globals()[fn]())
