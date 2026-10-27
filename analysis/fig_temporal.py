"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import tropics_style as st
import tropics_panels as tp
import fig_temporal_base as f1
import fig_temporal_panels as f2

T = f2.T
THREAT, NARROW = f2.THREAT, f2.NARROW
LIGHT = {THREAT: "#E7A9BF", NARROW: "#A9CFC2"}
PCOL = ["#DADAEB", "#9E9AC8", "#6A51A3", "#3F007D"]
SHORT = dict(f2.SHORT, **{"Melanesia": "Melanesia", "Australia and New Zealand": "Australia–NZ",
                          "Southern Africa": "S Africa"})
YRS = [2000, 2005, 2010, 2015, 2020]


def repeat():
    r = pd.read_csv(T / "temporal_subregion_repeat.csv")
    r = r[r["sub"] != "All tropics"]
    w = r.pivot(index="sub", columns="group", values="reached").fillna(0)
    keep = w[(w.max(axis=1) >= 30)].index
    r = r[r["sub"].isin(keep)]
    order = w.loc[keep].sort_values("Narrow-ranged", ascending=False).index.tolist()
    return r, order


def panel_repeat(ax, grp, col, order, xmax):
    r, _ = repeat()
    g = r[r.group == grp].set_index("sub").reindex(order).fillna(0)
    y = np.arange(len(order))
    ax.barh(y, g.n4, 0.68, color=col, lw=0)
    ax.barh(y, g.reached - g.n4, 0.68, left=g.n4, color=LIGHT[col], lw=0)
    for i, (n, k) in enumerate(zip(g.reached, g.n4)):
        ax.text(n + xmax * 0.015, i, f"{int(k):,} of {int(n):,}", va="center", ha="left", fontsize=st.MIN_PT, color=st.INK)
    ax.set_yticks(y, [SHORT[k] for k in order]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax); ax.set_xticks([0, 500, 1000]); ax.set_xlabel("Species reached, 2000–2020")
    f2.clean(ax)


def panel_every(ax):
    p = pd.read_csv(T / "temporal_periods.csv")
    x = np.arange(4); w = 0.36
    for k, (col, c, lab) in enumerate((("threat", THREAT, "Threatened"), ("narrow", NARROW, "Narrow-ranged"))):
        v = p[f"{col}_reached"].to_numpy() / 1000
        ax.bar(x + (k - 0.5) * w, v, w, color=c, lw=0)
    for k, (col, c) in enumerate((("threat", THREAT), ("narrow", NARROW))):
        tot = p[f"{col}_cum"].iloc[-1] / 1000
        ax.axhline(tot, color=c, lw=0.7, ls=(0, (3, 2)), xmin=0.02, xmax=0.98)
    ax.text(3.45, p.threat_cum.iloc[-1] / 1000 + 0.07, "Reached at least once", ha="right", va="bottom",
            fontsize=st.MIN_PT, color=st.INK)
    ax.set_xticks(x, f2.PLAB); ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 3.0); ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("Species reached (thousand)")
    f2.clean(ax)
    return ([Rectangle((0, 0), 1, 1, color=THREAT), Rectangle((0, 0), 1, 1, color=NARROW)],
            ["Threatened", "Narrow-ranged"])


def accrue():
    c = pd.read_csv(T / "temporal_subregion_cumulative.csv")
    s = c[(c["sub"] != "All tropics") & (c.group == "Narrow-ranged") & (c.j == 3)].set_index("sub").ge1
    order = s[s >= 5].sort_values(ascending=False).index.tolist()
    return c, order


def panel_accrual_all(ax):
    c, _ = accrue()
    a = c[c["sub"] == "All tropics"]
    for grp, col in (("Narrow-ranged", NARROW), ("Threatened", THREAT)):
        v = np.concatenate([[0], a[a.group == grp].sort_values("j").ge1.to_numpy()])
        ax.plot(YRS, v, "-o", color=col, lw=1.4, ms=3.4, mec="white", mew=0.4)
        ax.text(2020.8, v[-1], f"{int(v[-1]):,}", color=col, va="center", ha="left", fontsize=st.MIN_PT)
        tx, ty = (2001.2, 470) if grp == "Narrow-ranged" else (2011.5, 105)
        ax.text(tx, ty, grp, color=col, va="center", ha="left", fontsize=st.MIN_PT)
    ax.set_xlim(1999, 2021); ax.set_xticks(YRS); ax.set_ylim(0, 600); ax.set_yticks([0, 200, 400, 600])
    ax.set_xlabel("Year"); ax.set_ylabel("Species with ≥1% of range\nconverted since 2000")
    f2.clean(ax)


def panel_accrual_sub(ax, grp, order, xmax):
    c, _ = accrue()
    g = c[(c.group == grp) & c["sub"].isin(order)].pivot(index="sub", columns="j", values="ge1").reindex(order).fillna(0)
    y = np.arange(len(order)); left = np.zeros(len(order))
    for j in range(4):
        inc = g[j].to_numpy() - (g[j - 1].to_numpy() if j else 0)
        ax.barh(y, inc, 0.68, left=left, color=PCOL[j], lw=0)
        left += inc
    for i, n in enumerate(left):
        ax.text(n + xmax * 0.015, i, f"{int(n)}", va="center", ha="left", fontsize=st.MIN_PT, color=st.INK)
    ax.set_yticks(y, [SHORT[k] for k in order]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax); ax.set_xlabel("Species with ≥1% converted")
    f2.clean(ax)
    return [Rectangle((0, 0), 1, 1, color=k) for k in PCOL], f2.PER


def figure(kind):
    st.apply_rc()
    W, H = tp.PAGE_W, 7.2
    fig = plt.figure(figsize=(W, H))
    L, R, B, Tp = 0.090, 0.985, (0.110 if kind == "repeat" else 0.075), 0.962
    hg, vg = 0.098, 0.215
    cw = (R - L - 2 * hg) / 3
    ch = (Tp - B - vg) / 2
    pos = lambda i, j: [L + j * (cw + hg), B + (1 - i) * (ch + vg), cw, ch]
    key_y = B + ch + vg - 0.075

    def key(a, hl, ncol, dx=-0.05, **kw):
        bb = a.get_position()
        fig.legend(*hl, loc="upper left", bbox_to_anchor=(bb.x0 + dx, key_y), ncol=ncol, frameon=False,
                   handlelength=1.1, handletextpad=0.4, columnspacing=0.8, borderaxespad=0, fontsize=st.MIN_PT, **kw)

    a = fig.add_axes(pos(0, 0)); key(a, f1.panel_land(a), 2); f1.title(a, "a", "Urban land added by period")
    b = fig.add_axes(pos(0, 1)); key(b, f1.panel_aggl(b), 1); f1.title(b, "b", "Agglomerations by first appearance")
    c = fig.add_axes(pos(0, 2)); hl = f2.panel_shift(c)
    key(c, hl, 2, dx=-0.06, title="Agglomeration area at period start", title_fontsize=st.MIN_PT, alignment="left")
    f1.title(c, "c", "Land added by agglomeration size")
    lab_w = 0.075
    if kind == "repeat":
        _, order = repeat()
        d = fig.add_axes([pos(1, 0)[0] + lab_w, pos(1, 0)[1], cw - lab_w + 0.02, ch])
        panel_repeat(d, "Threatened", THREAT, order, 1060)
        f1.title(d, "d", "Threatened, by sub-region", x=-0.47, dx=0.085)
        e = fig.add_axes([pos(1, 1)[0] + lab_w, pos(1, 1)[1], cw - lab_w + 0.02, ch])
        panel_repeat(e, "Narrow-ranged", NARROW, order, 1060)
        f1.title(e, "e", "Narrow-ranged, by sub-region", x=-0.47, dx=0.085)
        f = fig.add_axes([pos(1, 2)[0] + 0.02, pos(1, 2)[1], cw - 0.02, ch])
        hl = panel_every(f)
        f1.title(f, "f", "Species reached in each period", x=-0.33, dx=0.075)
        bb = f.get_position()
        fig.legend(*hl, loc="upper left", bbox_to_anchor=(bb.x0 - 0.06, B - 0.062), ncol=2, frameon=False,
                   fontsize=st.MIN_PT, handlelength=1.0, handletextpad=0.4, columnspacing=0.8, borderaxespad=0)
        bb = d.get_position()
        fig.legend([Rectangle((0, 0), 1, 1, color="#4A4A4A"), Rectangle((0, 0), 1, 1, color="#C8C8C8")],
                   ["Reached in all four periods", "Reached in fewer periods"], loc="upper left",
                   bbox_to_anchor=(bb.x0 - 0.06, B - 0.062), ncol=2, frameon=False, fontsize=st.MIN_PT,
                   handlelength=1.0, handletextpad=0.4, columnspacing=0.8, borderaxespad=0)
        name = "figure_temporal_v10_repeat"
    else:
        _, order = accrue()
        d = fig.add_axes([pos(1, 0)[0] + 0.01, pos(1, 0)[1], cw - 0.05, ch]); panel_accrual_all(d)
        f1.title(d, "d", "Species with ≥1% of range converted", x=-0.33, dx=0.095)
        e = fig.add_axes([pos(1, 1)[0] + lab_w - 0.02, pos(1, 1)[1], cw - lab_w + 0.02, ch])
        hl = panel_accrual_sub(e, "Narrow-ranged", order, 150)
        f1.title(e, "e", "Narrow-ranged, by sub-region", x=-0.47, dx=0.085)
        f = fig.add_axes([pos(1, 2)[0] + lab_w - 0.02, pos(1, 2)[1], cw - lab_w + 0.02, ch])
        panel_accrual_sub(f, "Threatened", order, 150)
        f1.title(f, "f", "Threatened, by sub-region", x=-0.47, dx=0.085)
        f.legend(*hl, loc="lower right", bbox_to_anchor=(145, len(order) - 1 + 0.34), bbox_transform=f.transData,
                 ncol=1, frameon=False, fontsize=st.MIN_PT, handlelength=1.0, handletextpad=0.4, borderaxespad=0,
                 borderpad=0, labelspacing=0.3, title="Period in which\n1% was passed", title_fontsize=st.MIN_PT,
                 alignment="left")
        name = "figure_temporal_v10_accrual"
    return tp.save(fig, name)


LANDC = "#7A7A7A"


def mpl_same(a, b):
    from matplotlib.colors import to_rgba
    return np.allclose(to_rgba(a), to_rgba(b))


def panel_accrual_trend(ax):
    c, _ = accrue()
    a = c[c["sub"] == "All tropics"]
    p = pd.read_csv(T / "temporal_periods.csv")
    land = np.concatenate([[0], p.added_km2.cumsum().to_numpy() / 1000])
    ax2 = ax.twinx()
    ax2.plot(YRS, land, color=LANDC, lw=1.1, ls=(0, (3, 2)), marker="s", ms=2.8, mec="white", mew=0.3, zorder=1)
    ax2.set_ylim(0, 85); ax2.set_yticks([0, 20, 40, 60, 80])
    ax2.set_ylabel("Urban land added since 2000\n(thousand km²)", color=LANDC)
    ax2.tick_params(axis="y", colors=LANDC)
    ax2.spines["right"].set_color(LANDC); ax2.spines["top"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    for grp, col, off, ha, va in (("Narrow-ranged", NARROW, (-3, 3), "right", "bottom"),
                                  ("Threatened", THREAT, (3, -3), "left", "top")):
        v = np.concatenate([[0], a[a.group == grp].sort_values("j").ge1.to_numpy()])
        ax.plot(YRS, v, "-o", color=col, lw=1.4, ms=3.4, mec="white", mew=0.4, zorder=3)
        for x, y, dv in zip(YRS[1:], v[1:], np.diff(v)):
            last = grp == "Threatened" and x == YRS[-1]
            ax.annotate(f"+{int(dv)}", (x, y), xytext=(0, 4) if last else off, textcoords="offset points",
                        ha="center" if last else ha, va="bottom" if last else va, fontsize=st.MIN_PT, color=col)
        tx, ty = (2001.0, 570) if grp == "Narrow-ranged" else (2011.2, 55)
        ax.text(tx, ty, f"{grp}\n({int(v[-1])})", color=col, va="center", ha="left", fontsize=st.MIN_PT, linespacing=1.05)
    ax.set_xlim(1999, 2022.2); ax.set_xticks(YRS, ["2000", "", "2010", "", "2020"]); ax.set_ylim(0, 600); ax.set_yticks([0, 200, 400, 600])
    ax2.set_xlim(1999, 2022.2)
    ax.set_xlabel("Year"); ax.set_ylabel("Species with ≥1% of range\nconverted since 2000")
    ax.spines["top"].set_visible(False)
    return float(land[-1])


def figure_full(name="figure_temporal_v15"):
    st.apply_rc()
    W, H = tp.PAGE_W, 9.235
    fig = plt.figure(figsize=(W, H))
    box = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    L, R = 0.63, 6.895
    hg = 0.686
    cw = (R - L - 2 * hg) / 3
    cx = lambda j: L + j * (cw + hg)
    r1, h1 = 6.985, 2.00
    r2, h2 = 3.55, 2.10
    r3, h3 = 0.80, 2.00
    key_y = (r1 - 0.435) / H

    def key(a, hl, ncol, dx=-0.05, **kw):
        bb = a.get_position()
        fig.legend(*hl, loc="upper left", bbox_to_anchor=(bb.x0 + dx, key_y), ncol=ncol, frameon=False,
                   handlelength=1.1, handletextpad=0.4, columnspacing=0.8, borderaxespad=0, fontsize=st.MIN_PT, **kw)

    a = box(cx(0), r1, cw, h1); key(a, f1.panel_land(a), 2); f1.title(a, "a", "Urban land added by period")
    b = box(cx(1), r1, cw, h1); key(b, f1.panel_aggl(b), 1); f1.title(b, "b", "Agglomerations by first appearance")
    c = box(cx(2), r1, cw, h1); hl = f2.panel_shift(c)
    key(c, hl, 2, dx=-0.06, title="Agglomeration area at period start", title_fontsize=st.MIN_PT, alignment="left")
    f1.title(c, "c", "Land added by agglomeration size")
    _, order = accrue()
    dw = cw - 0.30
    d = box(cx(0) + 0.07, r2, dw, h2); land = panel_accrual_trend(d)
    f1.title(d, "d", "Species with ≥1% of range converted", x=-0.33 * cw / dw, dx=0.095 * cw / dw)
    ew = cw - 0.385 + 0.14
    e = box(cx(1) + 0.385, r2, ew, h2)
    hl = panel_accrual_sub(e, "Narrow-ranged", order, 150)
    f1.title(e, "e", "Narrow-ranged, by sub-region", x=-0.47 * (cw - 0.245) / ew, dx=0.085 * (cw - 0.245) / ew)
    f = box(R - ew, r2, ew, h2)
    panel_accrual_sub(f, "Threatened", order, 150)
    f1.title(f, "f", "Threatened, by sub-region", x=-0.47 * (cw - 0.245) / ew, dx=0.085 * (cw - 0.245) / ew)
    f.legend(*hl, loc="lower right", bbox_to_anchor=(145, len(order) - 1 + 0.34), bbox_transform=f.transData,
             ncol=1, frameon=False, fontsize=st.MIN_PT, handlelength=1.0, handletextpad=0.4, borderaxespad=0,
             borderpad=0, labelspacing=0.3, title="Period in which\n1% was passed", title_fontsize=st.MIN_PT,
             alignment="left")
    gw = 2.55
    g = box(L, r3, gw, h3); rho, pv = f2.panel_sub_scatter(g)
    f1.title(g, "g", "Concentration and rarity, by sub-region", x=-0.44 / gw, dx=0.12 / gw)
    hw = 1.95
    hx = L + (R - L) / 2 + 0.55
    hh = box(hx, r3, hw, h3); f2.panel_rarity(hh)
    f1.title(hh, "h", "Rarity-weighted exposure intensity", x=-0.50 / hw, dx=0.12 / hw)
    print("land 2020", land, "rho", rho, pv)
    return tp.save(fig, name)


def figure_compact(name="figure_temporal_v18", W=6.3, H=7.2):
    st.apply_rc(min_pt=6)
    fig = plt.figure(figsize=(W, H))
    box = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])

    def ttl(x, ax, letter, text, up=0.07):
        y = ax.get_position().y1 * H + up
        fig.text(x / W, y / H, letter, fontsize=st.MIN_PT + 1, fontweight="bold", ha="left", va="bottom",
                 color=st.INK)
        fig.text((x + 0.13) / W, y / H, text, fontsize=st.MIN_PT, fontweight="bold", ha="left", va="bottom",
                 color=st.INK)

    L, R, lab = 0.64, W - 0.12, 0.62
    hg = 0.60
    cw = (R - L - 2 * hg) / 3
    cx = lambda j: L + j * (cw + hg)
    h1 = 1.36
    r1 = H - 0.36 - h1
    key_y = (r1 - 0.40) / H

    def key(a, hl, ncol, dx, where="center", **kw):
        bb = a.get_position()
        xa = {"center": bb.x0 + bb.width / 2, "right": bb.x1}[where]
        fig.legend(*hl, loc=f"upper {where}", bbox_to_anchor=(xa, key_y), ncol=ncol,
                   frameon=False, handlelength=1.0, handletextpad=0.35, columnspacing=0.7, borderaxespad=0,
                   fontsize=7, labelspacing=0.3, **kw)

    def left_of(ax):
        fig.canvas.draw()
        return ax.yaxis.label.get_window_extent().x0 / fig.dpi

    a = box(cx(0), r1, cw, h1)
    ha, la = f1.panel_land(a)
    la = ["Other" if str(t).startswith("Other") else t for t in la]
    key(a, (ha, la), 3, 0)
    a.set_ylabel("Urban land added\n(thousand km²)")
    b = box(cx(1), r1, cw, h1); key(b, f1.panel_aggl(b), 1, 0)
    c = box(cx(2), r1, cw, h1); hl = f2.panel_shift(c)
    key(c, hl, 2, 0, where="right", title="Agglomeration area at period start", title_fontsize=7,
        alignment="left")
    ttl(left_of(a), a, "a", "Urban land added by period", up=0.20)
    ttl(left_of(b), b, "b", "Agglomerations by year formed", up=0.20)
    ttl(left_of(c), c, "c", "Land added by agglomeration size", up=0.20)

    h2 = 1.60
    r2 = r1 - 1.22 - h2
    _, order = accrue()
    dx0, dw = L + 0.06, 1.50
    d = box(dx0, r2, dw, h2); land = panel_accrual_trend(d)
    for t in d.texts:
        if t.get_text().startswith("Threatened"):
            t.set_position((2013.0, 42))
    ttl(0.0, d, "d", "Species with ≥1% of range converted")
    rlab, elab, fg = 0.62, 0.80, 0.30
    ew = (R - (dx0 + dw + rlab + elab) - fg) / 2
    ex = dx0 + dw + rlab + elab
    e = box(ex, r2, ew, h2)
    hl = panel_accrual_sub(e, "Narrow-ranged", order, 150)
    ttl(ex - elab + 0.08, e, "e", "Narrow-ranged species")
    fx = ex + ew + fg
    f = box(fx, r2, ew, h2)
    panel_accrual_sub(f, "Threatened", order, 100)
    f.set_xticks([0, 50, 100])
    f.set_yticklabels([])
    ttl(fx - 0.10, f, "f", "Threatened species")
    e.set_xlabel(""); f.set_xlabel("")
    fig.text((ex + fx + ew) / 2 / W, (r2 - 0.30) / H, "Species with ≥1% of range converted", ha="center",
             va="top", fontsize=st.MIN_PT, color=st.INK)
    f.legend(*hl, loc="lower right", bbox_to_anchor=(100, len(order) - 1 + 0.40), bbox_transform=f.transData,
             ncol=1, frameon=False, fontsize=6, handlelength=0.9, handletextpad=0.35, borderaxespad=0,
             borderpad=0, labelspacing=0.2, title="Period in which\n1% was passed", title_fontsize=6,
             alignment="left")

    r3 = 0.52
    h3 = r2 - 0.63 - r3
    gw = 2.30
    g = box(L, r3, gw, h3); rho, pv = f2.panel_sub_scatter(g)
    g.set_ylabel("Change in rarity-weighted\nintensity (%)")
    for t in g.texts:
        if t.get_text().startswith("Spearman"):
            t.set_position((0.97, 0.99))
    ttl(0.0, g, "g", "Concentration and rarity, by sub-region")
    hx = L + gw + 0.95
    hw = R + 0.08 - hx - 0.70
    hh = box(hx, r3, hw, h3); f2.panel_rarity(hh, gap=16)
    ends = {ln.get_color(): ln.get_ydata()[-1] for ln in hh.get_lines() if len(ln.get_ydata()) == 4}
    for t in hh.texts:
        nm = t.get_text().split("\n")[0]
        if nm in ("Indomalayan", "All tropics"):
            col = t.get_color()
            ye = [v for k, v in ends.items() if mpl_same(k, col)]
            if ye:
                ty = t.get_position()[1]
                hh.plot([3.04, 3.13], [ye[0], ty], color=st.MUTED, lw=0.5, clip_on=False, zorder=1)
    ttl(hx - 0.62, hh, "h", "Rarity-weighted exposure intensity")
    print("land 2020", land, "rho", rho, pv, "r2", r2, "h3", h3)
    return tp.save(fig, name)

if __name__ == "__main__":
    if sys.argv[1] == "compact":
        print(figure_compact())
    else:
        print(figure_full() if sys.argv[1] == "full" else figure(sys.argv[1]))
