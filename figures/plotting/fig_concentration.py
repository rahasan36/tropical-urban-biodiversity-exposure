"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import tropics_style as st
import tropics_panels as tp
import tropics_main_figures as f4

LEG_PT = 7.0
NAMES = {"IDN": "Indonesia", "IND": "India", "BRA": "Brazil", "VNM": "Vietnam", "MEX": "Mexico", "THA": "Thailand"}


def legend_one_col(ax, loc, anchor=None):
    kw = dict(fontsize=LEG_PT, loc=loc, ncol=1, handlelength=0.9, handletextpad=0.4,
              labelspacing=0.25, frameon=False, borderaxespad=0.3)
    if anchor is not None:
        kw["bbox_to_anchor"] = anchor
    return ax.legend(**kw)


def legend_at_left(ax, x0=-1.0, pad=0.04):
    lo, hi = ax.get_xlim()
    lg = ax.legend(fontsize=LEG_PT, loc="upper left", ncol=1, handlelength=0.9, handletextpad=0.4,
                   labelspacing=0.25, frameon=False, borderaxespad=0, borderpad=0,
                   bbox_to_anchor=((x0 - lo) / (hi - lo) + pad, 1.0), bbox_transform=ax.transAxes)
    return lg


def place_share_labels(ax, fig, pts, frac, cumc, cuma, rows=(51, 58.3, 65.6, 72.9, 80.2), gap=2.5):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for (q, yq, k), yr in zip(sorted(pts, key=lambda t: t[0]), rows):
        txt = f"$e$ = {yq:.1f}%, $n$ = {k:,}"
        t = ax.text(0, 0, txt, fontsize=LEG_PT, ha="left", va="center")
        e = t.get_window_extent(rend).transformed(inv)
        h = e.y1 - e.y0
        t.remove()
        top = yr + h / 2
        x0 = max(float(np.interp(top, cumc, frac)), float(np.interp(top, cuma, frac))) + gap
        ax.annotate(txt, (q, yq), xytext=(x0, yr), textcoords="data", fontsize=LEG_PT, color=st.INK,
                    ha="left", va="center", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
                    arrowprops=dict(arrowstyle="-", color=st.MUTED, lw=0.5, shrinkA=0, shrinkB=2,
                                    relpos=(0, 0.5)))

def figure(name="figure4_concentration_v36", ncol=3):
    st.apply_rc(min_pt=LEG_PT)
    d = tp.cities()
    nrow, W = 6 // ncol, tp.PAGE_W
    lm, rm, tm, bm, hg, vg, ch = ((0.70, 0.10, 0.22, 0.47, 0.70, 0.71, 1.52) if ncol == 2 else
                                  (0.52, 0.06, 0.22, 0.47, 0.56, 0.72, 1.72))
    cw = (W - lm - rm - (ncol - 1) * hg) / ncol
    H = tm + nrow * ch + (nrow - 1) * vg + bm
    fig = plt.figure(figsize=(W, H))

    def cell(k):
        i, j = divmod(k, ncol)
        return fig.add_axes([(lm + j * (cw + hg)) / W, (bm + (nrow - 1 - i) * (ch + vg)) / H, cw / W, ch / H])

    ax = cell(0)
    order = np.argsort(-d.urban_km2_base.values)
    cost = d.cost_km2.values[order]
    add = d.urban_added_km2.values[order]
    frac = np.arange(1, cost.size + 1) / cost.size * 100
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.plot(frac, np.cumsum(cost) / cost.sum() * 100, color=f4.NULL, lw=1.6, label="Exposure ($e$)")
    ax.plot(frac, np.cumsum(add) / add.sum() * 100, color=st.BLUE, lw=1.6, ls=(0, (4, 2)), label="Urban land added")
    ax.plot([0, 100], [0, 100], color=st.FAINT, lw=0.8, zorder=0)
    cumc = np.cumsum(cost) / cost.sum() * 100
    pts = []
    for q in (1, 5, 10, 15, 35):
        k = int(round(cost.size * q / 100))
        yq = float(cumc[k - 1])
        ax.plot([q, q], [0, yq], color=st.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=2)
        ax.plot([0, q], [yq, yq], color=st.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=2)
        ax.plot([q], [yq], "o", ms=3.0, color=f4.NULL, mec="white", mew=0.4, zorder=5)
        pts.append((q, yq, k))
    legend_one_col(ax, "lower right")
    place_share_labels(ax, fig, pts, frac, cumc, np.cumsum(add) / add.sum() * 100)
    ax.set_xlabel("Cities, ranked by 2000 urban extent (%)")
    ax.set_ylabel("Cumulative share (%)")
    st.despine(ax)
    tp.head(ax, "a", "")

    ax = cell(1)
    rel = f4._release()
    al = rel["alphas"]; a0 = rel["alpha_hat"]
    ax.fill_between(al, rel["m_lo"], rel["m_hi"], color=f4.NULL, alpha=0.13, lw=0, zorder=1)
    ax.plot(al, rel["measured"], color=f4.NULL, lw=1.7, zorder=3, label="Measured intensity ($m$)")
    ax.fill_between(al, rel["p_lo"], rel["p_hi"], color=st.BLUE, alpha=0.11, lw=0, zorder=1)
    ax.plot(al, rel["predicted"], color=st.BLUE, lw=1.7, ls=(0, (4, 2)), zorder=3,
            label="Model-fitted intensity ($p$)")
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, ymax=0.70, color=f4.STAT, lw=0.8, ls=(0, (1, 2)))
    ylo = min(float(np.min(rel["m_lo"])), float(np.min(rel["p_lo"])))
    yhi = max(float(np.max(rel["m_hi"])), float(np.max(rel["p_hi"])))
    ax.set_xlim(al.min(), al.max()); ax.set_ylim(ylo - 0.10 * (yhi - ylo), yhi + 0.45 * (yhi - ylo))
    ax.text(a0 + 0.06, 0.12, f"$\\alpha$ = {a0:.2f}", transform=ax.get_xaxis_transform(),
            fontsize=st.MIN_PT, color=f4.STAT, ha="left", va="center")
    note = (f"At $\\alpha$ = {al[-1]:+.0f}", f"$p$ − $m$ = {rel['d'][-1]:+.1f} pp", f"[{rel['d_lo'][-1]:+.1f}, {rel['d_hi'][-1]:+.1f}]")
    if ncol == 2:
        ax.text(0.985, 0.985, "\n".join(note).replace("-", "−"), transform=ax.transAxes, fontsize=st.MIN_PT,
                color=f4.STAT, ha="right", va="top", linespacing=1.15)
    else:
        ax.text(0.04, 0.76, f"{note[0]}, {note[1]}\n{note[2]}".replace("-", "−"), transform=ax.transAxes,
                fontsize=st.MIN_PT, color=f4.STAT, ha="left", va="top", linespacing=1.15)
    ax.set_xlabel("Concentration exponent $\\alpha$")
    ax.set_ylabel(f"Change relative to\n$\\alpha$ = {a0:.2f} (%)", linespacing=1.2)
    legend_one_col(ax, "upper left")
    st.despine(ax)
    tp.head(ax, "b", "")
    yw = -0.195 if ncol == 2 else 0.04
    ax.text(0.0 if ncol == 2 else 0.03, yw, "Dispersed", transform=ax.transAxes, fontsize=st.MIN_PT, color=st.MUTED, ha="left", va="center")
    ax.text(1.0 if ncol == 2 else 0.98, yw, "Concentrated", transform=ax.transAxes, fontsize=st.MIN_PT, color=st.MUTED, ha="right", va="center")

    rb = f4._robust()
    ax = cell(2)
    for realm in f4.REALMS:
        r = rb["realm"].get(realm)
        if not r:
            continue
        ax.fill_between(al, r["ci_lo"], r["ci_hi"], color=f4.REALM_COLOR[realm], alpha=0.11, lw=0, zorder=1)
        ax.plot(al, r["pct"], lw=1.5, color=f4.REALM_COLOR[realm], zorder=3, label=f"{realm} ($n$ = {r['n']:,})")
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, ymax=1.0 if ncol == 2 else 0.66, color=st.FAINT, lw=0.8, ls=(0, (1, 2)))
    lo = min(np.min(r["ci_lo"]) for r in rb["realm"].values())
    hi = max(np.max(r["ci_hi"]) for r in rb["realm"].values())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.62 * (hi - lo))
    ax.set_xlabel("Concentration exponent $\\alpha$")
    ax.set_ylabel("Change in exposure (%)")
    ax.set_xlim(al.min(), al.max())
    legend_at_left(ax)
    st.despine(ax)
    tp.head(ax, "c", "")

    ax = cell(3)
    lab = {"all species of concern": "Species of concern", "threatened": "Threatened",
           "narrow range": "Narrow range", "rarity weighted": "Rarity weighted"}
    for key, r in rb["response"].items():
        wide = key == "all species of concern"
        ax.fill_between(al, r["ci_lo"], r["ci_hi"], color=f4.RESP_COLOR[key], alpha=0.10, lw=0, zorder=1)
        ax.plot(al, r["pct"], lw=2.6 if wide else 1.5,
                color=f4.RESP_LIGHT[key] if wide else f4.RESP_COLOR[key], label=lab.get(key, key),
                zorder=2 if wide else 3)
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, ymax=1.0 if ncol == 2 else 0.72, color=st.FAINT, lw=0.8, ls=(0, (1, 2)))
    lo = min(np.min(r["ci_lo"]) for r in rb["response"].values())
    hi = max(np.max(r["ci_hi"]) for r in rb["response"].values())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.14 * (hi - lo))
    ax.set_xlabel("Concentration exponent $\\alpha$")
    ax.set_ylabel("Change in response (%)")
    ax.set_xlim(al.min(), al.max())
    legend_at_left(ax)
    st.despine(ax)
    tp.head(ax, "d", "")

    ax = cell(4)
    t = pd.read_csv(tp.TAB_DIR / "cf_influence.csv")
    full, loo = t.iloc[0], t.iloc[1:-1]
    rng = np.random.default_rng(4)
    rows = [("measured_concentrated", "Measured intensity ($m$)", f4.NULL, rel["m_lo"][-1], rel["m_hi"][-1]),
            ("predicted_concentrated", "Model-fitted intensity ($p$)", st.BLUE, rel["p_lo"][-1], rel["p_hi"][-1]),
            ("paired_concentrated", "Difference ($p$ − $m$)", st.MUTED, rel["d_lo"][-1], rel["d_hi"][-1])]
    for i, (col, lab, c, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=c, lw=7, alpha=0.16, solid_capstyle="butt", zorder=1)
        ax.scatter(loo[col], i + rng.uniform(-0.10, 0.10, len(loo)), s=8, color=c, alpha=0.75, lw=0, zorder=3)
        ax.plot([full[col]] * 2, [i - 0.24, i + 0.24], color=st.INK, lw=1.0, zorder=4)
        ax.text(0.0, i - 0.36, lab, transform=ax.get_yaxis_transform(), fontsize=LEG_PT, color=st.INK, ha="left",
                va="center", zorder=5, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        for r, ha, dx in ((loo.loc[loo[col].idxmin()], "right", -1.0), (loo.loc[loo[col].idxmax()], "left", 1.0)):
            ax.text(r[col] + dx, i + 0.30, NAMES.get(r.dropped, r.dropped), fontsize=LEG_PT, color=st.INK, ha=ha,
                    va="center", zorder=5, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.axvline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.set_xlim(-12, 46)
    ax.set_ylim(2.62, -0.62)
    ax.set_yticks([])
    ax.set_xlabel("Change at $\\alpha$ = +2, one country left out (%)")
    st.despine(ax, keep=("bottom",))
    tp.head(ax, "e", "")

    ax = cell(5)
    ax.fill_between(al, rel["m_lo"], rel["m_hi"], color=f4.NULL, alpha=0.13, lw=0, zorder=1)
    ax.plot(al, rel["measured"], color=f4.NULL, lw=1.7, zorder=3, label="Within country")
    pool = json.loads((tp.TAB_DIR / "cf_pool_bootstrap.json").read_text())
    for key, lab, col, ls in (("realm", "Within realm", st.TEAL, (0, (4, 2))),
                              ("tropics", "Across the tropics", st.BLUE, (0, (1, 1.6)))):
        ax.fill_between(al, pool[key]["ci_lo"], pool[key]["ci_hi"], color=col, alpha=0.10, lw=0, zorder=1)
        ax.plot(al, pool[key]["pct"], color=col, lw=1.5, ls=ls, zorder=3, label=lab)
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, ymax=0.72, color=st.FAINT, lw=0.8, ls=(0, (1, 2)))
    ax.set_xlim(al.min(), al.max())
    ax.set_ylim(-18, 58)
    ax.set_xlabel("Concentration exponent $\\alpha$")
    ax.set_ylabel("Change in exposure (%)")
    legend_at_left(ax)
    st.despine(ax)
    tp.head(ax, "f", "")
    if ncol == 3:
        for k in (1, 2, 3, 5):
            fig.axes[k].set_xticks([-1, 0, 1, 2])
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
