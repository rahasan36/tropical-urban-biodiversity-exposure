"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tropics_style as st
import tropics_panels as tp
import tropics_main_figures as f4

LEG_PT = 7.0


def legend_one_col(ax, loc, anchor=None):
    kw = dict(fontsize=LEG_PT, loc=loc, ncol=1, handlelength=0.9, handletextpad=0.4,
              labelspacing=0.25, frameon=False, borderaxespad=0.3)
    if anchor is not None:
        kw["bbox_to_anchor"] = anchor
    return ax.legend(**kw)


def legend_at_left(ax, x0=-1.0):
    lo, hi = ax.get_xlim()
    lg = ax.legend(fontsize=LEG_PT, loc="upper left", ncol=1, handlelength=0.9, handletextpad=0.4,
                   labelspacing=0.25, frameon=False, borderaxespad=0, borderpad=0,
                   bbox_to_anchor=((x0 - lo) / (hi - lo), 1.0), bbox_transform=ax.transAxes)
    return lg


def place_share_labels(ax, fig, pts, frac, cumc, cuma, rows=(51, 58.3, 65.6, 72.9, 80.2), gap=2.5):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for (q, yq, k), yr in zip(sorted(pts, key=lambda t: t[0]), rows):
        txt = f"e = {yq:.1f}%, n = {k:,}"
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

def figure(name="figure4_concentration_v29"):
    st.apply_rc(min_pt=LEG_PT)
    d = tp.cities()
    fig = plt.figure(figsize=(tp.PAGE_W, 4.45))
    L, R, T, B = 0.100, 0.985, 0.950, 0.105
    hgap, vgap = 0.100, 0.160
    cw = (R - L - hgap) / 2
    ch = (T - B - vgap) / 2
    cell = lambda i, j: fig.add_axes([L + j * (cw + hgap), B + (1 - i) * (ch + vgap), cw, ch])

    ax = cell(0, 0)
    order = np.argsort(-d.urban_km2_base.values)
    cost = d.cost_km2.values[order]
    add = d.urban_added_km2.values[order]
    frac = np.arange(1, cost.size + 1) / cost.size * 100
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.plot(frac, np.cumsum(cost) / cost.sum() * 100, color=f4.NULL, lw=1.6, label="Exposure (e)")
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

    ax = cell(0, 1)
    rel = f4._release()
    al = rel["alphas"]; a0 = rel["alpha_hat"]
    ax.fill_between(al, rel["m_lo"], rel["m_hi"], color=f4.NULL, alpha=0.13, lw=0, zorder=1)
    ax.plot(al, rel["measured"], color=f4.NULL, lw=1.7, zorder=3, label="Measured intensity (m)")
    ax.fill_between(al, rel["p_lo"], rel["p_hi"], color=st.BLUE, alpha=0.11, lw=0, zorder=1)
    ax.plot(al, rel["predicted"], color=st.BLUE, lw=1.7, ls=(0, (4, 2)), zorder=3,
            label="Model-fitted intensity (p)")
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, ymax=0.70, color=f4.STAT, lw=0.8, ls=(0, (1, 2)))
    ylo = min(float(np.min(rel["m_lo"])), float(np.min(rel["p_lo"])))
    yhi = max(float(np.max(rel["m_hi"])), float(np.max(rel["p_hi"])))
    ax.set_xlim(al.min(), al.max()); ax.set_ylim(ylo - 0.10 * (yhi - ylo), yhi + 0.45 * (yhi - ylo))
    ax.text(a0 + 0.06, 0.12, f"$\\alpha$ = {a0:.2f}", transform=ax.get_xaxis_transform(),
            fontsize=st.MIN_PT, color=f4.STAT, ha="left", va="center")
    ax.text(0.985, 0.985, f"At $\\alpha$ = {al[-1]:+.0f}\np − m = {rel['d'][-1]:+.1f} pp\n"
            f"[{rel['d_lo'][-1]:+.1f}, {rel['d_hi'][-1]:+.1f}]".replace("-", "−"),
            transform=ax.transAxes, fontsize=st.MIN_PT, color=f4.STAT, ha="right", va="top", linespacing=1.15)
    ax.set_xlabel("Concentration exponent $\\alpha$")
    ax.set_ylabel(f"Change relative to\n$\\alpha$ = {a0:.2f} (%)", linespacing=1.2)
    legend_one_col(ax, "upper left")
    st.despine(ax)
    tp.head(ax, "b", "")
    ax.text(0.0, -0.195, "Dispersed", transform=ax.transAxes, fontsize=st.MIN_PT, color=st.MUTED, ha="left", va="center")
    ax.text(1.0, -0.195, "Concentrated", transform=ax.transAxes, fontsize=st.MIN_PT, color=st.MUTED, ha="right", va="center")

    rb = f4._robust()
    ax = cell(1, 0)
    for realm in f4.REALMS:
        r = rb["realm"].get(realm)
        if not r:
            continue
        ax.fill_between(al, r["ci_lo"], r["ci_hi"], color=f4.REALM_COLOR[realm], alpha=0.11, lw=0, zorder=1)
        ax.plot(al, r["pct"], lw=1.5, color=f4.REALM_COLOR[realm], zorder=3, label=f"{realm} (n = {r['n']:,})")
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, color=st.FAINT, lw=0.8, ls=(0, (1, 2)))
    lo = min(np.min(r["ci_lo"]) for r in rb["realm"].values())
    hi = max(np.max(r["ci_hi"]) for r in rb["realm"].values())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.62 * (hi - lo))
    ax.set_xlabel("Concentration exponent $\\alpha$")
    ax.set_ylabel("Change in exposure (%)")
    ax.set_xlim(al.min(), al.max())
    legend_at_left(ax)
    st.despine(ax)
    tp.head(ax, "c", "")

    ax = cell(1, 1)
    lab = {"all species of concern": "All species", "threatened": "Threatened",
           "narrow range": "Narrow range", "rarity weighted": "Rarity weighted"}
    for key, r in rb["response"].items():
        wide = key == "all species of concern"
        ax.fill_between(al, r["ci_lo"], r["ci_hi"], color=f4.RESP_COLOR[key], alpha=0.10, lw=0, zorder=1)
        ax.plot(al, r["pct"], lw=2.6 if wide else 1.5, color=f4.RESP_COLOR[key], label=lab.get(key, key),
                zorder=2 if wide else 3)
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, color=st.FAINT, lw=0.8, ls=(0, (1, 2)))
    lo = min(np.min(r["ci_lo"]) for r in rb["response"].values())
    hi = max(np.max(r["ci_hi"]) for r in rb["response"].values())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.14 * (hi - lo))
    ax.set_xlabel("Concentration exponent $\\alpha$")
    ax.set_ylabel("Change in response (%)")
    ax.set_xlim(al.min(), al.max())
    legend_at_left(ax)
    st.despine(ax)
    tp.head(ax, "d", "")
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
