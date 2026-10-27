"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import tropics_style as st
import tropics_panels as tp
import tropics_main_figures as f4

SIZE_C, GROW_C, SET_C = "#5F7391", "#C98A18", "#2E6F5B"
RED = "#C8102E"
HEX = LinearSegmentedColormap.from_list("hex", mpl.colormaps["Blues"](np.linspace(0.22, 1.0, 256)))
PLAIN = FuncFormatter(lambda v, _: f"{v:,.3f}".rstrip("0").rstrip("."))


def data():
    import tropics_refit as trf
    d = tp.cities().merge(trf.footprint_setting(), on="city_id", how="left")
    d = d[d.urban_km2_base > 0].copy()
    d["setting_footprint"] = d.setting_footprint.fillna(0.0)
    d["growth"] = 100 * d.urban_added_km2 / d.urban_km2_base
    d["sz"] = pd.cut(d.urban_km2_base, [0, 1, 5, 25, 100, np.inf], labels=False, right=False)
    d["gq"], ge = pd.qcut(d.growth, 5, labels=False, retbins=True)
    d["sq"], se = pd.qcut(d.setting_footprint, 5, labels=False, retbins=True, duplicates="drop")
    sz_lab = ["Under 1 km²", "1 to 5 km²", "5 to 25 km²", "25 to 100 km²", "100 km² or more"]
    pc = [f"{v:.0f}" for v in ge]
    g_lab = [f"Under {pc[1]}%", f"{pc[1]} to {pc[2]}%", f"{pc[2]} to {pc[3]}%", f"{pc[3]} to {pc[4]}%",
             f"Over {pc[4]}%"]
    hi = [int(v) for v in se[1:]]
    lo = [int(np.ceil(se[0]))] + [h + 1 for h in hi[:-1]]
    s_lab = [f"{a} to {b} species" for a, b in zip(lo, hi)]
    groups = [("Urban extent in 2000", "sz", "urban_km2_base", sz_lab, SIZE_C),
              ("Urban growth, 2000 to 2020", "gq", "growth", g_lab, GROW_C),
              ("Threatened species within 50 km", "sq", "setting_footprint", s_lab, SET_C)]
    return d, groups


def class_stats(d, key, xcol):
    out = []
    for k in range(5):
        v = np.log10(d.loc[d[key] == k, "cost_per_km2"].to_numpy())
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
        out.append(dict(k=k, gm=10 ** m, lo=10 ** (m - 1.96 * se), hi=10 ** (m + 1.96 * se), n=v.size,
                        x=float(np.median(d.loc[d[key] == k, xcol]))))
    return pd.DataFrame(out)


def title(fig, W, H, x_in, ax, letter, text, up=0.07):
    y = ax.get_position().y1 * H + up
    fig.text(x_in / W, y / H, letter, fontsize=st.MIN_PT + 1, fontweight="bold", ha="left", va="bottom",
             color=st.INK)
    fig.text((x_in + 0.14) / W, y / H, text, fontsize=st.MIN_PT, fontweight="bold", ha="left", va="bottom",
             color=st.INK)


def shapley(ax):
    sh = f4._plan()["shapley"]
    items = sorted(sh["shapley"].items(), key=lambda kv: kv[1])
    names = [f4.SHAP_LABEL[k] for k, _ in items]
    vals = np.array([v for _, v in items]) * 100
    cols = [f4.STAT if k == "size" else (SET_C if k == "setting" else SIZE_C) for k, _ in items]
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color=cols, height=0.64, edgecolor="white", lw=0.5)
    for yi, v in zip(yy, vals):
        ax.text(v + max(vals) * 0.02, yi, f"{v:.1f}", va="center", ha="left", fontsize=st.MIN_PT, color=st.INK)
    ax.set_yticks(yy, names)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("Share of the variance in log\nexposure intensity (%)")
    ax.tick_params(axis="y", length=0)
    st.despine(ax, keep=("bottom",))
    ax.text(0.98, 0.02, f"Shapley shares sum to the\nmodel R$^2$ of {sh['full_R2']:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=st.MIN_PT, color=st.INK)


def option_hexbin(name="figure3_alt_hexbin_v1"):
    st.apply_rc()
    d, groups = data()
    W, H = tp.PAGE_W, 5.0
    fig = plt.figure(figsize=(W, H))
    L, gap, R = 0.62, 0.30, W - 0.10
    pw = (R - L - 2 * gap) / 3
    top_y, top_h = 2.85, 1.75
    y = d.cost_per_km2.to_numpy()
    xlab = ["Urban extent in 2000 (km²)", "Urban growth, 2000 to 2020 (%)", "Threatened species within 50 km"]
    axs = []
    for i, (ttl, key, xcol, labs, col) in enumerate(groups):
        ax = fig.add_axes([(L + i * (pw + gap)) / W, top_y / H, pw / W, top_h / H])
        x = d[xcol].clip(lower=0.5 if xcol == "setting_footprint" else 1e-3).to_numpy()
        ax.hexbin(x, y, xscale="log", yscale="log", gridsize=(18, 14), cmap=HEX, mincnt=1, bins="log", lw=0.1)
        cs = class_stats(d, key, xcol)
        ax.vlines(cs.x, cs.lo, cs.hi, color=RED, lw=1.2, zorder=3)
        ax.plot(cs.x, cs.gm, "-o", color=RED, lw=1.3, ms=3.6, mec="white", mew=0.6, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(1.5, 150)
        ax.yaxis.set_major_formatter(PLAIN); ax.xaxis.set_major_formatter(PLAIN)
        ax.set_xlabel(xlab[i])
        if i == 0:
            ax.set_ylabel("Exposure intensity\n(km² per km² added)")
        else:
            ax.set_yticklabels([])
        st.despine(ax)
        lo_, hi_ = cs.gm.min(), cs.gm.max()
        ax.text(0.03, 0.97, f"Class means {lo_:.1f} to {hi_:.1f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=st.MIN_PT, color=RED)
        axs.append(ax)
    for i, ax in enumerate(axs):
        title(fig, W, H, (L - 0.60) if i == 0 else (L + i * (pw + gap) - 0.05), ax, "abc"[i], groups[i][0])
    dx = fig.add_axes([1.55 / W, 0.55 / H, 4.6 / W, 1.55 / H])
    shapley(dx)
    title(fig, W, H, 0.02, dx, "d", "Variance in exposure intensity explained")
    return tp.save(fig, name)


def option_dots(name="figure3_alt_dots_v1"):
    st.apply_rc()
    d, groups = data()
    W, H = tp.PAGE_W, 4.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([1.45 / W, 0.55 / H, 2.15 / W, 3.0 / H])
    to_fig = lambda yd: (ax.transData.transform((1, yd))[1]) / fig.dpi
    ref = 10 ** np.log10(d.cost_per_km2).mean()
    yy, ylab, ycol, row = [], [], [], 0
    heads = []
    for ttl, key, xcol, labs, col in groups:
        cs = class_stats(d, key, xcol)
        heads.append((row, ttl, col))
        row += 1
        for _, r in cs.iterrows():
            ax.hlines(row, r.lo, r.hi, color=col, lw=1.6)
            ax.plot(r.gm, row, "o", color=col, ms=5, mec="white", mew=0.6, zorder=3)
            ax.text(r.hi * 1.04, row, f"{r.gm:.1f}", va="center", ha="left", fontsize=st.MIN_PT, color=st.INK,
                    bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)
            yy.append(row); ylab.append(labs[int(r.k)])
            row += 1
        row += 0.4
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for r0, ttl, col in heads:
        ax.text(0.10 / W, r0, ttl, transform=tr, ha="left", va="center", fontsize=st.MIN_PT, fontweight="bold",
                color=col)
    ax.axvline(ref, color=st.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(yy, ylab); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log"); ax.set_xlim(7, 32)
    ax.set_xticks([8, 10, 15, 20, 30]); ax.xaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Geometric mean exposure intensity\n(km² per km² added), 95% interval")
    st.despine(ax, keep=("bottom",))
    title(fig, W, H, 0.02, ax, "a", "Exposure intensity by class", up=0.12)
    dx = fig.add_axes([5.05 / W, 0.75 / H, 1.85 / W, 2.8 / H])
    shapley(dx)
    title(fig, W, H, 3.95, dx, "b", "Variance explained",
          up=(ax.get_position().y1 - dx.get_position().y1) * H + 0.12)
    return tp.save(fig, name)


def option_slope(name="figure3_alt_slope_v1"):
    st.apply_rc()
    d, groups = data()
    W, H = tp.PAGE_W, 2.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.70 / W, 0.62 / H, 3.10 / W, 1.95 / H])
    x = np.arange(1, 6)
    short = {"sz": "Urban extent\nin 2000", "gq": "Urban growth", "sq": "Threatened species\nwithin 50 km"}
    for ttl, key, xcol, labs, col in groups:
        cs = class_stats(d, key, xcol)
        ax.fill_between(x, cs.lo, cs.hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, cs.gm, "-o", color=col, lw=1.6, ms=3.8, mec="white", mew=0.6)
        j = ["sq", "sz", "gq"].index(key)
        ax.text(0.03, 0.98 - j * 0.105, f"{short[key].replace(chr(10), ' ')}, {cs.gm.iloc[0]:.1f} to "
                f"{cs.gm.iloc[-1]:.1f}", transform=ax.transAxes, color=col, va="top", ha="left",
                fontsize=st.MIN_PT)
    ax.set_yscale("log"); ax.set_ylim(7, 40)
    ax.set_yticks([8, 10, 15, 20, 30]); ax.yaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlim(0.8, 5.2); ax.set_xticks(x, ["1\nLowest", "2", "3", "4", "5\nHighest"])
    ax.set_xlabel("Class of the predictor")
    ax.set_ylabel("Geometric mean exposure\nintensity (km² per km² added)")
    st.despine(ax)
    title(fig, W, H, 0.02, ax, "a", "Exposure intensity across classes")
    dx = fig.add_axes([5.05 / W, 0.72 / H, 1.85 / W, 1.85 / H])
    shapley(dx)
    title(fig, W, H, 3.95, dx, "b", "Variance explained")
    return tp.save(fig, name)


def option_raincloud(name="figure3_alt_raincloud_v1"):
    from scipy.stats import gaussian_kde
    st.apply_rc()
    d, groups = data()
    W, H = tp.PAGE_W, 5.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([1.45 / W, 0.62 / H, 2.55 / W, 4.62 / H])
    rng = np.random.default_rng(7)
    allv = np.log10(d.cost_per_km2.to_numpy())
    ref = 10 ** allv.mean()
    grid = np.linspace(np.log10(1.5), np.log10(150), 300)
    yy, ylab, heads, row = [], [], [], 0
    for ttl, key, xcol, labs, col in groups:
        heads.append((row, ttl, col))
        row += 1
        for k in range(5):
            v = np.log10(d.loc[d[key] == k, "cost_per_km2"].to_numpy())
            a_, b_ = np.percentile(v, [0.5, 99.5])
            g = grid[(grid >= a_) & (grid <= b_)]
            kd = gaussian_kde(v)(g)
            kd = 0.42 * kd / kd.max()
            ax.fill_between(10 ** g, row - kd, row, color=col, alpha=0.35, lw=0, zorder=2)
            ax.plot(10 ** np.r_[g[0], g, g[-1]], np.r_[row, row - kd, row], color=col, lw=0.6, zorder=2)
            jy = row + rng.uniform(0.10, 0.40, v.size)
            ax.scatter(10 ** v, jy, s=0.8, color=col, alpha=0.25, lw=0, zorder=1, rasterized=True)
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            ax.add_patch(plt.Rectangle((10 ** q1, row - 0.05), 10 ** q3 - 10 ** q1, 0.10, facecolor="white",
                                       edgecolor=st.INK, lw=0.6, zorder=4))
            ax.plot([10 ** q2] * 2, [row - 0.05, row + 0.05], color=st.INK, lw=1.0, zorder=5)
            p5, p95 = np.percentile(v, [5, 95])
            ax.plot([10 ** p5, 10 ** q1], [row, row], color=st.INK, lw=0.6, zorder=3)
            ax.plot([10 ** q3, 10 ** p95], [row, row], color=st.INK, lw=0.6, zorder=3)
            gm = 10 ** v.mean()
            ax.plot(gm, row, "o", ms=3.6, color=RED, mec="white", mew=0.5, zorder=6)
            ax.text(1.0, row, f"{gm:.1f}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=st.MIN_PT, color=RED)
            yy.append(row); ylab.append(labs[k])
            row += 1
        row += 0.3
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for r0, ttl, col in heads:
        ax.text(0.10 / W, r0 + 0.1, ttl, transform=tr, ha="left", va="center", fontsize=st.MIN_PT,
                fontweight="bold", color=col)
    ax.axvline(ref, color=st.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(yy, ylab); ax.set_ylim(row - 0.6, -0.6); ax.tick_params(axis="y", length=0)
    ax.set_xscale("log"); ax.set_xlim(1.5, 150)
    ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.xaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Exposure intensity (km² per km² added)")
    st.despine(ax, keep=("bottom",))
    ax.text(1.0, -0.55, "Geometric\nmean", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=7, color=RED)
    title(fig, W, H, 0.02, ax, "a", "Exposure intensity by class", up=0.10)
    dx = fig.add_axes([5.30 / W, 2.45 / H, 1.62 / W, 2.55 / H])
    shapley(dx)
    title(fig, W, H, 4.45, dx, "b", "Variance explained",
          up=(ax.get_position().y1 - dx.get_position().y1) * H + 0.10)
    return tp.save(fig, name)


def raincloud_rows(ax, d, key, labs, col, ref, show_head):
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(7)
    grid = np.linspace(np.log10(1.5), np.log10(150), 300)
    for k in range(5):
        row = k
        v = np.log10(d.loc[d[key] == k, "cost_per_km2"].to_numpy())
        a_, b_ = np.percentile(v, [0.5, 99.5])
        g = grid[(grid >= a_) & (grid <= b_)]
        kd = gaussian_kde(v)(g)
        kd = 0.44 * kd / kd.max()
        ax.fill_between(10 ** g, row - kd, row, color=col, alpha=0.35, lw=0, zorder=2)
        ax.plot(10 ** np.r_[g[0], g, g[-1]], np.r_[row, row - kd, row], color=col, lw=0.6, zorder=2)
        ax.scatter(10 ** v, row + rng.uniform(0.10, 0.42, v.size), s=0.7, color=col, alpha=0.25, lw=0, zorder=1,
                   rasterized=True)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        p5, p95 = np.percentile(v, [5, 95])
        ax.add_patch(plt.Rectangle((10 ** q1, row - 0.06), 10 ** q3 - 10 ** q1, 0.12, facecolor="white",
                                   edgecolor=st.INK, lw=0.6, zorder=4))
        ax.plot([10 ** q2] * 2, [row - 0.06, row + 0.06], color=st.INK, lw=1.0, zorder=5)
        ax.plot([10 ** p5, 10 ** q1], [row, row], color=st.INK, lw=0.6, zorder=3)
        ax.plot([10 ** q3, 10 ** p95], [row, row], color=st.INK, lw=0.6, zorder=3)
        gm = 10 ** v.mean()
        ax.plot(gm, row, "o", ms=3.6, color=RED, mec="white", mew=0.5, zorder=6)
        ax.text(1.02, row, f"{gm:.1f}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=st.MIN_PT, color=RED)
    if show_head:
        ax.text(1.02, -0.62, "Geometric\nmean", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=7, color=RED, linespacing=1.0)
    ax.axvline(ref, color=st.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(range(5), labs); ax.set_ylim(4.6, -0.62); ax.tick_params(axis="y", length=0)
    ax.set_xscale("log"); ax.set_xlim(1.5, 150)
    ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.xaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v_, _: ""))
    ax.set_xlabel("Exposure intensity (km² per km² added)")
    st.despine(ax, keep=("bottom",))


def option_grid(name="figure3_distributions_v8"):
    st.apply_rc(min_pt=7)
    d, groups = data()
    W, H = tp.PAGE_W, 5.52
    fig = plt.figure(figsize=(W, H))
    lab, valc = 1.08, 0.36
    pw = (W - 0.04 - 2 * lab - 2 * valc) / 2
    ph = 1.95
    xs = [lab, lab + pw + valc + lab]
    ys = [H - 0.28 - ph, H - 0.28 - ph - 0.78 - ph]
    ref = 10 ** np.log10(d.cost_per_km2).mean()
    cells = [(0, 0), (0, 1), (1, 0)]
    for (i, j), (ttl, key, xcol, labs, col), letter in zip(cells, groups, "abc"):
        ax = fig.add_axes([xs[j] / W, ys[i] / H, pw / W, ph / H])
        raincloud_rows(ax, d, key, labs, col, ref, show_head=False)
        title(fig, W, H, xs[j] - lab + 0.02, ax, letter, ttl, up=0.04)
    dx = fig.add_axes([xs[1] / W, (ys[1] + 0.12) / H, (pw + valc - 0.05) / W, (ph - 0.12) / H])
    shapley(dx)
    title(fig, W, H, xs[1] - lab + 0.02, dx, "d", "Variance in exposure intensity explained",
          up=(ys[1] + ph - (ys[1] + 0.12) - (ph - 0.12)) + 0.04)
    return tp.save(fig, name)


if __name__ == "__main__":
    fn = dict(hexbin=option_hexbin, dots=option_dots, slope=option_slope, raincloud=option_raincloud, grid=option_grid)
    for k in (sys.argv[1:] or fn):
        print(fn[k]())
