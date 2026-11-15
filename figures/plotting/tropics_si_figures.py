"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import tropics_style as st
import tropics_panels as tp
from tropics_config import TABLE_DIR

PAGE_W = tp.PAGE_W
STAT = tp.STAT
GREEN, BLUE = "#1E7B57", "#2C5FAA"


_SENS_V3 = TABLE_DIR / "sensitivity_delineation_v3.csv"
SENS_DELIN = (_SENS_V3 if _SENS_V3.exists()
              else TABLE_DIR / "sensitivity_delineation_v2.csv")


def _plan():
    return json.loads((TABLE_DIR / "plan4.json").read_text())


def _forest(ax, labels, beta, lo, hi, note=None, colors=None, xlabel=""):
    y = np.arange(len(labels))[::-1]
    colors = colors or [st.INK] * len(labels)
    ax.axvline(0.0, color=STAT, lw=0.9, ls=(0, (3, 2)), zorder=1)
    for i, (b, l, h, c) in enumerate(zip(beta, lo, hi, colors)):
        ax.plot([l, h], [y[i], y[i]], color=c, lw=1.6, solid_capstyle="butt",
                zorder=3)
        ax.plot([b], [y[i]], "o", ms=4.0, mfc="white", mec=c, mew=1.3, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, fontsize=7.2)
    st.despine(ax, keep=("bottom",))
    if note is not None:
        for i, t in enumerate(note):
            ax.annotate(t, xy=(1.01, y[i]), xycoords=("axes fraction", "data"),
                        fontsize=6.0, color=st.MUTED, va="center", ha="left",
                        annotation_clip=False)


def fig_quantile(name="figS09_quantile_v5"):
    st.apply_rc()
    q = pd.read_csv(TABLE_DIR / "quantile_scaling.csv")
    fig = plt.figure(figsize=(PAGE_W, 2.85))
    ML, MR, gap = 0.62, 0.16, 0.72
    pw = (PAGE_W - ML - MR - gap) / 2
    ax = fig.add_axes([ML / PAGE_W, 0.60 / 2.85, pw / PAGE_W, 1.95 / 2.85])
    bx = fig.add_axes([(ML + pw + gap) / PAGE_W, 0.60 / 2.85, pw / PAGE_W,
                       1.95 / 2.85])
    for a, col, lab, c in ((ax, "beta", "Size exponent $\\beta$", st.INK),
                           (bx, "gamma", "Setting coefficient $\\gamma$", GREEN)):
        if col == "beta":
            a.fill_between(q["quantile"], q.ci_lower, q.ci_upper, color=c,
                           alpha=0.16, lw=0)
        a.plot(q["quantile"], q[col], "-o", color=c, lw=1.4, ms=3.4, mfc="white",
               mew=1.2)
        a.axhline(0.0, color=STAT, lw=0.9, ls=(0, (3, 2)))
        a.set_xlabel("Quantile of the conditional distribution", fontsize=7.2)
        a.set_ylabel(lab, fontsize=7.2)
        st.despine(a)
    ax.text(0.0, 1.06, "a", transform=ax.transAxes, fontsize=9.0,
            fontweight="bold", va="bottom", color=st.INK)
    bx.text(0.0, 1.06, "b", transform=bx.transAxes, fontsize=9.0,
            fontweight="bold", va="bottom", color=st.INK)
    ax.annotate(f"n = {int(q.n.iloc[0]):,} cities at every quantile",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=6.2,
                color=st.MUTED)
    return tp.save(fig, name)


def fig_delineation(name="figS10_delineation_v4"):
    st.apply_rc()
    d = pd.read_csv(SENS_DELIN)
    prod = json.loads((TABLE_DIR / "numbers.json").read_text())["models"]["A"]["beta"]
    lab = [s.replace("headline (fraction 0.20, link 2, floor 5)",
                     "headline: fraction 0.20, link 2, floor 5 km$^2$")
            .replace("built-up fraction", "built-up fraction")
            .replace("size floor 15 km2", "size floor 15 km$^2$") for s in d.label]
    fig = plt.figure(figsize=(PAGE_W, 3.15))
    ML, MR, gap = 1.95, 0.62, 0.70
    pw = (PAGE_W - ML - MR - gap) / 2
    ax = fig.add_axes([ML / PAGE_W, 0.72 / 3.15, pw / PAGE_W, 1.95 / 3.15])
    _forest(ax, lab, d.beta, d.ci_lower, d.ci_upper,
            colors=[STAT] + [st.INK] * (len(d) - 1),
            xlabel="Size exponent $\\beta$")
    ax.axvline(prod, color=BLUE, lw=1.0, ls=(0, (1, 1.6)))
    ax.annotate("production measurement", xy=(prod, len(d) - 0.42),
                fontsize=5.8, color=BLUE, ha="center", va="bottom",
                annotation_clip=False)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.01))
    bx = fig.add_axes([(ML + pw + gap) / PAGE_W, 0.72 / 3.15, pw / PAGE_W,
                       1.95 / 3.15])
    y = np.arange(len(d))[::-1]
    bx.barh(y, d.n_cities, color=st.SEQ_ORANGE(0.45), height=0.62)
    for i, (n, c) in enumerate(zip(d.n_cities, d.cost_km2)):
        bx.text(n * 1.02, y[i], f"{n:,}", fontsize=6.0, va="center",
                color=st.MUTED)
    bx.set_yticks([]); bx.set_ylim(-0.7, len(d) - 0.3)
    bx.set_xlim(0, d.n_cities.max() * 1.28)
    bx.set_xlabel("Agglomerations delineated", fontsize=7.2)
    st.despine(bx, keep=("bottom",))
    for a, l in ((ax, "a"), (bx, "b")):
        a.text(0.0, 1.10, l, transform=a.transAxes, fontsize=9.0,
               fontweight="bold", va="bottom", color=st.INK)
    return tp.save(fig, name)


def fig_sizevars(name="figS11_size_variables_v4"):
    st.apply_rc()
    d = pd.read_csv(TABLE_DIR / "size_variables.csv")
    fig = plt.figure(figsize=(PAGE_W, 3.05))
    ML, MR = 2.45, 0.95
    ax = fig.add_axes([ML / PAGE_W, 0.66 / 3.05, (PAGE_W - ML - MR) / PAGE_W,
                       2.05 / 3.05])
    lab = [s.replace(" km2", " km$^2$") for s in d.label]
    cols = [STAT] + [st.INK] * (len(d) - 1)
    _forest(ax, lab, d.beta, d.ci_lower, d.ci_upper, colors=cols,
            note=[f"n = {n:,}" for n in d.n],
            xlabel="Size exponent $\\beta$, with 95% confidence interval")
    return tp.save(fig, name)


def fig_scale_robustness(name="figS06_scale_robustness_v1"):
    st.apply_rc()
    dl = pd.read_csv(SENS_DELIN)
    sv = pd.read_csv(TABLE_DIR / "size_variables.csv")
    prod = json.loads((TABLE_DIR / "numbers.json").read_text())["models"]["A"]["beta"]

    ROW, BOT, GAP, TOP = 0.255, 0.62, 0.82, 0.46
    h_dl_in, h_sv_in = ROW * len(dl), ROW * len(sv)
    fig_h = BOT + h_sv_in + GAP + h_dl_in + TOP
    fig = plt.figure(figsize=(PAGE_W, fig_h))
    ML, MR = 2.55, 0.95
    aw = (PAGE_W - ML - MR) / PAGE_W
    ax_b = fig.add_axes([ML / PAGE_W, BOT / fig_h, aw, h_sv_in / fig_h])
    ax_a = fig.add_axes([ML / PAGE_W, (BOT + h_sv_in + GAP) / fig_h, aw,
                         h_dl_in / fig_h])

    lab_a = [s.replace("headline (fraction 0.20, link 2, floor 5)",
                       "headline: fraction 0.20, link 2, floor 5 km$^2$")
              .replace("size floor 15 km2", "size floor 15 km$^2$")
             for s in dl.label]
    _forest(ax_a, lab_a, dl.beta, dl.ci_lower, dl.ci_upper,
            colors=[STAT] + [st.INK] * (len(dl) - 1),
            note=[f"{n:,} cities" for n in dl.n_cities],
            xlabel="")
    ax_a.axvline(prod, color=BLUE, lw=1.0, ls=(0, (1, 1.6)))
    ax_a.annotate("production measurement", xy=(prod - 0.002, len(dl) - 0.42),
                  fontsize=5.8, color=BLUE, ha="right", va="bottom",
                  annotation_clip=False)
    ax_a.text(0.0, 1.16, "a", transform=ax_a.transAxes, fontsize=9.0,
              fontweight="bold", va="bottom", color=st.INK)

    lab_b = [s.replace(" km2", " km$^2$") for s in sv.label]
    _forest(ax_b, lab_b, sv.beta, sv.ci_lower, sv.ci_upper,
            colors=[STAT] + [st.INK] * (len(sv) - 1),
            note=[f"n = {n:,}" for n in sv.n],
            xlabel="Size exponent $\\beta$, with 95% confidence interval")
    ax_b.text(0.0, 1.10, "b", transform=ax_b.transAxes, fontsize=9.0,
              fontweight="bold", va="bottom", color=st.INK)

    lo = min(dl.ci_lower.min(), sv.ci_lower.min())
    hi = max(dl.ci_upper.max(), sv.ci_upper.max())
    pad = 0.06 * (hi - lo)
    for a in (ax_a, ax_b):
        a.set_xlim(lo - pad, hi + pad)
    ax_a.tick_params(axis="x", labelbottom=False)
    return tp.save(fig, name)


def _taxon_setting_frame():
    import tropics_models as p4
    d = p4.table()
    se = p4._taxon_setting(d)
    x = d[["city_id"]].copy()
    for s in se.values():
        x = x.join(s, on="city_id")
    return x.fillna(0.0)


def fig_crosstaxon(name="figS12_cross_taxon_v4"):
    st.apply_rc()
    rows = _plan()["cross_taxon"]
    x = _taxon_setting_frame()
    cols = [c for c in ("set_birds", "set_mammals", "set_herps") if c in x.columns]
    nm = {"set_birds": "birds", "set_mammals": "mammals",
          "set_herps": "amphibians\nand reptiles"}

    fig = plt.figure(figsize=(PAGE_W, 3.35))
    ML, MR, gap = 2.32, 0.28, 1.12
    pw = (PAGE_W - ML - MR - gap) / 2
    ax = fig.add_axes([ML / PAGE_W, 0.74 / 3.10, pw / PAGE_W, 1.95 / 3.10])
    tax = {"birds": "birds", "mammals": "mammals",
           "herps": "amphibians and reptiles"}
    def _grp(t):
        return " + ".join(tax.get(k.strip(), k.strip()) for k in t.split("+"))
    lab = [f"{_grp(r['setting_from'])} setting\n→ {_grp(r['exposure_of'])}"
           for r in rows]
    b = np.array([r["gamma"] for r in rows])
    se = np.array([r["gamma_se"] for r in rows])
    pal = [st.INK, GREEN, BLUE]
    _forest(ax, lab, b, b - 1.96 * se, b + 1.96 * se,
            colors=[pal[i % len(pal)] for i in range(len(rows))],
            xlabel="Setting coefficient $\\gamma$ on the other groups")
    ax.set_xlim(0, max(b + 1.96 * se) * 1.30)
    for i, r in enumerate(rows):
        pv = float(r["p"])
        txt = "P < 10$^{-15}$" if pv < 1e-15 else f"P = {pv:.1g}"
        ax.annotate(txt, xy=(0.985, len(rows) - 1 - i),
                    xycoords=("axes fraction", "data"), fontsize=st.MIN_PT,
                    color=st.MUTED, va="center", ha="right",
                    annotation_clip=False)
    bx = fig.add_axes([(ML + pw + gap) / PAGE_W, 0.74 / 3.10, pw / PAGE_W,
                       1.95 / 3.10])
    m = np.ones((len(cols), len(cols)))
    for i, a in enumerate(cols):
        for j, c in enumerate(cols):
            m[i, j] = x[a].corr(x[c], method="spearman")
    im = bx.imshow(m, cmap=st.SEQ_ORANGE, vmin=0.0, vmax=1.0)
    bx.set_xticks(range(len(cols))); bx.set_yticks(range(len(cols)))
    bx.set_xticklabels([nm[c] for c in cols], fontsize=6.4)
    bx.set_yticklabels([nm[c] for c in cols], fontsize=6.4)
    for i in range(len(cols)):
        for j in range(len(cols)):
            bx.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.4, color="white" if m[i, j] > 0.6 else st.INK)
    bx.tick_params(length=0)
    for s in bx.spines.values():
        s.set_visible(False)
    for a, l in ((ax, "a"), (bx, "b")):
        a.text(0.0, 1.07, l, transform=a.transAxes, fontsize=9.0,
               fontweight="bold", va="bottom", color=st.INK)
    bx.annotate(f"n = {len(x):,} agglomerations", xy=(0.0, -0.30),
                xycoords="axes fraction", fontsize=6.0, color=st.MUTED)
    ax.annotate("95% intervals are narrower than the markers; the widest is "
                "$\\pm$0.026", xy=(0.0, -0.30), xycoords="axes fraction",
                fontsize=6.0, color=st.MUTED)
    return tp.save(fig, name)


def fig_species_atlas(name="figS11_species_atlas_v5"):
    import tropics_speciesmaps as sm
    return sm.fig_worst_species(name=name, n=20)


def fig_pairs_selection(name="figS12_pair_selection_v1"):
    import tropics_models as p4
    st.apply_rc()
    gaps, n_named = p4.admissible_gaps()
    pl = _plan()
    drawn = [m["gap_log10"] for m in pl["matched"]]
    ranked = [m["gap_log10"] for m in pl["matched_ranked"]]
    fig = plt.figure(figsize=(PAGE_W, 2.70))
    ax = fig.add_axes([0.095, 0.290, 0.86, 0.545])
    ax.hist(gaps, bins=70, color=st.HAIR, edgecolor="none")
    ax.set_yscale("log")
    for v in ranked:
        ax.axvline(v, color=BLUE, lw=1.0, ls=(0, (3, 2)), zorder=4)
    for v in drawn:
        ax.axvline(v, color=STAT, lw=1.3, zorder=5)
    ax.set_xlabel("Difference in log$_{10}$ exposure intensity within a "
                  "size-matched pair", fontsize=7.2)
    ax.set_ylabel("Admissible pairs", fontsize=7.2)
    st.despine(ax)
    hs = [plt.Line2D([0], [0], color=STAT, lw=1.3),
          plt.Line2D([0], [0], color=BLUE, lw=1.0, ls=(0, (3, 2)))]
    ax.legend(hs, ["drawn in main-text Figure 6",
                   "unconstrained ranking, top three"],
              fontsize=6.2, loc="lower left", handlelength=1.8, frameon=True,
              framealpha=0.94, edgecolor="none", facecolor="white",
              borderpad=0.3, bbox_to_anchor=(0.015, 0.03))
    q = float((gaps <= min(drawn)).mean() * 100)
    print(f"    {len(gaps):,} admissible pairs among {n_named:,} named "
          f"agglomerations; the smallest drawn contrast exceeds {q:.1f}%")
    return tp.save(fig, name)


def fig_pairs_ranked(name="figS14_pairs_ranked_v3"):
    import tropics_main_figures as f4
    return f4.figure6(name=name, pairs=_plan()["matched_ranked"])


if __name__ == "__main__":
    import sys
    JOBS = dict(quantile=fig_quantile, delineation=fig_delineation,
                sizevars=fig_sizevars, crosstaxon=fig_crosstaxon,
                atlas=fig_species_atlas, ranked=fig_pairs_ranked,
                cfrobust=fig_cf_robust)
    for k in (sys.argv[1:] or list(JOBS)):
        print(JOBS[k]().name if hasattr(JOBS[k](), "name") else k, flush=True)


def _cfr():
    return json.loads((TABLE_DIR / "cf_robust.json").read_text())


def fig_cf_robust(name="figS15_counterfactual_robustness_v4"):
    st.apply_rc()
    r = _cfr()
    al = np.asarray(r["alphas"]); a0 = r["alpha_hat"]
    fig = plt.figure(figsize=(PAGE_W, 4.95))
    L, R, T, B = 0.132, 0.980, 0.945, 0.132
    hg, vg = 0.225, 0.195
    cw = (R - L - hg) / 2; ch = (T - B - vg) / 2
    def cell(i, j):
        return fig.add_axes([L + j * (cw + hg), B + (1 - i) * (ch + vg), cw, ch])

    def frame(ax, ylab):
        ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
        ax.axvline(a0, color=STAT, lw=0.8, ls=(0, (1, 2)), zorder=0)
        ax.set_xlim(al.min(), al.max())
        ax.set_xlabel("Concentration exponent $\\alpha$", fontsize=7.2)
        ax.set_ylabel(ylab, fontsize=7.2, linespacing=1.5)
        st.despine(ax)

    ax = cell(0, 0)
    g = r["aggregate"]
    ax.fill_between(al, g["ci_lo"], g["ci_hi"], color=STAT, alpha=0.14, lw=0,
                    label="95% cluster bootstrap")
    ax.plot(al, g["pct"], color=STAT, lw=1.8, label="point estimate")
    frame(ax, "Change relative to the fitted\nreference, $\\alpha$ = %.2f (%%)" % a0)
    ax.legend(loc="lower left", handlelength=1.5, frameon=False)
    ax.annotate("the fitted allocation reproduces the observed\n"
                f"total exposure to within {abs(g['fitted_vs_observed_pct']):.1f}%",
                xy=(0.03, 0.97), xycoords="axes fraction", fontsize=5.8,
                color=st.MUTED, va="top", linespacing=1.5)
    tp.head(ax, "a", "Estimation uncertainty")

    ax = cell(0, 1)
    for key, lab, col, ls in (("cap2", "no city above 2x its own growth", GREEN, "-"),
                              ("cap5", "no city above 5x its own growth", BLUE,
                               (0, (4, 2)))):
        ax.plot(al, r[key]["pct"], color=col, lw=1.6, ls=ls, label=lab)
    ax.plot(al, g["pct"], color=STAT, lw=1.8, label="unconstrained")
    frame(ax, "Change relative to the fitted\nreference (%)")
    ax.legend(loc="upper left", handlelength=1.7, frameon=False)
    tp.head(ax, "b", "A capacity constraint")

    ax = cell(1, 0)
    for key, lab, col, ls in (("by_realm_pool", "within realm", GREEN, (0, (4, 2))),
                              ("pantropical_pool", "across the tropics", BLUE,
                               (0, (1, 1.6)))):
        ax.plot(al, r[key]["pct"], color=col, lw=1.6, ls=ls, label=lab)
    ax.plot(al, g["pct"], color=STAT, lw=1.8, label="within country (main text)")
    frame(ax, "Change relative to the fitted\nreference (%)")
    ax.legend(loc="upper left", handlelength=1.7, frameon=False)
    tp.head(ax, "c", "The boundary land may cross")

    ax = cell(1, 1)
    rows = ([(k, v, "response") for k, v in r["response"].items()]
            + [(k, v, "realm") for k, v in r["realm"].items()])
    lab = [k for k, _, _ in rows]
    pt = [v["pct"][-1] for _, v, _ in rows]
    lo = [v["ci_lo"][-1] for _, v, _ in rows]
    hi = [v["ci_hi"][-1] for _, v, _ in rows]
    cols = [STAT if l * h > 0 else st.INK for l, h in zip(lo, hi)]
    _forest(ax, lab, pt, lo, hi, colors=cols,
            xlabel="Change at $\\alpha$ = 2, the most concentrated\nallocation (%)")
    ax.xaxis.label.set_linespacing(1.5)
    tp.head(ax, "d", "What is resolved at the concentrated end")
    return tp.save(fig, name)
