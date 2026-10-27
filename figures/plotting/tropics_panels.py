"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy
import matplotlib.patheffects as pe

import tropics_style as st
from tropics_config import TABLE_DIR as TAB_DIR, FIG_DIR, INTERIM_DIR

HEX_CMAP = "magma_r"
NULL = "#B0184E"
STAT = "#B0184E"


def stat(ax, text, loc="lower right", fs=6.4, pad=0.030):
    va, ha = loc.split()
    ax.text(pad if ha == "left" else 1 - pad,
            pad if va == "lower" else 1 - pad, text,
            transform=ax.transAxes, fontsize=fs, color=STAT, ha=ha,
            va="bottom" if va == "lower" else "top", linespacing=1.35,
            zorder=12, bbox=dict(fc="white", ec="none", alpha=0.80, pad=1.6))


def label_line(ax, x, y, text, fs=6.0, color=st.MUTED, dx=0.0, dy=0.0):
    import numpy as _np
    ang = _np.degrees(_np.arctan2(y[1] - y[0], x[1] - x[0]))
    tr = ax.transData.transform
    p0, p1 = tr((x[0], y[0])), tr((x[1], y[1]))
    scr = _np.degrees(_np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(0.5 * (x[0] + x[1]) + dx, 0.5 * (y[0] + y[1]) + dy, text,
            rotation=scr, rotation_mode="anchor", fontsize=fs, color=color,
            ha="center", va="bottom", zorder=6)
POS, NEG = "#0F8A6E", "#C05A0F"


PAGE_W = 7.0


def head(ax, letter, title, subtitle=None, dy=0.0):
    t = ax.transAxes
    def at(x_pt, y_pt):
        return offset_copy(t, fig=ax.figure, x=x_pt, y=y_pt, units="points")
    y = 4.5 + dy
    ax.text(0, 1, letter, transform=at(0, y), fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", color=st.INK)


def suptitle(fig, text, x_in=None, fs=10.0):
    return None


def clipping_report(fig):
    r = fig.canvas.get_renderer()
    W, H = (v * fig.dpi for v in fig.get_size_inches())
    out = []
    for a in fig.get_children():
        try:
            bb = a.get_window_extent(r)
        except Exception:
            continue
        if bb.width == 0 or bb.height == 0:
            continue
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            out.append((type(a).__name__, round(bb.x0), round(bb.y0),
                        round(bb.x1), round(bb.y1)))
    return out, (round(W), round(H))


def note(ax, text, loc="upper right", fs=6.2, pad=0.028):
    va, ha = loc.split()
    x = pad if ha == "left" else 1 - pad
    y = pad if va == "lower" else 1 - pad
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, color=st.INK,
            ha=ha, va="bottom" if va == "lower" else "top", linespacing=1.35,
            zorder=12, bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.6))


def hexpanel(ax, x, y, gridsize=44, cmap=HEX_CMAP, mincnt=1):
    ok = np.isfinite(x) & np.isfinite(y)
    hb = ax.hexbin(x[ok], y[ok], gridsize=gridsize, cmap=cmap, mincnt=mincnt,
                   norm=LogNorm(), linewidths=0.0)
    st.despine(ax)
    return hb


FITLINE = "#1357C8"
FITLINE_HALO = [pe.withStroke(linewidth=2.6, foreground="white")]


def binned_median(ax, x, y, nbin=14, color=FITLINE, label=None, halo=True):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = np.nanquantile(x, np.linspace(0, 1, nbin + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    cx, med, lo, hi = [], [], [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() < 15:
            continue
        cx.append(np.median(x[m])); med.append(np.median(y[m]))
        lo.append(np.quantile(y[m], 0.25)); hi.append(np.quantile(y[m], 0.75))
    ax.fill_between(cx, lo, hi, color=color, alpha=0.13, lw=0, zorder=4)
    ax.plot(cx, med, color=color, lw=1.5, zorder=5, label=label,
            solid_capstyle="round",
            path_effects=FITLINE_HALO if halo else None)
    return np.array(cx), np.array(med)


def forest(ax, labels, est, lo, hi, null=0.0, color=st.INK, fs=6.8,
           highlight=None):
    y = np.arange(len(labels))[::-1]
    ax.axvline(null, color=NULL, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for i, (e, l, h) in enumerate(zip(est, lo, hi)):
        c = color if highlight is None or not highlight[i] else NULL
        w = 1.9 if highlight is not None and highlight[i] else 1.1
        ax.plot([l, h], [y[i], y[i]], color=c, lw=w, solid_capstyle="round",
                zorder=3)
        ax.plot([e], [y[i]], "o", ms=3.4, color=c, mec="white", mew=0.6, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fs)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    st.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)


def colorbar(fig, hb, ax, label="cities", loc=(0.60, 0.06), w=0.34, h=0.030):
    p = ax.get_position()
    cax = fig.add_axes([p.x0 + loc[0] * p.width, p.y0 + loc[1] * p.height,
                        w * p.width, h * p.height])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=5.8, length=2, pad=1.2, color=st.MUTED)
    cax.text(1.03, 0.5, label, transform=cax.transAxes, fontsize=6.2,
             color=st.MUTED, va="center", ha="left")
    return cax


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    return FIG_DIR / f"{name}.png"


def cities(sample=True):
    d = pd.read_csv(TAB_DIR / "cities_cost.csv")
    if sample:
        d = d[d.urban_km2_base > 0].reset_index(drop=True)
    f = pd.read_csv(INTERIM_DIR / "city_form.csv")
    d = d.merge(f, on="city_id", how="left")
    tot = d[["form_infill_km2", "form_edge_km2", "form_leapfrog_km2"]].sum(axis=1)
    for k in ("infill", "edge", "leapfrog"):
        d[f"share_{k}"] = d[f"form_{k}_km2"] / tot.replace(0, np.nan)
    d["growth_ratio"] = d.urban_km2_final / d.urban_km2_base.replace(0, np.nan)
    d["growth_pct"] = (d.growth_ratio - 1.0) * 100.0
    d["abs_lat"] = d.lat.abs()
    return d


def fig_scaling(name="fig_scaling_v2"):
    st.apply_rc()
    d = cities()
    fits = pd.read_csv(TAB_DIR / "scaling_results.csv")
    hq = fits[(fits.response == "cost_per_km2") &
              (fits.label == "headline, spatial 2 deg")]
    if len(hq) == 0:
        hq = fits[fits.response == "cost_per_km2"].iloc[[0]]
    sp = json.loads((TAB_DIR / "scaling_spatial.json").read_text())["cost_per_km2"]
    qr = pd.read_csv(TAB_DIR / "quantile_scaling.csv")
    sen = pd.read_csv(TAB_DIR / "sensitivity_delineation_all.csv")
    vd = pd.read_csv(TAB_DIR / "variance_decomposition.csv")
    dm = pd.read_csv(TAB_DIR / "driver_model.csv")

    fig = plt.figure(figsize=(PAGE_W, 5.3))
    L, R, T, B = 0.132, 0.958, 0.895, 0.100
    hgap, vgap = 0.128, 0.185
    cw = (R - L - 2 * hgap) / 3
    ch = (T - B - vgap) / 2
    def cell(i, j):
        return fig.add_axes([L + j * (cw + hgap), B + (1 - i) * (ch + vgap), cw, ch])

    import tropics_refit as trf
    d = d.merge(trf.footprint_setting(), on="city_id", how="left")
    d["setting_footprint"] = d.setting_footprint.fillna(0.0)
    x = np.log10(d.urban_km2_base.replace(0, np.nan).values)
    y = np.log10(d.cost_per_km2.values)
    s = np.log10(d.setting_footprint.clip(lower=0.5).values)

    ax = cell(0, 0)
    hb = hexpanel(ax, x, y)
    binned_median(ax, x, y)
    ax.set_xlabel("City urban extent, 2000 (km$^2$)", fontsize=7.4)
    ax.set_ylabel("Species-range exposure per km$^2$", fontsize=7.4)
    _logticks(ax, "x", [1, 10, 100, 1000]); _logticks(ax, "y", [1, 10, 100, 1000])
    head(ax, "a", "Size carries almost none")
    stat(ax, f"$\\beta$ = {sp['beta']:+.4f}\n"
         f"[{sp['ci_lower']:+.4f}, {sp['ci_upper']:+.4f}]\nP = {sp['p_value']:.1g}",
         "upper right")
    colorbar(fig, hb, ax)

    ax = cell(0, 1)
    hb = hexpanel(ax, s, y)
    binned_median(ax, s, y)
    ax.set_xlabel("Threatened species within 50 km of the 2000 footprint",
                  fontsize=7.0)
    ax.set_ylabel("Species-range exposure per km$^2$", fontsize=7.4)
    _logticks(ax, "x", [3, 10, 30, 100]); _logticks(ax, "y", [1, 10, 100, 1000])
    head(ax, "b", "Setting carries some")
    stat(ax, f"$\\gamma$ = {sp['gamma_setting']:.3f} $\\pm$ {sp['gamma_se']:.3f}\n"
         f"n = {sp['n']:,}", "upper left")
    colorbar(fig, hb, ax)

    ax = cell(0, 2)
    forest(ax, [f"$q_{{{q:g}}}$" for q in qr["quantile"]], qr.beta, qr.ci_lower,
           qr.ci_upper, highlight=(qr.ci_lower > 0) | (qr.ci_upper < 0), fs=7.4)
    ax.set_xlabel("Size coefficient $\\beta$", fontsize=7.4)
    head(ax, "c", "At every quantile")

    ax = cell(1, 0)
    lab = [_short(l, t) for l, t in zip(sen.label, sen.taxa)]
    forest(ax, lab, sen.beta, sen.ci_lower, sen.ci_upper,
           highlight=(sen.ci_lower > 0) | (sen.ci_upper < 0), fs=5.9)
    ax.set_xlabel("Size coefficient $\\beta$", fontsize=7.4)
    head(ax, "d", "Under every delineation")

    ax = cell(1, 1)
    seq = vd[~vd.model.str.contains("except|last")].copy()
    names = ["size", "setting", "biome", "realm", "country"]
    r2 = seq.R2.values * 100
    inc = seq.increment.values * 100
    base = np.concatenate([[0.0], r2[:-1]])
    ax.bar(np.arange(len(r2)), inc, bottom=base, width=0.66,
           color=[NULL] + [st.BLUE] * (len(r2) - 1), edgecolor="white", lw=0.6)
    for i, (b, v) in enumerate(zip(base, inc)):
        ax.text(i, b + v + 1.4, f"{v:.1f}", ha="center", va="bottom", fontsize=6.4,
                color=st.INK)
    ax.set_xticks(np.arange(len(r2))); ax.set_xticklabels(names, fontsize=6.2, rotation=22, ha="right")
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=7.4)
    ax.set_ylim(0, 92); st.despine(ax)
    last = vd[vd.model == "size added last"].increment.values[0] * 100
    head(ax, "e", "Size explains almost nothing")
    stat(ax, f"size entered first {r2[0]:.2f}%\nentered last {last:.3f}%\n"
         f"location together {r2[-1]:.0f}%", "upper left", fs=6.0)

    ax = cell(1, 2)
    dm = dm.reindex(dm.beta_std.abs().sort_values(ascending=False).index)
    lab = [_pretty(p) for p in dm.predictor]
    forest(ax, lab, dm.beta_std, dm.beta_std - 1.96 * dm.se,
           dm.beta_std + 1.96 * dm.se,
           highlight=[p == "log_size" for p in dm.predictor], fs=6.4)
    ax.set_xlabel("Standardized coefficient", fontsize=7.4)
    head(ax, "f", "Against the others")
    note(ax, "standardized\nn = 7,207", "lower right", fs=6.0)

    suptitle(fig, "Three-quarters of the variation in what a square kilometer of "
             "urban land converts is accounted for by where it is")
    return save(fig, name)


def _logticks(ax, which, vals):
    lv = np.log10(vals)
    if which == "x":
        ax.set_xticks(lv); ax.set_xticklabels([f"{v:g}" for v in vals])
    else:
        ax.set_yticks(lv); ax.set_yticklabels([f"{v:g}" for v in vals])


def _short(label, taxa):
    t = "all taxa" if str(taxa) == "all" else "MAR"
    l = (str(label).replace("headline (fraction 0.20, link 2, floor 5)", "headline")
         .replace("built-up fraction", "fraction").replace("linking distance", "link")
         .replace("size floor ", "floor ").replace(" km2", "")
         .replace(" cells", "").replace(" cell", ""))
    return f"{l}, {t}"


def _pretty(p):
    return {"log_setting": "setting richness", "abs_lat": "|latitude|",
            "narrow_cost_share": "narrow-range share",
            "form_infill_share": "infill share", "growth_pct": "growth rate",
            "form_leapfrog_share": "leapfrog share",
            "log_size": "city size"}.get(p, p)


def city_names(d, path=None, tol_cells=3):
    from tropics_config import RAW_DIR, DELIN_DEG, MOSAIC_LAT_TOP
    from tropics_cities import load_city_cells
    p = path or (RAW_DIR / "tropical_places.csv")
    d = d.copy()
    if not p.exists():
        d["name"] = ""
        return d
    pl = pd.read_csv(p).sort_values("pop", ascending=False)
    rr, cc, ids = load_city_cells()
    key = rr.astype(np.int64) * 43200 + cc.astype(np.int64)
    order = np.argsort(key)
    key, kid = key[order], ids[order]

    prow = ((MOSAIC_LAT_TOP - pl.lat.values) / DELIN_DEG - 0.5)
    pcol = ((pl.lon.values + 180.0) / DELIN_DEG - 0.5)
    best = {}
    offs = range(-tol_cells, tol_cells + 1)
    for i in range(len(pl)):
        r0, c0 = int(round(prow[i])), int(round(pcol[i]))
        found = None
        for dr in offs:
            for dc in offs:
                k = (r0 + dr) * 43200 + (c0 + dc)
                j = np.searchsorted(key, k)
                if j < key.size and key[j] == k:
                    found = int(kid[j]); break
            if found is not None:
                break
        if found is not None and found not in best:
            best[found] = pl.name.values[i]
    d["name"] = [best.get(int(c), "") for c in d.city_id]
    return d


def label_for(row):
    n = str(row.get("name") or "")
    if n:
        return f"{n}, {row['country']}"
    return f"{row['country']} {abs(row['lat']):.1f}°{'N' if row['lat'] >= 0 else 'S'}"


BIOME_SHORT = {
    "Tropical & Subtropical Moist Broadleaf Forests": "Moist broadleaf",
    "Tropical & Subtropical Dry Broadleaf Forests": "Dry broadleaf",
    "Tropical & Subtropical Coniferous Forests": "Conifer",
    "Tropical & Subtropical Grasslands, Savannas & Shrublands": "Savanna",
    "Temperate Grasslands, Savannas & Shrublands": "Temperate grass",
    "Flooded Grasslands & Savannas": "Flooded grass",
    "Montane Grasslands & Shrublands": "Montane grass",
    "Deserts & Xeric Shrublands": "Desert",
    "Mangroves": "Mangrove",
}


def fig_concentration(name="fig_concentration_v2"):
    st.apply_rc()
    d = city_names(cities())
    nums = json.loads((TAB_DIR / "numbers.json").read_text())

    fig = plt.figure(figsize=(PAGE_W, 6.6))
    L, R, T, B = 0.100, 0.955, 0.905, 0.150
    hgap = 0.128
    colw = (R - L - 2 * hgap) / 3
    x1 = L; x2 = L + colw + hgap; x3 = x2 + colw + hgap
    vgap = 0.105
    row_h = (T - B - 2 * vgap) / 3
    ax_a = fig.add_axes([x1, B + 2 * (row_h + vgap), colw, row_h])
    ax_e = fig.add_axes([x1, B + row_h + vgap, colw, row_h])
    ax_b = fig.add_axes([x1 + 0.022, B, colw - 0.022, row_h])
    ax_c = fig.add_axes([x2, B, colw, T - B])
    ax_d = fig.add_axes([x3, B, colw, T - B])

    order = np.argsort(-d.urban_km2_base.values)
    cost = d.cost_km2.values[order]
    add = d.urban_added_km2.values[order]
    frac = np.arange(1, len(cost) + 1) / len(cost) * 100
    ax_a.plot(frac, np.cumsum(cost) / cost.sum() * 100, color=NULL, lw=1.6,
              label="species-range exposure")
    ax_a.plot(frac, np.cumsum(add) / add.sum() * 100, color=st.BLUE, lw=1.6,
              ls=(0, (4, 2)), label="urban land added")
    ax_a.plot([0, 100], [0, 100], color=st.FAINT, lw=0.8, zorder=0)
    k1 = max(1, int(round(len(cost) * 0.01)))
    share_cost_1 = cost[:k1].sum() / cost.sum()
    share_land_1 = add[:k1].sum() / add.sum()
    cum = np.cumsum(cost) / cost.sum() * 100
    rows = []
    for share in (25, 50, 75):
        xk = frac[int(np.searchsorted(cum, share))]
        ax_a.plot([xk], [share], "o", ms=3.0, mfc="white", mec=STAT, mew=1.0,
                  zorder=6)
        rows.append(f"{share}% of exposure: largest {xk:.1f}% of cities")
    rows.append(f"top 1%: {share_cost_1*100:.1f}% of exposure")
    rows.append(f"on {share_land_1*100:.1f}% of the land")
    ax_a.text(0.30, 0.62, "\n".join(rows), transform=ax_a.transAxes,
              fontsize=5.7, color=STAT, ha="left", va="top", linespacing=1.5,
              zorder=6)
    label_line(ax_a, [62, 84], [62, 84], "equal shares", fs=5.8, dy=1.0)
    ax_a.set_xlim(0, 100); ax_a.set_ylim(0, 100)
    ax_a.set_xlabel("Cities, ranked by 2000 urban extent (%)", fontsize=7.2)
    ax_a.set_ylabel("Cumulative share (%)", fontsize=7.2)
    ax_a.legend(fontsize=6.2, loc="lower right", handlelength=1.6, frameon=False,
                bbox_to_anchor=(1.02, -0.02))
    st.despine(ax_a)
    head(ax_a, "a", "Concentration")

    cf = nums["counterfactual"]
    al = np.array([float(k) for k in cf["ratio_observed_m"]])
    o = np.array([cf["ratio_observed_m"][k] for k in cf["ratio_observed_m"]])
    m = np.array([cf["ratio_modeled_m"][k] for k in cf["ratio_modeled_m"]])
    ax_e.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax_e.plot(al, (o - 1) * 100, color=NULL, lw=1.6, label="observed intensity")
    ax_e.plot(al, (m - 1) * 100, color=st.BLUE, lw=1.6, ls=(0, (4, 2)),
              label="predicted intensity")
    a0 = cf["implied_alpha"]
    ax_e.axvline(a0, color=STAT, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax_e.annotate(f"observed\nallocation\n$\\alpha$ = {a0:.2f}", xy=(a0, 0),
                  xytext=(a0 + 0.12, -6.4), fontsize=5.9, color=STAT,
                  ha="left", va="center", linespacing=1.4)
    ax_e.set_xlim(al.min(), al.max())
    ax_e.set_ylim(-11, 6)
    ax_e.set_xlabel("Concentration exponent $\\alpha$", fontsize=7.2)
    ax_e.set_ylabel("Change in pan-tropical\nexposure (%)", fontsize=7.2)
    ax_e.legend(fontsize=6.2, loc="upper left", handlelength=1.6, frameon=False)
    st.despine(ax_e)
    head(ax_e, "e", "Redistribution within countries")
    ax_e.text(0.0, -0.30, "dispersed", transform=ax_e.transAxes, fontsize=5.8,
              color=st.MUTED, ha="left", va="top")
    ax_e.text(1.0, -0.30, "concentrated", transform=ax_e.transAxes, fontsize=5.8,
              color=st.MUTED, ha="right", va="top")

    realms = ["Neotropic", "Afrotropic", "Indomalayan", "Australasia"]
    piv = (d[d.realm.isin(realms)]
           .assign(bs=lambda x: x.biome.map(BIOME_SHORT))
           .pivot_table(index="bs", columns="realm", values="cost_km2",
                        aggfunc="sum").reindex(columns=realms))
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index][:6]
    M = piv.values / d.cost_km2.sum() * 100
    cmap_b = st.SEQ_GREEN.copy(); cmap_b.set_bad("#F0EFEC")
    im = ax_b.imshow(np.ma.masked_invalid(M), cmap=cmap_b, aspect="auto",
                     vmin=0, vmax=np.nanmax(M))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v) and v >= 0.05:
                ax_b.text(j, i, f"{v:.0f}" if v >= 1 else f"{v:.1f}",
                          ha="center", va="center", fontsize=6.0,
                          color="white" if v > np.nanmax(M) * 0.55 else st.INK)
    ax_b.set_xticks(range(len(realms)))
    ax_b.set_xticklabels([r.replace("Indomalayan", "Indomalay.") for r in realms],
                         fontsize=6.2, rotation=32, ha="right")
    ax_b.set_yticks(range(M.shape[0]))
    ax_b.set_yticklabels(piv.index, fontsize=6.2)
    ax_b.tick_params(length=0)
    for s in ax_b.spines.values():
        s.set_visible(False)
    head(ax_b, "b", "Biome and realm (%)")

    ct = (d.groupby("country")
            .agg(cost=("cost_km2", "sum"), add=("urban_added_km2", "sum"),
                 n=("city_id", "size")).reset_index())
    ct["per"] = ct.cost / ct["add"]
    ct = ct.nlargest(18, "cost").iloc[::-1]
    y = np.arange(len(ct))
    cmax = ct.per.max()
    cols = st.SEQ_ORANGE(np.clip(ct.per / cmax, 0.08, 1))
    ax_c.barh(y, ct.cost / 1e3, color=cols, edgecolor="white", lw=0.4, height=0.72)
    tot = d.cost_km2.sum()
    for yi, r in zip(y, ct.itertuples()):
        ax_c.text(r.cost / 1e3 + ct.cost.max() / 1e3 * 0.02, yi,
                  f"{r.cost/tot*100:.0f}%   {r.per:.0f}", va="center",
                  ha="left", fontsize=5.6, color=st.MUTED)
    ax_c.set_yticks(y); ax_c.set_yticklabels(ct.country, fontsize=6.3)
    ax_c.set_xlabel("Species-range exposure (10$^3$ km$^2$)", fontsize=7.2)
    ax_c.set_xlim(0, ct.cost.max() / 1e3 * 1.30)
    ax_c.tick_params(axis="y", length=0)
    st.despine(ax_c, keep=("bottom",))
    head(ax_c, "c", "Countries")
    note(ax_c, "at each bar: share of the\npan-tropical total, then\nexposure per km$^2$",
         "lower right", fs=5.8, pad=0.075)

    tc = d.nlargest(25, "cost_km2").iloc[::-1]
    y = np.arange(len(tc))
    cols = st.SEQ_ORANGE(np.clip(tc.cost_per_km2 / tc.cost_per_km2.max(), 0.08, 1))
    ax_d.hlines(y, 0, tc.cost_km2 / 1e3, color=st.HAIR, lw=0.8, zorder=1)
    ax_d.scatter(tc.cost_km2 / 1e3, y, s=26, c=cols, edgecolor="white",
                 linewidth=0.5, zorder=3)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([label_for(r) for _, r in tc.iterrows()], fontsize=5.5)
    ax_d.set_xlabel("Species-range exposure (10$^3$ km$^2$)", fontsize=7.2)
    ax_d.set_xlim(0, tc.cost_km2.max() / 1e3 * 1.08)
    ax_d.tick_params(axis="y", length=0)
    st.despine(ax_d, keep=("bottom",))
    head(ax_d, "d", "Places")

    cax = fig.add_axes([x3 + colw * 0.45, B - 0.098, colw * 0.52, 0.012])
    import matplotlib as mpl
    cb = mpl.colorbar.ColorbarBase(cax, cmap=st.SEQ_ORANGE,
                                   norm=mpl.colors.Normalize(0, cmax),
                                   orientation="horizontal")
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=5.8, length=2, pad=1.2, color=st.MUTED)
    cax.set_xlabel("exposure per km$^2$", fontsize=6.2, color=st.MUTED, labelpad=1.5)

    suptitle(fig, "Exposure is concentrated in a few places, in proportion to the "
             "land they added, and redistributing that land moves the total by "
             "less than a tenth")
    return save(fig, name)


HAB_ORDER = ["forest", "shrub", "grass", "sparse", "wetland",
             "cropland", "impervious", "bare", "water"]
HAB_LABEL = {"forest": "Forest", "shrub": "Shrubland", "grass": "Grassland",
             "sparse": "Sparse vegetation", "wetland": "Wetland",
             "cropland": "Cropland", "impervious": "Already impervious",
             "bare": "Bare", "water": "Water"}


def fig_form_habitat(name="fig_form_habitat_v2", landcover=True):
    st.apply_rc()
    d = cities()
    hb_ = pd.read_csv(INTERIM_DIR / "city_habitat.csv")
    hb_ = hb_[hb_.city_id.isin(set(d.city_id))].copy()
    _hcols = [f"hab_{k}_km2_scaled" for k in HAB_ORDER]
    cov = {"cities": int(hb_.city_id.nunique()),
           "covered_share": float(hb_[_hcols].to_numpy().sum()
                                  / d.urban_added_km2.sum())}

    fig_h = 3.75 if landcover else 2.30
    k_ = 3.75 / fig_h
    L, R = 0.100, 0.958
    cw = (R - L - 0.115) / 2
    y_ab = 0.5147 if landcover else 0.23
    fig = plt.figure(figsize=(PAGE_W, fig_h))
    ax_a = fig.add_axes([L, y_ab, cw, 0.400 * k_])
    ax_b = fig.add_axes([L + cw + 0.115, y_ab, cw, 0.400 * k_])
    ax_c = fig.add_axes([L, 0.1232, R - L, 0.112]) if landcover else None

    forms = ["infill", "edge", "leapfrog"]
    fc = [st.FORM_COLORS[k] for k in forms]

    dec = pd.qcut(d.urban_km2_final, 10, labels=False)
    g = d.groupby(dec)[[f"form_{k}_km2" for k in forms]].sum()
    sh = (g.T / g.sum(axis=1)).T * 100
    x = np.arange(10)
    bot = np.zeros(10)
    for k, c in zip(forms, fc):
        v = sh[f"form_{k}_km2"].values
        ax_a.bar(x, v, bottom=bot, width=0.80, color=c, edgecolor="white",
                 lw=0.5, label=k.capitalize())
        bot += v
    med = d.groupby(dec).urban_km2_final.median()
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{v:.0f}" if v >= 10 else f"{v:.1f}" for v in med],
                         fontsize=6.0, rotation=45)
    ax_a.set_xlabel("Median 2020 extent of the decile (km$^2$)", fontsize=7.2)
    ax_a.set_ylabel("Share of converted area (%)", fontsize=7.2)
    ax_a.set_ylim(0, 126); ax_a.set_yticks([0, 20, 40, 60, 80, 100])
    st.despine(ax_a)
    ax_a.legend(fontsize=6.4, ncol=3, loc="upper center", handlelength=1.1,
                columnspacing=1.2, bbox_to_anchor=(0.5, 1.01), frameon=False)
    head(ax_a, "a", "Form does not shift with size")

    y = np.log10(d.cost_per_km2.values)
    for k, c in zip(("infill", "leapfrog"), (st.FORM_COLORS["infill"],
                                             st.FORM_COLORS["leapfrog"])):
        binned_median(ax_b, d[f"share_{k}"].values * 100, y, nbin=10, color=c,
                      label=f"{k.capitalize()} share")
    ax_b.set_xlabel("Share of the city's converted area (%)", fontsize=7.2)
    ax_b.set_ylabel("Species-range exposure intensity\n(km$^2$ km$^{-2}$)",
                    fontsize=7.2)
    _logticks(ax_b, "y", [5, 10, 20, 40])
    ax_b.legend(fontsize=6.4, loc="upper right", handlelength=1.4)
    st.despine(ax_b)
    dm = pd.read_csv(TAB_DIR / "driver_model.csv").set_index("predictor")
    head(ax_b, "b", "And it barely moves exposure intensity")
    note(ax_b, "Edge share omitted: the three\nshares are compositional\n"
         f"Standardized  infill {dm.loc['form_infill_share','beta_std']:+.4f}, "
         f"leapfrog {dm.loc['form_leapfrog_share','beta_std']:+.4f}",
         "lower left", fs=5.8)

    if landcover:
        cols = [f"hab_{k}_km2_scaled" for k in HAB_ORDER]
        tot = hb_[cols].sum()
        tot.index = [f"hab_{k}_km2" for k in HAB_ORDER]
        sh = tot / tot.sum() * 100
        left = 0.0
        for k in HAB_ORDER:
            v = sh[f"hab_{k}_km2"]
            ax_c.barh([0], [v], left=[left], height=0.50,
                      color=st.HABITAT_COLORS[k], edgecolor="white", lw=0.6)
            if v > 3.0:
                ax_c.text(left + v / 2, 0, f"{v:.1f}", ha="center", va="center",
                          fontsize=6.4, color="white", fontweight="bold")
            if k == "cropland":
                crop_mid = left + v / 2
            left += v
        nat = sum(sh[f"hab_{k}_km2"] for k in
                  ("forest", "shrub", "grass", "sparse", "wetland"))
        ax_c.plot([0, nat], [0.34, 0.34], color=st.INK, lw=1.0, solid_capstyle="butt")
        ax_c.text(nat / 2, 0.375, f"natural cover, {nat:.1f}%", ha="center",
                  va="bottom", fontsize=6.6, color=st.INK)
        handles = [Rectangle((0, 0), 1, 1, facecolor=st.HABITAT_COLORS[k])
                   for k in HAB_ORDER]
        ax_c.set_xlim(0, 100); ax_c.set_ylim(-0.45, 0.46)
        ax_c.set_yticks([]); ax_c.set_xlabel("Share of converted area (%)",
                                             fontsize=7.2)
        fig.canvas.draw()
        x_mid = ax_c.transData.transform((crop_mid, 0))[0] / (fig.get_size_inches()[0] * fig.dpi)
        fig.legend(handles, [HAB_LABEL[k] for k in HAB_ORDER], fontsize=6.0, ncol=5,
                   loc="center", bbox_to_anchor=(x_mid, 0.3301),
                   bbox_transform=fig.transFigure, handlelength=1.0,
                   handleheight=0.9, columnspacing=1.0, labelspacing=0.35,
                   frameon=False)
        st.despine(ax_c, keep=("bottom",))
        head(ax_c, "c", "What was converted")
        note(ax_c, f"{cov['cities']:,} cities, "
             f"{cov['covered_share']*100:.1f}% of the converted area",
             "upper right", fs=6.0)

    suptitle(fig, "Tropical urban growth extends edges at every size, and "
             "replaces cropland as often as forest")
    return save(fig, name)

def fig_typology(name="fig_typology_v2"):
    import tropics_cluster as tc
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    st.apply_rc()
    d0, cols = tc.feature_frame()
    lab = pd.read_csv(TAB_DIR / "city_clusters.csv")
    d0 = d0.merge(lab, on="city_id")
    X = StandardScaler().fit_transform(d0[cols].to_numpy())
    pca = PCA(n_components=3, random_state=tc.SEED).fit(X)
    Z = pca.transform(X)
    load = pd.read_csv(TAB_DIR / "cluster_pca_loadings.csv", index_col=0)
    prof = pd.read_csv(TAB_DIR / "cluster_profiles.csv")
    diag = pd.read_csv(TAB_DIR / "cluster_diagnostics.csv")
    pretty = dict(tc.FEATURES)

    fig = plt.figure(figsize=(PAGE_W, 5.2))
    L, R, T, B = 0.130, 0.935, 0.865, 0.115
    wide = 0.40; narrow = (R - L - wide - 0.135 * 2) / 2 + 0.135 / 2
    ch = (T - B - 0.175) / 2
    ax_a = fig.add_axes([L, B + ch + 0.175, wide, ch])
    ax_b = fig.add_axes([L + wide + 0.145, B + ch + 0.175, R - L - wide - 0.145, ch])
    ax_c = fig.add_axes([L, B, wide, ch])
    ax_d = fig.add_axes([L + wide + 0.145, B, R - L - wide - 0.145, ch])

    for k in range(4):
        m = d0.cluster.values == k
        ax_a.scatter(Z[m, 0], Z[m, 1], s=1.8, c=CLUSTER_COLORS[k], alpha=0.22,
                     linewidths=0, rasterized=True, zorder=2)
    for k in range(4):
        m = d0.cluster.values == k
        ax_a.scatter(np.median(Z[m, 0]), np.median(Z[m, 1]), s=86, marker="o",
                     c=CLUSTER_COLORS[k], edgecolor="white", linewidth=1.5,
                     zorder=10)
        ax_a.text(np.median(Z[m, 0]), np.median(Z[m, 1]), str(k), fontsize=7.0,
                  color="white", fontweight="bold", ha="center", va="center",
                  zorder=11)
    lo = load.loc[cols, ["PC1", "PC2"]].to_numpy()
    xr = np.percentile(np.abs(Z[:, 0]), 98)
    yr = np.percentile(np.abs(Z[:, 1]), 98)
    sc = 0.86 * min(xr / np.abs(lo[:, 0]).max(), yr / np.abs(lo[:, 1]).max())
    angs = np.degrees(np.arctan2(lo[:, 1] * sc / yr, lo[:, 0] * sc / xr))
    order_a = np.argsort(angs)
    radius = np.full(len(cols), 1.09)
    for a, b in zip(order_a[:-1], order_a[1:]):
        if abs(angs[b] - angs[a]) < 14:
            radius[b] = radius[a] + 0.24
    for i, (v, (lx, ly)) in enumerate(zip(cols, lo)):
        x, y = lx * sc, ly * sc
        ax_a.annotate("", xy=(x, y), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", lw=0.8, color=st.INK,
                                      mutation_scale=7), zorder=8)
        ang = np.degrees(np.arctan2(y, x))
        ax_a.text(x * radius[i], y * radius[i], SHORT_FEATURE.get(v, v),
                  fontsize=5.6, color=st.INK, zorder=9,
                  ha="left" if abs(ang) < 90 else "right",
                  va="bottom" if y >= 0 else "top",
                  bbox=dict(fc="white", ec="none", alpha=0.80, pad=0.8))
    ax_a.set_xlim(-1.20 * xr, 1.20 * xr); ax_a.set_ylim(-1.20 * yr, 1.20 * yr)
    ax_a.axhline(0, color=st.HAIR, lw=0.5, zorder=0)
    ax_a.axvline(0, color=st.HAIR, lw=0.5, zorder=0)
    ev = pca.explained_variance_ratio_
    ax_a.set_xlabel(f"PC1 ({ev[0]*100:.0f}% of variance)", fontsize=7.2)
    ax_a.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)", fontsize=7.2)
    st.despine(ax_a)
    head(ax_a, "a", "Ordination")
    note(ax_a, f"{len(d0):,} footprints\nnine standardized features",
         "lower left", fs=5.9)

    show = ["cost_per_km2", "rarity_per_km2", "setting", "urban_km2",
            "growth_pct", "infill", "leapfrog", "narrow_share", "abs_lat"]
    rowlab = ["exposure intensity", "rarity-weighted intensity", "setting richness",
              "urban extent", "growth (%)", "infill share", "leapfrog share",
              "narrow-range share", "|latitude|"]
    M = prof[show].to_numpy(float).T
    Zs = (M - M.mean(axis=1, keepdims=True)) / np.where(
        M.std(axis=1, keepdims=True) == 0, 1, M.std(axis=1, keepdims=True))
    v = np.nanmax(np.abs(Zs))
    im = ax_b.imshow(Zs, cmap=st.DIV, vmin=-v, vmax=v, aspect="auto")
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels([f"{k}\nn = {int(prof.n[k]):,}" for k in range(4)],
                         fontsize=6.2)
    ax_b.set_yticks(range(len(show))); ax_b.set_yticklabels(rowlab, fontsize=6.2)
    ax_b.tick_params(length=0)
    for s in ax_b.spines.values():
        s.set_visible(False)
    for i in range(4):
        ax_b.add_patch(Rectangle((i - 0.5, -0.5), 1, len(show), fill=False,
                                 edgecolor=CLUSTER_COLORS[i], lw=1.4, zorder=5))
    cax = fig.add_axes([ax_b.get_position().x1 + 0.012, ax_b.get_position().y0,
                        0.008, ax_b.get_position().height * 0.55])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cax.tick_params(labelsize=5.6, length=2, color=st.MUTED)
    cax.set_ylabel("z across groups", fontsize=6.0, color=st.MUTED, labelpad=2)
    head(ax_b, "b", "What separates them")

    ax_c_y = np.arange(4)[::-1]
    wbar = 0.36
    ax_c.barh(ax_c_y + wbar / 2, prof.cost_share_pct, height=wbar,
              color=[CLUSTER_COLORS[k] for k in range(4)], edgecolor="white",
              lw=0.5)
    ax_c.barh(ax_c_y - wbar / 2, prof.added_share_pct, height=wbar,
              color=[CLUSTER_COLORS[k] for k in range(4)], edgecolor="white",
              lw=0.5, alpha=0.45)
    for k in range(4):
        ax_c.text(prof.cost_share_pct[k] + 1.2, ax_c_y[k] + wbar / 2,
                  f"{prof.cost_share_pct[k]:.0f}% cost", va="center", fontsize=5.9,
                  color=st.MUTED)
        ax_c.text(prof.added_share_pct[k] + 1.2, ax_c_y[k] - wbar / 2,
                  f"{prof.added_share_pct[k]:.0f}% land", va="center", fontsize=5.9,
                  color=st.MUTED)
    ax_c.set_yticks(ax_c_y)
    import textwrap
    ax_c.set_yticklabels(
        [f"{k}  " + "\n     ".join(textwrap.wrap(tc.NAMES[k], 17))
         for k in range(4)], fontsize=5.8, linespacing=1.2)
    ax_c.set_xlabel("Share of the pan-tropical total (%)", fontsize=7.2)
    ax_c.set_xlim(0, max(prof.cost_share_pct.max(), prof.added_share_pct.max()) * 1.34)
    ax_c.tick_params(axis="y", length=0)
    st.despine(ax_c, keep=("bottom",))
    head(ax_c, "c", "What each group carries")

    ax_d.plot(diag.k, diag.silhouette, "-o", ms=3.4, lw=1.2, color=st.BLUE,
              mec="white", mew=0.6, label="silhouette")
    ax_d.set_xlabel("Number of groups $k$", fontsize=7.2)
    ax_d.set_ylabel("Silhouette", fontsize=7.2, color=st.BLUE)
    ax_d.tick_params(axis="y", colors=st.BLUE)
    ax2 = ax_d.twinx()
    ax2.plot(diag.k, diag.calinski / 1e3, "-s", ms=3.2, lw=1.2, color=st.ORANGE,
             mec="white", mew=0.6, label="Calinski-Harabasz")
    ax2.set_ylabel("Calinski-Harabasz (10$^3$)", fontsize=7.2, color=st.ORANGE)
    ax2.tick_params(axis="y", colors=st.ORANGE)
    for s in ("top",):
        ax_d.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    ax_d.axvline(4, color=NULL, lw=0.9, ls=(0, (4, 3)), zorder=0)
    rng_s = diag.silhouette.max() - diag.silhouette.min()
    ax_d.text(4, ax_d.get_ylim()[1], " k = 4 used", fontsize=6.2, color=NULL,
              va="top", ha="left")
    ax_d.text(0.02, 0.03, f"criteria are near-flat (silhouette range {rng_s:.3f});\n"
              "four groups chosen for interpretability",
              transform=ax_d.transAxes, fontsize=5.8, color=st.MUTED,
              va="bottom", ha="left", zorder=12,
              bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.6))
    kb_s = int(diag.loc[diag.silhouette.idxmax(), "k"])
    kb_c = int(diag.loc[diag.calinski.idxmax(), "k"])
    head(ax_d, "d", "How many groups")
    note(ax_d, f"silhouette peaks at k = {kb_s}\nCalinski-Harabasz at k = {kb_c}",
         "upper right", fs=5.9)

    suptitle(fig, "Footprints separate by setting, rarity and growth rate, "
             "and not by size")
    return save(fig, name)


REALMS = ["Indomalayan", "Neotropic", "Afrotropic", "Australasia"]
REALM_COLORS = ["#B0184E", "#2B62C4", "#C05A0F", "#0F8A6E"]


def _violin(ax, groups, values, colors, log=True):
    parts = ax.violinplot(values, showextrema=False, widths=0.82)
    for b, c in zip(parts["bodies"], colors):
        b.set_facecolor(c); b.set_alpha(0.42); b.set_edgecolor(c)
        b.set_linewidth(0.6)
    for i, v in enumerate(values, start=1):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=st.INK, lw=2.4, solid_capstyle="round",
                zorder=5)
        ax.plot([i], [med], "o", ms=3.2, color="white", mec=st.INK, mew=0.9,
                zorder=6)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, fontsize=6.2, rotation=28, ha="right")
    st.despine(ax)


def fig_distributions(name="fig_distributions_v2"):
    st.apply_rc()
    d = cities()
    fig = plt.figure(figsize=(PAGE_W, 2.9))
    L, R, B, T = 0.100, 0.958, 0.200, 0.835
    cw = (R - L - 2 * 0.105) / 3
    axes = [fig.add_axes([L + i * (cw + 0.105), B, cw, T - B]) for i in range(3)]

    ax = axes[0]
    vals = [np.log10(d[d.realm == r].cost_per_km2.values) for r in REALMS]
    _violin(ax, [r for r in REALMS], vals, REALM_COLORS)
    _logticks(ax, "y", [3, 10, 30, 100])
    ax.set_ylabel("Species-range exposure intensity\n(km$^2$ km$^{-2}$)", fontsize=7.4)
    head(ax, "a", "By realm")
    note(ax, "\n".join(f"{r[:11]} n = {int((d.realm == r).sum()):,}"
                       for r in REALMS), "upper left", fs=5.6)

    ax = axes[1]
    top = d.biome.value_counts().index[:5]
    vals = [np.log10(d[d.biome == b].cost_per_km2.values) for b in top]
    _violin(ax, [BIOME_SHORT.get(b, b) for b in top], vals,
            [st.SEQ_GREEN(x) for x in np.linspace(0.35, 0.85, len(top))])
    _logticks(ax, "y", [3, 10, 30, 100])
    ax.set_ylabel("Species-range exposure intensity\n(km$^2$ km$^{-2}$)", fontsize=7.4)
    head(ax, "b", "By biome")

    ax = axes[2]
    for r, c in zip(REALMS, REALM_COLORS):
        v = np.sort(d[d.realm == r].cost_per_km2.values)
        ax.plot(np.log10(v), np.arange(1, len(v) + 1) / len(v) * 100, color=c,
                lw=1.4, label=r)
    _logticks(ax, "x", [3, 10, 30, 100])
    ax.set_xlabel("Species-range exposure intensity (km$^2$ km$^{-2}$)", fontsize=7.4)
    ax.set_ylabel("Cities at or below (%)", fontsize=7.4)
    ax.legend(fontsize=6.0, loc="lower right", handlelength=1.3, borderpad=0.2)
    st.despine(ax)
    head(ax, "c", "Cumulative")

    suptitle(fig, "The cost distribution differs by realm and by biome, "
             "and overlaps heavily in both")
    return save(fig, name)


PAIRS_SPEC = [
    ("urban_km2_base", "2000 impervious extent (km$^2$)", True),
    ("setting_richness", "Setting richness", True),
    ("growth_pct", "Land added, % of 2000 extent", True),
    ("share_infill", "Infill share", False),
    ("share_leapfrog", "Leapfrog share", False),
    ("abs_lat", "|latitude| (\u00b0)", False),
]


def fig_pairs(name="fig_pairs_v2"):
    st.apply_rc()
    d = cities()
    y = np.log10(d.cost_per_km2.values)
    fig = plt.figure(figsize=(PAGE_W, 4.5))
    L, R, B, T = 0.100, 0.958, 0.108, 0.945
    cw = (R - L - 2 * 0.095) / 3
    ch = (T - B - 0.215) / 2
    for i, (col, lab, logx) in enumerate(PAIRS_SPEC):
        r, c = divmod(i, 3)
        ax = fig.add_axes([L + c * (cw + 0.095), B + (1 - r) * (ch + 0.215),
                           cw, ch])
        v = d[col].values.astype(float)
        if logx:
            v = np.log10(np.where(v > 0, v, np.nan))
        hb = hexpanel(ax, v, y, gridsize=34)
        binned_median(ax, v, y, nbin=10)
        nn = int(np.isfinite(v) & np.isfinite(y))if False else int((np.isfinite(v) & np.isfinite(y)).sum())
        ax.text(0.97, 0.04, f"n = {nn:,}", transform=ax.transAxes, fontsize=5.6,
                color=st.MUTED, ha="right", va="bottom")
        ax.set_xlabel(lab, fontsize=7.0)
        if c == 0:
            ax.set_ylabel("Species-range exposure intensity\n(km$^2$ km$^{-2}$)", fontsize=7.4)
        _logticks(ax, "y", [3, 10, 30, 100])
        if logx:
            hi = np.nanmax(v)
            ticks = [t for t in (1, 100, 10000) if np.log10(t) <= hi + 0.3]
            _logticks(ax, "x", ticks)
        head(ax, "abcdef"[i], lab)
    bar_w = 0.20
    mid_x = L + (cw + 0.095) + (cw - bar_w) / 2
    bar_y = B + ch + 0.0846
    cax = fig.add_axes([mid_x, bar_y, bar_w, 0.014])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=5.8, length=2, pad=1.2, color=st.MUTED)
    cax.text(0.5, 1.9, "cities per bin", transform=cax.transAxes, fontsize=6.2,
             color=st.MUTED, va="bottom", ha="center")
    suptitle(fig, "Species-range exposure intensity against every candidate "
             "predictor, on a common vertical scale")
    return save(fig, name)


def fig_regimes(name="fig_regimes_v2"):
    st.apply_rc()
    rr = pd.read_csv(TAB_DIR / "regime_fits_realm.csv")
    rb = pd.read_csv(TAB_DIR / "regime_fits_biome.csv")
    fig = plt.figure(figsize=(PAGE_W, 3.05))
    L, R, B, T = 0.212, 0.976, 0.225, 0.830
    GAP = 0.215
    cw = (R - L - GAP) / 2
    ax1 = fig.add_axes([L, B, cw, T - B])
    ax2 = fig.add_axes([L + cw + GAP, B, cw, T - B])
    for ax, df, letter, title in ((ax1, rr, "a", "By realm"),
                                  (ax2, rb, "b", "By biome")):
        lab = [f"{BIOME_SHORT.get(g, g)}  n = {int(n):,}"
               for g, n in zip(df.group, df.n)]
        forest(ax, lab, df.beta, df.ci_lower, df.ci_upper,
               highlight=(df.ci_lower > 0) | (df.ci_upper < 0), fs=6.4)
        ax.set_xlabel("Size coefficient $\\beta$", fontsize=7.4)
        head(ax, letter, title)
    suptitle(fig, "The size exponent is small in every region fitted on its "
             "own, and resolved in few")
    return save(fig, name)


ISO_NAME = {
    "IDN": "Indonesia", "IND": "India", "BRA": "Brazil", "CHN": "China",
    "VNM": "Viet Nam", "MYS": "Malaysia", "THA": "Thailand", "NGA": "Nigeria",
    "MEX": "Mexico", "MMR": "Myanmar", "PHL": "Philippines", "SDN": "Sudan",
    "SGP": "Singapore",
    "GHA": "Ghana", "COD": "DR Congo", "TZA": "Tanzania", "ETH": "Ethiopia",
    "KEN": "Kenya", "PER": "Peru", "VEN": "Venezuela", "COL": "Colombia",
    "BGD": "Bangladesh", "PAK": "Pakistan", "LKA": "Sri Lanka",
    "KHM": "Cambodia", "LAO": "Laos", "AGO": "Angola", "UGA": "Uganda",
    "CMR": "Cameroon", "CIV": "Cote d'Ivoire", "MDG": "Madagascar",
    "ECU": "Ecuador", "BOL": "Bolivia", "GTM": "Guatemala", "CUB": "Cuba",
    "PNG": "Papua New Guinea", "AUS": "Australia", "ZAF": "South Africa",
    "MOZ": "Mozambique", "ZMB": "Zambia", "SEN": "Senegal", "MWI": "Malawi",
    "ZWE": "Zimbabwe", "BFA": "Burkina Faso", "MLI": "Mali", "NER": "Niger",
    "TCD": "Chad", "SOM": "Somalia", "YEM": "Yemen", "OMN": "Oman",
    "SAU": "Saudi Arabia", "NPL": "Nepal", "TWN": "Taiwan",
}


def iso_names(codes):
    return [ISO_NAME.get(str(c), str(c)) for c in codes]


def fig_countries(name="fig_countries_v2"):
    st.apply_rc()
    d = cities()
    fig = plt.figure(figsize=(PAGE_W, 4.3))
    L, R, B, T = 0.130, 0.958, 0.170, 0.850
    cw = (R - L - 0.135) / 2
    ax_a = fig.add_axes([L, B, cw, T - B])
    ax_b = fig.add_axes([L + cw + 0.135, B, cw, T - B])

    ct = (d.groupby("country").agg(cost=("cost_km2", "sum"),
                                   add=("urban_added_km2", "sum")).reset_index())
    ct["per"] = ct.cost / ct["add"]
    top = ct.nlargest(20, "cost").country.tolist()
    sub = d[d.country.isin(top)]
    piv = (sub.assign(bs=sub.biome.map(BIOME_SHORT))
              .pivot_table(index="country", columns="bs", values="cost_km2",
                           aggfunc="sum").fillna(0.0))
    piv = piv.loc[ct.set_index("country").loc[top].sort_values("cost").index]
    order = piv.sum().sort_values(ascending=False).index.tolist()
    cols = [st.SEQ_GREEN(x) for x in np.linspace(0.85, 0.22, len(order))]
    y = np.arange(len(piv))
    left = np.zeros(len(piv))
    for k, c in zip(order, cols):
        v = piv[k].values / 1e3
        ax_a.barh(y, v, left=left, height=0.72, color=c, edgecolor="white",
                  lw=0.4, label=k)
        left += v
    ax_a.set_yticks(y); ax_a.set_yticklabels(iso_names(piv.index), fontsize=6.3)
    ax_a.set_xlabel("Converted range area (10$^3$ km$^2$)", fontsize=7.4)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(fontsize=5.6, ncol=2, loc="lower right", handlelength=1.0,
                handleheight=0.85, columnspacing=0.9, borderpad=0.25,
                frameon=False)
    st.despine(ax_a, keep=("bottom",))
    head(ax_a, "a", "National totals by biome")

    tot_cost = d.cost_km2.sum(); tot_add = d.urban_added_km2.sum()
    cc = (sub.groupby("country").agg(cost=("cost_km2", "sum"),
                                     add=("urban_added_km2", "sum"))
             .reindex(piv.index))
    s_cost = cc.cost / tot_cost * 100
    s_add = cc["add"] / tot_add * 100
    wbar = 0.36
    ax_b.barh(y + wbar / 2, s_cost, height=wbar, color=NULL, edgecolor="white",
              lw=0.4, label="converted range area")
    ax_b.barh(y - wbar / 2, s_add, height=wbar, color=st.BLUE, edgecolor="white",
              lw=0.4, label="urban land added")
    ax_b.set_yticks(y); ax_b.set_yticklabels(iso_names(piv.index), fontsize=6.3)
    ax_b.set_xlabel("Share of the pan-tropical total (%)", fontsize=7.4)
    ax_b.tick_params(axis="y", length=0)
    ax_b.legend(fontsize=6.2, loc="lower right", handlelength=1.1,
                handleheight=0.85, borderpad=0.25, frameon=False)
    st.despine(ax_b, keep=("bottom",))
    head(ax_b, "b", "Converted-range share against land share")

    suptitle(fig, "The twenty countries carrying the largest converted "
             "range area, and what that area is made of")
    return save(fig, name)
