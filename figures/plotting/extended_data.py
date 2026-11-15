"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation

import supplementary_data as sd
import tropics_style as st
import tropics_panels as tp
from tropics_config import TABLE_DIR

CCOL = {"CR": "#8B0F35", "EN": "#C05A0F", "VU": "#D9A62E"}
CLASSES = [("Amphibia", "Amphibians"), ("Reptilia", "Reptiles"), ("Mammalia", "Mammals"), ("Aves", "Birds")]
RULES = [("within", "Wholly inside"), ("centroid", "Center inside"), ("touches", "Touches"), ("all_three", "All three")]
RULE_COL = ["#C9CED8", "#7D8BA2", "#2B3A52", "#B0184E"]
GREY = "#8C8884"
INVENTORY = "#C24A10"
INV_SCALE = 0.45
LABEL_MOVES = {"Mexico City": (-106.0, 3.0, "right"), "Lagos": (-23.0, -1.0, "center")}
RELPOS = {"right": (1.0, 0.5), "left": (0.0, 0.5), "center": (0.5, 0.5)}
MAP_LABEL_MOVES = {"Mexico City": (-112.0, -6.0, "right")}


def capture(draw, *a, **k):
    prev = st.HELD
    held = st.hold()
    draw(*a, **k)
    st.HELD = prev
    return held[-1]


def extend(fig, extra):
    W, H = fig.get_size_inches()
    k, off = H / (H + extra), extra / (H + extra)
    anchors = []
    for leg in fig.legends:
        b = leg.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        anchors.append((leg, b))
    for ax in fig.axes:
        for which in ("original", "active"):
            p = ax.get_position(original=which == "original")
            ax._set_position([p.x0, p.y0 * k + off, p.width, p.height * k], which=which)
    texts = list(fig.texts) + [t for ax in fig.axes for t in ax.texts]
    for t in texts:
        if t.get_transform() is fig.transFigure and not isinstance(t, Annotation):
            x, y = t.get_position()
            t.set_position((x, y * k + off))
    for ln in fig.lines:
        if ln.get_transform() is fig.transFigure:
            ln.set_ydata(np.asarray(ln.get_ydata(), float) * k + off)
    for p in fig.patches:
        if isinstance(p, Rectangle) and p.get_transform() is fig.transFigure:
            p.set_y(p.get_y() * k + off)
            p.set_height(p.get_height() * k)
    fig.set_size_inches(W, H + extra)
    for leg, b in anchors:
        leg.set_bbox_to_anchor((b.x0, b.y0 * k + off, b.width, b.height * k), transform=fig.transFigure)
    return W, H + extra


def relayout(fig, H, rects, letters):
    W, H0 = fig.get_size_inches()
    old = []
    for ax in fig.axes:
        p = ax.get_position()
        old.append((p.x0 * W, p.y0 * H0, p.width * W, p.height * H0))
    fig.set_size_inches(W, H)
    for t in list(fig.texts):
        t.remove()
    for ax in fig.axes:
        for t in list(ax.texts):
            if t.get_fontweight() in ("bold", 700) and len(t.get_text().strip()) == 1:
                t.remove()
    for i, ax in enumerate(fig.axes):
        r = rects.get(i)
        if r is None:
            continue
        if r[0] == "dy":
            x, y, w, h = old[i]
            r = (x, y + r[1], w, h)
        elif r[0] == "with":
            j, dx, dy = r[1], r[2], r[3]
            x, y, w, h = old[i]
            r = (rects[j][0] + dx, rects[j][1] + dy, w, h)
        x, y, w, h = r
        ax.set_position([x / W, y / H, w / W, h / H])
    for x, y, s in letters:
        letter(fig, x, y, s)


def axes(fig, x, y, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def letter(fig, x, y, s, head=None):
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, s, fontsize=st.MIN_PT + 1, fontweight="bold", va="bottom", ha="left", color=st.INK)
    if head:
        fig.text((x + 0.15) / W, y / H, head, fontsize=8, fontweight="bold", va="bottom", ha="left", color=st.INK)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_reached_rules(ax):
    d = pd.read_csv(TABLE_DIR / "species_accounting_by_class.csv").set_index("Class")
    rows = CLASSES[::-1] + [("Total", "All classes")]
    y = np.arange(len(rows))[::-1]
    for yi, (key, lab) in zip(y, rows):
        r = d.loc[key]
        n = r["Threatened mapped"]
        lo, mid, hi = (100 * r[c] / n for c in ("Threatened reached, within rule", "Threatened reached, center rule",
                                                 "Threatened reached, touch rule"))
        ax.barh(yi, mid, height=0.55, color="#C9CED8", lw=0)
        ax.plot([lo, hi], [yi, yi], color=st.INK, lw=1.0, solid_capstyle="butt")
        ax.text(101, yi, f"{int(r['Threatened reached, center rule']):,} of {int(n):,}", va="center", ha="left",
                fontsize=st.MIN_PT, color=st.INK)
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Threatened species reached (%)")
    ax.tick_params(axis="y", length=0)
    clean(ax)


def panel_top_species(ax, n=15):
    d = pd.read_csv(TABLE_DIR / "species_boundary_shares.csv").nlargest(n, "share_centroid").iloc[::-1]
    y = np.arange(len(d))
    ax.hlines(y, 100 * d.share_min, 100 * d.share_max, color=GREY, lw=1.0)
    ax.scatter(100 * d.share_centroid, y, s=16, c=[CCOL[c] for c in d.category], zorder=3, lw=0)
    ax.set_yticks(y, d.sci)
    for t in ax.get_yticklabels():
        t.set_fontstyle("italic")
    ax.set_xlim(0, 42)
    ax.set_xlabel("Share of mapped range converted (%)")
    ax.tick_params(axis="y", length=0)
    hl = [Line2D([], [], marker="o", ls="", color=CCOL[c], ms=4) for c in ("CR", "EN", "VU")]
    hl.append(Line2D([], [], color=GREY, lw=1.0))
    ax.legend(hl, ["CR", "EN", "VU", "Range across rules"], loc="lower right", frameon=False, fontsize=st.MIN_PT,
              handlelength=1.2, borderaxespad=0.1, markerfirst=False)
    clean(ax)


def panel_class_concentration(ax):
    c = sd.concentration()
    c = c[c.grouping == "taxonomic class"].set_index("group")
    x = np.arange(len(CLASSES))
    top = [100 * c.loc[k, "top1_median"] for k, _ in CLASSES]
    gt = [100 * c.loc[k, "share_top1_gt50"] for k, _ in CLASSES]
    ax.bar(x - 0.19, top, width=0.36, color="#7D8BA2", lw=0, label="Median largest-agglomeration share")
    ax.bar(x + 0.19, gt, width=0.36, color="#C9CED8", lw=0, label="Species with largest share above 50%")
    ax.set_xticks(x, [lab for _, lab in CLASSES])
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Percent")
    ax.legend(loc="upper right", frameon=False, fontsize=st.MIN_PT, handlelength=1.0, ncol=1, borderaxespad=0.1)
    clean(ax)


def panel_rule_concentration(ax):
    r = pd.DataFrame(sd.rule_sensitivity()).set_index("band")
    bands = [(">=1%", "≥1% of range converted"), (">=5%", "≥5% of range converted")]
    w = 0.2
    for i, (b, lab) in enumerate(bands):
        for j, (k, rl) in enumerate(RULES):
            ax.bar(i + (j - 1.5) * w, r.loc[b, k], width=w * 0.92, color=RULE_COL[j], lw=0,
                   label=rl if i == 0 else None)
        ax.text(i + 1.5 * w, r.loc[b, "all_three"] + 18,
                f"{100 * r.loc[b, 'stable_top1_median']:.1f}%", ha="center", va="bottom", fontsize=st.MIN_PT,
                color=RULE_COL[3])
    ax.set_xticks(range(len(bands)), [lab for _, lab in bands])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 820)
    ax.set_ylabel("Species")
    ax.legend(loc="upper right", frameon=False, fontsize=st.MIN_PT, handlelength=1.0, borderaxespad=0.1,
              title="Boundary rule", title_fontsize=st.MIN_PT, alignment="left")
    clean(ax)


def panel_cv(ax):
    cv = sd.spatial_cv()
    labs = ["Size", "Setting", "Size +\nsetting", "+ biome", "+ country"]
    for i, m in enumerate(sd.CV_MODELS):
        v = cv[m].to_numpy()
        ax.scatter(np.full(v.size, i) + np.linspace(-0.18, 0.18, v.size), v, s=9, color="#7D8BA2", lw=0, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=st.INK, lw=1.2)
    ax.axhline(0, color=st.MUTED, lw=0.6)
    ax.set_xticks(range(len(labs)), labs)
    ax.set_ylim(-0.05, 0.85)
    ax.set_ylabel("Held-out $R^2$")
    clean(ax)


def panel_cross_taxon(ax):
    d = sd.cross_taxon()
    names = {"birds": "birds", "mammals": "mammals", "herps": "amphibians and reptiles"}
    y = np.arange(len(d))[::-1]
    ax.errorbar(d.setting_coefficient, y, xerr=1.96 * d.se, fmt="o", color="#2B3A52", ms=4, lw=1.0, capsize=0)
    for yi, r in zip(y, d.itertuples()):
        ax.text(1.2, yi, fmt(r.size_exponent), va="center", ha="left", fontsize=st.MIN_PT, color=st.MUTED)
    ax.text(1.2, y.max() + 0.75, "Size\nexponent", va="bottom", ha="left", fontsize=st.MIN_PT, color=st.MUTED)
    ax.set_yticks(y, [names[s].capitalize() for s in d.setting_measured_on])
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.5, len(d) - 0.2)
    ax.set_xlabel("Setting coefficient")
    ax.tick_params(axis="y", length=0)
    clean(ax)


def panel_decomposition(ax):
    d = sd.decomposition()
    d = d[d.grouping == "species subset"].iloc[::-1]
    y = np.arange(len(d))
    h = 0.26
    for j, (c, lab, col) in enumerate([("beta_unweighted", "Unweighted", "#C9CED8"),
                                       ("beta_rarity", "Rarity weighted", "#2B3A52"),
                                       ("beta_mean_range", "Mean global range", "#B0184E")]):
        ax.barh(y + (1 - j) * h, d[c], height=h * 0.92, color=col, lw=0, label=lab)
    ax.axvline(0, color=st.MUTED, lw=0.6)
    ax.set_yticks(y, d.subset)
    ax.set_xlabel("Size exponent")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.3), ncol=3, frameon=False, fontsize=st.MIN_PT,
              handlelength=1.0, columnspacing=0.8, borderaxespad=0)
    clean(ax)


def panel_region_share(ax):
    d = json.loads((TABLE_DIR / "region_window_overlap.json").read_text())
    regs = ["Eastern Arc and coastal Tanzania", "Northern South America", "Mesoamerica",
            "Sumatra and the Malay Peninsula", "Mainland Southeast Asia"]
    labs = ["Eastern Arc and\ncoastal Tanzania", "Northern South\nAmerica", "Mesoamerica",
            "Sumatra and the\nMalay Peninsula", "Mainland\nSoutheast Asia"]
    v = [d[r]["share_pct"] for r in regs][::-1]
    y = np.arange(len(regs))
    ax.barh(y, v, height=0.6, color="#7C5BA6", lw=0)
    for yi, vi in zip(y, v):
        ax.text(vi + 0.6, yi, f"{vi:.1f}", va="center", ha="left", fontsize=st.MIN_PT, color=st.INK)
    ax.axvline(d["tropics"]["share_pct"], color=st.MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(d["tropics"]["share_pct"] + 0.4, len(regs) - 0.45, "Tropics", ha="left", va="center",
            fontsize=st.MIN_PT, color=st.MUTED)
    ax.set_yticks(y, labs[::-1])
    ax.set_xlim(0, 32)
    ax.set_xlabel("New surface over narrow ranges (%)")
    ax.tick_params(axis="y", length=0)
    clean(ax)


def panel_two_part_coefficients(ax):
    c = sd.two_part_coefficients()
    terms = ["log10 year 2000 extent", "log10 setting richness + 1", "log10 urban land added"]
    tl = {"log10 year 2000 extent": "Size", "log10 setting richness + 1": "Setting",
          "log10 urban land added": "Land added"}
    rows = [(p, t) for p in ("occurrence", "intensity") for t in terms]
    y = np.arange(len(rows))[::-1]
    for spec, dy, col in (("B", 0.14, "#2B3A52"), ("A", -0.14, "#B0184E")):
        for yi, (p, t) in zip(y, rows):
            r = c[(c.specification == spec) & (c.part == p) & (c.term == t)]
            if len(r):
                r = r.iloc[0]
                ax.errorbar(r.estimate, yi + dy, xerr=1.96 * r.se, fmt="o", color=col, ms=3.6, lw=1.0, capsize=0)
    ax.axvline(0, color=st.MUTED, lw=0.6)
    ax.set_yticks(y, [f"{p.capitalize()}, {tl[t].lower()}" for p, t in rows])
    ax.set_xlabel("Coefficient with 95% interval")
    ax.tick_params(axis="y", length=0)
    hl = [Line2D([], [], marker="o", color=c_, lw=1.0, ms=3.6) for c_ in ("#2B3A52", "#B0184E")]
    ax.legend(hl, ["Size and setting", "With land added"], loc="lower right", frameon=False, fontsize=st.MIN_PT,
              handlelength=1.2, borderaxespad=0.1)
    clean(ax)


def panel_two_part_validation(ax):
    v = sd.two_part_validation()
    metrics = [("auc", "AUC, occurrence"), ("cal_slope", "Calibration slope"), ("brier", "Brier score"),
               ("intensity_r2_log10", "$R^2$, intensity"), ("combined_r2", "$R^2$, combined")]
    y = np.arange(len(metrics))[::-1]
    for spec, dy, col in (("B", 0.14, "#2B3A52"), ("A", -0.14, "#B0184E")):
        s = v[v.specification == spec].set_index("metric")
        ax.scatter([s.loc[m, "value"] for m, _ in metrics], y + dy, s=14, color=col, lw=0, zorder=3)
    ax.axvline(0, color=st.MUTED, lw=0.6)
    ax.set_yticks(y, [lab for _, lab in metrics])
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel("Value on held-out blocks")
    ax.tick_params(axis="y", length=0)
    clean(ax)


def world_map(fig, rect):
    import fig_headline as fh
    import tropics_main_figures as f4
    import tropics_robinson as rb
    ax = fig.add_axes(rect)
    rb.draw_globe(ax, zorder=0)
    rb.frame(ax)
    ax.set_ylim(*fh.world_ylim())
    f4._tropic_lines(ax)
    return ax


def window_outlines(ax, num_ax=None, num_y=None):
    import matplotlib.patheffects as pe
    import fig_headline as fh
    import tropics_frame as tf
    import tropics_robinson as rb
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    for k, r in enumerate(tf.REGIONS, start=1):
        name, lon0, lon1, lat0, lat1, col = r[:6]
        lo = np.concatenate([np.linspace(lon0, lon1, 40), np.full(40, lon1), np.linspace(lon1, lon0, 40),
                             np.full(40, lon0)])
        la = np.concatenate([np.full(40, lat1), np.linspace(lat1, lat0, 40), np.full(40, lat0),
                             np.linspace(lat0, lat1, 40)])
        bx, by = rb.project(lo, la)
        ax.plot(bx, by, color=col, lw=1.0, zorder=8, solid_joinstyle="round")
        tx, ty = rb.project(lon0, lat1)
        ty = num_y(float(ty)) if num_y else ty
        (num_ax or ax).annotate(str(k), xy=(float(tx), float(ty)), xytext=(-2.0, 1.5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=fh.PT + 1, fontweight="bold", color=col, zorder=9,
                    path_effects=halo)


def window_strip(fig, rect, draw, below):
    import matplotlib.patheffects as pe
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    from matplotlib.transforms import offset_copy
    from shapely.geometry import box
    import fig_headline as fh
    import tropics_frame as tf
    import tropics_robinson as rb
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    W, H = fig.get_size_inches()
    _, land = rb._load()
    for k, r in enumerate(tf.REGIONS, start=1):
        name, lon0, lon1, lat0, lat1, col = r[:6]
        x_in = fh.STRIP_X0 + (k - 1) * (fh.STRIP_IN + fh.STRIP_GAP)
        y_in = rect[1] * H - fh.STRIP_DROP - fh.STRIP_IN
        ia = fig.add_axes([x_in / W, y_in / H, fh.STRIP_IN / W, fh.STRIP_IN / H])
        ia.set_zorder(10)
        clip = land.intersection(box(lon0 - 1, lat0 - 1, lon1 + 1, lat1 + 1))
        ia.add_collection(PatchCollection([Polygon(q, closed=True) for q in rb._rings(clip)],
                                          facecolors=rb.TOP_FACE, edgecolors=rb.BORDER, linewidths=0.3, zorder=1))
        draw(ia)
        ia.set_xlim(lon0, lon1)
        ia.set_ylim(lat0, lat1)
        ia.set_xticks([])
        ia.set_yticks([])
        for sp in ia.spines.values():
            sp.set_color(col)
            sp.set_linewidth(1.1)
        ia.text(0.0, 1.03, f"{k}  {name}", transform=ia.transAxes, ha="left", va="bottom", fontsize=fh.PT,
                color=col, zorder=9, path_effects=halo)
        tf.corner_labels(ia, lon0, lon1, lat0, lat1)
        off = offset_copy(ia.transAxes, fig=fig, x=0, y=-13.5, units="points")
        ia.text(0.5, 0.0, below(lon0, lon1, lat0, lat1), transform=off, ha="center", va="top", fontsize=fh.PT,
                color=st.INK)
        if k == 1:
            km_per_deg = 111.32 * np.cos(np.radians(0.5 * (lat0 + lat1)))
            bx0 = lon0 + 0.06 * (lon1 - lon0)
            by = lat0 + 0.07 * (lat1 - lat0)
            ia.plot([bx0, bx0 + 500.0 / km_per_deg], [by, by], color=st.INK, lw=1.2, solid_capstyle="butt",
                    zorder=9)
            ia.text(bx0, by + 0.03 * (lat1 - lat0), "500 km", ha="left", va="bottom", fontsize=fh.PT - 0.5,
                    color=st.INK, zorder=9, path_effects=halo)


def map_layout(low, right=0.0):
    import fig_headline as fh
    W = tp.PAGE_W
    head, ml, mr = 0.17, 0.08, 0.08
    map_w = W - ml - mr - right
    map_h = map_w * fh.world_aspect()
    strip_h = fh.STRIP_DROP + fh.STRIP_IN + 0.40
    fig = plt.figure(figsize=(W, head + map_h + strip_h + low))
    H = fig.get_size_inches()[1]
    fig.text(ml / W, 1 - 0.01 / H, "a", fontsize=st.MIN_PT + 1.2, fontweight="bold", va="top", ha="left",
             color=st.INK)
    return fig, [ml / W, (low + strip_h) / H, map_w / W, map_h / H], map_h


def robinson(draw):
    import tropics_main_figures as f4
    old = f4.PROJECTION
    f4.PROJECTION = "robinson"
    try:
        return draw()
    finally:
        f4.PROJECTION = old


def panel_inventory(fig, rect):
    import matplotlib.patheffects as pe
    import fig_headline as fh
    import tropics_citymaps as cm
    import tropics_frame as tf
    import tropics_robinson as rb
    d = cm.cities(sample=False).sort_values("urban_km2_final")
    area = INV_SCALE * fh.marker_area(d.urban_km2_final.values)
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]

    def draw():
        ax = world_map(fig, rect)
        x, y = rb.project(d.lon.values, d.lat.values)
        ax.scatter(x, y, s=area, c=INVENTORY, linewidths=0.0, zorder=6, rasterized=True)
        for k, km2 in enumerate((10, 100, 1000)):
            kx, ky = rb.project(-134.0 + 19.0 * k, -45.0)
            ax.scatter([kx], [ky], s=INV_SCALE * fh.marker_area(np.array([km2])), c=INVENTORY, linewidths=0.0,
                       zorder=6)
            tx, ty = rb.project(-134.0 + 19.0 * k, -53.0)
            ax.text(tx, ty, f"{km2:,}", ha="center", va="center", fontsize=fh.PT, color=st.INK, zorder=7)
        tx, ty = rb.project(-146.0, -36.5)
        ax.text(tx, ty, "Urban extent, 2020 (km$^2$)", ha="left", va="center", fontsize=fh.PT, color=st.INK,
                zorder=7)
        for name, lon, lat, tlon, tlat, ha in fh.LABELS:
            tlon, tlat, ha = LABEL_MOVES.get(name, (tlon, tlat, ha))
            q = d[(abs(d.lon - lon) < 1.2) & (abs(d.lat - lat) < 1.2)].sort_values("urban_km2_final").iloc[-1]
            px, py = rb.project(q.lon, q.lat)
            qx, qy = rb.project(tlon, tlat)
            ax.annotate(f"{name}, {q.urban_km2_final:,.0f}", xy=(px, py), xytext=(qx, qy), ha=ha, va="center",
                        fontsize=fh.PT, color=st.INK, zorder=9, path_effects=halo,
                        arrowprops=dict(arrowstyle="-", color=st.INK, lw=0.45, shrinkA=1.5, shrinkB=1.0,
                                        relpos=RELPOS[ha]))
        tf.robinson_degree_labels(ax, lat_side="right", lat_inside=True)
        window_outlines(ax)
    robinson(draw)

    def count(lon0, lon1, lat0, lat1):
        n = int(((d.lon >= lon0) & (d.lon <= lon1) & (d.lat >= lat0) & (d.lat <= lat1)).sum())
        return f"{n:,} agglomerations"
    window_strip(fig, rect, lambda ia: ia.scatter(d.lon.values, d.lat.values, s=1.15 * area, c=INVENTORY,
                                                  linewidths=0.0, zorder=6, rasterized=True), count)


def ed_fig1(name):
    import fig_temporal_size as fts
    st.apply_rc()
    fig, rect, _ = map_layout(2.15)
    panel_inventory(fig, rect)
    g = fts.classes4()
    xs = [0.55, 2.95, 5.15]
    ab = axes(fig, xs[0], 0.42, 1.75, 1.35)
    hl = fts.panel_ranksize(ab)
    ab.set_ylim(1.5e-3, 4e4)
    ab.set_yticks([1, 10, 100, 1e3, 1e4])
    ab.legend(*hl, loc="lower left", ncol=2, columnspacing=0.8, frameon=False, handlelength=1.2, fontsize=st.MIN_PT,
              title="Year   $\\alpha$ (≥50 km²)", title_fontsize=st.MIN_PT, alignment="left", borderaxespad=0.1)
    ac = axes(fig, xs[1], 0.42, 1.55, 1.35)
    fts.panel_gibrat(ac)
    ad = axes(fig, xs[2], 0.42, 1.75, 1.35)
    hl = fts.panel_ratio(ad, g)
    ad.set_ylim(0.8, 1.22)
    ad.legend(*hl, loc="upper right", ncol=2, frameon=False, handlelength=1.4, handletextpad=0.4,
              columnspacing=0.8, fontsize=st.MIN_PT, borderaxespad=0.1)
    for x, s in zip(xs, "bcd"):
        letter(fig, x - 0.5, 1.9, s)
    return tp.save(fig, name)


def offset_points(fig, transform, dx, dy):
    from matplotlib.transforms import offset_copy
    return offset_copy(transform, fig=fig, x=dx, y=dy, units="points")


def country_lines(ax, color="#9C9995", lw=0.3, zorder=0.7):
    import tropics_robinson as rb
    geoms, _ = rb._load()
    for g in geoms:
        for r in rb._rings(rb._to_robin(g)):
            ax.plot(r[:, 0], r[:, 1], color=color, lw=lw, solid_joinstyle="round", zorder=zorder)


def nice(v):
    return max(t for t in (1, 2, 5, 10, 20, 50, 100) if t <= 0.9 * v)


def panel_expansion(fig, rect, map_h, prof_w, prof_gap):
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch
    import fig_headline as fh
    import tropics_bandmaps as m2
    import tropics_frame as tf
    import tropics_main_figures as f4
    import tropics_robinson as rb
    import tropics_surfaces as sf
    u = sf.load_urban()
    add, base = u["added"], u["base"]
    g_cmap, g_norm = f4._ramp_cmap(["#B5B5B5", "#8C8C8C", "#5E5E5E", "#2F2F2F"], "grey"), LogNorm(0.05, 2.0)
    u_cmap, u_norm = f4._ramp_cmap(m2.RAMP_URBAN, "urban"), LogNorm(0.05, 6.0)
    b0, a0 = np.nan_to_num(base), np.nan_to_num(add)
    grey = f4._continuous_raster(np.where(b0 >= a0, base, np.nan), g_cmap, g_norm, 0.05)
    rgba = f4._continuous_raster(np.where(a0 > b0, add, np.nan), u_cmap, u_norm, 0.05)
    c_base, c_add = "#5E5E5E", m2.RAMP_URBAN[4]
    lon0, lon1, lat0, lat1 = m2.EXTENT
    ny, nx = add.shape
    per = int(round(1.0 / sf.DISP_DEG))
    nc = nx // per
    cb = b0[:, :nc * per].reshape(ny, nc, per).sum(axis=(0, 2)) / 1e3
    ca = a0[:, :nc * per].reshape(ny, nc, per).sum(axis=(0, 2)) / 1e3
    lon_c = lon0 + (np.arange(nc) + 0.5)
    lon_c = np.where(lon_c < rb.LON_LO, lon_c + 360.0, lon_c)
    o = np.argsort(lon_c)
    lon_c, cb, ca = lon_c[o], cb[o], ca[o]
    vb, edges = f4._lat_profile(b0, lat1, lat0, deg=f4.PROFILE_BIN)
    va, _ = f4._lat_profile(a0, lat1, lat0, deg=f4.PROFILE_BIN)
    vb, va = vb / 1e3, va / 1e3
    W, H = fig.get_size_inches()
    out = {}

    def draw():
        ax = fig.add_axes(rect)
        rb.draw_globe(ax, zorder=0, borders=False)
        rb.frame(ax)
        ax.set_ylim(*fh.world_ylim())
        f4._draw_raster(ax, grey)
        f4._draw_raster(ax, rgba)
        country_lines(ax)
        f4._tropic_lines(ax)
        y_n = float(rb.project([0.0], [rb.TROPIC])[1][0])
        y_top = float(rb.project([0.0], [66.0])[1][0])
        top = float((cb + ca).max())
        k = (y_top - y_n) / (top * 1.05)
        x = np.asarray(rb.project(lon_c, np.full_like(lon_c, rb.TROPIC))[0])
        lx = fig.add_axes(rect)
        lx.patch.set_alpha(0)
        lx.set_axis_off()
        lx.set_xlim(ax.get_xlim())
        yl = ax.get_ylim()
        lx.set_ylim((yl[0] - y_n) / k, (yl[1] - y_n) / k)
        lx.fill_between(x, 0, cb, color=c_base, lw=0, zorder=7)
        lx.fill_between(x, cb, cb + ca, color=c_add, lw=0, zorder=7)
        lx.plot(x, np.where(cb + ca > 0.005, cb + ca, np.nan), color=c_add, lw=0.5, zorder=7)
        ax.legend([Patch(color=c_base), Patch(color=c_add)], ["Impervious by 2000", "Added 2000–2020"],
                  loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=fh.PT, handlelength=1.0,
                  handleheight=0.9, borderaxespad=0.1)
        f4._fig5_map_key(ax, map_h, g_cmap, g_norm, ticks=(0.05, 0.2, 2), title="Impervious by 2000 (km$^2$)",
                         x0=0.075, w=0.15, y0=0.085)
        f4._fig5_map_key(ax, map_h, u_cmap, u_norm, ticks=(0.05, 0.2, 1, 6), title="Added 2000–2020 (km$^2$)",
                         x0=0.497, w=0.168, y0=0.085)
        tf.robinson_degree_labels(ax, lat_side="left", lat_inside=True)
        window_outlines(ax, num_ax=lx, num_y=lambda y: (y - y_n) / k)
        out.update(ax=ax, k=k, y_n=y_n, top=top)
    robinson(draw)
    ax = out["ax"]
    yl = ax.get_ylim()

    def fy(y):
        return rect[1] + (y - yl[0]) / (yl[1] - yl[0]) * rect[3]
    x_end = float(rb.project([rb.LON_HI - 1e-6], [rb.TROPIC])[0][0])
    x_col = fig.transFigure.inverted().transform(ax.transData.transform((x_end, out["y_n"])))[0] + prof_gap / W
    t = nice(out["top"])
    sx = fig.add_axes([x_col, fy(out["y_n"]), 1e-4,
                       fy(out["y_n"] + t * out["k"]) - fy(out["y_n"])])
    sx.set_ylim(0, t)
    sx.set_yticks([0, t], ["", f"{t:g}"])
    sx.set_xticks([])
    for side in ("top", "right", "bottom"):
        sx.spines[side].set_visible(False)
    sx.patch.set_alpha(0)
    sx.yaxis.tick_right()
    sx.tick_params(axis="y", length=2, width=0.5, pad=1.5, labelsize=fh.PT, colors=st.INK)
    sx.text(0.0, 1.0, "10$^3$ km$^2$ per\n1° longitude", transform=offset_points(fig, sx.transAxes, 0, 5),
            ha="left", va="bottom", fontsize=fh.PT, color=st.INK, clip_on=False, linespacing=1.1)
    y_s = float(rb.project([0.0], [-rb.TROPIC])[1][0])
    px = fig.add_axes([x_col, fy(y_s), prof_w / W, fy(out["y_n"]) - fy(y_s)])
    ye = np.asarray(rb.project(np.zeros_like(edges), edges)[1])
    yy = np.repeat(ye, 2)[1:-1]
    bb, aa = np.repeat(vb, 2), np.repeat(va, 2)
    px.fill_betweenx(yy, 0, bb, color=c_base, lw=0)
    px.fill_betweenx(yy, bb, bb + aa, color=c_add, lw=0)
    px.plot(bb + aa, yy, color=c_add, lw=0.5)
    px.set_ylim(y_s, out["y_n"])
    tl = nice(float((vb + va).max()))
    px.set_xlim(0, float((vb + va).max()) * 1.08)
    px.set_xticks([0, tl], ["0", f"{tl:g}"])
    px.set_yticks([])
    for side in ("top", "right"):
        px.spines[side].set_visible(False)
    px.patch.set_alpha(0)
    px.axhline(float(rb.project([0.0], [0.0])[1][0]), color=rb.GRATICULE, lw=0.3, zorder=0)
    px.tick_params(axis="x", length=2, width=0.5, pad=1.5, labelsize=fh.PT, colors=st.INK)
    px.set_xlabel("10$^3$ km$^2$ per\n2° latitude", fontsize=fh.PT, labelpad=1.5)

    def added(w0, w1, h0, h1):
        lons = lon0 + (np.arange(nx) + 0.5) * (lon1 - lon0) / nx
        lats = lat1 - (np.arange(ny) + 0.5) * (lat1 - lat0) / ny
        cols = (lons >= w0) & (lons <= w1)
        rows = (lats >= h0) & (lats <= h1)
        return f"{a0[np.ix_(rows, cols)].sum():,.0f} km$^2$ added"

    def cells(ia):
        m2._draw(ia, grey)
        m2._draw(ia, rgba)
        tf._outline(ia, lw=0.3)
    window_strip(fig, rect, cells, added)


def panel_conversion(fig, low):
    from matplotlib.patches import Rectangle as Box
    import tropics_main_figures as f4
    W = tp.PAGE_W
    d = tp.cities()
    realms = ["Neotropic", "Afrotropic", "Indomalayan", "Australasia"]
    rows = ["All tropics"] + realms
    y = np.arange(len(rows))[::-1]

    def stack(ax, mat, colors):
        left = np.zeros(len(rows))
        for j, c in enumerate(colors):
            ax.barh(y, mat[:, j], left=left, height=0.62, color=c, edgecolor="white", lw=0.5)
            tc = st.text_on(c)
            for yi, li, vi in zip(y, left, mat[:, j]):
                if vi > 7:
                    ax.text(li + vi / 2, yi, f"{vi:.0f}", ha="center", va="center", fontsize=st.MIN_PT, color=tc)
            left = left + mat[:, j]
        ax.set_yticks(y)
        ax.set_yticklabels(rows, fontsize=st.MIN_PT)
        ax.set_xlim(0, 100)
        ax.tick_params(axis="y", length=0)
        st.despine(ax, keep=("bottom",))

    hab = pd.read_csv(f4.INTERIM_DIR / "city_habitat.csv")
    hab = hab[hab.city_id.isin(set(d.city_id))].merge(d[["city_id", "realm"]], on="city_id", how="left")
    hcols = [f"hab_{k}_km2_scaled" for k in tp.HAB_ORDER]
    hm = np.array([hab[hcols].sum().values] + [hab[hab.realm == r][hcols].sum().values for r in realms], float)
    hm = hm / hm.sum(axis=1, keepdims=True) * 100
    fcols = [f"form_{k}_km2" for k in f4.FORMS]
    fm = np.array([d[fcols].sum().values] + [d[d.realm == r][fcols].sum().values for r in realms], float)
    fm = fm / fm.sum(axis=1, keepdims=True) * 100
    lab_w, gap, y0, h = 0.82, 0.30, low - 1.58, 1.30
    pw = (W - 0.08 - 2 * lab_w - gap - 0.10) / 2
    for i, (mat, colors, labels, xlabel) in enumerate([
            (hm, [st.HABITAT_COLORS[k] for k in tp.HAB_ORDER], [tp.HAB_LABEL[k] for k in tp.HAB_ORDER],
             "Share of converted area by prior land cover (%)"),
            (fm, [st.FORM_COLORS[k] for k in f4.FORMS], [k.capitalize() for k in f4.FORMS],
             "Share of converted area by expansion form (%)")]):
        x = 0.08 + i * (lab_w + pw + gap)
        ax = axes(fig, x + lab_w, y0, pw, h)
        stack(ax, mat, colors)
        ax.set_xlabel(xlabel)
        ax.legend([Box((0, 0), 1, 1, facecolor=c) for c in colors], labels, fontsize=st.MIN_PT, ncol=3,
                  loc="upper center", bbox_to_anchor=(0.5, -0.30), handlelength=1.0, handleheight=0.9,
                  columnspacing=1.0, labelspacing=0.35, frameon=False)
        letter(fig, x, y0 + h + 0.08, "bc"[i])


def ed_fig2(name):
    st.apply_rc()
    low, prof_w, prof_gap = 2.75, 0.62, 0.05
    fig, rect, map_h = map_layout(low, right=prof_w - 0.10)
    panel_expansion(fig, rect, map_h, prof_w, prof_gap)
    panel_conversion(fig, low)
    return tp.save(fig, name)


def ed_fig3(name):
    import fig_threat_taxa as ftt
    fig = capture(ftt.figure, name)
    extend(fig, 2.15)
    ag = axes(fig, 0.95, 0.45, 1.50, 1.30)
    panel_reached_rules(ag)
    ah = axes(fig, 5.05, 0.45, 1.85, 1.30)
    panel_top_species(ah, n=12)
    y2, y1 = 2.45, 5.50
    relayout(fig, 7.62, {0: (0.52, y1, 3.0, 1.95), 1: (4.05, y1, 1.4, 1.95), 2: (6.12, y1 + 0.15, 0.7, 1.4),
                         3: (0.52, y2, 1.71, 1.95), 4: (2.85, y2, 1.71, 1.95), 5: (4.95, y2, 1.93, 1.27),
                         6: (0.95, 0.45, 1.5, 1.30), 7: (5.05, 0.45, 1.85, 1.30)},
             [(0.02, y1 + 2.0, "a"), (3.60, y1 + 2.0, "b"), (5.50, y1 + 2.0, "c"),
              (0.02, y2 + 2.0, "d"), (2.35, y2 + 2.0, "e"), (4.67, y2 + 2.0, "f"),
              (0.02, 1.85, "g"), (3.60, 1.85, "h")])
    old = fig.legends[0]
    hs, labs = old.legend_handles, [t.get_text() for t in old.get_texts()]
    size = old.get_texts()[0].get_fontsize()
    old.remove()
    W, H = fig.get_size_inches()
    fig.legend(hs, labs, loc="upper center", bbox_to_anchor=((4.05 + 0.7) / W, (y1 - 0.24) / H), ncol=1,
               frameon=False, fontsize=size, handlelength=1.0, handleheight=0.9, labelspacing=0.3, borderaxespad=0)
    return tp.save(fig, name)


def ed_fig4(name):
    import render_figures as rf
    fig = capture(rf._conc_fig, name)
    extend(fig, 2.0)
    ac = axes(fig, 0.75, 0.40, 2.55, 1.25)
    panel_class_concentration(ac)
    ad = axes(fig, 4.25, 0.40, 2.60, 1.25)
    panel_rule_concentration(ad)
    y1 = 2.25
    relayout(fig, 3.95, {0: (0.76, y1, 2.42, 1.55), 1: (4.12, y1, 2.78, 1.55),
                         2: (0.75, 0.40, 2.55, 1.25), 3: (3.92, 0.40, 2.98, 1.25)},
             [(0.05, y1 + 1.6, "a"), (3.30, y1 + 1.6, "b"), (0.05, 1.72, "c"), (3.30, 1.72, "d")])
    return tp.save(fig, name)


def lift(fig, height):
    W, H = fig.get_size_inches()
    k = H / height
    anchors = [(leg, leg.get_bbox_to_anchor().transformed(fig.transFigure.inverted())) for leg in fig.legends]
    for ax in fig.axes:
        for which in ("original", "active"):
            q = ax.get_position(original=which == "original")
            ax._set_position([q.x0, q.y0 * k, q.width, q.height * k], which=which)
    for t in list(fig.texts):
        if t.get_transform() is fig.transFigure and not isinstance(t, Annotation):
            x, y = t.get_position()
            t.set_position((x, y * k))
    fig.set_size_inches(W, height)
    for leg, b in anchors:
        leg.set_bbox_to_anchor((b.x0, b.y0 * k, b.width, b.height * k), transform=fig.transFigure)


def drop_old_map(fig, n=2):
    W, H = fig.get_size_inches()
    boxes = {}
    for ax in list(fig.axes)[:n]:
        fig.delaxes(ax)
    for t in list(fig.texts):
        if t.get_text().strip() == "a":
            t.remove()
    tops = []
    for ax in fig.axes:
        q = ax.get_position()
        tops.append((q.y0 + q.height) * H)
    return max(tops)


def lat_profile_right(fig, ax, rect, prof_w, prof_gap, lat, v, color, xlabel, log=False):
    import tropics_main_figures as f4
    import tropics_robinson as rb
    W, H = fig.get_size_inches()
    yl = ax.get_ylim()

    def fy(y):
        return rect[1] + (y - yl[0]) / (yl[1] - yl[0]) * rect[3]
    y_n = float(rb.project([0.0], [rb.TROPIC])[1][0])
    y_s = float(rb.project([0.0], [-rb.TROPIC])[1][0])
    x_end = float(rb.project([rb.LON_HI - 1e-6], [rb.TROPIC])[0][0])
    x_col = fig.transFigure.inverted().transform(ax.transData.transform((x_end, y_n)))[0] + prof_gap / W
    ctr, med = f4._lat_median(np.asarray(lat), np.asarray(v), f4.PROFILE_BIN)
    lo = np.asarray(rb.project(np.zeros_like(ctr), ctr - f4.PROFILE_BIN / 2)[1])
    hi = np.asarray(rb.project(np.zeros_like(ctr), ctr + f4.PROFILE_BIN / 2)[1])
    px = fig.add_axes([x_col, fy(y_s), prof_w / W, fy(y_n) - fy(y_s)])
    keep = med > 0
    px.barh(0.5 * (lo + hi)[keep], med[keep], height=0.86 * (hi - lo)[keep], color=color, lw=0, align="center")
    px.set_ylim(y_s, y_n)
    top = float(med.max())
    if log:
        px.set_xscale("log")
        px.set_xlim(0.5, top * 1.6)
    else:
        t = nice(top)
        px.set_xlim(0, top * 1.08)
        px.set_xticks([0, t], ["0", f"{t:g}"])
    px.set_yticks([])
    for side in ("top", "right"):
        px.spines[side].set_visible(False)
    px.patch.set_alpha(0)
    px.axhline(float(rb.project([0.0], [0.0])[1][0]), color=rb.GRATICULE, lw=0.3, zorder=0)
    px.tick_params(axis="x", length=2, width=0.5, pad=1.5, labelsize=st.MIN_PT, colors=st.INK)
    px.set_xlabel(xlabel, fontsize=st.MIN_PT, labelpad=1.5)
    return px


def panel_point_map(fig, rect, map_h, d, col, cmap, norm, ticks, key_title, prof_w, prof_gap, prof_label,
                    label_fmt=None, strip_below=None, boxes=None):
    import matplotlib.patheffects as pe
    import fig_headline as fh
    import tropics_frame as tf
    import tropics_main_figures as f4
    import tropics_robinson as rb
    d = d[np.isfinite(d[col].values)].sort_values(col)
    vals = np.clip(d[col].values.astype(float), norm.vmin, norm.vmax)
    colors = cmap(norm(vals))
    area = fh.marker_area(d.urban_added_km2.values)
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    out = {}

    def draw():
        ax = world_map(fig, rect)
        x, y = rb.project(d.lon.values, d.lat.values)
        ax.scatter(x, y, s=area, c=colors, linewidths=0.0, zorder=6, rasterized=True)
        f4._fig5_map_key(ax, map_h, cmap, norm, ticks=ticks, title=key_title, x0=0.497, w=0.168, y0=0.085)
        for k, km2 in enumerate((10, 100, 1000)):
            kx, ky = rb.project(-134.0 + 19.0 * k, -45.0)
            ax.scatter([kx], [ky], s=fh.marker_area(np.array([km2])), c="#8C8884", linewidths=0.0, zorder=6)
            tx, ty = rb.project(-134.0 + 19.0 * k, -53.0)
            ax.text(tx, ty, f"{km2:,}", ha="center", va="center", fontsize=fh.PT, color=st.INK, zorder=7)
        tx, ty = rb.project(-146.0, -36.5)
        ax.text(tx, ty, "Urban land added (km$^2$)", ha="left", va="center", fontsize=fh.PT, color=st.INK,
                zorder=7)
        if label_fmt:
            for name, lon, lat, tlon, tlat, ha in fh.LABELS:
                tlon, tlat, ha = MAP_LABEL_MOVES.get(name, (tlon, tlat, ha))
                q = d[(abs(d.lon - lon) < 1.2) & (abs(d.lat - lat) < 1.2)].sort_values("urban_km2_final").iloc[-1]
                px_, py_ = rb.project(q.lon, q.lat)
                qx, qy = rb.project(tlon, tlat)
                ax.annotate(f"{name}, {label_fmt(q[col])}", xy=(px_, py_), xytext=(qx, qy), ha=ha, va="center",
                            fontsize=fh.PT, color=st.INK, zorder=9, path_effects=halo,
                            arrowprops=dict(arrowstyle="-", color=st.INK, lw=0.45, shrinkA=1.5, shrinkB=1.0,
                                            relpos=RELPOS[ha]))
        tf.robinson_degree_labels(ax, lat_side="left", lat_inside=True)
        if strip_below:
            window_outlines(ax)
        for tag, lon0, lon1, lat0, lat1 in boxes or []:
            lo = np.concatenate([np.linspace(lon0, lon1, 20), np.full(20, lon1), np.linspace(lon1, lon0, 20),
                                 np.full(20, lon0)])
            la = np.concatenate([np.full(20, lat1), np.linspace(lat1, lat0, 20), np.full(20, lat0),
                                 np.linspace(lat0, lat1, 20)])
            bx, by = rb.project(lo, la)
            ax.plot(bx, by, color=st.INK, lw=0.9, zorder=8)
            tx, ty = rb.project(lon0, lat1)
            ax.annotate(tag, xy=(float(tx), float(ty)), xytext=(0.0, 1.5), textcoords="offset points",
                        ha="left", va="bottom", fontsize=fh.PT + 1, fontweight="bold", color=st.INK, zorder=9,
                        path_effects=halo)
        out["ax"] = ax
        lat_profile_right(fig, ax, rect, prof_w, prof_gap, d.lat.values, d[col].values, cm_color(), prof_label)
    robinson(draw)
    if strip_below:
        window_strip(fig, rect, lambda ia: ia.scatter(d.lon.values, d.lat.values, s=1.15 * area, c=colors,
                                                      linewidths=0.0, zorder=6, rasterized=True),
                     lambda lon0, lon1, lat0, lat1: strip_below(d[(d.lon >= lon0) & (d.lon <= lon1) &
                                                                  (d.lat >= lat0) & (d.lat <= lat1)]))
    return out["ax"]


def cm_color():
    import tropics_citymaps as cm
    return cm.CITY_RAMP[3]


def new_map_block(fig, low, prof_w, strip):
    import fig_headline as fh
    W = fig.get_size_inches()[0]
    head, ml, mr = 0.17, 0.08, 0.08
    map_w = W - ml - mr - (prof_w - 0.10)
    map_h = map_w * fh.world_aspect()
    strip_h = (fh.STRIP_DROP + fh.STRIP_IN + 0.40) if strip else 0.30
    H = low + strip_h + map_h + head
    lift(fig, H)
    fig.text(ml / W, 1 - 0.01 / H, "a", fontsize=st.MIN_PT + 1.2, fontweight="bold", va="top", ha="left",
             color=st.INK)
    return [ml / W, (low + strip_h) / H, map_w / W, map_h / H], map_h


def ed_fig5(name):
    import render_figures as rf
    import tropics_main_figures as mf
    fig = capture(rf._robin(mf.figure2, realms=False), name)
    extend(fig, 2.05)
    ad = axes(fig, 0.60, 0.55, 2.45, 1.20)
    panel_cv(ad)
    ae = axes(fig, 5.15, 0.55, 0.85, 1.05)
    panel_cross_taxon(ae)
    y, h = 0.62, 1.40
    dy = 2.55 - 4.72
    relayout(fig, 6.47 + dy, {0: ("dy", dy), 1: ("dy", dy), 2: (0.48, y, 1.22, h), 4: ("with", 2, 0.10, 0.30),
                              3: (2.62, y, 0.88, h), 5: (3.98, y, 1.22, h), 6: (6.02, y, 0.42, h * 0.85)},
             [(0.07, 6.47 + dy - 0.01 - 0.12, "a"), (0.02, y + h + 0.10, "b"), (1.73, y + h + 0.10, "c"),
              (3.55, y + h + 0.10, "d"), (5.25, y + h + 0.10, "e")])
    W, H = fig.get_size_inches()
    fig.axes[4].set_position([(0.48 + 0.06) / W, (y + 0.21) / H, 0.34 / W, 0.05 / H])
    lo, hi = fig.axes[2].get_ylim()
    fig.axes[2].set_ylim(lo, hi + 0.55 * (hi - lo) / 2.5)
    ae = fig.axes[6]
    ae.set_yticklabels([t.get_text().replace("Amphibians and reptiles", "Amphibians\nand reptiles")
                        for t in ae.get_yticklabels()])
    ad = fig.axes[5]
    ad.set_xticks(ad.get_xticks(), [t.get_text().replace("\n", " ") for t in ad.get_xticklabels()])
    for t in ad.get_xticklabels():
        t.set_rotation(40)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    import tropics_citymaps as cm
    import tropics_main_figures as f4
    from matplotlib.colors import LogNorm
    low = drop_old_map(fig) + 0.30
    prof_w, prof_gap = 0.62, 0.05
    rect, map_h = new_map_block(fig, low, prof_w, strip=True)
    e_cmap, e_norm = f4._ramp_cmap(cm.CITY_RAMP, "exposure"), LogNorm(5, 60)
    panel_point_map(fig, rect, map_h, cm.cities(), "cost_per_km2", e_cmap, e_norm, (5, 10, 20, 50),
                    "Exposure intensity\n(km$^2$ km$^{-2}$)", prof_w, prof_gap, "Median, 2° bands",
                    label_fmt=lambda v: f"{v:.0f}",
                    strip_below=lambda w: f"median {w.cost_per_km2.median():.1f}")
    return tp.save(fig, name)


def ed_fig6(name):
    import render_figures as rf
    import tropics_main_figures as mf
    fig = capture(rf._robin(mf.figure5, bottom="regions", middle=False), name)
    extend(fig, 2.05)
    extend(fig, 0.2)
    ae = axes(fig, 1.55, 0.70, 1.75, 1.35)
    panel_decomposition(ae)
    af = axes(fig, 4.80, 0.70, 1.95, 1.35)
    panel_region_share(af)
    letter(fig, 0.05, 2.12, "e")
    letter(fig, 3.55, 2.12, "f")
    import tropics_citymaps as cm
    import tropics_main_figures as f4
    from matplotlib.colors import LogNorm
    regs = [a for a in fig.axes[2:] if any(t.get_text()[:3] in ("b  ", "c  ", "d  ") for t in a.texts)]
    boxes = []
    for a in regs:
        tag = [t.get_text()[0] for t in a.texts if t.get_text()[:3] in ("b  ", "c  ", "d  ")][0]
        (x0, x1), (y0, y1) = a.get_xlim(), a.get_ylim()
        boxes.append((tag, x0, x1, y0, y1))
    low = drop_old_map(fig) + 0.30
    prof_w, prof_gap = 0.62, 0.05
    rect, map_h = new_map_block(fig, low, prof_w, strip=False)
    r_cmap, r_norm = f4._ramp_cmap(cm.CITY_RAMP, "rarity"), LogNorm(1, 1000)
    d = cm.cities()
    d = d.assign(rarity_1e5=d.rarity_per_km2 * 1e5)
    panel_point_map(fig, rect, map_h, d, "rarity_1e5", r_cmap, r_norm, (1, 10, 100, 1000),
                    "Rarity-weighted intensity\n(10$^{-5}$ km$^{-2}$)", prof_w, prof_gap, "Median, 2° bands",
                    boxes=boxes)
    return tp.save(fig, name)


def ed_fig7(name):
    import two_part_figs as tpf
    fig = capture(tpf.main_figure, name)
    extend(fig, 2.05)
    ae = axes(fig, 1.50, 0.45, 1.75, 1.35)
    panel_two_part_coefficients(ae)
    af = axes(fig, 5.10, 0.45, 1.75, 1.35)
    panel_two_part_validation(af)
    y2, y1, h = 0.45, 2.60, 1.45
    relayout(fig, y1 + h + 0.22, {0: (0.72, y1, 1.52, h), 1: (3.05, y1, 1.52, h), 2: (5.38, y1, 1.52, h),
                                  3: (0.82, y2, 1.38, h), 4: (3.50, y2, 1.30, h), 5: (5.85, y2, 1.05, h)},
             [(0.02, y1 + h + 0.06, "a"), (2.35, y1 + h + 0.06, "b"), (4.68, y1 + h + 0.06, "c"),
              (0.02, y2 + h + 0.06, "d"), (2.35, y2 + h + 0.06, "e"), (4.68, y2 + h + 0.06, "f")])
    fig.axes[5].set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    fig.axes[3].set_ylim(-0.60, 3.45)
    fig.axes[4].set_xlim(-0.8, 5.6)
    fig.axes[4].set_xticks([0, 2, 4])
    leg = fig.axes[4].get_legend()
    hs, labs = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
    fig.axes[4].legend(hs, [{"Size and setting": "Size and\nsetting", "With land added": "With land\nadded"}.get(t, t)
                            for t in labs], loc="lower right", frameon=False, fontsize=st.MIN_PT, handlelength=1.0,
                       borderaxespad=0.1, labelspacing=0.4)
    return tp.save(fig, name)


def fmt(v, nd=4):
    return f"{v:+.{nd}f}".replace("-", "−")


def pfmt(v):
    if v >= 0.001:
        return f"{v:.3f}"
    m, e = f"{v:.1e}".split("e")
    return f"{m} × 10$^{{{int(e)}}}$"


def span(v, nd=4):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    a, b = fmt(v.min(), nd), fmt(v.max(), nd)
    return a if a == b else f"{a} to {b}"


def primary_table():
    d = sd.responses().iloc[:6].copy()
    out = pd.DataFrame({
        "Response": np.where(d.response.str.startswith("range-rarity"), "Rarity weighted", "Species of concern"),
        "Size variable": np.where(d.specification.str.startswith("size 2020"), "2020 extent", "2000 extent"),
        "Setting at": np.where(d.specification.str.contains("footprint"), "2000 footprint", "2020 extent"),
        "n": [f"{int(v):,}" for v in d.n],
        "Size exponent": [fmt(v) for v in d.beta],
        "95% interval": [f"{fmt(a)} to {fmt(b)}" for a, b in zip(d.ci_lower, d.ci_upper)],
        "P": [pfmt(v) for v in d.p_value],
        "Setting coef.": [fmt(v, 3) for v in d.gamma_setting],
        "R²": [f"{v:.3f}" for v in d.R2]})
    return out


def robustness_rows():
    rows = []

    def add(check, beta, lo, hi, gamma, n_fits=None):
        beta, lo, hi = (np.asarray(x, float) for x in (beta, lo, hi))
        neg = int((beta < 0).sum())
        exc = int((hi < 0).sum()) if np.isfinite(hi).all() else None
        rows.append(dict(check=check, fits=len(beta) if n_fits is None else n_fits, exponent=span(beta),
                         negative=f"{neg} of {len(beta)}",
                         excludes_zero="–" if exc is None else f"{exc} of {len(beta)}",
                         setting=span(gamma, 3)))

    r = sd.responses().iloc[:3]
    add("Size variable and setting extent", r.beta, r.ci_lower, r.ci_upper, r.gamma_setting)
    rb = sd.realm_biome()
    for lev, lab in (("realm", "Within realm"), ("biome", "Within biome")):
        s = rb[rb.level == lev]
        add(lab, s.beta, s.ci_lower, s.ci_upper, s.gamma_setting)
    q = sd.quantiles()
    add("Quantile regression, 10th to 90th percentile", q.beta, q.ci_lower, q.ci_upper, q.gamma_setting)
    dl = pd.read_csv(sd._source("tableS4_delineation"))
    add("Delineation rules", dl.beta, dl.ci_lower, dl.ci_upper, dl.gamma_setting)
    sv = pd.read_csv(sd._source("tableS7_size_variables"))
    add("Size variable and minimum size", sv.beta, sv.ci_lower, sv.ci_upper, sv.gamma_setting)
    rt = pd.read_csv(sd._source("resolution_test_v2"))
    add("Lattice resolution and boundary rule", rt.beta, rt.lo, rt.hi, rt.gamma)
    lc = sd._block("tableS14_landcover_disagreement", "estimates")
    lc = lc[lc.response.str.contains("xposure|cost", regex=True)]
    add("Agglomerations where land-cover products agree", lc.beta, lc.lo, lc.hi, lc.gamma)
    sr = pd.read_csv(sd._source("setting_radius"))
    add("Setting buffer of 25, 50 and 100 km", sr.beta, sr.ci_lower, sr.ci_upper, sr.gamma_setting)
    ct = sd.cross_taxon()
    add("Setting from one taxonomic group", ct.size_exponent, np.full(len(ct), np.nan),
        np.full(len(ct), np.nan), ct.setting_coefficient)
    return pd.DataFrame(rows)


def robustness_table():
    return robustness_rows().rename(columns=dict(
        check="Check", fits="Fits", exponent="Size exponent", negative="Negative",
        excludes_zero="Interval below zero", setting="Setting coefficient"))


def draw_table(d, name, widths, title=None, fs=7.0):
    st.apply_rc()
    FontProperties._min_pt = fs
    row_h = fs * 1.75 / 72
    W = sum(widths) + 0.1
    H = row_h * (len(d) + 1) + 0.12
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    xs = np.concatenate([[0.05], 0.05 + np.cumsum(widths)[:-1]])
    y = 0.06 + row_h * 0.5
    for x, c in zip(xs, d.columns):
        ax.text(x, y, c, fontsize=fs, fontweight="bold", va="center", ha="left", color=st.INK)
    ax.plot([0.05, W - 0.05], [0.06, 0.06], color=st.INK, lw=0.6)
    ax.plot([0.05, W - 0.05], [0.06 + row_h, 0.06 + row_h], color=st.INK, lw=0.4)
    for i, row in enumerate(d.itertuples(index=False)):
        y = 0.06 + row_h * (i + 1.5)
        for x, v in zip(xs, row):
            ax.text(x, y, str(v), fontsize=fs, va="center", ha="left", color=st.INK)
    ax.plot([0.05, W - 0.05], [0.06 + row_h * (len(d) + 1)] * 2, color=st.INK, lw=0.6)
    FontProperties._min_pt = st.MIN_PT
    return tp.save(fig, name)


def ed_table1(name):
    return draw_table(primary_table(), name, [1.05, 0.75, 0.85, 0.45, 0.8, 1.15, 0.75, 0.8, 0.35])


def ed_table2(name):
    return draw_table(robustness_table(), name, [2.55, 0.35, 1.25, 0.55, 0.95, 0.95])
