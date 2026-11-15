"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import box as shapely_box
from matplotlib.ticker import NullLocator
from matplotlib.transforms import offset_copy
import tropics_style as st
import tropics_frame as tf
import tropics_panels as tp
import tropics_robinson as rb
import tropics_citymaps as cm
import tropics_main_figures as f4
import tropics_speciesmaps as sm

PT = 7.0
CATS = ("CR", "EN", "VU")
CLASSES = [("AMPHIBIA", "Amphibia", "Amphibians"), ("REPTILIA", "Reptilia", "Reptiles"),
           ("MAMMALIA", "Mammalia", "Mammals"), ("AVES", "Aves", "Birds")]
REST = "#E4E2DF"
YEARS = (2005, 2010, 2015, 2020)
INSETS = [("Atlantic Forest", 0.95), ("Gulf of Guinea", 2.98), ("Sunda", 5.02)]
INSET_IN = 0.82
VIOLIN = ["#E3E0DC", "#D3D0CB", "#C2BFBA", "#B0ADA8"]
STRIP_IN, STRIP_GAP, STRIP_DROP, STRIP_X0 = 0.98, 0.176, 0.48, 0.20
REGIONS = tp.TAB_DIR / "temporal" / "region_threatened_reached.csv"
REACHED = tp.TAB_DIR / "temporal" / "agglomeration_threatened_reached.csv"
EPOCHS = tp.TAB_DIR / "temporal" / "threatened_share_by_epoch.csv"
LABELS = [("Mexico City", -99.1, 19.4, -112.0, 8.0, "right"),
          ("Lima", -77.0, -12.0, -91.0, -15.0, "right"),
          ("Rio de Janeiro", -43.2, -22.9, -29.0, -29.0, "left"),
          ("Lagos", 3.4, 6.5, -4.0, -6.0, "right"),
          ("Nairobi", 36.8, -1.3, 50.0, -4.0, "left"),
          ("Mumbai", 72.9, 19.1, 67.0, 30.0, "right"),
          ("Kuala Lumpur", 101.7, 3.1, 95.0, -11.5, "right"),
          ("Jakarta", 106.8, -6.2, 101.0, -18.0, "right"),
          ("Bangkok", 100.5, 13.75, 103.0, 30.5, "center"),
          ("Pearl River Delta", 113.5, 22.9, 127.0, 29.0, "left")]


def species():
    t = pd.read_csv(tp.TAB_DIR / "species_concentration.csv")
    t = t[(t.rule == "centroid") & t.is_threat & (t.converted_km2 > 0)].rename(columns={"sci_name": "sci"})
    t["pct"] = 100 * t.converted_km2 / t.range_km2
    return t


def world_ylim():
    span = 2 * rb.X_MAX
    return rb.Y_BOT - 0.008 * span, rb.Y_TOP + 0.012 * span


def world_aspect():
    span = 2 * rb.X_MAX
    y0, y1 = world_ylim()
    return (y1 - y0) / (span + 0.012 * span * 3.2)


def derive():
    import tropics_temporal as tt
    import fig_threat_taxa as ft
    a, b, V = tt.load_pairs()
    C, _, R = ft.species()
    th = np.isin(C[a], CATS)
    pos = th & (V[:, 4] > 0)
    n = pd.Series(a[pos]).groupby(b[pos]).nunique().rename("n_threatened_reached")
    n.index.name = "city_id"
    n.to_csv(REACHED)
    per = pd.DataFrame(V[th, :4]).groupby(a[th]).sum()
    per = per[per.sum(1) > 0]
    sp = per.index.to_numpy()
    sh = 100 * per.cumsum(axis=1).div(R[sp], axis=0)
    sh.columns = [f"share_{y}" for y in YEARS]
    sh.insert(0, "category", C[sp])
    sh.insert(1, "range_km2", R[sp])
    sh.index.name = "sp_i"
    sh.to_csv(EPOCHS)
    c = cm.cities(sample=False).set_index("city_id")
    lon, lat = c.lon.reindex(b[pos]).to_numpy(), c.lat.reindex(b[pos]).to_numpy()
    rows = [(r[0], int(len(np.unique(a[pos][(lon >= r[1]) & (lon <= r[2]) & (lat >= r[3]) & (lat <= r[4])]))))
            for r in tf.REGIONS]
    pd.DataFrame(rows, columns=["region", "n_threatened_reached"]).to_csv(REGIONS, index=False)


def marker_area(added):
    return 0.5 + 48.0 * np.sqrt(np.clip(added, 0, None) / 2900.0)


def panel_map(fig, rect, d, map_h, strip=False):
    ax = fig.add_axes(rect)
    old = f4.PROJECTION
    f4.PROJECTION = "robinson"
    try:
        rb.draw_globe(ax, zorder=0)
        rb.frame(ax)
        ax.set_ylim(*world_ylim())
        cmap = LinearSegmentedColormap.from_list("reached", plt.cm.magma_r(np.linspace(0.06, 0.92, 256)))
        norm = LogNorm(4, 80)
        d = d.sort_values("n_threatened_reached")
        x, y = rb.project(d.lon.values, d.lat.values)
        v = np.clip(d.n_threatened_reached.values.astype(float), norm.vmin, norm.vmax)
        ax.scatter(x, y, s=marker_area(d.urban_added_km2.values), c=cmap(norm(v)), linewidths=0.0, zorder=6,
                   rasterized=True)
        f4._tropic_lines(ax)
        f4._fig5_map_key(ax, map_h, cmap, norm, ticks=(5, 10, 20, 40, 80),
                         title="Threatened species reached", x0=0.497, w=0.168, y0=0.085)
        for k, km2 in enumerate((10, 100, 1000)):
            kx, ky = rb.project(-134.0 + 19.0 * k, -45.0)
            ax.scatter([kx], [ky], s=marker_area(np.array([km2])), c="#8C8884", linewidths=0.0, zorder=6)
            tx, ty = rb.project(-134.0 + 19.0 * k, -53.0)
            ax.text(tx, ty, f"{km2:,}", ha="center", va="center", fontsize=PT, color=st.INK, zorder=7)
        tx, ty = rb.project(-146.0, -36.5)
        ax.text(tx, ty, "Urban land added (km$^2$)", ha="left", va="center", fontsize=PT, color=st.INK, zorder=7)
        halo = [pe.withStroke(linewidth=1.8, foreground="white")]
        for name, lon, lat, tlon, tlat, ha in LABELS:
            q = d[(abs(d.lon - lon) < 1.2) & (abs(d.lat - lat) < 1.2)].sort_values("urban_km2_final").iloc[-1]
            px, py = rb.project(q.lon, q.lat)
            qx, qy = rb.project(tlon, tlat)
            ax.annotate(f"{name}, {int(q.n_threatened_reached)}", xy=(px, py), xytext=(qx, qy), ha=ha,
                        va="center", fontsize=PT, color=st.INK, zorder=9, path_effects=halo,
                        arrowprops=dict(arrowstyle="-", color=st.INK, lw=0.45, shrinkA=1.5, shrinkB=1.0))
        tf.robinson_degree_labels(ax, lat_side="right", lat_inside=True)
        boxes = {r[0]: r for r in tf.REGIONS}
        wins = ([(r[0], r[5]) for r in tf.REGIONS] if strip else [(n, st.INK) for n, _ in INSETS])
        for k, (name, col) in enumerate(wins, start=1):
            _, lon0, lon1, lat0, lat1 = boxes[name][:5]
            lo = np.concatenate([np.linspace(lon0, lon1, 40), np.full(40, lon1), np.linspace(lon1, lon0, 40),
                                 np.full(40, lon0)])
            la = np.concatenate([np.full(40, lat1), np.linspace(lat1, lat0, 40), np.full(40, lat0),
                                 np.linspace(lat0, lat1, 40)])
            bx, by = rb.project(lo, la)
            ax.plot(bx, by, color=col, lw=1.0 if strip else 0.8, zorder=8, solid_joinstyle="round")
            tx, ty = rb.project(lon0, lat1)
            ax.annotate(str(k), xy=(float(tx), float(ty)), xytext=(-2.0, 1.5), textcoords="offset points",
                        ha="right", va="bottom", fontsize=PT + 1, fontweight="bold", color=col, zorder=9,
                        path_effects=halo)
    finally:
        f4.PROJECTION = old
    W, H = fig.get_size_inches()
    _, land = rb._load()
    colors = cmap(norm(v))
    area = marker_area(d.urban_added_km2.values)
    size = STRIP_IN if strip else INSET_IN
    counts = pd.read_csv(REGIONS).set_index("region").n_threatened_reached if strip else None
    for k, (name, col) in enumerate(wins, start=1):
        _, lon0, lon1, lat0, lat1 = boxes[name][:5]
        if strip:
            x_in = STRIP_X0 + (k - 1) * (STRIP_IN + STRIP_GAP)
            y_in = rect[1] * H - STRIP_DROP - STRIP_IN
        else:
            x_in = rect[0] * W + dict(INSETS)[name]
            y_in = (rect[1] + rect[3]) * H - 0.17 - INSET_IN
        ia = fig.add_axes([x_in / W, y_in / H, size / W, size / H])
        ia.set_zorder(10)
        clip = land.intersection(shapely_box(lon0 - 1, lat0 - 1, lon1 + 1, lat1 + 1))
        ia.add_collection(PatchCollection([MplPolygon(r, closed=True) for r in rb._rings(clip)],
                                          facecolors=rb.TOP_FACE, edgecolors=rb.BORDER, linewidths=0.3, zorder=1))
        ia.scatter(d.lon.values, d.lat.values, s=(1.15 if strip else 0.9) * area, c=colors, linewidths=0.0,
                   zorder=6, rasterized=True)
        ia.set_xlim(lon0, lon1)
        ia.set_ylim(lat0, lat1)
        ia.set_xticks([])
        ia.set_yticks([])
        for sp in ia.spines.values():
            sp.set_color(col)
            sp.set_linewidth(1.1 if strip else 0.6)
        ia.text(0.0, 1.03, f"{k}  {name}", transform=ia.transAxes, ha="left", va="bottom", fontsize=PT,
                color=col, zorder=9, path_effects=halo)
        if strip:
            tf.corner_labels(ia, lon0, lon1, lat0, lat1)
            below = offset_copy(ia.transAxes, fig=fig, x=0, y=-13.5, units="points")
            ia.text(0.5, 0.0, f"{int(counts[name]):,} species", transform=below, ha="center", va="top",
                    fontsize=PT, color=st.INK)
        if not strip or k == 1:
            km_per_deg = 111.32 * np.cos(np.radians(0.5 * (lat0 + lat1)))
            bx0 = lon0 + 0.06 * (lon1 - lon0)
            by = lat0 + 0.07 * (lat1 - lat0)
            ia.plot([bx0, bx0 + 500.0 / km_per_deg], [by, by], color=st.INK, lw=1.2, solid_capstyle="butt",
                    zorder=9)
            ia.text(bx0, by + 0.03 * (lat1 - lat0), "500 km", ha="left", va="bottom", fontsize=PT - 0.5,
                    color=st.INK, zorder=9, path_effects=halo)
    return ax


def panel_reached(ax, t):
    acc = pd.read_csv(tp.TAB_DIR / "species_accounting_by_class.csv").set_index("Class")
    rows = [("All classes", t, int(acc.loc["Total", "Threatened mapped"]))]
    rows += [(lab, t[t.taxon == key], int(acc.loc[name, "Threatened mapped"])) for key, name, lab in CLASSES]
    for i, (lab, q, mapped) in enumerate(rows):
        ax.add_patch(Rectangle((0, i + 0.02), 100, 0.34, facecolor=REST, edgecolor="none"))
        left = 0.0
        for c in CATS:
            w = 100 * (q.category == c).sum() / mapped
            ax.add_patch(Rectangle((left, i + 0.02), w, 0.34, facecolor=sm.CAT_COLORS[c], edgecolor="none"))
            left += w
        ax.text(0, i - 0.20, lab, ha="left", va="center", fontsize=PT, color=st.INK)
        ax.text(100, i - 0.20, f"{len(q):,} of {mapped:,} ({100 * len(q) / mapped:.0f}%)", ha="right",
                va="center", fontsize=PT, color=st.INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(rows) - 0.45, -0.45)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([])
    ax.set_xlabel("Threatened species reached (%)", fontsize=PT)
    ax.tick_params(labelsize=PT)
    st.despine(ax, keep=("bottom",))
    tp.head(ax, "b", "")


def panel_share(ax, t):
    for k in ("VU", "EN", "CR"):
        q = t[t.category == k]
        ax.scatter(q.range_km2, q.pct, s=4, color=sm.CAT_COLORS[k], alpha=0.55, lw=0, zorder=2, rasterized=True)
    edges = np.logspace(np.log10(t.range_km2.min()), np.log10(t.range_km2.max()), 19)
    b = np.digitize(t.range_km2, edges[1:-1])
    med, cnt = t.groupby(b).pct.median(), t.groupby(b).size()
    keep = cnt[cnt >= 30].index
    ctr = np.sqrt(edges[:-1] * edges[1:])
    ax.plot(ctr[keep], med.loc[keep], color=st.INK, lw=1.1, zorder=4)
    r = t.loc[t.sci == "Rattus ranjiniae"].iloc[0]
    ax.annotate(r.sci, xy=(r.range_km2, r.pct), xytext=(1.5e3, r.pct), textcoords="data", style="italic",
                color=st.INK, va="center", ha="left", fontsize=PT,
                arrowprops=dict(arrowstyle="-", color=st.MUTED, lw=0.5, shrinkA=1, shrinkB=3))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e9)
    ax.set_ylim(1e-7, 100)
    ax.set_xticks([1e1, 1e3, 1e5, 1e7, 1e9])
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1, 100])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Global mapped range (km$^2$)", fontsize=PT)
    ax.set_ylabel("Share of mapped range\nconverted (%)", fontsize=PT)
    ax.tick_params(labelsize=PT)
    st.despine(ax, keep=("bottom", "left"))
    hs = [plt.Line2D([0], [0], marker="o", ls="none", ms=3.5, color=sm.CAT_COLORS[k]) for k in CATS]
    hs.append(plt.Line2D([0], [0], color=st.INK, lw=1.1))
    ax.legend(hs, ["CR", "EN", "VU", "Median"], loc="lower left", ncol=1, frameon=False, handlelength=0.9,
              handletextpad=0.4, labelspacing=0.25, borderpad=0, borderaxespad=0, fontsize=PT,
              bbox_to_anchor=(3, 3e-7), bbox_transform=ax.transData)
    tp.head(ax, "c", "")


def panel_deepening(ax, e):
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(7)
    for j, yr in enumerate(YEARS):
        q = e[e[f"share_{yr}"] > 0]
        v = q[f"share_{yr}"].to_numpy()
        lv = np.log10(v)
        yy = np.linspace(lv.min(), lv.max(), 240)
        w = gaussian_kde(lv)(yy)
        w = 0.36 * w / w.max()
        ax.fill_betweenx(10 ** yy, j - w, j + w, color=VIOLIN[j], lw=0, zorder=2)
        hi = q[v >= 1]
        for c in ("VU", "EN", "CR"):
            p = hi.loc[hi.category == c, f"share_{yr}"].to_numpy()
            ax.scatter(j + 0.60 * (rng.random(len(p)) - 0.5), p, s=4.5, color=sm.CAT_COLORS[c], alpha=0.9, lw=0,
                       zorder=4, rasterized=True)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((j - 0.05, q1), 0.10, q3 - q1, facecolor="white", edgecolor=st.INK, lw=0.7,
                               zorder=5))
        ax.plot([j - 0.05, j + 0.05], [q2, q2], color=st.INK, lw=1.2, zorder=6)
        ax.text(j, 42, f"{len(hi)}", ha="center", va="center", fontsize=PT, color=f4.STAT, zorder=7)
    ax.axhline(1, color=st.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 100)
    ax.set_yticks([1e-4, 1e-2, 1, 100])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.55, len(YEARS) - 0.45)
    ax.set_xticks(range(len(YEARS)), [str(y) for y in YEARS])
    ax.set_ylabel("Cumulative share of mapped\nrange converted (%)", fontsize=PT)
    ax.tick_params(labelsize=PT)
    ax.tick_params(axis="x", length=0)
    ax.text(1.0, 1.02, "Species above 1%", transform=ax.transAxes, ha="right", va="bottom", fontsize=PT,
            color=f4.STAT)
    st.despine(ax, keep=("bottom", "left"))
    tp.head(ax, "d", "")


def panel_ridges(ax, e):
    from scipy.stats import gaussian_kde
    xs = np.linspace(-5.5, 2.0, 400)
    greys = ["#DDDAD6", "#C9C6C1", "#B3B0AB", "#9B9893"]
    n = len(YEARS)
    for j, yr in enumerate(YEARS):
        v = e[f"share_{yr}"].to_numpy()
        v = v[v > 0]
        w = gaussian_kde(np.log10(v))(xs)
        w = 0.92 * w / w.max()
        base = n - 1 - j
        ax.fill_between(xs, base, base + w, color=greys[j], lw=0, zorder=2 + j)
        hi = xs >= 0
        ax.fill_between(xs[hi], base, base + w[hi], color=f4.STAT, lw=0, zorder=2 + j)
        ax.plot(xs, base + w, color="white", lw=0.6, zorder=2 + j)
        med = np.log10(np.median(v))
        ax.plot([med, med], [base, base + 0.92 * np.interp(med, xs, w / 0.92)], color=st.INK, lw=0.9, zorder=3 + j)
        ax.text(2.0, base + 0.30, f"{int((v >= 1).sum())}", ha="right", va="center", fontsize=PT, color=f4.STAT,
                zorder=9)
    ax.axvline(0, color=st.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(-5.5, 2.0)
    ax.set_ylim(0, n + 0.05)
    ax.set_xticks([-4, -2, 0, 2], ["10$^{-4}$", "10$^{-2}$", "1", "100"])
    ax.set_yticks(range(n), [str(y) for y in YEARS[::-1]])
    ax.set_xlabel("Cumulative share of mapped\nrange converted (%)", fontsize=PT)
    ax.tick_params(labelsize=PT)
    ax.tick_params(axis="y", length=0)
    ax.text(1.0, 1.02, "Species above 1%", transform=ax.transAxes, ha="right", va="bottom", fontsize=PT,
            color=f4.STAT)
    st.despine(ax, keep=("bottom",))
    tp.head(ax, "d", "")


def figure_strip(name="figure1_headline_v10"):
    st.apply_rc()
    if not (REACHED.exists() and EPOCHS.exists() and REGIONS.exists()):
        derive()
    d = cm.cities(sample=False).merge(pd.read_csv(REACHED), on="city_id", how="left")
    t = species()
    e = pd.read_csv(EPOCHS)
    W = tp.PAGE_W
    head, ml, mr, ph, bot = 0.17, 0.08, 0.08, 1.80, 0.50
    map_w = W - ml - mr
    map_h = map_w * world_aspect()
    strip_h = STRIP_DROP + STRIP_IN + 0.40
    gap = 0.34
    H = head + map_h + strip_h + gap + ph + bot
    fig = plt.figure(figsize=(W, H))
    panel_map(fig, [ml / W, (bot + ph + gap + strip_h) / H, map_w / W, map_h / H], d, map_h, strip=True)
    fig.text(ml / W, 1 - 0.01 / H, "a", fontsize=st.MIN_PT + 1.2, fontweight="bold", va="top", ha="left",
             color=st.INK)
    bx = fig.add_axes([0.14 / W, bot / H, 1.92 / W, ph / H])
    cx = fig.add_axes([2.86 / W, bot / H, 1.62 / W, ph / H])
    dx = fig.add_axes([5.22 / W, bot / H, 1.70 / W, ph / H])
    panel_reached(bx, t)
    panel_share(cx, t)
    panel_deepening(dx, e)
    return tp.save(fig, name)


def figure(name="figure1_headline_v7"):
    st.apply_rc()
    if not (REACHED.exists() and EPOCHS.exists() and REGIONS.exists()):
        derive()
    d = cm.cities(sample=False).merge(pd.read_csv(REACHED), on="city_id", how="left")
    t = species()
    e = pd.read_csv(EPOCHS)
    W = tp.PAGE_W
    head, ml, mr, gap, ph, bot = 0.17, 0.08, 0.08, 0.44, 1.80, 0.50
    map_w = W - ml - mr
    map_h = map_w * world_aspect()
    H = head + map_h + gap + ph + bot
    fig = plt.figure(figsize=(W, H))
    panel_map(fig, [ml / W, (bot + ph + gap) / H, map_w / W, map_h / H], d, map_h)
    fig.text(ml / W, 1 - 0.01 / H, "a", fontsize=st.MIN_PT + 1.2, fontweight="bold", va="top", ha="left",
             color=st.INK)
    bx = fig.add_axes([0.14 / W, bot / H, 1.92 / W, ph / H])
    cx = fig.add_axes([2.86 / W, bot / H, 1.62 / W, ph / H])
    dx = fig.add_axes([5.22 / W, bot / H, 1.70 / W, ph / H])
    panel_reached(bx, t)
    panel_share(cx, t)
    panel_deepening(dx, e)
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
