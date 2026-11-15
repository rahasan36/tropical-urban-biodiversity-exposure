"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as Pg, Rectangle
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import box
from shapely.affinity import translate
import tropics_style as st
import tropics_panels as tp
import tropics_robinson as rb
import tropics_frame as tf

BREAKS = [0, 8, 10, 12, 15, 18, 22, 26, 1e9]
BLAB = ["<8", "8–10", "10–12", "12–15", "15–18", "18–22", "22–26", "≥26"]
CMAP = ListedColormap(["#2B5C8A", "#5C8FBE", "#93BBDC", "#C8DDEE", "#F6CDB4", "#EE9672", "#D2553D", "#9E2A2B"])
NORM = BoundaryNorm(BREAKS, CMAP.N)
NAMES = dict(BRA="Brazil", IND="India", IDN="Indonesia", MEX="Mexico", THA="Thailand", NGA="Nigeria",
             CHN="China", VEN="Venezuela", VNM="Vietnam", PER="Peru", COD="DR Congo", COL="Colombia")
GREY = "#E4E2DF"


def table():
    import tropics_numbers as tn
    d = tn.analysis_sample(tn.table())
    d["threat_per_km2"] = d.cost_threat_km2 / d.urban_added_km2
    d["narrow_per_km2"] = d.cost_narrow_km2 / d.urban_added_km2
    cp = pd.read_csv(tp.TAB_DIR / "temporal" / "temporal_city_periods.csv.gz", usecols=["city_id", "j", "existed"])
    ex = cp[cp.j == 0].set_index("city_id").existed
    d["existed"] = d.city_id.map(ex).fillna(False).astype(bool)
    return d


def country_polys(codes):
    import pyogrio
    g = pyogrio.read_dataframe(rb.NE_PATH, columns=["ADM0_A3"])
    band = box(-180, -rb.TROPIC, 180, rb.TROPIC)
    east = box(rb.LON_LO + 1e-6, -rb.TROPIC, 180.0, rb.TROPIC)
    west = box(-180.0, -rb.TROPIC, rb.LON_LO - 1e-6, rb.TROPIC)
    out = []
    for geom, a3 in zip(g.geometry, g.ADM0_A3):
        if geom is None or geom.is_empty:
            continue
        gb = geom.intersection(band)
        if gb.is_empty:
            continue
        parts = [gb.intersection(east), translate(gb.intersection(west), xoff=360.0)]
        for p in parts:
            if p.is_empty:
                continue
            p = p.simplify(0.05, preserve_topology=True)
            for r in rb._rings(rb._to_robin(p)):
                out.append((a3, r))
    return out


def panel_map(ax, d, min_n=10):
    rb.draw_globe(ax, zorder=0, top_face="white")
    c = d.groupby("country").agg(n=("city_id", "size"), med=("cost_per_km2", "median"))
    polys, cols = [], []
    for a3, r in country_polys(c.index):
        polys.append(Pg(r, closed=True))
        if a3 in c.index and c.loc[a3, "n"] >= min_n:
            cols.append(CMAP(NORM(c.loc[a3, "med"])))
        else:
            cols.append(GREY)
    ax.add_collection(PatchCollection(polys, facecolors=cols, edgecolors="white", linewidths=0.25, zorder=1))
    for la in (rb.TROPIC, -rb.TROPIC):
        ll = np.linspace(rb.LON_LO + 1e-6, rb.LON_HI - 1e-6, 240)
        xx, yy = rb.project(ll, np.full_like(ll, la))
        ax.plot(xx, yy, color="#8C8884", lw=0.6, ls=(0, (4, 3)), zorder=4)
    (x0, x1), _ = rb.band_limits(0.797)
    yv = float(rb.lat_to_y(30.0))
    ax.set_xlim(x0, x1); ax.set_ylim(-yv, yv); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#B9B6B1"); s.set_linewidth(0.6)
    return c


def key(ax, rect):
    k = ax.inset_axes(rect)
    for i in range(len(BLAB)):
        k.add_patch(Rectangle((0, len(BLAB) - 1 - i), 1, 1, facecolor=CMAP(i), edgecolor="white", lw=0.5))
        k.text(1.25, len(BLAB) - 0.5 - i, BLAB[i], va="center", ha="left", fontsize=8)
    k.add_patch(Rectangle((0, -1.5), 1, 1, facecolor=GREY, edgecolor="white", lw=0.5))
    k.text(1.25, -1.0, "Fewer than 10", va="center", ha="left", fontsize=8)
    k.set_xlim(0, 4.2); k.set_ylim(-1.6, len(BLAB) + 4.6); k.axis("off")
    k.text(0, len(BLAB) + 0.3, "Median km²\nconverted per\nkm² added", fontsize=8, va="bottom", ha="left",
           fontweight="bold", linespacing=1.1)
    return k


def panel_violins(ax, d, c, n=10):
    top = c.sort_values("n", ascending=False).head(n).sort_values("med", ascending=False)
    for i, a3 in enumerate(top.index):
        v = d.loc[d.country == a3, "cost_per_km2"].to_numpy()
        col = CMAP(NORM(top.loc[a3, "med"]))
        vp = ax.violinplot([v], positions=[i], widths=0.85, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(col); b.set_edgecolor("none"); b.set_alpha(1.0)
        ax.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color="white", lw=1.2)
    ax.axhline(d.cost_per_km2.median(), color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(top)), [NAMES.get(a, a) for a in top.index], rotation=45, ha="right",
                  rotation_mode="anchor")
    ax.set_xlim(-0.6, len(top) - 0.4); ax.set_ylim(0, 60)
    ax.set_ylabel("Range converted per km² added (km²)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)


def half(ax, x, v, col, side):
    kde = __import__("scipy.stats", fromlist=["gaussian_kde"]).gaussian_kde(np.log10(v))
    yy = np.linspace(np.log10(v).min(), np.log10(v).max(), 200)
    w = kde(yy); w = 0.34 * w / w.max()
    s = 1 if side == "right" else -1
    ax.fill_betweenx(10 ** yy, x, x + s * w, color=col, alpha=0.55, lw=0)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    bx = x - s * 0.10
    ax.add_patch(Rectangle((bx - 0.06, q1), 0.12, q3 - q1, facecolor="white", edgecolor=col, lw=1.0, zorder=3))
    ax.plot([bx - 0.06, bx + 0.06], [q2, q2], color=col, lw=1.4, zorder=4)
    lo, hi = np.percentile(v, [5, 95])
    ax.plot([bx, bx], [lo, q1], color=col, lw=0.8, zorder=2)
    ax.plot([bx, bx], [q3, hi], color=col, lw=0.8, zorder=2)


def panel_existing(axs, d):
    specs = [("cost_per_km2", "Species of concern", 1.0),
             ("narrow_per_km2", "Narrow-ranged", 1.0),
             ("rarity_per_km2", "Rarity ×10$^4$", 1e4)]
    out = {}
    for ax, (col, lab, sc) in zip(axs, specs):
        for x, (flag, c) in enumerate(((True, "#5F7391"), (False, "#D9752D"))):
            v = d.loc[(d.existed == flag) & (d[col] > 0), col].to_numpy() * sc
            half(ax, x, v, c, "right")
            out[(col, flag)] = float(np.median(v))
        if col != "cost_per_km2":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 60)
        ax.set_xticks([0, 1], ["Present\nby 2000", "Formed\nafter"])
        ax.set_xlim(-0.35, 1.5)
        ax.text(0.5, 1.05, lab, transform=ax.transAxes, fontsize=8, ha="center", va="bottom")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axs[0].set_ylabel("Per km² added (km²)")
    return out


def figure(name="figure_country_violin_v3"):
    st.apply_rc()
    d = table()
    W = tp.PAGE_W
    (x0, x1), _ = rb.band_limits(0.797)
    yv = float(rb.lat_to_y(30.0))
    map_h = W * (2 * yv) / (x1 - x0)
    low_h = 1.75
    H = map_h + 0.30 + low_h + 0.55
    fig = plt.figure(figsize=(W, H))
    KW = 0.95
    map_h = (W - KW) * (2 * yv) / (x1 - x0)
    H = map_h + 0.44 + low_h + 0.62
    fig.set_size_inches(W, H)
    ax = fig.add_axes([KW / W, 1 - map_h / H, (W - KW) / W, map_h / H])
    c = panel_map(ax, d)
    tf.robinson_degree_labels(ax, lat_side="right", lat_inside=True, lat_va="outward")
    kax = fig.add_axes([0.10 / W, 1 - map_h / H, (KW - 0.12) / W, map_h / H])
    kax.axis("off")
    key(kax, [0.0, 0.02, 1.0, 0.96])
    fig.text(0.005, 1 - 0.02 / H, "a", fontsize=st.MIN_PT + 1, fontweight="bold", va="top")
    bx = fig.add_axes([0.55 / W, 0.62 / H, 2.85 / W, low_h / H])
    panel_violins(bx, d, c)
    fig.text(0.005, (0.62 + low_h + 0.12) / H, "b", fontsize=st.MIN_PT + 1, fontweight="bold", va="bottom")
    fig.text(0.12 / W, (0.62 + low_h + 0.12) / H, "   Ten countries with the most agglomerations",
             fontsize=8, fontweight="bold", va="bottom")
    cw = (W - 3.95 - 0.10 - 2 * 0.30) / 3
    axs = [fig.add_axes([(3.95 + i * (cw + 0.30)) / W, 0.62 / H, cw / W, (low_h - 0.12) / H]) for i in range(3)]
    med = panel_existing(axs, d)
    fig.text(3.55 / W, (0.62 + low_h + 0.12) / H, "c", fontsize=st.MIN_PT + 1, fontweight="bold", va="bottom")
    fig.text(3.68 / W, (0.62 + low_h + 0.12) / H, "Agglomerations present by 2000 or formed after",
             fontsize=8, fontweight="bold", va="bottom")
    print({f"{k[0]}|{'existed' if k[1] else 'new'}": round(v, 3) for k, v in med.items()})
    print("countries mapped", int((c.n >= 10).sum()))
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
