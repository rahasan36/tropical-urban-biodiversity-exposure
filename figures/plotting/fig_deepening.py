"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import tropics_style as st
import tropics_panels as tp
import tropics_citymaps as cm
from tropics_config import TABLE_DIR

PT = 7.0
T = TABLE_DIR / "temporal"
DEEP = T / "agglomeration_deepening.csv"
PER = ["2000–2005", "2005–2010", "2010–2015", "2015–2020"]
PLAB = ["2000–\n2005", "2005–\n2010", "2010–\n2015", "2015–\n2020"]
YRS = [2000, 2005, 2010, 2015, 2020]
SIZE_ORDER = ["Not yet an agglomeration", "5 to 25 km²", "25 to 100 km²", "100 to 500 km²", "500 km² or more"]
SIZE_LAB = ["Not yet formed", "5–25 km²", "25–100 km²", "100–500 km²", "≥500 km²"]
SIZE_COL = ["#E7C9A9", "#C9CED8", "#94A1B5", "#5F7391", "#2B3A52"]
PCOL = ["#DADAEB", "#9E9AC8", "#6A51A3", "#3F007D"]
REALMS = ["Indomalayan", "Neotropic", "Afrotropic", "Australasia"]
REACHED = LinearSegmentedColormap.from_list("reached", plt.cm.magma_r(np.linspace(0.06, 0.92, 256)))
NORM = LogNorm(4, 80)
THREAT, NARROW = st.SET_COLOR["threatened"][0], st.SET_COLOR["narrow"][0]
LAND = "#7A7A7A"
SHORT = {"South-Eastern Asia": "SE Asia", "South America": "S America", "Southern Asia": "S Asia",
         "Western Africa": "W Africa", "Central America": "C America", "Eastern Asia": "E Asia",
         "Eastern Africa": "E Africa", "Middle Africa": "Middle Africa", "Caribbean": "Caribbean",
         "Western Asia": "W Asia", "Northern Africa": "N Africa", "Melanesia": "Melanesia",
         "Australia and New Zealand": "Australia–NZ", "Southern Africa": "S Africa"}
LEFT, RIGHT = 0.02, 6.96
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
CITIES = [("Pearl River\nDelta", 113.5, 22.9, (1.1e5, 38), "right", (0.88, 1.0)),
          ("Bangkok", 100.5, 13.75, (1.1e5, 95), "right", (0.70, 0.0)),
          ("Kuala Lumpur", 101.7, 3.1, (5.6e4, 49.5), "right", (0.85, 1.0)),
          ("Lagos", 3.4, 6.5, (1.45e4, 88), "center", (0.2, 0.0))]


def derive():
    import tropics_temporal as tt
    import fig_threat_taxa as ft
    a, b, V = tt.load_pairs()
    C, _, _ = ft.species()
    th = np.isin(C[a], ("CR", "EN", "VU"))
    a, b, V = a[th], b[th], V[th]
    early, later = V[:, 0] > 0, V[:, 1:4].sum(1)
    g = pd.DataFrame(dict(city_id=b, by_2005_km2=V[:, 0], after_2005_same_species_km2=np.where(early, later, 0.0),
                          after_2005_new_species_km2=np.where(early, 0.0, later))).groupby("city_id").sum()
    g["species_2005"] = pd.Series(a[early]).groupby(b[early]).nunique().reindex(g.index).fillna(0).astype(int)
    hit = V[:, 4] > 0
    g["species_2020"] = pd.Series(a[hit]).groupby(b[hit]).nunique().reindex(g.index).fillna(0).astype(int)
    g.to_csv(DEEP)


def deepening():
    if not DEEP.exists():
        derive()
    g = pd.read_csv(DEEP).merge(cm.cities(sample=False)[["city_id", "realm", "urban_added_km2", "urban_km2_final",
                                                         "lon", "lat"]], on="city_id")
    g["total"] = g.by_2005_km2 + g.after_2005_same_species_km2 + g.after_2005_new_species_km2
    g = g[g.by_2005_km2 > 0].copy()
    g["after_pct"] = 100 * (1 - g.by_2005_km2 / g.total)
    return g


def summary():
    from scipy import stats
    g = deepening()
    rho = stats.spearmanr(g.after_pct, g.species_2020)
    rich = g.species_2020 >= 40
    out = {"agglomerations_all": len(pd.read_csv(DEEP)),
           "agglomerations_converting_by_2005": len(g),
           "no_new_threatened_species_after_2005_pct": 100 * (g.species_2020 == g.species_2005).mean(),
           "median_share_converted_after_2005_pct": g.after_pct.median(),
           "share_converted_after_2005_q25_pct": g.after_pct.quantile(0.25),
           "share_converted_after_2005_q75_pct": g.after_pct.quantile(0.75),
           "majority_converted_after_2005_pct": 100 * (g.after_pct > 50).mean(),
           "threatened_range_converted_km2": g.total.sum(),
           "converted_after_2005_on_new_species_pct": 100 * g.after_2005_new_species_km2.sum() / g.total.sum(),
           "spearman_share_after_2005_vs_species": rho.statistic,
           "spearman_p": rho.pvalue,
           "agglomerations_40_or_more_species": int(rich.sum()),
           "agglomerations_40_or_more_species_pct": 100 * rich.mean(),
           "range_share_40_or_more_species_pct": 100 * g.total[rich].sum() / g.total.sum(),
           "left_of_axis": int((g.total < 0.8).sum())}
    for lab, lo, hi in (("under_10", 0, 10), ("10_to_19", 10, 20), ("20_to_39", 20, 40), ("40_or_more", 40, 10 ** 6)):
        out[f"median_share_after_2005_{lab}_species_pct"] = g.after_pct[(g.species_2020 >= lo) & (g.species_2020 < hi)].median()
    pd.Series(out, name="value").rename_axis("quantity").to_csv(T / "agglomeration_deepening_summary.csv")
    return out


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=PT)


def panel_size(ax):
    cl = pd.read_csv(T / "temporal_classes.csv")
    cl = cl[cl.classed_by == "size_c"]
    p = pd.read_csv(T / "temporal_periods.csv")
    x, bot = np.arange(4), np.zeros(4)
    for k, lab, col in zip(SIZE_ORDER, SIZE_LAB, SIZE_COL):
        v = 100 * cl[cl["class"] == k].set_index("period").loc[PER, "share_added"].to_numpy()
        ax.bar(x, v, 0.64, bottom=bot, color=col, edgecolor="white", lw=0.4)
        if k == SIZE_ORDER[-1]:
            for i in range(4):
                ax.text(i, bot[i] + v[i] / 2, f"{v[i]:.0f}%", ha="center", va="center", fontsize=PT, color="white")
        ax.text(3.44, max(bot[3] + v[3] / 2, 3.6), lab, ha="left", va="center", fontsize=PT, color=st.INK)
        bot += v
    for i, val in enumerate(p.intensity_concern):
        ax.text(i, 102, f"{val:.1f}", ha="center", va="bottom", fontsize=PT, color=st.INK)
    ax.text(-0.45, 111, "Range converted per km² added (km²)", ha="left", va="bottom", fontsize=PT, color=st.MUTED)
    ax.set_xticks(x, PLAB)
    ax.set_xlim(-0.5, 5.35)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.spines["bottom"].set_bounds(-0.5, 3.5)
    ax.set_ylabel("Share of land added (%)", fontsize=PT)
    clean(ax)


def panel_accrual(ax):
    c = pd.read_csv(T / "temporal_subregion_cumulative.csv")
    a = c[c["sub"] == "All tropics"]
    p = pd.read_csv(T / "temporal_periods.csv")
    land = np.concatenate([[0], p.added_km2.cumsum().to_numpy() / 1000])
    ax2 = ax.twinx()
    ax2.plot(YRS, land, color=LAND, lw=1.1, ls=(0, (3, 2)), marker="s", ms=2.8, mec="white", mew=0.3, zorder=1)
    ax2.set_ylim(0, 85)
    ax2.set_yticks([0, 20, 40, 60, 80])
    ax2.set_ylabel("Urban land added since 2000\n(thousand km²)", color=LAND, fontsize=PT)
    ax2.tick_params(axis="y", colors=LAND, labelsize=PT)
    ax2.spines["right"].set_color(LAND)
    ax2.spines["top"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for grp, lab, col, off, ha, va in (("Narrow-ranged", "Narrow range", NARROW, (-3, 3), "right", "bottom"),
                                       ("Threatened", "Threatened", THREAT, (3, -3), "left", "top")):
        v = np.concatenate([[0], a[a.group == grp].sort_values("j").ge1.to_numpy()])
        ax.plot(YRS, v, "-o", color=col, lw=1.4, ms=3.4, mec="white", mew=0.4, zorder=3)
        for x, y, dv in zip(YRS[1:], v[1:], np.diff(v)):
            last = grp == "Threatened" and x == YRS[-1]
            ax.annotate(f"+{int(dv)}", (x, y), xytext=(0, 4) if last else off, textcoords="offset points",
                        ha="center" if last else ha, va="bottom" if last else va, fontsize=PT,
                        color=st.darker(col, 0.18))
        tx, ty = (2000.6, 585) if grp == "Narrow-ranged" else (2012.6, 62)
        ax.text(tx, ty, f"{lab} ({int(v[-1])})" if grp == "Narrow-ranged" else f"{lab}\n({int(v[-1])})",
                color=st.darker(col, 0.18), va="center", ha="left", fontsize=PT, linespacing=1.05)
    ax.set_xlim(1999, 2022.4)
    ax2.set_xlim(1999, 2022.4)
    ax.set_xticks(YRS, ["2000", "", "2010", "", "2020"])
    ax.set_ylim(0, 630)
    ax.set_yticks([0, 200, 400, 600])
    ax.set_ylabel("Species with ≥1% of range\nconverted since 2000", fontsize=PT)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=PT)
    return ax2


def area(km2):
    return 1.5 + 60.0 * np.sqrt(np.asarray(km2, float) / 2900.0)


def panel_cities(ax, scale):
    fig = ax.figure
    g = deepening()
    if scale == "realm":
        d = g.sample(frac=1, random_state=1)
        col, alpha = [st.REALM_COLOR.get(r, st.FAINT) for r in d.realm], 0.8
    else:
        d = g.sort_values(["species_2020", "urban_added_km2"], ascending=[True, False])
        col, alpha = REACHED(NORM(np.clip(d.species_2020.values.astype(float), NORM.vmin, NORM.vmax))), 0.9
    lo, hi = 0.8, 1.2e5
    keep = ((d.total >= lo) & (d.total <= hi)).values
    ax.scatter(d.total[keep], d.after_pct[keep], s=area(d.urban_added_km2[keep]), c=np.asarray(col)[keep], alpha=alpha,
               lw=0.1, edgecolor="white", rasterized=True, clip_on=False, zorder=2)
    med = float(g.after_pct.median())
    ax.axhline(med, color="white", lw=2.0, zorder=4.8)
    ax.axhline(med, color=st.INK, lw=0.8, ls=(0, (3, 2)), zorder=4.9)
    ax.text(0.03, med + 2.5, f"Median {med:.0f}%", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=PT, color=st.INK, path_effects=HALO, zorder=6)
    for name, lon, lat, pos, ha, rel in CITIES:
        q = g[(abs(g.lon - lon) < 1.2) & (abs(g.lat - lat) < 1.2)].sort_values("urban_km2_final").iloc[-1]
        ax.scatter([q.total], [q.after_pct], s=area(q.urban_added_km2), facecolor="none", edgecolor=st.INK, lw=0.6,
                   zorder=5)
        ax.annotate(name, xy=(q.total, q.after_pct), xytext=pos, ha=ha, va="center", fontsize=PT, color=st.INK,
                    linespacing=1.05, zorder=6, path_effects=HALO,
                    arrowprops=dict(arrowstyle="-", color=st.INK, lw=0.45, shrinkA=0, shrinkB=2.5, relpos=rel))
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks([1, 100, 10000], ["1", "100", "10,000"])
    ax.set_xticks([], minor=True)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Threatened range converted\nby 2020 (km²)", fontsize=PT, linespacing=1.1)
    ax.set_ylabel("Share converted after 2005 (%)", fontsize=PT)
    clean(ax)
    p = ax.get_position()
    w, h = p.width * fig.get_figwidth(), p.height * fig.get_figheight()
    if scale == "realm":
        hs = [Line2D([], [], marker="o", ls="", color=st.REALM_COLOR[r], ms=3.6, mec="none") for r in REALMS]
        ax.legend(hs, REALMS, loc="lower right", fontsize=PT, ncol=1, handletextpad=0.0, borderaxespad=0.2,
                  borderpad=0.1, labelspacing=0.2, frameon=False)
    else:
        bw, bh, y0 = 0.055, 0.46, 0.02
        e = np.linspace(0, 1, 65)
        for a, b in zip(e[:-1], e[1:]):
            k = REACHED((a + b) / 2)
            ax.add_patch(Rectangle((1.0 - bw / w, (y0 + a * bh) / h), bw / w, (b - a) * bh / h,
                                   transform=ax.transAxes, facecolor=k, edgecolor=k, lw=0.2, zorder=6))
        for v in (5, 10, 20, 40, 80):
            ax.text(1.0 - (bw + 0.03) / w, (y0 + float(NORM(v)) * bh) / h, str(v), transform=ax.transAxes,
                    ha="right", va="center", fontsize=PT, color=st.INK, zorder=6)
        ax.text(1.0 - (bw + 0.22) / w, (y0 + bh / 2) / h, "Threatened\nspecies", transform=ax.transAxes, ha="right",
                va="center", fontsize=PT, color=st.INK, linespacing=1.05, zorder=6)
    r = fig.canvas.get_renderer()
    left = lambda t: ax.transAxes.inverted().transform((t.get_window_extent(r).x0, 0))[0]
    y = 1.0 + 0.115 / h
    x = left(ax.text(1.0, y, "km² added", transform=ax.transAxes, ha="right", va="center", fontsize=PT,
                     color=st.INK)) - 0.035 / w
    for v, lab in ((1000, "1,000"), (100, "100"), (10, "10")):
        x = left(ax.text(x, y, lab, transform=ax.transAxes, ha="right", va="center", fontsize=PT, color=st.INK))
        rad = np.sqrt(area(v) / np.pi) / 72.0
        ax.scatter([x - (0.03 + rad) / w], [y], s=area(v), transform=ax.transAxes, facecolor="#BDBDBD",
                   edgecolor="white", lw=0.1, clip_on=False, zorder=6)
        x -= (0.03 + 2 * rad + 0.07) / w


def accrue():
    c = pd.read_csv(T / "temporal_subregion_cumulative.csv")
    s = c[(c["sub"] != "All tropics") & (c.group == "Narrow-ranged") & (c.j == 3)].set_index("sub").ge1
    return c, s[s >= 5].sort_values(ascending=False).index.tolist()


def panel_sub(ax, grp, xmax, xlabel, names):
    c, order = accrue()
    g = c[(c.group == grp) & c["sub"].isin(order)].pivot(index="sub", columns="j", values="ge1").reindex(order).fillna(0)
    y, left = np.arange(len(order)), np.zeros(len(order))
    for j in range(4):
        inc = g[j].to_numpy() - (g[j - 1].to_numpy() if j else 0)
        ax.barh(y, inc, 0.68, left=left, color=PCOL[j], lw=0)
        left += inc
    for i, n in enumerate(left):
        ax.text(n + xmax * 0.02, i, f"{int(n)}", va="center", ha="left", fontsize=PT, color=st.INK)
    ax.set_yticks(y, [SHORT[k] for k in order] if names else [""] * len(order))
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, fontsize=PT)
    clean(ax)


def panel_rarity(ax, gap=10):
    r = pd.read_csv(T / "temporal_realms.csv")
    p = pd.read_csv(T / "temporal_periods.csv")
    x = np.arange(4)
    cols = dict(zip(REALMS, st.CAT))
    series = [(rl, r[r.realm == rl].set_index("period").loc[PER, "rarity_per_km2"].to_numpy(), cols[rl], 1.2)
              for rl in REALMS]
    series.append(("All tropics", p.rarity_per_km2.to_numpy(), st.INK, 2.0))
    ends = []
    for nm, v, c, lw in series:
        idx = 100 * v / v[0]
        ax.plot(x, idx, "-o", color=c, lw=lw, ms=3.2, mec="white", mew=0.4, zorder=3 if nm == "All tropics" else 2)
        ends.append([idx[-1], nm, c])
    ends.sort()
    ys = [e[0] for e in ends]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    for (y0, nm, c), y in zip(ends, ys):
        ax.plot([3.05, 3.2], [y0, y], color=st.MUTED, lw=0.4, clip_on=False, zorder=1)
        ax.text(3.24, y, f"{nm} {y0 - 100:+.0f}%".replace("-", "−"), color=st.darker(c) if c != st.INK else c,
                fontsize=PT, va="center", ha="left", clip_on=False)
    ax.axhline(100, color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(x, ["2005", "2010", "2015", "2020"])
    ax.set_xlim(-0.2, 3.1)
    ax.set_ylim(55, 128)
    ax.set_xlabel("End of five-year period", fontsize=PT)
    ax.set_ylabel("Rarity-weighted intensity\n(2000–2005 = 100)", fontsize=PT)
    clean(ax)


def letter(fig, x, y, ch):
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, ch, fontsize=PT + 2, fontweight="bold", ha="left", va="bottom", color=st.INK)


LAYOUT = {"species": ([["size", "accrual", "species"], ["narrow", "threat", "rarity"]],
                      [[1.72, 1.40, 1.70], [1.45, 1.45, None]]),
          "realm": ([["size", "accrual", "realm"], ["narrow", "threat", "rarity"]],
                    [[1.72, 1.40, 1.70], [1.45, 1.45, None]]),
          "both": ([["size", "accrual"], ["realm", "species"], ["narrow", "threat", "rarity"]],
                   [[1.92, None], [2.99, 2.99], [1.45, 1.45, None]])}


def draw(ax, key):
    if key == "size":
        panel_size(ax)
    elif key == "accrual":
        return ax, panel_accrual(ax)
    elif key in ("species", "realm"):
        panel_cities(ax, key)
    elif key == "narrow":
        panel_sub(ax, "Narrow-ranged", 150, "Narrow-range species above 1%", True)
    elif key == "threat":
        panel_sub(ax, "Threatened", 100, "Threatened species above 1%", False)
        ax.legend([Rectangle((0, 0), 1, 1, color=k) for k in PCOL], PER, loc="lower right", ncol=1, frameon=False,
                  fontsize=PT, handlelength=0.9, handletextpad=0.35, borderaxespad=0.1, borderpad=0, labelspacing=0.2,
                  title="Period 1% was passed", title_fontsize=PT, alignment="left")
    else:
        panel_rarity(ax)
    return (ax,)


def compose(keys, xs, ws):
    W = tp.PAGE_W
    ph, bot, vg = 1.75, 0.50, 0.68
    n = len(keys)
    H = bot + n * ph + (n - 1) * vg + 0.24
    fig = plt.figure(figsize=(W, H))
    rows, k = [], 0
    for i, row in enumerate(keys):
        y = bot + (n - 1 - i) * (ph + vg)
        out = []
        for key, x, w in zip(row, xs[i], ws[i]):
            ax = fig.add_axes([x / W, y / H, w / W, ph / H])
            out.append(draw(ax, key))
            p = ax.get_position()
            letter(fig, p.x0 * W - (0.0 if key == "threat" else 0.30), p.y1 * H + 0.05, "abcdefg"[k])
            k += 1
        rows.append(out)
    return fig, rows


def overhang(fig, row):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for axes in row:
        boxes = [x.get_tightbbox(r) for x in axes]
        p = axes[0].get_window_extent(r)
        out.append(((p.x0 - min(q.x0 for q in boxes)) / fig.dpi, (max(q.x1 for q in boxes) - p.x1) / fig.dpi))
    return out


def place(widths, hang, gap):
    xs, x = [], LEFT
    for w, (l, r) in zip(widths, hang):
        xs.append(x + l)
        x += l + w + r + gap
    return xs


def figure(name="figure_temporal_v25", scale="species"):
    st.apply_rc(min_pt=7)
    keys, fixed = LAYOUT[scale]
    span = RIGHT - LEFT
    ws = [[1.50 if w is None else w for w in row] for row in fixed]
    xs = [place(w, [(0.5, 0.1)] * len(w), 0.18) for w in ws]
    ref = next(i for i, row in enumerate(fixed) if None not in row)
    used = lambda w, h: sum(v + l + r for v, (l, r) in zip(w, h))
    for _ in range(2):
        fig, rows = compose(keys, xs, ws)
        hang = [overhang(fig, r) for r in rows]
        plt.close(fig)
        gap = (span - used(ws[ref], hang[ref])) / (len(ws[ref]) - 1)
        for i, row in enumerate(fixed):
            free = [j for j, w in enumerate(row) if w is None]
            rest = span - sum(l + r for l, r in hang[i]) - sum(w for w in row if w is not None) - (len(row) - 1) * gap
            for j in free:
                ws[i][j] = rest / len(free)
        xs = [place(w, h, gap) for w, h in zip(ws, hang)]
    fig, _ = compose(keys, xs, ws)
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
    print(figure("figure_temporal_v26", "realm"))
    print(figure("figure_temporal_v27", "both"))
