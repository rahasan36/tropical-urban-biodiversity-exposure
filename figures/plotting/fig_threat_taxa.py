"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tropics_style as st
import tropics_panels as tp
import tropics_temporal as t

CATS = ["VU", "EN", "CR"]
CCOL = {"VU": "#F2C94C", "EN": "#E8892B", "CR": "#B5541C"}
PCOL = {"VU": "#E9B829", "EN": "#D9711A", "CR": "#8E3A12"}
TAXA = [("AMPHIBIA", "Amphibians"), ("REPTILIA", "Reptiles"), ("MAMMALIA", "Mammals"), ("AVES", "Birds")]
GROUPS = [("all", "All agglomerations"), ("old", "Present by 2000"), ("new", "Formed after 2000")]


def species():
    import tropics_rangecache as rc, tropics_flags as tfl
    from tropics_cost import PAIRS_PATH
    sp, _ = rc.load_index()
    cat, tax, _sci, area = tfl.flags(PAIRS_PATH)
    order = json.loads(str(np.load(PAIRS_PATH, allow_pickle=False)["sp_index"]))
    j = np.array([order.get(str(v), -1) for v in sp.sp_id])
    n = int(sp.sp_i.max()) + 1
    C = np.full(n, "", object); T = np.full(n, "", object); R = np.zeros(n)
    C[sp.sp_i] = np.char.upper(cat[j].astype(str)); T[sp.sp_i] = tax[j]; R[sp.sp_i] = area[j]
    return C, T, R


def compute():
    C, T, R = species()
    a, b, V = t.load_pairs()
    cp = pd.read_csv(tp.TAB_DIR / "temporal" / "temporal_city_periods.csv.gz", usecols=["city_id", "j", "existed"])
    ex = np.zeros(int(b.max()) + 1, bool)
    e = cp[cp.j == 0]
    ex[e.city_id.to_numpy()] = e.existed.to_numpy()
    th = np.isin(C[a], CATS)
    pos = (V[:, 4] > 0) & th
    rows = []
    u_old = np.unique(a[pos & ex[b]])
    u_new = np.unique(a[pos & ~ex[b]])
    sets = dict(all=np.unique(a[pos]), old=u_old, new=u_new, only_new=np.setdiff1d(u_new, u_old))
    for g in sets:
        u = sets[g]
        for tx, _ in TAXA:
            for c in CATS:
                rows.append(dict(group=g, taxon=tx, cat=c, n=int(((T[u] == tx) & (C[u] == c)).sum())))
    counts = pd.DataFrame(rows)
    conv = pd.Series(V[th, 4]).groupby(a[th]).sum()
    pm = th & (V[:, 4] > 0)
    nagg = pd.Series(b[pm]).groupby(a[pm]).nunique()
    pv = pd.DataFrame(dict(sp=a[pm], v=V[pm, 4])).groupby("sp").v
    topshare = pv.max() / pv.sum()
    s = pd.DataFrame(dict(sp=conv.index, conv=conv.values))
    s["cat"] = C[s.sp]; s["taxon"] = T[s.sp]; s["range"] = R[s.sp]
    s = s[(s.conv > 0) & (s.range > 0)]
    s["frac"] = 100 * s.conv / s.range
    s["nagg"] = s.sp.map(nagg).fillna(0).astype(int)
    s["top"] = 100 * s.sp.map(topshare)
    per = np.column_stack([pd.Series(V[th, k]).groupby(a[th]).sum().reindex(conv.index).to_numpy() for k in range(4)])
    tim = {}
    for c in CATS:
        m = C[conv.index] == c
        tot = per[m].sum(0)
        tim[c] = 100 * np.cumsum(tot) / tot.sum()
    out = tp.TAB_DIR / "temporal"
    counts.to_csv(out / "threatened_reached_by_group_v1.csv", index=False)
    s.to_csv(out / "threatened_range_fraction_v1.csv", index=False)
    return counts, s, tim


def panel_counts(ax, counts):
    x = 0.0; ticks, labs, gx = [], [], []
    for gi, (g, gl) in enumerate(GROUPS):
        start = x
        for tx, tl in TAXA:
            bot = 0
            for c in CATS:
                v = counts[(counts.group == g) & (counts.taxon == tx) & (counts.cat == c)].n.iloc[0]
                ax.bar(x, v, 0.8, bottom=bot, color=CCOL[c], edgecolor="white", lw=0.4)
                bot += v
            ax.text(x, bot + 10, f"{bot}", ha="center", va="bottom", fontsize=7)
            ticks.append(x); labs.append(tl); x += 1
        gx.append(((start + x - 1) / 2, gl, start, x - 1))
        x += 0.8
    ax.set_xticks(ticks, labs, rotation=90, fontsize=8, va="top")
    ax.tick_params(axis="x", length=0)
    for cx, gl, s0, s1 in gx:
        ax.text(cx, 735, gl.replace(" agglomerations", "").replace("All", "All"), ha="center",
                va="bottom", fontsize=8)
        ax.plot([s0 - 0.4, s1 + 0.4], [725, 725], color=st.MUTED, lw=0.6)
    ax.set_ylim(0, 790); ax.set_xlim(-0.7, x - 0.5)
    ax.set_ylabel("Threatened species reached")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def panel_frac(ax, s):
    rng = np.random.default_rng(3)
    for i, c in enumerate(CATS[::-1]):
        v = s[s.cat == c].frac.to_numpy()
        lv = np.log10(np.clip(v, 1e-4, None))
        from scipy.stats import gaussian_kde
        yy = np.linspace(lv.min(), lv.max(), 200)
        w = gaussian_kde(lv)(yy); w = 0.36 * w / w.max()
        ax.fill_betweenx(10 ** yy, i, i + w, color=CCOL[c], alpha=0.6, lw=0)
        ax.scatter(i - 0.18 + rng.uniform(-0.07, 0.07, v.size), np.clip(v, 1e-4, None), s=1.2,
                   color=CCOL[c], alpha=0.5, lw=0, rasterized=True)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((i - 0.06, q1), 0.12, q3 - q1, facecolor="white", edgecolor=st.INK, lw=0.7, zorder=3))
        ax.plot([i - 0.06, i + 0.06], [q2, q2], color=st.INK, lw=1.2, zorder=4)
        ax.text(i + 0.07, 3e1, f"{q2:.2f}%", fontsize=7, va="bottom", ha="center", zorder=5)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 100)
    ax.set_xticks(range(3), ["CR", "EN", "VU"], fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.45, 2.6)
    ax.set_ylabel("Share of global range (%)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def panel_taxon_frac(ax, s):
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(5)
    cols = ["#009E73", "#0072B2", "#CC79A7", "#56B4E9"]
    for i, ((tx, tl), c) in enumerate(zip(TAXA, cols)):
        v = s[s.taxon == tx].frac.to_numpy()
        lv = np.log10(np.clip(v, 1e-4, None))
        yy = np.linspace(lv.min(), lv.max(), 200)
        w = gaussian_kde(lv)(yy); w = 0.36 * w / w.max()
        ax.fill_betweenx(10 ** yy, i, i + w, color=c, alpha=0.6, lw=0)
        ax.scatter(i - 0.18 + rng.uniform(-0.07, 0.07, v.size), np.clip(v, 1e-4, None), s=1.2,
                   color=c, alpha=0.5, lw=0, rasterized=True)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((i - 0.06, q1), 0.12, q3 - q1, facecolor="white", edgecolor=st.INK, lw=0.7, zorder=3))
        ax.plot([i - 0.06, i + 0.06], [q2, q2], color=st.INK, lw=1.2, zorder=4)
        ax.text(i + 0.07, 3e1, f"{q2:.2f}%", fontsize=7, va="bottom", ha="center", zorder=5)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 100)
    ax.set_xticks(range(4), [tl for _, tl in TAXA], fontsize=8, rotation=25, ha="center")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.45, 3.6)
    ax.set_ylabel("Share of global range (%)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def panel_range_scatter(ax, s):
    for c in ["VU", "EN", "CR"]:
        m = s.cat == c
        ax.scatter(s.range[m], s.frac[m], s=3.0, color=PCOL[c], alpha=0.75, lw=0, rasterized=True)
    from scipy.stats import spearmanr
    rho = spearmanr(np.log10(s.range), np.log10(s.frac.clip(1e-6))).correlation
    ax.text(0.97, 0.95, f"Spearman $\\rho$ = {rho:.2f}".replace("-", "\u2212"), transform=ax.transAxes, ha="right",
            va="top", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 100)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1, 100])
    ax.set_xlabel("Global range (km²)")
    ax.set_ylabel("Share of global range (%)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return rho


def panel_top(ax, s):
    bins = [(100, 100.01, "100%"), (75, 100, "75–99%"), (50, 75, "50–74%"), (0, 50, "Under 50%")]
    tc = ["#3B4A61", "#5F7391", "#94A1B5", "#D2D7E0"]
    out = {}
    for i, c in enumerate(CATS[::-1]):
        v = s[s.cat == c].top.to_numpy()
        left = 0
        for (lo, hi, lab), col in zip(bins, tc):
            f = 100 * ((v >= lo) & (v < hi)).mean() if hi < 100.01 else 100 * (v >= 99.999).mean()
            if lo == 75:
                f = 100 * ((v >= 75) & (v < 99.999)).mean()
            ax.barh(i, f, 0.62, left=left, color=col, edgecolor="white", lw=0.4)
            if f >= 8:
                ax.text(left + f / 2, i, f"{f:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if col in tc[:2] else st.INK)
            left += f
        out[c] = float(100 * (v >= 50).mean())
    ax.set_yticks(range(3), ["CR", "EN", "VU"], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of species (%)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    h = [Rectangle((0, 0), 1, 1, color=c) for c in tc]
    ax.legend(h, [b[2] for b in bins], loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=2, frameon=False,
              fontsize=8, handlelength=0.9, handletextpad=0.3, columnspacing=1.0,
              title="Share from the largest agglomeration", title_fontsize=8,
              alignment="left", borderaxespad=0.2)
    return out


def panel_nagg(ax, s):
    for c in CATS[::-1]:
        v = np.sort(s[s.cat == c].nagg.to_numpy())
        y = 100 * (1 - np.arange(v.size) / v.size)
        ax.step(v, y, where="post", color=CCOL[c], lw=1.4)
    ax.set_xscale("log"); ax.set_xlim(1, s.nagg.max() * 1.2); ax.set_ylim(0, 102)
    ax.set_xlabel("Agglomerations reaching the species")
    ax.set_ylabel("Species reached by at least x (%)")
    med = s.groupby("cat").nagg.median()
    ax.text(0.97, 0.95, "Median: " + ", ".join(f"{c} {int(med[c])}" for c in CATS[::-1]),
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return med


def panel_timing(ax, counts):
    y = np.arange(len(TAXA))
    for i, (tx, tl) in enumerate(TAXA):
        left = 0
        for c in CATS[::-1]:
            v = counts[(counts.group == "only_new") & (counts.taxon == tx) & (counts.cat == c)].n.iloc[0]
            ax.barh(i, v, 0.7, left=left, color=CCOL[c], edgecolor="white", lw=0.4)
            left += v
        ax.text(left + 2, i, f"{left}", va="center", ha="left", fontsize=8)
    ax.set_yticks(y, [tl for _, tl in TAXA], fontsize=8)
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 175)
    ax.set_xlabel("Species")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def figure(name="figure_threat_taxa_v4"):
    st.apply_rc()
    counts, s, tim = compute()
    W = tp.PAGE_W; H = 6.1
    ph = 1.95
    B2, B1 = 0.60, 0.60 + ph + 1.25
    fig = plt.figure(figsize=(W, H))
    a1 = fig.add_axes([0.52 / W, B1 / H, 3.00 / W, ph / H])
    panel_counts(a1, counts)
    a2 = fig.add_axes([4.05 / W, B1 / H, 1.40 / W, ph / H])
    panel_frac(a2, s)
    a3 = fig.add_axes([6.12 / W, (B1 + 0.15) / H, 0.70 / W, (ph - 0.55) / H])
    panel_timing(a3, counts)
    pw = (W - 0.52 - 0.12 - 2 * 0.62) / 3
    xs = [0.52 + k * (pw + 0.62) for k in range(3)]
    a4 = fig.add_axes([xs[0] / W, B2 / H, pw / W, ph / H]); panel_taxon_frac(a4, s)
    a5 = fig.add_axes([xs[1] / W, B2 / H, pw / W, ph / H]); rho = panel_range_scatter(a5, s)
    a6 = fig.add_axes([(xs[2] - 0.22) / W, B2 / H, (pw + 0.22) / W, (ph - 0.68) / H]); med = panel_top(a6, s)

    def head(x, y, l, t):
        fig.text(x / W, y / H, l, fontsize=st.MIN_PT + 1, fontweight="bold", va="bottom")
        fig.text((x + 0.15) / W, y / H, t, fontsize=8, fontweight="bold", va="bottom")
    y1 = B1 + ph + 0.10
    head(0.02, y1, "a", "Threatened species reached, 2000–2020")
    head(3.60, y1, "b", "Global range converted")
    head(5.50, y1, "c", "Species reached only by")
    fig.text(5.65 / W, (y1 - 0.02) / H, "new agglomerations", fontsize=8, fontweight="bold", va="top")
    y2 = B2 + ph + 0.10
    head(xs[0] - 0.50, y2, "d", "Global range converted by taxon")
    head(xs[1] - 0.50, y2, "e", "Share of range converted by range size")
    head(xs[2] - 0.50, y2, "f", "Dominance of one agglomeration")
    h = [Rectangle((0, 0), 1, 1, color=CCOL[c]) for c in CATS[::-1]]
    fig.legend(h, ["Critically endangered", "Endangered", "Vulnerable"], loc="lower center",
               bbox_to_anchor=(0.5, (B1 - 1.08) / H), ncol=3, frameon=False, fontsize=8, handlelength=1.0,
               columnspacing=1.5)
    print("rho", round(rho, 3), "top>=50%", med)
    print({c: round(float(100 * (s[s.cat == c].top >= 99.999).mean()), 1) for c in CATS})
    print(s.groupby("taxon").frac.median().round(3).to_dict())
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
