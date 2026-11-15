"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Pg, Rectangle
from matplotlib.colors import to_rgba
import tropics_style as st
import tropics_panels as tp
import tropics_robinson as rb
import tropics_temporal as t
from tropics_measure import MEASURE_PATH
from tropics_config import MOSAIC_LAT_TOP, DELIN_DEG

SEG_COL = ["#5F7391", "#F3C98B", "#E9A35A", "#D9752D", "#A84C0C"]
ISO = dict(NGA="Nigeria", SAU="Saudi Arabia", AGO="Angola", IND="India", BOL="Bolivia", TZA="Tanzania",
           SDN="Sudan", VNM="Vietnam", CHN="China", ZMB="Zambia", BRA="Brazil", GHA="Ghana", MMR="Myanmar",
           COD="DR Congo", MYS="Malaysia", MEX="Mexico", IDN="Indonesia", COL="Colombia", ECU="Ecuador",
           THA="Thailand", PER="Peru", PHL="Philippines", VEN="Venezuela", GTM="Guatemala", AUS="Australia",
           ETH="Ethiopia", KEN="Kenya", UGA="Uganda", CIV="Côte d'Ivoire", CMR="Cameroon", MOZ="Mozambique",
           MDG="Madagascar", BGD="Bangladesh", LAO="Laos", KHM="Cambodia", YEM="Yemen", OMN="Oman",
           EGY="Egypt", NER="Niger", MLI="Mali", BFA="Burkina Faso", SEN="Senegal", HND="Honduras",
           NIC="Nicaragua", CUB="Cuba", DOM="Dominican Republic", HTI="Haiti", PRY="Paraguay", TWN="Taiwan",
           SGP="Singapore", LKA="Sri Lanka", ZWE="Zimbabwe", MWI="Malawi", BEN="Benin", TGO="Togo",
           SLV="El Salvador", CRI="Costa Rica", PAN="Panama", RWA="Rwanda", BDI="Burundi", SSD="South Sudan",
           TCD="Chad", GIN="Guinea", ZAF="South Africa", NAM="Namibia", BWA="Botswana", ARE="United Arab Emirates")
SEG_LAB = ["Impervious by 2000", "Added 2000–2005", "Added 2005–2010", "Added 2010–2015",
           "Added 2015–2020"]


def cells():
    z = np.load(MEASURE_PATH, allow_pickle=False)
    a = np.load(t.OUT_DIR / "cell_epoch_area.npz")["area"]
    seg = np.column_stack([a[:, 1:5].sum(1), a[:, 5], a[:, 6], a[:, 7], a[:, 8]])
    lat = MOSAIC_LAT_TOP - (z["cell_row"].astype(float) + 0.5) * DELIN_DEG
    lon = -180.0 + (z["cell_col"].astype(float) + 0.5) * DELIN_DEG
    return lon, lat, seg, z["cell_city"].astype(int)


def city_table():
    d = tp.cities(sample=False)
    ce = np.load(MEASURE_PATH)["city_epoch"]
    seg = np.column_stack([ce[d.city_id, 1:5].sum(1)] + [ce[d.city_id, k] for k in (5, 6, 7, 8)])
    for k in range(5):
        d[f"s{k}"] = seg[:, k]
    d["tot"] = seg.sum(1)
    return d


def stacked_profile(ax, pos, seg, edges, horizontal=True):
    b = np.clip(np.digitize(pos, edges) - 1, 0, len(edges) - 2)
    mid = (edges[:-1] + edges[1:]) / 2
    V = np.column_stack([np.bincount(b, weights=seg[:, k], minlength=len(mid)) for k in range(5)])
    V = V / 1000.0
    cum = np.zeros(len(mid))
    for k in range(5):
        top = cum + V[:, k]
        if horizontal:
            ax.fill_between(mid, cum, top, color=SEG_COL[k], lw=0, zorder=2)
            ax.plot(mid, top, color=SEG_COL[k], lw=0.5, zorder=3)
        else:
            ax.fill_betweenx(mid, cum, top, color=SEG_COL[k], lw=0, zorder=2)
            ax.plot(top, mid, color=SEG_COL[k], lw=0.5, zorder=3)
        cum = top
    return cum.max()


def ranked_bars(ax, names, S, label_rot=90):
    P = 100 * S / S.sum(1, keepdims=True)
    x = np.arange(len(names))
    bot = np.zeros(len(names))
    for k in range(5):
        ax.bar(x, P[:, k], 0.78, bottom=bot, color=SEG_COL[k], edgecolor="white", lw=0.3)
        bot += P[:, k]
    ax.set_xticks(x, names, rotation=label_rot, ha="center", fontsize=8)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.6, len(names) - 0.4); ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Share of 2020 impervious area (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure(name="figure_profile_map_v3", lat_view=27.0, grid_deg=0.25, n_bar=25):
    st.apply_rc()
    lon, lat, seg, city = cells()
    W = tp.PAGE_W
    yv = float(rb.lat_to_y(lat_view))
    (xl0, xl1), _ = rb.band_limits(0.797)
    ML, MR, prof_w, gap_r = 0.05, 0.40, 0.55, 0.04
    map_w = W - ML - MR - prof_w - gap_r
    map_h = map_w * (2 * yv) / (xl1 - xl0)
    top_h, gap_t = 0.62, 0.04
    bar_h, gap_b, lab_h = 1.45, 0.62, 0.95
    H = 0.15 + top_h + gap_t + map_h + gap_b + bar_h + lab_h + 0.05
    fig = plt.figure(figsize=(W, H))
    y_map = (lab_h + bar_h + gap_b) / H
    ax = fig.add_axes([ML / W, y_map, map_w / W, map_h / H])
    rb.draw_globe(ax, zorder=0, top_face="white")

    gi = np.floor((lon + 180) / grid_deg).astype(int); gj = np.floor((MOSAIC_LAT_TOP - lat) / grid_deg).astype(int)
    k = gi * 100000 + gj
    u, inv = np.unique(k, return_inverse=True)
    G = np.column_stack([np.bincount(inv, weights=seg[:, q]) for q in range(5)])
    cumf = np.cumsum(G, 1) / G.sum(1, keepdims=True)
    dom = (cumf < 0.5).sum(1)
    ncol = int(round(360 / grid_deg)); nrow = int(np.ceil((MOSAIC_LAT_TOP + 25) / grid_deg))
    rgba = np.zeros((nrow, ncol, 4))
    rgba[u % 100000, u // 100000] = [to_rgba(SEG_COL[q]) for q in dom]
    rgba = (rgba * 255).astype(np.uint8)
    warped, ext = rb.warp_raster(rgba, (-180, 180, MOSAIC_LAT_TOP - nrow * grid_deg, MOSAIC_LAT_TOP),
                                 out_w=4200, lat_clip=(-rb.TROPIC, rb.TROPIC))
    ax.imshow(warped, extent=ext, origin="upper", interpolation="nearest", zorder=0.6, aspect="auto")
    lats = np.linspace(-lat_view, lat_view, 200)
    lx, ly = rb.project(np.full_like(lats, rb.LON_LO + 1e-6), lats)
    rx, ry = rb.project(np.full_like(lats, rb.LON_HI - 1e-6), lats)
    clip = Pg(np.vstack([np.column_stack([lx, ly]), np.column_stack([rx, ry])[::-1]]), closed=True,
              facecolor="none", edgecolor="#B9B6B1", lw=0.6, zorder=6)
    for sp in ax.spines.values():
        sp.set_visible(True)
    for la in (rb.TROPIC, -rb.TROPIC):
        ll = np.linspace(rb.LON_LO + 1e-6, rb.LON_HI - 1e-6, 240)
        xx, yy = rb.project(ll, np.full_like(ll, la))
        ax.plot(xx, yy, color="#8C8884", lw=0.6, ls=(0, (4, 3)), zorder=4)
    ax.set_xlim(xl0, xl1); ax.set_ylim(-yv, yv); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_color("#B9B6B1"); sp_.set_linewidth(0.6)
    tot = seg.sum(0)
    h = [Rectangle((0, 0), 1, 1, color=c) for c in SEG_COL]
    short = ["By 2000", "2000–2005", "2005–2010", "2010–2015", "2015–2020"]
    labs = [f"{l} ({100 * v / tot.sum():.1f}%)" for l, v in zip(short, tot)]
    fig.legend(h, labs, loc="upper left", bbox_to_anchor=(ML / W, y_map - 0.05 / H), ncol=5,
               frameon=False, fontsize=8, handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
               borderaxespad=0)
    fig.text(0.02 / W, 1 - 0.04 / H, "a", fontsize=st.MIN_PT + 1, fontweight="bold", va="top", ha="left")

    x, y = rb.project(lon, lat)
    dx = rb.X_MAX / 180.0
    tax = fig.add_axes([ML / W, y_map + (map_h + gap_t) / H, map_w / W, top_h / H])
    vmax = stacked_profile(tax, np.asarray(x), seg, np.arange(xl0, xl1 + dx, dx))
    tax.set_xlim(xl0, xl1); tax.set_ylim(0, vmax * 1.08)
    lt = [-90, -60, -30, 0, 30, 60, 90, 120, 150]
    tax.set_xticks(rb.project(np.array(lt, float), np.zeros(len(lt)))[0],
                   [f"{abs(v)}°{'W' if v < 0 else ('E' if v > 0 else '')}" for v in lt])
    tax.xaxis.tick_top(); tax.tick_params(axis="x", length=2, pad=1)
    tax.yaxis.tick_right(); tax.yaxis.set_label_position("right")
    tax.set_ylabel("10³ km²\nper 1°", fontsize=8, rotation=0, ha="left", va="center", labelpad=4)
    for s in ("left", "bottom"):
        tax.spines[s].set_visible(False)

    rax = fig.add_axes([(ML + map_w + gap_r) / W, y_map, prof_w / W, map_h / H])
    dy = float(rb.lat_to_y(1.0))
    vmax_r = stacked_profile(rax, np.asarray(y), seg, np.arange(-yv, yv + dy, dy), horizontal=False)
    rax.set_ylim(-yv, yv); rax.set_xlim(0, vmax_r * 1.08)
    la_t = [-20, -10, 0, 10, 20]
    rax.set_yticks(rb.lat_to_y(np.array(la_t, float)),
                   [f"{abs(v)}°{'S' if v < 0 else ('N' if v > 0 else '')}" for v in la_t])
    rax.yaxis.tick_right(); rax.tick_params(axis="y", length=2, pad=1)
    rax.set_xlabel("10³ km² per 1°", fontsize=8, labelpad=1)
    for s in ("left", "top"):
        rax.spines[s].set_visible(False)

    d = city_table()
    top = d.nlargest(n_bar, "tot")
    top = tp.city_names(top)
    S = top[[f"s{q}" for q in range(5)]].to_numpy()
    o = np.argsort(S[:, 0] / S.sum(1))
    bw = (W - 0.55 - 0.55) / 2
    bx = fig.add_axes([0.50 / W, lab_h / H, bw / W, bar_h / H])
    ranked_bars(bx, top["name"].fillna("Unnamed").to_numpy()[o], S[o])
    bx.text(-0.30 / bw, 1.06, "b", transform=bx.transAxes, fontsize=st.MIN_PT + 1, fontweight="bold",
            va="bottom", ha="left")
    bx.text(-0.30 / bw + 0.06, 1.06, f"{n_bar} largest agglomerations in 2020", transform=bx.transAxes,
            fontsize=st.MIN_PT, fontweight="bold", va="bottom", ha="left")

    c = d.groupby("country")[[f"s{q}" for q in range(5)] + ["tot"]].sum().nlargest(n_bar, "tot")
    Sc = c[[f"s{q}" for q in range(5)]].to_numpy()
    oc = np.argsort(Sc[:, 0] / Sc.sum(1))
    cx = fig.add_axes([(0.50 + bw + 0.55) / W, lab_h / H, bw / W, bar_h / H])
    cn = [ISO.get(n, n) for n in c.index.to_numpy()[oc]]
    ranked_bars(cx, cn, Sc[oc])
    cx.set_ylabel("")
    cx.text(-0.30 / bw, 1.06, "c", transform=cx.transAxes, fontsize=st.MIN_PT + 1, fontweight="bold",
            va="bottom", ha="left")
    cx.text(-0.30 / bw + 0.06, 1.06, f"{n_bar} countries with most agglomerated area", transform=cx.transAxes,
            fontsize=st.MIN_PT, fontweight="bold", va="bottom", ha="left")
    return tp.save(fig, name)


if __name__ == "__main__":
    print(figure())
