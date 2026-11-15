"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.transforms import offset_copy

import tropics_style as st
import tropics_basemap as bm
import tropics_rangecache as rc
from tropics_config import (INTERIM_DIR, TABLE_DIR, FIG_DIR, DELIN_DEG,
                            MOSAIC_LAT_TOP, THREAT_CATS)
from tropics_cities import load_city_cells
from tropics_panels import head, save as _psave

PAIRS = INTERIM_DIR / "species_city_pairs.npz"
RANGE_FILL = "#0F8A6E"
RANGE_EDGE = "#0A5A48"
URBAN = "#C05A0F"
CAT_COLORS = {"CR": "#8B0F35", "EN": "#C05A0F", "VU": "#D9A62E"}


def impact_table():
    import tropics_flags as tfl
    z = np.load(PAIRS, allow_pickle=True)
    sp_index = json.loads(str(z["sp_index"]))
    inv = {v: k for k, v in sp_index.items()}
    conv = np.zeros(len(sp_index))
    np.add.at(conv, z["pair_sp"], z["pair_conv"])
    cat, tax, sci, area = tfl.flags(PAIRS)
    d = pd.DataFrame(dict(
        sp_i=np.arange(len(sp_index)),
        sp_id=[inv[i] for i in range(len(sp_index))],
        sci=sci, category=cat, taxon=tax,
        converted_km2=conv, range_km2=area))
    d["share"] = d.converted_km2 / d.range_km2
    return d[d.converted_km2 > 0].copy()


def worst(n=20, cats=None, min_range_km2=50.0):
    d = impact_table()
    if cats:
        d = d[d.category.isin(cats)]
    d = d[d.range_km2 >= min_range_km2]
    return d.nlargest(n, "share").reset_index(drop=True)


ATLAS_GEOMS = INTERIM_DIR / "atlas_geoms.npz"


def _atlas_geoms(sp_ids):
    import shapely
    z = np.load(ATLAS_GEOMS, allow_pickle=False)
    ids, counts, lens, buf = z["ids"], z["counts"], z["lens"], z["buf"]
    raw, at, k = buf.tobytes(), 0, 0
    want = {str(x): x for x in sp_ids}
    out = {}
    for sid, n in zip(ids, counts):
        parts = []
        for _ in range(int(n)):
            ln = int(lens[k]); parts.append(shapely.from_wkb(raw[at:at + ln]))
            at += ln; k += 1
        if str(sid) in want:
            out[want[str(sid)]] = parts
    return out


def _geoms_for(sp_ids):
    if not rc.INDEX.exists() and ATLAS_GEOMS.exists():
        return _atlas_geoms(sp_ids), None
    sp, parts = rc.load_index()
    want = sp[sp.sp_id.isin(set(sp_ids))]
    idx = set(want.sp_i.tolist())
    part_sp, off, ln = parts
    keep = np.flatnonzero(np.isin(part_sp, list(idx)))
    out = {}
    for si, g in rc.iter_parts((part_sp[keep], off[keep], ln[keep]), batch=400):
        for a, b in zip(si, g):
            out.setdefault(int(a), []).append(b)
    id_of = dict(zip(want.sp_i, want.sp_id))
    return {id_of[k]: v for k, v in out.items() if k in id_of}, sp


def _city_cells_index():
    rr, cc, ids = load_city_cells()
    conv = np.load(INTERIM_DIR / "city_measurements.npz")["cell_conv"]
    o = np.argsort(ids, kind="stable")
    return rr[o], cc[o], ids[o], conv[o]


def _cells_of(city_ids, rr, cc, ids, conv):
    lo = np.searchsorted(ids, city_ids, "left")
    hi = np.searchsorted(ids, city_ids, "right")
    if not len(lo):
        e = np.array([])
        return e, e, e
    r = np.concatenate([rr[a:b] for a, b in zip(lo, hi)])
    c = np.concatenate([cc[a:b] for a, b in zip(lo, hi)])
    v = np.concatenate([conv[a:b] for a, b in zip(lo, hi)])
    lat = MOSAIC_LAT_TOP - (r + 0.5) * DELIN_DEG
    lon = -180.0 + (c + 0.5) * DELIN_DEG
    return lon, lat, v


def _draw_polys(ax, geoms, fc, ec, lw=0.35, alpha=1.0, zorder=3):
    patches = []
    for g in geoms:
        for p in bm._polys(g):
            patches.append(MplPolygon(np.asarray(p.exterior.coords), closed=True))
    if patches:
        ax.add_collection(PatchCollection(patches, facecolors=fc, edgecolors=ec,
                                          linewidths=lw, alpha=alpha,
                                          zorder=zorder))


SPARSE_FILL = 0.06


def _patch_frames(g, lon, lat, cv):
    import shapely, math
    if not g:
        return [], 1.0
    u = shapely.union_all(g)
    parts = list(u.geoms) if hasattr(u, "geoms") else [u]
    x0, y0, x1, y1 = u.bounds
    kx = 111.32 * math.cos(math.radians(0.5 * (y0 + y1)))
    bbox = max((x1 - x0) * kx * (y1 - y0) * 111.32, 1e-9)
    fill = sum(p.area for p in parts) * kx * 111.32 / bbox
    out = []
    for p in parts:
        if len(lon):
            m = shapely.contains_xy(p, lon, lat)
            conv = float(cv[m].sum()) if cv is not None else float(m.sum())
        else:
            conv = 0.0
        out.append((conv, p))
    out.sort(key=lambda t: (-t[0], -t[1].area))
    return out, fill


def _locator(ax, g, focus, aspect):
    import shapely
    u = shapely.union_all(g)
    x0, y0, x1, y1 = u.bounds
    span = max(x1 - x0, y1 - y0) * 0.70
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w_ax, top = 0.30, 0.972
    h_ax = w_ax * aspect
    ix = ax.inset_axes([0.665, top - h_ax, w_ax, h_ax])
    ix.set_xlim(cx - span, cx + span)
    ix.set_ylim(cy - span, cy + span)
    ix.set_xticks([]); ix.set_yticks([])
    bmg, _ = bm._load_countries()
    _draw_polys(ix, bmg, "white", "#C9C7C3", lw=0.2, zorder=1)
    for p in (u.geoms if hasattr(u, "geoms") else [u]):
        c = p.centroid
        ix.plot([c.x], [c.y], "o", ms=2.2, mfc=RANGE_FILL, mec=RANGE_EDGE,
                mew=0.35, zorder=4)
    fx0, fy0, fx1, fy1 = focus.bounds
    pad = max(fx1 - fx0, fy1 - fy0) * 1.4 + 0.02
    ix.add_patch(Rectangle((fx0 - pad, fy0 - pad),
                           (fx1 - fx0) + 2 * pad, (fy1 - fy0) + 2 * pad,
                           fill=False, ec=URBAN, lw=0.7, zorder=6))
    for s_ in ix.spines.values():
        s_.set_color("#B9B6B1"); s_.set_linewidth(0.4)
    ix.set_facecolor("white")
    n = len(list(u.geoms)) if hasattr(u, "geoms") else 1
    ix.text(1.0, -0.06, f"{n} patches", transform=ix.transAxes, fontsize=4.2,
            ha="right", va="top", color=st.MUTED, zorder=8)
    return ix


def species_panel(ax, row, g, pair_sp, pair_city, cells, aspect):
    import shapely
    rr, cc, ids, conv = cells
    m = pair_sp == row.sp_i
    city_ids = np.unique(pair_city[m])
    lon, lat, cv = _cells_of(np.sort(city_ids), rr, cc, ids, conv)
    keep = cv > 0
    lon, lat = lon[keep], lat[keep]
    patches, fill = _patch_frames(g, lon, lat, cv[keep] if len(cv) else cv)
    sparse = bool(g) and fill < SPARSE_FILL and len(patches) > 1
    frame = [patches[0][1]] if sparse else g
    if frame:
        xs = np.concatenate([np.asarray(p.exterior.coords)[:, 0]
                             for gg in frame for p in bm._polys(gg)])
        ys = np.concatenate([np.asarray(p.exterior.coords)[:, 1]
                             for gg in frame for p in bm._polys(gg)])
    else:
        xs, ys = lon, lat
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    span = max((x1 - x0), (y1 - y0))
    pad = span * 0.28 + min(0.25, span * 1.5)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    sx, sy = x1 - x0, y1 - y0
    if sx / sy > aspect:
        k = sx / aspect - sy
        y0 -= k / 2; y1 += k / 2
    else:
        k = sy * aspect - sx
        x0 -= k / 2; x1 += k / 2
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    bmg, _ = bm._load_countries()
    _draw_polys(ax, bmg, "white", "#C9C7C3", lw=0.25, zorder=1)
    _draw_polys(ax, g, RANGE_FILL, RANGE_EDGE, lw=0.4, alpha=0.55, zorder=3)
    if len(lon):
        inside = np.zeros(len(lon), bool)
        for gg in g:
            inside |= shapely.contains_xy(gg, lon, lat)
        ax.plot(lon[~inside], lat[~inside], ".", ms=1.0, color="#E8B98F",
                mew=0, zorder=4)
        ax.plot(lon[inside], lat[inside], ".", ms=1.4, color=URBAN, mew=0,
                zorder=6)
    if sparse:
        _locator(ax, g, patches[0][1], aspect)
    for s_ in ax.spines.values():
        s_.set_color("#B9B6B1"); s_.set_linewidth(0.5)
    cat = CAT_COLORS.get(row.category, st.MUTED)
    ax.text(0.0, 1.20, row.sci, transform=ax.transAxes, fontsize=5.9,
            style="italic", ha="left", va="bottom", color=st.INK)
    ax.text(0.0, 1.02, f"{row.share*100:.1f}% converted",
            transform=ax.transAxes, fontsize=5.4, ha="left", va="bottom",
            color=st.MUTED)
    ax.text(1.0, 1.02, row.category, transform=ax.transAxes, fontsize=5.4,
            ha="right", va="bottom", color=cat, fontweight="bold")
    from tropics_frame import north_arrow as _na
    _na(ax, x0, x1, y0, y1)
    latm = 0.5 * (y0 + y1)
    kmdeg = 111.32 * np.cos(np.radians(latm))
    target = (x1 - x0) * kmdeg * 0.28
    km = min([5, 10, 25, 50, 100, 250, 500], key=lambda v: abs(v - target))
    dx = km / kmdeg
    ax.plot([x0 + (x1 - x0) * 0.06, x0 + (x1 - x0) * 0.06 + dx],
            [y0 + (y1 - y0) * 0.07] * 2, color=st.INK, lw=1.1, zorder=9)
    ax.text(x0 + (x1 - x0) * 0.06 + dx / 2, y0 + (y1 - y0) * 0.095,
            f"{km:g} km", fontsize=4.8, ha="center", va="bottom",
            color=st.INK, zorder=9)
    return x0, x1, y0, y1


def fig_worst_species(name="fig_worst_species_v2", n=20,
                      cats=("CR", "EN", "VU")):
    st.apply_rc()
    w = worst(n=n, cats=list(cats))
    geoms, _ = _geoms_for(w.sp_id.tolist())
    z = np.load(PAIRS, allow_pickle=True)
    pair_sp, pair_city, pair_conv = z["pair_sp"], z["pair_city"], z["pair_conv"]
    rr, cc, ids, conv = _city_cells_index()

    ncol, nrow = 4, int(np.ceil(n / 4))
    fig = plt.figure(figsize=(7.0, 1.72 * nrow + 0.45))
    L, R, T, B = 0.020, 0.972, 0.945, 0.075
    cw = (R - L) / ncol - 0.018
    chh = (T - B) / nrow - 0.060
    for i, row in w.iterrows():
        r, c = divmod(i, ncol)
        ax = fig.add_axes([L + c * ((R - L) / ncol),
                           B + (nrow - 1 - r) * ((T - B) / nrow), cw, chh])
        species_panel(ax, row, geoms.get(row.sp_id, []), pair_sp, pair_city,
                      (rr, cc, ids, conv), aspect=(cw * 7.0) / (chh * fig.get_figheight()))

    from tropics_panels import suptitle
    suptitle(fig, "The twenty threatened species with the largest share of "
             "mapped range converted, 2000 to 2020", x_in=L * 7.0, fs=9.6)
    handles = [Rectangle((0, 0), 1, 1, facecolor=RANGE_FILL, alpha=0.55),
               plt.Line2D([0], [0], marker="s", ls="none", ms=3.6, color=URBAN),
               plt.Line2D([0], [0], marker="s", ls="none", ms=3.6, color="#E8B98F")]
    labels = ["expert range polygon",
              "urban land added inside the range, 2000\u20132020",
              "the same footprints outside the range"]
    fig.legend(handles, labels, ncol=3, fontsize=6.4, loc="lower center",
               bbox_to_anchor=(0.5, 0.004), handlelength=1.1, handleheight=0.9,
               columnspacing=1.6, frameon=False)
    return _psave(fig, name)


EPOCH_COLORS = {5: "#F2AE6B", 6: "#E4863C", 7: "#C85B15", 8: "#8A3703"}
EPOCH_YEARS = {5: "2000-2005", 6: "2005-2010", 7: "2010-2015", 8: "2015-2020"}
PRE = "#9B9791"


def _tiles_for(x0, x1, y0, y1):
    from tropics_config import GISD30_SRC
    from tropics_grid import tile_name
    out, missing = [], []
    for lat_n in range(int(np.ceil(y1 / 5.0)) * 5, int(np.floor(y0 / 5.0)) * 5, -5):
        for lon_w in range(int(np.floor(x0 / 5.0)) * 5,
                           int(np.ceil(x1 / 5.0)) * 5, 5):
            p = GISD30_SRC / f"GISD30_1985-2020_{tile_name(lon_w, lat_n)}.tif"
            (out if p.exists() else missing).append((p, lon_w, lat_n))
    return out, missing


def read_epochs(x0, x1, y0, y1, max_px=1600):
    import rasterio
    from rasterio.windows import from_bounds
    from tropics_config import GISD30_SRC
    span = max(x1 - x0, y1 - y0)
    res = span / max_px
    W = max(int(round((x1 - x0) / res)), 8)
    H = max(int(round((y1 - y0) / res)), 8)
    out = np.zeros((H, W), np.uint8)
    have, missing = _tiles_for(x0, x1, y0, y1)
    if not have:
        raise FileNotFoundError(
            "no GISD30 tile covering "
            f"{x0:.3f}..{x1:.3f} E, {y0:.3f}..{y1:.3f} N under {GISD30_SRC}. "
            f"Expected {', '.join(q.name for q, _, _ in missing)}. "
            "Set TROPICS_GISD30 to the tile folder, or run this figure on a "
            "machine that holds it. Returning an empty raster would draw a "
            "blank panel that looks finished.")
    for p, lon_w, lat_n in have:
        with rasterio.open(p) as src:
            b = src.bounds
            ax0, ax1 = max(x0, b.left), min(x1, b.right)
            ay0, ay1 = max(y0, b.bottom), min(y1, b.top)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            win = from_bounds(ax0, ay0, ax1, ay1, src.transform)
            oh = max(int(round((ay1 - ay0) / res)), 1)
            ow = max(int(round((ax1 - ax0) / res)), 1)
            a = src.read(1, window=win, out_shape=(oh, ow))
            r0 = int(round((y1 - ay1) / res)); c0 = int(round((ax0 - x0) / res))
            r1, c1 = min(r0 + oh, H), min(c0 + ow, W)
            if r1 > r0 and c1 > c0:
                sub = out[r0:r1, c0:c1]
                np.maximum(sub, a[:r1 - r0, :c1 - c0], out=sub)
    if missing:
        warnings.warn(
            f"{len(missing)} of {len(have) + len(missing)} GISD30 tiles for "
            f"{x0:.3f}..{x1:.3f} E, {y0:.3f}..{y1:.3f} N are absent "
            f"({', '.join(q.name for q, _, _ in missing)}); the window is drawn "
            "from the tiles that are present.", RuntimeWarning, stacklevel=2)
    return out


def _epoch_rgba(a, pre_color=None):
    from matplotlib.colors import to_rgb
    rgba = np.zeros(a.shape + (4,), np.uint8)
    pre = (a >= 1) & (a <= 4)
    pc = pre_color or PRE
    rgba[pre] = np.array(list(np.round(np.array(to_rgb(pc)) * 255)) + [255], np.uint8)
    for e, c in EPOCH_COLORS.items():
        m = a == e
        if m.any():
            rgba[m] = np.array(list(np.round(np.array(to_rgb(c)) * 255)) + [255],
                               np.uint8)
    return rgba


CITY_WINDOW_KM = 480.0


def fig_city_growth(name="fig_city_growth_v2", cities_wanted=None, max_px=1400,
                    window_km=CITY_WINDOW_KM):
    import shapely
    from tropics_panels import cities as _cities, city_names
    st.apply_rc()
    d = city_names(_cities())
    if cities_wanted is None:
        cities_wanted = ["Guangzhou", "Ho Chi Minh City", "Jakarta", "Bengaluru",
                         "Lagos", "Addis Ababa", "Campinas", "Manila"]
    sel = (d[d.name.isin(cities_wanted)]
           .drop_duplicates("name")
           .set_index("name").reindex(cities_wanted).dropna(subset=["city_id"])
           .reset_index())

    z = np.load(PAIRS, allow_pickle=True)
    sp_index = json.loads(str(z["sp_index"]))
    inv = {v: k for k, v in sp_index.items()}
    pair_sp, pair_city = z["pair_sp"], z["pair_city"]
    sp_area = z["sp_area"]
    meta = json.loads(str((TABLE_DIR / "cities_cost_meta.json").read_text()))
    thr = meta["narrow_threshold_km2"]

    rr, cc, ids, conv = _city_cells_index()
    want_ids = []
    per_city = {}
    for _, row in sel.iterrows():
        m = pair_city == int(row.city_id)
        s = np.unique(pair_sp[m])
        s = s[sp_area[s] <= thr]
        per_city[int(row.city_id)] = s
        want_ids += [inv[int(i)] for i in s]
    geoms, _ = _geoms_for(sorted(set(want_ids)))

    ncol, nrow = 4, 2
    W = 7.0
    L_in, R_in = 0.09, 0.26
    col_gap, row_gap, label_in = 0.075, 0.115, 0.30
    head_in, foot_in = 0.46, 0.52
    pw_in = (W - L_in - R_in - (ncol - 1) * col_gap) / ncol
    ph_in = pw_in
    fig_h = (head_in + nrow * (ph_in + label_in) + (nrow - 1) * row_gap
             + foot_in)
    fig = plt.figure(figsize=(W, fig_h))
    L, R = L_in / W, 1 - R_in / W
    for i, row in sel.iterrows():
        r, c = divmod(i, ncol)
        x = L_in + c * (pw_in + col_gap)
        y = fig_h - head_in - (r + 1) * (ph_in + label_in) - r * row_gap
        ax = fig.add_axes([x / W, y / fig_h, pw_in / W, ph_in / fig_h])
        lon, lat, cv = _cells_of(np.array([int(row.city_id)]), rr, cc, ids, conv)
        lon_c, lat_c = 0.5 * (lon.min() + lon.max()), 0.5 * (lat.min() + lat.max())
        dlat = window_km / 110.57
        dlon = window_km / (111.32 * np.cos(np.radians(lat_c)))
        x0, x1 = lon_c - dlon / 2, lon_c + dlon / 2
        y0, y1 = lat_c - dlat / 2, lat_c + dlat / 2
        a = read_epochs(x0, x1, y0, y1, max_px=max_px)
        ax.imshow(_epoch_rgba(a), extent=(x0, x1, y0, y1), origin="upper",
                  interpolation="nearest", zorder=2)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        n_here = 0
        for si in per_city[int(row.city_id)]:
            for g in geoms.get(inv[int(si)], []):
                if g.intersects(shapely.box(x0, y0, x1, y1)):
                    _draw_polys(ax, [g], "none", "#0A5A48", lw=0.6, zorder=6)
                    n_here += 1
        for s in ax.spines.values():
            s.set_color("#B9B6B1"); s.set_linewidth(0.5)
        from tropics_frame import north_arrow, scale_bar
        north_arrow(ax, x0, x1, y0, y1)
        scale_bar(ax, x0, x1, y0, y1, km=100)
        ax.text(0.0, 1.15, f"{row['name']}, {row.country}", transform=ax.transAxes,
                fontsize=6.8, fontweight="bold", ha="left", va="bottom",
                color=st.INK)
        ax.text(0.0, 1.02, f"{row.urban_added_km2:,.0f} km$^2$ added  ·  "
                f"{n_here} narrow range{'s' if n_here != 1 else ''}",
                transform=ax.transAxes, fontsize=5.8, ha="left", va="bottom",
                color=st.MUTED)

    from matplotlib.lines import Line2D as _L2D
    bar_w = (100.0 / window_km) * pw_in / W
    yb = 0.13 / fig_h
    fig.add_artist(_L2D([R - bar_w, R], [yb, yb], color=st.INK, lw=1.5,
                        solid_capstyle="butt", transform=fig.transFigure))
    fig.text(R - bar_w / 2, yb + 0.035 / fig_h, f"100 km, every panel "
             f"{window_km:,.0f} \u00d7 {window_km:,.0f} km", fontsize=6.2,
             ha="center", va="bottom", color=st.INK)

    handles = [Rectangle((0, 0), 1, 1, facecolor=PRE)] + \
              [Rectangle((0, 0), 1, 1, facecolor=EPOCH_COLORS[e]) for e in (5, 6, 7, 8)] + \
              [plt.Line2D([0], [0], color="#0A5A48", lw=1.0)]
    labels = ["impervious by 2000"] + [EPOCH_YEARS[e] for e in (5, 6, 7, 8)] + \
             ["narrow-range boundary"]
    fig.legend(handles, labels, ncol=6, fontsize=6.2, loc="lower left",
               bbox_to_anchor=(L, 0.09 / fig_h), handlelength=1.1,
               handleheight=0.85, columnspacing=1.2, frameon=False)
    from tropics_panels import suptitle
    suptitle(fig, "Eight tropical cities at 30 m, with the narrow-range species "
             "boundaries that cross them", x_in=L_in)
    return _psave(fig, name)
