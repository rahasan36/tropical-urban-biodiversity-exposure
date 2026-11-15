"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tropics_style as st
import tropics_frame as tf
import tropics_bandmaps as m2
import tropics_surfaces as sf
import tropics_panels as tp
from tropics_config import TABLE_DIR, INTERIM_DIR

PAGE_W = tp.PAGE_W
NULL, STAT = tp.NULL, tp.STAT
FORMS = ["infill", "edge", "leapfrog"]
FOUR = [1, 2, 4, 5]


def _nums():
    return json.loads((TABLE_DIR / "numbers.json").read_text())


def _robust():
    return json.loads((TABLE_DIR / "cf_robust.json").read_text())


def _release():
    r = json.loads((TABLE_DIR / "cf_release.json").read_text())
    a = np.asarray(r["alphas"], float)
    m = np.asarray(r["reps_m"], float)
    p_ = np.asarray(r["reps_p"], float)
    ok = ~np.isnan(p_).any(axis=1)
    pm = np.asarray(r["point"]["measured"], float)
    pp = np.asarray(r["point"]["predicted"], float)
    mlo, mhi = np.percentile(m, [2.5, 97.5], axis=0)
    plo, phi = np.percentile(p_[ok], [2.5, 97.5], axis=0)
    diff = p_[ok] - m[ok]
    dlo, dhi = np.percentile(diff, [2.5, 97.5], axis=0)
    return dict(alphas=a, alpha_hat=float(r["alpha_hat"]), n=int(r["n"]),
                n_reps=int(ok.sum()), smearing=float(r["smearing"]),
                measured=pm, predicted=pp,
                m_lo=mlo, m_hi=mhi, p_lo=plo, p_hi=phi,
                d=pp - pm, d_lo=dlo, d_hi=dhi)


def _plan():
    return json.loads((TABLE_DIR / "plan4.json").read_text())


ROBIN_MARGINS = (0.07, 0.07)
ROBIN_PROF_W = 0.34
ROBIN_GAP_P = 0.035


def _band_layout(ML, MR, prof_w, gap_p):
    if _is_robin():
        return ROBIN_MARGINS[0], ROBIN_MARGINS[1], ROBIN_PROF_W, ROBIN_GAP_P
    return ML, MR, prof_w, gap_p

def _band_axes(fig, W, fig_h, y_top, ML=0.06, MR=0.15, left_w=0.0, right_w=0.0,
               gap=0.055, land=None):
    live = W - ML - MR
    n_side = (left_w > 0) + (right_w > 0)
    map_w = live - left_w - right_w - gap * n_side
    robin = _is_robin()
    if robin:
        import tropics_robinson as rb
        map_h = map_w * rb.band_aspect(ROBIN_COVER)
    else:
        map_h = map_w * (tf.LAT1 - tf.LAT0) / (tf.LON1 - tf.LON0)
    y_map = y_top - map_h
    x_map = ML + (left_w + gap if left_w else 0.0)
    ax = fig.add_axes([x_map / W, y_map / fig_h, map_w / W, map_h / fig_h])
    if robin:
        rb.draw_globe(ax, zorder=0, top_face=land)
        rb.frame_band(ax, ROBIN_COVER)
        (_, _), (py0, py1) = rb.band_limits(ROBIN_COVER)
        y_trop, h_trop = y_map, map_h
    else:
        ax.set_xlim(tf.LON0, tf.LON1); ax.set_ylim(tf.LAT0, tf.LAT1)
        ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#B9B6B1"); sp.set_linewidth(0.5)
        tf._graticule(ax); tf._draw_land(ax, color="white", context=True)
        span = tf.LAT1 - tf.LAT0
        y_trop = y_map + map_h * (tf.TROP0 - tf.LAT0) / span
        h_trop = map_h * (tf.TROP1 - tf.TROP0) / span
    lax = rax = None
    if left_w:
        lax = fig.add_axes([ML / W, y_trop / fig_h, left_w / W, h_trop / fig_h])
    if right_w:
        rax = fig.add_axes([(x_map + map_w + gap) / W, y_trop / fig_h,
                            right_w / W, h_trop / fig_h])
    return dict(ax=ax, left=lax, right=rax, map_w=map_w, map_h=map_h,
                y_map=y_map, x_map=x_map)


PROFILE_BIN = 2.0

PROJECTION = "band"
ROBIN_COVER = 0.797


def _is_robin():
    return PROJECTION == "robinson"


def _lonlat(lon, lat):
    if _is_robin():
        import tropics_robinson as rb
        return rb.project(lon, lat)
    return np.asarray(lon, float), np.asarray(lat, float)


def _draw_raster(ax, rgba, tropics_only=True):
    import tropics_bandmaps as _m2
    if not _is_robin():
        return _m2._draw(ax, rgba, tropics_only=tropics_only)
    import tropics_robinson as rb
    clip = (-rb.TROPIC, rb.TROPIC) if tropics_only else None
    warped, ext = rb.warp_raster(rgba, _m2.EXTENT, out_w=2600, lat_clip=clip)
    return ax.imshow(warped, extent=ext, origin="upper",
                     interpolation="nearest", zorder=0.55, aspect="auto")


def _tropic_lines(ax):
    import tropics_robinson as rb
    for la in (rb.TROPIC, -rb.TROPIC):
        if _is_robin():
            lons = np.linspace(rb.LON_LO + 1e-6, rb.LON_HI - 1e-6, 240)
            xx, yy = rb.project(lons, np.full_like(lons, la))
            ax.plot(xx, yy, color="#8C8884", lw=0.7, ls=(0, (4, 3)), zorder=4)
        else:
            ax.axhline(la, color="#8C8884", lw=0.7, ls=(0, (4, 3)), zorder=4)


def _nice_tick(top):
    m = top / 1.06
    k = 10 ** np.floor(np.log10(m))
    for f in (5, 2.5, 2, 1):
        if f * k <= m:
            return f * k
    return k


def _profile(pax, ctr, vals, color, label, flip, top, unit,
             label_ha="center", unit_ha="center", bars=None):
    if _is_robin():
        import tropics_robinson as rb
        from matplotlib.patches import Rectangle as _R
        bc, bv, bh = bars if bars is not None else (ctr, vals, 1.0)
        for c, v in zip(np.asarray(bc, float), np.asarray(bv, float)):
            y_lo = float(rb.lat_to_y(c - bh / 2.0))
            y_hi = float(rb.lat_to_y(c + bh / 2.0))
            pax.add_patch(_R((0, y_lo), v, y_hi - y_lo, facecolor=color,
                             edgecolor="white", lw=0.3, zorder=2))
        (_, _), (py0, py1) = rb.band_limits(ROBIN_COVER)
        pax.set_ylim(py0, py1)
    else:
        if bars is not None:
            bc, bv, bh = bars
            pax.barh(bc, bv, height=bh * 0.86, color=color, edgecolor="white",
                     lw=0.35, zorder=2)
            pax.plot(vals, ctr, color=st.INK, lw=0.5, zorder=4)
        else:
            pax.fill_betweenx(ctr, 0, vals, color=color, lw=0)
            pax.plot(vals, ctr, color=st.INK, lw=0.6)
        pax.set_ylim(tf.TROP0, tf.TROP1)
    pax.set_xlim(top, 0) if flip else pax.set_xlim(0, top)
    pax.set_yticks([])
    if _is_robin():
        t = _nice_tick(top)
        pax.set_xticks([0, t])
        pax.set_xticklabels(["0", f"{t:g}"])
        pax.tick_params(axis="x", length=2, width=0.5, pad=1.5,
                        labelsize=st.MIN_PT, colors=st.INK)
    else:
        pax.set_xticks([])
    for sp in ("top", "left", "right"):
        pax.spines[sp].set_visible(False)
    pax.spines["bottom"].set_color(st.MUTED)
    anchor = {"center": 0.5, "left": 0.0, "right": 1.0}
    l_x, l_ha = anchor[label_ha], label_ha
    if _is_robin():
        l_x, l_ha = (1.0, "left") if flip else (0.0, "right")
    pax.text(l_x, 1.05, label, transform=pax.transAxes,
             fontsize=st.MIN_PT, ha=l_ha, va="bottom", color=st.INK)
    unit_txt = unit
    u_x, u_ha, u_y = anchor[unit_ha], unit_ha, -0.09
    if _is_robin():
        u_x, u_ha = (1.0, "left") if flip else (0.0, "right")
        pax.annotate(unit_txt, xy=(u_x, 0), xycoords="axes fraction",
                     xytext=(0, -(st.MIN_PT + 5)), textcoords="offset points",
                     fontsize=st.MIN_PT, ha=u_ha, va="top", color=st.INK,
                     linespacing=1.2)
        return
    pax.text(u_x, u_y, unit_txt, transform=pax.transAxes,
             fontsize=st.MIN_PT, ha=u_ha, va="top", color=st.MUTED,
             linespacing=1.5)


def _ramp_cmap(ramp, name="ramp"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, ramp)


def _continuous_raster(a, cmap, norm, floor):
    v = np.where(np.isfinite(a), a, np.nan)
    ok = np.isfinite(v) & (v >= floor)
    rgba = np.zeros(v.shape + (4,), np.uint8)
    rgba[ok] = (cmap(norm(np.clip(v[ok], norm.vmin, norm.vmax))) * 255).astype(np.uint8)
    rgba[ok, 3] = 255
    return rgba


def _points_continuous(ax, d, col, cmap, norm, size_by="urban_km2_final", zorder=6):
    import tropics_citymaps as cm
    v = np.clip(d[col].values.astype(float), norm.vmin, norm.vmax)
    x, y = _lonlat(d.lon.values, d.lat.values)
    ax.scatter(x, y, s=cm._size(d[size_by].values), c=cmap(norm(v)),
               linewidths=0.0, zorder=zorder, rasterized=True)


def _cbar(ax, x_frac, w_frac, below_in, cmap, norm, ticks, label, fmt=None):
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.ticker import NullLocator
    fig = ax.figure
    F = fig.get_size_inches()[1]
    bb = ax.get_position()
    thick = 0.085
    cax = fig.add_axes([bb.x0 + x_frac * bb.width, bb.y0 - (below_in + thick) / F,
                        w_frac * bb.width, thick / F])
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal",
                      ticks=ticks, extend="neither")
    cax.xaxis.set_minor_locator(NullLocator())
    cb.set_ticklabels([fmt(t) if fmt else f"{t:g}" for t in ticks])
    cax.tick_params(labelsize=st.MIN_PT, length=2, width=0.5, colors=st.INK, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cax.annotate(label, xy=(0.0, 1.0), xycoords="axes fraction",
                 xytext=(0, 2.0), textcoords="offset points", ha="left",
                 va="bottom", fontsize=st.MIN_PT, fontweight="normal",
                 color=st.INK)
    return cb


def _vlegend(ax, x_lon, y_lat, title, colors, labels, w_deg=58.0,
             sw_deg=(7.0, 2.2), row_deg=4.4, fs=6.0, pad=(1.8, 1.2)):
    from matplotlib.patches import Rectangle as _R
    sw_w, sw_h = sw_deg
    lines = title.split("\n")
    line_deg = 4.3
    head_h = pad[1] + line_deg * len(lines) + 1.0
    h = head_h + row_deg * len(colors) + pad[1]
    ax.add_patch(_R((x_lon, y_lat - h), w_deg, h, facecolor="white",
                    edgecolor="#D0CDC8", lw=0.5, zorder=10, clip_on=False))
    for i, ln in enumerate(lines):
        ax.text(x_lon + pad[0], y_lat - pad[1] - line_deg * i, ln, fontsize=fs,
                fontweight="bold", ha="left", va="top", color=st.INK, zorder=12,
                clip_on=False)
    y = y_lat - head_h
    for c, lab in zip(colors, labels):
        ax.add_patch(_R((x_lon + pad[0], y - sw_h), sw_w, sw_h, facecolor=c,
                        edgecolor="#B9B6B1", lw=0.3, zorder=12, clip_on=False))
        ax.text(x_lon + pad[0] + sw_w + 1.8, y - sw_h * 0.55, lab, fontsize=fs,
                ha="left", va="center", color=st.INK, zorder=12, clip_on=False)
        y -= row_deg


def _class_labels(breaks, unit=""):
    lab = [f"{breaks[i]:g}\u2013{breaks[i+1]:g}" for i in range(len(breaks) - 1)]
    return lab + [f"> {breaks[-1]:g}"]


def _lat_median(lat, val, deg):
    lo = np.floor(tf.TROP0 / deg) * deg
    hi = np.ceil(tf.TROP1 / deg) * deg
    edges = np.arange(lo, hi + deg / 2, deg)
    idx = np.clip(np.digitize(lat, edges[1:-1]), 0, len(edges) - 2)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    out = np.zeros(len(ctr))
    for i in range(len(ctr)):
        m = idx == i
        if m.any():
            out[i] = np.median(val[m])
    return ctr, out


def _lat_profile(arr, lat_top, lat_bot, deg=1.0):
    rows_per = int(round(deg / sf.DISP_DEG))
    n = arr.shape[0] // rows_per
    v = np.nansum(arr[:n * rows_per].reshape(n, rows_per, -1), axis=(1, 2))
    edges = lat_top - np.arange(n + 1) * deg
    return v, edges


def _stacked_profiles(fig, W, fig_h, band, base, add, c_base, c_add, top_h, top_gap, ML):
    import tropics_robinson as rb
    from matplotlib.patches import Patch
    ax, rax = band["ax"], band["right"]
    per = int(round(1.0 / sf.DISP_DEG))
    ncol = base.shape[1] // per
    cb = np.nansum(np.nan_to_num(base)[:, :ncol * per].reshape(base.shape[0], ncol, per), axis=(0, 2)) / 1e3
    ca = np.nansum(np.nan_to_num(add)[:, :ncol * per].reshape(add.shape[0], ncol, per), axis=(0, 2)) / 1e3
    lon = -180.0 + (np.arange(ncol) + 0.5)
    lon = np.where(lon < rb.LON_LO, lon + 360.0, lon)
    o = np.argsort(lon)
    lon, cb, ca = lon[o], cb[o], ca[o]
    x = np.asarray(rb.project(lon, np.zeros_like(lon))[0])
    xm = band["x_map"]
    tax = fig.add_axes([xm / W, (band["y_map"] + band["map_h"] + top_gap) / fig_h,
                        band["map_w"] / W, top_h / fig_h])
    tax.fill_between(x, 0, cb, color=c_base, lw=0, zorder=2)
    tax.fill_between(x, cb, cb + ca, color=c_add, lw=0, zorder=2)
    tax.plot(x, cb + ca, color=c_add, lw=0.5, zorder=3)
    tax.set_xlim(ax.get_xlim())
    vmax = float((cb + ca).max())
    tax.set_ylim(0, vmax * 1.08)
    t = _nice_tick(vmax * 1.08)
    tax.set_yticks([0, t], ["0", f"{t:g}"])
    tax.yaxis.tick_right()
    tax.yaxis.set_label_position("right")
    tax.tick_params(axis="y", length=2, width=0.5, pad=1.5, labelsize=st.MIN_PT, colors=st.INK)
    lt = [-90, -60, -30, 0, 30, 60, 90, 120, 150]
    lt = [v for v in lt if rb.LON_LO < (v if v >= rb.LON_LO else v + 360) < rb.LON_HI]
    lx = rb.project(np.array([v if v >= rb.LON_LO else v + 360 for v in lt], float), np.zeros(len(lt)))[0]
    tax.set_xticks(lx, [f"{abs(v)}°{'W' if v < 0 else ('E' if 0 < v < 180 else '')}" for v in lt])
    tax.xaxis.tick_top()
    tax.tick_params(axis="x", length=2, width=0.5, pad=1.0, labelsize=st.MIN_PT, colors=st.INK)
    for sp in ("left", "bottom"):
        tax.spines[sp].set_visible(False)
    for sp in ("top", "right"):
        tax.spines[sp].set_color(st.MUTED); tax.spines[sp].set_linewidth(0.5)
    from matplotlib.transforms import offset_copy
    tax.text(1.0, 0.42, "10$^3$ km$^2$\nper 1°", ha="left", va="center", fontsize=st.MIN_PT, color=st.INK,
             transform=offset_copy(tax.transAxes, fig=fig, x=14, y=0, units="points"))

    vb, edges = _lat_profile(base, sf.LAT_TOP, sf.LAT_BOT, deg=1.0)
    va, _ = _lat_profile(add, sf.LAT_TOP, sf.LAT_BOT, deg=1.0)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    y = np.asarray(rb.lat_to_y(ctr), float)
    vb, va = vb / 1e3, va / 1e3
    rax.fill_betweenx(y, 0, vb, color=c_base, lw=0, zorder=2)
    rax.fill_betweenx(y, vb, vb + va, color=c_add, lw=0, zorder=2)
    rax.plot(vb + va, y, color=c_add, lw=0.5, zorder=3)
    (_, _), (py0, py1) = rb.band_limits(ROBIN_COVER)
    rax.set_ylim(py0, py1)
    rmax = float((vb + va).max())
    rax.set_xlim(0, rmax * 1.08)
    t = _nice_tick(rmax * 1.08)
    rax.set_xticks([0, t], ["0", f"{t:g}"])
    la = [-20, -10, 0, 10, 20]
    rax.set_yticks(rb.lat_to_y(np.array(la, float)), [f"{abs(v)}°{'S' if v < 0 else ('N' if v > 0 else '')}" for v in la])
    rax.yaxis.tick_right()
    rax.tick_params(axis="both", length=2, width=0.5, pad=1.5, labelsize=st.MIN_PT, colors=st.INK)
    for sp in ("top", "left"):
        rax.spines[sp].set_visible(False)
    for sp in ("bottom", "right"):
        rax.spines[sp].set_color(st.MUTED); rax.spines[sp].set_linewidth(0.5)
    rax.text(0.97, 0.008, "10$^3$ km$^2$\nper 1°", transform=rax.transAxes, ha="right",
             va="bottom", fontsize=KEY_PT, color=st.INK, linespacing=1.0)
    hs = [Patch(facecolor=c_base), Patch(facecolor=c_add)]
    tax.legend(hs, ["Impervious by 2000", "Added 2000\u20132020"], loc="upper left", ncol=2,
               bbox_to_anchor=(0.0, 0.98), frameon=False, fontsize=st.MIN_PT,
               handlelength=1.0, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)


def figure1(name="figure1_expansion_v11", profiles="sides"):
    st.apply_rc()
    u = sf.load_urban()
    add, base = u["added"], u["base"]
    breaks = [0.05, 0.15, 0.4, 1.0, 2.5, 6.0]
    grey_breaks = [0.05, 0.4, 2.0]
    grey_ramp = ["#B5B5B5", "#8C8C8C", "#5E5E5E", "#2F2F2F"]

    W = PAGE_W
    ML, MR = 0.10, 0.20
    live = W - ML - MR
    prof_w, gap_p = 0.78, 0.075
    ML, MR, prof_w, gap_p = _band_layout(ML, MR, prof_w, gap_p)
    top_h, top_gap = 0.62, 0.05
    if profiles == "top_right":
        prof_w, gap_p, lab_r = 0.62, 0.06, 0.34
        map_w = live - prof_w - gap_p - lab_r
    else:
        map_w = live - 2 * prof_w - 2 * gap_p
    if _is_robin():
        import tropics_robinson as rb
        map_h = map_w * rb.band_aspect(ROBIN_COVER)
    else:
        map_h = map_w * (tf.LAT1 - tf.LAT0) / (tf.LON1 - tf.LON0)
    head = 0.36 if profiles != "top_right" else 0.36 + top_h + top_gap
    gap1, label_h, gap2 = (0.20 if _is_robin() else 0.92), 0.16, 0.12
    icols, igap = 4, 0.17
    pw = (live - (icols - 1) * igap) / icols
    ph = pw
    cd_h, foot = 1.34, 0.98
    ins_l = 0.15
    gap2 = 0.27
    pw = (live - ins_l - (icols - 1) * igap) / icols
    ph = pw
    lab_w, pgap = 0.95, 0.30
    pw2 = (live - lab_w - pgap) / 2
    fig_h = head + map_h + gap1 + label_h + ph + gap2 + cd_h + foot
    fig = plt.figure(figsize=(W, fig_h))

    if profiles == "top_right":
        band = _band_axes(fig, W, fig_h, fig_h - head, ML=ML, MR=W - ML - live + lab_r,
                          left_w=0.0, right_w=prof_w, gap=gap_p, land="white")
    else:
        band = _band_axes(fig, W, fig_h, fig_h - head, ML=ML, MR=MR,
                          left_w=prof_w, right_w=prof_w, gap=gap_p, land="white")
    ax, y_map = band["ax"], band["y_map"]
    from matplotlib.colors import LogNorm
    g_cmap, g_norm = _ramp_cmap(grey_ramp, "grey"), LogNorm(0.05, 2.0)
    u_cmap, u_norm = _ramp_cmap(m2.RAMP_URBAN, "urban"), LogNorm(0.05, 6.0)
    b0, a0 = np.nan_to_num(base), np.nan_to_num(add)
    grey = _continuous_raster(np.where(b0 >= a0, base, np.nan), g_cmap, g_norm, 0.05)
    rgba = _continuous_raster(np.where(a0 > b0, add, np.nan), u_cmap, u_norm, 0.05)
    _draw_raster(ax, grey); _draw_raster(ax, rgba)
    if _is_robin():
        _tropic_lines(ax)
    else:
        tf._outline(ax, context=True)
    a_lab = fig.text(ML / W, 1 - 0.055 / fig_h, "a", fontsize=9.2, fontweight="bold",
                     va="top", ha="left", color=st.INK)

    vb, edges = _lat_profile(base, sf.LAT_TOP, sf.LAT_BOT)
    va, _ = _lat_profile(add, sf.LAT_TOP, sf.LAT_BOT)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    top = max(vb.max(), va.max()) / 1e3 * 1.06
    unit = f"0\u2013{top:.0f}\n10$^3$ km$^2$\nper degree"
    bb, bedges = _lat_profile(base, sf.LAT_TOP, sf.LAT_BOT, deg=PROFILE_BIN)
    ba, _ = _lat_profile(add, sf.LAT_TOP, sf.LAT_BOT, deg=PROFILE_BIN)
    bctr = 0.5 * (bedges[:-1] + bedges[1:])
    if _is_robin():
        bdiv, top = 1.0, max(bb.max(), ba.max()) / 1e3 * 1.06
        unit = "10$^3$ km$^2$ per\n2\u00b0 band"
    else:
        bdiv = PROFILE_BIN
    if profiles == "top_right":
        _stacked_profiles(fig, W, fig_h, band, base, add, grey_ramp[2], m2.RAMP_URBAN[4],
                          top_h, top_gap, ML)
        fig.canvas.draw()
        rnd_ = fig.canvas.get_renderer()
        for ax_ in fig.axes:
            lon_ = [t for t in ax_.get_xticklabels() if "\u00b0W" in t.get_text()]
            if lon_:
                bb_ = lon_[0].get_window_extent(rnd_)
                a_lab.set_y(0.5 * (bb_.y0 + bb_.y1) / fig.bbox.height)
                a_lab.set_va("center")
                break
    else:
        _profile(band["left"], ctr, vb / 1e3, grey_ramp[2], "By 2000", True, top, unit,
                 bars=(bctr, bb / 1e3 / bdiv, PROFILE_BIN))
        _profile(band["right"], ctr, va / 1e3, m2.RAMP_URBAN[4],
                 "Added", False, top, unit, label_ha="right",
                 bars=(bctr, ba / 1e3 / bdiv, PROFILE_BIN))

    def _rect(lo, hi, la, lb):
        return ((lo - tf.LON0) / (tf.LON1 - tf.LON0),
                (la - tf.LAT0) / (tf.LAT1 - tf.LAT0),
                (hi - lo) / (tf.LON1 - tf.LON0), (lb - la) / (tf.LAT1 - tf.LAT0))
    _fig5_map_key(ax, map_h, g_cmap, g_norm, ticks=(0.05, 0.2, 2),
                  title="Impervious, 2000", x0=0.318, w=0.12)
    _fig5_map_key(ax, map_h, u_cmap, u_norm, ticks=(0.05, 0.2, 1, 6),
                  title="Added 2000\u20132020", x0=0.612, w=0.15)

    FURNITURE = [(0.56, 0.935, 0.075), (0.56, 0.470, 0.075),
                 (0.030, 0.935, 0.075), (0.030, 0.940, 0.820)]

    y_ins = y_map - gap1 - label_h - ph
    for i, idx in enumerate(FOUR):
        nm, lo, hi, la, lb, _ = tf.REGIONS[idx]
        col = tf.FRAME_COLORS[i]
        x = ML + ins_l + i * (pw + igap)
        iax = fig.add_axes([x / W, y_ins / fig_h, pw / W, ph / fig_h])
        iax.set_xlim(lo, hi); iax.set_ylim(la, lb)
        iax.set_xticks([]); iax.set_yticks([]); iax.set_facecolor("white")
        for sp in iax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.1)
        tf._draw_land(iax, lw=0.3, color="white")
        m2._draw(iax, grey); m2._draw(iax, rgba); tf._outline(iax, lw=0.3)
        bar_x, arr_x, arr_y = FURNITURE[i]
        tf.north_arrow(iax, lo, hi, la, lb, x_frac=arr_x, y_frac=arr_y)
        tf.scale_bar(iax, lo, hi, la, lb, km=500, x_frac=bar_x)
        tf.corner_labels(iax, lo, hi, la, lb)
        iax.text(0.0, 1.020, f"{i + 1}  {nm}", transform=iax.transAxes,
                 fontsize=7.0, fontweight="bold", color=col, ha="left",
                 va="bottom")
        bx_deg, by_deg = lo + (hi - lo) * 0.5, lb + (lb - la) * 0.10
        if _is_robin():
            e = np.linspace(0, 1, 40)
            ring_lon = np.r_[lo + (hi - lo) * e, np.full(40, hi),
                             hi - (hi - lo) * e, np.full(40, lo)]
            ring_lat = np.r_[np.full(40, la), la + (lb - la) * e,
                             np.full(40, lb), lb - (lb - la) * e]
            rx, ry = _lonlat(ring_lon, ring_lat)
            ax.plot(rx, ry, color=col, lw=0.9, zorder=8)
            bx, by = (float(v) for v in _lonlat([bx_deg], [by_deg]))
        else:
            ax.add_patch(Rectangle((lo, la), hi - lo, lb - la, fill=False,
                                   edgecolor=col, lw=0.9, zorder=8))
            bx, by = bx_deg, by_deg
        ax.plot([bx], [by], marker="o", ms=7.0, mfc="white", mec=col, mew=0.9,
                ls="none", zorder=9, clip_on=False)
        ax.text(bx, by, str(i + 1), fontsize=6.0, fontweight="bold", color=col,
                ha="center", va="center", zorder=10, clip_on=False)
        if i == 0:
            iax.text(0.025, 0.965, "b", transform=iax.transAxes, fontsize=9.2,
                     fontweight="bold", color=st.INK, ha="left", va="top",
                     zorder=9,
                     bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))

    y_cd = foot
    d = tp.cities()
    realms = ["Neotropic", "Afrotropic", "Indomalayan", "Australasia"]
    rows = ["All tropics"] + realms
    y = np.arange(len(rows))[::-1]
    from matplotlib.colors import to_rgb

    def _stack(ax, mat, colors, show_rows):
        left = np.zeros(len(rows))
        for j, c in enumerate(colors):
            ax.barh(y, mat[:, j], left=left, height=0.62, color=c,
                    edgecolor="white", lw=0.5)
            tc = st.text_on(c)
            for yi, li, vi in zip(y, left, mat[:, j]):
                if vi > 7:
                    ax.text(li + vi / 2, yi, f"{vi:.0f}", ha="center", va="center",
                            fontsize=6.2, color=tc)
            left = left + mat[:, j]
        ax.set_yticks(y)
        ax.set_yticklabels(rows if show_rows else [], fontsize=8)
        ax.set_xlim(0, 100)
        ax.tick_params(axis="y", length=0)
        st.despine(ax, keep=("bottom",))

    hab = pd.read_csv(INTERIM_DIR / "city_habitat.csv")
    hab = hab[hab.city_id.isin(set(d.city_id))].merge(
        d[["city_id", "realm"]], on="city_id", how="left")
    hcols = [f"hab_{k}_km2_scaled" for k in tp.HAB_ORDER]
    hm = [hab[hcols].sum().values]
    for r in realms:
        hm.append(hab[hab.realm == r][hcols].sum().values)
    hm = np.array(hm, dtype=float)
    hm = hm / hm.sum(axis=1, keepdims=True) * 100
    rnd = fig.canvas.get_renderer()
    probe = [fig.text(0, 0, r, fontsize=8) for r in rows]
    lab_w = max(t.get_window_extent(rnd).width for t in probe) / fig.dpi + 0.07
    for t in probe:
        t.remove()
    pgap = 0.22
    pw2 = (live - 2 * lab_w - pgap) / 2
    x_c, x_d = ML, ML + lab_w + pw2 + pgap
    from matplotlib.transforms import offset_copy
    y_top = (y_cd + cd_h) / fig_h
    cx = fig.add_axes([(x_c + lab_w) / W, y_cd / fig_h, pw2 / W, cd_h / fig_h])
    _stack(cx, hm, [st.HABITAT_COLORS[k] for k in tp.HAB_ORDER], True)
    cx.set_xlabel("Share of converted area by prior land cover (%)", fontsize=7.0)
    fig.canvas.draw()
    _row = [t for t in cx.get_yticklabels() if t.get_text() == "All tropics"][0]
    _bb = _row.get_window_extent(fig.canvas.get_renderer())
    y_row = 0.5 * (_bb.y0 + _bb.y1) / fig.bbox.height
    fig.text(x_c / W, y_row, "c", fontsize=9.5, fontweight="bold", va="center", ha="left", color=st.INK)
    hs = [Rectangle((0, 0), 1, 1, facecolor=st.HABITAT_COLORS[k])
          for k in tp.HAB_ORDER]
    cx.legend(hs, [tp.HAB_LABEL[k] for k in tp.HAB_ORDER], fontsize=6.2, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.30), handlelength=1.0,
              handleheight=0.9, columnspacing=1.0, labelspacing=0.35,
              frameon=False)

    dx = fig.add_axes([(x_d + lab_w) / W, y_cd / fig_h, pw2 / W, cd_h / fig_h])
    tot = d[[f"form_{k}_km2" for k in FORMS]].sum()
    mat = [(tot / tot.sum() * 100).values]
    for r in realms:
        t = d[d.realm == r][[f"form_{k}_km2" for k in FORMS]].sum()
        mat.append((t / t.sum() * 100).values)
    _stack(dx, np.array(mat), [st.FORM_COLORS[k] for k in FORMS], True)
    dx.set_xlabel("Share of converted area by expansion form (%)", fontsize=7.0)
    fig.text(x_d / W, y_row, "d", fontsize=9.5, fontweight="bold", va="center", ha="left", color=st.INK)
    hs = [Rectangle((0, 0), 1, 1, facecolor=st.FORM_COLORS[k]) for k in FORMS]
    dx.legend(hs, [k.capitalize() for k in FORMS], fontsize=6.2, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.30), handlelength=1.0,
              handleheight=0.9, columnspacing=1.4, frameon=False)
    return tf.save(fig, name)


REALMS = ["Indomalayan", "Neotropic", "Afrotropic", "Australasia"]


def figure2(name="figure2_exposure_v11", realms=True):
    import tropics_citymaps as cm
    st.apply_rc()
    d = cm.cities()
    breaks = [8, 12, 17, 24, 35, 55]

    W = PAGE_W
    ML, MR = 0.12, 0.15
    live = W - ML - MR
    head, prof_w, gap_p = 0.17, 0.80, 0.075
    ML, MR, prof_w, gap_p = _band_layout(ML, MR, prof_w, gap_p)
    pgap, ph, y_pan = 0.62, 1.62, 0.62
    pw = (live - 2 * pgap) / 3
    map_w = live - prof_w - gap_p
    if _is_robin():
        import tropics_robinson as rb
        map_h = map_w * rb.band_aspect(ROBIN_COVER)
    else:
        map_h = map_w * (tf.LAT1 - tf.LAT0) / (tf.LON1 - tf.LON0)
    gap_mid = 0.46 if _is_robin() else 1.26
    fig_h = head + map_h + gap_mid + ph + y_pan
    fig = plt.figure(figsize=(W, fig_h))

    band = _band_axes(fig, W, fig_h, fig_h - head, ML=ML, MR=MR, left_w=prof_w,
                      gap=gap_p)
    ax = band["ax"]
    from matplotlib.colors import LogNorm
    e_cmap, e_norm = _ramp_cmap(cm.CITY_RAMP, "exposure"), LogNorm(5, 60)
    _points_continuous(ax, d, "cost_per_km2", e_cmap, e_norm)
    if _is_robin():
        _tropic_lines(ax)
    else:
        tf._outline(ax, context=True)
    fig.text(ML / W, 1 - 0.01 / fig_h, "a", fontsize=9.2, fontweight="bold",
             va="top", ha="left", color=st.INK)

    def _rect(lo, hi, la, lb):
        return ((lo - tf.LON0) / (tf.LON1 - tf.LON0),
                (la - tf.LAT0) / (tf.LAT1 - tf.LAT0),
                (hi - lo) / (tf.LON1 - tf.LON0), (lb - la) / (tf.LAT1 - tf.LAT0))
    _fig5_map_key(ax, map_h, e_cmap, e_norm, ticks=(5, 10, 20, 50),
                  title="Exposure intensity\n(km$^2$ km$^{-2}$)",
                  y0=0.10 + (0.15 if _is_robin() else 0.0))
    if _is_robin():
        tf.robinson_degree_labels(ax, lat_side=FIG2_LAT_SIDE, lat_inside=True)

    edges = np.arange(sf.LAT_TOP, sf.LAT_BOT - 0.001, -1.0)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    band_id = np.clip(((sf.LAT_TOP - d.lat.to_numpy()) // 1.0).astype(int), 0,
                      len(ctr) - 1)
    med = (pd.DataFrame(dict(b=band_id, v=d.cost_per_km2.to_numpy()))
             .groupby("b").v.median().reindex(range(len(ctr))).fillna(0.0)
             .to_numpy())
    bctr, bmed = _lat_median(d.lat.to_numpy(), d.cost_per_km2.to_numpy(),
                             PROFILE_BIN)
    if _is_robin():
        _profile(band["left"], ctr, med, cm.CITY_RAMP[3], "", True,
                 np.nanmax(bmed) * 1.06, "",
                 unit_ha="left", bars=(bctr, bmed, PROFILE_BIN))
        band["left"].text(0.0, 0.012, "Median, 2\u00b0 bands (km$^2$ km$^{-2}$)",
                          transform=band["left"].transAxes, fontsize=KEY_PT,
                          ha="left", va="bottom", color=st.INK)
    else:
        _profile(band["left"], ctr, med, cm.CITY_RAMP[3], "median", True,
                 med.max() * 1.06,
                 f"0\u2013{med.max():.0f} km$^2$ km$^{{-2}}$\nper degree",
                 unit_ha="left", bars=(bctr, bmed, PROFILE_BIN))

    xs = [ML + i * (pw + pgap) for i in range(3)]
    inset = [0.66, 0.40, 0.86]
    if not realms:
        pw = (live - pgap) / 2
        xs, inset = [0.0, ML, ML + pw + pgap], [0.0, 0.40, 0.86]
    bx, cx, dx = (fig.add_axes([(xs[i] + inset[i]) / W, y_pan / fig_h,
                                (pw - inset[i]) / W, ph / fig_h])
                  for i in range(3))

    v = [np.log10(d[d.realm == r].cost_per_km2.dropna().values) for r in REALMS]
    parts = bx.violinplot(v, vert=False, showextrema=False, widths=0.86)
    for pc in parts["bodies"]:
        pc.set_facecolor(st.SEQ_ORANGE(0.45)); pc.set_alpha(0.55)
        pc.set_edgecolor("none")
    for i, a in enumerate(v, start=1):
        q1, me, q3 = np.percentile(a, [25, 50, 75])
        bx.plot([q1, q3], [i, i], color=st.INK, lw=1.6, solid_capstyle="butt")
        bx.plot([me], [i], "o", ms=3.0, mfc="white", mec=st.INK, mew=1.0)
    bx.set_yticks(range(1, len(REALMS) + 1))
    bx.set_yticklabels([f"{r}\n$n$ = {len(x):,}" for r, x in zip(REALMS, v)],
                       fontsize=6.0, linespacing=1.4)
    tp._logticks(bx, "x", [3, 10, 30, 100])
    bx.set_xlabel("Exposure intensity (km$^2$ km$^{-2}$)", fontsize=7.0)
    bx.tick_params(axis="y", length=0)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x_left = min(t.get_window_extent(rend).x0 for t in bx.get_yticklabels()
                 if t.get_text()) / fig.bbox.width
    st.despine(bx, keep=("bottom",))
    if realms:
        tp.head(bx, "b", "Within realms as well as between")
    else:
        bx.remove()

    x = np.log10(d.urban_added_km2.replace(0, np.nan).values)
    y = np.log10(d.cost_per_km2.values)
    ok = np.isfinite(x) & np.isfinite(y)
    hb = tp.hexpanel(cx, x[ok], y[ok])
    mx, my = np.nanmedian(x[ok]), np.nanmedian(y[ok])
    cx.axvline(mx, color=st.MUTED, lw=0.6, ls=(0, (2, 2)))
    cx.axhline(my, color=st.MUTED, lw=0.6, ls=(0, (2, 2)))
    top_ = d[(np.log10(d.urban_added_km2.replace(0, np.nan)) > mx)
             & (np.log10(d.cost_per_km2) > my)]
    cx.set_ylim(0.0, np.log10(300))
    cx.text(0.97, 0.97, f"{len(top_):,} cities above\nboth medians hold\n"
            f"{top_.cost_km2.sum() / d.cost_km2.sum() * 100:.0f}% of the exposure",
            transform=cx.transAxes,
            fontsize=5.9, color=STAT, ha="right", va="top", linespacing=1.5,
            bbox=dict(fc="white", ec="none", pad=1.0), zorder=6)
    tp._logticks(cx, "x", [0.1, 10, 1000])
    tp._logticks(cx, "y", [1, 3, 10, 30, 100, 300])
    cx.set_xlabel("Urban land added (km$^2$)", fontsize=7.0)
    cx.set_ylabel("Exposure intensity (km$^2$ km$^{-2}$)", fontsize=7.0)
    tp.head(cx, "c" if realms else "b", "Large expansion against high intensity")
    from matplotlib.ticker import FuncFormatter, NullLocator
    kax = tp.colorbar(fig, hb, cx, label="", loc=(0.05, 0.115), w=0.24)
    kax.annotate("Cities", xy=(0.0, 1.0), xycoords="axes fraction",
                 xytext=(0, 2.0), textcoords="offset points", ha="left",
                 va="bottom", fontsize=6.2, color=st.MUTED)
    kax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    kax.xaxis.set_minor_locator(NullLocator())

    BIOMES = [("Tropical & Subtropical Moist Broadleaf Forests", "Moist forest"),
              ("Mangroves", "Mangroves"),
              ("Tropical & Subtropical Grasslands, Savannas & Shrublands", "Savanna"),
              ("Tropical & Subtropical Dry Broadleaf Forests", "Dry forest"),
              ("Deserts & Xeric Shrublands", "Desert and xeric"),
              ("Tropical & Subtropical Coniferous Forests", "Coniferous forest")]
    vb_ = [np.log10(d[d.biome == b].cost_per_km2.dropna().values) for b, _ in BIOMES]
    order = np.argsort([np.median(a) for a in vb_])
    vb_ = [vb_[k] for k in order]
    labs = [BIOMES[k][1] for k in order]
    bp = dx.boxplot(vb_, vert=False, whis=(5, 95), showfliers=False,
                    widths=0.56, patch_artist=True,
                    medianprops=dict(color=st.INK, lw=1.4),
                    whiskerprops=dict(color=st.INK, lw=0.8),
                    capprops=dict(color=st.INK, lw=0.8),
                    boxprops=dict(edgecolor=st.INK, lw=0.6))
    for b_ in bp["boxes"]:
        b_.set_facecolor(st.SEQ_ORANGE(0.45)); b_.set_alpha(0.75)
    dx.set_yticks(range(1, len(labs) + 1))
    dx.set_yticklabels([f"{l}\n$n$ = {len(a):,}" for l, a in zip(labs, vb_)],
                       fontsize=6.0, linespacing=1.4)
    dx.set_xlim(bx.get_xlim())
    tp._logticks(dx, "x", [3, 10, 30, 100])
    dx.set_xlabel("Exposure intensity (km$^2$ km$^{-2}$)", fontsize=7.0,
                  loc="right")
    dx.tick_params(axis="y", length=0)
    st.despine(dx, keep=("bottom",))
    tp.head(dx, "d" if realms else "c", "Within biomes")
    return tp.save(fig, name)


SHAP_LABEL = {"country": "Country", "realm": "Realm",
              "biome": "Biome", "setting": "Setting richness",
              "form": "Expansion form", "size": "City size, 2000"}
DRIVER_LABEL = {"log_setting": "Setting richness", "abs_lat": "|latitude|",
                "log_size": "City size, 2000", "form_infill_share": "Infill share",
                "form_leapfrog_share": "Leapfrog share", "growth_pct": "Growth rate"}


def figure3(name="figure3_scale_setting_v11"):
    st.apply_rc()
    import tropics_refit as trf
    d = tp.cities().merge(trf.footprint_setting(), on="city_id", how="left")
    d["setting_footprint"] = d.setting_footprint.fillna(0.0)
    sp = json.loads((TABLE_DIR / "scaling_spatial.json").read_text())["cost_per_km2"]
    dm = pd.read_csv(TABLE_DIR / "driver_model.csv")
    sh = _plan()["shapley"]

    fig = plt.figure(figsize=(PAGE_W, 4.65))
    L, R, T, B = 0.158, 0.952, 0.952, 0.118
    hgap, vgap = 0.150, 0.190
    cw = (R - L - hgap) / 2
    ch = (T - B - vgap) / 2
    def cell(i, j):
        return fig.add_axes([L + j * (cw + hgap), B + (1 - i) * (ch + vgap), cw, ch])

    x = np.log10(d.urban_km2_base.replace(0, np.nan).values)
    y = np.log10(d.cost_per_km2.values)
    sset = np.log10(d.setting_footprint.clip(lower=0.5).values)

    ax = cell(0, 0)
    hb = tp.hexpanel(ax, x, y)
    tp.binned_median(ax, x, y)
    ax.set_xlabel("City urban extent, 2000 (km$^2$)", fontsize=7.4)
    ax.set_ylabel("Species-range exposure per km$^2$", fontsize=7.4)
    tp._logticks(ax, "x", [1, 10, 100, 1000]); tp._logticks(ax, "y", [3, 10, 30, 100])
    tp.head(ax, "a", "Size carries almost none")
    tp.stat(ax, f"$\\beta$ = {sp['beta']:+.4f}\n"
            f"[{sp['ci_lower']:+.4f}, {sp['ci_upper']:+.4f}]\n$P$ = {sp['p_value']:.1g}",
            "upper right")
    tp.colorbar(fig, hb, ax, loc=(0.60, 0.155))

    ax = cell(0, 1)
    hb = tp.hexpanel(ax, sset, y)
    tp.binned_median(ax, sset, y)
    ax.set_xlabel("Threatened species within 50 km of the 2000 footprint",
                  fontsize=6.9)
    ax.set_ylabel("Species-range exposure per km$^2$", fontsize=7.4)
    tp._logticks(ax, "x", [1, 3, 10, 30, 100]); tp._logticks(ax, "y", [3, 10, 30, 100])
    tp.head(ax, "b", "Setting carries some")
    tp.stat(ax, f"$\\gamma$ = {sp['gamma_setting']:.3f} $\\pm$ {sp['gamma_se']:.3f}\n"
            f"$n$ = {sp['n']:,}", "upper left")
    tp.colorbar(fig, hb, ax, loc=(0.60, 0.155))

    ax = cell(1, 0)
    items = sorted(sh["shapley"].items(), key=lambda kv: kv[1])
    names = [SHAP_LABEL[k] for k, _ in items]
    vals = np.array([v for _, v in items]) * 100
    cols = [STAT if k == "size" else st.BLUE for k, _ in items]
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color=cols, height=0.64, edgecolor="white", lw=0.5)
    for yi, v in zip(yy, vals):
        ax.text(v + max(vals) * 0.02, yi, f"{v:.1f}", va="center", ha="left",
                fontsize=6.2, color=st.INK)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlim(0, max(vals) * 1.20)
    ax.set_xlabel("Share of the variance in log exposure (%)", fontsize=7.4)
    ax.tick_params(axis="y", length=0)
    st.despine(ax, keep=("bottom",))
    tp.head(ax, "c", "Where the variance sits")
    tp.note(ax, f"Shapley shares, exact over\n64 subsets; they sum to the\n"
            f"model $R^2$ of {sh['full_R2']:.2f}", "lower right", fs=5.9)

    ax = cell(1, 1)
    dm = dm.reindex(dm.beta_std.abs().sort_values(ascending=False).index)
    lab = [DRIVER_LABEL.get(p, p) for p in dm.predictor]
    tp.forest(ax, lab, dm.beta_std, dm.beta_std - 1.96 * dm.se,
              dm.beta_std + 1.96 * dm.se,
              highlight=[p == "log_size" for p in dm.predictor], fs=6.4)
    ax.set_xlabel("Change in log$_{10}$ exposure per SD of the predictor",
                  fontsize=7.0)
    tp.head(ax, "d", "Against the others")
    tp.note(ax, f"country random intercept\nand biome fixed effects\n$n$ = {sp['n']:,}",
            "lower right", fs=5.9)

    return tp.save(fig, name)


def _ridges(ax, groups, labels, colors, xlim, ref=None, scale_to=0.82, seed=7):
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(seed)
    xs = np.linspace(xlim[0], xlim[1], 500)
    kdes = [gaussian_kde(g) for g in groups]
    dens = [kd(xs) for kd in kdes]
    k = scale_to / max(dn.max() for dn in dens)
    n = len(groups)
    for i, (g, kd, dn, lab, c) in enumerate(zip(groups, kdes, dens, labels, colors)):
        b0 = n - 1 - i
        h = kd(g) * k
        gx = g + rng.uniform(-0.012, 0.012, g.size)
        ax.scatter(gx, b0 + rng.uniform(0.04, 0.94, g.size) * h, s=1.1, color=c,
                   alpha=0.35, linewidths=0, zorder=2, rasterized=True)
        ax.plot(xs, b0 + dn * k, color=c, lw=1.3, zorder=3)
        ax.plot(xlim, [b0, b0], color="#BDBAB5", lw=0.5, zorder=3)
        on = lambda x: b0 + float(kd(x)[0]) * k
        mu, sd = float(np.mean(g)), float(np.std(g))
        for x in (mu - sd, mu + sd):
            ax.plot([x, x], [b0, on(x)], color=st.INK, lw=0.7, ls=(0, (1, 1.5)),
                    zorder=4)
        q1, me, q3 = np.percentile(g, [25, 50, 75])
        pk = float(xs[np.argmax(dn)])
        for x in (q1, q3):
            ax.plot([x], [on(x)], "s", ms=3.0, mfc="white", mec=st.INK, mew=0.9,
                    zorder=5)
        ax.plot([me], [on(me)], "D", ms=3.2, mfc=st.INK, mec=st.INK, zorder=6)
        ax.plot([mu], [on(mu)], "o", ms=3.6, mfc="white", mec=STAT, mew=1.1,
                zorder=6)
        ax.plot([pk], [on(pk)], "^", ms=3.8, mfc=STAT, mec=STAT, zorder=6)
        ax.text(xlim[0] - 0.03 * (xlim[1] - xlim[0]), b0 + 0.28,
                f"{lab}\n$n$ = {len(g):,}", ha="right", va="center", fontsize=6.2,
                linespacing=1.35, color=st.INK)
    if ref is not None:
        ax.axvline(ref, color=st.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.15, n - 1 + scale_to + 0.1)
    ax.set_yticks([])
    tp._logticks(ax, "x", [2, 5, 10, 20, 50, 100])
    ax.set_xlabel("Exposure intensity (km$^2$ km$^{-2}$)", fontsize=7.0)
    st.despine(ax, keep=("bottom",))


def figure3_shift(name="figure3_distributions_v1"):
    st.apply_rc()
    import tropics_refit as trf
    d = tp.cities().merge(trf.footprint_setting(), on="city_id", how="left")
    d = d[d.urban_km2_base > 0].copy()
    d["setting_footprint"] = d.setting_footprint.fillna(0.0)
    y = np.log10(d.cost_per_km2.values)
    ref = float(np.median(y))
    xlim = (np.log10(1.5), np.log10(150))
    sh = _plan()["shapley"]

    fig = plt.figure(figsize=(PAGE_W, 5.6))
    L, R, T, B = 0.152, 0.975, 0.958, 0.090
    hgap, vgap = 0.160, 0.195
    cw = (R - L - hgap) / 2
    ch = (T - B - vgap) / 2

    def _t(ax, letter, text):
        ax.text(-0.40, 1.035, letter, transform=ax.transAxes, fontsize=st.MIN_PT + 1,
                fontweight="bold", ha="left", va="bottom", color=st.INK)
        ax.text(-0.40 + 0.075, 1.035, text, transform=ax.transAxes,
                fontsize=st.MIN_PT, fontweight="bold", ha="left", va="bottom",
                color=st.INK)

    def cell(i, j):
        return fig.add_axes([L + j * (cw + hgap), B + (1 - i) * (ch + vgap), cw, ch])

    slate = "#5F7391"
    sz_edges = [0, 1, 5, 25, 100, np.inf]
    sz_lab = ["Under 1 km$^2$", "1 to 5 km$^2$", "5 to 25 km$^2$",
              "25 to 100 km$^2$", "100 km$^2$ or more"]
    sz = pd.cut(d.urban_km2_base, sz_edges, labels=False, right=False)
    ax = cell(0, 0)
    _ridges(ax, [y[sz == k] for k in range(5)][::-1], sz_lab[::-1],
            [slate] * 5, xlim, ref)
    _t(ax, "a", "Urban extent in 2000")

    gr = d.urban_added_km2 / d.urban_km2_base
    gq, ge = pd.qcut(gr, 5, labels=False, retbins=True)
    pc = [f"{v * 100:.0f}" for v in ge]
    g_lab = [f"Under {pc[1]}%", f"{pc[1]} to {pc[2]}%", f"{pc[2]} to {pc[3]}%",
             f"{pc[3]} to {pc[4]}%", f"Over {pc[4]}%"]
    ax = cell(0, 1)
    _ridges(ax, [y[gq == k] for k in range(5)][::-1], g_lab[::-1],
            [slate] * 5, xlim, ref)
    _t(ax, "b", "Urban growth, 2000 to 2020")

    sq, se = pd.qcut(d.setting_footprint, 5, labels=False, retbins=True, duplicates="drop")
    lo = [int(np.ceil(v)) for v in se[:-1]]
    hi = [int(v) for v in se[1:]]
    lo = [lo[0]] + [h + 1 for h in hi[:-1]]
    s_lab = [f"{a} to {b} species" for a, b in zip(lo, hi)]
    greens = ["#BFA9D6", "#9C7DC0", "#7651A5", "#5E3D91", "#4E2A84"]
    ax = cell(1, 0)
    _ridges(ax, [y[sq == k] for k in range(5)][::-1], s_lab[::-1],
            greens[::-1], xlim, ref)
    _t(ax, "c", "Threatened species within 50 km")

    ax = cell(1, 1)
    items = sorted(sh["shapley"].items(), key=lambda kv: kv[1])
    names = [SHAP_LABEL[k] for k, _ in items]
    vals = np.array([v for _, v in items]) * 100
    cols = [STAT if k == "size" else (greens[3] if k == "setting" else slate)
            for k, _ in items]
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color=cols, height=0.64, edgecolor="white", lw=0.5)
    for yi, v in zip(yy, vals):
        ax.text(v + max(vals) * 0.02, yi, f"{v:.1f}", va="center", ha="left",
                fontsize=6.2, color=st.INK)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlim(0, max(vals) * 1.20)
    ax.set_xlabel("Share of the variance in log exposure intensity (%)", fontsize=7.0)
    ax.tick_params(axis="y", length=0)
    st.despine(ax, keep=("bottom",))
    _t(ax, "d", "Variance in exposure intensity explained")
    tp.note(ax, f"Shapley shares sum to the\nmodel $R^2$ of {sh['full_R2']:.2f}",
            "lower right", fs=5.9)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="s", ls="none", ms=3.0, mfc="white", mec=st.INK, mew=0.9),
          Line2D([], [], marker="D", ls="none", ms=3.2, mfc=st.INK, mec=st.INK),
          Line2D([], [], marker="o", ls="none", ms=3.6, mfc="white", mec=STAT, mew=1.1),
          Line2D([], [], marker="^", ls="none", ms=3.8, mfc=STAT, mec=STAT),
          Line2D([], [], color=st.INK, lw=0.8, ls=(0, (1, 1.5))),
          Line2D([], [], marker="o", ls="none", ms=2.4, mfc=slate, mec="none", alpha=0.6),
          Line2D([], [], color=st.MUTED, lw=0.8, ls=(0, (3, 2)))]
    fig.legend(hs, ["25th and 75th percentiles", "Median", "Geometric mean", "Peak",
                    "Mean \u00b1 1 SD", "One city", "Median of all cities"],
               loc="center", ncol=4, fontsize=6.2, frameon=False,
               bbox_to_anchor=(0.5, B + ch + vgap * 0.46),
               handlelength=1.6, columnspacing=1.4, labelspacing=0.5)
    return tp.save(fig, name)


REALM_COLOR = dict(st.REALM_COLOR)
RESP_COLOR = {"all species of concern": st.SET_COLOR["all"][0],
              "threatened": st.SET_COLOR["threatened"][0],
              "narrow range": st.SET_COLOR["narrow"][0],
              "rarity weighted": st.SET_COLOR["rarity"][0]}
RESP_LIGHT = {"all species of concern": st.SET_COLOR["all"][1],
              "threatened": st.SET_COLOR["threatened"][1],
              "narrow range": st.SET_COLOR["narrow"][1],
              "rarity weighted": st.SET_COLOR["rarity"][1]}
FIG2_LAT_SIDE = "right"
FIG5_LAT_SIDE = "right"


def figure4(name="figure4_concentration_v13"):
    st.apply_rc()
    d = tp.cities()
    n = _nums(); pl = _plan()

    fig = plt.figure(figsize=(PAGE_W, 4.45))
    L, R, T, B = 0.104, 0.980, 0.948, 0.112
    hgap, vgap = 0.150, 0.180
    cw = (R - L - hgap) / 2
    ch = (T - B - vgap) / 2
    def cell(i, j):
        return fig.add_axes([L + j * (cw + hgap), B + (1 - i) * (ch + vgap), cw, ch])

    ax = cell(0, 0)
    order = np.argsort(-d.urban_km2_base.values)
    cost = d.cost_km2.values[order]; add = d.urban_added_km2.values[order]
    frac = np.arange(1, cost.size + 1) / cost.size * 100
    ax.plot(frac, np.cumsum(cost) / cost.sum() * 100, color=NULL, lw=1.6,
            label="species-range exposure")
    ax.plot(frac, np.cumsum(add) / add.sum() * 100, color=st.BLUE, lw=1.6,
            ls=(0, (4, 2)), label="urban land added")
    ax.plot([0, 100], [0, 100], color=st.FAINT, lw=0.8, zorder=0)
    k1 = max(1, int(round(cost.size * 0.01)))
    ax.text(0.30, 0.60,
            f"top 1% of cities: {cost[:k1].sum() / cost.sum() * 100:.1f}% of exposure\n"
            f"on {add[:k1].sum() / add.sum() * 100:.1f}% of the land\n"
            f"ratio {cost[:k1].sum() / cost.sum() / (add[:k1].sum() / add.sum()):.2f}",
            transform=ax.transAxes, fontsize=5.9, color=STAT, ha="left", va="top",
            linespacing=1.5)
    tp.label_line(ax, [62, 84], [62, 84], "equal shares", fs=5.8, dy=1.0)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Cities, ranked by 2000 urban extent (%)", fontsize=7.2)
    ax.set_ylabel("Cumulative share (%)", fontsize=7.2)
    ax.legend(fontsize=6.2, loc="lower right", handlelength=1.6, frameon=False,
              bbox_to_anchor=(1.02, -0.02))
    st.despine(ax)
    tp.head(ax, "a", "What actually happened")

    ax = cell(0, 1)
    rel = _release()
    rb = _robust()
    al = rel["alphas"]; a0 = rel["alpha_hat"]
    ax.fill_between(al, rel["m_lo"], rel["m_hi"], color=NULL, alpha=0.13,
                    lw=0, zorder=1)
    ax.plot(al, rel["measured"], color=NULL, lw=1.7, zorder=3,
            label="measured exposure intensity")
    ax.fill_between(al, rel["p_lo"], rel["p_hi"], color=st.BLUE, alpha=0.11,
                    lw=0, zorder=1)
    ax.plot(al, rel["predicted"], color=st.BLUE, lw=1.7,
            ls=(0, (4, 2)), zorder=3,
            label="model-fitted exposure intensity")
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, color=STAT, lw=0.8, ls=(0, (1, 2)))
    _obs_note = (a0 + 0.10, None)
    ylo = min(float(np.min(rel["m_lo"])), float(np.min(rel["p_lo"])))
    yhi = max(float(np.max(rel["m_hi"])), float(np.max(rel["p_hi"])))
    pad = 0.08 * (yhi - ylo)
    ax.set_xlim(al.min(), al.max()); ax.set_ylim(ylo - pad, yhi + pad)
    ax.text(_obs_note[0], 0.17, f"fitted reference\n$\\alpha$ = {a0:.2f}",
            transform=ax.get_xaxis_transform(), fontsize=5.9, color=STAT,
            ha="left", va="center", linespacing=1.5)
    ax.set_xlabel("Concentration exponent $\\alpha$", fontsize=7.2)
    ax.set_ylabel(f"Change relative to the fitted\nreference, "
                  f"$\\alpha$ = {a0:.2f} (%)", fontsize=7.2, linespacing=1.5)
    ax.legend(fontsize=5.8, loc="upper left", handlelength=1.6, frameon=True,
              framealpha=0.94, edgecolor="none", facecolor="white",
              borderpad=0.25, bbox_to_anchor=(0.02, 1.05)).set_zorder(6)
    ax.text(0.030, 0.795,
            f"at $\\alpha$ = {al[-1]:+.0f}, predicted minus measured\n"
            f"{rel['d'][-1]:+.1f} percentage points\n"
            f"[{rel['d_lo'][-1]:+.1f}, {rel['d_hi'][-1]:+.1f}]",
            transform=ax.transAxes, fontsize=5.6, color=STAT,
            ha="left", va="top", linespacing=1.5)
    st.despine(ax)
    tp.head(ax, "b", "The same land, reallocated")
    ax.text(0.0, -0.345, "dispersed", transform=ax.transAxes, fontsize=st.MIN_PT,
            color=st.MUTED, ha="left", va="top")
    ax.text(1.0, -0.345, "concentrated", transform=ax.transAxes,
            fontsize=st.MIN_PT, color=st.MUTED, ha="right", va="top")

    ax = cell(1, 0)
    for realm in REALMS:
        r = rb["realm"].get(realm)
        if not r:
            continue
        ax.fill_between(al, r["ci_lo"], r["ci_hi"], color=REALM_COLOR[realm],
                        alpha=0.11, lw=0, zorder=1)
        ax.plot(al, r["pct"], lw=1.5, color=REALM_COLOR[realm], zorder=3,
                label=f"{realm} (n = {r['n']:,})")
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, color=st.FAINT, lw=0.8, ls=(0, (1, 2)))
    _lo = min(np.min(r["ci_lo"]) for r in rb["realm"].values())
    _hi = max(np.max(r["ci_hi"]) for r in rb["realm"].values())
    ax.set_ylim(_lo - 0.06 * (_hi - _lo), _hi + 0.26 * (_hi - _lo))
    ax.set_xlabel("Concentration exponent $\\alpha$", fontsize=7.2)
    ax.set_ylabel("Change in the realm's exposure (%)", fontsize=7.2)
    ax.legend(fontsize=5.9, loc="upper center", handlelength=1.8, frameon=False,
              ncol=2, columnspacing=1.0)
    st.despine(ax)
    tp.head(ax, "c", "Realms pull in opposite directions")

    ax = cell(1, 1)
    for key, r in rb["response"].items():
        wide = key == "all species of concern"
        ax.fill_between(al, r["ci_lo"], r["ci_hi"], color=RESP_COLOR[key],
                        alpha=0.10, lw=0, zorder=1)
        ax.plot(al, r["pct"], lw=2.6 if wide else 1.5, color=RESP_COLOR[key],
                label=key, zorder=2 if wide else 3)
    ax.axhline(0, color=st.FAINT, lw=0.8, zorder=0)
    ax.axvline(a0, color=st.FAINT, lw=0.8, ls=(0, (1, 2)))
    _lo = min(np.min(r["ci_lo"]) for r in rb["response"].values())
    _hi = max(np.max(r["ci_hi"]) for r in rb["response"].values())
    ax.set_ylim(_lo - 0.06 * (_hi - _lo), _hi + 0.22 * (_hi - _lo))
    ax.set_xlabel("Concentration exponent $\\alpha$", fontsize=7.2)
    ax.set_ylabel("Change in the response (%)", fontsize=7.2)
    ax.legend(fontsize=5.9, loc="upper center", handlelength=1.8, frameon=False,
              ncol=2, columnspacing=1.0)
    st.despine(ax)
    tp.head(ax, "d", "Rarity changes the answer")

    return tp.save(fig, name)


RARITY_ROW_COLOR = {"Threatened, unweighted": st.SET_COLOR["threatened"][0],
                    "Narrow range, unweighted": st.SET_COLOR["narrow"][0],
                    "Species of concern, rarity weighted": st.SET_COLOR["all"][0],
                    "Threatened, rarity weighted": st.SET_COLOR["threatened"][0],
                    "Narrow range, rarity weighted": st.SET_COLOR["narrow"][0]}


def _exemplars(w, geoms, k=3, min_km=3000.0):
    import shapely
    pts = {}
    for sid, g in geoms.items():
        if g:
            c = shapely.union_all(g).centroid
            pts[sid] = (c.x, c.y)
    chosen = []
    for _, row in w.iterrows():
        if row.sp_id not in pts:
            continue
        x, y = pts[row.sp_id]
        far = True
        for c in chosen:
            cx, cy = pts[c.sp_id]
            dx = np.radians(x - cx)
            d = 6371.0 * np.arccos(np.clip(
                np.sin(np.radians(y)) * np.sin(np.radians(cy)) +
                np.cos(np.radians(y)) * np.cos(np.radians(cy)) * np.cos(dx), -1, 1))
            far &= d >= min_km
        if far:
            chosen.append(row)
        if len(chosen) == k:
            break
    return chosen


REGION_EDGE = "#1A1A1A"


def _region_outline(ax, box, letter, n=80):
    name, lon0, lon1, lat0, lat1 = box[:5]
    lon = np.linspace(lon0, lon1, n)
    lat = np.linspace(lat0, lat1, n)
    edges = [(lon, np.full(n, lat1)), (np.full(n, lon1), lat[::-1]),
             (lon[::-1], np.full(n, lat0)), (np.full(n, lon0), lat)]
    xs, ys = [], []
    for a, b in edges:
        x, y = _lonlat(np.asarray(a, float), np.asarray(b, float))
        xs.append(x); ys.append(y)
    x = np.concatenate(xs + [xs[0][:1]]); y = np.concatenate(ys + [ys[0][:1]])
    ax.plot(x, y, color=REGION_EDGE, lw=0.85, zorder=8,
            solid_joinstyle="round")
    xt, yt = _lonlat(np.array([lon0]), np.array([lat1]))
    ax.text(float(xt[0]), float(yt[0]), f" {letter}", fontsize=st.MIN_PT,
            fontweight="bold", color=REGION_EDGE, ha="left", va="bottom",
            zorder=9)


FIG5_REGIONS = [
    ("Colombian Andes", -76.5, -71.5, 4.0, 8.3, 100),
    ("Gulf of Guinea", -0.503, 4.503, 4.5975, 8.9025, 100),
    ("West Java", 104.7469, 109.7531, -8.9277, -4.6223, 100),
]


FIG5_SET_COLORS = {"Species of concern": st.SET_COLOR["all"][::-1],
                   "Threatened": st.SET_COLOR["threatened"][::-1],
                   "Narrow range": st.SET_COLOR["narrow"][::-1]}


def _fig5_region_keys(ae, af, pw, ph, live):
    import region_panels as rp
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.ticker import NullLocator
    thick = 0.085 / ph
    sw = ae.inset_axes([0.06, 0.075 - thick / 2, 0.24 / pw, thick])
    sw.add_patch(Rectangle((0, 0), 1, 1, facecolor=rp.URBAN, edgecolor="none"))
    sw.set_xlim(0, 1); sw.set_ylim(0, 1); sw.axis("off")
    ae.text(0.06 + 0.30 / pw, 0.075, "Urban land added", transform=ae.transAxes,
            ha="left", va="center", fontsize=st.MIN_PT, color=st.INK, zorder=10)
    x0, yb_ = 0.47, 0.105
    cax = af.inset_axes([x0, yb_, live * 0.62 * 0.62 / 3 / pw, thick])
    cmap, norm = rp._rich_cmap_norm()
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal",
                      ticks=rp.RICH_TICKS, extend="neither")
    cax.xaxis.set_minor_locator(NullLocator())
    cb.set_ticklabels([f"{t:g}" for t in rp.RICH_TICKS])
    cax.tick_params(labelsize=st.MIN_PT, length=2, width=0.5, colors=st.INK, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cax.annotate("Narrow-range species",
                 xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 2.0),
                 textcoords="offset points", ha="left", va="bottom",
                 fontsize=st.MIN_PT, color=st.INK, linespacing=1.1)


def _fig5_panels_bc(fig, W, fig_h, yb, bh):
    import tropics_speciesmaps as sm
    rv = {r["label"]: r for r in _plan()["rarity_variants"]}
    s0 = json.loads((TABLE_DIR / "scaling_spatial.json").read_text())["cost_per_km2"]
    sets = [("Species of concern", dict(beta=s0["beta"], ci=[s0["ci_lower"], s0["ci_upper"]], n=s0["n"]),
             rv["Species of concern, rarity weighted"]),
            ("Threatened", rv["Threatened, unweighted"], rv["Threatened, rarity weighted"]),
            ("Narrow range", rv["Narrow range, unweighted"], rv["Narrow range, rarity weighted"])]
    bx = fig.add_axes([1.30 / W, yb / fig_h, 2.10 / W, bh / fig_h])
    bx.axvline(0, color=st.FAINT, lw=0.7, zorder=0)
    xs = np.linspace(-0.32, 0.17, 1200)
    ticks, labels, y = [], [], 0.0
    for g, (lab, u, r) in enumerate(sets[::-1]):
        for e, kind, c in ((r, "Rarity weighted", FIG5_SET_COLORS[lab][1]),
                           (u, "Unweighted", FIG5_SET_COLORS[lab][0])):
            se = (e["ci"][1] - e["ci"][0]) / (2 * 1.959964)
            dens = np.exp(-0.5 * ((xs - e["beta"]) / se) ** 2)
            bx.fill_between(xs, y, y + 0.78 * dens, color=c, alpha=0.45, lw=0, zorder=2)
            bx.plot(xs, y + 0.78 * dens, color=c, lw=0.7, zorder=3)
            bx.plot(e["ci"], [y, y], color=FIG5_SET_COLORS[lab][1], lw=1.4, solid_capstyle="butt", zorder=4)
            bx.scatter([e["beta"]], [y], s=26, color=c, edgecolor=FIG5_SET_COLORS[lab][1], lw=0.9, zorder=5)
            ticks.append(y); labels.append(kind)
            y += 1.0
        bx.text(-0.335, y - 0.12, f"{lab}, $n$ = {u['n']:,}", ha="left", va="bottom",
                fontweight="bold", color=st.darker(FIG5_SET_COLORS[lab][1], 0.18))
        y += 0.65
    bx.set_yticks(ticks)
    bx.set_yticklabels(labels)
    bx.set_ylim(-0.35, y - 0.35)
    bx.set_xlim(-0.335, 0.165)
    bx.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1])
    bx.set_xlabel("Size exponent $\\beta$")
    bx.tick_params(axis="y", length=0)
    st.despine(bx, keep=("bottom",))
    tp.head(bx, "b", "Rarity weighting and the size exponent")

    t = pd.read_csv(tp.TAB_DIR / "species_concentration.csv")
    t = t[(t.rule == "centroid") & t.is_threat & (t.converted_km2 > 0)].rename(columns={"sci_name": "sci"})
    t["pct"] = 100 * t.converted_km2 / t.range_km2
    cx = fig.add_axes([4.12 / W, yb / fig_h, 2.78 / W, bh / fig_h])
    for k in ("VU", "EN", "CR"):
        q = t[t.category == k]
        cx.scatter(q.range_km2, q.pct, s=5, color=sm.CAT_COLORS[k], alpha=0.55, lw=0, zorder=2,
                   rasterized=True)
    edges = np.logspace(np.log10(t.range_km2.min()), np.log10(t.range_km2.max()), 19)
    b_ = np.digitize(t.range_km2, edges[1:-1])
    med = t.groupby(b_).pct.median()
    cnt = t.groupby(b_).size()
    keep = cnt[cnt >= 30].index
    ctr = np.sqrt(edges[:-1] * edges[1:])
    cx.plot(ctr[keep], med.loc[keep], color=st.INK, lw=1.1, zorder=4)
    r_ = t.loc[t.sci == "Rattus ranjiniae"].iloc[0]
    cx.annotate(r_.sci, xy=(r_.range_km2, r_.pct), xytext=(1.5e3, r_.pct), textcoords="data",
                style="italic", color=st.INK, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=st.MUTED, lw=0.5, shrinkA=1, shrinkB=3))
    cx.set_xscale("log"); cx.set_yscale("log")
    cx.set_xlim(1, 1e9); cx.set_ylim(1e-7, 100)
    cx.set_xticks([1e1, 1e3, 1e5, 1e7, 1e9])
    cx.set_yticks([1e-6, 1e-4, 1e-2, 1, 100])
    cx.set_xlabel("Global mapped range (km$^2$)")
    cx.set_ylabel("Share of mapped range converted (%)")
    st.despine(cx, keep=("bottom", "left"))
    hs = [plt.Line2D([0], [0], marker="o", ls="none", ms=4.0, color=sm.CAT_COLORS[k]) for k in ("CR", "EN", "VU")]
    hs.append(plt.Line2D([0], [0], color=st.INK, lw=1.1))
    cx.legend(hs, ["CR", "EN", "VU", "Median"], loc="lower left", ncol=1, frameon=False, handlelength=0.9,
              handletextpad=0.4, labelspacing=0.3, borderpad=0, borderaxespad=0,
              bbox_to_anchor=(10, 1e-6), bbox_transform=cx.transData)
    tp.head(cx, "c", "Share of range converted, all threatened species")
    return sm.worst(n=15, cats=("CR", "EN", "VU"))


KEY_PT = 7.0


def _fig5_map_key(ax, map_h, cmap, norm, ticks=(1, 10, 100, 1000),
                  title="Exposure intensity\n(10$^{-5}$ km$^{-2}$)", x0=0.625, w=0.15,
                  y0=0.10):
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.ticker import NullLocator
    cax = ax.inset_axes([x0, y0, w, 0.075 / map_h])
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal",
                      ticks=list(ticks), extend="neither")
    cax.xaxis.set_minor_locator(NullLocator())
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    cax.tick_params(labelsize=KEY_PT, length=1.5, width=0.4, colors=st.INK, pad=1.0)
    cb.outline.set_linewidth(0.4)
    cax.annotate(title,
                 xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 2.0),
                 textcoords="offset points", ha="left", va="bottom",
                 fontsize=KEY_PT, color=st.INK, linespacing=1.1)
    return cb


def figure5(name="figure5_rarity_v15", bottom="species", regions=None, middle=True):
    import tropics_citymaps as cm
    import tropics_speciesmaps as sm
    st.apply_rc()
    d = cm.cities()
    v = d.rarity_per_km2.values * 1e5
    d = d.assign(rarity_1e5=v)
    breaks = [float(f"{x:.2g}") for x in np.percentile(v[v > 0], [40, 60, 75, 87, 95, 99])]

    W = PAGE_W
    ML, MR = 0.09, 0.26
    live = W - ML - MR
    head, prof_w, gap_p = 0.17, 0.80, 0.075
    ML, MR, prof_w, gap_p = _band_layout(ML, MR, prof_w, gap_p)
    map_w = live - prof_w - gap_p
    if _is_robin():
        import tropics_robinson as rb
        map_h = map_w * rb.band_aspect(ROBIN_COVER)
    else:
        map_h = map_w * (tf.LAT1 - tf.LAT0) / (tf.LON1 - tf.LON0)
    bh = 2.05 if bottom == "regions" else 1.32
    if bottom == "regions":
        import region_panels as rp
        reg = sorted(regions or FIG5_REGIONS,
                     key=lambda b: 0.5 * (b[1] + b[2]))
        ML3, MR3, gap3 = 0.30, 0.10, 0.16
        pw2 = (W - ML3 - MR3 - 2 * gap3) / 3
        ph2 = pw2 * 0.86
        key3, y2 = 0.0, 0.27
        yb = y2 + ph2 + 0.68
    else:
        yb = 2.92
        ph2, y2, pw2 = 1.48, 0.44, 1.96
    gap_mid = 0.42 if _is_robin() else 1.18
    if not middle:
        yb, bh = y2 + ph2 + 0.26, 0.0
    fig_h = yb + bh + gap_mid + map_h + head
    fig = plt.figure(figsize=(W, fig_h))

    band = _band_axes(fig, W, fig_h, fig_h - head, ML=ML, MR=MR,
                      right_w=prof_w, gap=gap_p)
    ax = band["ax"]
    from matplotlib.colors import LogNorm
    r_cmap, r_norm = _ramp_cmap(cm.CITY_RAMP, "rarity"), LogNorm(1, 1000)
    _points_continuous(ax, d, "rarity_1e5", r_cmap, r_norm)
    if _is_robin():
        _tropic_lines(ax)
    else:
        tf._outline(ax, context=True)
    fig.text(ML / W, 1 - 0.01 / fig_h, "a", fontsize=9.2, fontweight="bold",
             va="top", ha="left", color=st.INK)

    def _rect(lo, hi, la, lb):
        return ((lo - tf.LON0) / (tf.LON1 - tf.LON0),
                (la - tf.LAT0) / (tf.LAT1 - tf.LAT0),
                (hi - lo) / (tf.LON1 - tf.LON0), (lb - la) / (tf.LAT1 - tf.LAT0))
    _fig5_map_key(ax, map_h, r_cmap, r_norm, y0=0.10 + (0.15 if _is_robin() else 0.0))
    if _is_robin():
        tf.robinson_degree_labels(ax, lat_side=FIG5_LAT_SIDE, lat_inside=True)

    edges = np.arange(sf.LAT_TOP, sf.LAT_BOT - 0.001, -1.0)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    bid = np.clip(((sf.LAT_TOP - d.lat.to_numpy()) // 1.0).astype(int), 0,
                  len(ctr) - 1)
    med = (pd.DataFrame(dict(b=bid, v=v)).groupby("b").v.median()
             .reindex(range(len(ctr))).fillna(0.0).to_numpy())
    bctr, bmed = _lat_median(d.lat.to_numpy(), v, PROFILE_BIN)
    if _is_robin():
        _profile(band["right"], ctr, med, cm.CITY_RAMP[3], "", False,
                 np.nanmax(bmed) * 1.06, "",
                 unit_ha="right", bars=(bctr, bmed, PROFILE_BIN))
        band["right"].text(1.0, 0.012, "Median, 2\u00b0 bands",
                           transform=band["right"].transAxes, fontsize=KEY_PT,
                           ha="right", va="bottom", color=st.INK)
    else:
        _profile(band["right"], ctr, med, cm.CITY_RAMP[3], "latitudinal\nmedian",
                 False,
                 med.max() * 1.06,
                 f"0\u2013{med.max():.0f}\n10$^{{-5}}$ km$^{{-2}}$\nper degree",
                 unit_ha="right", bars=(bctr, bmed, PROFILE_BIN))

    w = _fig5_panels_bc(fig, W, fig_h, yb, bh) if middle else None
    tags = "def" if middle else "bcd"

    if bottom == "regions":
        rich, added = rp.load_layers()
        rbreaks = rp._rich_breaks(rich)
        boxes = [rp._fit_box(b, ph2 / pw2) for b in reg]
        axs = []
        for i, box in enumerate(boxes):
            x = ML3 + i * (pw2 + gap3)
            a = fig.add_axes([x / W, y2 / fig_h, pw2 / W, ph2 / fig_h])
            rp.region_panel(a, box, rich, added, rbreaks,
                            scale_side="left" if i == 2 else "right",
                            corners=True)
            a.text(0.0, 1.035, f"{tags[i]}  {box[0]}", transform=a.transAxes,
                   fontsize=st.MIN_PT, fontweight="bold", ha="left",
                   va="bottom", color=st.INK)
            _region_outline(ax, box, tags[i])
            axs.append(a)
        _fig5_region_keys(axs[1], axs[2], pw2, ph2, W - ML3 - MR3)
        return tp.save(fig, name)

    geoms, _ = sm._geoms_for(w.sp_id.tolist())
    z = np.load(sm.PAIRS, allow_pickle=True)
    pair_sp, pair_city = z["pair_sp"], z["pair_city"]
    cells = sm._city_cells_index()
    ex = _exemplars(w, geoms)
    x_start = 0.46
    avail = W - MR - x_start
    gap2 = (avail - 3 * pw2) / 2
    for i, row in enumerate(ex):
        x = x_start + i * (pw2 + gap2)
        a = fig.add_axes([x / W, y2 / fig_h, pw2 / W, ph2 / fig_h])
        sm.species_panel(a, row, geoms.get(row.sp_id, []), pair_sp, pair_city,
                         cells, aspect=pw2 / ph2)
        a.text(-0.11, 1.22, "def"[i], transform=a.transAxes, fontsize=9.0,
               fontweight="bold", ha="left", va="bottom", color=st.INK)
    return tp.save(fig, name)


PRE_2000 = "#F2E3A0"


def figure6(name="figure6_matched_cities_v10", max_px=900, pairs=None,
            title=None):
    import tropics_speciesmaps as sm
    from tropics_panels import cities as _cities, city_names
    from tropics_frame import north_arrow as _na
    from tropics_refit import footprint_setting
    st.apply_rc()
    named = city_names(_cities()).set_index("city_id")
    setting = footprint_setting()
    pairs = _plan()["matched"] if pairs is None else pairs
    rr, cc, ids, conv = sm._city_cells_index()

    ncol, nrow = len(pairs), 2
    W = PAGE_W
    L_in, R_in = 0.34, 0.06
    col_gap, row_gap = 0.19, 0.25
    label_in, head_in, foot_in = 0.50, 0.10, 0.46
    pw_in = (W - L_in - R_in - (ncol - 1) * col_gap) / ncol
    ph_in = pw_in * 0.72
    fig_h = head_in + nrow * (ph_in + label_in) + (nrow - 1) * row_gap + foot_in
    fig = plt.figure(figsize=(W, fig_h))

    letters = "abcdef"
    BAR_KM_COL = {1: 5.0, 2: 5.0}
    ROW_LAB = ["lower exposure", "higher exposure"]

    for c, pair in enumerate(pairs):
        for r, side in enumerate(("low", "high")):
            info = pair[side]
            cid = int(info["city_id"])
            lon, lat, _ = sm._cells_of(np.array([cid]), rr, cc, ids, conv)
            sp_km = max((lon.max() - lon.min()) * 111.32
                        * np.cos(np.radians(np.mean(lat))),
                        (lat.max() - lat.min()) * 110.57)
            win = max(30.0, sp_km * 1.45)
            x = L_in + c * (pw_in + col_gap)
            y = fig_h - head_in - (r + 1) * (ph_in + label_in) - r * row_gap
            ax = fig.add_axes([x / W, y / fig_h, pw_in / W, ph_in / fig_h])
            lon_c = 0.5 * (lon.min() + lon.max())
            lat_c = 0.5 * (lat.min() + lat.max())
            dlat = win / 110.57
            dlon = win / (111.32 * np.cos(np.radians(lat_c))) / (ph_in / pw_in)
            x0, x1 = lon_c - dlon / 2, lon_c + dlon / 2
            y0, y1 = lat_c - dlat / 2, lat_c + dlat / 2
            a = sm.read_epochs(x0, x1, y0, y1, max_px=max_px)
            ax.imshow(sm._epoch_rgba(a, PRE_2000), extent=(x0, x1, y0, y1),
                      origin="upper", interpolation="nearest", zorder=2)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.set_aspect("auto")
            ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_color("#B9B6B1"); sp_.set_linewidth(0.5)
            nm = str(named.name.get(cid, "") or "").strip()
            if not nm:
                ns = "N" if info["lat"] >= 0 else "S"
                ew = "E" if info["lon"] >= 0 else "W"
                nm = (f"unnamed agglomeration {abs(info['lat']):.1f}\u00b0{ns} "
                      f"{abs(info['lon']):.1f}\u00b0{ew}")
            k = r * ncol + c
            ax.text(-0.020, 1.230, letters[k], transform=ax.transAxes,
                    fontsize=8.8, fontweight="bold", ha="left", va="baseline",
                    color=st.INK)
            ax.text(0.058, 1.230, f"{nm}, {info['country']}",
                    transform=ax.transAxes, fontsize=7.2, fontweight="bold",
                    ha="left", va="baseline", color=st.INK)
            ax.text(0.0, 1.125,
                    f"2000 extent {info['urban_km2_base']:.1f} km$^2$   "
                    f"added {info['urban_added_km2']:.0f} km$^2$",
                    transform=ax.transAxes, fontsize=5.9, ha="left",
                    va="baseline", color=st.MUTED)
            ax.text(0.0, 1.030, f"setting {int(setting.get(cid, 0))} spp.",
                    transform=ax.transAxes, fontsize=5.9, ha="left",
                    va="baseline", color=st.MUTED)
            ax.text(1.0, 1.030, f"{info['cost_per_km2']:.0f} km$^2$ per km$^2$",
                    transform=ax.transAxes, fontsize=6.6, ha="right",
                    va="baseline", color=STAT, fontweight="normal")
            tf.corner_labels(ax, x0, x1, y0, y1, nd=2)
            _na(ax, x0, x1, y0, y1)
            kmdeg = 111.32 * np.cos(np.radians(lat_c))
            target = (x1 - x0) * kmdeg * 0.26
            km = min([5, 10, 25, 50, 100, 250], key=lambda v: abs(v - target))
            km = BAR_KM_COL.get(c, km)
            dxk = km / kmdeg
            ax.plot([x0 + (x1 - x0) * 0.06, x0 + (x1 - x0) * 0.06 + dxk],
                    [y0 + (y1 - y0) * 0.08] * 2, color=st.INK, lw=1.1, zorder=9)
            ax.text(x0 + (x1 - x0) * 0.06 + dxk / 2, y0 + (y1 - y0) * 0.115,
                    f"{km:g} km", fontsize=5.8, ha="center", va="bottom",
                    color=st.INK, zorder=9)
            if c == 0:
                ax.text(-0.105, 0.5, ROW_LAB[r], transform=ax.transAxes,
                        fontsize=6.4, rotation=90, ha="right", va="center",
                        color=st.MUTED)

    hs_ = [Rectangle((0, 0), 1, 1, facecolor=PRE_2000)] + \
          [Rectangle((0, 0), 1, 1, facecolor=sm.EPOCH_COLORS[k])
           for k in (5, 6, 7, 8)]
    labs_ = ["impervious by 2000"] + [sm.EPOCH_YEARS[k] for k in (5, 6, 7, 8)]
    fig.legend(hs_, labs_, ncol=5, fontsize=6.2, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), handlelength=1.1, handleheight=0.9,
               columnspacing=1.5, frameon=False)
    return tp.save(fig, name)


if __name__ == "__main__":
    import sys
    for fn in sys.argv[1:]:
        print(globals()[fn]())
