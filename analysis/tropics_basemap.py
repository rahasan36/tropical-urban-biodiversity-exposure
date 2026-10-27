"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pyogrio, shapely, pickle
from shapely.geometry import box
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from tropics_config import DOWNLOADS, INTERIM_DIR, TROPIC_LAT
import tropics_style as st

NE_PATH = DOWNLOADS / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp"
CACHE = INTERIM_DIR / "basemap_tropics.pkl"
WORLD_CACHE = INTERIM_DIR / "basemap_world_v2.pkl"
WORLD_LAT = (-62.0, 62.0)
WORLD_SIMPLIFY = 0.05
WORLD_DROP = {"ATA"}

TROPIC_CACHE = INTERIM_DIR / "basemap_tropic_exact.pkl"
SIMPLIFY = 0.02

PANELS = [("Americas", -110.0, -33.0),
          ("Africa and western Asia", -20.0, 62.0),
          ("Eastern Asia and the Pacific", 62.0, 162.0)]
LAT_LIM = (-25.5, 25.5)


def _load_countries():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    gdf = pyogrio.read_dataframe(NE_PATH, columns=["ADM0_A3", "NAME", "CONTINENT"])
    clip = box(-180, LAT_LIM[0] - 2, 180, LAT_LIM[1] + 2)
    geoms, attrs = [], []
    for g, a3, nm, ct in zip(gdf.geometry, gdf.ADM0_A3, gdf.NAME, gdf.CONTINENT):
        if g is None or g.is_empty:
            continue
        gi = g.intersection(clip)
        if gi.is_empty:
            continue
        gi = gi.simplify(SIMPLIFY, preserve_topology=True)
        if gi.is_empty:
            continue
        geoms.append(gi); attrs.append((a3, nm, ct))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump((geoms, attrs), fh, protocol=4)
    return geoms, attrs


def _load_tropic():
    if TROPIC_CACHE.exists():
        with open(TROPIC_CACHE, "rb") as fh:
            return pickle.load(fh)
    from tropics_config import TROPIC_LAT as _T
    geoms, attrs = _load_countries()
    clip = box(-180, -_T, 180, _T)
    out = []
    for g in geoms:
        gi = g.intersection(clip)
        if not gi.is_empty:
            out.append(gi)
    TROPIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(TROPIC_CACHE, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


def _load_world():
    if WORLD_CACHE.exists():
        with open(WORLD_CACHE, "rb") as fh:
            return pickle.load(fh)
    gdf = pyogrio.read_dataframe(NE_PATH, columns=["ADM0_A3", "NAME", "CONTINENT"])
    clip = box(-180, WORLD_LAT[0], 180, WORLD_LAT[1])
    geoms = []
    for g, a3 in zip(gdf.geometry, gdf.ADM0_A3):
        if g is None or g.is_empty or a3 in WORLD_DROP:
            continue
        gi = g.intersection(clip)
        if gi.is_empty:
            continue
        gi = gi.simplify(WORLD_SIMPLIFY, preserve_topology=True)
        if gi.is_empty:
            continue
        geoms.append(gi)
    WORLD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(WORLD_CACHE, "wb") as fh:
        pickle.dump(geoms, fh, protocol=4)
    return geoms


def _polys(geom):
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in geom.geoms if g.geom_type == "Polygon"]


def draw_land(ax, face=None, edge=None, lw=0.25, values=None, cmap=None,
              vmin=None, vmax=None, colors=None):
    geoms, attrs = _load_countries()
    patches, facecols = [], []
    for g, (a3, nm, ct) in zip(geoms, attrs):
        if values is not None:
            v = values.get(a3)
            if v is None or not np.isfinite(v):
                fc = "#FBFAF8"
            else:
                t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
                fc = cmap(np.clip(t, 0, 1))
        elif colors is not None:
            fc = colors.get(a3, "#FBFAF8")
        else:
            fc = face or st.LAND
        for p in _polys(g):
            patches.append(MplPolygon(np.asarray(p.exterior.coords), closed=True))
            facecols.append(fc)
    pc = PatchCollection(patches, facecolors=facecols,
                         edgecolors=(edge or st.LAND_EDGE), linewidths=lw, zorder=1)
    ax.add_collection(pc)
    return pc


DEG_PER_IN = 17.0
HEADER_IN = 0.62
GAP_IN = 0.16
LEFT_IN, RIGHT_IN, BOTTOM_IN = 0.30, 0.16, 0.95


def panel_figure_legacy(draw=True, deg_per_in=DEG_PER_IN, footer_in=BOTTOM_IN, **land_kw):
    st.apply_rc()
    lat_span = LAT_LIM[1] - LAT_LIM[0]
    widths = [(hi - lo) / deg_per_in for _, lo, hi in PANELS]
    ph = lat_span / deg_per_in
    fig_w = LEFT_IN + max(widths) + RIGHT_IN
    fig_h = HEADER_IN + len(PANELS) * ph + (len(PANELS) - 1) * GAP_IN + footer_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for i, ((name, lo, hi), w) in enumerate(zip(PANELS, widths)):
        bottom = footer_in + (len(PANELS) - 1 - i) * (ph + GAP_IN)
        ax = fig.add_axes([LEFT_IN / fig_w, bottom / fig_h, w / fig_w, ph / fig_h])
        ax.set_xlim(lo, hi); ax.set_ylim(*LAT_LIM)
        ax.set_facecolor(st.SEA)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(st.FAINT); sp.set_linewidth(0.5)
        if draw:
            draw_land(ax, **land_kw)
        for y in (-TROPIC_LAT, TROPIC_LAT):
            ax.axhline(y, color=st.MUTED, lw=0.45, ls=(0, (5, 4)), zorder=2, alpha=0.6)
        ax.text(0.004, 0.965, name, transform=ax.transAxes, fontsize=7,
                va="top", ha="left", color=st.MUTED, zorder=6,
                bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none", alpha=0.82))
        axes.append(ax)
    fig._panel_geom = dict(fig_w=fig_w, fig_h=fig_h, header=HEADER_IN,
                           footer=footer_in, left=LEFT_IN)
    return fig, axes


def title_block(fig, title, subtitle):
    g = fig._panel_geom
    x = g["left"] / g["fig_w"]
    fig.text(x, 1 - 0.10 / g["fig_h"], title, fontsize=11.5, fontweight="bold",
             va="top", ha="left", color=st.INK)
    fig.text(x, 1 - 0.30 / g["fig_h"], subtitle, fontsize=7.5, va="top",
             ha="left", color=st.MUTED, wrap=True)


def footer_axes(fig, w_in, h_in, x_in=None, y_in=0.30):
    g = fig._panel_geom
    x_in = g["left"] if x_in is None else x_in
    return fig.add_axes([x_in / g["fig_w"], y_in / g["fig_h"],
                         w_in / g["fig_w"], h_in / g["fig_h"]])


def scatter_panels(axes, lon, lat, **kw):
    lon = np.asarray(lon); lat = np.asarray(lat)
    out = []
    for ax, (_, lo, hi) in zip(axes, PANELS):
        m = (lon >= lo) & (lon <= hi)
        sub = {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == lon.shape else v)
               for k, v in kw.items()}
        out.append(ax.scatter(lon[m], lat[m], **sub))
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    geoms, attrs = _load_countries()
    print(f"basemap cached: {len(geoms)} country geometries, "
          f"{CACHE.stat().st_size/1e6:.1f} MB")
    import collections
    print(collections.Counter(a[2] for a in attrs).most_common(8))


def panel_figure_raster(land_face=None, land_edge=None, footer_in=1.05, **kw):
    fig, axes = panel_figure_legacy(footer_in=footer_in, **kw)
    if land_face or land_edge:
        for ax in axes:
            for coll in ax.collections:
                if land_face:
                    coll.set_facecolor(land_face)
                if land_edge:
                    coll.set_edgecolor(land_edge)
    return fig, axes, 0.34 / fig._panel_geom["fig_h"]


def panel_figure(**kw):
    return panel_figure_legacy(**kw)
