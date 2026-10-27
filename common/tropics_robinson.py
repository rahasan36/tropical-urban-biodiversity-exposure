"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import pickle
import numpy as np
import pyogrio
import pyproj
import shapely
from shapely.geometry import box
from shapely.ops import transform as shp_transform, unary_union
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from tropics_config import NE_COUNTRIES, INTERIM_DIR

NE_PATH = NE_COUNTRIES
CACHE = INTERIM_DIR / "basemap_robinson_v2.pkl"

LAT_LIM = (-60.0, 84.0)
DROP = {"ATA"}
SIMPLIFY = 0.06

LON_0 = 31.0
LON_LO, LON_HI = LON_0 - 180.0, LON_0 + 180.0

CRS_ROBIN = pyproj.CRS.from_proj4(
    f"+proj=robin +lon_0={LON_0} +datum=WGS84 +units=m +no_defs")
_FWD = pyproj.Transformer.from_crs("EPSG:4326", CRS_ROBIN, always_xy=True)


def wrap_lon(lon):
    lon = np.asarray(lon, float)
    return ((lon - LON_0 + 180.0) % 360.0) - 180.0 + LON_0


X_MAX = _FWD.transform(LON_HI - 1e-9, 0.0)[0]
Y_TOP = _FWD.transform(0.0, LAT_LIM[1])[1]
Y_BOT = _FWD.transform(0.0, LAT_LIM[0])[1]

TOP_FACE = "#E3E1DE"
SIDE_WALL = "#BFBCB8"
BORDER = "#CFCCC8"
SHADOW = "#9A9794"
GRATICULE = "#D8D6D3"


def project(lon, lat):
    return _FWD.transform(wrap_lon(lon), np.asarray(lat, float))


def lat_to_y(lat):
    lat = np.asarray(lat, float)
    return _FWD.transform(np.zeros_like(lat), lat)[1]


def _load():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    gdf = pyogrio.read_dataframe(NE_PATH, columns=["ADM0_A3"])
    from shapely.affinity import translate as _translate
    east = box(LON_LO + 1e-6, LAT_LIM[0], 180.0, LAT_LIM[1])
    west = box(-180.0, LAT_LIM[0], LON_LO - 1e-6, LAT_LIM[1])
    geoms = []
    for g, a3 in zip(gdf.geometry, gdf.ADM0_A3):
        if g is None or g.is_empty or a3 in DROP:
            continue
        parts = []
        a = g.intersection(east)
        if not a.is_empty:
            parts.append(a)
        b = g.intersection(west)
        if not b.is_empty:
            parts.append(_translate(b, xoff=360.0))
        for gi in parts:
            gi = gi.simplify(SIMPLIFY, preserve_topology=True)
            if not gi.is_empty:
                geoms.append(gi)
    land = unary_union(geoms).buffer(0)
    out = (geoms, land)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


def _to_robin(geom):
    return shp_transform(lambda x, y, z=None: _FWD.transform(x, y), geom)


def _rings(geom):
    polys = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
    for p in polys:
        if p.is_empty:
            continue
        yield np.asarray(p.exterior.coords)
        for r in p.interiors:
            yield np.asarray(r.coords)


def _patches(geom, **kw):
    return PatchCollection([MplPolygon(r, closed=True) for r in _rings(geom)], **kw)


def draw_globe(ax, depth=0.0055, shadow=0.011, borders=True,
               graticule=True, zorder=0, min_span=4.0e5, top_face=None):
    geoms, land = _load()
    land_r = _to_robin(land)
    span = 2 * X_MAX
    dx, dy = depth * span, -depth * span * 0.70
    sx, sy = shadow * span, -shadow * span * 0.70

    big = [r for r in _rings(land_r)
           if (r[:, 0].max() - r[:, 0].min()) > min_span
           or (r[:, 1].max() - r[:, 1].min()) > min_span]

    def collection(rings, off, **kw):
        pc = PatchCollection(
            [MplPolygon(r + np.asarray(off), closed=True) for r in rings], **kw)
        ax.add_collection(pc)
        return pc

    for i, a in enumerate((0.10, 0.13, 0.16)):
        f = (i + 1) / 3
        collection(big, (sx * f, sy * f), facecolor=SHADOW, edgecolor="none",
                   alpha=a, zorder=zorder)
    collection(big, (dx, dy), facecolor=SIDE_WALL, edgecolor="none",
               zorder=zorder + 0.2)

    if graticule:
        draw_graticule(ax, zorder=zorder + 0.4)

    collection(list(_rings(land_r)), (0, 0), facecolor=top_face or TOP_FACE,
               edgecolor="none", zorder=zorder + 0.5)
    if borders:
        for g in geoms:
            for r in _rings(_to_robin(g)):
                ax.plot(r[:, 0], r[:, 1], color=BORDER, lw=0.25,
                        solid_joinstyle="round", zorder=zorder + 0.6)


def draw_graticule(ax, lon_step=30, lat_step=30, zorder=0.4, lw=0.3):
    lats = np.linspace(LAT_LIM[0], LAT_LIM[1], 200)
    for lon in np.arange(LON_LO, LON_HI + 1e-6, lon_step):
        x, y = project(np.full_like(lats, lon), lats)
        ax.plot(x, y, color=GRATICULE, lw=lw, zorder=zorder)
    lons = np.linspace(LON_LO + 1e-6, LON_HI - 1e-6, 200)
    for lat in range(-60, 91, lat_step):
        if not LAT_LIM[0] <= lat <= LAT_LIM[1]:
            continue
        x, y = project(lons, np.full_like(lons, lat))
        ax.plot(x, y, color=GRATICULE, lw=lw, zorder=zorder)


def frame(ax, pad=0.012):
    span = 2 * X_MAX
    ax.set_xlim(-X_MAX - pad * span, X_MAX + pad * span * 2.2)
    ax.set_ylim(Y_BOT - pad * span * 2.2, Y_TOP + pad * span)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("white")


def aspect():
    span = 2 * X_MAX
    w = span + 0.012 * span * 3.2
    h = (Y_TOP - Y_BOT) + 0.012 * span * 3.2
    return h / w


_INV = pyproj.Transformer.from_crs(CRS_ROBIN, "EPSG:4326", always_xy=True)


def warp_raster(rgba, extent, out_w=2200, lat_clip=None, origin="upper"):
    lon0, lon1, lat_b, lat_t = extent
    lat_top, lat_bot = max(lat_t, lat_b), min(lat_t, lat_b)
    h, w = rgba.shape[:2]
    out_h = int(round(out_w * (Y_TOP - Y_BOT) / (2 * X_MAX)))
    xs = np.linspace(-X_MAX, X_MAX, out_w)
    ys = np.linspace(Y_TOP, Y_BOT, out_h)
    gx, gy = np.meshgrid(xs, ys)
    lon, lat = _INV.transform(gx.ravel(), gy.ravel())
    lon = lon.reshape(gx.shape); lat = lat.reshape(gx.shape)

    out = np.zeros((out_h, out_w, 4), dtype=rgba.dtype)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(lon) & np.isfinite(lat)
    ok &= (lon >= lon0) & (lon <= lon1) & (lat <= lat_top) & (lat >= lat_bot)
    if lat_clip is not None:
        ok &= (lat >= lat_clip[0]) & (lat <= lat_clip[1])
    lon_s = np.where(ok, lon, lon0)
    lat_s = np.where(ok, lat, lat_top)
    col = np.clip(((lon_s - lon0) / (lon1 - lon0) * w).astype(int), 0, w - 1)
    frac = (lat_top - lat_s) / (lat_top - lat_bot)
    if origin != "upper":
        frac = 1.0 - frac
    row = np.clip((frac * h).astype(int), 0, h - 1)
    out[ok] = rgba[row[ok], col[ok]]
    return out, (-X_MAX, X_MAX, Y_BOT, Y_TOP)


def inside_sheet(ax, zorder=0.45, facecolor="white"):
    lats = np.linspace(LAT_LIM[0], LAT_LIM[1], 200)
    left = np.column_stack(project(np.full_like(lats, -180.0), lats))
    right = np.column_stack(project(np.full_like(lats, 180.0), lats[::-1]))
    return np.vstack([left, right])


TROPIC = 23.4394


LON_CROP = (-113.0, 158.0)


def band_limits(cover=0.90, pad_x=0.004):
    y_tropic = float(lat_to_y(TROPIC))
    y_half = y_tropic / cover
    x0 = float(project([LON_CROP[0]], [0.0])[0][0])
    x1 = float(project([LON_CROP[1]], [0.0])[0][0])
    span = x1 - x0
    return (x0 - pad_x * span, x1 + pad_x * span), (-y_half, y_half)


def frame_band(ax, cover=0.90, **kw):
    (x0, x1), (y0, y1) = band_limits(cover, **kw)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("white")


def band_aspect(cover=0.90, **kw):
    (x0, x1), (y0, y1) = band_limits(cover, **kw)
    return (y1 - y0) / (x1 - x0)
