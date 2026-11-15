"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import argparse
import glob
import os
import pickle
import re
import runpy
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
CODE_DIRS = ["common", "measurement", "sensitivity", "temporal", "models", "reallocation",
             "supplementary_tables", "figures/plotting"]
OUT = os.path.join(REPO, "data", "workbook", "Source_Data_Figures.xlsx")
COLUMNS = ["panel", "axes", "element", "series", "id", "color", "x", "y", "x2", "y2", "value", "size"]
DECOR = {"tropics_robinson": ["draw_globe", "draw_graticule", "inside_sheet"],
         "tropics_basemap": ["draw_land"],
         "tropics_frame": ["_draw_land", "_outline", "_graticule", "scale_bar", "north_arrow", "corner_labels",
                           "band_degree_labels", "robinson_degree_labels", "class_legend", "class_legend_h",
                           "map_legend", "footer_legend", "shared_scale_bar"],
         "tropics_style": ["add_bivariate_key"],
         "tropics_main_figures": ["_vlegend", "_region_outline", "_tropic_lines", "_fig5_region_keys",
                                  "_fig5_map_key"],
         "region_panels": ["_key", "_inmap_keys"],
         "fig_bivariate": ["key_square"],
         "fig_bivariate_decomp": ["strip_key"],
         "fig_trivariate": ["tri_key"],
         "fig_country_violin": ["key"]}
LONLAT = ("draw_land", "_draw_land", "_outline")
MARKS = ["points", "circles", "hexbin", "cells"]
MAX_VERTICES = 20000
MAX_PATCH = 2000
NOTES = {"line": "x and y of each vertex of a plotted line",
         "points": "x and y of each marker, with the value mapped to color and the marker area in points squared",
         "circles": "x and y of each circle center, with its fill color and its radius in the units of the drawing",
         "bar": "x and y of the lower left corner of a bar, x2 its width and y2 its height",
         "band": "x and y of each vertex of a filled band or distribution outline",
         "hexbin": "x and y of each hexagon center, with the count or statistic mapped to color",
         "cells": "x and y of each cell center, with the value mapped to color",
         "segment": "x and y of each vertex of a line segment",
         "polygon": "x and y of each vertex of a drawn polygon, including the boxes of box plots",
         "area": "one row per country of a shaded map, with the median mapped to its color as value and the "
                 "number of agglomerations behind it as size. The value is blank where the country is drawn "
                 "in grey"}


def country_medians(ax, d, min_n=10, **k):
    c = d.groupby("country").agg(n=("city_id", "size"), med=("cost_per_km2", "median")).reset_index()
    return pd.DataFrame({"element": "area", "series": c.country.astype(str), "value": c.med.where(c.n >= min_n),
                         "size": c.n.astype(float)})


EXTRA = {"fig_country_violin": {"panel_map": country_medians}}


def all_axes(fig):
    out, stack = [], list(fig.axes)
    while stack:
        ax = stack.pop(0)
        if ax not in out:
            out.append(ax)
            stack.extend(getattr(ax, "child_axes", []))
    return out


def every(fig):
    out = list(fig.texts) + list(fig.patches) + list(fig.lines) + list(fig.images)
    for ax in all_axes(fig):
        out += list(ax.lines) + list(ax.patches) + list(ax.collections) + list(ax.images) + list(ax.texts)
    return out


def target(a, k):
    for v in list(a) + list(k.values()):
        if hasattr(v, "transData"):
            return v, v.figure
        if hasattr(v, "add_axes"):
            return None, v
    import matplotlib.pyplot as plt
    return None, plt.gcf() if plt.get_fignums() else None


def tag(m, name, extra=None):
    f = getattr(m, name)

    def wrapped(*a, **k):
        ax, fig = target(a, k)
        seen = {id(x) for x in every(fig)} if fig is not None else set()
        axes = {id(x) for x in all_axes(fig)} if fig is not None else set()
        r = f(*a, **k)
        if extra is not None:
            if ax is not None:
                ax._extra = extra(*a, **k)
            return r
        if fig is not None:
            for x in every(fig):
                if id(x) not in seen:
                    x._decor = True
            for x in all_axes(fig):
                if id(x) not in axes:
                    x._skip = True
        if ax is not None and name == "draw_globe":
            ax._robinson = True
        if ax is not None and name in LONLAT:
            ax._lonlat = True
        return r

    setattr(m, name, wrapped)


def watch():
    import builtins
    keep = builtins.__import__
    todo = [(mod, name, None) for mod, names in DECOR.items() for name in names]
    todo += [(mod, name, f) for mod, names in EXTRA.items() for name, f in names.items()]

    def imported(name, *a, **k):
        r = keep(name, *a, **k)
        for item in list(todo):
            m = sys.modules.get(item[0])
            if m is not None and hasattr(m, item[1]):
                todo.remove(item)
                tag(m, item[1], item[2])
        return r

    builtins.__import__ = imported


def track(root):
    import builtins
    keep, seen, root = builtins.open, [], os.path.realpath(root)

    def note(path):
        try:
            p = os.path.realpath(os.fspath(path))
        except TypeError:
            return
        if p.startswith(root + os.sep):
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if rel not in seen:
                seen.append(rel)

    def opened(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        if not isinstance(file, int) and "w" not in mode and "a" not in mode and "x" not in mode:
            note(file)
        return keep(file, *a, **k)

    def wrap(f):
        def g(path, *a, **k):
            note(path)
            return f(path, *a, **k)
        return g

    import io
    import pathlib
    builtins.open = io.open = opened
    for name in ("open", "read_text", "read_bytes"):
        setattr(pathlib.Path, name, (lambda f: lambda self, *a, **k: (note(self), f(self, *a, **k))[1])(
            getattr(pathlib.Path, name)))
    pd.read_parquet = wrap(pd.read_parquet)
    for mod, name in (("pyogrio", "read_dataframe"), ("geopandas", "read_file"), ("shapefile", "Reader")):
        try:
            m = __import__(mod)
            setattr(m, name, wrap(getattr(m, name)))
        except (ImportError, AttributeError):
            pass
    return seen


def in_data(ax, transform, both=True):
    try:
        hit = transform.contains_branch_seperately(ax.transData)
    except Exception:
        return True
    return all(hit) if both else any(hit)


def clean(text):
    for a, b in (("\\alpha", "α"), ("\\rho", "ρ"), ("\\beta", "β"), ("\\geq", "≥"), ("\\times", "×"), ("$", ""),
                 ("\n", " ")):
        text = text.replace(a, b)
    return " ".join(text.split())


def hexes(colors, n):
    from matplotlib.colors import to_hex
    c = np.atleast_2d(colors)
    if len(c) == n:
        return [to_hex(v) for v in c]
    return [to_hex(c[0])] * n if len(c) else [None] * n


def block(element, series, x, y, **extra):
    d = pd.DataFrame({"element": element, "series": series, "x": np.asarray(x, float), "y": np.asarray(y, float)})
    for k, v in extra.items():
        d[k] = v
    return d


def kind(ax):
    from matplotlib.collections import PatchCollection
    if getattr(ax, "_robinson", False):
        return "robinson"
    if getattr(ax, "_lonlat", False):
        return "lonlat"
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    bare = not ax.axison or (len(ax.get_xticks()) == 0 and len(ax.get_yticks()) == 0)
    drawn = len(ax.images) > 0 or any(isinstance(c, PatchCollection) for c in ax.collections)
    if bare and drawn and x0 >= -180.5 and x1 <= 180.5 and y0 >= -90.5 and y1 <= 90.5:
        return "lonlat"
    return None


def same(a, b):
    a, b = sorted(a), sorted(b)
    return bool(np.allclose(a, b, rtol=1e-6) and max(abs(v) for v in b) > 1e5)


def lon_of(x):
    import tropics_robinson as rb
    x = np.asarray(x, float)
    out, ok = np.full(x.shape, np.nan), np.isfinite(x)
    out[ok] = ((np.asarray(rb._INV.transform(x[ok], np.zeros(ok.sum()))[0]) + 180.0) % 360.0) - 180.0
    return out


def lat_of(y):
    import tropics_robinson as rb
    y = np.asarray(y, float)
    out, ok = np.full(y.shape, np.nan), np.isfinite(y)
    out[ok] = np.asarray(rb._INV.transform(np.zeros(ok.sum()), y[ok])[1])
    return out


def harvest_axes(ax, sort, rx=False, ry=False):
    from matplotlib.collections import LineCollection, PatchCollection, PathCollection, PolyCollection, QuadMesh
    from matplotlib.colors import to_hex
    from matplotlib.patches import Rectangle
    out, notes, plain = [], [], sort is None
    if plain:
        for i, ln in enumerate(ax.lines):
            x, y = np.asarray(ln.get_xdata(), float).ravel(), np.asarray(ln.get_ydata(), float).ravel()
            if getattr(ln, "_decor", False) or x.size == 0 or x.size > MAX_VERTICES:
                continue
            if not in_data(ax, ln.get_transform(), False):
                continue
            lab = ln.get_label()
            out.append(block("line", lab if lab and not lab.startswith("_") else f"line {i + 1}", x, y,
                             color=to_hex(ln.get_color())))
    k = 0
    for c in ax.collections:
        if getattr(c, "_decor", False):
            continue
        if not in_data(ax, c.get_offset_transform() if isinstance(c, PathCollection) else c.get_transform()):
            continue
        k += 1
        lab = c.get_label()
        name = lab if lab and not str(lab).startswith("_") else f"series {k}"
        arr = c.get_array()
        arr = None if arr is None else np.ma.filled(np.ma.asarray(arr, float), np.nan).ravel()
        if isinstance(c, QuadMesh):
            shape = tuple(c.get_coordinates().shape[:2])
            if min(shape) > 3:
                notes.append(f"{shape[0] - 1} × {shape[1] - 1} cells")
        elif isinstance(c, PathCollection):
            off = np.asarray(c.get_offsets(), float)
            if len(off) == 0 or (not plain and len(off) < 4):
                continue
            size = np.asarray(c.get_sizes(), float)
            extra = {"color": hexes(c.get_facecolors(), len(off))}
            if arr is not None and len(arr) == len(off):
                extra["value"] = arr
            extra["size"] = size if len(size) == len(off) else (size[0] if len(size) else np.nan)
            out.append(block("points", name, off[:, 0], off[:, 1], **extra))
        elif isinstance(c, PatchCollection):
            paths = c.get_paths()
            count = {len(p.vertices) for p in paths}
            if len(paths) > 3 and len(count) == 1 and min(count) >= 20:
                v = np.array([p.vertices for p in paths], float)
                lo, hi = v.min(axis=1), v.max(axis=1)
                out.append(block("circles", name, 0.5 * (lo[:, 0] + hi[:, 0]), 0.5 * (lo[:, 1] + hi[:, 1]),
                                 color=hexes(c.get_facecolors(), len(paths)), size=0.5 * (hi[:, 0] - lo[:, 0])))
        elif isinstance(c, PolyCollection):
            paths, off = c.get_paths(), np.asarray(c.get_offsets(), float)
            if arr is not None and len(paths) == 1 and len(off) == len(arr) and len(arr) > 1:
                out.append(block("hexbin", name, off[:, 0], off[:, 1], value=arr))
            elif arr is not None and len(paths) == len(arr) and len(arr) > 1:
                cen = np.array([p.vertices.mean(axis=0) for p in paths])
                out.append(block("cells", name, cen[:, 0], cen[:, 1], value=arr))
            elif plain and sum(len(p.vertices) for p in paths) <= MAX_VERTICES:
                col = hexes(c.get_facecolors(), len(paths))
                for j, p in enumerate(paths):
                    out.append(block("band", f"{name}, part {j + 1}" if len(paths) > 1 else name,
                                     p.vertices[:, 0], p.vertices[:, 1], color=col[j]))
        elif isinstance(c, LineCollection) and plain:
            segs = c.get_segments()
            if sum(len(s) for s in segs) <= MAX_VERTICES:
                col = hexes(c.get_colors(), len(segs))
                for j, s in enumerate(segs):
                    s = np.asarray(s, float)
                    extra = {"color": col[j]}
                    if arr is not None and len(arr) == len(segs):
                        extra["value"] = arr[j]
                    out.append(block("segment", f"{name}, part {j + 1}", s[:, 0], s[:, 1], **extra))
    for im in ax.images:
        shape = tuple(np.shape(im.get_array())[:2])
        if not getattr(im, "_decor", False) and min(shape) > 2:
            notes.append(f"{shape[0]} × {shape[1]} cells")
    rows = []
    for p in ax.patches if plain else []:
        if getattr(p, "_decor", False) or p is ax.patch or not in_data(ax, p.get_data_transform()):
            continue
        if isinstance(p, Rectangle):
            rows.append(("bar", p.get_x(), p.get_y(), p.get_width(), p.get_height(), to_hex(p.get_facecolor())))
        else:
            v = p.get_path().vertices
            if len(v) <= MAX_PATCH:
                v = p.get_patch_transform().transform(v)
                out.append(block("polygon", f"polygon {len(out) + 1}", v[:, 0], v[:, 1],
                                 color=to_hex(p.get_facecolor())))
    if rows:
        b = pd.DataFrame(rows, columns=["element", "x", "y", "x2", "y2", "color"])
        b["series"] = [f"bar {i + 1}" for i in range(len(b))]
        out.append(b)
    d = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLUMNS[2:])
    d = d.reindex(columns=COLUMNS[2:])
    if len(d):
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        inside = d.x.between(x0, x1) & d.y.between(y0, y1)
        d = d[~d.element.isin(MARKS) | inside].reset_index(drop=True)
    extra = getattr(ax, "_extra", None)
    if extra is not None:
        d = pd.concat([d, extra.reindex(columns=COLUMNS[2:])], ignore_index=True)
    if len(d) and sort == "robinson":
        import tropics_robinson as rb
        x, y = np.array(d.x, float), np.array(d.y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        lon, lat = rb._INV.transform(x[ok], y[ok])
        x[ok], y[ok] = ((np.asarray(lon) + 180.0) % 360.0) - 180.0, lat
        d["x"], d["y"] = x, y
    elif len(d):
        for on, a, b, f in ((rx, "x", "x2", lon_of), (ry, "y", "y2", lat_of)):
            if on:
                v, w = np.array(d[a], float), np.array(d[b], float)
                d[b] = f(v + w) - f(v)
                d[a] = f(v)
    return d, notes


def ticks(axis):
    t = [x.get_text() for x in axis.get_ticklabels() if x.get_text()]
    return "; ".join(clean(x) for x in t) if t and not all(re.fullmatch(r"[−\-+\d.,% ×^{}$\\a-z]*", x) for x in t) else ""


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def harvest(fig):
    from matplotlib.text import Annotation
    inv = fig.transFigure.inverted()
    axes = [ax for ax in all_axes(fig) if not getattr(ax, "_skip", False) and ax.get_visible()]
    maps = [ax for ax in axes if getattr(ax, "_robinson", False)]
    texts = []
    renderer = fig.canvas.get_renderer()
    for ax in [None] + axes:
        for t in fig.texts if ax is None else list(ax.texts) + [ax.title]:
            s = t.get_text()
            if not s.strip() or getattr(t, "_decor", False) or isinstance(t, Annotation) or not t.get_visible():
                continue
            try:
                fx, fy = inv.transform(t.get_transform().transform(t.get_position()))
            except Exception:
                continue
            w = t.get_fontweight()
            bold = w in ("bold", "heavy", "semibold", "demibold", "black") or (not isinstance(w, str) and w >= 600)
            try:
                box = t.get_window_extent(renderer).transformed(inv)
                extent = (float(box.x0), float(box.y0), float(box.x1), float(box.y1))
            except Exception:
                extent = (float(fx), float(fy), float(fx), float(fy))
            texts.append((s, float(fx), float(fy), bold, extent))
    data = []
    for i, ax in enumerate(axes):
        sort = kind(ax)
        rx = sort is None and any(same(ax.get_xlim(), m.get_xlim()) for m in maps)
        ry = sort is None and any(same(ax.get_ylim(), m.get_ylim()) for m in maps)
        d, notes = harvest_axes(ax, sort, rx, ry)
        if len(d) or notes:
            b = ax.get_position()
            data.append(dict(i=i + 1, ax=ax, d=d, notes=notes, box=(b.x0, b.y0, b.x1, b.y1), panel=None,
                             sort=sort, rx=rx, ry=ry))
    area = lambda a: (a["box"][2] - a["box"][0]) * (a["box"][3] - a["box"][1])
    found = {}
    for s, fx, fy, bold, _ in texts:
        m = re.fullmatch(r"\s*([a-z])(?:\s{2,}\S.*)?\s*", s, re.S)
        if not m or not bold:
            continue
        cand = []
        for k, a in enumerate(data):
            x0, y0, x1, y1 = a["box"]
            if fx <= x0 + 0.5 * (x1 - x0) and fy >= y1 - 0.35 * (y1 - y0):
                cand.append((float(np.hypot(fx - x0, fy - y1)), k))
        if cand:
            near = min(cand)[0]
            dist, k = max((c for c in cand if c[0] <= near + 0.10), key=lambda c: area(data[c[1]]))
            if m.group(1) not in found or near < found[m.group(1)][0]:
                found[m.group(1)] = (near, k)
    best = {}
    for s, (dist, k) in found.items():
        if k not in best or dist < best[k][0]:
            best[k] = (dist, s)
    for k, (_, s) in best.items():
        data[k]["panel"] = s
    lettered = [a for a in data if a["panel"]]
    for a in data:
        if a["panel"] or not lettered:
            continue
        x0, y0, x1, y1 = a["box"]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        inside = [b for b in lettered if b["box"][0] <= cx <= b["box"][2] and b["box"][1] <= cy <= b["box"][3]]
        row = [b for b in lettered if b["box"][0] < x0 and overlap(y0, y1, b["box"][1], b["box"][3]) >
               0.5 * min(y1 - y0, b["box"][3] - b["box"][1])]
        above = [b for b in lettered if b["box"][1] >= y1 - 1e-6 and overlap(x0, x1, b["box"][0], b["box"][2]) >
                 0.5 * min(x1 - x0, b["box"][2] - b["box"][0])]
        if inside:
            pick = min(inside, key=area)
        elif row:
            pick = max(row, key=lambda b: b["box"][0])
        elif above:
            pick = min(above, key=lambda b: b["box"][1])
        else:
            pick = min(lettered, key=lambda b: np.hypot(max(b["box"][0] - cx, 0, cx - b["box"][2]),
                                                        max(b["box"][1] - cy, 0, cy - b["box"][3])))
        a["panel"] = pick["panel"]
    frames, contents = [], []
    for a in data:
        ax, d = a["ax"], a["d"]
        x0, y0, x1, y1 = a["box"]
        panel = a["panel"] or "all"
        others = [b["box"] for b in data if b is not a]
        head = []
        for s, fx, fy, bold, (tx0, ty0, tx1, ty1) in texts:
            cx, cy = 0.5 * (tx0 + tx1), 0.5 * (ty0 + ty1)
            if re.fullmatch(r"\s*[a-z]\s*", s) or re.fullmatch(r"[\d.,%\s−+≥<>-]*", s):
                continue
            if not (y1 - 0.004 <= ty0 <= y1 + 0.06 and x0 - 0.02 <= cx <= x1):
                continue
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in others):
                continue
            if any(b[0] <= cx <= b[2] and 0 <= b[1] - ty1 < ty0 - y1 for b in others):
                continue
            head.append((-ty0, tx0, re.sub(r"^\s*[a-z]\s{2,}", "", s).strip()))
        head = sorted(head)
        if len(d):
            d = d.copy()
            d.insert(0, "axes", a["i"])
            d.insert(0, "panel", panel)
            frames.append(d)
        counts = d.groupby("element").size().to_dict() if len(d) else {}
        on_map = a["sort"] is not None
        contents.append(dict(panel=panel, axes=a["i"], title="; ".join(clean(s) for _, _, s in head),
                             map="yes" if on_map else "",
                             x_axis="longitude (degrees)" if on_map or a["rx"] else clean(ax.get_xlabel()),
                             y_axis="latitude (degrees)" if on_map or a["ry"] else clean(ax.get_ylabel()),
                             x_categories=ticks(ax.xaxis), y_categories=ticks(ax.yaxis),
                             elements="; ".join(f"{k} {v:,}" for k, v in sorted(counts.items())),
                             gridded="; ".join(a["notes"]), rows=int(len(d))))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return table.reindex(columns=COLUMNS), pd.DataFrame(contents)


def identify(table, contents, root, inputs):
    on_map = set(contents.loc[contents["map"] == "yes", "axes"]) if len(contents) else set()
    rows = table.index[(table.element == "points") & table["axes"].isin(on_map)]
    for rel in inputs if len(rows) else []:
        path = os.path.join(root, rel)
        if not rel.endswith((".csv", ".parquet")):
            continue
        cols = list(pd.read_csv(path, nrows=0).columns) if rel.endswith(".csv") else list(pd.read_parquet(path).columns)
        if not {"city_id", "lon", "lat"} <= set(cols):
            continue
        use = ["city_id", "lon", "lat"]
        t = pd.read_csv(path, usecols=use) if rel.endswith(".csv") else pd.read_parquet(path, columns=use)
        t = t.drop_duplicates("city_id").dropna()
        if not len(t):
            continue
        ref = t[["lon", "lat"]].to_numpy(float)
        xy = table.loc[rows, ["x", "y"]].to_numpy(float)
        ids = np.full(len(xy), None, dtype=object)
        for s in range(0, len(xy), 500):
            q = xy[s:s + 500]
            gap = np.abs(q[:, None, :] - ref[None, :, :]).max(axis=2)
            j = gap.argmin(axis=1)
            hit = gap[np.arange(len(q)), j] < 1e-3
            ids[s:s + 500][hit] = t.city_id.to_numpy()[j[hit]]
        table["id"] = table["id"].astype(object)
        table.loc[rows, "id"] = ids
        return rel
    return ""


def run(folder, data, pipeline, tmp):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    got, seen = [], set()

    def savefig(self, *a, **k):
        if id(self) not in seen:
            seen.add(id(self))
            got.append(harvest(self))

    script = glob.glob(os.path.join(HERE, folder, "make_*.py"))[0]
    root = os.environ.get("TROPICS_PROJECT", data) if pipeline else data
    inputs = track(root)
    keep, old = Figure.savefig, sys.argv
    Figure.savefig = savefig
    try:
        sys.argv = [script, "--out", tmp] + (["--pipeline"] if pipeline else ["--data", data])
        runpy.run_path(script, run_name="__main__")
    finally:
        Figure.savefig, sys.argv = keep, old
    table, contents = got[-1]
    for f in list(inputs):
        if f.endswith(".shp"):
            inputs += [f[:-4] + e for e in (".dbf", ".shx", ".prj", ".cpg") if os.path.exists(os.path.join(root, f[:-4] + e))]
    inputs = sorted(set(inputs))
    return table, contents, inputs, identify(table, contents, root, inputs)


def label(folder):
    m = re.match(r"(ED)?Fig0*(\d+)", folder)
    return ("Extended Data Fig. " if m.group(1) else "Fig. ") + m.group(2), ("EDFig" if m.group(1) else "Fig") + m.group(2)


def signif(d):
    for c in ("x", "y", "x2", "y2", "value", "size"):
        v = np.array(pd.to_numeric(d[c], errors="coerce"), float)
        ok = np.isfinite(v)
        v[ok] = [float(f"{x:.6g}") for x in v[ok]]
        d[c] = v
    return d


def assemble(cache, folders, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    index, files, tables, keyed = [], [], {}, []
    for f in folders:
        with open(os.path.join(cache, f + ".pkl"), "rb") as h:
            table, contents, inputs, key = pickle.load(h)
        name, sheet = label(f)
        contents.insert(0, "sheet", sheet)
        contents.insert(0, "figure", name)
        index.append(contents)
        files.append(pd.DataFrame({"figure": name, "sheet": sheet, "file": inputs}))
        tables[sheet] = signif(table)
        keyed += [key] if key else []
    keyed = ", ".join(sorted(set(keyed)))
    readme = pd.DataFrame({"item": ["Workbook", "Sheets", "Contents", "Inputs", "panel", "axes", "element", "series",
                                    "id", "color", "x, y", "x2, y2", "value", "size", "Maps", "Gridded surfaces"]
                                   + [f"element: {k}" for k in NOTES],
                           "description": [
        "Values plotted in each figure and Extended Data figure of the paper, read from the figures "
        "as the scripts in figures/ draw them",
        "One sheet per figure (Fig1 to Fig4, EDFig1 to EDFig8), one row per plotted value. Extended Data Tables 1 and 2 are in Supplementary Data 2",
        "Panels of each figure with their axes, the quantities on the axes, the kinds of marks and the number of rows",
        "Files in data/figures that each figure script reads",
        "Panel letter. A panel drawn on several axes (a map with its windows, a profile beside a map, an inset or "
        "a second scale) repeats the letter and the axes column separates them. A figure without letters is marked all",
        "Number of the axes within the figure, in the order they are drawn. Contents gives the title of each",
        "Kind of mark the row belongs to (see the element rows below)",
        "Name of the plotted series, from the legend where the figure has one",
        "Identifier of the agglomeration (city_id) for each marker on a map"
        + (f", as in data/figures/{keyed}" if keyed else "")
        + ". The quantities mapped to marker color and size are in the files listed for that figure in Inputs",
        "Color of the mark as a hexadecimal code",
        "Position in the units of the axes given in Contents",
        "Width and height of a bar",
        "Value mapped to the color of a mark",
        "Marker area in points squared where marker size carries a quantity, or the radius of a circle",
        "Map panels give longitude and latitude in degrees and list the plotted markers. Coastlines, borders, "
        "graticules, window outlines and keys are left out",
        "A surface drawn as an image is listed in Contents by its number of cells. Its values are in the gridded "
        "files named for that figure in Inputs"] + list(NOTES.values())})
    import xlsxwriter
    sheets = {"README": readme, "Contents": pd.concat(index, ignore_index=True),
              "Inputs": pd.concat(files, ignore_index=True)}
    sheets.update(tables)
    book = xlsxwriter.Workbook(out, {"in_memory": True})
    bold = book.add_format({"bold": True})
    for sheet, t in sheets.items():
        ws = book.add_worksheet(sheet)
        ws.freeze_panes(1, 0)
        for j, c in enumerate(t.columns):
            width = max([len(str(c))] + [len(str(v)) for v in t[c].head(200)])
            ws.set_column(j, j, min(max(width + 2, 8), 110 if sheet == "README" else 48))
        ws.write_row(0, 0, [str(c) for c in t.columns], bold)
        for i, row in enumerate(zip(*[t[c].tolist() for c in t.columns]), start=1):
            ws.write_row(i, 0, [None if v is None or v != v else v.item() if hasattr(v, "item") else v for v in row])
    book.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--data", default=os.path.join(REPO, "data", "figures"))
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--one", action="store_true")
    a = ap.parse_args()
    every_folder = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(HERE, "*Fig*", "make_*.py")))
    every_folder = sorted(every_folder, key=lambda f: (f.startswith("ED"), f))
    cache = a.cache or tempfile.mkdtemp()
    os.makedirs(cache, exist_ok=True)
    data = os.path.abspath(a.data)
    if a.one:
        tmp = tempfile.mkdtemp()
        if not a.pipeline:
            os.environ["TROPICS_INTERIM"] = os.path.join(data, "interim")
            os.environ["TROPICS_TABLES"] = os.path.join(data, "tables")
            os.environ["TROPICS_RAW"] = os.path.join(data, "raw")
            os.environ["TROPICS_NE"] = os.path.join(data, "naturalearth", "ne_10m_admin_0_countries.shp")
            os.environ["TROPICS_GISD30"] = os.path.join(data, "gisd30")
        os.environ["TROPICS_FIG_DIR"] = tmp
        os.environ.setdefault("MPLBACKEND", "Agg")
        sys.path[:0] = [os.path.join(REPO, *d.split("/")) for d in CODE_DIRS]
        watch()
        with open(os.path.join(cache, a.folders[0] + ".pkl"), "wb") as h:
            pickle.dump(run(a.folders[0], data, a.pipeline, tmp), h)
        return
    if not a.assemble:
        for f in a.folders or every_folder:
            cmd = [sys.executable, os.path.abspath(__file__), f, "--one", "--cache", cache, "--data", data]
            subprocess.run(cmd + (["--pipeline"] if a.pipeline else []), check=True, stdout=subprocess.DEVNULL)
            print(f, flush=True)
    if a.assemble or not a.folders:
        print(assemble(cache, every_folder, a.out))


if __name__ == "__main__":
    main()
