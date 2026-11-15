"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

INK, MUTED, FAINT, HAIR = "#1A1A1A", "#5A5A5A", "#C9C9C9", "#E6E6E6"
SURFACE = "#FFFFFF"
LAND, LAND_EDGE, SEA = "#F2F1EE", "#D6D4D0", "#FFFFFF"

CAT = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
CAT_MARKERS = ["o", "s", "^", "D"]
BLUE, ORANGE, TEAL, PINK = CAT
CRIMSON = "#B0184E"
OKABE_ITO = ["#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#56B4E9"]
REALM_COLOR = {"Indomalayan": "#0072B2", "Neotropic": "#E69F00",
               "Afrotropic": "#009E73", "Australasia": "#CC79A7"}
SET_COLOR = {"all": ("#555555", "#BBBBBB"), "threatened": ("#0072B2", "#9EC9E2"),
             "narrow": ("#CC79A7", "#E8C3D8"), "rarity": ("#E69F00", "#F5D08A")}

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#EDF2FC", "#C3D3F0", "#8FAEE4", "#5585D2", "#2B62C4", "#173F86"])
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange", ["#FCF0E6", "#F6D6B8", "#EDB07C", "#DC8640", "#C05A0F", "#7E3B08"])
SEQ_PURPLE = LinearSegmentedColormap.from_list(
    "seq_purple", ["#F2EEF6", "#DCD0E8", "#BFA9D6", "#9C7DC0", "#7651A5", "#4E2A84"])
SEQ_GREEN = SEQ_PURPLE
DIV = LinearSegmentedColormap.from_list(
    "div", ["#173F86", "#5585D2", "#C3D3F0", "#EEEDEA", "#F6D6B8", "#DC8640", "#7E3B08"])

HABITAT_COLORS = {
    "forest": "#117733", "shrub": "#999933", "grass": "#DDCC77",
    "sparse": "#E5DEC8", "wetland": "#44AA99", "cropland": "#CC6677",
    "impervious": "#882255", "bare": "#BBBBBB", "water": "#88CCEE",
}
FORM_COLORS = {"infill": "#0072B2", "edge": "#E69F00", "leapfrog": "#CC79A7"}


def darker(color, f=0.28):
    r, g, b = to_rgb(color)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def text_on(color):
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in to_rgb(color))
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if (1.05 / (L + 0.05)) > ((L + 0.05) / 0.05) else INK


FONT_STACK = ["Arial", "Helvetica"]
MIN_PT = 8.0


def enforce_min_pt(min_pt=MIN_PT):
    from matplotlib.font_manager import FontProperties
    if getattr(FontProperties, "_min_pt_patched", False):
        FontProperties._min_pt = min_pt
        return
    original = FontProperties.set_size

    def clamped(self, size):
        try:
            if size is not None and not isinstance(size, str):
                size = max(float(size), FontProperties._min_pt)
        except (TypeError, ValueError):
            pass
        return original(self, size)

    FontProperties.set_size = clamped
    FontProperties.set_fontsize = clamped
    FontProperties._min_pt = min_pt
    FontProperties._min_pt_patched = True


HELD = None


def hold():
    global HELD
    HELD = []
    return HELD


def held(fig):
    if HELD is None:
        return False
    HELD.append(fig)
    return True


def glyph_xheight():
    import matplotlib._mathtext as mtx

    def xheight(self, fontname, fontsize, dpi):
        return self.get_metrics(fontname, mpl.rcParams["mathtext.default"], "x", fontsize, dpi).iceberg

    mtx.TruetypeFonts.get_xheight = xheight


def require_font():
    from matplotlib import font_manager as fm
    for name in FONT_STACK:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    raise RuntimeError("Figures require Arial or Helvetica. Install one of them, then clear the Matplotlib "
                       "font cache (" + mpl.get_cachedir() + ") and rerun.")


def apply_rc(min_pt=MIN_PT):
    enforce_min_pt(min_pt)
    require_font()
    glyph_xheight()
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300,
        "font.family": "sans-serif", "font.sans-serif": FONT_STACK,
        "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 5,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 8,
        "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.bbox": None,
        "mathtext.fontset": "custom", "mathtext.rm": "sans",
        "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
        "mathtext.sf": "sans", "mathtext.cal": "sans:italic",
        "mathtext.default": "it",
    })


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def bivariate_colors(n=3, c_x=BLUE, c_y=ORANGE, base="#EDEBE7"):
    bx, by, b0 = np.array(to_rgb(c_x)), np.array(to_rgb(c_y)), np.array(to_rgb(base))
    corner = 0.55 * (bx + by) / 2.0
    grid = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            fx, fy = j / (n - 1), i / (n - 1)
            col = (b0 * (1 - fx) * (1 - fy) + bx * fx * (1 - fy)
                   + by * fy * (1 - fx) + corner * fx * fy)
            grid[i, j] = np.clip(col, 0, 1)
    return grid


def bivariate_class(x, y, n=3):
    def cls(v):
        v = np.asarray(v, float)
        qs = np.nanquantile(v, np.linspace(0, 1, n + 1)[1:-1])
        return np.clip(np.searchsorted(qs, v, side="right"), 0, n - 1)
    return cls(x), cls(y)


def trivariate_rgb(a, b, c, gamma=0.75):
    m = np.column_stack([np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)])
    s = m.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    m = m / s
    return np.clip(m ** gamma, 0, 1)


def add_bivariate_key(fig, rect, grid, xlabel, ylabel):
    ax = fig.add_axes(rect)
    ax.imshow(grid, origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=6.5, labelpad=2, color=INK)
    ax.set_ylabel(ylabel, fontsize=6.5, labelpad=2, color=INK)
    for s in ax.spines.values():
        s.set_color(FAINT); s.set_linewidth(0.5)
    return ax
