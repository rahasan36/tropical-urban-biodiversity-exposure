"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import tropics_panels as tp
import tropics_style as st
import validate_setting as vs
import spatial_cv_blocks as cv2
from tropics_config import TABLE_DIR

SPECS = {
    "size only": "log_cost ~ log_size",
    "setting only": "log_cost ~ log_setting",
    "size + setting": "log_cost ~ log_size + log_setting",
    "size + setting + biome": "log_cost ~ log_size + log_setting + C(biome)",
    "+ country": ("log_cost ~ log_size + log_setting + C(biome) "
                  "+ C(country)"),
}
DEGS = (5.0, 10.0, 20.0)
DEG_COLOR = {5.0: "#9A9A9A", 10.0: "#1357C8", 20.0: "#B5361F"}
PANEL_DEG = 10.0
BUFFERS = (0.0, 100.0, 250.0)
BUF_COLOR = {0.0: "#9A9A9A", 100.0: "#1357C8", 250.0: "#B5361F"}


def held_out_predictions(d, formula, deg=10.0, k=10, seed=3, buffer_km=0.0):
    d = d.copy()
    d["_blk"] = cv2.blocks(d.lon.to_numpy(), d.lat.to_numpy(), deg)
    u = np.unique(d._blk)
    rng = np.random.default_rng(seed)
    fold = dict(zip(u, rng.integers(0, k, len(u))))
    d["_fold"] = d._blk.map(fold)
    xyz = cv2._unit_xyz(d.lon.to_numpy(), d.lat.to_numpy())
    out = []
    for f in range(k):
        tr_i = np.flatnonzero((d._fold != f).to_numpy())
        te_i = np.flatnonzero((d._fold == f).to_numpy())
        if len(te_i) < 20 or len(tr_i) < 200:
            continue
        if buffer_km > 0:
            tr_i = tr_i[~cv2._near_any(xyz[tr_i], xyz[te_i], buffer_km)]
            if len(tr_i) < 200:
                continue
        tr, te = d.iloc[tr_i], d.iloc[te_i]
        m = smf.ols(formula, tr).fit()
        p = m.predict(te)
        out.append(pd.DataFrame(dict(obs=te.log_cost.to_numpy(),
                                     pred=np.asarray(p, float),
                                     fold=f)))
    return pd.concat(out, ignore_index=True).dropna()


def figure(name="figS13_spatial_cv_v1"):
    import tropics_models as p4
    d = p4.table()

    scores = pd.read_csv(TABLE_DIR / "_spatial_cv_buffered.csv")
    st.apply_rc()
    W = tp.PAGE_W
    fig = plt.figure(figsize=(W, 3.05))
    axa = fig.add_axes([0.185, 0.175, 0.355, 0.735])
    axb = fig.add_axes([0.665, 0.175, 0.310, 0.735])

    order = list(SPECS)
    yy = np.arange(len(order))[::-1]
    axa.axvline(0, color=st.MUTED, lw=0.8, ls=(0, (3, 2)))
    for buf in BUFFERS:
        s = scores[(scores.block_deg == PANEL_DEG)
                   & (scores.buffer_km == buf)].set_index("model")
        v = [float(s.loc[m, "R2_oos"]) for m in order]
        n = int(s.n_scored.iloc[0])
        lab = ("no buffer" if buf == 0 else f"{buf:.0f} km buffer")
        wide = buf == 0
        axa.plot(v, yy, "o-", ms=3.6, lw=3.0 if wide else 1.2,
                 color=BUF_COLOR[buf], mfc="white", mew=1.2,
                 alpha=0.55 if wide else 1.0, zorder=2 if wide else 3,
                 label=f"{lab} (n = {n:,})")
    axa.set_yticks(yy)
    axa.set_yticklabels(order, fontsize=6.2)
    axa.set_xlim(-0.06, 0.82)
    axa.set_xlabel("Out-of-sample $R^2$ against the training mean", fontsize=7.2)
    axa.tick_params(axis="y", length=0)
    axa.legend(fontsize=5.8, loc="upper right", frameon=False, handlelength=1.6,
               bbox_to_anchor=(1.03, 1.03))
    st.despine(axa, keep=("bottom",))
    tp.head(axa, "a", "Blocks held out, then buffered")

    pr = held_out_predictions(d, SPECS["setting only"], deg=PANEL_DEG,
                              buffer_km=max(BUFFERS))
    lo = float(min(pr.obs.quantile(0.002), pr.pred.quantile(0.002)))
    hi = float(max(pr.obs.quantile(0.998), pr.pred.quantile(0.998)))
    axb.hexbin(pr.pred, pr.obs, gridsize=46, bins="log", mincnt=1,
               extent=(lo, hi, lo, hi), cmap="Blues", linewidths=0)
    axb.plot([lo, hi], [lo, hi], color=st.MUTED, lw=0.9, zorder=3)
    r2 = 1 - float(((pr.obs - pr.pred) ** 2).sum()) / \
        float(((pr.obs - pr.obs.mean()) ** 2).sum())
    axb.text(0.035, 0.965,
             f"held-out $R^2$ = {r2:.3f}\n{max(BUFFERS):.0f} km buffer, "
             f"n = {len(pr):,}",
             transform=axb.transAxes, fontsize=5.9, color=st.MUTED,
             ha="left", va="top", linespacing=1.5)
    axb.set_xlim(lo, hi); axb.set_ylim(lo, hi)
    axb.set_xlabel("Predicted log$_{10}$ exposure intensity", fontsize=7.2)
    axb.set_ylabel("Measured log$_{10}$ exposure intensity", fontsize=7.2)
    st.despine(axb)
    tp.head(axb, "b", "Setting alone, on cities never seen")
    return tp.save(fig, name)


def _selftest():
    rng = np.random.default_rng(7)
    n = 2000
    d = pd.DataFrame(dict(lon=rng.uniform(-180, 180, n),
                          lat=rng.uniform(-25, 25, n),
                          log_size=rng.normal(0, 1, n),
                          log_setting=rng.normal(0, 1, n)))
    d["log_cost"] = rng.normal(0, 1, n)
    pr = held_out_predictions(d, "log_cost ~ log_setting", deg=10.0)
    r2 = 1 - float(((pr.obs - pr.pred) ** 2).sum()) / \
        float(((pr.obs - pr.obs.mean()) ** 2).sum())
    assert r2 < 0.02, r2
    assert len(pr) > 1500, len(pr)
    print(f"ok  a noise predictor scores {r2:+.4f} out of fold")


if __name__ == "__main__":
    _selftest()
    print(figure())
