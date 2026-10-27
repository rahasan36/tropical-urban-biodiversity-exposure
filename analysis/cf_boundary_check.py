"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cf_fix as CFIX
import cf_release as R
import tropics_counterfactual as cf
from tropics_config import TABLE_DIR

OUT = TABLE_DIR / "cf_boundary_check.json"
A = CFIX.ALPHAS


def one(d, alpha_hat, rng):
    ks = d.country.dropna().unique()
    pick = rng.choice(ks, size=len(ks), replace=True)
    ndist = int(len(set(pick)))
    r = pd.concat([d[d.country == k].assign(country=f"{k}__{j}")
                   for j, k in enumerate(pick)], ignore_index=True)
    m = CFIX.rel_to_fitted(CFIX.totals(r.assign(_p=r.cost_per_km2), "_p", alphas=A),
                        A, alpha_hat)
    try:
        import statsmodels.formula.api as smf
        x = r[r.cost_per_km2 > 0].copy()
        x["y"] = np.log10(x.cost_per_km2)
        x["logSet"] = np.log10(x.setting_footprint.clip(lower=0.5))
        x["_biome"] = x.biome.astype(str)
        x["_country"] = x.country.astype(str)
        f = smf.mixedlm("y ~ logSet + C(_biome)", x,
                        groups=x["_country"]).fit(reml=True, method="lbfgs")
        var = float(np.asarray(f.cov_re).ravel()[0])
        p_, _ = CFIX.predict_price(r)
        p_ = p_[p_.m_hat_model.notna()]
        p = CFIX.rel_to_fitted(CFIX.totals(p_, "m_hat_model", alphas=A), A, alpha_hat)
        return m, p, var, None, p, ndist
    except Exception as e:
        try:
            q_, _ = CFIX.predict_price(r, use_ranef=False)
            q_ = q_[q_.m_hat_model.notna()]
            q = CFIX.rel_to_fitted(CFIX.totals(q_, "m_hat_model", alphas=A),
                                   A, alpha_hat)
        except Exception:
            q = np.full(len(A), np.nan)
        return m, np.full(len(A), np.nan), np.nan, f"{type(e).__name__}: {e}"[:160], q, ndist


CHUNK = TABLE_DIR / "_cfbc_chunk{k}.npz"


def chunk(k, n):
    d = R.sample()
    ah = float(cf.implied_alpha(d))
    rng = np.random.default_rng(1000 * (k + 1) + 11)
    M, P, V, E, Q, ND = [], [], [], [], [], []
    t0 = time.time()
    for _ in range(n):
        m, p, v, e, q, nd = one(d, ah, rng)
        M.append(m); P.append(p); V.append(v); E.append(e); Q.append(q); ND.append(nd)
    f = CHUNK.name.format(k=k)
    np.savez(TABLE_DIR / f, M=np.asarray(M), P=np.asarray(P),
             V=np.asarray(V, float), Q=np.asarray(Q),
             ND=np.asarray(ND, float),
             E=np.asarray([e or "" for e in E], dtype=object), allow_pickle=True)
    print(f"chunk {k}: {n} replicates, "
          f"{int(np.isfinite(np.asarray(P)).all(axis=1).sum())} fitted "
          f"[{time.time()-t0:.0f}s] -> {f}", flush=True)


def load_chunks():
    M, P, V, E, Q, ND = [], [], [], [], [], []
    for k in range(20):
        f = TABLE_DIR / CHUNK.name.format(k=k)
        if not f.exists():
            continue
        with np.load(f, allow_pickle=True) as z:
            M.append(z["M"]); P.append(z["P"]); V.append(z["V"])
            Q.append(z["Q"]); ND.append(z["ND"])
            E += [str(e) if str(e) else None for e in z["E"]]
    if not M:
        raise SystemExit("no chunks on disk; run `chunk k n` first")
    return (np.vstack(M), np.vstack(P), np.concatenate(V),
            E, np.vstack(Q), np.concatenate(ND))


def main(pooled=False):
    argv = [] if pooled else sys.argv[1:]
    n_boot = int(argv[0]) if argv else 300
    budget = float(argv[1]) if len(argv) > 1 else 150.0
    t0 = time.time()
    if pooled:
        M, P, V, E, Q, ND = load_chunks()
    else:
        d = R.sample()
        ah = float(cf.implied_alpha(d))
        rng = np.random.default_rng(11)
        M, P, V, E, Q, ND = [], [], [], [], [], []
        while len(M) < n_boot and time.time() - t0 < budget:
            m, p, v, e, q, nd = one(d, ah, rng)
            M.append(m.tolist()); P.append(p.tolist()); V.append(v)
            E.append(e); Q.append(q.tolist()); ND.append(nd)
        M, P, V, Q, ND = (np.asarray(M), np.asarray(P), np.asarray(V, float),
                          np.asarray(Q), np.asarray(ND, float))
    ok = np.isfinite(P).all(axis=1)
    print(f"replicates {len(M)}   fitted curve formed {int(ok.sum())}   "
          f"failed {int((~ok).sum())}   [{time.time()-t0:.0f}s]\n")

    fails = collections.Counter(e.split(":")[0] for e in E if e)
    print("failure classes:", dict(fails) or "none")
    for e in [e for e in E if e][:4]:
        print("   ", e)

    good = V[np.isfinite(V)]
    if good.size:
        nzero = int((good <= 1e-10).sum())
        print(f"\ncountry variance among the {good.size} fits that succeeded: "
              f"{nzero} at the boundary (<=1e-10), median {np.median(good):.3e}")

    def band(arr, mask):
        lo, hi = np.percentile(arr[mask], [2.5, 97.5], axis=0)
        return lo, hi
    rows = []
    for name, arr in (("measured", M), ("predicted", P)):
        for lab, mask in (("succeeded only", ok), ("all replicates", np.ones(len(M), bool))):
            if name == "predicted" and lab == "all replicates":
                continue
            lo, hi = band(arr, mask)
            rows.append((name, lab, lo[0], hi[0], lo[-1], hi[-1]))
    paired = P - M
    for lab, mask in (("succeeded only", ok),):
        lo, hi = band(paired, mask)
        rows.append(("paired", lab, lo[0], hi[0], lo[-1], hi[-1]))
    filled = np.where(np.isfinite(P), P, Q)
    fok = np.isfinite(filled).all(axis=1)
    lo, hi = band(filled, fok)
    rows.append(("predicted", "with FE fallback", lo[0], hi[0], lo[-1], hi[-1]))
    lo, hi = band(filled - M, fok)
    rows.append(("paired", "with FE fallback", lo[0], hi[0], lo[-1], hi[-1]))
    print(f"\nfixed-effect fallback formed a curve for "
          f"{int(fok.sum() - ok.sum())} of the {int((~ok).sum())} singular replicates")
    print(f"\n{'quantity':<11}{'replicates':<17}{'dispersed CI':>24}{'concentrated CI':>24}")
    for n, l, a, b, c, e_ in rows:
        print(f"{n:<11}{l:<17}{a:>10.2f} to {b:<11.2f}{c:>10.2f} to {e_:<11.2f}")

    n_fb = int(fok.sum() - ok.sum())
    print(f"\ndistinct countries drawn: {ND[ok].mean():.1f} where the fit "
          f"succeeded, {ND[~ok].mean():.1f} where it did not"
          if (~ok).any() else "\nno failures to contrast")
    print(f"measured curve at the concentrated end: {M[ok, -1].mean():+.1f}% "
          f"where the fit succeeded, {M[~ok, -1].mean():+.1f}% where it did not"
          if (~ok).any() else "")

    OUT.write_text(json.dumps(dict(
        n=len(M), n_ok=int(ok.sum()), failures=dict(fails),
        failure_messages=[e for e in E if e][:20],
        country_var_boundary=int((good <= 1e-10).sum()) if good.size else None,
        country_var_median=float(np.median(good)) if good.size else None,
        n_fe_fallback=n_fb,
        n_countries_ok=float(ND[ok].mean()) if ok.any() else None,
        n_countries_fail=float(ND[~ok].mean()) if (~ok).any() else None,
        measured_conc_ok=float(M[ok, -1].mean()) if ok.any() else None,
        measured_conc_fail=float(M[~ok, -1].mean()) if (~ok).any() else None,
        measured_conc_ci_all=[float(v) for v in np.percentile(M[:, -1], [2.5, 97.5])],
        measured_conc_ci_ok=[float(v) for v in np.percentile(M[ok, -1], [2.5, 97.5])],
        alphas=[float(a) for a in A]), indent=1))
    print(f"\nwritten to {OUT.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chunk":
        chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        main(pooled=True)
    else:
        main()
