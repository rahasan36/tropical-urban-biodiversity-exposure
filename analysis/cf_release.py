"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import json, sys, time, pathlib
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tropics_numbers as tn
import cf_fix as C
from tropics_config import TABLE_DIR

OUT = TABLE_DIR / "cf_release.json"
A = C.ALPHAS


def sample():
    d = tn.table()
    d = d[d.urban_km2_base > 0].copy()
    return d


def curves(d, alpha_hat, use_ranef=True):
    meas = C.totals(d.assign(_p=d.cost_per_km2), "_p", alphas=A)
    x, s = C.predict_price(d, use_ranef=use_ranef)
    x = x[x.m_hat_model.notna()]
    pred = C.totals(x, "m_hat_model", alphas=A)
    return (C.rel_to_fitted(meas, A, alpha_hat),
            C.rel_to_fitted(pred, A, alpha_hat), s, len(x))


def one_replicate(d, alpha_hat, rng):
    ks = d.country.dropna().unique()
    pick = rng.choice(ks, size=len(ks), replace=True)
    r = pd.concat([d[d.country == k].assign(country=f"{k}__{j}")
                   for j, k in enumerate(pick)], ignore_index=True)
    m = C.rel_to_fitted(C.totals(r.assign(_p=r.cost_per_km2), "_p", alphas=A),
                        A, alpha_hat)
    try:
        x, _ = C.predict_price(r)
        x = x[x.m_hat_model.notna()]
        p = C.rel_to_fitted(C.totals(x, "m_hat_model", alphas=A), A, alpha_hat)
    except Exception:
        p = np.full(len(A), np.nan)
    return m, p


def run(n_boot=300, seed=11, budget_s=480):
    d = sample()
    import tropics_counterfactual as cf
    ah = float(cf.implied_alpha(d))
    st = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "point" not in st:
        m, p, s, n = curves(d, ah)
        st = dict(alpha_hat=ah, alphas=[float(a) for a in A], n=len(d),
                  smearing=s, n_priced=n,
                  point=dict(measured=m.tolist(), predicted=p.tolist()))
        fe_m, fe_p, _, _ = curves(d, ah, use_ranef=False)
        st["point"]["predicted_fe_only"] = fe_p.tolist()
        OUT.write_text(json.dumps(st, indent=1))
        print(f"point estimates  n={len(d)}  alpha_hat={ah:.4f}  smearing={s:.4f}")
        print(f"  measured   a=+2 {m[-1]:+.2f}%  a=-1 {m[0]:+.2f}%")
        print(f"  predicted  a=+2 {p[-1]:+.2f}%  a=-1 {p[0]:+.2f}%  "
              f"(fixed effects only {fe_p[-1]:+.2f} / {fe_p[0]:+.2f})", flush=True)
    reps_m = st.get("reps_m", []); reps_p = st.get("reps_p", [])
    rng = np.random.default_rng(seed + len(reps_m))
    t0 = time.time()
    while len(reps_m) < n_boot and time.time() - t0 < budget_s:
        m, p = one_replicate(d, ah, rng)
        reps_m.append(m.tolist()); reps_p.append(p.tolist())
        if len(reps_m) % 10 == 0:
            st["reps_m"], st["reps_p"] = reps_m, reps_p
            OUT.write_text(json.dumps(st))
            print(f"  {len(reps_m):3d}/{n_boot} replicates "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    st["reps_m"], st["reps_p"] = reps_m, reps_p
    OUT.write_text(json.dumps(st))
    print(f"stored {len(reps_m)} replicates", flush=True)


if __name__ == "__main__":
    run(n_boot=int(sys.argv[1]) if len(sys.argv) > 1 else 300,
        budget_s=float(sys.argv[2]) if len(sys.argv) > 2 else 480)
