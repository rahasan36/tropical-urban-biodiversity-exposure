"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import json, time, numpy as np, pandas as pd, pathlib
import tropics_counterfactual as cf, tropics_cfrobust as cr
import tropics_models as p4
from tropics_config import TABLE_DIR

t0 = time.time()
d = p4.table()
d["m_hat"] = d.cost_per_km2
ah = cf.implied_alpha(d)
A = cr.ALPHAS
out = {"alpha_hat": float(ah), "alphas": [float(a) for a in A]}

base = cr.curve_generic(d, A)
pct, ref = cr.rebase(A, base, ah)
obs = float((d.m_hat * d.urban_added_km2).sum())
out["aggregate"] = dict(pct=pct.tolist(), ref=ref, observed=obs,
                        fitted_vs_observed_pct=100.0 * (ref / obs - 1.0))
print(f"alpha_hat {ah:.3f}   fitted total is {100*(ref/obs-1):+.2f}% of observed",
      f"  [{time.time()-t0:.0f}s]", flush=True)

lo, hi = cr.bootstrap(d, n_boot=300, alpha_hat=ah)
out["aggregate"]["ci_lo"] = lo.tolist(); out["aggregate"]["ci_hi"] = hi.tolist()
print(f"bootstrap done [{time.time()-t0:.0f}s]  band at alpha=2: "
      f"{lo[-1]:+.2f} to {hi[-1]:+.2f}%", flush=True)

for mult, key in ((2.0, "cap2"), (5.0, "cap5")):
    c = cr.curve_generic(d, A, cap_mult=mult)
    out[key] = dict(pct=cr.rebase(A, c, ah)[0].tolist(), cap_multiple=mult)
    print(f"cap {mult}x done [{time.time()-t0:.0f}s]", flush=True)

for grp, key in (("realm", "by_realm_pool"), ("_all", "pantropical_pool")):
    z = d.assign(_all="tropics") if grp == "_all" else d
    c = cr.curve_generic(z, A, group=grp)
    out[key] = dict(pct=cr.rebase(A, c, ah)[0].tolist())
    print(f"pool {grp} done [{time.time()-t0:.0f}s]", flush=True)

pathlib.Path(TABLE_DIR/"cf_robust.json").write_text(json.dumps(out, indent=1))
print("written", flush=True)
