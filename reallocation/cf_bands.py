"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import json, sys, time, numpy as np, pandas as pd, pathlib
import tropics_counterfactual as cf, tropics_cfrobust as cr
import tropics_numbers as tn
from tropics_config import TABLE_DIR

P = TABLE_DIR / "cf_robust.json"
out = json.loads(P.read_text())
A = cr.ALPHAS
ah = out["alpha_hat"]
x = tn.analysis_sample()
x["threat_per_km2"] = x.cost_threat_km2 / x.urban_added_km2
x["narrow_per_km2"] = x.cost_narrow_km2 / x.urban_added_km2
NB = 200
t0 = time.time()

which = sys.argv[1]
if which == "response":
    out.setdefault("response", {})
    for name, col in (("all species of concern", "cost_per_km2"), ("threatened", "threat_per_km2"),
                      ("narrow range", "narrow_per_km2"), ("rarity weighted", "rarity_per_km2")):
        if name in out["response"]:
            continue
        y = x.assign(m_hat=x[col].fillna(0.0))
        pct, ref = cr.rebase(A, cr.curve_generic(y, A), ah)
        lo, hi = cr.bootstrap(y, n_boot=NB, alpha_hat=ah)
        out["response"][name] = dict(pct=pct.tolist(), ci_lo=lo.tolist(),
                                     ci_hi=hi.tolist())
        print(f"{name:<18} [{time.time()-t0:.0f}s]  a=2 {pct[-1]:+.1f}% "
              f"[{lo[-1]:+.1f}, {hi[-1]:+.1f}]", flush=True)
        P.write_text(json.dumps(out, indent=1))
else:
    out.setdefault("realm", {})
    x["m_hat"] = x.cost_per_km2.fillna(0.0)
    for realm, sub in x.groupby("realm"):
        if len(sub) < 150 or str(realm) in out["realm"]:
            continue
        pct, ref = cr.rebase(A, cr.curve_generic(sub, A), ah)
        lo, hi = cr.bootstrap(sub, n_boot=NB, alpha_hat=ah)
        out["realm"][str(realm)] = dict(n=int(len(sub)), pct=pct.tolist(),
                                        ci_lo=lo.tolist(), ci_hi=hi.tolist())
        print(f"{realm:<14} n={len(sub):<6} [{time.time()-t0:.0f}s]  a=2 "
              f"{pct[-1]:+.1f}% [{lo[-1]:+.1f}, {hi[-1]:+.1f}]", flush=True)
        P.write_text(json.dumps(out, indent=1))
print("saved", flush=True)
