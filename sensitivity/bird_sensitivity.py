"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np, pandas as pd, json
import tropics_refit as trf
from tropics_scaling import fit_scaling
from tropics_config import INTERIM_DIR, TABLE_DIR, THREAT_CATS

z = np.load(INTERIM_DIR / "species_city_pairs.npz", allow_pickle=False)
f = np.load(INTERIM_DIR / "sp_flags_merged.npz", allow_pickle=False)
cat, tax, area = f["cat"], f["tax"], f["area"]
n = len(cat)
thr_km2 = float(json.loads((TABLE_DIR / "cities_cost_meta.json").read_text())
                ["narrow_threshold_km2"])
ps, pc, pv = z["pair_sp"], z["pair_city"], z["pair_conv"]
ar = np.zeros(n); ar[:min(area.size, n)] = area[:min(area.size, n)]
w = np.where(ar[ps] > 0, pv / np.maximum(ar[ps], 1e-6), 0.0)

threat = np.isin(cat, list(THREAT_CATS))
narrow = (ar > 0) & (ar <= thr_km2)
concern = threat | narrow
is_bird = (tax == "AVES")

d = trf.table()
d = d[d.urban_km2_base > 0].copy()
rows = []
for lbl, keep in (("all four classes", np.ones(n, bool)),
                  ("birds excluded", ~is_bird)):
    x = d.copy()
    for nm, mask, val in (("conc", concern & keep, pv),
                          ("rar", concern & keep, w)):
        s = (pd.DataFrame(dict(city_id=pc[mask[ps]], v=val[mask[ps]]))
               .groupby("city_id").v.sum().rename(nm))
        x = x.join(s, on="city_id")
        x[nm] = x[nm].fillna(0.0) / x.urban_added_km2
    for col, resp in (("conc", "exposure intensity"),
                      ("rar", "rarity-weighted intensity")):
        y = x[x[col] > 0]
        r = fit_scaling(y, response=col, size="urban_km2_base",
                        setting="setting_footprint", spatial=True)
        rows.append(dict(pool=lbl, response=resp, n=r["n"], beta=r["beta"],
                         lo=r["ci_lower"], hi=r["ci_upper"],
                         gamma=r.get("gamma_setting", np.nan),
                         total_km2=float((x[col] * x.urban_added_km2).sum())))
o = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(o.to_string(index=False, formatters={
    "beta": "{:+.4f}".format, "lo": "{:+.4f}".format, "hi": "{:+.4f}".format,
    "gamma": "{:+.4f}".format, "total_km2": "{:,.0f}".format}))
o.to_csv(TABLE_DIR / "bird_sensitivity.csv", index=False)
print("\nspecies of concern: all %d   non-bird %d   birds %d"
      % (concern.sum(), (concern & ~is_bird).sum(), (concern & is_bird).sum()))
