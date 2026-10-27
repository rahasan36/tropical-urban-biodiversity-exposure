"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import json
import numpy as np
from pathlib import Path
import tropics_temporal as t
from tropics_measure import MEASURE_PATH

z = np.load(MEASURE_PATH, allow_pickle=False)
cell_city = z["cell_city"].astype(np.int64)
a, b, V = t.load_pairs()
m = V[:, 4] > 0
a, b, v = a[m], b[m], V[m, 4]
per = V[:, :4].sum(1)
R = Path(t.INTERIM_DIR) / "restest" / "30_centroid"
rs, rc_, rv = [], [], []
for f in sorted(R.glob("slice_*.npz")):
    with np.load(f) as q:
        rs.append(q["sp_i"]); rc_.append(q["city_id"]); rv.append(q["conv_km2"])
rs, rc_, rv = map(np.concatenate, (rs, rc_, rv))
n = int(max(cell_city.max(), rc_.max())) + 1
k1 = a.astype(np.int64) * n + b
k2 = rs.astype(np.int64) * n + rc_
u = np.union1d(k1, k2)
x1 = np.zeros(u.size); x1[np.searchsorted(u, k1)] = v
x2 = np.zeros(u.size); x2[np.searchsorted(u, k2)] = rv
d = np.abs(x1 - x2)
out = dict(pairs_hits=int(k1.size), pairs_ref=int(k2.size), union=int(u.size),
           only_hits=int(np.setdiff1d(k1, k2).size), only_ref=int(np.setdiff1d(k2, k1).size),
           maxabs=float(d.max()), total_hits=float(v.sum()), total_ref=float(rv.sum()),
           species_hits=int(np.unique(a).size), species_ref=int(np.unique(rs).size),
           period_sum_vs_total=float(np.abs(per - V[:, 4]).max()))
print(json.dumps(out, indent=1))
(Path(t.OUT_DIR) / "hits_check.json").write_text(json.dumps(out, indent=1))
