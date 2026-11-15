"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import tropics_scaling as ts
import tropics_sizevars as sv
from tropics_config import TABLE_DIR

OUT = TABLE_DIR / "residual_moran.csv"


def selftest():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, (1500, 2))
    i0, _, p0 = ts.morans_i(xy, rng.normal(size=1500), n_perm=199)
    i1, _, p1 = ts.morans_i(xy, xy[:, 0] / 1000 + 0.1 * rng.normal(size=1500), n_perm=199)
    assert abs(i0) < 0.05 and p0 > 0.05 and i1 > 0.5 and p1 < 0.05
    return True


def main():
    selftest()
    d = sv.table()
    rows = []
    for label, spatial in (("Country random intercept", False), ("Country random intercept and 2 degree blocks", True)):
        r = ts.fit_scaling(d, response="cost_per_km2", size="urban_km2_base", setting="setting_footprint", spatial=spatial)
        x = r["_d"]
        xy = ts.project_equal_area(x.lon.to_numpy(), x.lat.to_numpy())
        i, z, p = ts.morans_i(xy, r["_resid"])
        rows.append(dict(model=label, n=int(r["n"]), beta=r["beta"], gamma=r.get("gamma_setting"), residuals=r["resid_kind"],
                         neighbors=8, moran_I=i, z=z, p=p, permutations=999))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
