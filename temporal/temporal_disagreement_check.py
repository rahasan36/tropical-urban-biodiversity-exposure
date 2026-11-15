"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
from tropics_config import TABLE_DIR, INTERIM_DIR

OUT = TABLE_DIR / "temporal" / "disagreement_check.csv"
EPOCHS = [2005, 2010, 2015, 2020]


def counts(sp, V, R):
    cum = np.cumsum(V, axis=1)
    n = int(sp.max()) + 1
    tot = np.zeros((n, 4))
    np.add.at(tot, sp, cum)
    seen = np.zeros(n, bool)
    seen[sp] = True
    share = np.where(R[:n, None] > 0, 100 * tot / np.maximum(R[:n, None], 1e-12), 0.0)
    out = {}
    for k, y in enumerate(EPOCHS):
        out[f"reached_{y}"] = int(((tot[:, k] > 0) & seen).sum())
        out[f"above_1pct_{y}"] = int(((share[:, k] >= 1) & seen).sum())
    return out


def factors(d, A):
    D = d * A.sum(axis=1)
    r1 = np.minimum(D, A[:, 0])
    f1 = np.where(A[:, 0] > 0, 1 - r1 / np.maximum(A[:, 0], 1e-12), 1.0)
    later = A[:, 1:].sum(axis=1)
    fl = np.where(later > 0, 1 - (D - r1) / np.maximum(later, 1e-12), 1.0)
    return np.column_stack([f1, fl, fl, fl]).clip(0, 1)


def variants(sp, city, V, R, d, A):
    rows = [dict(variant="As measured", agglomerations=int(np.unique(city).size), **counts(sp, V, R))]
    rows.append(dict(variant="Disputed share removed from every period", agglomerations=int(np.unique(city).size),
                     **counts(sp, V * (1 - d[city])[:, None], R)))
    rows.append(dict(variant="Disputed surface removed from 2000 to 2005 first", agglomerations=int(np.unique(city).size),
                     **counts(sp, V * factors(d, A)[city], R)))
    for cut in (0.5, 0.25):
        keep = d[city] <= cut
        rows.append(dict(variant=f"Agglomerations with disputed share above {cut:.2f} excluded",
                         agglomerations=int(np.unique(city[keep]).size), **counts(sp[keep], V[keep], R)))
    return pd.DataFrame(rows)


def selftest():
    sp = np.array([0, 0, 1])
    city = np.array([0, 1, 1])
    V = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 0.5]])
    R = np.array([100.0, 100.0])
    d = np.array([0.0, 0.5])
    A = np.array([[4.0, 0, 0, 0], [0.0, 2.0, 0, 2.0]])
    t = variants(sp, city, V, R, d, A).set_index("variant")
    assert t.loc["As measured", ["reached_2005", "reached_2020", "above_1pct_2005", "above_1pct_2010"]].tolist() == [1, 2, 1, 1]
    assert t.loc["As measured", "above_1pct_2020"] == 1
    p = counts(sp, V * (1 - d[city])[:, None], R)
    assert p["above_1pct_2010"] == 1 and p["reached_2020"] == 2
    f = factors(d, A)
    assert np.allclose(f[0], 1) and np.allclose(f[1], [1, 0.5, 0.5, 0.5])
    assert t.loc["Agglomerations with disputed share above 0.25 excluded", "reached_2020"] == 1
    return True


def main():
    import tropics_temporal as tt
    import fig_threat_taxa as ft
    selftest()
    a, b, V = tt.load_pairs()
    C, _, R = ft.species()
    th = np.isin(C[a], ("CR", "EN", "VU"))
    a, b, V = a[th], b[th], V[th, :4]
    h = pd.read_csv(INTERIM_DIR / "city_habitat.csv").set_index("city_id").hab_impervious_share
    p = pd.read_csv(TABLE_DIR / "temporal" / "temporal_city_periods.csv.gz")
    A = p.pivot_table(index="city_id", columns="j", values="added_km2", aggfunc="sum").fillna(0.0)
    n = int(max(b.max(), A.index.max(), h.index.max())) + 1
    d = np.zeros(n)
    d[h.index.to_numpy()] = h.fillna(0.0).to_numpy()
    Am = np.zeros((n, 4))
    Am[A.index.to_numpy()] = A.to_numpy()[:, :4]
    t = variants(a, b, V, R, d, Am)
    t.to_csv(OUT, index=False)
    w = Am.sum(axis=1)
    print("area-weighted disputed share", round(100 * (d * w).sum() / w.sum(), 1), "| missing share", int(h.isna().sum()))
    print(t.to_string())


if __name__ == "__main__":
    main()
