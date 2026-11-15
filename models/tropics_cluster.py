"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from tropics_config import TABLE_DIR, SEED
import tropics_analysis as ta

FEATURES = [
    ("log_cost", "marginal cost"),
    ("log_rarity", "rarity-weighted cost"),
    ("log_setting", "setting richness"),
    ("log_size", "city size"),
    ("log_growth", "growth 2000-2020"),
    ("narrow_cost_share", "share of cost from narrow-range species"),
    ("form_infill_share", "share of growth by infilling"),
    ("form_leapfrog_share", "share of growth by leapfrog"),
    ("abs_lat", "distance from the equator"),
]


def feature_frame(df=None):
    df = ta.load_full() if df is None else df
    d = df.copy()
    d["log_rarity"] = np.log10(d.rarity_per_km2.replace(0, np.nan))
    d["log_growth"] = np.log10(d.growth_ratio.clip(lower=1.0))
    d["abs_lat"] = d.lat.abs()
    cols = [c for c, _ in FEATURES if c in d.columns]
    d = d.dropna(subset=cols)
    return d, cols


def choose_k(X, ks=range(2, 9), seed=SEED, sample=4000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        lab = km.labels_
        rows.append(dict(k=k, inertia=float(km.inertia_),
                         silhouette=float(silhouette_score(X[idx], lab[idx])),
                         calinski=float(calinski_harabasz_score(X, lab))))
    return pd.DataFrame(rows)


def fit(df=None, k=4, seed=SEED):
    d, cols = feature_frame(df)
    X = StandardScaler().fit_transform(d[cols].to_numpy())
    pca = PCA(n_components=min(len(cols), 6), random_state=seed).fit(X)
    Z = pca.transform(X)
    diag = choose_k(Z)
    km = KMeans(n_clusters=k, n_init=25, random_state=seed).fit(Z)
    d = d.copy()
    d["cluster"] = km.labels_
    order = (d.groupby("cluster").cost_per_km2.median()
               .sort_values(ascending=False).index.tolist())
    remap = {old: new for new, old in enumerate(order)}
    d["cluster"] = d.cluster.map(remap)
    prof = (d.groupby("cluster")
              .agg(n=("city_id", "size"),
                   urban_km2=("urban_km2_final", "median"),
                   added_km2=("urban_added_km2", "sum"),
                   cost_per_km2=("cost_per_km2", "median"),
                   rarity_per_km2=("rarity_per_km2", "median"),
                   setting=("setting_richness", "median"),
                   growth_pct=("growth_pct", "median"),
                   infill=("form_infill_share", "median"),
                   leapfrog=("form_leapfrog_share", "median"),
                   narrow_share=("narrow_cost_share", "median"),
                   abs_lat=("abs_lat", "median"),
                   total_cost=("cost_km2", "sum")).reset_index())
    prof["cost_share_pct"] = prof.total_cost / prof.total_cost.sum() * 100
    prof["added_share_pct"] = prof.added_km2 / prof.added_km2.sum() * 100
    loadings = pd.DataFrame(pca.components_[:3].T, index=cols,
                            columns=["PC1", "PC2", "PC3"])
    loadings["explained"] = np.nan
    loadings.loc[cols[0], "explained"] = float(pca.explained_variance_ratio_[:3].sum())
    return d, prof, diag, loadings


NAMES = {
    0: "rich-setting, high-cost",
    1: "explosive growth, leapfrog-dominated",
    2: "drier setting, lower-cost",
    3: "endemic-dominated",
}
NAME_NOTES = {
    0: "the largest share of both tropical urban growth and tropical "
       "biodiversity cost; richest settings",
    1: "a small group that multiplied in area more than tenfold, growing "
       "mostly by leapfrog",
    2: "drier, further from the equator, lower cost per unit",
    3: "modest cost in range area but a rarity-weighted cost two orders of "
       "magnitude above the rest: the endemic-range group",
}


def run():
    d, prof, diag, loadings = fit()
    d[["city_id", "cluster"]].to_csv(TABLE_DIR / "city_clusters.csv", index=False)
    prof.to_csv(TABLE_DIR / "cluster_profiles.csv", index=False)
    diag.to_csv(TABLE_DIR / "cluster_diagnostics.csv", index=False)
    loadings.to_csv(TABLE_DIR / "cluster_pca_loadings.csv")
    return d, prof, diag, loadings


if __name__ == "__main__":
    d, prof, diag, loadings = run()
    pd.set_option("display.width", 220)
    print("=== how many groups the data support ===")
    print(diag.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("  Silhouette widths near 0.2 across the whole range mean the data are "
          "gradational, not sharply clustered: the typology is a descriptive "
          "device, not evidence of discrete city types.")
    print("\n=== cluster profiles (medians unless stated) ===")
    print(prof.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\n=== principal component loadings ===")
    print(loadings.round(3).to_string())
    print(f"\ncities clustered: {len(d):,}")
