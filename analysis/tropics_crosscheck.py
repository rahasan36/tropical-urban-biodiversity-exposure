"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json
from scipy import stats as sstats

from tropics_config import TABLE_DIR, THREAT_CATS
from tropics_cost import PAIRS_PATH
import tropics_analysis as ta
import tropics_scaling as ts

GROUPS = {"birds (BirdLife)": ["AVES"],
          "mammals (IUCN)": ["MAMMALIA"],
          "herpetofauna (IUCN)": ["AMPHIBIA", "REPTILIA"]}


def setting_by_group():
    z = np.load(PAIRS_PATH, allow_pickle=False)
    sp_index = json.loads(str(z["sp_index"]))
    sp_meta = json.loads(str(z["sp_meta"]))
    idx_to_sp = {v: k for k, v in sp_index.items()}
    n = len(sp_index)
    taxon = np.array([sp_meta.get(idx_to_sp[i], ["", "", ""])[2] for i in range(n)])
    cat = np.array([str(sp_meta.get(idx_to_sp[i], ["", "", ""])[1]).upper()
                    for i in range(n)])
    from tropics_species import read_bird_checklist
    from tropics_config import BOTW_CHECKLIST
    chk = read_bird_checklist(BOTW_CHECKLIST).set_index("sci_name")["category"]
    names = np.array([sp_meta.get(idx_to_sp[i], ["", "", ""])[0] for i in range(n)])
    fill = (cat == "") | (cat == "NAN") | (cat == "NONE")
    cat = np.where(fill, pd.Series(names).map(chk).fillna("").str.upper().to_numpy(), cat)
    is_threat = np.isin(cat, list(THREAT_CATS))

    s_sp, s_city = z["set_sp"], z["set_city"]
    out = {}
    for label, taxa in GROUPS.items():
        keep = np.isin(taxon[s_sp], taxa) & is_threat[s_sp]
        if not keep.any():
            continue
        s = pd.Series(s_city[keep]).value_counts()
        out[label] = s.rename(f"setting_{label.split()[0]}")
    return out


def run():
    df = ta.load_full()
    groups = setting_by_group()
    for label, s in groups.items():
        df = df.merge(s.rename_axis("city_id").reset_index(), on="city_id", how="left")
    cols = [c for c in df.columns if c.startswith("setting_")
            and c not in ("setting_richness", "setting_richness_all")]
    for c in cols:
        df[c] = df[c].fillna(0)

    rows = []
    keys = cols + ["setting_richness"]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            m = df[[a, b]].dropna()
            rho = sstats.spearmanr(m[a], m[b]).statistic
            rows.append(dict(a=a, b=b, spearman=float(rho), n=int(len(m))))
    agree = pd.DataFrame(rows)

    fits = []
    for c in keys:
        d = df.copy()
        d["setting_richness"] = d[c]
        r = ts.fit_scaling(d, response="cost_per_km2", spatial=True)
        fits.append(dict(covariate=c, n=r["n"], beta=r["beta"],
                         ci_lower=r["ci_lower"], ci_upper=r["ci_upper"],
                         gamma=r.get("gamma_setting", np.nan), R2=r["R2"],
                         verdict=r["verdict"]))
    fits = pd.DataFrame(fits)

    agree.to_csv(TABLE_DIR / "crosscheck_setting_agreement.csv", index=False)
    fits.to_csv(TABLE_DIR / "crosscheck_setting_fits.csv", index=False)
    df[["city_id"] + cols].to_csv(TABLE_DIR / "setting_by_taxon.csv", index=False)
    return agree, fits


if __name__ == "__main__":
    agree, fits = run()
    pd.set_option("display.width", 200)
    print("=== do independent taxonomic groups rank city settings the same way? ===")
    print(agree.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== does the size exponent depend on which group supplies the covariate? ===")
    print(fits.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
