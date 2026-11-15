"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import numpy as np
import pandas as pd
import tropics_flags as tfl
import species_concentration as sc
from tropics_cost import PAIRS_PATH
from tropics_config import TABLE_DIR

THREAT = ["CR", "EN", "VU"]
CLASSES = [("AMPHIBIA", "Amphibia"), ("REPTILIA", "Reptilia"), ("MAMMALIA", "Mammalia"), ("AVES", "Aves")]
RED_LIST_2023_1 = {"AMPHIBIA": 2876, "REPTILIA": 1848, "MAMMALIA": 1339}
BIRDLIFE_CHECKLIST_10 = {"AVES": 1256}
ASSESSED = {**RED_LIST_2023_1, **BIRDLIFE_CHECKLIST_10}


def build():
    cat, tax, _sci, _area = tfl.flags(PAIRS_PATH)
    mapped = pd.DataFrame(dict(taxon=tax, thr=pd.Series(cat).astype(str).str.upper().isin(THREAT)))
    pool, _ = sc.analysis_pool(sc.species_table(), sc.published_pool())
    rules = pd.read_csv(TABLE_DIR / "species_concentration_rules.csv")
    reached = rules[rules.is_threat].groupby(["taxon", "rule"]).size().unstack(fill_value=0)
    rows = []
    for key, lab in CLASSES + [(None, "Total")]:
        m = mapped if key is None else mapped[mapped.taxon == key]
        p = pool if key is None else pool[pool.taxon == key]
        r = reached.sum() if key is None else reached.loc[key]
        rows.append({"Class": lab,
                     "Species mapped": len(m),
                     "Threatened mapped": int(m.thr.sum()),
                     "Threatened assessed": sum(ASSESSED.values()) if key is None else ASSESSED[key],
                     "Species in pool": len(p),
                     "Threatened in pool": int(p.is_threat.sum()),
                     "Narrow range in pool": int(p.is_narrow.sum()),
                     "Threatened reached, within rule": int(r["within"]),
                     "Threatened reached, center rule": int(r["centroid"]),
                     "Threatened reached, touch rule": int(r["touches"]),
                     "Median global range in pool (km2)": float(np.median(p.range_km2))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = build()
    d.to_csv(TABLE_DIR / "species_accounting_by_class.csv", index=False)
    print(d.to_string())
