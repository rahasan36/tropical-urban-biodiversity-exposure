"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyogrio
from pathlib import Path

from tropics_config import (IUCN_SOURCES, BOTW_GPKG, BOTW_LAYER, BOTW_CHECKLIST,
                            PRESENCE_KEEP, ORIGIN_KEEP, SEASONAL_KEEP,
                            THREAT_CATS, NARROW_RANGE_PCTL, TROPIC_LAT)

IUCN_ATTRS = ["id_no", "sci_name", "presence", "origin", "seasonal",
              "category", "class", "terrestial", "marine"]
BOTW_ATTRS = ["sisid", "sci_name", "presence", "origin", "seasonal"]


def _keep_mask(df):
    def as_int(s):
        return pd.to_numeric(s, errors="coerce")
    m = as_int(df["presence"]).isin(PRESENCE_KEEP)
    m &= as_int(df["origin"]).isin(ORIGIN_KEEP)
    m &= as_int(df["seasonal"]).isin(SEASONAL_KEEP)
    return m


def read_iucn_attributes(paths=None):
    paths = paths or IUCN_SOURCES
    frames = []
    for taxon, files in paths.items():
        for f in files:
            f = Path(f)
            if not f.exists():
                raise FileNotFoundError(f)
            cols = [c for c in IUCN_ATTRS
                    if c in pyogrio.read_info(f)["fields"]]
            df = pyogrio.read_dataframe(f, columns=cols, read_geometry=False)
            df = df[_keep_mask(df)]
            df["taxon"] = taxon
            df["source_file"] = f.name
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["sp_id"] = out["taxon"].str[:4] + "_" + out["id_no"].astype("int64").astype(str)
    agg = (out.groupby("sp_id")
              .agg(id_no=("id_no", "first"), sci_name=("sci_name", "first"),
                   taxon=("taxon", "first"), category=("category", "first"),
                   n_parts=("sci_name", "size"))
              .reset_index())
    return agg


def read_bird_attributes(gpkg=BOTW_GPKG, layer=BOTW_LAYER, checklist=BOTW_CHECKLIST):
    df = pyogrio.read_dataframe(gpkg, layer=layer, columns=BOTW_ATTRS,
                                read_geometry=False)
    df = df[_keep_mask(df)]
    df["sp_id"] = "AVES_" + df["sisid"].astype("int64").astype(str)
    agg = (df.groupby("sp_id")
             .agg(id_no=("sisid", "first"), sci_name=("sci_name", "first"),
                  n_parts=("sci_name", "size"))
             .reset_index())
    agg["taxon"] = "AVES"
    cat = read_bird_checklist(checklist)
    agg = agg.merge(cat, on="sci_name", how="left")
    return agg


def read_bird_checklist(path=BOTW_CHECKLIST):
    raw = pd.read_excel(path, header=None, nrows=12)
    hdr = None
    for i in range(len(raw)):
        vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        if any("scientific name" in v for v in vals) and \
           any("iucn red list" in v or v == "category" for v in vals):
            hdr = i
            break
    if hdr is None:
        raise ValueError(f"could not locate header row in {path}")
    df = pd.read_excel(path, header=hdr)
    cols = {str(c).strip().lower(): c for c in df.columns}
    name_col = next(c for k, c in cols.items() if "scientific name" in k)
    cat_col = next(c for k, c in cols.items()
                   if "iucn red list" in k or k == "category")
    out = df[[name_col, cat_col]].copy()
    out.columns = ["sci_name", "category"]
    out["category"] = out["category"].astype(str).str.strip().str.upper()
    return out.dropna(subset=["sci_name"]).drop_duplicates("sci_name")


def build_species_pool(iucn=None, birds=None, range_km2=None,
                       narrow_pctl=NARROW_RANGE_PCTL):
    frames = [f for f in (iucn, birds) if f is not None and len(f)]
    pool = pd.concat(frames, ignore_index=True)
    if range_km2 is not None:
        pool = pool.merge(range_km2, on="sp_id", how="left")
    pool["category"] = pool["category"].astype(str).str.strip().str.upper()
    pool["is_threat"] = pool["category"].isin(THREAT_CATS)
    if "range_km2" in pool.columns and pool["range_km2"].notna().any():
        thr = np.nanpercentile(pool.loc[pool.range_km2 > 0, "range_km2"], narrow_pctl)
        pool["narrow_threshold_km2"] = thr
        pool["is_narrow"] = pool["range_km2"] <= thr
    else:
        pool["is_narrow"] = False
    return pool


if __name__ == "__main__":
    df = pd.DataFrame(dict(
        presence=[1, 2, 3, 1, 1], origin=[1, 1, 1, 3, 2],
        seasonal=[1, 3, 1, 1, 5]))
    assert _keep_mask(df).tolist() == [True, True, False, False, False]
    print("distribution-code filter  ok")

    pool = build_species_pool(
        iucn=pd.DataFrame(dict(sp_id=[f"MAMM_{i}" for i in range(8)],
                               sci_name=list("abcdefgh"), taxon="MAMMALIA",
                               category=["CR", "EN", "VU", "NT", "LC",
                                         "LC", "DD", "LC"])),
        birds=None,
        range_km2=pd.DataFrame(dict(sp_id=[f"MAMM_{i}" for i in range(8)],
                                    range_km2=[10, 20, 30, 40, 50, 60, 70, 80.])))
    assert pool.is_threat.sum() == 3
    assert pool.is_narrow.sum() == 2, pool[["sp_id", "range_km2", "is_narrow"]]
    print("threat / narrow-range flags ok  "
          f"(threshold {pool.narrow_threshold_km2.iloc[0]:.1f} km2)")
    print("ALL SPECIES-MODULE UNIT TESTS PASS")
