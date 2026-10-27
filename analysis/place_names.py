"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import sys
import numpy as np
import pandas as pd
import pyogrio
from tropics_config import DOWNLOADS, RAW_DIR

SRC = DOWNLOADS / "ne_10m_populated_places" / "ne_10m_populated_places.shp"
OUT = RAW_DIR / "tropical_places.csv"
MAX_LAT = 26.0


def build(src=SRC):
    g = pyogrio.read_dataframe(str(src))
    g = g[g.LATITUDE.abs() <= MAX_LAT]
    d = pd.DataFrame(dict(name=g.NAME, country=g.ADM0NAME, iso=g.ADM0_A3, pop=g.POP_MAX,
                          rank=np.where(g.SCALERANK == 0, 99, g.SCALERANK),
                          lon=g.geometry.x.round(6), lat=g.geometry.y.round(6)))
    d = d.sort_values("pop", ascending=False, kind="stable")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    return d


if __name__ == "__main__":
    print(f"{len(build(*sys.argv[1:2])):,} places written to {OUT}")
