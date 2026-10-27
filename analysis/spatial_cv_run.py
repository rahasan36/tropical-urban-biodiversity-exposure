"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spatial_cv_blocks as cv2
import tropics_models as p4
from tropics_config import TABLE_DIR

OUT = TABLE_DIR / "_spatial_cv_buffered.csv"
DEGS = (5.0, 10.0, 20.0)
BUFFERS = (0.0, 100.0, 250.0)
SPECS = {
    "size only": "log_cost ~ log_size",
    "setting only": "log_cost ~ log_setting",
    "size + setting": "log_cost ~ log_size + log_setting",
    "size + setting + biome": "log_cost ~ log_size + log_setting + C(biome)",
    "+ country": "log_cost ~ log_size + log_setting + C(biome) + C(country)",
}


def run():
    d = p4.table()
    rows = []
    for deg in DEGS:
        for buf in BUFFERS:
            r = cv2.block_cv(d, SPECS, deg=deg, buffer_km=buf,
                             common_sample=True)
            r["block_deg"] = deg
            r["buffer_km"] = buf
            rows.append(r)
            print(f"  {deg:4.0f} deg  {buf:5.0f} km  "
                  f"scored {int(r.n_scored.iloc[0]):,}  "
                  + "  ".join(f"{m} {v:+.3f}" for m, v
                              in zip(r.model, r.R2_oos)), flush=True)
    f = pd.concat(rows, ignore_index=True)
    cols = ["model", "block_deg", "buffer_km", "n_blocks", "n_scored", "R2_oos",
            "train_rows_dropped_by_buffer"]
    f = f[[c for c in cols if c in f.columns]]
    f.to_csv(OUT, index=False)
    print(f"written to {OUT.name}   {len(f)} rows")
    return f


def _selftest():
    f = pd.read_csv(OUT)
    for (deg, buf), g in f.groupby(["block_deg", "buffer_km"]):
        assert g.n_scored.nunique() == 1, (deg, buf, g.n_scored.tolist())
        assert len(g) == len(SPECS), (deg, buf, len(g))
    for deg, g in f.groupby("block_deg"):
        by = g.groupby("buffer_km").train_rows_dropped_by_buffer.first()
        assert by.is_monotonic_increasing, (deg, by.tolist())
    print(f"ok  {len(f)} rows, one scored sample per cell, buffers monotone")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        run()
        _selftest()
