"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
from __future__ import annotations
import numpy as np
from tropics_config import NATIVE_DEG, DELIN_DEG, TROPIC_LAT, TILE_DEG

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_E = np.sqrt(WGS84_E2)


def _authalic_zone_term(lat_deg):
    phi = np.radians(np.asarray(lat_deg, dtype=np.float64))
    s = np.sin(phi)
    e = WGS84_E
    return (1.0 - WGS84_E2) * (
        s / (1.0 - WGS84_E2 * s * s)
        + (1.0 / (2.0 * e)) * np.log((1.0 + e * s) / (1.0 - e * s))
    )


def row_area_km2(lat_top_deg, lat_bottom_deg, lon_span_deg):
    dlam = np.radians(lon_span_deg)
    q_top = _authalic_zone_term(lat_top_deg)
    q_bot = _authalic_zone_term(lat_bottom_deg)
    a2 = WGS84_A * WGS84_A
    return np.abs(a2 * dlam * 0.5 * (q_top - q_bot)) / 1.0e6


def cell_area_km2(lat_top_deg, n_rows, cell_deg):
    tops = lat_top_deg - cell_deg * np.arange(n_rows, dtype=np.float64)
    bots = tops - cell_deg
    return row_area_km2(tops, bots, cell_deg)


def earth_area_km2():
    return float(row_area_km2(90.0, -90.0, 360.0))


def tropics_row_bounds(lat_top_deg, n_rows, cell_deg, tropic_lat=TROPIC_LAT):
    tops = lat_top_deg - cell_deg * np.arange(n_rows + 1, dtype=np.float64)
    inside = (tops[:-1] > -tropic_lat) & (tops[1:] < tropic_lat)
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return 0, 0
    return int(idx[0]), int(idx[-1]) + 1


def tile_name(lon_west, lat_north):
    ew = "E" if lon_west >= 0 else "W"
    ns = "N" if lat_north >= 0 else "S"
    return f"{ew}{abs(int(lon_west))}{ns}{abs(int(lat_north))}"


def tropical_tiles(tile_deg=TILE_DEG, tropic_lat=TROPIC_LAT):
    out = []
    lat = tile_deg * np.ceil(tropic_lat / tile_deg)
    while lat - tile_deg < tropic_lat and lat > -tropic_lat:
        lon = -180.0
        while lon < 180.0:
            out.append((tile_name(lon, lat), float(lon), float(lat)))
            lon += tile_deg
        lat -= tile_deg
    return out


if __name__ == "__main__":
    a = earth_area_km2()
    assert abs(a - 510_065_600.0) / 510_065_600.0 < 1e-4, a
    print(f"ellipsoid surface area  {a:,.0f} km2   (expected ~510,065,600)")

    eq = float(row_area_km2(0.0 + NATIVE_DEG, 0.0, NATIVE_DEG))
    side_m = np.sqrt(eq) * 1000.0
    assert 29.5 < side_m < 30.3, side_m
    print(f"native cell at equator   {eq*1e6:,.0f} m2  ({side_m:.1f} m on a side)")

    hi = float(row_area_km2(23.4394 + NATIVE_DEG, 23.4394, NATIVE_DEG))
    ratio = hi / eq
    assert abs(ratio - np.cos(np.radians(23.4394))) < 0.002, ratio
    print(f"area ratio at 23.44 deg  {ratio:.4f}   (cos lat = "
          f"{np.cos(np.radians(23.4394)):.4f})")

    n = 2000
    top = 23.4394
    step = (2 * top) / n
    rows = cell_area_km2(top, n, step) * (360.0 / step)
    direct = float(row_area_km2(top, -top, 360.0))
    assert abs(rows.sum() - direct) / direct < 1e-9
    print(f"tropical band area       {direct:,.0f} km2  (row sum agrees to 1e-9)")

    tiles = tropical_tiles()
    lats = sorted({t[2] for t in tiles})
    assert lats == [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0], lats
    assert len(tiles) == len(lats) * 72, len(tiles)
    print(f"tropical tile grid       {len(tiles)} tiles, lat rows {lats}")
    print("ALL GRID TESTS PASS")
