"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import os
from pathlib import Path

REPO        = Path(__file__).resolve().parents[1]
PROJECT     = Path(os.environ.get("TROPICS_PROJECT", REPO))
DOWNLOADS   = Path(os.environ.get("TROPICS_DOWNLOADS", REPO / "data" / "source"))
GISD30_SRC  = Path(os.environ.get("TROPICS_GISD30", REPO / "data" / "source" / "gisd30" / "tiles"))
SCRATCH     = Path(os.environ.get("TROPICS_SCRATCH", PROJECT / "scratch"))

RAW_DIR     = Path(os.environ.get("TROPICS_RAW", PROJECT / "data" / "raw"))
INTERIM_DIR = Path(os.environ.get("TROPICS_INTERIM", PROJECT / "data" / "interim"))
PROC_DIR    = PROJECT / "data" / "processed"
TABLE_DIR   = Path(os.environ.get("TROPICS_TABLES", PROJECT / "outputs" / "tables"))
SUPP_TABLE_DIR = PROJECT / "manuscript" / "tables"
NE_COUNTRIES = Path(os.environ.get("TROPICS_NE", DOWNLOADS / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp"))
FIG_DIR     = Path(os.environ.get("TROPICS_FIG_DIR", PROJECT / "outputs" / "figures"))
LOG_DIR     = PROJECT / "logs"

GISD30_ARCHIVE_DIR = RAW_DIR / "gisd30"
GISD30_TILE_DIR    = GISD30_SRC
GLC_TILE_DIR       = RAW_DIR / "glc"

IUCN_SOURCES = {
    "MAMMALIA":  [DOWNLOADS / "MAMMALS_TERRESTRIAL_ONLY" / "MAMMALS_TERRESTRIAL_ONLY.shp"],
    "AMPHIBIA":  [DOWNLOADS / "AMPHIBIANS" / "AMPHIBIANS_PART1.shp",
                  DOWNLOADS / "AMPHIBIANS" / "AMPHIBIANS_PART2.shp"],
    "REPTILIA":  [DOWNLOADS / "REPTILES" / "REPTILES_PART1.shp",
                  DOWNLOADS / "REPTILES" / "REPTILES_PART2.shp"],
}
BOTW_GPKG      = DOWNLOADS / "species" / "BOTW_2025.gpkg"
BOTW_LAYER     = "all_species"
BOTW_CHECKLIST = (DOWNLOADS / "species" /
                  "Handbook of the Birds of the World and BirdLife International "
                  "Digital Checklist of the Birds of the World_Version_10.xlsx")

TROPICS_MODE = "band"
TROPIC_LAT   = 23.4394

NATIVE_DEG      = 0.0002694945852358564
DELIN_DEG       = 1.0 / 120.0
TILE_DEG        = 5.0
TILE_PX         = 18554

MOSAIC_LAT_TOP    = 25.0
MOSAIC_LAT_BOTTOM = -25.0

GISD30_EPOCHS = {1: 1985, 2: 1990, 3: 1995, 4: 2000,
                 5: 2005, 6: 2010, 7: 2015, 8: 2020}
GISD30_NODATA = 255
BASE_EPOCH    = 2000
FINAL_EPOCH   = 2020

GLC_LEGEND = {
    10: "Rainfed cropland", 11: "Herbaceous cover cropland",
    12: "Tree or shrub cover (orchard) cropland", 20: "Irrigated cropland",
    51: "Open evergreen broadleaved forest", 52: "Closed evergreen broadleaved forest",
    61: "Open deciduous broadleaved forest", 62: "Closed deciduous broadleaved forest",
    71: "Open evergreen needleleaved forest", 72: "Closed evergreen needleleaved forest",
    81: "Open deciduous needleleaved forest", 82: "Closed deciduous needleleaved forest",
    91: "Open mixed leaf forest", 92: "Closed mixed leaf forest",
    120: "Shrubland", 121: "Evergreen shrubland", 122: "Deciduous shrubland",
    130: "Grassland", 140: "Lichens and mosses",
    150: "Sparse vegetation", 152: "Sparse shrubland", 153: "Sparse herbaceous",
    181: "Swamp", 182: "Marsh", 183: "Flooded flat", 184: "Saline",
    185: "Mangrove", 186: "Salt marsh", 187: "Tidal flat",
    190: "Impervious surfaces",
    200: "Bare areas", 201: "Consolidated bare areas", 202: "Unconsolidated bare areas",
    210: "Water body", 220: "Permanent ice and snow",
    0: "Filled value", 250: "Filled value",
}
GLC_FOREST    = [51, 52, 61, 62, 71, 72, 81, 82, 91, 92]
GLC_SHRUB     = [120, 121, 122]
GLC_GRASS     = [130]
GLC_SPARSE    = [140, 150, 152, 153]
GLC_WETLAND   = [181, 182, 183, 184, 185, 186, 187]
GLC_CROPLAND  = [10, 11, 12, 20]
GLC_IMPERVIOUS = [190]
GLC_BARE      = [200, 201, 202]
GLC_WATER     = [210, 220]
GLC_NATURAL   = GLC_FOREST + GLC_SHRUB + GLC_GRASS + GLC_SPARSE + GLC_WETLAND
GLC_ANNUAL_BAND_YEARS = list(range(2000, 2023))

PRESENCE_KEEP = {1, 2}
ORIGIN_KEEP   = {1, 2}
SEASONAL_KEEP = {1, 2, 3}
THREAT_CATS   = {"CR", "EN", "VU"}
NARROW_RANGE_PCTL = 25

EXCLUDED_TAXA = ()

URBAN_FRACTION = float(os.environ.get("TROPICS_FRACTION", 0.20))

LINK_CELLS   = int(os.environ.get("TROPICS_LINK", 2))
MIN_CITY_KM2 = float(os.environ.get("TROPICS_FLOOR", 5.0))
SETTING_BUFFER_KM = 50.0

TAG = os.environ.get("TROPICS_TAG", "")


def tagged(path):
    from pathlib import Path as _P
    p = _P(path)
    return p if not TAG else p.with_name(f"{p.stem}{TAG}{p.suffix}")


SEED = 20260829
