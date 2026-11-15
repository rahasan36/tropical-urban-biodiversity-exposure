"""The geography of threatened species exposure to tropical urban growth

Author: Rakibul Ahasan
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
NAME = "EDFig8_country_recency"
DPI = 400
CODE_DIRS = ["common", "measurement", "sensitivity", "temporal", "models", "reallocation",
             "supplementary_tables", "figures/plotting"]


def out_name(folder):
    name, k = NAME, 1
    while os.path.exists(os.path.join(folder, f"{name}.png")) or os.path.exists(os.path.join(folder, f"{name}.pdf")):
        name = f"{NAME}_r{k}"
        k += 1
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--data", default=os.path.join(REPO, "data", "figures"))
    ap.add_argument("--pipeline", action="store_true")
    a = ap.parse_args()
    if not a.pipeline:
        d = os.path.abspath(a.data)
        os.environ["TROPICS_INTERIM"] = os.path.join(d, "interim")
        os.environ["TROPICS_TABLES"] = os.path.join(d, "tables")
        os.environ["TROPICS_RAW"] = os.path.join(d, "raw")
        os.environ["TROPICS_NE"] = os.path.join(d, "naturalearth", "ne_10m_admin_0_countries.shp")
        os.environ["TROPICS_GISD30"] = os.path.join(d, "gisd30")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.environ["TROPICS_FIG_DIR"] = out
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path[:0] = [os.path.join(REPO, *d.split("/")) for d in CODE_DIRS]
    import tropics_style as st
    import fig_country_recency as fcr
    held = st.hold()
    fcr.figure(NAME)
    fig = held[-1]
    name = out_name(out)
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=DPI)
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    print(os.path.join(out, f"{name}.png"))


if __name__ == "__main__":
    main()
