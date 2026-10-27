"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
NAME = "SFig07_surface_by_period"


def out_name(folder):
    name, k = NAME, 1
    while os.path.exists(os.path.join(folder, f"{name}.png")) or os.path.exists(os.path.join(folder, f"{name}.pdf")):
        name = f"{NAME}_r{k}"
        k += 1
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.environ["TROPICS_FIG_DIR"] = out
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, os.path.join(REPO, "analysis"))
    import fig_bivariate_decomp as fbd
    name = out_name(out)
    print(fbd.figure(name, mode="tall"))


if __name__ == "__main__":
    main()
