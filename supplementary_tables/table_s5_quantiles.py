"""A third of the world's threatened land vertebrates exposed to tropical urbanization

Author: Rakibul Ahasan
"""
import csv

from tropics_config import SUPP_TABLE_DIR, TABLE_DIR

OUT = SUPP_TABLE_DIR
LABEL = {"0.1": "10th percentile", "0.25": "25th percentile", "0.5": "50th percentile",
         "0.75": "75th percentile", "0.9": "90th percentile"}


def signed(x, k):
    return ("+" if x >= 0 else "-") + f"{abs(x):.{k}f}"


def main():
    path = OUT / "tableS5.csv"
    rows = [r for r in csv.reader(open(path, encoding="utf-8")) if r and r[0] != "quantile"]
    for r in csv.DictReader(open(TABLE_DIR / "quantile_scaling.csv", encoding="utf-8")):
        b, lo, hi, g = (float(r[k]) for k in ("beta", "ci_lower", "ci_upper", "gamma"))
        rows.append(["quantile", LABEL[r["quantile"]], f"{int(r['n']):,}", signed(b, 4),
                     f"{signed(lo, 4)} to {signed(hi, 4)}", signed(g, 3), "–", "–",
                     "yes" if hi < 0 or lo > 0 else "no"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(path, len(rows) - 1, "rows")


if __name__ == "__main__":
    main()
