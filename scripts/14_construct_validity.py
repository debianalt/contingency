"""
Construct-validity check for the Software Portfolio Sophistication Index (SPSI).

Responds to the reviewer request for project-level validation of the indicator.
Correlates the SPSI (eci_software) against repository-level signals of
sophistication that are NOT inputs to its eigenvalue-decomposition construction:
stars, codebase size, licensing, and the composition of development activity
(enterprise/systems vs commodity web). A high SPSI is expected to align with
more-starred, larger, licensed, enterprise/systems-oriented portfolios and to
be negatively associated with commodity web development.

Output: tables/table_S11_construct_validity.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data" / "departamentos_master.csv"
TABLES = PROJ / "tables"

LOG_VARS = {"gh_total_stars", "gh_total_size_mb"}

PROJECT_METRICS = [
    ("gh_total_stars",        "Total stars (log)",            "+"),
    ("gh_total_size_mb",      "Total codebase size, MB (log)", "+"),
    ("gh_pct_with_license",   "Repositories with a licence (%)", "+"),
    ("gh_pct_enterprise",     "Enterprise-oriented development (%)", "+"),
    ("gh_pct_systems",        "Systems-oriented development (%)", "+"),
    ("gh_pct_web_development", "Commodity web development (%)", "-"),
]


def main():
    df = pd.read_csv(DATA, dtype={"dpto5": str})
    sub = df.dropna(subset=["eci_software"]).copy()
    n = len(sub)
    print(f"SPSI sample N = {n}")

    rows = []
    for var, label, expected in PROJECT_METRICS:
        if var not in sub.columns:
            print(f"  MISSING: {var}")
            continue
        d = sub[["eci_software", var]].dropna()
        x = d[var].astype(float).values
        if var in LOG_VARS:
            x = np.log1p(x)
        r, p = pearsonr(d["eci_software"], x)
        rho, _ = spearmanr(d["eci_software"], x)
        rows.append({
            "metric": label,
            "expected_sign": expected,
            "pearson_r": round(r, 3),
            "spearman_rho": round(rho, 3),
            "N": len(d),
            "p": p,
        })
        print(f"  {label:36s} r={r:+.3f} rho={rho:+.3f} (N={len(d)}, p={p:.1e})")

    out = TABLES / "table_S11_construct_validity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
