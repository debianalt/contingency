"""
MCA + Ward typology robustness.

Refits the MCA-HAC pipeline used by the baseline typology under alternative
specifications and compares cluster membership to the baseline (mca_cluster_label
column in the master dataset). Cramer's V quantifies the agreement.

Variants:
  A. Discretisation: quartiles (vs baseline terciles)
  B. Active variables: drop rad_2014
  C. Active variables: drop ln(pob_2010)
  D. Number of retained MCA axes: k = 4, 5 (baseline), 6, 7
  E. Cluster count: 5, 6 (baseline), 7

Output:
  - tables/table_S6_typology_robustness.csv
  - appends summary to tables/revision_numbers.txt
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import prince
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import contingency

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJ = Path(__file__).resolve().parents[1]
TS_DATA = PROJ / "data" / "departamentos_master.csv"
TABLES = PROJ / "tables"
OUT_TXT = TABLES / "revision_numbers.txt"

ACTIVE_BASELINE = [
    "pct_jefe_sec_2010",
    "pct_jefe_uni_2010",
    "pct_pc_2010",
    "rad_2014",
    "tasa_empleo_2010",
    "pct_nbi_2010",
    "pct_hacinam_2010",
    "ln_pob_2010",
]


def cramers_v(x, y):
    tab = pd.crosstab(x, y)
    chi2 = contingency.chi2_contingency(tab.values, correction=False)[0]
    n = tab.values.sum()
    r, k = tab.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


def discretise(s, n_bins=3):
    labels = list(range(n_bins))
    try:
        return pd.qcut(s, q=n_bins, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.cut(s, bins=n_bins, labels=labels).astype(str)


def fit_typology(df, active_vars, n_bins=3, n_components=5, n_clusters=6):
    d = df.copy()
    discrete = pd.DataFrame(index=d.index)
    for v in active_vars:
        discrete[v] = discretise(d[v], n_bins=n_bins)
    mca = prince.MCA(n_components=n_components, random_state=0)
    mca = mca.fit(discrete)
    coords = mca.row_coordinates(discrete)
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = cluster.fit_predict(coords.values)
    return pd.Series(labels, index=d.index)


def main():
    df = pd.read_csv(TS_DATA, dtype={"dpto5": str})
    df["ln_pob_2010"] = np.log(df["pob_2010"].clip(lower=1))
    df = df.dropna(subset=ACTIVE_BASELINE + ["mca_cluster_label"]).reset_index(drop=True)
    print(f"Sample: N = {len(df)}")
    baseline = df["mca_cluster_label"].astype(str)

    variants = []

    # A. Discretisation
    for n_bins in [2, 3, 4]:
        labels = fit_typology(df, ACTIVE_BASELINE, n_bins=n_bins, n_components=5, n_clusters=6)
        v = cramers_v(baseline, labels)
        tag = "baseline (terciles)" if n_bins == 3 else f"bins={n_bins}"
        variants.append({"variant": f"Discretisation {tag}", "cramers_v": v})

    # B. Drop variable
    for drop in ["rad_2014", "ln_pob_2010"]:
        active = [v for v in ACTIVE_BASELINE if v != drop]
        labels = fit_typology(df, active, n_bins=3, n_components=5, n_clusters=6)
        v = cramers_v(baseline, labels)
        variants.append({"variant": f"Drop {drop}", "cramers_v": v})

    # C. Number of retained MCA axes
    for k in [4, 5, 6, 7]:
        labels = fit_typology(df, ACTIVE_BASELINE, n_bins=3, n_components=k, n_clusters=6)
        v = cramers_v(baseline, labels)
        tag = " (baseline)" if k == 5 else ""
        variants.append({"variant": f"MCA axes k={k}{tag}", "cramers_v": v})

    # D. Number of clusters
    for nc in [5, 6, 7]:
        labels = fit_typology(df, ACTIVE_BASELINE, n_bins=3, n_components=5, n_clusters=nc)
        v = cramers_v(baseline, labels)
        tag = " (baseline)" if nc == 6 else ""
        variants.append({"variant": f"Clusters n={nc}{tag}", "cramers_v": v})

    out_df = pd.DataFrame(variants)
    out_df.to_csv(TABLES / "table_S6_typology_robustness.csv", index=False)

    # Append to revision numbers
    with open(OUT_TXT, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("7. Typology robustness (MCA-HAC variants)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Sample for typology refit: N = {len(df)}\n")
        for r in variants:
            f.write(f"  {r['variant']:42s} Cramer's V = {r['cramers_v']:.3f}\n")
        v_values = [r["cramers_v"] for r in variants if "baseline" not in r["variant"]]
        if v_values:
            f.write(f"\nRange of Cramer's V across non-baseline variants: [{min(v_values):.3f}, {max(v_values):.3f}]\n")

    print(f"Wrote {TABLES / 'table_S6_typology_robustness.csv'}")


if __name__ == "__main__":
    main()
