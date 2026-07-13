"""
Selection sensitivity for the pooled OLS estimate.

Three sensitivity layers — all on the existing N=221 sample (recomputing SPSI
under truly relaxed thresholds requires access to the raw bipartite matrix
held in the companion PostgreSQL DB, which is unavailable here):

1. Tighter repo thresholds: drop departments with gh_total_repos below
   {20, 30, 50}. Tests whether the pooled coefficient depends on the
   lowest-volume departments in the analysis subsample.
2. Influence trimming: drop the top 10% and bottom 10% by STEM density,
   and separately by gh_total_repos. Tests whether the coefficient is
   driven by extreme observations.
3. Cluster leave-one-out: re-estimate the pooled OLS dropping each
   territorial type in turn.

Outputs:
  - tables/table_S9_selection_sensitivity.csv
  - appends to tables/revision_numbers.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJ = Path(__file__).resolve().parents[1]
TS_DATA = PROJ / "data" / "departamentos_master.csv"
TABLES = PROJ / "tables"
OUT_TXT = TABLES / "revision_numbers.txt"

KNOW = ["cyt_stem_per_10k", "dist_stem_uni_km", "pct_univ_adultos"]
WEALTH = ["log_wage_median", "rad_mean_2022", "pct_servicios_avanzados"]
CONTROLS = ["log_pob", "inet_penetracion_hog"]
ALL_PRED = KNOW + WEALTH + CONTROLS


def load_pooled():
    df = pd.read_csv(TS_DATA, dtype={"dpto5": str})
    df["log_pob"] = np.log(df["pob_2022"].clip(lower=1))
    cols = ["dpto5", "departamento", "mca_cluster_label", "eci_software", "gh_total_repos"] + ALL_PRED
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].dropna(subset=["eci_software"] + ALL_PRED).copy()
    return sub


def fit_pooled(sub):
    X = sm.add_constant(sub[ALL_PRED])
    m = sm.OLS(sub["eci_software"], X).fit(cov_type="HC1")
    y_sd = sub["eci_software"].std(ddof=1)
    stem_sd = sub["cyt_stem_per_10k"].std(ddof=1)
    beta_stem = m.params["cyt_stem_per_10k"] * stem_sd / y_sd
    return {
        "n": len(sub),
        "B_stem": m.params["cyt_stem_per_10k"],
        "SE_stem": m.bse["cyt_stem_per_10k"],
        "p_stem": m.pvalues["cyt_stem_per_10k"],
        "beta_stem": beta_stem,
        "R2": m.rsquared,
    }


def write_section(f, title):
    f.write("\n" + "=" * 70 + "\n")
    f.write(title + "\n")
    f.write("=" * 70 + "\n")


def main():
    sub = load_pooled()
    print(f"Pooled sample: N={len(sub)}")

    out = open(OUT_TXT, "a", encoding="utf-8")
    write_section(out, "6. Selection sensitivity (Step 4 of econometric strategy)")

    rows = []

    # Baseline
    base = fit_pooled(sub)
    out.write(f"Baseline (full N={base['n']}): beta_STEM = {base['beta_stem']:.3f}, p = {base['p_stem']:.4f}\n")
    rows.append({"variant": "Baseline (N={})".format(base["n"]), **base})

    # 1. Tighter repo thresholds
    write_section(out, "6a. Tighter repository thresholds")
    for thresh in [20, 30, 50]:
        s = sub[sub["gh_total_repos"] >= thresh].copy()
        r = fit_pooled(s)
        out.write(f"gh_total_repos >= {thresh}: N = {r['n']}, beta_STEM = {r['beta_stem']:.3f}, p = {r['p_stem']:.4f}, R2 = {r['R2']:.3f}\n")
        rows.append({"variant": f"repos>={thresh}", **r})

    # 2. Influence trimming
    write_section(out, "6b. Influence trimming")
    # By STEM density: drop top 10% and bottom 10%
    lo_stem = sub["cyt_stem_per_10k"].quantile(0.10)
    hi_stem = sub["cyt_stem_per_10k"].quantile(0.90)
    s = sub[(sub["cyt_stem_per_10k"] >= lo_stem) & (sub["cyt_stem_per_10k"] <= hi_stem)].copy()
    r = fit_pooled(s)
    out.write(f"Trim STEM top+bottom 10%: N = {r['n']}, beta_STEM = {r['beta_stem']:.3f}, p = {r['p_stem']:.4f}\n")
    rows.append({"variant": "Trim STEM 10/90", **r})

    lo_r = sub["gh_total_repos"].quantile(0.10)
    hi_r = sub["gh_total_repos"].quantile(0.90)
    s = sub[(sub["gh_total_repos"] >= lo_r) & (sub["gh_total_repos"] <= hi_r)].copy()
    r = fit_pooled(s)
    out.write(f"Trim repos top+bottom 10%: N = {r['n']}, beta_STEM = {r['beta_stem']:.3f}, p = {r['p_stem']:.4f}\n")
    rows.append({"variant": "Trim repos 10/90", **r})

    # 3. Cluster leave-one-out
    write_section(out, "6c. Cluster leave-one-out")
    for t in sub["mca_cluster_label"].dropna().unique():
        s = sub[sub["mca_cluster_label"] != t].copy()
        r = fit_pooled(s)
        out.write(f"Drop {t}: N = {r['n']}, beta_STEM = {r['beta_stem']:.3f}, p = {r['p_stem']:.4f}\n")
        rows.append({"variant": f"Drop {t}", **r})

    # Summary range
    write_section(out, "6d. Summary")
    betas = [row["beta_stem"] for row in rows]
    ps = [row["p_stem"] for row in rows]
    out.write(f"beta_STEM range across {len(rows)} variants: [{min(betas):.3f}, {max(betas):.3f}]\n")
    out.write(f"p_STEM range: [{min(ps):.4g}, {max(ps):.4g}]\n")
    out.write(f"Variants with p < 0.001: {sum(1 for p in ps if p < 0.001)} / {len(ps)}\n")
    out.write(f"Variants with p < 0.05: {sum(1 for p in ps if p < 0.05)} / {len(ps)}\n")

    pd.DataFrame(rows).to_csv(TABLES / "table_S9_selection_sensitivity.csv", index=False)
    out.close()
    print(f"Appended to {OUT_TXT}")


if __name__ == "__main__":
    main()
