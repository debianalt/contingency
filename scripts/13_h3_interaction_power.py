"""
Round-2 reinforcement analyses for Technology in Society resubmission:

  (a) Closes the H3 logical gap by adding interaction terms thick x STEM and
      thick x ln(wage) to the pooled specification (N = 221), so that H3 is
      tested with statistical power rather than relying on n = 49 in a single
      type. The dummy "thick" groups the three institutional-thickness types
      (Metropolitan-Diversified, Pampeana-Educated, Metropolitan-Core); "thin"
      groups Intermediate-Urban (plus the small types that are pooled here for
      the interaction even though they are dropped from the type-specific OLS).

  (b) Formal power analysis for the type-specific OLS regressions following
      Cohen (1988). For each estimable type, computes the minimum detectable
      f^2 at alpha = 0.05, power = 0.80, with df1 = 1 and df2 = n - k - 1, and
      converts it into a standardised-beta equivalent using R^2_full of the
      type-specific regression.

Inputs:
  - data/departamentos_master.csv

Outputs:
  - tables/table_06_h3_interaction.csv     (interaction-specification output)
  - tables/table_06b_power_analysis.csv    (MDE per type)
  - tables/h3_power_numbers.txt            (key numbers for the manuscript)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist, ncf

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data" / "departamentos_master.csv"
TABLES = PROJ / "tables"
TABLES.mkdir(exist_ok=True)
OUT_TXT = TABLES / "h3_power_numbers.txt"

KNOW = ["cyt_stem_per_10k", "dist_stem_uni_km", "pct_univ_adultos"]
WEALTH = ["log_wage_median", "rad_mean_2022", "pct_servicios_avanzados"]
CONTROLS = ["log_pob", "inet_penetracion_hog"]
ALL_PRED = KNOW + WEALTH + CONTROLS

# Type-specific reduced predictor set (matches the manuscript's Table 4)
TYPE_PRED = ["cyt_stem_per_10k", "log_wage_median", "log_pob", "inet_penetracion_hog"]

THICK_TYPES = ["Metro-Diversified", "Pampeana-Educated", "Metro-Core"]
# All other observable types in the SPSI subsample are pooled as "thin"

ALIAS = {
    "Metropolitan-Diversified": "Metro-Diversified",
    "Metropolitan Diversified": "Metro-Diversified",
    "Pampeana-Educated": "Pampeana-Educated",
    "Pampeana Educated": "Pampeana-Educated",
    "Metropolitan-Core": "Metro-Core",
    "Metropolitan Core": "Metro-Core",
    "Intermediate-Urban": "Intermediate-Urban",
    "Intermediate Urban": "Intermediate-Urban",
    "Peripheral-Deprived": "Peripheral-Deprived",
    "Peripheral Deprived": "Peripheral-Deprived",
    "Semi-Rural-Active": "Semi-Rural-Active",
    "Semi Rural Active": "Semi-Rural-Active",
}


def load_master():
    df = pd.read_csv(DATA, dtype={"dpto5": str})
    df["log_pob"] = np.log(df["pob_2022"].clip(lower=1))
    df["dept_type_norm"] = df["mca_cluster_label"].astype(str).map(lambda s: ALIAS.get(s.strip(), s.strip()))
    return df


def fit_ols(df, y_col, x_cols, cov="HC1"):
    sub = df.dropna(subset=[y_col] + x_cols).copy()
    X = sm.add_constant(sub[x_cols])
    model = sm.OLS(sub[y_col], X).fit(cov_type=cov)
    return model, sub


def standardise_betas(model, X):
    y_sd = model.model.endog.std(ddof=1)
    out = {}
    X_sd = X.std(ddof=1)
    for name, coef in model.params.items():
        if name == "const":
            continue
        if name in X_sd.index and X_sd[name] > 0:
            out[name] = coef * X_sd[name] / y_sd
        else:
            out[name] = np.nan
    return out


# ---------------------------------------------------------------------------
# (a) H3 interaction in the pooled sample
# ---------------------------------------------------------------------------

def h3_interaction(df, f):
    sub = df.dropna(subset=["eci_software"] + ALL_PRED + ["dept_type_norm"]).copy()
    sub["thick"] = sub["dept_type_norm"].isin(THICK_TYPES).astype(int)
    f.write(f"\nH3 interaction sample: N = {len(sub)}\n")
    f.write("Type distribution (thick=1 grouping):\n")
    for t, n in sub.groupby("dept_type_norm").size().sort_values(ascending=False).items():
        thk = int(t in THICK_TYPES)
        f.write(f"  {t:25s} N = {n:3d} | thick = {thk}\n")
    f.write(f"  Thick subsample N = {(sub['thick']==1).sum()}\n")
    f.write(f"  Thin subsample  N = {(sub['thick']==0).sum()}\n")

    # Centred predictors for cleaner interpretation of main effects in interaction model
    sub["stem_c"] = sub["cyt_stem_per_10k"] - sub["cyt_stem_per_10k"].mean()
    sub["lnw_c"] = sub["log_wage_median"] - sub["log_wage_median"].mean()
    sub["stem_x_thick"] = sub["stem_c"] * sub["thick"]
    sub["lnw_x_thick"] = sub["lnw_c"] * sub["thick"]

    # Specification: STEM (centred) + ln_wage (centred) + thick + STEMxThick + ln_wagexThick + controls
    int_cols = [
        "stem_c", "lnw_c", "thick", "stem_x_thick", "lnw_x_thick",
        "dist_stem_uni_km", "pct_univ_adultos",
        "rad_mean_2022", "pct_servicios_avanzados",
        "log_pob", "inet_penetracion_hog",
    ]
    X = sm.add_constant(sub[int_cols])
    m = sm.OLS(sub["eci_software"], X).fit(cov_type="HC1")
    betas = standardise_betas(m, sub[int_cols])

    f.write(f"\nInteraction OLS: N = {int(m.nobs)}, R2 = {m.rsquared:.4f}, Adj R2 = {m.rsquared_adj:.4f}\n")
    for v in int_cols:
        b = m.params[v]
        se = m.bse[v]
        p = m.pvalues[v]
        beta = betas.get(v, np.nan)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.10 else "")))
        f.write(f"  {v:25s} B = {b:+.4f} (SE = {se:.4f}) p = {p:.4f}{sig} | beta = {beta:+.3f}\n")

    # Implied conditional effects: skills association in thick vs thin
    stem_thin = m.params["stem_c"]
    stem_thick = m.params["stem_c"] + m.params["stem_x_thick"]
    lnw_thin = m.params["lnw_c"]
    lnw_thick = m.params["lnw_c"] + m.params["lnw_x_thick"]
    f.write("\nImplied conditional effects:\n")
    f.write(f"  STEM effect | thin  = {stem_thin:+.4f}\n")
    f.write(f"  STEM effect | thick = {stem_thick:+.4f}\n")
    f.write(f"  ln(wage) | thin     = {lnw_thin:+.4f}\n")
    f.write(f"  ln(wage) | thick    = {lnw_thick:+.4f}\n")

    # Joint Wald test of the two interaction terms
    R = np.zeros((2, len(m.params)))
    idx = list(m.params.index)
    R[0, idx.index("stem_x_thick")] = 1.0
    R[1, idx.index("lnw_x_thick")] = 1.0
    wald = m.wald_test(R, scalar=True)
    f.write(f"\nJoint Wald test (both interactions = 0): F = {float(wald.statistic):.4f}, p = {float(wald.pvalue):.4g}\n")

    # Persist
    rows = []
    for v in int_cols:
        rows.append({
            "variable": v,
            "B": float(m.params[v]),
            "SE": float(m.bse[v]),
            "p": float(m.pvalues[v]),
            "beta": float(betas.get(v, np.nan)),
        })
    rows.append({"variable": "STEM | thin",   "B": float(stem_thin),  "SE": "", "p": "", "beta": ""})
    rows.append({"variable": "STEM | thick",  "B": float(stem_thick), "SE": "", "p": "", "beta": ""})
    rows.append({"variable": "ln(wage) | thin",  "B": float(lnw_thin),  "SE": "", "p": "", "beta": ""})
    rows.append({"variable": "ln(wage) | thick", "B": float(lnw_thick), "SE": "", "p": "", "beta": ""})
    rows.append({"variable": "N",  "B": int(m.nobs), "SE": "", "p": "", "beta": ""})
    rows.append({"variable": "R2", "B": float(m.rsquared), "SE": "", "p": "", "beta": ""})
    rows.append({"variable": "Joint Wald F", "B": float(wald.statistic), "SE": "", "p": float(wald.pvalue), "beta": ""})
    pd.DataFrame(rows).to_csv(TABLES / "table_06_h3_interaction.csv", index=False)


# ---------------------------------------------------------------------------
# (b) Formal power analysis per type
# ---------------------------------------------------------------------------

def mde_f_sq(n, k, alpha=0.05, power=0.80):
    """Minimum detectable Cohen's f^2 for an OLS predictor (df1 = 1).

    Returns the noncentrality parameter required to achieve the target power
    against a one-degree-of-freedom F-test, and the corresponding f^2 = lambda / N.
    Computed by numerical inversion of the noncentral F CDF.
    """
    df1 = 1
    df2 = n - k - 1
    if df2 <= 0:
        return np.nan, np.nan
    crit = f_dist.ppf(1 - alpha, df1, df2)

    # Bisection on noncentrality nc
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pwr = 1 - ncf.cdf(crit, df1, df2, mid)
        if pwr < power:
            lo = mid
        else:
            hi = mid
    nc_required = 0.5 * (lo + hi)
    f2_required = nc_required / n
    return nc_required, f2_required


def power_analysis(df, f):
    f.write("\n" + "=" * 70 + "\n")
    f.write("Power analysis per territorial type (Cohen 1988)\n")
    f.write("=" * 70 + "\n")
    f.write("Reduced specification (4 predictors): STEM per 10k, ln(wage), ln(pob), Internet penetration\n")
    f.write("alpha = 0.05, target power = 0.80\n\n")

    rows = []
    sub_all = df.dropna(subset=["eci_software"] + TYPE_PRED + ["dept_type_norm"]).copy()
    for t in [
        "Metro-Diversified",
        "Pampeana-Educated",
        "Metro-Core",
        "Intermediate-Urban",
        "Peripheral-Deprived",
        "Semi-Rural-Active",
    ]:
        sub = sub_all[sub_all["dept_type_norm"] == t].copy()
        n = len(sub)
        k = 4
        if n <= k + 1:
            f.write(f"{t:25s} n = {n:3d}  (too small for k = {k} predictors)\n")
            rows.append({"type": t, "n": n, "k": k, "R2_full": "", "f2_MDE": "", "beta_MDE": ""})
            continue

        # Fit reduced spec to recover R^2_full
        X = sm.add_constant(sub[TYPE_PRED])
        m = sm.OLS(sub["eci_software"], X).fit(cov_type="HC1")
        r2_full = m.rsquared
        nc, f2 = mde_f_sq(n, k, alpha=0.05, power=0.80)

        # Translate f^2 to standardised beta for the focal predictor (STEM):
        # f^2 = sr^2 / (1 - R^2_full)  ==>  sr^2 = f^2 (1 - R^2_full)
        # For a predictor entered alongside the others, the semi-partial r^2
        # in standardised units approximately equals beta^2 * (1 - R^2_other),
        # where R^2_other is the R^2 of regressing STEM on the remaining
        # predictors. We estimate R^2_other empirically.
        x_oth = [c for c in TYPE_PRED if c != "cyt_stem_per_10k"]
        Xo = sm.add_constant(sub[x_oth])
        m_oth = sm.OLS(sub["cyt_stem_per_10k"], Xo).fit()
        r2_other = m_oth.rsquared
        sr2_required = f2 * (1 - r2_full)
        if (1 - r2_other) > 0:
            beta_mde = float(np.sqrt(sr2_required / (1 - r2_other)))
        else:
            beta_mde = np.nan

        f.write(f"{t:25s} n = {n:3d}  R2_full = {r2_full:.3f}  R2_other = {r2_other:.3f}  f2_MDE = {f2:.3f}  beta_MDE = {beta_mde:.3f}\n")
        rows.append({
            "type": t, "n": n, "k": k,
            "R2_full": round(r2_full, 4),
            "R2_other": round(r2_other, 4),
            "f2_MDE": round(f2, 4),
            "beta_MDE": round(beta_mde, 3),
        })
    pd.DataFrame(rows).to_csv(TABLES / "table_06b_power_analysis.csv", index=False)


def main():
    df = load_master()
    print(f"Loaded master: {len(df)} rows")

    f = open(OUT_TXT, "w", encoding="utf-8")
    f.write("Round-2 reinforcement analyses for TS resubmission\n")
    f.write("=" * 70 + "\n")

    f.write("\n----- (a) H3 INTERACTION in the pooled sample -----\n")
    h3_interaction(df, f)

    f.write("\n----- (b) FORMAL POWER ANALYSIS per type -----\n")
    power_analysis(df, f)

    f.close()
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
