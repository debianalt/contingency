"""
Revision analyses for Technology in Society resubmission.

Produces:
  - Table 5: moderation (STEM x ln_wage) and non-linearity (STEM^2)
  - Bundle SPSI robustness: correlation with language-level SPSI + pooled OLS with bundle
  - Temporal panel: year-over-year rank correlations 2015-2025; 2020 vs 2025
  - Top-quintile persistence 2020 -> 2025

Inputs:
  - data/departamentos_master.csv (TS master dataset, has SPSI, STEM, wage, panel eci_YYYY)
  - 2026_1_JEG_sent/github-subir/data/table_s_bundle_robustness.csv (bundle SPSI from companion)

Outputs (to tables/):
  - table_05_moderation_nonlinearity.csv
  - table_S7_bundle_robustness.csv
  - table_S8_temporal_panel.csv
  - revision_numbers.txt (key numbers to paste into manuscript)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr

PROJ = Path(__file__).resolve().parents[1]
TS_DATA = PROJ / "data" / "departamentos_master.csv"
JEG_DATA = PROJ.parents[0] / "2026_1_JEG_sent" / "github-subir" / "data" / "table_s_bundle_robustness.csv"
TABLES = PROJ / "tables"
TABLES.mkdir(exist_ok=True)
OUT_TXT = TABLES / "revision_numbers.txt"

KNOW = ["cyt_stem_per_10k", "dist_stem_uni_km", "pct_univ_adultos"]
WEALTH = ["log_wage_median", "rad_mean_2022", "pct_servicios_avanzados"]
CONTROLS = ["log_pob", "inet_penetracion_hog"]
ALL_PRED = KNOW + WEALTH + CONTROLS


def load_ts():
    df = pd.read_csv(TS_DATA, dtype={"dpto5": str})
    df["log_pob"] = np.log(df["pob_2022"].clip(lower=1))
    keep = ["dpto5", "departamento", "provincia", "dept_type", "eci_software"] + ALL_PRED
    keep += [f"eci_{y}" for y in range(2014, 2026)]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    return df


def pooled_sample(df):
    cols = ["eci_software"] + ALL_PRED
    sub = df.dropna(subset=cols).copy()
    return sub


def standardise_betas(model, X):
    """Compute standardised coefficients from an OLS model."""
    y_sd = model.model.endog.std(ddof=1)
    X_sd = X.std(ddof=1)
    betas = {}
    for name, coef in model.params.items():
        if name == "const":
            betas[name] = np.nan
            continue
        if name in X_sd.index:
            betas[name] = coef * X_sd[name] / y_sd
        else:
            betas[name] = np.nan
    return betas


def fit_ols(df, y_col, x_cols):
    sub = df.dropna(subset=[y_col] + x_cols).copy()
    X = sm.add_constant(sub[x_cols])
    model = sm.OLS(sub[y_col], X).fit(cov_type="HC1")
    betas = standardise_betas(model, sub[x_cols])
    return model, betas, len(sub)


def write_section(f, title):
    f.write("\n" + "=" * 70 + "\n")
    f.write(title + "\n")
    f.write("=" * 70 + "\n")


def fmt_coef(model, betas, name):
    if name not in model.params.index:
        return "(absent)"
    b = model.params[name]
    se = model.bse[name]
    p = model.pvalues[name]
    beta = betas.get(name, np.nan)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.10 else "")))
    return f"B={b:.4f} (SE={se:.4f}) p={p:.4f}{sig} | beta={beta:.3f}"


def main():
    df = load_ts()
    sub = pooled_sample(df)
    print(f"Loaded TS master: {len(df)} rows | pooled sample N={len(sub)}")

    out = open(OUT_TXT, "w", encoding="utf-8")
    out.write("Revision analyses — key numbers for TS resubmission\n")
    out.write("=" * 70 + "\n")
    out.write(f"Pooled sample: N = {len(sub)}\n")

    # -------------------------------------------------------------
    # 1. Baseline pooled (sanity check)
    # -------------------------------------------------------------
    write_section(out, "1. Baseline pooled (sanity check)")
    m0, b0, n0 = fit_ols(sub, "eci_software", ALL_PRED)
    out.write(f"N = {n0}, R2 = {m0.rsquared:.4f}, Adj R2 = {m0.rsquared_adj:.4f}\n")
    for v in ALL_PRED:
        out.write(f"  {v:32s} {fmt_coef(m0, b0, v)}\n")

    # -------------------------------------------------------------
    # 2. Moderation: STEM x ln_wage
    # -------------------------------------------------------------
    write_section(out, "2. Moderation specification: STEM x ln(wage)")
    sub_m = sub.copy()
    sub_m["stem_x_lnwage"] = sub_m["cyt_stem_per_10k"] * sub_m["log_wage_median"]
    mod_cols = ALL_PRED + ["stem_x_lnwage"]
    m1, b1, n1 = fit_ols(sub_m, "eci_software", mod_cols)
    out.write(f"N = {n1}, R2 = {m1.rsquared:.4f}, Adj R2 = {m1.rsquared_adj:.4f}\n")
    for v in mod_cols:
        out.write(f"  {v:32s} {fmt_coef(m1, b1, v)}\n")

    # -------------------------------------------------------------
    # 3. Non-linearity: STEM^2
    # -------------------------------------------------------------
    write_section(out, "3. Non-linearity specification: STEM^2")
    sub_q = sub.copy()
    sub_q["stem_sq"] = sub_q["cyt_stem_per_10k"] ** 2
    quad_cols = ALL_PRED + ["stem_sq"]
    m2, b2, n2 = fit_ols(sub_q, "eci_software", quad_cols)
    out.write(f"N = {n2}, R2 = {m2.rsquared:.4f}, Adj R2 = {m2.rsquared_adj:.4f}\n")
    for v in quad_cols:
        out.write(f"  {v:32s} {fmt_coef(m2, b2, v)}\n")

    # Save Table 5
    rows = []
    for v in mod_cols:
        if v in m1.params.index:
            rows.append({
                "spec": "Moderation",
                "variable": v,
                "B": m1.params[v],
                "SE": m1.bse[v],
                "p": m1.pvalues[v],
                "beta": b1.get(v, np.nan),
            })
    rows.append({"spec": "Moderation", "variable": "N", "B": n1, "SE": "", "p": "", "beta": ""})
    rows.append({"spec": "Moderation", "variable": "R2", "B": m1.rsquared, "SE": "", "p": "", "beta": ""})
    for v in quad_cols:
        if v in m2.params.index:
            rows.append({
                "spec": "Non-linearity",
                "variable": v,
                "B": m2.params[v],
                "SE": m2.bse[v],
                "p": m2.pvalues[v],
                "beta": b2.get(v, np.nan),
            })
    rows.append({"spec": "Non-linearity", "variable": "N", "B": n2, "SE": "", "p": "", "beta": ""})
    rows.append({"spec": "Non-linearity", "variable": "R2", "B": m2.rsquared, "SE": "", "p": "", "beta": ""})
    pd.DataFrame(rows).to_csv(TABLES / "table_05_moderation_nonlinearity.csv", index=False)

    # -------------------------------------------------------------
    # 4. Bundle SPSI robustness
    # -------------------------------------------------------------
    write_section(out, "4. Bundle SPSI (Juhász et al. 2026 mapping)")
    bun = pd.read_csv(JEG_DATA, dtype={"dpto5": str})
    out.write(f"Bundle file rows: {len(bun)} | columns: {list(bun.columns)}\n")

    merged = sub.merge(bun[["dpto5", "ECI_individual", "ECI_bundle"]], on="dpto5", how="inner")
    out.write(f"Merged rows (SPSI sample with bundle data): {len(merged)}\n")

    # Correlation language-level vs bundle
    r_p, p_p = pearsonr(merged["ECI_individual"], merged["ECI_bundle"])
    r_s, p_s = spearmanr(merged["ECI_individual"], merged["ECI_bundle"])
    out.write(f"Correlation language SPSI vs bundle SPSI: Pearson r = {r_p:.4f} (p={p_p:.3e}), Spearman rho = {r_s:.4f} (p={p_s:.3e})\n")

    # OLS with bundle as DV
    merged_bun = merged.copy()
    merged_bun["eci_software"] = merged_bun["ECI_bundle"]
    m3, b3, n3 = fit_ols(merged_bun, "eci_software", ALL_PRED)
    out.write(f"OLS with bundle DV: N = {n3}, R2 = {m3.rsquared:.4f}\n")
    for v in ALL_PRED:
        out.write(f"  {v:32s} {fmt_coef(m3, b3, v)}\n")

    # Save table S7
    rows = []
    rows.append({"metric": "Pearson r (lang vs bundle)", "value": f"{r_p:.4f}"})
    rows.append({"metric": "Spearman rho (lang vs bundle)", "value": f"{r_s:.4f}"})
    rows.append({"metric": "N", "value": n3})
    rows.append({"metric": "R2 (bundle DV)", "value": f"{m3.rsquared:.4f}"})
    for v in ALL_PRED:
        if v in m3.params.index:
            rows.append({
                "metric": f"{v} B (bundle DV)",
                "value": f"{m3.params[v]:.4f} (SE={m3.bse[v]:.4f}, p={m3.pvalues[v]:.4f}, beta={b3.get(v, np.nan):.3f})",
            })
    pd.DataFrame(rows).to_csv(TABLES / "table_S7_bundle_robustness.csv", index=False)

    # -------------------------------------------------------------
    # 5. Temporal panel: rho 2015-2025
    # -------------------------------------------------------------
    write_section(out, "5. Temporal panel SPSI 2015-2025")
    years = [y for y in range(2015, 2026) if f"eci_{y}" in df.columns]
    out.write(f"Available years: {years}\n")
    panel = df[["dpto5"] + [f"eci_{y}" for y in years]].copy()
    # Filter to deps with computable SPSI in at least 2 consecutive years
    consecutive_rhos = []
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        m = panel[[f"eci_{y1}", f"eci_{y2}"]].dropna()
        if len(m) < 30:
            continue
        rho, p = spearmanr(m[f"eci_{y1}"], m[f"eci_{y2}"])
        consecutive_rhos.append({"y1": y1, "y2": y2, "n": len(m), "rho": rho, "p": p})
        out.write(f"  rho({y1} vs {y2}) = {rho:.4f}, N = {len(m)}, p = {p:.3e}\n")
    rhos_only = [r["rho"] for r in consecutive_rhos]
    if rhos_only:
        out.write(f"Range of consecutive-year rho: [{min(rhos_only):.3f}, {max(rhos_only):.3f}]\n")

    # 2020 vs 2025
    if "eci_2020" in panel.columns and "eci_2025" in panel.columns:
        m_2025 = panel[["eci_2020", "eci_2025"]].dropna()
        rho_2020_25, p_2020_25 = spearmanr(m_2025["eci_2020"], m_2025["eci_2025"])
        out.write(f"  rho(2020 vs 2025) = {rho_2020_25:.4f}, N = {len(m_2025)}, p = {p_2020_25:.3e}\n")

        # Top-quintile persistence
        m2 = panel[["eci_2020", "eci_2025"]].dropna().copy()
        q2020 = m2["eci_2020"].quantile(0.8)
        q2025 = m2["eci_2025"].quantile(0.8)
        m2["top_2020"] = m2["eci_2020"] >= q2020
        m2["top_2025"] = m2["eci_2025"] >= q2025
        if m2["top_2020"].sum() > 0:
            persist = (m2["top_2020"] & m2["top_2025"]).sum() / m2["top_2020"].sum()
            out.write(f"  Top-quintile persistence 2020->2025: {persist:.3f} ({(m2['top_2020'] & m2['top_2025']).sum()}/{m2['top_2020'].sum()})\n")

    pd.DataFrame(consecutive_rhos).to_csv(TABLES / "table_S8_temporal_panel.csv", index=False)

    out.close()
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
