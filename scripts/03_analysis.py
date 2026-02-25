"""
03_analysis.py — Main statistical analysis for Article 3.

'The geography of digital complexity follows the geography of
scientific-technical knowledge, not wealth.'

Target: Technology in Society

Outputs:
  figures/  — 8 PNG files (300 DPI)
  tables/   — 7+ CSV files
"""

import warnings
warnings.filterwarnings("ignore")
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from scipy.spatial import KDTree
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns


# ============================================================
# SECTION 0 — Configuration & data loading
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

STYLE = {
    "title_size": 15,
    "axis_label_size": 13,
    "tick_size": 12,
    "annot_size": 12,
    "legend_size": 12,
    "small_annot_size": 11,
    "panel_label_size": 15,
    "dpi": 300,
}

# Muted academic palette (seaborn-inspired)
CLUSTER_COLOURS = {
    1: "#c44e52",   # muted red       (Peripheral-Deprived)
    2: "#4c72b0",   # steel blue      (Metropolitan-Core)
    3: "#55a868",   # sage green      (Metropolitan-Diversified)
    4: "#8172b3",   # muted purple    (Pampeana-Educated)
    5: "#dd8452",   # warm orange     (Semi-Rural-Active)
    6: "#937860",   # warm brown      (Intermediate-Urban)
}

# Muted histogram palette
HIST_COLOURS = {
    "eci": "#4c72b0",
    "devs": "#c44e52",
    "stem": "#55a868",
    "wage": "#8172b3",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": STYLE["tick_size"],
    "axes.titlesize": STYLE["title_size"],
    "axes.labelsize": STYLE["axis_label_size"],
    "xtick.labelsize": STYLE["tick_size"],
    "ytick.labelsize": STYLE["tick_size"],
    "legend.fontsize": STYLE["legend_size"],
    "figure.dpi": 100,
    "savefig.dpi": STYLE["dpi"],
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
})


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_axis(ax, grid=True):
    """Apply consistent academic styling to an axis."""
    despine(ax)
    if grid:
        ax.grid(True, alpha=0.15, linewidth=0.5, color="#888888")
        ax.set_axisbelow(True)


# --- Load data ---
DATA_PATH = ROOT / "data" / "departamentos_master.csv"
df = pd.read_csv(DATA_PATH)
print(f"\n{'=' * 70}")
print(f"  ARTICLE 3 — ANALYSIS SCRIPT")
print(f"{'=' * 70}")
print(f"  Loaded {len(df)} departments, {df.shape[1]} columns")

# Derived variables
df["log_pob"] = np.log(df["pob_2022"].replace(0, np.nan))
df["has_eci"] = df["eci_software"].notna().astype(int)
print(f"  ECI non-null: {df['eci_software'].notna().sum()}")
print(f"  gh_has_devs:  {int(df['gh_has_devs'].sum())}")

# --- Variable lists ---
KNOWLEDGE_VARS = ["cyt_stem_per_10k", "dist_stem_uni_km", "pct_univ_adultos"]
WEALTH_VARS = ["log_wage_median", "rad_mean_2022", "pct_servicios_avanzados"]
CONTROLS = ["log_pob", "inet_penetracion_hog"]
ALL_PREDICTORS = KNOWLEDGE_VARS + WEALTH_VARS + CONTROLS

DV_PRIMARY = "eci_software"
OUTCOMES = [
    "eci_software", "gh_devs_per_10k",
    "gh_language_diversity_index", "eci_diversity",
]

# Reduced spec for type-specific models (4 predictors for DoF)
TYPE_PREDS = ["cyt_stem_per_10k", "log_wage_median", "log_pob", "inet_penetracion_hog"]

VAR_LABELS = {
    "cyt_stem_per_10k": "STEM researchers per 10k",
    "dist_stem_uni_km": "Dist. to STEM uni (km)",
    "pct_univ_adultos": "% university-educated adults",
    "log_wage_median": "ln(Median wage)",
    "rad_mean_2022": "Nighttime radiance 2022",
    "pct_servicios_avanzados": "Advanced services (%)",
    "log_pob": "ln(Population)",
    "inet_penetracion_hog": "Internet penetration (%)",
    "knowledge_intensity": "Knowledge intensity",
    "imr_lambda": "Inverse Mills ratio (\u03bb)",
    "const": "Constant",
}

# Cluster metadata
cluster_labels = {}
if "mca_cluster_label" in df.columns:
    for c in df["mca_cluster"].dropna().unique():
        cluster_labels[int(c)] = df.loc[
            df["mca_cluster"] == c, "mca_cluster_label"
        ].iloc[0]
    print(f"  Clusters: {cluster_labels}")


# --- Helper functions ---

def sig_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "\u2020"
    else:
        return ""


def run_ols(data, predictors, dv, title):
    """OLS with HC1 standard errors.  Print summary, return (model, clean)."""
    cols = predictors + [dv]
    clean = data.dropna(subset=cols)
    n = len(clean)
    if n < len(predictors) + 2:
        print(f"\n  {title}: N = {n} — TOO FEW OBSERVATIONS")
        return None, clean
    X = add_constant(clean[predictors])
    y = clean[dv]
    model = sm.OLS(y, X).fit(cov_type="HC1")
    betas_std = compute_standardised_betas(clean, predictors, dv)
    print(f"\n  {'~' * 70}")
    print(f"  {title}")
    print(f"  {'~' * 70}")
    print(f"  N = {int(model.nobs)}, R\u00b2 = {model.rsquared:.4f}, "
          f"Adj R\u00b2 = {model.rsquared_adj:.4f}, F = {model.fvalue:.3f}")
    print(f"\n  {'Variable':<30} {'B':>8} {'SE':>10} {'t':>8} "
          f"{'p':>10} {'Beta':>8}")
    print(f"  {'-' * 74}")
    for var in model.params.index:
        b = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        label = VAR_LABELS.get(var, var)
        if var == "const":
            print(f"  {label:<30} {b:>8.4f} {se:>10.4f} "
                  f"{t:>8.3f} {p:>10.4f} {'---':>8}")
        else:
            beta = (betas_std[var]["beta"]
                    if betas_std and var in betas_std else np.nan)
            print(f"  {label:<30} {b:>8.4f} {se:>10.4f} "
                  f"{t:>8.3f} {p:>10.4f} {beta:>8.3f} {sig_star(p)}")
    return model, clean


def compute_standardised_betas(data, predictors, dv):
    """Return dict  {var: {beta, se, p, ci_low, ci_high}}."""
    cols = predictors + [dv]
    clean = data.dropna(subset=cols)
    if len(clean) < len(predictors) + 2:
        return None
    X = clean[predictors].copy()
    y = (clean[dv] - clean[dv].mean()) / clean[dv].std()
    for c in predictors:
        X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HC1")
    return {var: {
        "beta": model.params[var],
        "se": model.bse[var],
        "p": model.pvalues[var],
        "ci_low": model.conf_int().loc[var, 0],
        "ci_high": model.conf_int().loc[var, 1],
    } for var in predictors}


def bootstrap_correlations(data, var1, var2,
                           n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence intervals for Pearson r."""
    clean = data[[var1, var2]].dropna()
    n = len(clean)
    if n < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    r_obs = clean[var1].corr(clean[var2])
    boot_r = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = clean.iloc[idx]
        boot_r[b] = sample[var1].corr(sample[var2])
    alpha = (1 - ci) / 2
    return (r_obs,
            np.percentile(boot_r, alpha * 100),
            np.percentile(boot_r, (1 - alpha) * 100))


def partial_corr(data, x, y, z_list):
    """Partial correlation of x and y controlling for z_list."""
    cols = [x, y] + z_list
    clean = data[cols].dropna()
    if len(clean) < len(z_list) + 3:
        return np.nan, np.nan, len(clean)
    Z = add_constant(clean[z_list])
    res_x = sm.OLS(clean[x], Z).fit().resid
    res_y = sm.OLS(clean[y], Z).fit().resid
    r, p = stats.pearsonr(res_x, res_y)
    return r, p, len(clean)


def draw_confidence_ellipse(x, y, ax, n_std=1.5,
                            facecolor="none", **kwargs):
    """Concentration ellipse (n_std=1.5 ~ 78 % of points)."""
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_rx = np.sqrt(1 + pearson)
    ell_ry = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_rx * 2, height=ell_ry * 2,
                       facecolor=facecolor, **kwargs)
    sx = np.sqrt(cov[0, 0]) * n_std
    sy = np.sqrt(cov[1, 1]) * n_std
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(sx, sy)
              .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def compute_vif(data, predictors, dv):
    """Return {var: VIF} dict."""
    clean = data.dropna(subset=predictors + [dv])
    X = add_constant(clean[predictors])
    vifs = {}
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vifs[col] = variance_inflation_factor(X.values, i)
    return vifs


# ============================================================
# SECTION 1 — Descriptive statistics
# ============================================================
print(f"\n\n{'=' * 70}")
print(f"  SECTION 1 — Descriptive statistics")
print(f"{'=' * 70}")

desc_vars = ALL_PREDICTORS + [DV_PRIMARY, "gh_devs_per_10k",
                               "gh_language_diversity_index"]

# Panel (a): all 511 departments
desc_all = df[desc_vars].describe(percentiles=[.25, .5, .75]).T
desc_all = desc_all[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
desc_all.columns = ["N", "Mean", "SD", "Min", "P25", "P50", "P75", "Max"]
desc_all.index = [VAR_LABELS.get(v, v) for v in desc_vars]

# Panel (b): 224 with ECI
df_eci = df[df["eci_software"].notna()]
desc_eci = df_eci[desc_vars].describe(percentiles=[.25, .5, .75]).T
desc_eci = desc_eci[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
desc_eci.columns = ["N", "Mean", "SD", "Min", "P25", "P50", "P75", "Max"]
desc_eci.index = [VAR_LABELS.get(v, v) for v in desc_vars]

# Combine panels
desc_combined = pd.concat(
    {"Panel A: All departments (N=511)": desc_all,
     "Panel B: With ECI (N=224)": desc_eci},
)
desc_combined.to_csv(TAB_DIR / "table_01_descriptive.csv", float_format="%.3f")
print(f"  Saved: table_01_descriptive.csv")
print(desc_all.to_string())

# --- Figure 1: Distribution panels (2x2) ---
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))

plot_info = [
    ("eci_software", "ECI (software)",
     df_eci, HIST_COLOURS["eci"]),
    ("gh_devs_per_10k", "Developers per 10,000 inhab.",
     df[df["gh_devs_per_10k"] > 0], HIST_COLOURS["devs"]),
    ("cyt_stem_per_10k", "STEM researchers per 10,000 inhab.",
     df, HIST_COLOURS["stem"]),
    ("log_wage_median", "ln(Median wage)",
     df, HIST_COLOURS["wage"]),
]

for ax, (var, label, data, colour) in zip(axes1.flat, plot_info):
    vals = data[var].dropna()
    ax.hist(vals, bins=30, color=colour, alpha=0.45, edgecolor="white",
            linewidth=0.5, density=True)
    vals.plot.kde(ax=ax, color=colour, linewidth=2.2)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.text(0.95, 0.95, f"N = {len(vals)}\n\u03bc = {vals.mean():.2f}\n"
            f"\u03c3 = {vals.std():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=STYLE["small_annot_size"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc",
                      alpha=0.9, linewidth=0.5))
    style_axis(ax)

panel_letters = ["(a)", "(b)", "(c)", "(d)"]
for ax, letter in zip(axes1.flat, panel_letters):
    ax.text(-0.02, 1.05, letter, transform=ax.transAxes,
            fontsize=STYLE["panel_label_size"], fontweight="bold", va="bottom")

fig1.tight_layout()
fig1.savefig(FIG_DIR / "fig_01_distributions.png", dpi=STYLE["dpi"])
print(f"  Saved: fig_01_distributions.png")
plt.close(fig1)


# ============================================================
# SECTION 2 — Correlation analysis
# ============================================================
print(f"\n\n{'=' * 70}")
print(f"  SECTION 2 — Correlation analysis")
print(f"{'=' * 70}")

predictors_for_corr = KNOWLEDGE_VARS + WEALTH_VARS + CONTROLS
outcomes_for_corr = OUTCOMES

# Bivariate Pearson correlations (Table 2)
corr_rows = []
for pred in predictors_for_corr:
    row = {"Predictor": VAR_LABELS.get(pred, pred), "Group": (
        "Knowledge" if pred in KNOWLEDGE_VARS else
        "Wealth" if pred in WEALTH_VARS else "Control")}
    for outcome in outcomes_for_corr:
        clean = df[[pred, outcome]].dropna()
        if len(clean) > 5:
            r, p = stats.pearsonr(clean[pred], clean[outcome])
            row[f"r_{outcome}"] = r
            row[f"p_{outcome}"] = p
        else:
            row[f"r_{outcome}"] = np.nan
            row[f"p_{outcome}"] = np.nan
    corr_rows.append(row)

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(TAB_DIR / "table_02_correlations.csv",
               index=False, float_format="%.4f")
print(f"  Saved: table_02_correlations.csv")

# Partial correlations controlling for log_pob (Table 2b)
pcorr_rows = []
for pred in predictors_for_corr:
    if pred == "log_pob":
        continue  # cannot partial-out itself
    row = {"Predictor": VAR_LABELS.get(pred, pred)}
    for outcome in outcomes_for_corr:
        r, p, n = partial_corr(df, pred, outcome, ["log_pob"])
        row[f"r_partial_{outcome}"] = r
        row[f"p_{outcome}"] = p
        row[f"n_{outcome}"] = n
    pcorr_rows.append(row)

pcorr_df = pd.DataFrame(pcorr_rows)
pcorr_df.to_csv(TAB_DIR / "table_02b_partial_correlations.csv",
                index=False, float_format="%.4f")
print(f"  Saved: table_02b_partial_correlations.csv")

# --- Figure 2: Annotated correlation heatmap ---
fig2, ax2 = plt.subplots(figsize=(10, 8))
# Re-enable spines for heatmap
for spine in ax2.spines.values():
    spine.set_visible(True)

# Build matrix (rows = predictors, cols = outcomes)
r_matrix = np.full((len(predictors_for_corr), len(outcomes_for_corr)), np.nan)
annot_matrix = []
for i, pred in enumerate(predictors_for_corr):
    annot_row = []
    for j, outcome in enumerate(outcomes_for_corr):
        clean = df[[pred, outcome]].dropna()
        if len(clean) > 5:
            r, p = stats.pearsonr(clean[pred], clean[outcome])
            r_matrix[i, j] = r
            annot_row.append(f"{r:.2f}{sig_star(p)}")
        else:
            annot_row.append("")
    annot_matrix.append(annot_row)

annot_array = np.array(annot_matrix)
row_labels = [VAR_LABELS.get(v, v) for v in predictors_for_corr]
col_labels = ["ECI", "Devs/10k", "Lang. diversity", "ECI diversity"]

sns.heatmap(
    r_matrix, ax=ax2, annot=annot_array, fmt="",
    xticklabels=col_labels, yticklabels=row_labels,
    cmap="RdBu_r", center=0, vmin=-0.7, vmax=0.7,
    linewidths=0.5, linecolor="white",
    annot_kws={"fontsize": STYLE["annot_size"]},
    cbar_kws={"label": "Pearson r", "shrink": 0.8},
)

# Add group separators
ax2.axhline(len(KNOWLEDGE_VARS), color="black", linewidth=2)
ax2.axhline(len(KNOWLEDGE_VARS) + len(WEALTH_VARS), color="black",
            linewidth=2)
fig2.tight_layout()
fig2.savefig(FIG_DIR / "fig_02_correlation_heatmap.png", dpi=STYLE["dpi"])
print(f"  Saved: fig_02_correlation_heatmap.png")
plt.close(fig2)


# ============================================================
# SECTION 3 — Horse-race scatter plots
# ============================================================
print(f"\n\n{'=' * 70}")
print(f"  SECTION 3 — Horse-race scatter plots")
print(f"{'=' * 70}")

# --- Figure 3: 2x2 scatter panel ---
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

scatter_specs = [
    ("cyt_stem_per_10k", "eci_software",
     "STEM researchers per 10k", "ECI software"),
    ("log_wage_median", "eci_software",
     "ln(Median wage)", "ECI software"),
    ("cyt_stem_per_10k", "gh_devs_per_10k",
     "STEM researchers per 10k", "Developers per 10k"),
    ("log_wage_median", "gh_devs_per_10k",
     "ln(Median wage)", "Developers per 10k"),
]

for ax, (xvar, yvar, xlab, ylab) in zip(axes3.flat, scatter_specs):
    plot_df = df.dropna(subset=[xvar, yvar, "mca_cluster"]).copy()

    # Scatter coloured by cluster
    for c in sorted(plot_df["mca_cluster"].unique()):
        c_int = int(c)
        mask = plot_df["mca_cluster"] == c
        label = cluster_labels.get(c_int, f"Type {c_int}")
        ax.scatter(
            plot_df.loc[mask, xvar], plot_df.loc[mask, yvar],
            c=CLUSTER_COLOURS.get(c_int, "grey"), s=30, alpha=0.6,
            label=label, edgecolors="white", linewidths=0.3,
        )

    # OLS fit line + 95 % CI band
    clean = plot_df[[xvar, yvar]].dropna()
    if len(clean) > 10:
        x_fit = clean[xvar].values
        y_fit = clean[yvar].values
        r_val, p_val = stats.pearsonr(x_fit, y_fit)
        slope, intercept, _, _, se_slope = stats.linregress(x_fit, y_fit)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
        y_line = intercept + slope * x_line
        n_fit = len(x_fit)
        x_mean = x_fit.mean()
        se_line = (np.sqrt(np.sum((y_fit - (intercept + slope * x_fit))**2)
                           / (n_fit - 2))
                   * np.sqrt(1/n_fit + (x_line - x_mean)**2
                             / np.sum((x_fit - x_mean)**2)))
        t_crit = stats.t.ppf(0.975, n_fit - 2)
        ax.plot(x_line, y_line, color="#222222", linewidth=1.8, zorder=5)
        ax.fill_between(x_line, y_line - t_crit * se_line,
                        y_line + t_crit * se_line,
                        alpha=0.12, color="#888888", zorder=4)
        ax.text(0.05, 0.95, f"r = {r_val:.2f}{sig_star(p_val)}\nN = {n_fit}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=STYLE["annot_size"],
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc",
                          alpha=0.9, linewidth=0.5))

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    style_axis(ax)

for ax, letter in zip(axes3.flat, ["(a)", "(b)", "(c)", "(d)"]):
    ax.text(-0.02, 1.05, letter, transform=ax.transAxes,
            fontsize=STYLE["panel_label_size"], fontweight="bold", va="bottom")

# Shared legend
handles, labels = axes3[0, 0].get_legend_handles_labels()
fig3.legend(handles, labels, loc="lower center", ncol=3,
            frameon=False, fontsize=STYLE["legend_size"])
fig3.tight_layout(rect=[0, 0.06, 1, 1])
fig3.savefig(FIG_DIR / "fig_03_horse_race_scatter.png", dpi=STYLE["dpi"])
print(f"  Saved: fig_03_horse_race_scatter.png")
plt.close(fig3)

# --- Figure 4: Marginal R-squared bar chart ---
fig4, ax4 = plt.subplots(figsize=(10, 6))

r2_data = []
for var in KNOWLEDGE_VARS + WEALTH_VARS:
    clean = df[[var, DV_PRIMARY, "log_pob"]].dropna()
    if len(clean) < 10:
        continue
    # Bivariate R²
    X_biv = add_constant(clean[[var]])
    m_biv = sm.OLS(clean[DV_PRIMARY], X_biv).fit()
    r2_biv = m_biv.rsquared

    # Partial R² (incremental over log_pob)
    X_red = add_constant(clean[["log_pob"]])
    m_red = sm.OLS(clean[DV_PRIMARY], X_red).fit()
    X_full = add_constant(clean[["log_pob", var]])
    m_full = sm.OLS(clean[DV_PRIMARY], X_full).fit()
    r2_partial = (m_full.rsquared - m_red.rsquared) / (1 - m_red.rsquared)

    group = "Knowledge" if var in KNOWLEDGE_VARS else "Wealth"
    r2_data.append({
        "Variable": VAR_LABELS.get(var, var),
        "Bivariate R\u00b2": r2_biv,
        "Partial R\u00b2": r2_partial,
        "Group": group,
    })

r2_df = pd.DataFrame(r2_data)
x_pos = np.arange(len(r2_df))
bar_w = 0.35

colours_biv = [HIST_COLOURS["eci"] if g == "Knowledge" else HIST_COLOURS["devs"]
               for g in r2_df["Group"]]
colours_par = ["#8faec4" if g == "Knowledge" else "#d9a09a"
               for g in r2_df["Group"]]

ax4.bar(x_pos - bar_w/2, r2_df["Bivariate R\u00b2"], bar_w,
        color=colours_biv, edgecolor="white", label="Bivariate R\u00b2")
ax4.bar(x_pos + bar_w/2, r2_df["Partial R\u00b2"], bar_w,
        color=colours_par, edgecolor="white", label="Partial R\u00b2 (after ln Pop)")

ax4.set_xticks(x_pos)
ax4.set_xticklabels(r2_df["Variable"], rotation=30, ha="right",
                    fontsize=STYLE["small_annot_size"])
ax4.set_ylabel("R\u00b2")
ax4.legend(frameon=False)
style_axis(ax4)

fig4.tight_layout()
fig4.savefig(FIG_DIR / "fig_04_r2_comparison.png", dpi=STYLE["dpi"])
print(f"  Saved: fig_04_r2_comparison.png")
plt.close(fig4)


# ============================================================
# SECTION 4 — Cluster validation
# ============================================================
print(f"\n\n{'=' * 70}")
print(f"  SECTION 4 — Cluster validation")
print(f"{'=' * 70}")

# Cross-tabulation: cluster × ECI coverage
print(f"\n  Cluster sizes and ECI coverage:")
for c in sorted(df["mca_cluster"].dropna().unique()):
    c_int = int(c)
    label = cluster_labels.get(c_int, f"Type {c_int}")
    n_tot = (df["mca_cluster"] == c).sum()
    n_eci = df.loc[df["mca_cluster"] == c, "eci_software"].notna().sum()
    pct = 100 * n_eci / n_tot if n_tot > 0 else 0
    print(f"    {label:<28} N = {n_tot:>3}  |  N_eci = {n_eci:>3}  ({pct:.0f}%)")

# --- Figure 5: MCA plane — knowledge intensity with cluster ellipses ---
fig5, ax5 = plt.subplots(figsize=(10, 8))

mca_cols = ["mca_dim1", "mca_dim2"]
df_mca = df.dropna(subset=mca_cols + ["knowledge_intensity"]).copy()

sc5 = ax5.scatter(
    df_mca["mca_dim1"], df_mca["mca_dim2"],
    c=df_mca["knowledge_intensity"], cmap="viridis", s=40, alpha=0.7,
    edgecolors="none", zorder=2,
)
cbar5 = plt.colorbar(sc5, ax=ax5, shrink=0.75, pad=0.02)
cbar5.set_label("Knowledge intensity (composite)",
                fontsize=STYLE["axis_label_size"])
cbar5.ax.tick_params(labelsize=STYLE["tick_size"])

# Cluster ellipses with labels
for c in sorted(df_mca["mca_cluster"].dropna().unique()):
    c_int = int(c)
    mask = df_mca["mca_cluster"] == c
    label = cluster_labels.get(c_int, f"Type {c_int}")
    colour = CLUSTER_COLOURS.get(c_int, "grey")
    x_vals = df_mca.loc[mask, "mca_dim1"].values
    y_vals = df_mca.loc[mask, "mca_dim2"].values
    draw_confidence_ellipse(
        x_vals, y_vals, ax5, n_std=1.5,
        edgecolor=colour, linewidth=2.5, linestyle="-", alpha=0.9, zorder=3,
    )
    cx, cy = np.mean(x_vals), np.mean(y_vals)
    ax5.annotate(
        label, (cx, cy),
        fontsize=STYLE["axis_label_size"], fontweight="bold",
        ha="center", va="center", color=colour,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        zorder=4,
    )

ax5.set_xlabel("MCA Dimension 1")
ax5.set_ylabel("MCA Dimension 2")
ax5.axhline(0, color="#bbbbbb", linewidth=0.6, linestyle="--", zorder=1)
ax5.axvline(0, color="#bbbbbb", linewidth=0.6, linestyle="--", zorder=1)
style_axis(ax5, grid=False)

fig5.tight_layout()
fig5.savefig(FIG_DIR / "fig_05_mca_knowledge.png", dpi=STYLE["dpi"])
print(f"  Saved: fig_05_mca_knowledge.png")
plt.close(fig5)


# ============================================================
# SECTION 5 — Regression models
# ============================================================
print(f"\n\n{'=' * 70}")
print(f"  SECTION 5 — Regression models")
print(f"{'=' * 70}")

# --- Model 1: Knowledge only ---
m1, d1 = run_ols(df, KNOWLEDGE_VARS + CONTROLS, DV_PRIMARY,
                 "Model 1: Knowledge only")

# --- Model 2: Wealth only ---
m2, d2 = run_ols(df, WEALTH_VARS + CONTROLS, DV_PRIMARY,
                 "Model 2: Wealth only")

# --- Model 3: Horse race (full) ---
m3, d3 = run_ols(df, ALL_PREDICTORS, DV_PRIMARY,
                 "Model 3: Full horse race")

# VIF for Models 1-3
print(f"\n  VIF diagnostics:")
for label_m, preds_m in [("Model 1", KNOWLEDGE_VARS + CONTROLS),
                          ("Model 2", WEALTH_VARS + CONTROLS),
                          ("Model 3", ALL_PREDICTORS)]:
    vifs = compute_vif(df, preds_m, DV_PRIMARY)
    print(f"\n    {label_m}:")
    for var, vif in vifs.items():
        flag = " << HIGH" if vif > 5 else (" << MODERATE" if vif > 2.5 else "")
        print(f"      {VAR_LABELS.get(var, var):<30} VIF = {vif:.2f}{flag}")

# --- Model 4: Heckman selection correction ---
print(f"\n  {'~' * 70}")
print(f"  Model 4: Heckman two-step selection correction")
print(f"  {'~' * 70}")

SELECTION_VARS = ["log_pob", "pct_nbi_2022", "inet_penetracion_hog",
                  "pct_computadora"]
sel_cols = SELECTION_VARS + ["has_eci"]
df_sel = df.dropna(subset=sel_cols).copy()

# Stage 1: Probit
X_sel = add_constant(df_sel[SELECTION_VARS])
probit = sm.Probit(df_sel["has_eci"], X_sel).fit(disp=0)
print(f"\n  Stage 1 — Probit (N = {int(probit.nobs)})")
print(f"  Pseudo R\u00b2 = {probit.prsquared:.4f}")
print(f"  LR chi\u00b2 = {probit.llr:.2f}, p = {probit.llr_pvalue:.2e}")
for var in probit.params.index:
    b, se, p = probit.params[var], probit.bse[var], probit.pvalues[var]
    label = VAR_LABELS.get(var, var)
    print(f"    {label:<30} B = {b:>7.4f}  SE = {se:.4f}  "
          f"p = {p:.4f} {sig_star(p)}")

# Inverse Mills ratio
xb = (X_sel.values @ probit.params.values).astype(float)
df_sel["imr_lambda"] = norm.pdf(xb) / norm.cdf(xb)

# Stage 2: OLS on selected sample with IMR
df_stage2 = df_sel[df_sel["has_eci"] == 1].copy()
# Merge ECI values
df_stage2["eci_software"] = df.loc[df_stage2.index, "eci_software"]
stage2_preds = ALL_PREDICTORS + ["imr_lambda"]
m4, d4 = run_ols(df_stage2, stage2_preds, DV_PRIMARY,
                 "Model 4: Stage 2 OLS with IMR (\u03bb)")

if m4 is not None:
    lambda_p = m4.pvalues.get("imr_lambda", np.nan)
    print(f"\n  Selection bias test: \u03bb p-value = {lambda_p:.4f} "
          f"{'** SIGNIFICANT' if lambda_p < 0.05 else '(not significant)'}")

# --- Model 5: Type-specific ---
print(f"\n  {'~' * 70}")
print(f"  Model 5: Type-specific regressions")
print(f"  {'~' * 70}")

type_results = {}
type_betas = {}
clusters_sorted = sorted(df["mca_cluster"].dropna().unique())

for c in clusters_sorted:
    c_int = int(c)
    label = cluster_labels.get(c_int, f"Type {c_int}")
    subset = df[df["mca_cluster"] == c].copy()
    n_eci = subset["eci_software"].notna().sum()

    if n_eci >= 30:
        model_c, data_c = run_ols(
            subset, TYPE_PREDS, DV_PRIMARY,
            f"Type {c_int}: {label} (N_eci = {n_eci})")
        type_results[c_int] = model_c
        type_betas[c_int] = compute_standardised_betas(
            subset, TYPE_PREDS, DV_PRIMARY)
    else:
        print(f"\n  Type {c_int}: {label} — N_eci = {n_eci} < 30")
        print(f"  Using bivariate correlations with bootstrap CI.")
        type_results[c_int] = None
        type_betas[c_int] = None
        eci_sub = subset[subset["eci_software"].notna()]
        print(f"\n  {'Predictor':<30} {'r':>8} {'CI lo':>8} "
              f"{'CI hi':>8} {'N':>5}")
        print(f"  {'-' * 59}")
        for var in TYPE_PREDS:
            r, ci_lo, ci_hi = bootstrap_correlations(eci_sub, var, DV_PRIMARY)
            n_v = eci_sub[[var, DV_PRIMARY]].dropna().shape[0]
            print(f"  {VAR_LABELS.get(var, var):<30} {r:>8.3f} "
                  f"{ci_lo:>8.3f} {ci_hi:>8.3f} {n_v:>5}")

# --- Model 6: Extensive margin (Logit) ---
print(f"\n  {'~' * 70}")
print(f"  Model 6: Logit — extensive margin (gh_has_devs)")
print(f"  {'~' * 70}")

logit_cols = ALL_PREDICTORS + ["gh_has_devs"]
df_logit = df.dropna(subset=logit_cols).copy()
X_logit = add_constant(df_logit[ALL_PREDICTORS])
logit_model = sm.Logit(df_logit["gh_has_devs"], X_logit).fit(disp=0)

print(f"  N = {int(logit_model.nobs)}, "
      f"Pseudo R\u00b2 = {logit_model.prsquared:.4f}")
print(f"  LR chi\u00b2 = {logit_model.llr:.2f}, "
      f"p = {logit_model.llr_pvalue:.2e}")
print(f"\n  {'Variable':<30} {'B':>8} {'SE':>10} {'OR':>8} "
      f"{'p':>10}")
print(f"  {'-' * 66}")
for var in logit_model.params.index:
    b = logit_model.params[var]
    se = logit_model.bse[var]
    p = logit_model.pvalues[var]
    odds = np.exp(b) if var != "const" else np.nan
    label = VAR_LABELS.get(var, var)
    if var == "const":
        print(f"  {label:<30} {b:>8.4f} {se:>10.4f} {'---':>8} "
              f"{p:>10.4f}")
    else:
        print(f"  {label:<30} {b:>8.4f} {se:>10.4f} {odds:>8.3f} "
              f"{p:>10.4f} {sig_star(p)}")

# --- Model 6b: Type-specific logit (extensive margin by type) ---
print(f"\n  {'~' * 70}")
print(f"  Model 6b: Type-specific logit — extensive margin by type")
print(f"  {'~' * 70}")

logit_type_rows = []
for c in clusters_sorted:
    c_int = int(c)
    label = cluster_labels.get(c_int, f"Type {c_int}")
    subset = df[df["mca_cluster"] == c].copy()
    n_tot = len(subset)
    n_devs = int(subset["gh_has_devs"].sum())
    pct_devs = 100 * n_devs / n_tot if n_tot > 0 else 0

    # Need sufficient variation in DV and minimum N
    if n_tot < 50 or n_devs < 5 or (n_tot - n_devs) < 5:
        print(f"\n  {label} (N = {n_tot}): {n_devs} with devs ({pct_devs:.0f}%)"
              f" — descriptive only")
        logit_type_rows.append({
            "Type": label, "N": n_tot, "N_devs": n_devs,
            "pct_devs": f"{pct_devs:.1f}", "Pseudo_R2": "—",
            **{f"OR_{v}": "—" for v in TYPE_PREDS},
            **{f"p_{v}": "—" for v in TYPE_PREDS},
        })
        continue

    logit_cols_t = TYPE_PREDS + ["gh_has_devs"]
    sub_clean = subset.dropna(subset=logit_cols_t).copy()
    n_clean = len(sub_clean)
    n_devs_clean = int(sub_clean["gh_has_devs"].sum())

    if n_clean < 20 or n_devs_clean < 5 or (n_clean - n_devs_clean) < 5:
        print(f"\n  {label} (N_clean = {n_clean}): insufficient variation"
              f" — descriptive only")
        logit_type_rows.append({
            "Type": label, "N": n_clean, "N_devs": n_devs_clean,
            "pct_devs": f"{100 * n_devs_clean / n_clean:.1f}" if n_clean > 0 else "—",
            "Pseudo_R2": "—",
            **{f"OR_{v}": "—" for v in TYPE_PREDS},
            **{f"p_{v}": "—" for v in TYPE_PREDS},
        })
        continue

    try:
        X_t = add_constant(sub_clean[TYPE_PREDS])
        logit_t = sm.Logit(sub_clean["gh_has_devs"], X_t).fit(disp=0,
                           maxiter=100, method="newton")
        print(f"\n  {label} (N = {n_clean}, devs = {n_devs_clean}, "
              f"{100*n_devs_clean/n_clean:.0f}%)")
        print(f"  Pseudo R² = {logit_t.prsquared:.4f}")
        row = {"Type": label, "N": n_clean, "N_devs": n_devs_clean,
               "pct_devs": f"{100*n_devs_clean/n_clean:.1f}",
               "Pseudo_R2": f"{logit_t.prsquared:.4f}"}
        for v in TYPE_PREDS:
            if v in logit_t.params.index:
                b = logit_t.params[v]
                p = logit_t.pvalues[v]
                odds = np.exp(b)
                row[f"OR_{v}"] = f"{odds:.3f}"
                row[f"p_{v}"] = f"{p:.4f}"
                lbl = VAR_LABELS.get(v, v)
                print(f"    {lbl:<30} OR = {odds:>7.3f}  p = {p:.4f} "
                      f"{sig_star(p)}")
            else:
                row[f"OR_{v}"] = "—"
                row[f"p_{v}"] = "—"
        logit_type_rows.append(row)
    except Exception as e:
        print(f"\n  {label} (N = {n_clean}): convergence failed — {e}")
        logit_type_rows.append({
            "Type": label, "N": n_clean, "N_devs": n_devs_clean,
            "pct_devs": f"{100*n_devs_clean/n_clean:.1f}" if n_clean > 0 else "—",
            "Pseudo_R2": "FAILED",
            **{f"OR_{v}": "—" for v in TYPE_PREDS},
            **{f"p_{v}": "—" for v in TYPE_PREDS},
        })

pd.DataFrame(logit_type_rows).to_csv(TAB_DIR / "table_08_logit_type.csv",
                                      index=False)
print(f"\n  Saved: table_08_logit_type.csv")

# --- Chow test ---
print(f"\n  Chow test (structural break across types):")
eci_chow = df[df["eci_software"].notna() & df["mca_cluster"].notna()].copy()
eci_chow = eci_chow.dropna(subset=ALL_PREDICTORS + [DV_PRIMARY, "mca_cluster"])

ref_cluster = clusters_sorted[0]
type_dummies = pd.get_dummies(eci_chow["mca_cluster"], prefix="type",
                              dtype=float)
type_dummy_cols = [c for c in type_dummies.columns
                   if c != f"type_{ref_cluster}"]
eci_chow = pd.concat([eci_chow, type_dummies[type_dummy_cols]], axis=1)

# Restricted model
X_r = add_constant(eci_chow[ALL_PREDICTORS])
y_chow = eci_chow[DV_PRIMARY]
model_r = sm.OLS(y_chow, X_r).fit()
rss_r = model_r.ssr
k_r = model_r.df_model + 1

# Unrestricted model (+ type dummies + interactions)
interaction_cols = []
for dummy in type_dummy_cols:
    for pred in ALL_PREDICTORS:
        col_name = f"{dummy}_x_{pred}"
        eci_chow[col_name] = eci_chow[dummy] * eci_chow[pred]
        interaction_cols.append(col_name)

X_u = add_constant(
    eci_chow[ALL_PREDICTORS + type_dummy_cols + interaction_cols])
model_u = sm.OLS(y_chow, X_u).fit()
rss_u = model_u.ssr
k_u = model_u.df_model + 1
n_chow = len(y_chow)

q = k_u - k_r
f_chow = ((rss_r - rss_u) / q) / (rss_u / (n_chow - k_u))
p_chow = 1 - stats.f.cdf(f_chow, q, n_chow - k_u)

print(f"    Restricted R\u00b2 = {model_r.rsquared:.4f}, "
      f"Unrestricted R\u00b2 = {model_u.rsquared:.4f}")
print(f"    F = {f_chow:.4f}, q = {q}, p = {p_chow:.4f}")
if p_chow < 0.05:
    print(f"    -> SIGNIFICANT: coefficients differ across types")
else:
    print(f"    -> NOT significant at p < 0.05")

# --- Save regression tables ---

# Table 3: Models 1-3
def extract_model_col(model, predictors):
    """Extract {var: 'coef (SE) stars'} from fitted model."""
    if model is None:
        return {}
    result = {}
    betas = compute_standardised_betas(
        model.model.data.frame if hasattr(model.model.data, "frame")
        else pd.DataFrame(model.model.exog, columns=model.model.exog_names)
              .assign(**{model.model.endog_names: model.model.endog}),
        [v for v in predictors if v != "const"],
        model.model.endog_names
    )
    for var in model.params.index:
        b = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]
        beta = betas[var]["beta"] if betas and var in betas else np.nan
        result[var] = {
            "B": b, "SE": se, "p": p,
            "Beta": beta if var != "const" else np.nan,
            "display": f"{b:.4f} ({se:.4f}){sig_star(p)}",
        }
    result["_N"] = int(model.nobs)
    result["_R2"] = model.rsquared
    result["_AdjR2"] = model.rsquared_adj
    return result

tab3_rows = []
all_vars_tab = ["const"] + ALL_PREDICTORS

# Pre-compute standardised betas for each model
_m_betas = {}
for mname, model, preds in [("Model 1", m1, KNOWLEDGE_VARS + CONTROLS),
                             ("Model 2", m2, WEALTH_VARS + CONTROLS),
                             ("Model 3", m3, ALL_PREDICTORS)]:
    if model is not None:
        _m_betas[mname] = compute_standardised_betas(df, preds, DV_PRIMARY)

for var in all_vars_tab:
    row = {"Variable": VAR_LABELS.get(var, var)}
    for mname, model, preds in [("Model 1", m1, KNOWLEDGE_VARS + CONTROLS),
                                 ("Model 2", m2, WEALTH_VARS + CONTROLS),
                                 ("Model 3", m3, ALL_PREDICTORS)]:
        if model is not None and var in model.params.index:
            b = model.params[var]
            se = model.bse[var]
            p = model.pvalues[var]
            row[mname] = f"{b:.4f} ({se:.4f}){sig_star(p)}"
            # Add standardised beta column
            betas = _m_betas.get(mname, {})
            if var != "const" and betas and var in betas:
                row[f"\u03b2 {mname}"] = f"{betas[var]['beta']:.3f}"
            else:
                row[f"\u03b2 {mname}"] = ""
        else:
            row[mname] = ""
            row[f"\u03b2 {mname}"] = ""
    tab3_rows.append(row)

# Add diagnostics
for diag, key in [("N", "nobs"), ("R\u00b2", "rsquared"),
                  ("Adj R\u00b2", "rsquared_adj")]:
    row = {"Variable": diag}
    for mname, model in [("Model 1", m1), ("Model 2", m2), ("Model 3", m3)]:
        if model is not None:
            val = getattr(model, key)
            row[mname] = (f"{int(val)}" if key == "nobs"
                          else f"{val:.4f}")
        else:
            row[mname] = ""
    tab3_rows.append(row)

pd.DataFrame(tab3_rows).to_csv(TAB_DIR / "table_03_models.csv",
                                index=False)
print(f"  Saved: table_03_models.csv")

# Table 4: Heckman
tab4_rows = []
if probit is not None:
    for var in probit.params.index:
        tab4_rows.append({
            "Variable": VAR_LABELS.get(var, var),
            "Stage 1 (Probit)": (f"{probit.params[var]:.4f} "
                                 f"({probit.bse[var]:.4f})"
                                 f"{sig_star(probit.pvalues[var])}"),
        })
    tab4_rows.append({"Variable": "N",
                       "Stage 1 (Probit)": str(int(probit.nobs))})
    tab4_rows.append({"Variable": "Pseudo R\u00b2",
                       "Stage 1 (Probit)": f"{probit.prsquared:.4f}"})
if m4 is not None:
    for var in m4.params.index:
        # Find existing row or create new
        existing = [r for r in tab4_rows
                    if r["Variable"] == VAR_LABELS.get(var, var)]
        entry = (f"{m4.params[var]:.4f} ({m4.bse[var]:.4f})"
                 f"{sig_star(m4.pvalues[var])}")
        if existing:
            existing[0]["Stage 2 (OLS + \u03bb)"] = entry
        else:
            tab4_rows.append({
                "Variable": VAR_LABELS.get(var, var),
                "Stage 2 (OLS + \u03bb)": entry,
            })
    # Diagnostics for stage 2
    for r in tab4_rows:
        if r["Variable"] == "N":
            r["Stage 2 (OLS + \u03bb)"] = str(int(m4.nobs))
        elif r["Variable"] == "Pseudo R\u00b2":
            r["Stage 2 (OLS + \u03bb)"] = f"R\u00b2 = {m4.rsquared:.4f}"

pd.DataFrame(tab4_rows).to_csv(TAB_DIR / "table_04_heckman.csv",
                                index=False)
print(f"  Saved: table_04_heckman.csv")

# Table 5: Type-specific
tab5_rows = []
for var in TYPE_PREDS:
    row = {"Variable": VAR_LABELS.get(var, var)}
    if m3 is not None and var in m3.params.index:
        b, se, p = m3.params[var], m3.bse[var], m3.pvalues[var]
        row["Pooled"] = f"{b:.4f} ({se:.4f}){sig_star(p)}"
    for c in clusters_sorted:
        c_int = int(c)
        label = cluster_labels.get(c_int, f"Type {c_int}")
        m = type_results.get(c_int)
        if m is not None and var in m.params.index:
            b, se, p = m.params[var], m.bse[var], m.pvalues[var]
            row[label] = f"{b:.4f} ({se:.4f}){sig_star(p)}"
        else:
            row[label] = "—"
    tab5_rows.append(row)

# Diagnostics
diag_row_n = {"Variable": "N"}
diag_row_r2 = {"Variable": "R\u00b2"}
if m3 is not None:
    diag_row_n["Pooled"] = str(int(m3.nobs))
    diag_row_r2["Pooled"] = f"{m3.rsquared:.4f}"
for c in clusters_sorted:
    c_int = int(c)
    label = cluster_labels.get(c_int, f"Type {c_int}")
    m = type_results.get(c_int)
    if m is not None:
        diag_row_n[label] = str(int(m.nobs))
        diag_row_r2[label] = f"{m.rsquared:.4f}"
    else:
        n_eci = (df.loc[df["mca_cluster"] == c, "eci_software"]
                 .notna().sum())
        diag_row_n[label] = f"<30 ({n_eci})"
        diag_row_r2[label] = "—"
tab5_rows.extend([diag_row_n, diag_row_r2])

pd.DataFrame(tab5_rows).to_csv(TAB_DIR / "table_05_type_specific.csv",
                                index=False)
print(f"  Saved: table_05_type_specific.csv")

# Table 6: Logit
tab6_rows = []
for var in logit_model.params.index:
    b = logit_model.params[var]
    se = logit_model.bse[var]
    p = logit_model.pvalues[var]
    odds = np.exp(b) if var != "const" else np.nan
    tab6_rows.append({
        "Variable": VAR_LABELS.get(var, var),
        "B": f"{b:.4f}",
        "SE": f"{se:.4f}",
        "OR": f"{odds:.3f}" if not np.isnan(odds) else "—",
        "p": f"{p:.4f}",
        "sig": sig_star(p),
    })
tab6_rows.append({"Variable": "N", "B": str(int(logit_model.nobs))})
tab6_rows.append({"Variable": "Pseudo R\u00b2",
                   "B": f"{logit_model.prsquared:.4f}"})

pd.DataFrame(tab6_rows).to_csv(TAB_DIR / "table_06_logit.csv",
                                index=False)
print(f"  Saved: table_06_logit.csv")

# --- Figure 6: Forest plot (pooled + type-specific) ---
plot_data = []

# Pooled betas from Model 3 (using TYPE_PREDS for comparability)
pooled_betas = compute_standardised_betas(df, TYPE_PREDS, DV_PRIMARY)
if pooled_betas:
    for var in TYPE_PREDS:
        if var in pooled_betas:
            plot_data.append({
                "variable": VAR_LABELS.get(var, var),
                "type": "Pooled",
                "beta": pooled_betas[var]["beta"],
                "ci_low": pooled_betas[var]["ci_low"],
                "ci_high": pooled_betas[var]["ci_high"],
                "p": pooled_betas[var]["p"],
            })

# Per-type betas
for c in clusters_sorted:
    c_int = int(c)
    label_c = cluster_labels.get(c_int, f"Type {c_int}")
    betas = type_betas.get(c_int)
    if betas:
        for var in TYPE_PREDS:
            if var in betas:
                plot_data.append({
                    "variable": VAR_LABELS.get(var, var),
                    "type": label_c,
                    "beta": betas[var]["beta"],
                    "ci_low": betas[var]["ci_low"],
                    "ci_high": betas[var]["ci_high"],
                    "p": betas[var]["p"],
                })

plot_df = pd.DataFrame(plot_data)

if len(plot_df) > 0:
    variables = list(dict.fromkeys(plot_df["variable"]))
    types = list(dict.fromkeys(plot_df["type"]))
    n_vars = len(variables)
    n_types = len(types)

    fig6, ax6 = plt.subplots(figsize=(12, max(7, n_vars * 1.8)))

    forest_colours = ["black"]
    for c in clusters_sorted:
        c_int = int(c)
        if c_int in type_betas and type_betas[c_int] is not None:
            forest_colours.append(CLUSTER_COLOURS.get(c_int, "grey"))
    offsets = np.linspace(-0.3, 0.3, n_types)

    for j, type_name in enumerate(types):
        sub = plot_df[plot_df["type"] == type_name]
        colour = forest_colours[j % len(forest_colours)]
        for _, row in sub.iterrows():
            var_idx = variables.index(row["variable"])
            y_pos = n_vars - 1 - var_idx + offsets[j]
            marker = "D" if type_name == "Pooled" else "o"
            size = 13 if type_name == "Pooled" else 11

            ax6.errorbar(
                row["beta"], y_pos,
                xerr=[[row["beta"] - row["ci_low"]],
                      [row["ci_high"] - row["beta"]]],
                fmt=marker, color=colour, markersize=size,
                capsize=5, capthick=2.0, linewidth=2.5, alpha=0.85,
            )

    ax6.axvline(0, color="#999999", linestyle="--", linewidth=1, zorder=1)
    ax6.set_yticks(range(n_vars))
    ax6.set_yticklabels(list(reversed(variables)))
    ax6.set_xlabel("Standardised coefficient (\u03b2) with 95% CI")
    style_axis(ax6)
    # Horizontal guides for each variable
    for y_g in range(n_vars):
        ax6.axhline(y_g, color="#eeeeee", linewidth=0.5, zorder=0)

    legend_handles = []
    for j, type_name in enumerate(types):
        colour = forest_colours[j % len(forest_colours)]
        marker = "D" if type_name == "Pooled" else "o"
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=colour, markersize=12, label=type_name))
    ax6.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False,
               fontsize=STYLE["legend_size"])

    fig6.tight_layout(rect=[0, 0.10, 1, 1])
    fig6.savefig(FIG_DIR / "fig_06_forest_plot.png", dpi=STYLE["dpi"])
    print(f"  Saved: fig_06_forest_plot.png")
    plt.close(fig6)

# --- Figure 7: R-squared comparison bars ---
fig7, ax7 = plt.subplots(figsize=(8, 5))

model_names = ["Model 1\n(Knowledge)", "Model 2\n(Wealth)",
               "Model 3\n(Full)"]
r2_vals = []
adjr2_vals = []
for m in [m1, m2, m3]:
    if m is not None:
        r2_vals.append(m.rsquared)
        adjr2_vals.append(m.rsquared_adj)
    else:
        r2_vals.append(0)
        adjr2_vals.append(0)

x7 = np.arange(len(model_names))
w7 = 0.35
ax7.bar(x7 - w7/2, r2_vals, w7, color=HIST_COLOURS["eci"], label="R\u00b2",
        edgecolor="white")
ax7.bar(x7 + w7/2, adjr2_vals, w7, color="#8faec4", label="Adj R\u00b2",
        edgecolor="white")

# Annotate values
for i, (r2, ar2) in enumerate(zip(r2_vals, adjr2_vals)):
    ax7.text(i - w7/2, r2 + 0.01, f"{r2:.3f}", ha="center",
             fontsize=STYLE["small_annot_size"])
    ax7.text(i + w7/2, ar2 + 0.01, f"{ar2:.3f}", ha="center",
             fontsize=STYLE["small_annot_size"])

ax7.set_xticks(x7)
ax7.set_xticklabels(model_names)
ax7.set_ylabel("R\u00b2")
ax7.set_ylim(0, max(r2_vals) * 1.2 if max(r2_vals) > 0 else 1)
ax7.legend(frameon=False)
style_axis(ax7)

fig7.tight_layout()
fig7.savefig(FIG_DIR / "fig_07_r2_models.png", dpi=STYLE["dpi"])
print(f"  Saved: fig_07_r2_models.png")
plt.close(fig7)


# ============================================================
# SECTION 6 — Spatial diagnostics (Moran's I)
# ============================================================
print(f"\n\n{'=' * 70}")
print(f"  SECTION 6 — Spatial diagnostics")
print(f"{'=' * 70}")

GEOJSON_PATH = ROOT / "data" / "satelital" / "departamentos_polygons.geojson"

if m3 is not None and GEOJSON_PATH.exists():
    # Load centroids from GeoJSON
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)

    centroids = {}
    for feat in gj["features"]:
        dpto5 = feat["properties"]["dpto5"]
        geom = feat["geometry"]
        # Compute centroid from polygon coordinates
        if geom["type"] == "Polygon":
            coords = np.array(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            # Use largest polygon
            largest = max(geom["coordinates"], key=lambda p: len(p[0]))
            coords = np.array(largest[0])
        else:
            continue
        centroids[dpto5] = (coords[:, 0].mean(), coords[:, 1].mean())

    # Match Model 3 observations to centroids
    resid_data = d3.dropna(subset=ALL_PREDICTORS + [DV_PRIMARY]).copy()
    X_resid = add_constant(resid_data[ALL_PREDICTORS])
    y_resid = resid_data[DV_PRIMARY]
    m3_refit = sm.OLS(y_resid, X_resid).fit()
    residuals = m3_refit.resid.values

    # Get coordinates for each observation
    coord_list = []
    valid_idx = []
    for i, (idx, row) in enumerate(resid_data.iterrows()):
        dpto5 = str(row["dpto5"]).zfill(5)
        if dpto5 in centroids:
            coord_list.append(centroids[dpto5])
            valid_idx.append(i)

    if len(valid_idx) > 20:
        coords_arr = np.array(coord_list)
        e = residuals[valid_idx]
        N = len(e)

        # KNN=5 distance weights (row-standardised)
        tree = KDTree(coords_arr)
        K = 5
        W = np.zeros((N, N))
        for i in range(N):
            dists, indices = tree.query(coords_arr[i], k=K + 1)
            # Exclude self (index 0)
            neighbours = indices[1:]
            W[i, neighbours] = 1.0 / K

        # Moran's I
        e_dm = e - e.mean()
        numerator = e_dm @ W @ e_dm
        denominator = e_dm @ e_dm
        S0 = W.sum()
        I_obs = (N / S0) * (numerator / denominator)
        E_I = -1.0 / (N - 1)

        # Permutation test (999 permutations)
        n_perm = 999
        rng = np.random.RandomState(42)
        I_perm = np.zeros(n_perm)
        for p_idx in range(n_perm):
            e_shuf = rng.permutation(e_dm)
            I_perm[p_idx] = (N / S0) * (e_shuf @ W @ e_shuf) / denominator

        z_score = (I_obs - np.mean(I_perm)) / np.std(I_perm)
        p_moran = (np.sum(np.abs(I_perm) >= np.abs(I_obs)) + 1) / (n_perm + 1)

        print(f"\n  Moran's I on Model 3 residuals (KNN = {K}):")
        print(f"    I = {I_obs:.4f}")
        print(f"    E[I] = {E_I:.4f}")
        print(f"    z = {z_score:.3f}")
        print(f"    p (permutation, {n_perm} runs) = {p_moran:.4f}")
        if p_moran < 0.05:
            print(f"    -> Significant spatial autocorrelation detected")
        else:
            print(f"    -> No significant spatial autocorrelation")
        print(f"    N matched = {N} of {len(residuals)} observations")
    else:
        print(f"  WARNING: Only {len(valid_idx)} observations matched "
              f"to centroids. Skipping Moran's I.")
else:
    if m3 is None:
        print(f"  Model 3 not estimated; skipping Moran's I.")
    else:
        print(f"  GeoJSON not found at {GEOJSON_PATH}; skipping Moran's I.")


# ============================================================
# SECTION 7 — Robustness checks
# ============================================================
print(f"\n\n{'=' * 70}")
print(f"  SECTION 7 — Robustness checks")
print(f"{'=' * 70}")

robustness_specs = {}

# 7a: Alternative DVs
print(f"\n  7a — Alternative dependent variables")
m7a_devs, _ = run_ols(df, ALL_PREDICTORS, "gh_devs_per_10k",
                       "Robustness 7a: DV = gh_devs_per_10k")
robustness_specs["DV: devs_per_10k"] = m7a_devs

m7a_lang, _ = run_ols(df, ALL_PREDICTORS, "gh_language_diversity_index",
                       "Robustness 7a: DV = gh_language_diversity_index")
robustness_specs["DV: lang_diversity"] = m7a_lang

# 7b: Alternative knowledge specification
print(f"\n  7b — Alternative knowledge proxy: knowledge_intensity")
ALT_KNOWLEDGE = ["knowledge_intensity", "dist_stem_uni_km", "pct_univ_adultos"]
ALT_PREDICTORS = ALT_KNOWLEDGE + WEALTH_VARS + CONTROLS
m7b, _ = run_ols(df, ALT_PREDICTORS, DV_PRIMARY,
                 "Robustness 7b: knowledge_intensity replaces cyt_stem_per_10k")
robustness_specs["Alt knowledge spec"] = m7b

# 7c: Region fixed effects
print(f"\n  7c — Region fixed effects")
if "region" in df.columns:
    df_fe = df.dropna(subset=ALL_PREDICTORS + [DV_PRIMARY, "region"]).copy()
    region_dummies = pd.get_dummies(df_fe["region"], prefix="reg", dtype=float)
    # Drop first region as reference
    ref_region = region_dummies.columns[0]
    region_cols = [c for c in region_dummies.columns if c != ref_region]
    df_fe = pd.concat([df_fe, region_dummies[region_cols]], axis=1)
    fe_preds = ALL_PREDICTORS + region_cols
    m7c, _ = run_ols(df_fe, fe_preds, DV_PRIMARY,
                     "Robustness 7c: Model 3 + region FE")
    robustness_specs["Region FE"] = m7c
else:
    print(f"  No 'region' column found; skipping region FE.")
    m7c = None

# Table 7: Robustness summary
tab7_rows = []
target_var = "cyt_stem_per_10k"

# Map each spec to its (data_source, predictor_list, dv)
_spec_meta = {
    "DV: devs_per_10k":    (df, ALL_PREDICTORS, "gh_devs_per_10k"),
    "DV: lang_diversity":   (df, ALL_PREDICTORS, "gh_language_diversity_index"),
    "Alt knowledge spec":   (df, ALT_PREDICTORS, DV_PRIMARY),
    "Region FE":            (df_fe if "df_fe" in dir() else df,
                             fe_preds if "fe_preds" in dir() else ALL_PREDICTORS,
                             DV_PRIMARY),
}

for spec_name, model in robustness_specs.items():
    if model is not None and target_var in model.params.index:
        src, preds, dv_name = _spec_meta.get(
            spec_name, (df, ALL_PREDICTORS, DV_PRIMARY))
        betas = compute_standardised_betas(src, preds, dv_name)
        b = model.params[target_var]
        se = model.bse[target_var]
        p = model.pvalues[target_var]
        beta_val = (betas[target_var]["beta"]
                    if betas and target_var in betas else np.nan)
        tab7_rows.append({
            "Specification": spec_name,
            "B": f"{b:.4f}",
            "SE": f"{se:.4f}",
            "Beta": f"{beta_val:.4f}" if not np.isnan(beta_val) else "—",
            "p": f"{p:.4f}",
            "sig": sig_star(p),
            "N": int(model.nobs),
            "R2": f"{model.rsquared:.4f}",
        })

# Add baseline (Model 3) and knowledge_intensity variant
if m3 is not None and target_var in m3.params.index:
    base_betas = compute_standardised_betas(df, ALL_PREDICTORS, DV_PRIMARY)
    tab7_rows.insert(0, {
        "Specification": "Baseline (Model 3)",
        "B": f"{m3.params[target_var]:.4f}",
        "SE": f"{m3.bse[target_var]:.4f}",
        "Beta": (f"{base_betas[target_var]['beta']:.4f}"
                 if base_betas and target_var in base_betas else "—"),
        "p": f"{m3.pvalues[target_var]:.4f}",
        "sig": sig_star(m3.pvalues[target_var]),
        "N": int(m3.nobs),
        "R2": f"{m3.rsquared:.4f}",
    })

# Add Model 1 (knowledge only)
if m1 is not None and target_var in m1.params.index:
    m1_betas = compute_standardised_betas(df, KNOWLEDGE_VARS + CONTROLS, DV_PRIMARY)
    tab7_rows.insert(1, {
        "Specification": "Knowledge only (Model 1)",
        "B": f"{m1.params[target_var]:.4f}",
        "SE": f"{m1.bse[target_var]:.4f}",
        "Beta": (f"{m1_betas[target_var]['beta']:.4f}"
                 if m1_betas and target_var in m1_betas else "—"),
        "p": f"{m1.pvalues[target_var]:.4f}",
        "sig": sig_star(m1.pvalues[target_var]),
        "N": int(m1.nobs),
        "R2": f"{m1.rsquared:.4f}",
    })

# Add Heckman (Model 4) if cyt_stem_per_10k is there
if m4 is not None and target_var in m4.params.index:
    tab7_rows.append({
        "Specification": "Heckman (Model 4)",
        "B": f"{m4.params[target_var]:.4f}",
        "SE": f"{m4.bse[target_var]:.4f}",
        "Beta": "—",
        "p": f"{m4.pvalues[target_var]:.4f}",
        "sig": sig_star(m4.pvalues[target_var]),
        "N": int(m4.nobs),
        "R2": f"{m4.rsquared:.4f}",
    })

pd.DataFrame(tab7_rows).to_csv(TAB_DIR / "table_07_robustness.csv",
                                index=False)
print(f"  Saved: table_07_robustness.csv")

# --- Figure S1: Coefficient stability plot ---
fig_s1, ax_s1 = plt.subplots(figsize=(10, 6))

stability_points = []

# Collect beta of cyt_stem_per_10k across specifications
spec_labels = []

# Model 1 (knowledge only)
if m1 is not None:
    b1 = compute_standardised_betas(df, KNOWLEDGE_VARS + CONTROLS, DV_PRIMARY)
    if b1 and target_var in b1:
        stability_points.append(b1[target_var])
        spec_labels.append("M1: Knowledge")

# Model 3 (full)
if m3 is not None:
    b3 = compute_standardised_betas(df, ALL_PREDICTORS, DV_PRIMARY)
    if b3 and target_var in b3:
        stability_points.append(b3[target_var])
        spec_labels.append("M3: Full")

# Model 4 (Heckman) — compute standardised betas on stage 2 sample
if m4 is not None:
    b4 = compute_standardised_betas(df_stage2, stage2_preds, DV_PRIMARY)
    if b4 and target_var in b4:
        stability_points.append(b4[target_var])
        spec_labels.append("M4: Heckman")

# Region FE
if m7c is not None:
    b7c = compute_standardised_betas(df_fe, fe_preds, DV_PRIMARY)
    if b7c and target_var in b7c:
        stability_points.append(b7c[target_var])
        spec_labels.append("Region FE")

# Alt DV: devs_per_10k
if m7a_devs is not None:
    b7a = compute_standardised_betas(df, ALL_PREDICTORS, "gh_devs_per_10k")
    if b7a and target_var in b7a:
        stability_points.append(b7a[target_var])
        spec_labels.append("DV: devs/10k")

# Alt DV: language diversity
if m7a_lang is not None:
    b7al = compute_standardised_betas(df, ALL_PREDICTORS,
                                       "gh_language_diversity_index")
    if b7al and target_var in b7al:
        stability_points.append(b7al[target_var])
        spec_labels.append("DV: lang div.")

if stability_points:
    y_s1 = np.arange(len(stability_points))
    betas_s1 = [sp["beta"] for sp in stability_points]
    ci_lo_s1 = [sp["ci_low"] for sp in stability_points]
    ci_hi_s1 = [sp["ci_high"] for sp in stability_points]

    for i in range(len(stability_points)):
        colour = HIST_COLOURS["eci"] if betas_s1[i] > 0 else HIST_COLOURS["devs"]
        ax_s1.errorbar(
            betas_s1[i], y_s1[i],
            xerr=[[betas_s1[i] - ci_lo_s1[i]],
                  [ci_hi_s1[i] - betas_s1[i]]],
            fmt="o", color=colour, markersize=10,
            capsize=5, capthick=1.5, linewidth=2, alpha=0.85,
        )

    ax_s1.axvline(0, color="#999999", linestyle="--", linewidth=1, zorder=1)
    ax_s1.set_yticks(y_s1)
    ax_s1.set_yticklabels(spec_labels)
    ax_s1.set_xlabel("Standardised \u03b2 of STEM researchers per 10k "
                     "(95% CI)")
    style_axis(ax_s1)
    for y_g in y_s1:
        ax_s1.axhline(y_g, color="#eeeeee", linewidth=0.5, zorder=0)

fig_s1.tight_layout()
fig_s1.savefig(FIG_DIR / "fig_S1_robustness.png", dpi=STYLE["dpi"])
print(f"  Saved: fig_S1_robustness.png")
plt.close(fig_s1)


# ============================================================
# Summary
# ============================================================
print(f"\n\n{'=' * 70}")
print(f"  ANALYSIS COMPLETE")
print(f"{'=' * 70}")
print(f"  Figures saved to: {FIG_DIR}")
print(f"  Tables saved to:  {TAB_DIR}")

if m3 is not None and m2 is not None and m1 is not None:
    print(f"\n  Key results:")
    print(f"    Model 1 (Knowledge) R\u00b2 = {m1.rsquared:.4f}")
    print(f"    Model 2 (Wealth)    R\u00b2 = {m2.rsquared:.4f}")
    print(f"    Model 3 (Full)      R\u00b2 = {m3.rsquared:.4f}")
    gain = m3.rsquared - m2.rsquared
    print(f"    R\u00b2 gain (M3 vs M2) = {gain:.4f}")

    if target_var in m3.params.index:
        p_stem = m3.pvalues[target_var]
        print(f"    cyt_stem_per_10k in M3: p = {p_stem:.4f} "
              f"{'** SIGNIFICANT' if p_stem < 0.05 else ''}")

print(f"\n  Done.\n")
