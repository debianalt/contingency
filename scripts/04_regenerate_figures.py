"""
04_regenerate_figures.py — Regenerate all article figures with improved styling.

Changes from 03_analysis.py figure generation:
  - High-contrast, colorblind-friendly palette (Tol Bright + custom)
  - Thicker lines in forest plot (fig_06)
  - Two new choropleth maps of Argentina (fig_08, fig_09)
  - All other figures updated with new palette

Run:  cd scripts && python 04_regenerate_figures.py
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
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import matplotlib.transforms as transforms
import matplotlib.patheffects as pe

# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
FIG_DIR.mkdir(exist_ok=True)

STYLE = {
    "title_size": 16,
    "axis_label_size": 14,
    "tick_size": 12,
    "annot_size": 12,
    "legend_size": 13,
    "small_annot_size": 12,
    "panel_label_size": 16,
    "dpi": 300,
}

# ---------------------------------------------------------------
# HIGH-CONTRAST COLORBLIND-FRIENDLY PALETTE
# Based on Paul Tol's Bright scheme + manually tuned for 6 types
# ---------------------------------------------------------------
CLUSTER_COLOURS = {
    1: "#CC3311",   # vermillion     (Peripheral-Deprived)
    2: "#0077BB",   # strong blue    (Metropolitan-Core)
    3: "#009988",   # teal           (Metropolitan-Diversified)
    4: "#AA3377",   # magenta/wine   (Pampeana-Educated)
    5: "#EE7733",   # orange         (Semi-Rural-Active)
    6: "#BBBBBB",   # neutral grey   (Intermediate-Urban)
}

# Histogram palette — high contrast
HIST_COLOURS = {
    "eci": "#0077BB",
    "devs": "#CC3311",
    "stem": "#009988",
    "wage": "#AA3377",
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
    despine(ax)
    if grid:
        ax.grid(True, alpha=0.15, linewidth=0.5, color="#888888")
        ax.set_axisbelow(True)


def sig_star(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    elif p < 0.10: return "\u2020"
    else: return ""


def draw_confidence_ellipse(x, y, ax, n_std=1.5, facecolor="none", **kwargs):
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


def compute_standardised_betas(data, predictors, dv):
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


# ============================================================
# Load data
# ============================================================
DATA_PATH = ROOT / "data" / "departamentos_master.csv"
GEOJSON_PATH = ROOT / "data" / "satelital" / "departamentos_polygons.geojson"

df = pd.read_csv(DATA_PATH)
df["log_pob"] = np.log(df["pob_2022"].replace(0, np.nan))
df["has_eci"] = df["eci_software"].notna().astype(int)

KNOWLEDGE_VARS = ["cyt_stem_per_10k", "dist_stem_uni_km", "pct_univ_adultos"]
WEALTH_VARS = ["log_wage_median", "rad_mean_2022", "pct_servicios_avanzados"]
CONTROLS = ["log_pob", "inet_penetracion_hog"]
ALL_PREDICTORS = KNOWLEDGE_VARS + WEALTH_VARS + CONTROLS
DV_PRIMARY = "eci_software"
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
}

cluster_labels = {}
if "mca_cluster_label" in df.columns:
    for c in df["mca_cluster"].dropna().unique():
        cluster_labels[int(c)] = df.loc[
            df["mca_cluster"] == c, "mca_cluster_label"
        ].iloc[0]

# Sorted cluster order for consistent plotting
clusters_sorted = sorted(cluster_labels.keys())

print(f"\n{'=' * 70}")
print(f"  FIGURE REGENERATION — High-contrast palette + choropleths")
print(f"{'=' * 70}")
print(f"  Loaded {len(df)} departments")
print(f"  Clusters: {cluster_labels}")


# ============================================================
# Figure 1: Distribution panels (2x2)
# ============================================================
print("\n  Generating fig_01_distributions...")

df_eci = df[df["eci_software"].notna()]
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))

plot_info = [
    ("eci_software", "ECI (software)", df_eci, HIST_COLOURS["eci"]),
    ("gh_devs_per_10k", "Developers per 10,000 inhab.",
     df[df["gh_devs_per_10k"] > 0], HIST_COLOURS["devs"]),
    ("cyt_stem_per_10k", "STEM researchers per 10,000 inhab.",
     df, HIST_COLOURS["stem"]),
    ("log_wage_median", "ln(Median wage)", df, HIST_COLOURS["wage"]),
]

for ax, (var, label, data, colour) in zip(axes1.flat, plot_info):
    vals = data[var].dropna()
    ax.hist(vals, bins=30, color=colour, alpha=0.50, edgecolor="white",
            linewidth=0.5, density=True)
    vals.plot.kde(ax=ax, color=colour, linewidth=2.5)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.text(0.95, 0.95, f"N = {len(vals)}\n\u03bc = {vals.mean():.2f}\n"
            f"\u03c3 = {vals.std():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=STYLE["small_annot_size"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc",
                      alpha=0.9, linewidth=0.5))
    style_axis(ax)

for ax, letter in zip(axes1.flat, ["(a)", "(b)", "(c)", "(d)"]):
    ax.text(-0.02, 1.05, letter, transform=ax.transAxes,
            fontsize=STYLE["panel_label_size"], fontweight="bold", va="bottom")

fig1.tight_layout()
fig1.savefig(FIG_DIR / "fig_01_distributions.png", dpi=STYLE["dpi"])
print(f"    Saved: fig_01_distributions.png")
plt.close(fig1)


# ============================================================
# Figure 3: Horse-race scatter (2x2)
# ============================================================
print("  Generating fig_03_horse_race_scatter...")

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
    plot_df_sc = df.dropna(subset=[xvar, yvar, "mca_cluster"]).copy()

    for c in sorted(plot_df_sc["mca_cluster"].unique()):
        c_int = int(c)
        mask = plot_df_sc["mca_cluster"] == c
        label = cluster_labels.get(c_int, f"Type {c_int}")
        ax.scatter(
            plot_df_sc.loc[mask, xvar], plot_df_sc.loc[mask, yvar],
            c=CLUSTER_COLOURS.get(c_int, "grey"), s=35, alpha=0.7,
            label=label, edgecolors="white", linewidths=0.4,
        )

    clean = plot_df_sc[[xvar, yvar]].dropna()
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
        ax.plot(x_line, y_line, color="#222222", linewidth=2.0, zorder=5)
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

handles, labels = axes3[0, 0].get_legend_handles_labels()
fig3.legend(handles, labels, loc="lower center", ncol=3,
            frameon=False, fontsize=STYLE["legend_size"],
            columnspacing=1.5, handletextpad=0.5,
            bbox_to_anchor=(0.5, -0.01))
fig3.tight_layout(rect=[0, 0.07, 1, 1])
fig3.savefig(FIG_DIR / "fig_03_horse_race_scatter.png", dpi=STYLE["dpi"])
print(f"    Saved: fig_03_horse_race_scatter.png")
plt.close(fig3)


# ============================================================
# Figure 5: MCA plane with knowledge gradient + cluster ellipses
# ============================================================
print("  Generating fig_05_mca_knowledge...")

fig5, ax5 = plt.subplots(figsize=(12, 9))

mca_cols = ["mca_dim1", "mca_dim2"]
df_mca = df.dropna(subset=mca_cols + ["knowledge_intensity"]).copy()

sc5 = ax5.scatter(
    df_mca["mca_dim1"], df_mca["mca_dim2"],
    c=df_mca["knowledge_intensity"], cmap="viridis", s=40, alpha=0.7,
    edgecolors="none", zorder=2,
)
cbar5 = plt.colorbar(sc5, ax=ax5, shrink=0.70, pad=0.02)
cbar5.set_label("Knowledge intensity (composite)",
                fontsize=STYLE["axis_label_size"])
cbar5.ax.tick_params(labelsize=STYLE["tick_size"])

# Manual label offsets (in points) to prevent overlapping
LABEL_OFFSETS = {
    1: (30, -25),    # Peripheral-Deprived: down-right
    2: (0, -20),     # Metropolitan-Core: below centroid
    3: (-25, -25),   # Metropolitan-Diversified: down-left
    4: (-45, 25),    # Pampeana-Educated: up-left (away from Intermediate)
    5: (30, 15),     # Semi-Rural-Active: up-right
    6: (40, 25),     # Intermediate-Urban: up-right (away from Pampeana)
}

for c in sorted(df_mca["mca_cluster"].dropna().unique()):
    c_int = int(c)
    mask = df_mca["mca_cluster"] == c
    label = cluster_labels.get(c_int, f"Type {c_int}")
    colour = CLUSTER_COLOURS.get(c_int, "grey")
    x_vals = df_mca.loc[mask, "mca_dim1"].values
    y_vals = df_mca.loc[mask, "mca_dim2"].values
    draw_confidence_ellipse(
        x_vals, y_vals, ax5, n_std=1.5,
        edgecolor=colour, linewidth=3.0, linestyle="-", alpha=0.9, zorder=3,
    )
    cx, cy = np.mean(x_vals), np.mean(y_vals)
    offset = LABEL_OFFSETS.get(c_int, (0, 0))
    ann_kwargs = dict(
        fontsize=STYLE["small_annot_size"], fontweight="bold",
        ha="center", va="center", color=colour,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.90,
                  edgecolor=colour, linewidth=1.5),
        zorder=4,
    )
    if offset[0]**2 + offset[1]**2 > 400:
        ann_kwargs["arrowprops"] = dict(
            arrowstyle="-", color=colour, lw=1.2,
            connectionstyle="arc3,rad=0.1")
    ax5.annotate(
        label, (cx, cy),
        xytext=offset, textcoords="offset points",
        **ann_kwargs,
    )

ax5.set_xlabel("MCA Dimension 1")
ax5.set_ylabel("MCA Dimension 2")
ax5.axhline(0, color="#bbbbbb", linewidth=0.6, linestyle="--", zorder=1)
ax5.axvline(0, color="#bbbbbb", linewidth=0.6, linestyle="--", zorder=1)
style_axis(ax5, grid=False)

fig5.tight_layout()
fig5.savefig(FIG_DIR / "fig_05_mca_knowledge.png", dpi=STYLE["dpi"])
print(f"    Saved: fig_05_mca_knowledge.png")
plt.close(fig5)


# ============================================================
# Figure 6: Forest plot — THICKER lines + high-contrast palette
# ============================================================
print("  Generating fig_06_forest_plot...")

# Compute betas: pooled
pooled_betas = compute_standardised_betas(df, TYPE_PREDS, DV_PRIMARY)

# Compute betas: type-specific
type_betas = {}
for c in clusters_sorted:
    c_int = int(c)
    df_c = df[df["mca_cluster"] == c]
    n_eci = df_c[DV_PRIMARY].notna().sum()
    if n_eci >= 30:
        type_betas[c_int] = compute_standardised_betas(df_c, TYPE_PREDS, DV_PRIMARY)
    else:
        type_betas[c_int] = None

plot_data = []
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

plot_df_forest = pd.DataFrame(plot_data)

if len(plot_df_forest) > 0:
    variables = list(dict.fromkeys(plot_df_forest["variable"]))
    types = list(dict.fromkeys(plot_df_forest["type"]))
    n_vars = len(variables)
    n_types = len(types)

    fig6, ax6 = plt.subplots(figsize=(12, max(7, n_vars * 1.8)))

    # Colour assignment: black for pooled, cluster colours for types
    forest_colours = ["#222222"]  # Pooled = near-black
    for c in clusters_sorted:
        c_int = int(c)
        if c_int in type_betas and type_betas[c_int] is not None:
            forest_colours.append(CLUSTER_COLOURS.get(c_int, "grey"))
    offsets = np.linspace(-0.3, 0.3, n_types)

    for j, type_name in enumerate(types):
        sub = plot_df_forest[plot_df_forest["type"] == type_name]
        colour = forest_colours[j % len(forest_colours)]
        for _, row in sub.iterrows():
            var_idx = variables.index(row["variable"])
            y_pos = n_vars - 1 - var_idx + offsets[j]
            is_pooled = type_name == "Pooled"
            marker = "D" if is_pooled else "o"
            size = 14 if is_pooled else 11

            ax6.errorbar(
                row["beta"], y_pos,
                xerr=[[row["beta"] - row["ci_low"]],
                      [row["ci_high"] - row["beta"]]],
                fmt=marker, color=colour, markersize=size,
                capsize=6, capthick=2.5, linewidth=3.5,
                alpha=0.90, markeredgecolor="white",
                markeredgewidth=0.8 if not is_pooled else 0,
                zorder=5 if is_pooled else 4,
            )

    ax6.axvline(0, color="#999999", linestyle="--", linewidth=1.2, zorder=1)
    ax6.set_yticks(range(n_vars))
    ax6.set_yticklabels(list(reversed(variables)),
                        fontsize=STYLE["axis_label_size"])
    ax6.set_xlabel("Standardised coefficient (\u03b2) with 95% CI",
                   fontsize=STYLE["axis_label_size"])
    style_axis(ax6)

    # Alternating background bands for readability
    for y_g in range(n_vars):
        if y_g % 2 == 0:
            ax6.axhspan(y_g - 0.5, y_g + 0.5, color="#f0f0f0", zorder=0)
        # Divider lines between variable categories
        if y_g > 0:
            ax6.axhline(y_g - 0.5, color="#cccccc", linewidth=0.8,
                         linestyle="-", zorder=1)

    # Legend
    legend_handles = []
    for j, type_name in enumerate(types):
        colour = forest_colours[j % len(forest_colours)]
        marker = "D" if type_name == "Pooled" else "o"
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=colour, markersize=13,
                   markeredgecolor="white" if marker == "o" else colour,
                   markeredgewidth=0.8 if marker == "o" else 0,
                   label=type_name))
    ax6.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False,
               fontsize=STYLE["legend_size"], columnspacing=1.5,
               handletextpad=0.5)

    fig6.tight_layout(rect=[0, 0.15, 1, 1])
    fig6.savefig(FIG_DIR / "fig_06_forest_plot.png", dpi=STYLE["dpi"])
    print(f"    Saved: fig_06_forest_plot.png")
    plt.close(fig6)


# ============================================================
# Figure S1: Robustness coefficient stability
# ============================================================
print("  Generating fig_S1_robustness...")

rob_csv = TAB_DIR / "table_07_robustness.csv"
if rob_csv.exists():
    rob_df = pd.read_csv(rob_csv)
    rob_df["Beta"] = pd.to_numeric(rob_df["Beta"], errors="coerce")
    rob_df["SE"] = pd.to_numeric(rob_df["SE"], errors="coerce")
    rob_df["p"] = pd.to_numeric(rob_df["p"], errors="coerce")
    rob_df = rob_df.dropna(subset=["Beta"])
    fig_s1, ax_s1 = plt.subplots(figsize=(10, 5))

    y_pos_r = np.arange(len(rob_df))
    betas_r = rob_df["Beta"].values
    # Approximate CIs: beta ± 1.96 * (SE_unstd / SD_y)
    # Since Beta is already standardised, use B and SE to get rough CI
    b_raw = rob_df["B"].values
    se_raw = rob_df["SE"].values
    # CI on standardised scale: beta ± 1.96 * (se_raw * sd_x / sd_y)
    # Simpler: assume beta_se ≈ |beta/t| where t = B/SE
    t_vals = b_raw / se_raw
    beta_se = np.abs(betas_r / t_vals)
    cis_low = betas_r - 1.96 * beta_se
    cis_high = betas_r + 1.96 * beta_se

    colours_r = ["#0077BB" if p < 0.05 else "#BBBBBB"
                 for p in rob_df["p"].values]

    for i in range(len(rob_df)):
        ax_s1.errorbar(
            betas_r[i], i,
            xerr=[[betas_r[i] - cis_low[i]], [cis_high[i] - betas_r[i]]],
            fmt="o", color=colours_r[i], markersize=10,
            capsize=5, capthick=2.0, linewidth=2.5, alpha=0.85,
        )

    ax_s1.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax_s1.set_yticks(y_pos_r)
    ax_s1.set_yticklabels(rob_df["Specification"].values,
                          fontsize=STYLE["small_annot_size"])
    ax_s1.set_xlabel("Standardised \u03b2 (STEM researchers per 10k)")
    style_axis(ax_s1)

    fig_s1.tight_layout()
    fig_s1.savefig(FIG_DIR / "fig_S1_robustness.png", dpi=STYLE["dpi"])
    print(f"    Saved: fig_S1_robustness.png")
    plt.close(fig_s1)
else:
    print(f"    SKIP: fig_S1 — {rob_csv} not found")


# ============================================================
# Figure 2 (heatmap): regenerate with new palette
# ============================================================
print("  Generating fig_02_correlation_heatmap...")

import seaborn as sns

predictors_for_corr = KNOWLEDGE_VARS + WEALTH_VARS + CONTROLS
outcomes_for_corr = ["eci_software", "gh_devs_per_10k",
                     "gh_language_diversity_index"]
corr_vars = predictors_for_corr + outcomes_for_corr
corr_labels = [VAR_LABELS.get(v, v) for v in corr_vars]

df_corr = df[corr_vars].dropna()
corr_matrix = df_corr.corr()
corr_matrix.index = corr_labels
corr_matrix.columns = corr_labels

fig2, ax2 = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax2,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"})
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right",
                    fontsize=STYLE["small_annot_size"])
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0,
                    fontsize=STYLE["small_annot_size"])

fig2.tight_layout()
fig2.savefig(FIG_DIR / "fig_02_correlation_heatmap.png", dpi=STYLE["dpi"])
print(f"    Saved: fig_02_correlation_heatmap.png")
plt.close(fig2)


# ============================================================
# CHOROPLETH MAPS — 6 panels, one per territorial type
# ============================================================
print("\n  Generating choropleth maps...")

try:
    import geopandas as gpd
    USE_GPD = True
except ImportError:
    USE_GPD = False

if GEOJSON_PATH.exists() and USE_GPD:
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf["dpto5"] = gdf["dpto5"].astype(str).str.zfill(5)
    df["dpto5_str"] = df["dpto5"].astype(str).str.zfill(5)
    gdf = gdf.merge(df[["dpto5_str", "mca_cluster", "mca_cluster_label",
                        "eci_software", "cyt_stem_per_10k", "gh_devs_per_10k"]],
                    left_on="dpto5", right_on="dpto5_str", how="left")

    # -------------------------------------------------------
    # Figure 8: 6-panel choropleth — one per territorial type
    # -------------------------------------------------------
    print("  Generating fig_08_choropleth_typology (6 panels)...")

    # Panel order: follow the gradient from most deprived to most dense
    panel_order = [1, 5, 6, 4, 3, 2]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    fig8, axes8 = plt.subplots(2, 3, figsize=(12, 16))
    axes8_flat = axes8.flatten()

    for idx, (c_int, plabel) in enumerate(zip(panel_order, panel_labels)):
        ax = axes8_flat[idx]
        lbl = cluster_labels.get(c_int, f"Type {c_int}")
        colour = CLUSTER_COLOURS[c_int]
        n_c = (gdf["mca_cluster"] == c_int).sum()
        n_eci = gdf.loc[gdf["mca_cluster"] == c_int, "eci_software"].notna().sum()

        # Base: all departments in very light grey
        gdf.plot(ax=ax, color="#F0F0F0", edgecolor="#CCCCCC",
                 linewidth=0.15, aspect="equal")
        # Highlight this type
        mask = gdf["mca_cluster"] == c_int
        if mask.sum() > 0:
            gdf[mask].plot(ax=ax, color=colour, edgecolor="#444444",
                          linewidth=0.3, aspect="equal")

        ax.set_axis_off()
        ax.set_title(f"{plabel} {lbl}\nN = {n_c}  |  ECI coverage: {n_eci}",
                     fontsize=STYLE["title_size"], fontweight="bold",
                     pad=10, color=colour)

    fig8.tight_layout(h_pad=3.0, w_pad=1.5)
    fig8.savefig(FIG_DIR / "fig_08_choropleth_typology.png", dpi=STYLE["dpi"])
    print(f"    Saved: fig_08_choropleth_typology.png")
    plt.close(fig8)

    # -------------------------------------------------------
    # Figure 9: Choropleth — ECI software + STEM density (2 panels)
    # -------------------------------------------------------
    print("  Generating fig_09_choropleth_eci_stem...")

    fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(12, 10))

    # Compute shared geographic extent for consistent framing
    xmin, ymin, xmax, ymax = gdf.total_bounds
    x_margin = (xmax - xmin) * 0.02
    y_margin = (ymax - ymin) * 0.02

    # Panel (a): ECI software
    cmap_eci = plt.cm.RdYlBu_r
    gdf_eci = gdf.copy()
    has_eci = gdf_eci["eci_software"].notna()

    gdf_eci.plot(ax=ax9a, color="#EEEEEE", edgecolor="#999999",
                 linewidth=0.15, aspect="equal")
    if has_eci.sum() > 0:
        vmin_eci = gdf_eci.loc[has_eci, "eci_software"].quantile(0.02)
        vmax_eci = gdf_eci.loc[has_eci, "eci_software"].quantile(0.98)
        gdf_eci[has_eci].plot(ax=ax9a, column="eci_software",
                              cmap=cmap_eci, edgecolor="#555555",
                              linewidth=0.15, legend=True,
                              vmin=vmin_eci, vmax=vmax_eci,
                              aspect="equal",
                              legend_kwds={
                                  "label": "ECI software",
                                  "shrink": 0.6,
                                  "orientation": "horizontal",
                                  "pad": 0.04,
                              })

    ax9a.set_xlim(xmin - x_margin, xmax + x_margin)
    ax9a.set_ylim(ymin - y_margin, ymax + y_margin)
    ax9a.set_axis_off()
    ax9a.set_title("(a) Software complexity (ECI)",
                   fontsize=STYLE["title_size"], fontweight="bold", pad=12)

    # Panel (b): STEM researchers per 10k
    cmap_stem = plt.cm.YlGnBu
    has_stem = gdf["cyt_stem_per_10k"].notna()
    vmax_stem = gdf.loc[has_stem, "cyt_stem_per_10k"].quantile(0.95)

    gdf.plot(ax=ax9b, color="#EEEEEE", edgecolor="#999999",
             linewidth=0.15, aspect="equal")
    if has_stem.sum() > 0:
        gdf[has_stem].plot(ax=ax9b, column="cyt_stem_per_10k",
                           cmap=cmap_stem, edgecolor="#555555",
                           linewidth=0.15, legend=True,
                           vmin=0, vmax=vmax_stem, aspect="equal",
                           legend_kwds={
                               "label": "STEM researchers per 10k",
                               "shrink": 0.6,
                               "orientation": "horizontal",
                               "pad": 0.04,
                           })

    ax9b.set_xlim(xmin - x_margin, xmax + x_margin)
    ax9b.set_ylim(ymin - y_margin, ymax + y_margin)
    ax9b.set_axis_off()
    ax9b.set_title("(b) STEM research density",
                   fontsize=STYLE["title_size"], fontweight="bold", pad=12)

    fig9.tight_layout(w_pad=2.0)
    fig9.savefig(FIG_DIR / "fig_09_choropleth_eci_stem.png", dpi=STYLE["dpi"])
    print(f"    Saved: fig_09_choropleth_eci_stem.png")
    plt.close(fig9)

else:
    if not GEOJSON_PATH.exists():
        print(f"    WARNING: GeoJSON not found at {GEOJSON_PATH}")
    if not USE_GPD:
        print("    WARNING: geopandas not installed — choropleth maps skipped")


# ============================================================
# RESULTS FIGURES — additional visualisations
# ============================================================
print("\n  Generating results figures...")

# -------------------------------------------------------
# Figure 10: STEM β vs Wages β comparison by type (grouped bars)
# -------------------------------------------------------
print("  Generating fig_10_beta_comparison...")

# Collect betas for STEM and wages per type + pooled
beta_comp = []
if pooled_betas:
    beta_comp.append({
        "type": "Pooled", "colour": "#222222",
        "stem_beta": pooled_betas["cyt_stem_per_10k"]["beta"],
        "stem_p": pooled_betas["cyt_stem_per_10k"]["p"],
        "wage_beta": pooled_betas["log_wage_median"]["beta"],
        "wage_p": pooled_betas["log_wage_median"]["p"],
    })

# Order: Metro-Div, Pampeana, Metro-Core, Intermediate (matches §4.3)
type_display_order = [3, 4, 2, 6]
for c_int in type_display_order:
    betas = type_betas.get(c_int)
    if betas:
        beta_comp.append({
            "type": cluster_labels.get(c_int, f"Type {c_int}"),
            "colour": CLUSTER_COLOURS[c_int],
            "stem_beta": betas["cyt_stem_per_10k"]["beta"],
            "stem_p": betas["cyt_stem_per_10k"]["p"],
            "wage_beta": betas["log_wage_median"]["beta"],
            "wage_p": betas["log_wage_median"]["p"],
        })

if beta_comp:
    bc_df = pd.DataFrame(beta_comp)
    n_bc = len(bc_df)
    x_bc = np.arange(n_bc)
    w_bc = 0.35

    fig10, ax10 = plt.subplots(figsize=(12, 6))

    # STEM bars
    stem_colours = [row["colour"] for _, row in bc_df.iterrows()]
    # Use hatching for wages to distinguish
    bars_stem = ax10.bar(x_bc - w_bc/2, bc_df["stem_beta"], w_bc,
                         color=stem_colours, edgecolor="white",
                         linewidth=1.0, label="STEM researchers per 10k",
                         alpha=0.90)
    bars_wage = ax10.bar(x_bc + w_bc/2, bc_df["wage_beta"], w_bc,
                         color=stem_colours, edgecolor="white",
                         linewidth=1.0, label="ln(Median wage)",
                         alpha=0.45, hatch="///")

    # Significance annotations
    for i, row in bc_df.iterrows():
        # STEM star
        y_s = row["stem_beta"]
        offset_s = 0.03 if y_s >= 0 else -0.05
        ax10.text(i - w_bc/2, y_s + offset_s, sig_star(row["stem_p"]),
                  ha="center", va="bottom" if y_s >= 0 else "top",
                  fontsize=STYLE["annot_size"], fontweight="bold")
        # Wage star
        y_w = row["wage_beta"]
        offset_w = 0.03 if y_w >= 0 else -0.05
        ax10.text(i + w_bc/2, y_w + offset_w, sig_star(row["wage_p"]),
                  ha="center", va="bottom" if y_w >= 0 else "top",
                  fontsize=STYLE["annot_size"], fontweight="bold")

    ax10.axhline(0, color="#999999", linewidth=0.8, linestyle="-")
    ax10.set_xticks(x_bc)
    ax10.set_xticklabels(bc_df["type"], fontsize=STYLE["axis_label_size"],
                         rotation=15, ha="right")
    ax10.set_ylabel("Standardised coefficient (\u03b2)",
                    fontsize=STYLE["axis_label_size"])

    # Custom legend: solid = STEM, hatched = wages
    legend_handles = [
        Patch(facecolor="#666666", alpha=0.90, label="STEM researchers per 10k"),
        Patch(facecolor="#666666", alpha=0.45, hatch="///",
              label="ln(Median wage)"),
    ]
    ax10.legend(handles=legend_handles, frameon=False,
                fontsize=STYLE["legend_size"], loc="upper left")
    style_axis(ax10)

    fig10.tight_layout()
    fig10.savefig(FIG_DIR / "fig_10_beta_comparison.png", dpi=STYLE["dpi"])
    print(f"    Saved: fig_10_beta_comparison.png")
    plt.close(fig10)


# -------------------------------------------------------
# Figure 11: R² by territorial type (horizontal bar chart)
# -------------------------------------------------------
print("  Generating fig_11_r2_by_type...")

r2_data = []
# Pooled R²
pooled_cols = TYPE_PREDS + [DV_PRIMARY]
pooled_clean = df.dropna(subset=pooled_cols)
X_pool = add_constant(pooled_clean[TYPE_PREDS])
m_pool = sm.OLS(pooled_clean[DV_PRIMARY], X_pool).fit(cov_type="HC1")
r2_data.append({"type": "Pooled", "R2": m_pool.rsquared,
                "n": int(m_pool.nobs), "colour": "#222222"})

for c_int in type_display_order:
    df_c = df[df["mca_cluster"] == c_int]
    n_eci = df_c[DV_PRIMARY].notna().sum()
    if n_eci >= 30:
        clean_c = df_c.dropna(subset=pooled_cols)
        if len(clean_c) >= len(TYPE_PREDS) + 2:
            X_c = add_constant(clean_c[TYPE_PREDS])
            m_c = sm.OLS(clean_c[DV_PRIMARY], X_c).fit(cov_type="HC1")
            r2_data.append({
                "type": cluster_labels.get(c_int, f"Type {c_int}"),
                "R2": m_c.rsquared,
                "n": int(m_c.nobs),
                "colour": CLUSTER_COLOURS[c_int],
            })

if r2_data:
    r2_df = pd.DataFrame(r2_data)

    fig11, ax11 = plt.subplots(figsize=(10, 5))
    y_r2 = np.arange(len(r2_df))

    bars = ax11.barh(y_r2, r2_df["R2"], height=0.6,
                     color=r2_df["colour"].tolist(),
                     edgecolor="white", linewidth=1.0, alpha=0.85)

    # Annotate R² values and N
    for i, row in r2_df.iterrows():
        ax11.text(row["R2"] + 0.01, i,
                  f"R\u00b2 = {row['R2']:.3f}  (n = {row['n']})",
                  va="center", fontsize=STYLE["annot_size"])

    ax11.set_yticks(y_r2)
    ax11.set_yticklabels(r2_df["type"], fontsize=STYLE["axis_label_size"])
    ax11.set_xlabel("R\u00b2", fontsize=STYLE["axis_label_size"])
    ax11.set_xlim(0, max(r2_df["R2"]) * 1.35)
    ax11.invert_yaxis()
    style_axis(ax11)

    fig11.tight_layout()
    fig11.savefig(FIG_DIR / "fig_11_r2_by_type.png", dpi=STYLE["dpi"])
    print(f"    Saved: fig_11_r2_by_type.png")
    plt.close(fig11)


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 70}")
print(f"  Done. All figures saved to {FIG_DIR}")
print(f"{'=' * 70}")
