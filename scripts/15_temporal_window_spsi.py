"""
Temporal-window robustness for the SPSI (Round 2, Reviewer #2 objection on the
2008-2026 aggregation window).

Recomputes the SPSI from scratch using only repositories CREATED WITHIN a
restricted window, then re-runs the pooled OLS with the windowed index as
dependent variable. Methodology identical to the published index
(01_compute_eci.py / 09_eci_temporal_panel.py in the companion repo):
foreign-user exclusion, Cordoba/CABA geocode corrections, >=10 repos per
department, >=30 repos per language, RCA -> binary A -> method of
reflections -> second eigenvector -> standardise -> sign-align.

Requires the companion PostgreSQL DB (github_argentina.repos, created_at).

Windows:
  - 2008-2026  full window; sanity replication of the published index
               (must reproduce N=221, beta_STEM ~= 0.220)
  - 2015-2026  principal restricted window
  - 2020-2026  extreme stress window (N shrinks; reported transparently)

Outputs:
  - tables/table_S13_temporal_window.csv
  - appends to tables/revision_numbers.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import eig
from scipy.stats import pearsonr, spearmanr
from sqlalchemy import create_engine, text

PROJ = Path(__file__).resolve().parents[1]
TS_DATA = PROJ / "data" / "departamentos_master.csv"
TABLES = PROJ / "tables"
OUT_TXT = TABLES / "revision_numbers.txt"

COMPANION = PROJ.parent / "2026_1_JEG_sent" / "github-subir"
FOREIGN_CSV = COMPANION / "audit" / "audit_04_foreign_users.csv"

ENGINE_URL = "postgresql://postgres:postgres@localhost:5432/posadas"

KNOW = ["cyt_stem_per_10k", "dist_stem_uni_km", "pct_univ_adultos"]
WEALTH = ["log_wage_median", "rad_mean_2022", "pct_servicios_avanzados"]
CONTROLS = ["log_pob", "inet_penetracion_hog"]
ALL_PRED = KNOW + WEALTH + CONTROLS

MIN_DEPT_REPOS = 10
MIN_LANG_REPOS = 30

WINDOWS = [
    ("SPSI 2008-2026 (baseline replication)", "2008-01-01", "2026-12-31"),
    ("SPSI 2015-2026", "2015-01-01", "2026-12-31"),
    ("SPSI 2020-2026", "2020-01-01", "2026-12-31"),
]

# Geocode corrections, identical to 09_eci_temporal_panel.py
CORDOBA_CORRECTIONS = {
    "14112": "14119",
    "14119": "14126",
    "14126": "14133",
    "14133": "14140",
    "14140": "14147",
    "14154": "14161",
    "14175": "14182",
    "14182": "14112",
}
MAPPING = {
    "06217": "06218",
    "06466": "06218",
    "94007": "94008",
    "94014": "94015",
    "94011": "94015",
    **CORDOBA_CORRECTIONS,
}
EXCLUDE_CODES = {"94021", "94028"}
CABA_PREFIX = "02"


def apply_corrections(df):
    df = df.copy()
    df["dpto5"] = df["dpto5"].str.zfill(5)
    df["dpto5"] = df["dpto5"].apply(
        lambda x: "02000" if x.startswith(CABA_PREFIX) else MAPPING.get(x, x)
    )
    df = df[~df["dpto5"].isin(EXCLUDE_CODES)].copy()
    return df


def compute_spsi_window(engine, foreign_users, start, end):
    """Recompute the SPSI on repositories created within [start, end]."""
    query = text("""
        SELECT LEFT(redcode, 5) AS dpto5,
               primary_language,
               COUNT(*)         AS repos
        FROM   github_argentina.repos
        WHERE  primary_language IS NOT NULL
          AND  primary_language != ''
          AND  username NOT IN (SELECT unnest(:fu))
          AND  created_at >= :start
          AND  created_at <= :end
        GROUP  BY 1, 2
    """)
    df = pd.read_sql(
        query, engine,
        params={"fu": foreign_users,
                "start": f"{start}T00:00:00Z",
                "end": f"{end}T23:59:59Z"},
    )
    df = apply_corrections(df)
    df = df.groupby(["dpto5", "primary_language"], as_index=False)["repos"].sum()

    lang_totals = df.groupby("primary_language")["repos"].sum()
    valid_langs = lang_totals[lang_totals >= MIN_LANG_REPOS].index
    dept_totals = df.groupby("dpto5")["repos"].sum()
    valid_depts = dept_totals[dept_totals >= MIN_DEPT_REPOS].index
    df = df[df["primary_language"].isin(valid_langs)
            & df["dpto5"].isin(valid_depts)].copy()

    n_d = df["dpto5"].nunique()
    n_l = df["primary_language"].nunique()
    if n_d < 20 or n_l < 5:
        raise RuntimeError(f"Window {start}..{end}: too sparse "
                           f"(n_depts={n_d}, n_langs={n_l})")

    M = df.pivot_table(index="dpto5", columns="primary_language",
                       values="repos", fill_value=0)
    M_arr = M.values.astype(float)

    row_s = M_arr.sum(axis=1, keepdims=True)
    col_s = M_arr.sum(axis=0, keepdims=True)
    total = M_arr.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        RCA = (M_arr / row_s) / (col_s / total)
    RCA = np.nan_to_num(RCA, nan=0.0)
    A = (RCA >= 1).astype(float)

    diversity = A.sum(axis=1)
    ubiquity = A.sum(axis=0)
    diversity[diversity == 0] = 1e-10
    ubiquity[ubiquity == 0] = 1e-10

    D_inv = np.diag(1.0 / diversity)
    U_inv = np.diag(1.0 / ubiquity)
    M_tilde = D_inv @ A @ U_inv @ A.T
    eigenvalues, eigenvectors = eig(M_tilde)
    idx = np.argsort(-eigenvalues.real)
    eci_raw = eigenvectors[:, idx[1]].real
    eci_std = (eci_raw - eci_raw.mean()) / eci_raw.std()
    if np.corrcoef(eci_std, diversity)[0, 1] < 0:
        eci_std = -eci_std

    print(f"  {start}..{end}: {n_d} depts, {n_l} languages, "
          f"total repos={int(M_arr.sum()):,}")
    return pd.DataFrame({"dpto5": M.index, "spsi_window": eci_std}), n_d, n_l


def load_master():
    df = pd.read_csv(TS_DATA, dtype={"dpto5": str})
    df["dpto5"] = df["dpto5"].str.zfill(5)
    df["log_pob"] = np.log(df["pob_2022"].clip(lower=1))
    return df


def fit_pooled(sub, dv):
    X = sm.add_constant(sub[ALL_PRED])
    m = sm.OLS(sub[dv], X).fit(cov_type="HC1")
    y_sd = sub[dv].std(ddof=1)
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
    foreign_users = pd.read_csv(FOREIGN_CSV, usecols=["username"])["username"].tolist()
    print(f"Foreign-user exclusion list: {len(foreign_users)} users")
    engine = create_engine(ENGINE_URL)
    master = load_master()

    out = open(OUT_TXT, "a", encoding="utf-8")
    write_section(out, "8. Temporal-window SPSI robustness (Round 2, Table S13)")

    rows = []
    for label, start, end in WINDOWS:
        spsi, n_depts_idx, n_langs = compute_spsi_window(
            engine, foreign_users, start, end)
        merged = master.merge(spsi, on="dpto5", how="inner")

        # Correlation with the published index (common departments)
        common = merged.dropna(subset=["eci_software", "spsi_window"])
        r_pub, _ = pearsonr(common["eci_software"], common["spsi_window"])
        rho_pub, _ = spearmanr(common["eci_software"], common["spsi_window"])

        sub = merged.dropna(subset=["spsi_window"] + ALL_PRED).copy()
        res = fit_pooled(sub, dv="spsi_window")

        out.write(
            f"{label}: index depts = {n_depts_idx} ({n_langs} langs); "
            f"OLS N = {res['n']}, B_STEM = {res['B_stem']:.4f}, "
            f"beta_STEM = {res['beta_stem']:.3f}, p = {res['p_stem']:.4g}, "
            f"R2 = {res['R2']:.3f}; r/rho with published SPSI = "
            f"{r_pub:.3f}/{rho_pub:.3f} (n={len(common)})\n"
        )
        rows.append({
            "variant": label,
            "n_depts_index": n_depts_idx,
            "n_langs_index": n_langs,
            **res,
            "r_published": r_pub,
            "rho_published": rho_pub,
        })
        print(f"  -> OLS N={res['n']}, beta_STEM={res['beta_stem']:.3f}, "
              f"p={res['p_stem']:.4g}, R2={res['R2']:.3f}, "
              f"r_pub={r_pub:.3f}, rho_pub={rho_pub:.3f}")

    # Published-index baseline row for side-by-side comparison
    base_sub = master.dropna(subset=["eci_software"] + ALL_PRED).copy()
    base = fit_pooled(base_sub, dv="eci_software")
    out.write(
        f"Published SPSI (master, N={base['n']}): "
        f"beta_STEM = {base['beta_stem']:.3f}, p = {base['p_stem']:.4g}, "
        f"R2 = {base['R2']:.3f}\n"
    )
    rows.insert(0, {
        "variant": f"Published SPSI (N={base['n']})",
        "n_depts_index": np.nan, "n_langs_index": np.nan,
        **base, "r_published": 1.0, "rho_published": 1.0,
    })

    out_df = pd.DataFrame(rows)
    num_cols = [c for c in out_df.columns if c not in ("variant", "p_stem")]
    out_df[num_cols] = out_df[num_cols].astype(float).round(4)
    for c in ("n_depts_index", "n_langs_index", "n"):
        out_df[c] = out_df[c].astype("Int64")
    # Exact p when > 0.001, "< 0.001" otherwise (journal convention).
    out_df["p_stem"] = out_df["p_stem"].astype(float).map(
        lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}")
    out_df.to_csv(TABLES / "table_S13_temporal_window.csv", index=False)
    out.close()
    print(f"\nSaved tables/table_S13_temporal_window.csv; appended to {OUT_TXT}")


if __name__ == "__main__":
    main()
