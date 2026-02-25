"""
02_integrate_economic_indicators.py
Integrates CEP XXI economic indicators + census aggregates + satellite data
into art1.departamentos in PostgreSQL.

Key economic development proxies:
  - Median formal wage (CEP XXI) — direct income measure
  - Registered employment (CEP XXI) — labor market proxy
  - Establishments count (CEP XXI) — firm density
  - Billing variation (CEP XXI) — revenue growth
  - Census: employment, education, household goods, pensions
  - Satellite: GHSL built-up, NO2, EVI (if available)

Usage:
    python 02_integrate_economic_indicators.py
"""

import pandas as pd
import numpy as np
import sqlalchemy as sa
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CEP_DIR = DATA_DIR / "cep_xxi"
CENSO_DIR = DATA_DIR / "censo"
SAT_DIR = DATA_DIR / "satelital"
DB_URI = "postgresql://postgres:postgres@localhost:5432/posadas"


def to_dpto5(series):
    """Convert codigo_departamento_indec (float/int) to 5-char INDEC code."""
    return (
        series
        .dropna()
        .astype(float)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )


def load_cep_wages():
    """Load CEP XXI median and mean wages, compute 2022 annual average per department."""
    print("[1/5] CEP XXI wages...")

    # Median private wage
    w_med = pd.read_csv(CEP_DIR / "w_median_depto_priv.csv")
    w_med["fecha"] = pd.to_datetime(w_med["fecha"])
    w_med = w_med.dropna(subset=["codigo_departamento_indec"])
    w_med["dpto5"] = to_dpto5(w_med["codigo_departamento_indec"])

    # Annual 2022 median
    w22 = w_med[w_med["fecha"].dt.year == 2022].groupby("dpto5")["w_median"].mean().rename("wage_median_2022")

    # Also get 2020 for comparison
    w20 = w_med[w_med["fecha"].dt.year == 2020].groupby("dpto5")["w_median"].mean().rename("wage_median_2020")

    # Mean private wage
    w_mean = pd.read_csv(CEP_DIR / "w_mean_depto_priv.csv")
    w_mean["fecha"] = pd.to_datetime(w_mean["fecha"])
    w_mean = w_mean.dropna(subset=["codigo_departamento_indec"])
    w_mean["dpto5"] = to_dpto5(w_mean["codigo_departamento_indec"])
    wm22 = w_mean[w_mean["fecha"].dt.year == 2022].groupby("dpto5")["w_mean"].mean().rename("wage_mean_2022")

    result = pd.concat([w22, w20, wm22], axis=1)
    print(f"  Departments with wages: {len(result)}")
    print(f"  Median wage 2022 mean: ${result['wage_median_2022'].mean():,.0f}")
    return result


def load_cep_employment():
    """Load CEP XXI registered employment, compute 2022 annual average."""
    print("\n[2/5] CEP XXI employment...")

    # Private sector
    p = pd.read_csv(CEP_DIR / "puestos_depto_priv.csv")
    p = p.dropna(subset=["codigo_departamento_indec"])
    p["fecha"] = pd.to_datetime(p["fecha"])
    p["dpto5"] = to_dpto5(p["codigo_departamento_indec"])
    priv22 = p[p["fecha"].dt.year == 2022].groupby("dpto5")["puestos"].mean().rename("empleo_priv_2022")

    # Total employment
    t = pd.read_csv(CEP_DIR / "puestos_depto_total.csv")
    t = t.dropna(subset=["codigo_departamento_indec"])
    t["fecha"] = pd.to_datetime(t["fecha"])
    t["dpto5"] = to_dpto5(t["codigo_departamento_indec"])
    tot22 = t[t["fecha"].dt.year == 2022].groupby("dpto5")["puestos"].mean().rename("empleo_total_2022")

    # Public sector
    pub = pd.read_csv(CEP_DIR / "puestos_depto_pub.csv")
    pub = pub.dropna(subset=["codigo_departamento_indec"])
    pub["fecha"] = pd.to_datetime(pub["fecha"])
    pub["dpto5"] = to_dpto5(pub["codigo_departamento_indec"])
    pub22 = pub[pub["fecha"].dt.year == 2022].groupby("dpto5")["puestos"].mean().rename("empleo_pub_2022")

    result = pd.concat([priv22, pub22, tot22], axis=1)
    print(f"  Departments with employment data: {len(result)}")
    print(f"  Mean private employment: {result['empleo_priv_2022'].mean():,.0f}")
    return result


def load_cep_establishments():
    """Load CEP XXI productive establishments, aggregate to department."""
    print("\n[3/5] CEP XXI establishments...")

    est = pd.read_csv(CEP_DIR / "establecimientos_depto.csv")
    est = est.dropna(subset=["in_departamentos"])
    est["dpto5"] = to_dpto5(est["in_departamentos"])

    # Total establishments and employment per department (latest year)
    latest_year = est["anio"].max()
    est_latest = est[est["anio"] == latest_year]

    agg = est_latest.groupby("dpto5").agg(
        establecimientos=("Establecimientos", "sum"),
        empleo_establ=("Empleo", "sum"),
        empresas_export=("empresas_exportadoras", "sum"),
    ).rename(columns={
        "establecimientos": f"establ_total_{latest_year}",
        "empleo_establ": f"establ_empleo_{latest_year}",
        "empresas_export": f"establ_export_{latest_year}",
    })

    # Also compute: share of service-sector employment (letters J-N: info, finance, professional)
    service_letters = ["J", "K", "L", "M", "N"]
    services = est_latest[est_latest["letra"].isin(service_letters)]
    svc_emp = services.groupby("dpto5")["Empleo"].sum().rename("empleo_servicios_avanzados")

    total_emp = est_latest.groupby("dpto5")["Empleo"].sum().rename("empleo_total_establ")
    pct_svc = (svc_emp / total_emp * 100).rename("pct_servicios_avanzados").round(2)

    result = agg.join(pct_svc, how="left").fillna({"pct_servicios_avanzados": 0})
    print(f"  Departments: {len(result)}, year: {latest_year}")
    return result


def load_cep_billing():
    """Load CEP XXI billing variation, compute 2022 annual mean."""
    print("\n[4/5] CEP XXI billing variation...")

    fact = pd.read_csv(CEP_DIR / "fact_por_depto.csv")
    fact = fact.dropna(subset=["codigo_departamento_indec"])
    fact["fecha"] = pd.to_datetime(fact["fecha"])
    fact["dpto5"] = to_dpto5(fact["codigo_departamento_indec"])

    # Mean billing variation for 2022
    f22 = fact[fact["fecha"].dt.year == 2022].groupby("dpto5")["var_facturacion"].mean().rename("fact_var_2022")

    # Mean billing variation for 2023 (if available)
    f23 = fact[fact["fecha"].dt.year == 2023].groupby("dpto5")["var_facturacion"].mean().rename("fact_var_2023")

    result = pd.concat([f22, f23], axis=1)
    print(f"  Departments with billing: {len(result)}")
    return result


def load_census_aggregates():
    """Load pre-aggregated census variables."""
    print("\n[5/5] Census aggregates...")

    dfs = {}
    for fname in ["censo_empleo_depto.csv", "censo_bienes_depto.csv",
                   "censo_educacion_depto.csv", "censo_jubilaciones_depto.csv"]:
        fpath = CENSO_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, dtype={"dpto5": str})
            df["dpto5"] = df["dpto5"].str.zfill(5)
            df = df.set_index("dpto5")
            dfs[fname] = df
            print(f"  {fname}: {len(df)} departments, {len(df.columns)} vars")

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs.values(), axis=1)

    # Compute useful rates
    if "c22_adultos_univ" in result.columns and "c22_adultos_total" in result.columns:
        result["pct_univ_adultos"] = (
            result["c22_adultos_univ"] / result["c22_adultos_total"] * 100
        ).round(2)

    if "c22_cobertura_salud" in result.columns and "c22_pob_educ" in result.columns:
        result["pct_cobertura_salud"] = (
            result["c22_cobertura_salud"] / result["c22_pob_educ"] * 100
        ).round(2)

    if "c22_ocupados" in result.columns and "c22_activos" in result.columns:
        result["tasa_empleo_censo"] = (
            result["c22_ocupados"] / result["c22_activos"] * 100
        ).round(2)

    if "c22_h_computadora" in result.columns and "c22_hogares" in result.columns:
        result["pct_computadora"] = (
            result["c22_h_computadora"] / result["c22_hogares"] * 100
        ).round(2)

    return result


def load_satellite():
    """Load satellite indicators if available."""
    sat_file = SAT_DIR / "gee_indicators_depto.csv"
    if sat_file.exists():
        df = pd.read_csv(sat_file, dtype={"dpto5": str})
        df["dpto5"] = df["dpto5"].str.zfill(5)
        df = df.set_index("dpto5")
        print(f"\n[SAT] Loaded {len(df)} departments, {len(df.columns)} satellite vars")
        return df
    else:
        print("\n[SAT] gee_indicators_depto.csv not found yet (GEE still running?)")
        return pd.DataFrame()


def integrate():
    """Merge all and load to PostgreSQL."""
    engine = sa.create_engine(DB_URI)

    # Load existing
    existing = pd.read_sql(
        "SELECT dpto5, pob_2022 FROM art1.departamentos",
        engine,
    ).set_index("dpto5")
    print(f"\nExisting departments: {len(existing)}")

    # Load all sources
    wages = load_cep_wages()
    employment = load_cep_employment()
    establishments = load_cep_establishments()
    billing = load_cep_billing()
    census = load_census_aggregates()
    satellite = load_satellite()

    # Merge
    combined = existing.copy()
    for src in [wages, employment, establishments, billing, census, satellite]:
        if len(src) > 0:
            combined = combined.join(src, how="left")

    # Compute per-capita rates for key variables
    pop = combined["pob_2022"].astype(float)

    if "empleo_priv_2022" in combined.columns:
        combined["empleo_priv_per_10k"] = np.where(
            pop > 0, combined["empleo_priv_2022"] / pop * 10000, 0
        ).round(2)

    if "wage_median_2022" in combined.columns:
        # Log wage for regressions
        combined["log_wage_median"] = np.log1p(combined["wage_median_2022"]).round(4)

    if "ghsl_built_surface_m2" in combined.columns:
        # Built-up per capita (m2/person)
        combined["built_up_per_cap"] = np.where(
            pop > 0, combined["ghsl_built_surface_m2"] / pop, 0
        ).round(2)

    # Drop pob_2022 (already exists in DB)
    new_cols = [c for c in combined.columns if c != "pob_2022"]

    # Summary
    print("\n=== New Economic Variables ===")
    for col in new_cols:
        s = combined[col].dropna()
        if len(s) > 0:
            print(f"  {col:35s} n={len(s):4d}  mean={s.mean():12.2f}  "
                  f"std={s.std():12.2f}")

    # Update PostgreSQL
    print(f"\n[DB] Adding {len(new_cols)} columns to art1.departamentos...")

    with engine.begin() as conn:
        for col in new_cols:
            dtype = combined[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                pg_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                pg_type = "DOUBLE PRECISION"
            else:
                pg_type = "DOUBLE PRECISION"

            conn.execute(sa.text(
                f"ALTER TABLE art1.departamentos ADD COLUMN IF NOT EXISTS "
                f"{col} {pg_type}"
            ))

        for dpto5, row in combined.iterrows():
            set_clauses = []
            params = {"dpto5": dpto5}
            for col in new_cols:
                val = row[col]
                if pd.isna(val):
                    set_clauses.append(f"{col} = NULL")
                else:
                    param_name = f"v_{col}"
                    set_clauses.append(f"{col} = :{param_name}")
                    params[param_name] = float(val)

            if set_clauses:
                sql = f"UPDATE art1.departamentos SET {', '.join(set_clauses)} WHERE dpto5 = :dpto5"
                conn.execute(sa.text(sql), params)

    # Recreate the view to pick up new columns
    print("[DB] Recreating departamentos_geo view...")
    with engine.begin() as conn:
        conn.execute(sa.text("DROP VIEW IF EXISTS art1.departamentos_geo"))
        conn.execute(sa.text("""
            CREATE VIEW art1.departamentos_geo AS
            WITH geo_mapped AS (
                SELECT g.redcode AS geo_redcode,
                       g.nombre AS geo_nombre,
                       g.nombre_completo AS geo_nombre_completo,
                       g.geometry,
                       CASE
                           WHEN g.redcode LIKE '02%' THEN '02000'
                           WHEN g.redcode = '06466' THEN '06218'
                           WHEN g.redcode = '94011' THEN '94015'
                           WHEN g.redcode IN ('94021','94028') THEN NULL
                           ELSE g.redcode
                       END AS art1_dpto5
                FROM departamentos_argentina g
            )
            SELECT gm.geo_redcode, gm.geo_nombre, gm.geo_nombre_completo, gm.geometry, d.*
            FROM geo_mapped gm
            LEFT JOIN art1.departamentos d ON gm.art1_dpto5 = d.dpto5
        """))

    # Save CSV
    out_csv = DATA_DIR / "economic_indicators.csv"
    combined.to_csv(out_csv, index=True)
    print(f"\n[DONE] {len(new_cols)} economic variables integrated.")
    print(f"  CSV saved: {out_csv}")


if __name__ == "__main__":
    integrate()
