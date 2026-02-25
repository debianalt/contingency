"""
00_build_knowledge_institutions.py
Builds knowledge-institution variables at department level for the Research Policy article.

Sources:
  - MINCYT personas_2020.csv + organizaciones_localizacion.csv + ref_DISCIPLINA.csv
    → STEM researchers per department
  - SPU universidades_geo.geojson
    → University academic units per department + distance to nearest STEM faculty
  - Padrón Educativo padron_educativo.xlsx
    → Technical schools (INET) per department

Output: new columns added to art1.departamentos in PostgreSQL (posadas)

Usage:
    python 00_build_knowledge_institutions.py
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import sqlalchemy as sa

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
DB_URI = "postgresql://postgres:postgres@localhost:5432/posadas"
SCHEMA = "art1"
TABLE = "departamentos"

# STEM gran_area_codigo: 1 = Ciencias Naturales y Exactas, 2 = Ingenierías y Tecnologías
STEM_AREAS = {1, 2}
# Researchers and fellows (excludes admin/other)
RESEARCHER_TYPES = {1, 3}  # 1=BECARIO DE I+D, 3=INVESTIGADOR

# STEM-related keywords for university faculties
STEM_KEYWORDS = [
    "ingenier", "exacta", "tecnolog", "ciencia", "comput", "inform",
    "sistemas", "fisica", "quimica", "matematica", "agronomia",
    "biolog", "bioqu", "farmacia", "agropecuaria", "politecnic",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def is_stem_faculty(name):
    """Check if a university unit name suggests STEM orientation."""
    if not name:
        return False
    name_lower = name.lower()
    return any(kw in name_lower for kw in STEM_KEYWORDS)


# ═══════════════════════════════════════════════════════════════════════════
# 1. MINCYT: STEM researchers per department
# ═══════════════════════════════════════════════════════════════════════════
def build_mincyt_indicators():
    """
    Join personas → organizaciones_localizacion → ref_DISCIPLINA.
    Build dpto5 from dpt_codigo (2-digit INDEC province) + dpa__codigo (3-digit dept).
    Compute counts per department.
    """
    print("[1/3] Loading MINCYT data...")

    # Load CSVs
    personas = pd.read_csv(
        DATA_DIR / "mincyt" / "personas_2020.csv", sep=";",
        dtype=str, low_memory=False,
    )
    # Convert key columns to numeric
    for col in ["persona_id", "institucion_trabajo_id", "tipo_personal_id",
                "categoria_conicet_id", "disciplina_experticia_id"]:
        personas[col] = pd.to_numeric(personas[col], errors="coerce")

    loc = pd.read_csv(
        DATA_DIR / "mincyt" / "organizaciones_localizacion.csv", sep=";",
        dtype=str, low_memory=False,
    )
    disc = pd.read_csv(
        DATA_DIR / "refs" / "ref_DISCIPLINA.csv", sep=";",
        dtype=str, low_memory=False,
    )
    disc["disciplina_id"] = pd.to_numeric(disc["disciplina_id"], errors="coerce")
    disc["gran_area_codigo"] = pd.to_numeric(disc["gran_area_codigo"], errors="coerce")

    print(f"  personas: {len(personas):,} rows")
    print(f"  localizaciones: {len(loc):,} rows")
    print(f"  disciplinas: {len(disc):,} rows")

    # Build dpto5 in localizacion table
    # dpt_codigo = INDEC 2-digit province code; dpa__codigo = 3-digit department code
    loc = loc.dropna(subset=["dpt_codigo", "dpa__codigo"])
    loc = loc[loc["dpt_codigo"].str.strip() != ""]
    loc = loc[loc["dpa__codigo"].str.strip() != ""]
    loc["dpto5"] = loc["dpt_codigo"].str.strip().str.zfill(2) + loc["dpa__codigo"].str.strip().str.zfill(3)
    loc["organizacion_id"] = loc["organizacion_id"].astype(int)

    # Keep one location per organization (some orgs have multiple locations)
    loc_unique = loc.drop_duplicates(subset=["organizacion_id"], keep="first")

    # Join personas → location
    merged = personas.merge(
        loc_unique[["organizacion_id", "dpto5"]],
        left_on="institucion_trabajo_id",
        right_on="organizacion_id",
        how="inner",
    )
    print(f"  Matched personas with location: {len(merged):,} / {len(personas):,}")

    # Join with discipline reference
    merged = merged.merge(
        disc[["disciplina_id", "gran_area_codigo"]],
        left_on="disciplina_experticia_id",
        right_on="disciplina_id",
        how="left",
    )

    # ── Compute indicators per department ──
    # a) Total S&T personnel
    total_cyt = merged.groupby("dpto5").size().rename("cyt_total")

    # b) Total researchers (tipo_personal_id IN 1,3)
    researchers = merged[merged["tipo_personal_id"].isin(RESEARCHER_TYPES)]
    total_researchers = researchers.groupby("dpto5").size().rename("cyt_researchers")

    # c) STEM researchers (gran_area_codigo IN 1,2 AND tipo_personal_id IN 1,3)
    stem = researchers[researchers["gran_area_codigo"].isin(STEM_AREAS)]
    stem_researchers = stem.groupby("dpto5").size().rename("cyt_stem_researchers")

    # d) CONICET members (categoria_conicet_id != -1)
    conicet = merged[merged["categoria_conicet_id"] != -1]
    conicet_count = conicet.groupby("dpto5").size().rename("cyt_conicet")

    # e) CONICET STEM
    conicet_stem = conicet[
        conicet["gran_area_codigo"].isin(STEM_AREAS) &
        conicet["tipo_personal_id"].isin(RESEARCHER_TYPES)
    ]
    conicet_stem_count = conicet_stem.groupby("dpto5").size().rename("cyt_conicet_stem")

    # f) Computer science specifically (area_codigo 1.2)
    cs_disc = disc[disc["area_codigo"] == "1.2"]["disciplina_id"]
    cs_researchers = researchers[researchers["disciplina_experticia_id"].isin(cs_disc)]
    cs_count = cs_researchers.groupby("dpto5").size().rename("cyt_cs_researchers")

    # Combine
    mincyt_df = pd.concat(
        [total_cyt, total_researchers, stem_researchers, conicet_count,
         conicet_stem_count, cs_count],
        axis=1,
    ).fillna(0).astype(int)

    print(f"  Departments with CyT data: {len(mincyt_df)}")
    print(f"  Total STEM researchers: {mincyt_df['cyt_stem_researchers'].sum():,}")
    print(f"  Total CS researchers: {mincyt_df['cyt_cs_researchers'].sum():,}")

    return mincyt_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Universities: presence and distance per department
# ═══════════════════════════════════════════════════════════════════════════
def build_university_indicators(engine):
    """
    Load GeoJSON university points, assign to nearest department centroid,
    count total and STEM faculties per department, compute distance to nearest
    STEM faculty from each department centroid.
    """
    print("\n[2/3] Loading university data...")

    # Load GeoJSON
    with open(DATA_DIR / "spu" / "universidades_geo.geojson", "r", encoding="utf-8") as f:
        gj = json.load(f)

    # Extract point data
    unis = []
    for feat in gj["features"]:
        props = feat["properties"]
        geom = feat["geometry"]
        if geom and geom["type"] == "Point":
            lon, lat = geom["coordinates"]
            unis.append({
                "universidad": props.get("universidad", ""),
                "unidad_academica": props.get("unidad_academica", ""),
                "lat": float(lat),
                "lon": float(lon),
                "is_stem": is_stem_faculty(props.get("unidad_academica", "")),
                "regimen": props.get("regimen", ""),
            })

    uni_df = pd.DataFrame(unis)
    print(f"  University units: {len(uni_df):,}")
    print(f"  STEM faculties: {uni_df['is_stem'].sum()}")

    # Get department centroids from PostGIS
    centroids = pd.read_sql(
        """
        SELECT dpto5,
               ST_X(ST_Centroid(geometry)) as centroid_lon,
               ST_Y(ST_Centroid(geometry)) as centroid_lat
        FROM art1.departamentos_geo
        """,
        engine,
    )
    print(f"  Department centroids: {len(centroids)}")

    # Assign each university to the nearest department centroid
    c_lats = centroids["centroid_lat"].values
    c_lons = centroids["centroid_lon"].values
    c_dpto5 = centroids["dpto5"].values

    uni_depts = []
    for _, u in uni_df.iterrows():
        dists = np.array([
            haversine_km(u["lat"], u["lon"], c_lats[i], c_lons[i])
            for i in range(len(c_lats))
        ])
        nearest_idx = dists.argmin()
        uni_depts.append(c_dpto5[nearest_idx])

    uni_df["dpto5"] = uni_depts

    # Count universities per department
    uni_total = uni_df.groupby("dpto5").size().rename("uni_total")
    uni_stem = uni_df[uni_df["is_stem"]].groupby("dpto5").size().rename("uni_stem")

    # Distinct universities (not just faculties)
    uni_distinct = uni_df.groupby("dpto5")["universidad"].nunique().rename("uni_distinct")

    # Distance from each department centroid to nearest STEM faculty
    stem_unis = uni_df[uni_df["is_stem"]]
    stem_lats = stem_unis["lat"].values
    stem_lons = stem_unis["lon"].values

    dist_records = []
    for _, row in centroids.iterrows():
        if len(stem_lats) == 0:
            dist_records.append({"dpto5": row["dpto5"], "dist_stem_uni_km": np.nan})
            continue
        dists = np.array([
            haversine_km(row["centroid_lat"], row["centroid_lon"],
                         stem_lats[i], stem_lons[i])
            for i in range(len(stem_lats))
        ])
        dist_records.append({
            "dpto5": row["dpto5"],
            "dist_stem_uni_km": dists.min(),
        })

    dist_df = pd.DataFrame(dist_records)
    # Deduplicate centroids (some dpto5 may appear twice in geo table)
    dist_df = dist_df.drop_duplicates(subset=["dpto5"], keep="first").set_index("dpto5")

    # Also compute distance to nearest ANY university
    all_uni_lats = uni_df["lat"].values
    all_uni_lons = uni_df["lon"].values

    dist_any = []
    for _, row in centroids.iterrows():
        dists = np.array([
            haversine_km(row["centroid_lat"], row["centroid_lon"],
                         all_uni_lats[i], all_uni_lons[i])
            for i in range(len(all_uni_lats))
        ])
        dist_any.append({
            "dpto5": row["dpto5"],
            "dist_any_uni_km": dists.min(),
        })

    dist_any_df = pd.DataFrame(dist_any)
    dist_any_df = dist_any_df.drop_duplicates(subset=["dpto5"], keep="first").set_index("dpto5")

    # Combine using join to handle different index sizes
    uni_counts = pd.DataFrame({"uni_total": uni_total, "uni_stem": uni_stem, "uni_distinct": uni_distinct})
    uni_result = dist_df.join(dist_any_df, how="outer").join(uni_counts, how="outer")
    # Fill missing counts with 0, keep distances as-is
    for col in ["uni_total", "uni_stem", "uni_distinct"]:
        if col in uni_result.columns:
            uni_result[col] = uni_result[col].fillna(0).astype(int)

    print(f"  Departments with universities: {(uni_result['uni_total'] > 0).sum()}")
    print(f"  Mean distance to STEM uni: {uni_result['dist_stem_uni_km'].mean():.1f} km")

    return uni_result


# ═══════════════════════════════════════════════════════════════════════════
# 3. Technical schools (Padrón Educativo - INET)
# ═══════════════════════════════════════════════════════════════════════════
def build_technical_school_indicators():
    """
    Read padrón educativo, filter technical schools (INET columns),
    count per department.
    """
    print("\n[3/3] Loading Padron Educativo...")

    # Row 13 (0-indexed: 12) has column names, data starts at row 14 (0-indexed: 13)
    df = pd.read_excel(
        DATA_DIR / "padron" / "padron_educativo.xlsx",
        header=None,
        skiprows=13,
        dtype=str,
    )

    # Column names from inspection:
    # [0] Jurisdiccion, [1] Sector, [2] Ambito, [3] Departamento,
    # [4] Codigo de departamento, [5] Localidad, [6] Codigo de localidad,
    # [7] Cueanexo, [8] Nombre, [9] Domicilio, [10] C.P., [11] Telefono,
    # [12] Mail, [13] Comun, [14] Especial, [15] Adultos,
    # [16] Nivel inicial - Jardin maternal, ... [20] Secundario - INET,
    # [21] SNU, [22] SNU - INET, ... [31] Formacion Profesional,
    # [32] Formacion Profesional - INET
    col_names = {
        0: "jurisdiccion", 1: "sector", 2: "ambito", 3: "departamento",
        4: "dpto5", 7: "cueanexo", 8: "nombre",
        13: "comun", 14: "especial", 15: "adultos",
        20: "secundario_inet", 22: "snu_inet", 32: "fp_inet",
    }
    df = df.rename(columns=col_names)

    # Keep only needed columns
    cols_keep = ["dpto5", "cueanexo", "nombre", "secundario_inet", "snu_inet", "fp_inet"]
    df = df[[c for c in cols_keep if c in df.columns]].copy()

    # Drop rows without department code
    df = df.dropna(subset=["dpto5"])
    df["dpto5"] = df["dpto5"].astype(str).str.strip().str.zfill(5)

    # CABA communes (02101-02115) → single code 02000
    df.loc[df["dpto5"].str.startswith("02"), "dpto5"] = "02000"

    print(f"  Schools with department code: {len(df):,}")

    # Convert INET flags to numeric
    for col in ["secundario_inet", "snu_inet", "fp_inet"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # A school is INET-technical if any INET column == 1
    inet_cols = [c for c in ["secundario_inet", "snu_inet", "fp_inet"] if c in df.columns]
    df["is_inet"] = df[inet_cols].max(axis=1).clip(0, 1)

    # Count per department
    total_schools = df.groupby("dpto5").size().rename("escuelas_total")
    inet_schools = df[df["is_inet"] == 1].groupby("dpto5").size().rename("escuelas_inet")

    result = pd.concat([total_schools, inet_schools], axis=1).fillna(0).astype(int)

    print(f"  Departments with schools: {len(result)}")
    print(f"  Total INET schools: {result['escuelas_inet'].sum():,}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integration: merge all + update PostgreSQL
# ═══════════════════════════════════════════════════════════════════════════
def integrate_and_load():
    """Merge all indicators and add columns to art1.departamentos."""
    engine = sa.create_engine(DB_URI)

    # Build each indicator set
    mincyt = build_mincyt_indicators()
    unis = build_university_indicators(engine)
    schools = build_technical_school_indicators()

    # Load existing departamentos to get dpto5 list and population
    existing = pd.read_sql(
        f"SELECT dpto5, pob_2022 FROM {SCHEMA}.{TABLE}",
        engine,
    )
    existing = existing.set_index("dpto5")
    print(f"\n[Integration] Existing departments: {len(existing)}")

    # Merge all on dpto5
    combined = existing.join(mincyt, how="left")
    combined = combined.join(unis, how="left")
    combined = combined.join(schools, how="left")

    # Fill NaN counts with 0
    count_cols = [
        "cyt_total", "cyt_researchers", "cyt_stem_researchers",
        "cyt_conicet", "cyt_conicet_stem", "cyt_cs_researchers",
        "uni_total", "uni_stem", "uni_distinct",
        "escuelas_total", "escuelas_inet",
    ]
    for col in count_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0).astype(int)

    # Compute per-capita rates (per 10,000 population)
    pop = combined["pob_2022"].astype(float)
    combined["cyt_stem_per_10k"] = np.where(
        pop > 0, combined["cyt_stem_researchers"] / pop * 10000, 0
    ).round(2)
    combined["cyt_conicet_per_10k"] = np.where(
        pop > 0, combined["cyt_conicet"] / pop * 10000, 0
    ).round(2)
    combined["inet_per_10k"] = np.where(
        pop > 0, combined["escuelas_inet"] / pop * 10000, 0
    ).round(2)
    combined["uni_per_10k"] = np.where(
        pop > 0, combined["uni_total"] / pop * 10000, 0
    ).round(2)

    # Knowledge intensity index (composite): standardised sum of key indicators
    key_vars = ["cyt_stem_per_10k", "cyt_conicet_per_10k", "inet_per_10k", "uni_per_10k"]
    for v in key_vars:
        col_z = v + "_z"
        mean_v = combined[v].mean()
        std_v = combined[v].std()
        combined[col_z] = np.where(std_v > 0, (combined[v] - mean_v) / std_v, 0)

    combined["knowledge_intensity"] = (
        combined[[v + "_z" for v in key_vars]].mean(axis=1)
    ).round(4)

    # Drop z-score temp columns
    combined = combined.drop(columns=[v + "_z" for v in key_vars])

    # ── Summary statistics ──
    print("\n=== Summary Statistics ===")
    report_cols = [
        "cyt_total", "cyt_researchers", "cyt_stem_researchers",
        "cyt_conicet", "cyt_conicet_stem", "cyt_cs_researchers",
        "uni_total", "uni_stem", "uni_distinct",
        "dist_stem_uni_km", "dist_any_uni_km",
        "escuelas_total", "escuelas_inet",
        "cyt_stem_per_10k", "cyt_conicet_per_10k",
        "inet_per_10k", "uni_per_10k", "knowledge_intensity",
    ]
    for col in report_cols:
        if col in combined.columns:
            s = combined[col]
            print(f"  {col:30s} mean={s.mean():10.2f}  std={s.std():10.2f}  "
                  f"min={s.min():10.2f}  max={s.max():10.2f}")

    # ── Update PostgreSQL ──
    new_cols = [c for c in combined.columns if c != "pob_2022"]

    print(f"\n[DB] Adding {len(new_cols)} columns to {SCHEMA}.{TABLE}...")

    with engine.begin() as conn:
        for col in new_cols:
            dtype = combined[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                pg_type = "INTEGER"
            else:
                pg_type = "DOUBLE PRECISION"

            # Add column if not exists
            conn.execute(sa.text(
                f"ALTER TABLE {SCHEMA}.{TABLE} ADD COLUMN IF NOT EXISTS "
                f"{col} {pg_type}"
            ))

        # Update values
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
                    params[param_name] = float(val) if not pd.api.types.is_integer_dtype(combined[col].dtype) else int(val)

            if set_clauses:
                sql = f"UPDATE {SCHEMA}.{TABLE} SET {', '.join(set_clauses)} WHERE dpto5 = :dpto5"
                conn.execute(sa.text(sql), params)

    # departamentos_geo is a VIEW on departamentos — no mirroring needed
    # (view was recreated with d.* to pick up new columns automatically)

    print("\n[DONE] Knowledge institution variables integrated successfully.")
    print(f"  New columns: {new_cols}")

    # Save intermediate CSV for inspection
    out_csv = DATA_DIR / "knowledge_institutions.csv"
    combined.to_csv(out_csv, index=True)
    print(f"  CSV saved: {out_csv}")


if __name__ == "__main__":
    integrate_and_load()
