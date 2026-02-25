"""
01_extract_satellite_indicators.py
Extracts satellite-derived economic proxies at department level from GEE.

Indicators:
  1. GHSL Built-Up Surface (2020) — urban capital stock proxy
  2. Sentinel-5P NO2 (2022 annual mean) — industrial activity proxy
  3. MODIS EVI (2022 growing season integral) — agricultural productivity
  4. GHSL Built-Up Volume (2020) — floor-space stock

Output: satelital/gee_indicators_depto.csv

Usage:
    python 01_extract_satellite_indicators.py
"""

import ee
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path

ee.Initialize()

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "satelital"
OUT_DIR.mkdir(exist_ok=True)

# Load department polygons from GeoJSON exported from PostGIS
GEOJSON_PATH = OUT_DIR / "departamentos_polygons.geojson"


def export_departamentos_geojson():
    """Export department polygons from PostGIS to GeoJSON for GEE."""
    import sqlalchemy as sa
    engine = sa.create_engine("postgresql://postgres:postgres@localhost:5432/posadas")

    df = pd.read_sql("""
        SELECT dpto5, ST_AsGeoJSON(geometry) as geojson
        FROM art1.departamentos_geo
        WHERE dpto5 IS NOT NULL
    """, engine)

    features = []
    for _, row in df.iterrows():
        geom = json.loads(row["geojson"])
        features.append({
            "type": "Feature",
            "properties": {"dpto5": row["dpto5"]},
            "geometry": geom,
        })

    fc = {"type": "FeatureCollection", "features": features}
    with open(GEOJSON_PATH, "w") as f:
        json.dump(fc, f)

    print(f"Exported {len(features)} department polygons to {GEOJSON_PATH}")
    return fc


def load_departments_ee(geojson_fc):
    """Convert GeoJSON to EE FeatureCollection."""
    features = []
    for feat in geojson_fc["features"]:
        try:
            ee_feat = ee.Feature(
                ee.Geometry(feat["geometry"]),
                {"dpto5": feat["properties"]["dpto5"]}
            )
            features.append(ee_feat)
        except Exception:
            pass
    return ee.FeatureCollection(features)


def extract_in_batches(image, fc_list, band, reducer, scale, batch_size=50, max_retries=3):
    """Extract zonal statistics in batches to avoid GEE memory limits."""
    all_results = []

    for i in range(0, len(fc_list), batch_size):
        batch = fc_list[i:i + batch_size]
        batch_fc = ee.FeatureCollection(batch)

        for attempt in range(max_retries):
            try:
                reduced = image.select(band).reduceRegions(
                    collection=batch_fc,
                    reducer=reducer,
                    scale=scale,
                )
                results = reduced.getInfo()
                for feat in results["features"]:
                    props = feat["properties"]
                    all_results.append(props)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"    Batch {i // batch_size + 1} failed (attempt {attempt+1}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    Batch {i // batch_size + 1} FAILED after {max_retries} attempts: {e}")
                    # Fill with None for this batch
                    for feat in batch:
                        all_results.append({"dpto5": feat.getInfo()["properties"]["dpto5"]})

        pct = min(100, (i + batch_size) / len(fc_list) * 100)
        print(f"    Batch {i // batch_size + 1}: {pct:.0f}% done")

    return all_results


def extract_ghsl_built_surface(fc_list):
    """GHSL Built-Up Surface 2020 (m2 per cell)."""
    print("\n[1/4] GHSL Built-Up Surface 2020...")
    img = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")

    results = extract_in_batches(
        img, fc_list, "built_surface",
        ee.Reducer.sum(), scale=100, batch_size=30
    )

    records = []
    for r in results:
        records.append({
            "dpto5": r["dpto5"],
            "ghsl_built_surface_m2": r.get("sum", 0) or 0,
        })

    df = pd.DataFrame(records)
    print(f"  Departments: {len(df)}, total built: {df['ghsl_built_surface_m2'].sum() / 1e6:.1f} km2")
    return df


def extract_ghsl_built_volume(fc_list):
    """GHSL Built-Up Volume 2020 (m3 per cell)."""
    print("\n[2/4] GHSL Built-Up Volume 2020...")
    img = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_V/2020")

    results = extract_in_batches(
        img, fc_list, "built_volume_total",
        ee.Reducer.sum(), scale=100, batch_size=30
    )

    records = []
    for r in results:
        records.append({
            "dpto5": r["dpto5"],
            "ghsl_built_volume_m3": r.get("sum", 0) or 0,
        })

    df = pd.DataFrame(records)
    print(f"  Departments: {len(df)}, total volume: {df['ghsl_built_volume_m3'].sum() / 1e9:.2f} billion m3")
    return df


def extract_no2(fc_list):
    """Sentinel-5P tropospheric NO2, annual mean 2022."""
    print("\n[3/4] Sentinel-5P NO2 annual mean 2022...")

    # Use monthly means first to reduce computation load
    monthly_means = []
    for month in range(1, 13):
        start = f"2022-{month:02d}-01"
        end_month = month + 1 if month < 12 else 1
        end_year = 2022 if month < 12 else 2023
        end = f"{end_year}-{end_month:02d}-01"
        m = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
             .filterDate(start, end)
             .select("tropospheric_NO2_column_number_density")
             .mean())
        monthly_means.append(m)

    annual_mean = ee.ImageCollection(monthly_means).mean()

    results = extract_in_batches(
        annual_mean, fc_list, "tropospheric_NO2_column_number_density",
        ee.Reducer.mean(), scale=5000, batch_size=15
    )

    records = []
    for r in results:
        val = r.get("mean", None)
        records.append({
            "dpto5": r["dpto5"],
            # Convert from mol/m2 to umol/m2 for readability
            "no2_trop_umol_m2": (val * 1e6) if val is not None else None,
        })

    df = pd.DataFrame(records)
    print(f"  Departments: {len(df)}, mean NO2: {df['no2_trop_umol_m2'].mean():.2f} umol/m2")
    return df


def extract_evi(fc_list):
    """MODIS EVI growing season integral (Oct 2021 - Mar 2022)."""
    print("\n[4/4] MODIS EVI growing season 2021-2022...")

    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate("2021-10-01", "2022-03-31")
           .select("EVI"))

    # Sum of EVI composites (16-day) as proxy for cumulative vegetation productivity
    evi_sum = col.sum().multiply(0.0001)  # scale factor

    results = extract_in_batches(
        evi_sum, fc_list, "EVI",
        ee.Reducer.mean(), scale=500, batch_size=20
    )

    records = []
    for r in results:
        records.append({
            "dpto5": r["dpto5"],
            "evi_growing_season": r.get("mean", None),
        })

    df = pd.DataFrame(records)
    print(f"  Departments: {len(df)}, mean EVI integral: {df['evi_growing_season'].mean():.4f}")
    return df


def main():
    # Step 1: Export departments from PostGIS
    if GEOJSON_PATH.exists():
        print(f"Loading existing {GEOJSON_PATH}")
        with open(GEOJSON_PATH) as f:
            fc_geojson = json.load(f)
    else:
        fc_geojson = export_departamentos_geojson()

    # Step 2: Convert to EE features list
    print("Converting to EE features...")
    fc_list = []
    for feat in fc_geojson["features"]:
        try:
            ee_feat = ee.Feature(
                ee.Geometry(feat["geometry"]),
                {"dpto5": feat["properties"]["dpto5"]}
            )
            fc_list.append(ee_feat)
        except Exception:
            pass
    print(f"  {len(fc_list)} department features loaded")

    # Step 3: Extract each indicator, saving partial results
    partial_path = OUT_DIR / "gee_partial.csv"

    # Check for partial results from prior run
    if partial_path.exists():
        partial = pd.read_csv(partial_path, dtype={"dpto5": str})
        partial["dpto5"] = partial["dpto5"].str.zfill(5)
    else:
        partial = pd.DataFrame()

    if "ghsl_built_surface_m2" not in partial.columns:
        ghsl_s = extract_ghsl_built_surface(fc_list)
        if len(partial) == 0:
            partial = ghsl_s
        else:
            partial = partial.merge(ghsl_s, on="dpto5", how="outer")
        partial.to_csv(partial_path, index=False)

    if "ghsl_built_volume_m3" not in partial.columns:
        ghsl_v = extract_ghsl_built_volume(fc_list)
        partial = partial.merge(ghsl_v, on="dpto5", how="outer")
        partial.to_csv(partial_path, index=False)

    if "no2_trop_umol_m2" not in partial.columns:
        no2 = extract_no2(fc_list)
        partial = partial.merge(no2, on="dpto5", how="outer")
        partial.to_csv(partial_path, index=False)

    if "evi_growing_season" not in partial.columns:
        evi = extract_evi(fc_list)
        partial = partial.merge(evi, on="dpto5", how="outer")
        partial.to_csv(partial_path, index=False)

    # Save final
    out_path = OUT_DIR / "gee_indicators_depto.csv"
    partial.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved {len(partial)} departments to {out_path}")
    print(partial.describe().to_string())


if __name__ == "__main__":
    main()
