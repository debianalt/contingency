"""
Populate the replication package from the project working directory.
Run from the project root: python github-github/populate_repo.py
"""

import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent   # 2026-3/
REPO    = Path(__file__).resolve().parent           # github-github/

# ── Data ─────────────────────────────────────────────────────────────
data_files = [
    ("data/departamentos_master.csv",              "data/departamentos_master.csv"),
    ("data/satelital/departamentos_polygons.geojson", "data/departamentos_polygons.geojson"),
]

# ── Scripts ──────────────────────────────────────────────────────────
script_files = [
    "scripts/00_build_knowledge_institutions.py",
    "scripts/01_extract_satellite_indicators.py",
    "scripts/02_integrate_economic_indicators.py",
    "scripts/03_analysis.py",
    "scripts/04_regenerate_figures.py",
]

# ── Figures (body + supplementary, excluding unused) ─────────────────
figure_files = [
    "figures/fig_01_distributions.png",
    "figures/fig_02_correlation_heatmap.png",
    "figures/fig_03_horse_race_scatter.png",
    "figures/fig_05_mca_knowledge.png",
    "figures/fig_06_forest_plot.png",
    "figures/fig_08_choropleth_typology.png",
    "figures/fig_09_choropleth_eci_stem.png",
    "figures/fig_S1_robustness.png",
]

# ── Tables (used in article, excluding unused heckman/logit) ─────────
table_files = [
    "tables/table_01_descriptive.csv",
    "tables/table_02_correlations.csv",
    "tables/table_02b_partial_correlations.csv",
    "tables/table_03_models.csv",
    "tables/table_05_type_specific.csv",
    "tables/table_07_robustness.csv",
    "tables/table_08_logit_type.csv",
    "tables/table_S3_mca_coordinates.csv",
]


def copy_file(src_rel, dst_rel=None):
    """Copy a file from PROJECT to REPO."""
    src = PROJECT / src_rel
    dst = REPO / (dst_rel or src_rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  OK  {src_rel} -> {dst.relative_to(REPO)}")
    else:
        print(f"  MISSING  {src_rel}")


def main():
    print("Populating replication package...\n")

    print("Data:")
    for item in data_files:
        if isinstance(item, tuple):
            copy_file(item[0], item[1])
        else:
            copy_file(item)

    print("\nScripts:")
    for f in script_files:
        copy_file(f)

    print("\nFigures:")
    for f in figure_files:
        copy_file(f)

    print("\nTables:")
    for f in table_files:
        copy_file(f)

    # Clean up .gitkeep placeholders
    for gk in REPO.rglob(".gitkeep"):
        gk.unlink()
        print(f"\n  Removed placeholder: {gk.relative_to(REPO)}")

    print(f"\nDone. Repo contents at: {REPO}")
    print(f"Total files: {sum(1 for _ in REPO.rglob('*') if _.is_file() and _.name != 'populate_repo.py')}")


if __name__ == "__main__":
    main()
