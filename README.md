# Replication materials

Replication package for:

> Gomez, R. E. (2026). Where knowledge predicts digital sophistication and where it does not: institutional contingency of the second-level divide in subnational software production. Working paper.

## Structure

```
contingency/
├── data/
│   ├── departamentos_master.csv         # 511 departments × 277 variables (integrated dataset)
│   └── departamentos_polygons.geojson   # Department boundary polygons for mapping
├── scripts/
│   ├── 00_build_knowledge_institutions.py  # Knowledge infrastructure integration (MinCyT, SPU)
│   ├── 01_extract_satellite_indicators.py  # Satellite proxies (VIIRS nighttime radiance)
│   ├── 02_integrate_economic_indicators.py # CEP-XXI wages, employment, ENACOM internet
│   ├── 03_analysis.py                     # Main analysis: OLS, Chow test, type-specific models
│   ├── 04_regenerate_figures.py           # Standalone figure regeneration
│   ├── 10_revision_analyses.py            # Moderation, non-linearity, temporal panel (Tables S8, S10)
│   ├── 11_selection_sensitivity.py        # Twelve-variant selection sensitivity (Table S9)
│   ├── 12_mca_robustness.py               # Typology robustness across specifications (Table S6)
│   ├── 13_h3_interaction_power.py         # Power analysis (Cohen 1988 minimum detectable effects)
│   ├── 14_construct_validity.py           # Project-level construct validity (Table S11)
│   └── 15_temporal_window_spsi.py         # Restricted-window SPSI recomputation (Table S13)
├── figures/                               # Output figures (PNG, 300 DPI)
├── tables/                                # Output tables (CSV)
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# Run the full analysis (generates the core figures and tables)
python scripts/03_analysis.py

# Or regenerate figures only (standalone)
python scripts/04_regenerate_figures.py
```

## Data

The integrated dataset (`data/departamentos_master.csv`) contains 511 Argentine departments with 277 variables from the following sources:

| Source | Variables | Period |
|--------|-----------|--------|
| **GitHub Argentina** | Developer counts, language portfolios, SPSI | 2008–2026 |
| **MinCyT (CVar)** | STEM researchers per 10,000 population | 2022 |
| **SPU** | University locations, STEM programme offerings | 2023 |
| **INDEC Census** | Population, education, employment, poverty | 2010, 2022 |
| **CEP-XXI** | Formal wages, sectoral employment | 2020–2022 |
| **ENACOM** | Internet household penetration | 2022 |
| **VIIRS** | Nighttime radiance composites | 2014, 2022 |

The dependent variable is the Software Portfolio Sophistication Index (SPSI), computed from the bipartite network linking departments to programming languages via the eigenvalue-decomposition method (stored as `eci_software` in the dataset for historical continuity of the variable name).

Scripts `00`–`02` document the pipeline from raw sources to the master dataset. Since raw source files are not redistributed (census microdata, MinCyT personnel records), these scripts serve as methodological documentation. The core analysis is fully reproducible from `departamentos_master.csv` alone using `03_analysis.py`. Scripts `10`–`14` document the revision-stage robustness analyses and run from the master dataset and the CSVs in `tables/`. Script `15` recomputes the SPSI from repository-level records held in a PostgreSQL database that is not redistributed (GitHub terms of service); it is included as methodological documentation, and its output is `tables/table_S13_temporal_window.csv`.

## Territorial typology

The six territorial types used in this study derive from a Multiple Correspondence Analysis (MCA) and hierarchical clustering computed in a companion study:

> Gomez, R. E. (2026). The spatiality of software: subnational economic complexity from GitHub data in Argentina. Working paper. Replication materials: [https://doi.org/10.5281/zenodo.18674718](https://doi.org/10.5281/zenodo.18674718)

The MCA cluster assignments are included in `departamentos_master.csv` (variable `cluster_6`). The MCA estimation code is available in the companion repository linked above.

## Key results

| Model | *N* | *R*² | Key finding |
|-------|-----|------|-------------|
| Pooled OLS | 221 | 0.468 | STEM β = 0.220*** (knowledge > wealth) |
| Chow test | 221 | — | *F* = 2.14, *p* < 0.001 (reject coefficient homogeneity) |
| Metro-Diversified | 45 | 0.453 | STEM β = 0.467*** |
| Pampeana-Educated | 58 | 0.326 | STEM β = 0.225*** |
| Metro-Core | 53 | 0.456 | STEM β = 0.236* |
| Intermediate-Urban | 49 | 0.172 | STEM n.s.; wages β = 0.301* |
| Pooled OLS, SPSI 2015–2026 | 221 | 0.452 | STEM β = 0.203*** (*r* = 0.995 with full-window index) |
| Pooled OLS, SPSI 2020–2026 | 214 | 0.485 | STEM β = 0.186*** (*r* = 0.960 with full-window index) |

## Output mapping

### Figures

| Article | File | Content |
|---------|------|---------|
| Fig. 1 | `fig_08_choropleth_typology.png` | Choropleth: six territorial types |
| Fig. 2 | `fig_09_choropleth_eci_stem.png` | Dual choropleth: SPSI and STEM density |
| Fig. 3 | `fig_05_mca_knowledge.png` | MCA factorial plane with skills gradient |
| Fig. 4 | `fig_06_forest_plot.png` | Forest plot: standardised betas by type |
| Fig. 5 | `fig_03_horse_race_scatter.png` | Scatter: SPSI vs STEM and wages by type |
| Fig. S1 | `fig_S1_robustness.png` | Coefficient stability across specifications |
| Fig. S2 | `fig_02_correlation_heatmap.png` | Correlation heatmap |
| Fig. S3 | `fig_01_distributions.png` | Distributions: SPSI, developers, STEM, wages |

### Tables

| Article | File | Content |
|---------|------|---------|
| Table 2 | `table_01_descriptive.csv` | Descriptive statistics |
| Table 3 | `table_03_models.csv` | Pooled OLS (knowledge-only, wealth-only, full) |
| Table 4 | `table_05_type_specific.csv` | Type-specific OLS coefficients |
| Table 5 / S10 | `table_05_moderation_nonlinearity.csv` | Moderation and non-linearity extensions |
| Table S1 | `table_02_correlations.csv` + `table_02b_partial_correlations.csv` | Bivariate and partial correlations |
| Table S2 | `table_07_robustness.csv` | Coefficient stability across specifications |
| Table S3 | `table_S3_mca_coordinates.csv` | MCA modality coordinates and contributions |
| Table S4 | `table_08_logit_type.csv` | Type-specific logit (participation) |
| Table S5 | `table_03_models.csv` | Skills-only and wealth-only models |
| Table S6 | `table_S6_typology_robustness.csv` | Typology robustness (Cramér's *V*) |
| Table S7 | `table_S7_bundle_robustness.csv` | Bundle-based alternative indicator |
| Table S8 | `table_S8_temporal_panel.csv` | Year-over-year rank stability 2015–2025 |
| Table S9 | `table_S9_selection_sensitivity.csv` | Selection sensitivity (twelve variants) |
| Table S11 | `table_S11_construct_validity.csv` | Construct validity against project-level signals |
| Table S13 | `table_S13_temporal_window.csv` | Restricted-window SPSI recomputation |

Table 1 (cluster profiles), Table S12 (MCA eigenvalue decomposition), and the in-text Chow statistics are produced inline by `03_analysis.py` and the manuscript build; they have no standalone CSV.

## Colour palette

Paul Tol Bright (colorblind-friendly):

| Type | Hex |
|------|-----|
| Peripheral-Deprived | `#CC3311` |
| Metro-Core | `#0077BB` |
| Metro-Diversified | `#009988` |
| Pampeana-Educated | `#AA3377` |
| Semi-Rural-Active | `#EE7733` |
| Intermediate-Urban | `#BBBBBB` |

## Author

Raimundo Elias Gomez
CONICET / Universidad Nacional de Misiones, Argentina
Institute of Sociology, University of Porto, Portugal
ORCID: [0000-0002-4468-9618](https://orcid.org/0000-0002-4468-9618)

## License

This work is licensed under the [MIT License](LICENSE).
