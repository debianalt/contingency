# Replication materials

Replication package for:

> Gomez, R. E. (2026). Where knowledge drives digital sophistication and where it does not: institutional contingency of the second-level divide in subnational software production. Working paper.

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
│   └── 04_regenerate_figures.py           # Standalone figure regeneration (all 8 body + 2 supp)
├── figures/                               # Output figures (PNG, 300 DPI)
├── tables/                                # Output tables (CSV)
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# Run the full analysis (generates all figures and tables)
python scripts/03_analysis.py

# Or regenerate figures only (standalone, improved formatting)
python scripts/04_regenerate_figures.py
```

## Data

The integrated dataset (`data/departamentos_master.csv`) contains 511 Argentine departments with 277 variables from the following sources:

| Source | Variables | Period |
|--------|-----------|--------|
| **GitHub Argentina** | Developer counts, language portfolios, ECI | 2008–2026 |
| **MinCyT (CVar)** | STEM researchers per 10,000 population | 2022 |
| **SPU** | University locations, STEM programme offerings | 2023 |
| **INDEC Census** | Population, education, employment, poverty | 2010, 2022 |
| **CEP-XXI** | Formal wages, sectoral employment | 2020–2022 |
| **ENACOM** | Internet household penetration | 2022 |
| **VIIRS** | Nighttime radiance composites | 2014, 2022 |

Scripts `00`–`02` document the pipeline from raw sources to the master dataset. Since raw source files are not redistributed (census microdata, MinCyT personnel records), these scripts serve as methodological documentation. The analysis is fully reproducible from `departamentos_master.csv` alone using `03_analysis.py`.

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

## Output mapping

### Figures

| Article | File | Content |
|---------|------|---------|
| Fig. 1 | `fig_08_choropleth_typology.png` | Choropleth: six territorial types |
| Fig. 2 | `fig_01_distributions.png` | Histograms: ECI, developers, STEM, wages |
| Fig. 3 | `fig_09_choropleth_eci_stem.png` | Dual choropleth: ECI and STEM density |
| Fig. 4 | `fig_05_mca_knowledge.png` | MCA factorial plane with knowledge gradient |
| Fig. 5 | `fig_06_forest_plot.png` | Forest plot: standardised betas by type |
| Fig. 6 | `fig_03_horse_race_scatter.png` | Scatter: ECI vs STEM and wages by type |
| Fig. S1 | `fig_S1_robustness.png` | Coefficient stability across specifications |
| Fig. S2 | `fig_02_correlation_heatmap.png` | Correlation heatmap |

### Tables

| Article | File | Content |
|---------|------|---------|
| Table 2 | `table_01_descriptive.csv` | Descriptive statistics |
| Table 3 | `table_03_models.csv` | Pooled OLS (knowledge-only, wealth-only, full) |
| Table 4 | `table_05_type_specific.csv` | Type-specific OLS coefficients |
| Table S1 | `table_02_correlations.csv` | Bivariate correlations |
| Table S1 | `table_02b_partial_correlations.csv` | Partial correlations |
| Table S2 | `table_07_robustness.csv` | Robustness checks |
| Table S3 | `table_S3_mca_coordinates.csv` | MCA modality coordinates and contributions |
| Table S4 | `table_08_logit_type.csv` | Type-specific logit (participation) |

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
