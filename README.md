![Yahia Suhaib Abudoulah – research and product portfolio](portfolio-assets/banner.svg)

[**Finance & alternative data**](#01--finance--alternative-data) · [**Scientific research**](#02--scientific-research) · [**Product development**](#03--product-development) · [**Professional development**](#04--professional-development)

I am a final-year Astrophysics undergraduate at Swansea University (expected graduation June 2027), interested in computational physics, quantitative analysis and the commercial applications of frontier technology. This portfolio collects research code, numerical findings, financial models and product-development case studies.

## Selected work

| Project | Evidence to explore | Highlights |
|---|---|---|
| **[Commercial Space Economics](space-economics/)** | [Excel model](space-economics/Space_Economics_Model.xlsx) · [Report](space-economics/Space_Economics_Report.pdf) · [Python](space-economics/analyze.py) | 3 companies, 9 funding scenarios |
| **[Orbit Intelligence](orbit-intelligence/)** | [Python](orbit-intelligence/analyze.py) · [Model results](orbit-intelligence/results.json) | 93 monthly observations; 3.86% historical holdout MAPE |
| **[Boundary-Dominated Curvature](astrophysics/emergent-gravity/)** | [Symbolic validation](astrophysics/emergent-gravity/closed_form_symbolic_curvature_validation.py) · [Motif derivation](astrophysics/emergent-gravity/motif_count_derivation.py) | Exact model-specific boundary law; 1,125 historical validation points |
| **[GhostHunter](astrophysics/ghosthunter/)** | [v11.1 detector](astrophysics/ghosthunter/GHOSTHUNTER_V11_1_DISCOVERY_PILOT_FIXED.py) · [Candidate data](astrophysics/ghosthunter/results/followup_six_candidates.csv) | Galaxy candidate screening and follow-up |
| **[ZTF & ATLAS](astrophysics/ztf-atlas/)** | [Python analysis](astrophysics/ztf-atlas/GHOSTHUNTER_TIME_DOMAIN_SPRINT.py) · [Independent check](astrophysics/ztf-atlas/results/atlas_independent_report.json) | Periodicity, artefact rejection, cross-survey validation |
| **[LAMOST stellar analysis](astrophysics/lamost/)** | [Portable reproduction script](astrophysics/lamost/analyze.py) · [Historical findings](astrophysics/lamost/recorded_results.json) | Historical mass–luminosity exponent ≈3.46 |
| **[ATHAR](athar/)** | [Development case study](athar/README.md) · [Geometry QA](athar/geometry-qa.md) | Student-focused web product, 11-room world prototype |
| **[BLACKOUT](blackout/)** | [Product case study](blackout/README.md) | Immersive narrative mobile application concept |
| **[Standard Chartered](professional-development/)** | [Simulation overview](professional-development/README.md) · [Completion record](professional-development/completion-record.md) | Corporate credit analysis, Forage |

## 01 — Finance & alternative data

### Commercial Space Economics

How do different space-sector business models fund growth? The model compares Rocket Lab, Planet Labs and Intuitive Machines using published annual disclosures, then tests five-year cash-flow sensitivity under downside, base and upside assumptions.

![Historical free cash flow](space-economics/historical_fcf.png)

[Research report](space-economics/Space_Economics_Report.pdf) · [Excel workbook](space-economics/Space_Economics_Model.xlsx) · [Source data and methodology](space-economics/)

### Orbit Intelligence

A historical maritime nowcasting experiment combining AIS and official Singapore container throughput. The test uses a chronological holdout rather than a random split.

![Model comparison](portfolio-assets/orbit-validation.svg)

[Model code](orbit-intelligence/analyze.py) · [Results](orbit-intelligence/results.json) · [Reproduction and provider sources](orbit-intelligence/)

## 02 — Scientific research

### Boundary-Dominated Curvature Response in Weighted Network Defects

An investigation of weighted graph curvature and the mathematical role of interface motifs. The July 2026 manuscript reports symbolic/direct graph agreement at floating-point precision across 1,125 parameter points. It deliberately limits its claim to the specified network model.

![Weighted network schematic](portfolio-assets/gravity-concept.svg)

[Research overview and code](astrophysics/emergent-gravity/) · [Full astrophysics index](astrophysics/)

### Astronomical data analysis

**GhostHunter** screens low-surface-brightness galaxy candidates and examines detector performance and source identity. **ZTF & ATLAS** tests time-domain signals using independent survey data, controls and quality cuts; non-confirmation is a scientifically meaningful outcome. **LAMOST** examines the empirical mass–luminosity relation of FGK stars.

[GhostHunter](astrophysics/ghosthunter/) · [Time-domain research](astrophysics/ztf-atlas/) · [Stellar analysis](astrophysics/lamost/)

## 03 — Product development

| [ATHAR](athar/) | [BLACKOUT](blackout/) |
|---|---|
| Student productivity, career evidence and an interactive world-inspired interface. October geometry handoff includes a validated 11-room reference layout. | Immersive horror storytelling, sound, investigations and vintage editorial design. Development-stage concept; current approved visual build is not yet in the public repository. |

Product work is documented separately from scientific results. Private source repositories and assets are not redistributed.

## 04 — Professional development

[**Standard Chartered — Credit Analyst Job Simulation (Forage)**](professional-development/) · completed 19 November 2026. Simulated client financial assessment, credit risk, industry research and a lending recommendation presentation; not bank employment.

## Methods & tools

Python · NumPy · SciPy · pandas · Matplotlib · Astropy · Excel · statistical modelling · time-series analysis · financial statements · scenario analysis · React/Vite/Phaser product prototyping

## Reproducibility and research integrity

The commercial models are AI-assisted research prototypes. Historical results are not new runs; the LAMOST script is a documented reimplementation rather than a recovered original. The gravity manuscript is an unpublished model study; no validated theory of gravity, galaxy discovery, variable-star discovery, product launch or employer endorsement is claimed. Data availability, permissions and independent reproduction status are documented in the project pages and [data/reuse notes](DATA_AND_REUSE.md).

For code, start with [requirements](requirements.txt) and each project's own instructions. Research measurements and conclusions should be independently checked before professional reliance.
