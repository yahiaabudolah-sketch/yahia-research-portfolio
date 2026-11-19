# GhostHunter | Low-surface-brightness galaxy research

Galaxy candidate screening using DESI Legacy Imaging Surveys imagery and follow-up measurements. The archived workflow separates catalogue validation, source-identity diagnostics and exploratory discovery screening.

| Version | Purpose |
|---|---|
| [v10 catalogue validation](GHOSTHUNTER_V10_CATALOGUE_VALIDATION.py) | Frozen catalogue-based validation pilot |
| [v10.1 source audit](GHOSTHUNTER_V10_1_SOURCE_AUDIT.py) | Post-hoc source-profile and matching diagnostics |
| [v11.1 discovery pilot](GHOSTHUNTER_V11_1_DISCOVERY_PILOT_FIXED.py) | Corrected integrity-check compatibility; exploratory candidate screening |

## Recorded output

[Six follow-up candidates](results/followup_six_candidates.csv) · [Three-field measurements](results/three_field_measurements.csv)

Candidate coordinates and flags are recorded for follow-up, not presented as confirmed galaxies. Image contamination, compact neighbours, blending, survey masks and catalogue identity need investigation. Read the script headers for required cache files, dependencies, survey access and output structure before attempting reproduction.

The historical results are archived outputs, not a newly executed validation. [DESI Legacy Imaging Surveys](https://www.legacysurvey.org/) supplies survey context; its imagery and catalogue terms apply.
