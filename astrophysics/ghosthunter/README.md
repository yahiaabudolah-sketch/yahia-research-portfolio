# GhostHunter | Low-surface-brightness galaxy research

## Objective

Screen DESI Legacy Imaging Surveys images for diffuse sources and determine whether candidate detections survive catalogue and image-level checks.

## Method

The project uses multi-scale image filtering, background estimation, source masks, candidate ranking, position matching and visual follow-up. Known diffuse galaxies and published catalogue positions served as controls. Image-level source profiles were compared with nearby compact-source profiles, but are not calibrated point-spread function fits.

| Experiment | Recorded findings |
|---|---|
| v9 catalogue controls | Two held-out published galaxies (Dragonfly 44 and NGC5846-UDG1) recovered in the top ten; DGSAT I lacked usable image coverage. |
| v10 catalogue pilot | 12 predeclared SMUDGes controls. 9/12 top-ten recovery within 20 arcsec, tightening to 8/12 within 5 arcsec. |
| v11.1 exploratory survey | All 49 tiles around NGC 7713 processed; 1,247 raw peaks, 1,026 merged peaks and 370 Tier A candidates. |
| Follow-up | Seven priority sources were examined; none established as a new galaxy. |

These are recorded results from the 31 October research handoff. They do not measure survey-wide completeness or constitute a discovery. A single-catalogue non-match does not establish novelty.

## Source code

| File | Purpose |
|---|---|
| [v10 catalogue validation](GHOSTHUNTER_V10_CATALOGUE_VALIDATION.py) | Frozen catalogue-based pilot |
| [v10.1 source audit](GHOSTHUNTER_V10_1_SOURCE_AUDIT.py) | Local profile and positional checks |
| [v11.1 discovery pilot](GHOSTHUNTER_V11_1_DISCOVERY_PILOT_FIXED.py) | Exploratory screening and candidate ranking |

## Measurement tables

[Six follow-up candidates](results/followup_six_candidates.csv) | [Three-field measurements](results/three_field_measurements.csv) | [Two-field measurements](results/two_field_measurements.csv) | [GH11-0643 measurements](results/GH11_0643_measurements.csv)

Several candidates were weak, inconsistent or affected by blends, bright neighbours and mask contamination. The follow-up therefore prioritises source rejection and verification rather than counting image peaks as galaxies.

## Dependencies and data

See individual script headers for cache paths and required files. The scripts use numerical image processing and publicly obtainable DESI data; original survey images are subject to the provider's terms. [Legacy Surveys](https://www.legacysurvey.org/) | [SMUDGes](https://www.legacysurvey.org/) (catalogue context).

Recorded results have not been independently rerun for this repository.
