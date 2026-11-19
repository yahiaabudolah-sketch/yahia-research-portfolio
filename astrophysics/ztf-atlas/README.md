# ZTF & ATLAS | Time-domain astronomical analysis

## Research question

Can apparent periodic variability survive quality control, tests for shared observational artefacts and comparison with independent survey photometry?

## ZTF search and follow-up

The 31 October handoff records a bounded 12-region search. Ten regions encountered network failures; two completed. Field M09 contained 30 sources and 13 preliminary periodic candidates. Periodogram rankings alone do not provide a calibrated global false-alarm probability.

Two neighbouring source peaks, near 0.249654 and 0.499280 days, shared anomalous high-cadence observations on 31 December 2018. Their same-night correlation was approximately 0.793 across 100 matched exposures; both were substantially weakened by excluding that epoch.

The original image audit showed seeing degrading from 2.696 to 6.237 arcsec, limiting magnitude falling from 20.63 to 19.21 and the later exposure carrying INFOBITS=8. Faint comparison sources brightened in the same exposure. These findings support a quality-dependent photometry bias, rather than robust persistent periodicity.

## Independent ATLAS check

A separate ZTF source (OID 514211100003577; Gaia DR3 3161310025388614272) was evaluated at a 1.49927-day trial period using 4,262 ATLAS input measurements. In the independently acquired data, 653 orange-filter nights and 199 cyan-filter nights survived the documented nightly analysis. Phase instability and weak independent modulation did not support a persistent periodic signal. The preselected-frequency permutation checks are not global false-alarm probabilities.

## Code and numerical evidence

- [Time-domain screening code](GHOSTHUNTER_TIME_DOMAIN_SPRINT.py)
- [ZTF quality reassessment](results/ztf_quality_reassessment.csv)
- [Same-epoch evidence summary](results/ztf_evidence_summary.json)
- [Original image audit](results/ztf_image_audit.json)
- [Candidate comparison](results/ztf_candidate_3577_comparison.csv)
- [Independent ATLAS report](results/atlas_independent_report.json)
- [ATLAS nightly quality-controlled measurements](results/atlas_nightly_quality_control.csv)

## Interpretation and limits

The current observations do not establish a newly discovered variable star. Cross-catalogue non-matches do not establish novelty. The project demonstrates a workflow for rejecting false positives using real photometry and image quality, and records the limitations of the initial candidate search.

Original datasets can be obtained from the relevant survey providers where permitted. [ZTF](https://www.ztf.caltech.edu/) | [ATLAS](https://fallingstar.com/).
