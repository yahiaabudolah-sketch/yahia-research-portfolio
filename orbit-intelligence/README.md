# Orbit Intelligence | Singapore maritime nowcasting

## Research question
Can contemporary shipping signals improve estimates of monthly official container throughput in Singapore?

## Historical experiment

The model was evaluated on 93 monthly observations (January 2019–September 2026), with a fixed training cutoff of December 2024 and a 21-month holdout. Year-on-year daily rates reduce seasonality, and an ordinary least-squares regression combines two contemporaneous AIS traffic measures. AIS is collected by satellite and terrestrial receivers; this experiment does not use optical imagery.

| Model | Holdout MAPE |
|---|---:|
| Seasonal baseline | 6.26% |
| Historical growth baseline | 4.47% |
| AIS nowcast | 3.86% |

The results are historical associations, not proof of a live predictive advantage. The data are revised historical observations, and release timing and real-time vintages were not tested. The sample is small, covers one port and cannot establish causality. Negative partial coefficients need further investigation rather than causal interpretation.

## Reproduction

1. Obtain the monthly observations from the original or permitted data providers below. Check their redistribution terms before publishing copies.
2. Assemble `singapore_monthly.csv` in this directory using `input_template.csv` as the schema: ISO month start, monthly container-ship call count, average all-vessel calls per day, and monthly container throughput in thousands of TEU. Include every month from January 2019 through September 2026. Ensure dates, units and port definitions match the historical experiment.
3. From the repository root install `requirements.txt`; then run `python orbit-intelligence/analyze.py`.
4. Examine `results.json`, generated figures and `dashboard.html`. The dashboard is generated locally and is not committed because it embeds the source values. Updated inputs may yield different metrics.

The original AIS snapshot was transcribed from republished tables; the raw data, data-bearing figures, PDF and interactive dashboard are retained in a separate local project bundle, not published here. The aggregate numerical results from that experiment are included. The optional `nightlights_extension.js` requires Earth Engine and has not been run.

## Sources and attribution (accessed 19 November 2026)
- Source: International Monetary Fund, [IMF PortWatch](https://portwatch.imf.org/), developed with the University of Oxford using UN Global Platform AIS data. [Methodology](https://www.elibrary.imf.org/view/journals/001/2025/093/article-A001-en.xml).
- Historical monthly container calls: [IndexBox republication](https://app.indexbox.io/indicators/symbol/port-calls-of-container-ships-monthly-imf-portwatch-from-ais-data-singapore/), updated 19 October 2026.
- Historical all-vessel monthly means: [World Freight Monitor](https://worldfreightmonitor.com/ports/singapore/), retrieved 10 November 2026. [Provider terms and source licences](https://worldfreightmonitor.com/sources/).
- Singapore Department of Statistics, [Sea Cargo and Shipping Statistics, Monthly](https://data.gov.sg/datasets/d_9eba4566b01827e386c9ae54c6b01629/view), available under the [Singapore Open Data Licence](https://data.gov.sg/open-data-licence).
- [MPA Singapore 2025 annual port results](https://www.mpa.gov.sg/media-centre/details/singapore-posts-record-port-performance-in-2025-and-develops-future-readiness-through-industry-collaborations-for-2026).
- Optional optical extension: [NOAA VIIRS monthly composites](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMCFG).

## Next steps
Reproduce the historical experiment, test alternative holdout periods, compare a second port and check the actual dates on which the input data became available.
