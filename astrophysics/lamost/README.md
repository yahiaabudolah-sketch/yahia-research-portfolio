# Stellar mass–luminosity relation | LAMOST DR2

**April 2026 | Stellar astrophysics, TOPCAT, FITS and Python**

## Data and question

The original project began with the VizieR catalogue of estimated radii, masses and luminosities for A and FGK stars from LAMOST DR2, containing 700,480 rows. It investigated the empirical relation L ∝ Mᵃ for selected main-sequence candidates.

## Selection

| Criterion | Historical filter |
|---|---|
| Temperature | 3,700–7,500 K |
| Surface gravity | log g > 4.0 in the April project description; a later working selection used > 4.1 |
| Mass | 0 < M < 2.5 solar masses |
| Luminosity | 0 < L < 100 solar luminosities |

The initial catalogue was filtered in TOPCAT. Positive masses and luminosities were transformed to base-10 logarithms and fitted to log L = a log M + b. The earlier analysis also examined asymmetric uncertainties using catalogue quartiles; the luminosity columns LumQ1 and LumQ3 were used to estimate mean log-space uncertainty.

## Recorded results

| Quantity | Historical reported value |
|---|---:|
| Fitted exponent a | 3.46 |
| Coefficient of determination R² | 0.516 |
| Mean mass uncertainty (reported) | 0.0331 |
| Mean luminosity uncertainty (reported) | 0.3329 |

The exact units and aggregation conventions of the last two reported quantities require checking against the original notebook. Sample selection matters: earlier fits changed when stronger filtering was applied, so these values should not be treated as universal stellar parameters.

[Historical result record](recorded_results.json) | [Reconstructed Python workflow](analyze.py)

## Reproduction

The original filtered catalogue, final notebook and scatter plot have not been recovered. Obtain the source catalogue from [VizieR](https://vizier.cds.unistra.fr/) or [LAMOST](https://www.lamost.org/), identify its original columns, and apply the documented cuts. The reconstructed script accepts column names and the surface-gravity threshold as parameters:

```bash
pip install numpy astropy matplotlib
python analyze.py FGK.fit --logg-min 4.0 --out output
python analyze.py FGK.fit --logg-min 4.1 --out output_stricter
```

Compare the selection counts and regression coefficients. The script produces a density plot and fitted values from the supplied catalogue; it does not generate simulated stars to reproduce the historical slope.
