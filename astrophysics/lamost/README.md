# Stellar mass–luminosity relation | LAMOST DR2

**Historical analysis:** April 2026 · FGK main-sequence candidates · FITS / TOPCAT / Python

The project used public LAMOST DR2 stellar data and a filtered FGK candidate sample to investigate the power-law relation L ∝ Mᵃ. The historical reported fit yielded a ≈ 3.46 and R² ≈ 0.516. The original FITS and scatter plot have not been recovered; these figures are documented historical results and have not been independently reproduced in this repository.

## Reproduction workflow

[Reconstructed analysis script](analyze.py) · [Recorded results](recorded_results.json)

Obtain the original prefiltered FGK FITS catalogue from your own archive or construct and document a matching public catalogue from [LAMOST](https://www.lamost.org/) / [VizieR](https://vizier.cds.unistra.fr/). The previous workflow selected effective temperatures 3,700–7,500 K; positive masses and luminosities; mass below 2.5 solar masses; and luminosity below 100 solar luminosities. Earlier working code used log g > 4.1, while the later report described log g > 4.0: the portable script exposes the threshold as a parameter and defaults to 4.1.

```bash
pip install numpy astropy matplotlib
python analyze.py FGK.fit --logg-min 4.1 --out output
```

Use `--prefiltered` only when temperature and gravity cuts have already been applied to the input file and the optional columns are absent. Column names may be overridden. The script fits the log–log relation, saves a density plot and writes JSON fit results. It does not recreate unknown sample selection or uncertainty handling, and does not fabricate data to reproduce the historical slope.

Compare the new fit with the archived result and report differences.
