# Boundary-Dominated Curvature Response in Weighted Network Defects

**Exact Motif Laws and Continuum Tests** · Unpublished manuscript version 1.0 · 25 July 2026

![Conceptual weighted network](../../portfolio-assets/gravity-concept.svg)

## Research question

For a graph with a localized defect, what controls the baseline-corrected weighted Forman-inspired curvature diagnostic: interior coupling alone, or interface motifs?

## Archived findings

| Test | Result reported in July manuscript |
|---|---|
| Critical boundary-coupling relation | Agreement with independent Dijkstra/bisection calculation within 3.1 × 10⁻¹⁶ in evaluated cases |
| Exact motif decomposition | Pure interior channels absent under the chosen edge classification |
| Direct vs. closed-form curvature | Approximately 10⁻¹⁵ numerical precision over 1,125 parameter points |
| Continuum/source tests | Support within an explicitly resolved validity domain |
| Poisson illustration | Expected 1/r potential and inverse-square force only after insertion of the equation |

## Code and reproduction

- [Closed-form symbolic curvature validation](closed_form_symbolic_curvature_validation.py)
- [Motif-count derivation](motif_count_derivation.py)

Install NumPy and matplotlib in a separate Python environment. Inspect each script's parameters and output paths before running, as original scripts were authored for a local workstation. Additional models and the complete final package are archived separately.

The supplementary verification report states that core results were rerun or independently/archivally checked; a complete uninterrupted run of all canonical scripts did not finish in the original time budget. This repository update does not constitute a new verification of those results.

**Scope:** exact consequences of the specified graph, weights and diagnostic. Physical spacetime, general relativity and quantum gravity are not derived.
