# Reproducibility and verification

We distinguish numerical validation, reproduction of available study summaries, and a complete rerun from native source archives. We completed the first two checks during packaging; we did not repeat the full source-data download or the full 5,000-replicate observational calculation.

## What we checked

We ran thirteen automated tests covering known TC/EC covariance moments, the identity of the model variances across the implemented EC and triplet equations, negative-variance and zero-denominator cases, duplicate-date selection and conflicts, minimum storm-day requirements, constant-field remapping, heterogeneous-field volume conservation, missing-area masking, and PERSIANN daily-to-event assembly. The assembly checks verify native values, coordinates, units, filename dates, missing-day rejection, and grid-mismatch rejection. All thirteen tests passed in the available scientific Python environment.

We also ran the synthetic ten-storm gridded example, including EC, 64 bootstrap replicates, and leave-one-storm-out estimates. We use 64 draws only for this quick numerical example; our observational study used 5,000 draws with seed 404.

An independent abstract-syntax-tree comparison confirmed that the copied EC, dataset-agreement, gridded-preparation, and shared preparation functions preserve the original scientific routines. The separate-triplet changes specify the NetCDF backend without changing its formulas or masks.

We reran the portable direct-reference routine on the available aligned observational archive. All eight bias and RMSE summaries matched the saved study benchmarks exactly. We also reproduced its 683-cell coverage, 78 core dates, 127 expanded dates, and removal of one duplicate expanded date.

For the supplemental figures, we checked grid alignment, source-file content hashes, paired-selection masks, the identity of error correlation with covariance divided by the two error SDs, common raw/log subsets, negative-variance counts, raster dimensions, 600-dpi metadata, and text placement. We visually reviewed all six figures.

## Observational benchmarks

These values are rounded reference summaries from the saved study results. They are not independent observations and must not be used as additional statistical samples.

| Quantity | Core | Expanded |
|---|---:|---:|
| Unique matched calendar days in the window design | 78 | 127 |
| Sample-eligible center-selected cells | 683 | 683 |
| Primary EC point-admissible cells | 671 | 682 |
| Retained paired-comparison cells | 527 | 660 |
| Retained paired weighted watershed area (%) | 70.8053 | 89.2457 |
| Common admissible cells across the two triplets | 314 | 382 |
| Mean model-error correlation on paired cells | 0.975724 | 0.976368 |
| CONUS404 mean random-error SD on paired cells (mm/day) | 14.3724 | 10.9229 |
| CONUS404BA mean random-error SD on paired cells (mm/day) | 15.2611 | 11.5462 |
| Mean paired SD difference, BA minus CONUS404 (mm/day) | 0.888697 | 0.623317 |

The mean SD is an area-weighted arithmetic mean of pixelwise SDs, not the square root of the mean variance. The paired and common-triplet subsets are not required to be nested. We use separate validity criteria for product-specific maps.

| Direct daily model-reference comparison | Core bias | Core RMSE | Expanded bias | Expanded RMSE |
|---|---:|---:|---:|---:|
| CONUS404 minus MRMS | +0.012689 | 14.709888 | -0.119035 | 12.655920 |
| CONUS404BA minus MRMS | +0.191341 | 15.630356 | +0.016390 | 13.372189 |
| CONUS404 minus PERSIANN-CDR | +1.726821 | 16.765492 | +0.976217 | 14.187858 |
| CONUS404BA minus PERSIANN-CDR | +1.905472 | 17.525015 | +1.111643 | 14.788537 |

All four comparisons use the same matched dates and 683 eligible cells within a window. Units are mm/day. These are weighted means of pixelwise temporal metrics, without EC or bootstrap screening.

## Running the checks

```bash
python -m pytest tests
python examples/run_synthetic.py
python scripts/run_gridded_extended_collocation.py --self-test
python scripts/run_pipeline.py --config configs/study.example.json --dry-run
```

We provide an environment specification and dependency ranges for reuse. We did not validate a newly solved conda environment on every operating system, perform fresh network transfers from every provider, or benchmark the full observational runtime during release preparation. Native GRIB reading requires ecCodes/cfgrib; the acquisition and event-window stages also need their stated geospatial/catalog dependencies.

## Reusing the method for another study

We designed the numerical routines for reuse with explicit inputs and settings. Several publication-figure and manuscript-summary scripts additionally assert our exact grid, sample sizes, or benchmark values. Those checks protect reproduction of this paper. For a different region, storm sample, grid, or threshold configuration, we update those study-specific checks and captions deliberately rather than interpreting an assertion failure as a reason to loosen scientific screening.

We preserve the distinction between the final gridded `all_windows` workflow and the optional watershed-average or event-assigned alternatives. The older EC-tail agreement score is not a p-value and is not the direct-reference RMSE used in the final paper.

## Before a public release

We credit Miguel Diaz in `CITATION.cff` and complete the public repository URL and archived release metadata when available. We retain the data attribution notices, freeze the event inventory and relevant input versions, review the generated validation report, and archive a tagged software release. We have not assigned a DOI or published this local package to GitHub.
