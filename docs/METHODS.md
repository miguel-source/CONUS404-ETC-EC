# Computational methods

We document the final gridded analysis used in our study. We retain the watershed-average routines as an optional alternative; those routines use a different statistical support and do not reproduce the primary gridded results.

## Study sample and event windows

We evaluated Matthew (2016), Harvey (2017), Irma (2017), Florence (2018), Imelda (2019), Laura (2020), Zeta (2020), Eta (2020), Ida (2021), and Ian (2022). Their assigned watershed domains comprise 23 unique watersheds: 20 HUC6 and three HUC8 domains.

We interpolated consecutive Atlantic HURDAT2 positions along great-circle paths at hourly intervals. The implementation uses a spherical Earth radius of 6,371,008.8 m and does not interpolate gaps longer than 18 hours. We measured the shortest distance from each sampled storm position to each watershed polygon. The 500-km criterion is centered on the moving storm position; it is not a buffer around a watershed centroid. The original track-processing routine retained all available status records for a selected storm and did not impose a TS/HU-only filter.

For each assigned watershed, we defined the core interval from one calendar day before first contact within 500 km through two days after last contact. We combined the assigned watershed intervals to obtain each storm's calendar window. We added four preceding and one following day to form the expanded window. The checked-in event inventory records these decisions. A later HURDAT2 revision can change the windows, so reproducing the paper requires the recorded inventory rather than silently regenerating it from the newest tracks.

## Daily precipitation and common grid

We used UTC daily accumulation intervals [D 00:00, D+1 00:00). For MRMS RadarOnly_QPE_24H, we assigned the field valid at D+1 00:00 UTC to day D. We treated negative and nonfinite source values as missing, not as dry days. The code raises a quality-control error for daily precipitation exceeding the configured 2,000-mm ceiling; it does not clip those observations.

We retained PERSIANN-CDR on its native 0.25-degree longitude-latitude grid (EPSG:4326). We aligned CONUS404, CONUS404BA, and MRMS using source-target overlap areas measured in the CONUS Albers equal-area projection (EPSG:5070). For target cell j and day t, our depth estimate is:

    P[j,t] = sum_i(A[i,j] * valid[i,t] * P[i,t]) / sum_i(A[i,j] * valid[i,t])

Here A[i,j] is the source-target intersection area, and valid indicates a finite, nonnegative source value. This is an overlap-area-weighted mean precipitation depth, not a sum of millimeters from finer cells. We required at least 0.95 valid source-area coverage and at least 0.98 geometric grid coverage. The treatment of partially missing source areas is recorded in the preparation metadata. Any remapping already performed by an upstream data provider is distinct from this alignment step.

We used a 70-by-125 rectangular computational subset and retained the 683 cells whose centers intersected the watershed union. We also stored fractional watershed overlap for spatial weighting. Our summaries use watershed fraction multiplied by cosine of latitude; the area-coverage denominator includes all stored watershed fractions, including boundary fractions outside the center-selected mask.

## Matched observations and temporal pooling

At each selected cell, we required coincident, finite, nonnegative precipitation from all four products. A contributing storm required at least three matched days. Eligible cells required at least eight contributing storms and at least 60 core-window or 100 expanded-window observations.

Our final `all_windows` analysis pooled the selected storm-calendar days at every retained watershed cell. It does not establish that every storm directly affected every cell. We checked duplicate dates for identical precipitation fields and retained the first record in the fixed event order. One overlapping expanded-window date was removed, leaving 78 unique core days and 127 unique expanded days. The retained record determines the bootstrap storm block for that date.

## TC and ETC triplets

We evaluated two triplets separately:

- A: CONUS404, MRMS Radar-Only QPE, PERSIANN-CDR.
- B: CONUS404BA, MRMS Radar-Only QPE, PERSIANN-CDR.

For covariance matrix C of a matched triplet X, Y, Z, the native-unit TC random-error variance of X is:

    error_variance[X] = C[X,X] - C[X,Y] * C[X,Z] / C[Y,Z]

The ETC squared correlation with the unknown latent precipitation signal is:

    latent_R2[X] = C[X,Y] * C[X,Z] / (C[X,X] * C[Y,Z])

We applied analogous expressions to the other triplet members. TC and ETC use the same error-variance equation; ETC adds latent-correlation estimation. We calculated random-error SD as the square root of a nonnegative variance. Under the zero-mean error model this is random-error RMSE, and it differs from RMSE calculated directly against another precipitation product.

We preserved negative variances and out-of-range latent R2 estimates as diagnostics. A negative variance is an inadmissible algebraic estimate, not negative physical uncertainty. We did not clip negative estimates to zero. Separate-triplet results retain separate MRMS and PERSIANN-CDR estimates; those reference estimates were not averaged across triplets in the final triplet maps and tables.

## Four-product extended collocation

We modeled each product as an affine response to latent precipitation plus zero-mean error. We allowed only the CONUS404-CONUS404BA error covariance to be nonzero, equivalent to error groups [1, 1, 2, 3]. We assumed the MRMS and PERSIANN-CDR errors to be uncorrelated with each other and with both model errors. These are modeling assumptions, not demonstrated independence.

We estimated model-error correlation by dividing the CONUS404-CONUS404BA error covariance by the product of their error SDs. This is correlation between their inferred random errors; it is neither their observed precipitation correlation nor their latent-signal R2. The five other pairwise error correlations are constrained by the model and cannot be reported as independently estimated zeros.

In our implemented equations, individual CONUS404 and CONUS404BA error variances agree with their corresponding triplet values at identical cells and samples. Different reported means can result from different masks. The original EC files contain averaged MRMS and PERSIANN-CDR variance diagnostics; we kept these distinguishable from the unaveraged triplet estimates.

## Bootstrap and storm-omission sensitivity

We drew 5,000 bootstrap samples of complete storm blocks with seed 404 and used the same storm draws across the map. We formed pixelwise intervals from the 2.5th and 97.5th percentiles of admissible estimates. We retained the paired model comparison where the point estimate met the primary criteria, the paired difference-error SD was finite and positive, and at least 90% of bootstrap replicates were admissible. We did not require reference-diagnostic bootstrap admissibility for this model-pair subset.

For the paired difference in error SD (CONUS404BA minus CONUS404), we classified intervals as entirely positive, entirely negative, or including zero. These are conditional pixelwise intervals, not simultaneous spatial confidence bands or a confidence interval for the domain mean. Bootstrap resampling does not correct structural violations of collocation assumptions.

We omitted each storm once, recalculated the estimates, and rechecked eligibility. We summarized omission results using the fixed full-sample paired spatial subset, with the availability of omission estimates recorded. We used this analysis to assess event sensitivity, not to construct confidence intervals or rank individual storms by their own error variance.

## Daily agreement with comparison references

We calculated model-minus-reference daily bias, MAE, and RMSE for CONUS404 and CONUS404BA against MRMS and PERSIANN-CDR. All four comparisons used the same matched dates and sample-eligible cells within a window. We imposed no EC-admissibility or bootstrap screen on these diagnostics.

For each cell we first calculated temporal bias and RMSE. We then area-averaged those cell-level metrics. The reported mean pixelwise RMSE is not a single RMSE pooled across all cell-days. Positive bias means wetter model precipitation. MRMS and PERSIANN-CDR are imperfect references; these metrics quantify disagreement with them.

## Spatial subsets and transformations

| Summary | Core cells | Expanded cells | Interpretation |
|---|---:|---:|---|
| Matched sample-eligible cells | 683 | 683 | Direct-reference agreement; no collocation screening |
| Primary EC point-admissible cells | 671 | 682 | Point-estimate criteria before paired bootstrap screening |
| Retained paired EC comparison | 527 | 660 | 70.81% and 89.25% of weighted watershed area |
| Common admissible cells across both triplets | 314 | 382 | Same cells for all six triplet products within each window |
| Product-specific map cells | Varies | Varies | Finite nonnegative variance/SD or latent R2 in [0,1] |

We used raw precipitation as the primary analysis and log1p(P) as a sensitivity analysis. Log-transformed variance and SD have transformed units and must not be compared numerically with raw mm-per-day estimates. Dimensionless latent R2 and model-error correlation can be compared across transforms when their retained cells and interpretation are stated.

## Method references

- McColl et al. (2014), Extended triple collocation: https://doi.org/10.1002/2014GL061322
- Gruber et al. (2016), Extended collocation and error cross-correlations: https://doi.org/10.1002/2015JD024027
- Yilmaz and Crow (2014), Collocation assumptions: https://doi.org/10.1175/JHM-D-13-0158.1
