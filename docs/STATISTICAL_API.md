# Statistical routines for reuse

We expose the covariance estimators as a small Python API in addition to the complete study scripts. We use the same numerical equations and admissibility diagnostics as the archived routines.

## Matched daily observations

We arrange a daily matrix with columns in this order:

    CONUS404, CONUS404BA, MRMS Radar-Only QPE, PERSIANN-CDR

Rows must refer to the same UTC dates and the same spatial support. Before calculating a covariance, we retain only rows that are finite and nonnegative in all four columns. The full gridded pipeline additionally applies storm counts, minimum matched days per storm, duplicate-date checks, and window-specific eligibility. The covariance API by itself cannot enforce those temporal requirements because it receives moments, not dated observations.

For an existing matched daily array named `matched_daily`, we calculate:

```python
import numpy as np
from tc_precip_collocation import four_product_ec, triple_collocation

# matched_daily has shape (n_days, 4), with the column order given above.
assert matched_daily.ndim == 2 and matched_daily.shape[1] == 4
assert np.isfinite(matched_daily).all() and (matched_daily >= 0).all()
covariance = np.cov(matched_daily, rowvar=False, ddof=1)

ec = four_product_ec(covariance)
triplet_a = triple_collocation(covariance[np.ix_([0, 2, 3], [0, 2, 3])])
triplet_b = triple_collocation(covariance[np.ix_([1, 2, 3], [1, 2, 3])])

print(ec['valid_primary_ec'])
print(ec['error_variance'])
print(ec['error_correlation_model_pair'])
print(triplet_a['valid_triplet'], triplet_b['valid_triplet'])
```

We accept batches of covariance matrices with trailing dimensions (4,4) for EC and (3,3) for TC/ETC. The product order remains fixed. The routines return arrays and validity flags rather than silently forcing an inadmissible estimate into its physical range.

## What the outputs represent

- `error_variance` in four-product EC contains the two primary model estimates and the two averaged reference diagnostics used in the original EC calculation.
- `error_covariance_model_pair` and `error_correlation_model_pair` describe the permitted CONUS404-CONUS404BA error dependence.
- `valid_primary_ec` applies to the model-error block and its associated numerical conditions; it is not a bootstrap screen.
- `reference_diagnostics_valid` is separate from the primary EC validity flag.
- `error_variance_unmasked` in each triplet preserves its separate error-variance formulations, including negative diagnostic values where algebraically defined.
- `valid_product_estimate` and `valid_triplet` distinguish product-level validity from validity of all three members.
- `estimated_r2_to_truth_unmasked` is a latent-signal diagnostic provided where the API's product validity conditions permit it. The complete triplet script also retains the additional algebraic diagnostics used in the maps.

We do not interpret raw native-unit error variance as mean bias or direct-reference RMSE. A rescaled or transformed series has a different error scale; raw and log1p error SD cannot be compared as if both had units of mm/day.

## Uncertainty and agreement

We use `scripts/run_gridded_extended_collocation.py` for storm-block resampling and omission sensitivity because the dated observations and storm labels are required. We use `scripts/extract_direct_reference_agreement.py` for the paper's daily bias and RMSE table, without EC or bootstrap screening. The separate older EC-tail agreement diagnostics are optional and are not p-values.

We provide a fully runnable synthetic example in `examples/run_synthetic.py`. We use the tests in `tests/` to check known covariance moments, algebraic boundary cases, sampling rules, and overlap-area remapping. To adapt the analysis to a new set of products, we specify and justify the permitted error-correlation structure before using the four-product equations.
