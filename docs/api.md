# Reference

## Testing

```{eval-rst}
.. autofunction:: spicyr.spicy
```

## Checking calibration

```{eval-rst}
.. autofunction:: spicyr.calibration_check
.. autoclass:: spicyr.SpicyCalibration
   :members: summary, plot
```

## Writing a store

```{eval-rst}
.. autofunction:: spicyr.write_cell_store
```

## Results

```{eval-rst}
.. autoclass:: spicyr.SpicyResults
   :members: top_pairs, top_follow_ups, bind, box_plot, signif_plot, radius_plot, coefficient, p_value, se
```

The attributes of a results object:

| Attribute | Contents |
|---|---|
| `cell_results` | every column of the test (what `top_pairs(n=None, full=True)` lists, unsorted): one row per pair (and per group with more than two groups): `excess_ref`, `excess_comp`, `excess_difference`, `se`, `df`, `p_value`, `p_adj`, `tau2`; with covariates or `adjust_abundance=True`, what the test was adjusted for (`adjusted_for`), the effect and p-value of each adjustment (`<covariate>_effect`, `<covariate>_p_value`, `abundance_effect`) and the unadjusted test (`unadjusted_difference`, `unadjusted_se`, `unadjusted_df`, `unadjusted_p_value`, `unadjusted_p_adj`). For survival: `score_coefficient`, `p_value`, `p_adj`, `hazard_ratio_sd`, `log_hr_se` (model-based and approximate), `n_deaths`, `not_tested` and, with `adjust_abundance=True`, the p-value and hazard ratio without it. With several radii: `r` and `p_value_best_radius`; with `combine="maxT"`, `maxT_converged` (whether the max-T p-value met its error target of about 0.1%) |
| `follow_up_results`, `alpha` | with more than two conditions, the follow-up comparisons of two levels for the pairs called by the overall test at level `alpha` (what `top_follow_ups()` lists) |
| `radius_results` | with several radii, one row per pair and radius (what `radius_plot()` shows) |
| `pairwise_assoc` | the effect of every pair in every image (what `bind()` returns) |
| `image_weights` | the weight of every image in the test of every pair: its share of its group's information (the point sizes of `box_plot()`, and the `weight` column of `bind(from_=, to=)`) |
| `levels`, `image_ids`, `condition`, `subject`, `r`, `k`, `effect` | the groups, images and settings of the analysis (`effect`: "allocation" or "count") |

## Proportions

```{eval-rst}
.. autofunction:: spicyr.get_prop
.. autofunction:: spicyr.col_test
```

## Plotting an image

```{eval-rst}
.. autofunction:: spicyr.plot_image
```

## Data

```{eval-rst}
.. autofunction:: spicyr.datasets.metabric_ali2020
.. autofunction:: spicyr.datasets.schurch2020
.. autofunction:: spicyr.datasets.example_cells
.. autofunction:: spicyr.format_data
```
