# Reference

## Testing

```{eval-rst}
.. autofunction:: spicyr.spicy
```

## Results

```{eval-rst}
.. autoclass:: spicyr.SpicyResults
   :members: top_pairs, bind, box_plot, signif_plot, coefficient, p_value, se
```

The attributes of a results object:

| Attribute | Contents |
|---|---|
| `cell_results` | one row per pair (and per group with more than two groups): `excess_ref`, `excess_comp`, `excess_difference`, `se`, `df`, `p_value`, `p_adj`, `tau2`; with covariates or `adjust_abundance=True`, what the test was adjusted for (`adjusted_for`), the effect and p-value of each adjustment (`<covariate>_effect`, `<covariate>_p_value`, `abundance_effect`) and the unadjusted test (`unadjusted_difference`, `unadjusted_se`, `unadjusted_df`, `unadjusted_p_value`, `unadjusted_p_adj`). For survival: `score_coefficient`, `p_value`, `p_adj`, `hazard_ratio_sd`, `hr_p_value` and, with `adjust_abundance=True`, the p-value and hazard ratio without it. With several radii: `r` and `p_value_best_radius` |
| `radius_results` | with several radii, one row per pair and radius |
| `pairwise_assoc` | the effect of every pair in every image (what `bind()` returns) |
| `image_weights` | the weight of every image in the test of every pair: its share of its group's information (the point sizes of `box_plot()`) |
| `levels`, `image_ids`, `condition`, `subject`, `r`, `k`, `effect` | the groups, images and settings of the analysis (`effect`: "allocation" or "count") |

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
