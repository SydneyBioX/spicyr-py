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
| `cell_results` | one row per pair (and per group with more than two groups): `excess_ref`, `excess_comp`, `excess_difference`, `se`, `df`, `p_value`, `p_adj`, `tau2`, the `adjusted_*` columns, and the `covariate_*` columns when `covariates` is given. For survival: `score_coefficient`, `p_value`, `p_adj`, `hazard_ratio_sd`, `hr_p_value`. With several radii: `r` and `p_value_best_radius` |
| `radius_results` | with several radii, one row per pair and radius |
| `pairwise_assoc` | the excess of every pair in every image (what `bind()` returns) |
| `levels`, `image_ids`, `condition`, `subject`, `r`, `k` | the groups, images and settings of the analysis |

## Data

```{eval-rst}
.. autofunction:: spicyr.datasets.metabric_ali2020
.. autofunction:: spicyr.datasets.schurch2020
.. autofunction:: spicyr.datasets.example_cells
.. autofunction:: spicyr.format_data
```
