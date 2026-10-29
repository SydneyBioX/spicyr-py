# spicyr news

## 1.99.33

- Self-pairs (`from_` = `to`): each image's variance is the exact random-labelling variance of the self-pair, for the
  radius graph and for k nearest neighbours (fraction: Var(O); count: Var(O - pL)). It was the variance of a pair of
  two types, which for a self-pair is too small. As spicyR 1.99.48.
- Allocation effect: an image whose room on the pair's side (n_to - E when attracting, E when avoiding) is below one
  cell is not used; the side is chosen first, from every image.
- The variance of the difference is each condition's own CR2 variance; the dispersion is no longer pooled over the
  conditions.
- Every requested pair has a row in `cell_results`; the new column `not_tested` gives why a pair (or a radius) was
  not tested, and its p-value is NaN and not counted in the BH adjustment. With more than two conditions a pair that
  could not be tested is listed instead of stopping the analysis.
- Faster with many patients: each image's patient is looked up directly.
- `tests/shared_cases/make_neff_cases.R` and `make_noinfo_cases.R` (the generators of `neff_expected.json` and
  `noinfo_expected.json`) are now in the repository.

## 1.99.32

- `top_pairs()` shows `side` only with `full=True`: the sign of the effects gives the direction. The tutorial
  describes only the columns `top_pairs()` shows by default. As spicyR 1.99.44.

## 1.99.31

- `top_pairs()` lists the columns of the test: `from`, `to`, `side`, `excess_ref`, `excess_comp`,
  `excess_difference`, `p_value` and `p_adj`, with `r` for several radii, `level` for more than two conditions,
  `hazard_ratio_sd` for survival and each covariate's `<covariate>_effect` and `<covariate>_p_value`. `full=True`
  gives every column of `cell_results`. `from_` and `to` choose pairs (an error names a cell type with no pair),
  `n=None` lists every pair, and with `cutoff` and no `n` every pair below the cutoff is listed, not only the first
  ten. The spicyR 1.x columns `intercept`, `coefficient`, `p.value` and `adj.pvalue` are gone. As spicyR 1.99.42.
- `bind()` has the image IDs as index, and `bind(from_=, to=)` gives one pair's table: `imageID`, `condition`,
  `subject`, `effect` and each image's `weight` in the test.
- `plot_image()` takes several images and draws them as panels of one figure, in the order given, sharing the axes
  and the density scale, and returns the figure.
- `radius_plot()` shows -log10 p at each radius for chosen pairs of a result with several radii, as spicyR's
  `radiusPlot()`.
- `calibration_check()` returns a `SpicyCalibration`, still a DataFrame, with `summary()` (the mean percentage over
  the shuffles at each threshold, with the expected one) and `plot()` (each shuffle against the expected
  percentage).
- The printed results give the patients in each group (`Units: 14 patients (A 7, B 7) with 28 images`) and point to
  `top_pairs()`.
- The tutorial uses these functions in place of hand-written code.
- Shared cases (`tests/shared_cases/make_accessor_cases.R`) check `top_pairs()` and `bind()` against spicyR's
  `topPairs()` and `bind()`.

## 1.99.30

- `write_cell_store()` works with anndata 0.13, which lists `X` among the layers (as `layers[None]`); `X` was
  written a second time there and failed. Tested with anndata 0.13.4, zarr 3.4.1 and pandas 3.0.6 (as CI) and
  with anndata 0.12.19.

## 1.99.29

- `write_cell_store()` writes an AnnData object whole, as spicyR's `writeCellStore()` writes a SpatialExperiment:
  `X`, `layers`, `var`, `obsm`, `varm`, `obsp`, `varp` and `uns` are written by anndata's own `write_elem()`,
  chunked (a sparse matrix's arrays in chunks of `chunk_rows` values, a dense matrix in chunks of whole rows), with
  the table's `obs`, region key and instance IDs as before. An `obsm["spatial"]` that holds the coordinates is
  written as it is.
- The cells of each image are also written as a points element named after the image (x, y and the instance ID
  that links each point to its row of the table), so the table annotates elements of the store and spatialdata reads
  it without warnings. Each chunk of rows is sorted by image and each image's rows appended to its own parquet
  file, so every element is one file and memory stays that of a chunk. `points=False` skips them; image IDs that
  are not valid element names give a warning and no points. Needs pyarrow (in the `store` extra).

## 1.99.28

- `col_test()` takes `subject`: with a table of cells and `feature`, the proportions are then those of each
  patient's cells, all images pooled, and each patient is one unit of the test. As spicyR 1.99.38.
- `col_test(type="wilcox")` reports the median in each group (`"median in group <level>"`), as the t-test reports
  the means.
- `col_test()` matches the rows of `df` to a `condition` Series or DataFrame whose index holds row names of `df`,
  and leaves out rows without a condition; otherwise the condition is taken in row order, as before.
- Values are rounded to two significant digits exactly as R's `signif()` does (scale, round half to even,
  unscale), so the few values that fall on a tie in floating point (such as 0.205000...02, which R rounds to 0.2)
  now match R as well; checked on 2,006 values against R.

## 1.99.27

- `calibration_check()` shuffles the condition (or the survival time and event together) across patients, or across
  images without `subject`, runs `spicy()` on each shuffle and reports the percentage of pairs below each p-value
  threshold: the check the tutorial recommends before trusting a discovery. Patients without a condition keep it
  missing, a categorical condition keeps its categories, and the input is left as it was. As spicyR 1.99.37's
  `calibrationCheck()`.
- `plot_image()` without `from_` and `to` shows every cell of the image, coloured by its cell type, with the same
  colour for a cell type in every image (as spicyR's `plotImage()` since 1.99.33).
- The tutorial is simpler: `imageID` and `cellType` are set once (noted as optional), `ER.Status` is used as it is
  (`neg` is the reference), the two example cores are named and
  drawn with `plot_image()`, the shuffles use `calibration_check()`, and the store section writes the whole data set
  and compares with the results in memory.

## 1.99.26

- Cell weights: `spicy(cell_weights=, weight_labels=)` takes weights in [0, 1], in the order of the cells: a 1-D
  array (numpy array, list or pandas Series) with one weight per cell for its own cell type, for example a quality
  score, or a cells x labels matrix of soft cell type membership (a pandas DataFrame, whose columns are the labels,
  or a numpy array or scipy sparse matrix with `weight_labels`), for example a classifier's confidence. A 1-D array
  is the same as the cells x cell types matrix with each cell's weight in its own type's column. With a matrix,
  `from_` and `to` name labels, which may overlap. A `to` cell counts with its weight, its score is the chance that
  it has a `from` neighbour within `r` if the labels were drawn from the weights (allocation) or the summed `from`
  weight of its neighbours (count), and the null shuffles the `to` weights over the cells whose `from` weight is
  below 1. 0/1 weights give the cell-type results. One radius, cells in memory; pairs of a label with itself are not
  tested. As spicyR 1.99.32 (core synced from it).

## 1.99.25

- A radius is now not tested when a condition's effective patients fall below min(3, m_g / 2), where m_g is the number
  of that condition's patients that contribute at that radius; before, the bound was 3. For conditions of 6 or more
  patients nothing changes. The rule applies wherever the old one did: `spicy()` (both effects, with or without
  adjustment, Hartung-Knapp, and each level with more than two conditions). The label stays
  `"few_effective_units"`.
- Small studies get most of their tests back. Over the small real-data analyses, the radii of the grid that are tested
  rise from 78% to 95%, and a condition of 3 patients can now be tested (none could before).
- Core synced from spicyR 1.99.30.

## 1.99.24

- `spicy()` (both effects) no longer tests a radius where a condition has fewer than 3 effective patients: Kish's
  (sum w)^2 / sum w^2 of the patients' weights in that condition. A patient's weight grows with the information of its
  images, and when the patients do not differ (frailty variance 0) nothing caps it, so one patient with many cells can
  carry nearly all of a condition's weight. Such a radius has p-value NaN (and the unadjusted one; with adjustment,
  `adjusted_for` is `"none (few_effective_units)"`) and is left out of the combination over radii, as a radius with no
  informative patient is. With more than two conditions, a pair where any condition has fewer than 3 is not tested. A
  condition with fewer than 3 patients is never tested, and one with 3 or 4 rarely is.
- The test now shares one dispersion between the conditions. Each condition's CR2 variance is divided by its
  model-based variance, the ratios are averaged with the conditions' numbers of patients as weights, and the
  model-based variance of the difference is multiplied by this average; the Satterthwaite df are unchanged. Before,
  each condition's CR2 variance entered on its own, so with conditions of unequal size and skewed patient effects a
  small condition's low spread set the standard error and the test gave too many small p-values. With adjustment, the
  ratios are those of each condition's coefficient in the adjusted design, and every coefficient's variance (also the
  reported effect of each adjustment) is scaled by their average. `variance="hartung_knapp"` is unchanged.
- Images with a missing condition are now also left out of the choice of an allocation pair's side and of the
  between-image variance (`tau2_image`), so the result equals the one with those images removed, also with adjustment
  and several images per patient.
- Survival is unchanged.
- Calibration, on 11 real-data analyses with 30 relabellings of the conditions between patients each (spicyR 1.99.29,
  which gives the same results), as the share of relabellings with any BH-significant pair over 25 radii (davies): for
  the allocation effect, from 37% to 3.3% in the worst analysis (404 patients, 340 against 64), from 23% to 3.3% in
  the next (one patient carried 99% of a condition's weight), and from 5.2% to 3.0% on average over the other 9; for
  the count effect, from 33% to 3.3%, from 27% to 0% and from 4.1% to 3.0%. With the true labels, BH-significant pairs
  go from 535 to 510 (allocation) and from 641 to 546 (count).
- Of the 62 reference configurations, the 4 of survival and the 2 with Hartung-Knapp and covariates are identical to
  1.99.23's; the 28 of each effect change in their standard errors and p-values.
- The shared C++ core is synced from spicyR 1.99.29.

## 1.99.23

- `spicy()` leaves images whose condition is missing out of the unadjusted test. Before, the test of two conditions
  counted them in the second condition: its excess, the difference, the standard error, the df, the p-value and the
  frailty variance all included them. This affected the test without adjustment and the `unadjusted_*` columns of an
  adjusted test. The adjusted design already left these images out, so its results are unchanged, except where its
  frailty variance is held at the unadjusted value (an added column that varies between the images of a patient, such
  as `adjust_abundance=True` with several images per patient). There the held variance is now that of the images with
  a condition, and the adjusted test changes with it (spicyR#83).
- With more than two conditions, images with a missing condition entered the test with no condition and changed its
  results. They are now left out, as with two conditions.
- The test now equals the one with these images removed, except for two steps that use every image: the side of an
  allocation pair (attraction or avoidance), and the between-image variance (`tau2_image`) when patients have several
  images.
- Results with no missing condition are unchanged: every reference configuration is identical to 1.99.22's.
- The shared C++ core is synced from spicyR 1.99.28, which gives the same results.
