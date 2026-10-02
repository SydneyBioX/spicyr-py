# Tutorial

This tutorial uses the small simulated data set shipped with the package: 14 patients (7 per condition), two
images each, six cell types. In condition B, a third of the T cells were moved next to tumour cells.

```python
import spicyr

cells = spicyr.datasets.example_cells()
cells[["imageID", "patient", "condition", "cellType", "x", "y"]].head()
```

```text
  imageID patient condition cellType       x       y
0   A01_1     A01         A   tumour   70.65    0.16
1   A01_1     A01         A   stroma  123.93  383.34
2   A01_1     A01         A   stroma  344.57  223.78
3   A01_1     A01         A   tumour  269.71  154.52
4   A01_1     A01         A   stroma  125.52  181.27
```

## Test every pair

Give the condition, the patient of each image (`subject`) and the radius. Patients, not images, are the units.

```python
res = spicyr.spicy(cells, condition="condition", subject="patient", r=30)
res
```

```text
SpicyResults (spicyR Cell): 36 tests, conditions ['A', 'B']; 3 with BH-adjusted p < 0.05
```

```python
res.top_pairs(5)
```

```text
                  from      to  intercept  coefficient    p.value  adj.pvalue
T__tumour            T  tumour      0.010        0.299  6.083e-05       0.002
tumour__tumour  tumour  tumour      0.016       -0.184  2.037e-03       0.031
T__stroma            T  stroma     -0.095        0.114  2.554e-03       0.031
tumour__T       tumour       T      0.042        0.071  2.279e-02       0.205
B__tumour            B  tumour     -0.024       -0.097  3.381e-02       0.243
```

`T__tumour` is read *from* T *to* tumour: the number of extra tumour cells within 30 units of each T cell. That
is the direction the moved T cells change most, and the direction whose excess does not depend on how common
tumour cells are (the recruited type is at the centre). `coefficient` is the change in the excess from
condition A (the reference, `intercept`) to B.

## The full table

```python
res.cell_results.loc[["T__tumour", "tumour__T"], ["excess_ref", "excess_comp", "excess_difference", "se", "df", "p_value"]]
```

```text
           excess_ref  excess_comp  excess_difference     se      df    p_value
T__tumour       0.010        0.310              0.299  0.049  11.417  6.083e-05
tumour__T       0.042        0.113              0.071  0.027  11.287  2.279e-02
```

`df` are Satterthwaite degrees of freedom: with 14 patients they are about 11, not thousands of cells.

## Abundance or attraction?

Each pair also gets the difference at equal availability of the counted type (`adjusted_*`): the excess is
adjusted for the log share of the `to` type in each image.

```python
res.cell_results.loc[["T__tumour", "tumour__T"], ["excess_difference", "p_value", "adjusted_difference", "adjusted_p_value"]]
```

```text
           excess_difference    p_value  adjusted_difference  adjusted_p_value
T__tumour              0.299  6.083e-05                0.299         7.578e-05
tumour__T              0.071  2.279e-02                0.069         1.941e-02
```

Here composition does not differ between conditions, so the two agree.

## Several radii

```python
m = spicyr.spicy(cells, condition="condition", subject="patient", r=[10, 20, 40], from_="T", to=["tumour", "B"])
m.cell_results[["from", "to", "r", "excess_difference", "p_value", "p_value_best_radius"]]
```

```text
          from      to     r  excess_difference    p_value  p_value_best_radius
T__tumour    T  tumour  10.0              0.306  1.271e-06            4.361e-07
T__B         T       B  10.0              0.008  8.436e-01            5.564e-01
```

The radii are combined by max-T, using the correlation between radii estimated from the patients' CR2
influences (`combine="cauchy"` is an alternative). `m.radius_results` has every radius.

## Covariates

```python
spicyr.spicy(cells, condition="condition", subject="patient", r=30, from_="T", to="tumour",
             covariates=["age", "batch"]).cell_results
```

```text
           excess_difference    p_value  covariate_difference  covariate_p_value
T__tumour              0.307  6.938e-05                 0.306          8.909e-05
```

(The unadjusted difference differs slightly from the all-pairs analysis above because the label-clustering
factor of a target type is pooled over the pairs requested.)

## Survival

```python
s = spicyr.spicy(cells, condition="time", survival="event", subject="patient", r=30, from_="T", to=["tumour", "B"])
s.cell_results[["from", "to", "score_coefficient", "p_value", "hazard_ratio_sd", "hr_p_value"]]
```

```text
          from      to  score_coefficient  p_value  hazard_ratio_sd  hr_p_value
T__tumour    T  tumour              0.050    0.537            1.249       0.521
T__B         T       B             -0.008    0.802            0.863       0.766
```

The p-value is a score test: each patient's excess is regressed on the martingale residual of a Cox model with no
spatial term, so it does not rely on the Cox model holding. The hazard ratio is per standard deviation of the
patients' excess after shrinkage.

## From AnnData and SpatialData

```python
# AnnData: cells as observations, coordinates in obsm["spatial"]
res = spicyr.spicy(adata, condition="condition", subject="patient", image_id="imageID", cell_type="cellType")

# SpatialData: the annotation table; images from its region key, coordinates from obsm["spatial"]
# or the centroids of the annotated shapes or labels
res = spicyr.spicy(sdata, condition="condition", subject="patient", image_id="region", table_key="table")
```

## Plots

```python
res.signif_plot()                 # every pair: colour = difference, size = -log10 p, ring = BH < 0.05
res.box_plot("T", "tumour")       # per-image excess by condition
res.bind()                        # the per-image excess of every pair, as a table
```
