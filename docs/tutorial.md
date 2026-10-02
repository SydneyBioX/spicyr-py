---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tutorial

This tutorial is a notebook: it runs every time the documentation is built, so the output below is what the
current version of spicyr produces.

It uses the small simulated data set shipped with the package: 14 patients, 7 in each of two conditions (A and
B), two images per patient and six cell types. In condition B, a third of the T cells were moved next to tumour
cells, so the co-localisation of T cells and tumour cells is the true change. Everything else is random.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import spicyr

pd.set_option("display.precision", 3)
cells = spicyr.datasets.example_cells()
cells[["imageID", "patient", "condition", "cellType", "x", "y"]].head()
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
for ax, img in zip(axes, ["A01_1", "B01_1"]):
    z = cells[cells["imageID"] == img]
    for t, g in z.groupby("cellType"):
        ax.scatter(g["x"], g["y"], s=6, label=t)
    ax.set_title(f"{img} (condition {z['condition'].iloc[0]})")
    ax.set_aspect("equal")
axes[1].legend(markerscale=2, fontsize=8, bbox_to_anchor=(1, 1))
plt.tight_layout()
```

## Test every pair

Give the condition, the patient of each image (`subject`) and the radius. Patients, not images, are the units.

```{code-cell} ipython3
res = spicyr.spicy(cells, condition="condition", subject="patient", r=30)
res
```

```{code-cell} ipython3
res.top_pairs(5)
```

`T__tumour` reads *from* T *to* tumour: the number of extra tumour cells within 30 units of each T cell. That is
the direction the moved T cells change most, and the direction whose excess does not depend on how common tumour
cells are (the recruited type is at the centre). `coefficient` is the change in the excess from condition A (the
reference, `intercept`) to condition B.

```{code-cell} ipython3
res.signif_plot();
```

Each disc is a pair: its left half is coloured by the excess in condition A, its right half by the excess in
condition B, the size grows with -log10 p and a black ring marks p < 0.05 (`fdr=True` uses the BH-adjusted p).
This is the same plot as `signifPlot()` in spicyR.

```{code-cell} ipython3
res.signif_plot(fdr=True);
```

## One pair

```{code-cell} ipython3
res.box_plot("T", "tumour");
```

The full table has one row per pair. `df` are Satterthwaite degrees of freedom: with 14 patients they are about
11, not thousands of cells.

```{code-cell} ipython3
res.cell_results.loc[["T__tumour", "tumour__T"],
                     ["excess_ref", "excess_comp", "excess_difference", "se", "df", "p_value", "p_adj"]]
```

## Abundance or attraction?

Each pair also gets the difference at equal availability of the counted type (`adjusted_*`): the excess
adjusted for the log share of the `to` type in each image. Here composition does not differ between conditions,
so the two agree.

```{code-cell} ipython3
t = res.cell_results
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.scatter(t["excess_difference"], t["adjusted_difference"], s=15)
lim = [t[["excess_difference", "adjusted_difference"]].min().min(), t[["excess_difference", "adjusted_difference"]].max().max()]
ax.plot(lim, lim, ls="--", c="grey")
ax.set_xlabel("difference in excess")
ax.set_ylabel("difference at equal availability");
```

## Several radii

Give several radii and they are combined by a max-T test that uses the correlation between radii, estimated from
the patients' CR2 influences (`combine="cauchy"` is an alternative). The reported effect is the one at the
radius with the strongest evidence.

```{code-cell} ipython3
m = spicyr.spicy(cells, condition="condition", subject="patient", r=[10, 20, 30, 40], from_="T", to=["tumour", "B"])
m.cell_results[["from", "to", "r", "excess_difference", "p_value", "p_value_best_radius"]]
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 3.5))
for (f, to), g in m.radius_results.groupby(["from", "to"]):
    ax.plot(g["r"], g["excess_difference"], marker="o", label=f"{f} -> {to}")
ax.axhline(0, c="grey", ls="--")
ax.set_xlabel("radius")
ax.set_ylabel("difference in excess (B - A)")
ax.legend();
```

## Covariates

```{code-cell} ipython3
cv = spicyr.spicy(cells, condition="condition", subject="patient", r=30, from_="T", to="tumour",
                  covariates=["age", "batch"])
cv.cell_results[["excess_difference", "p_value", "covariate_difference", "covariate_p_value"]]
```

The unadjusted difference differs slightly from the all-pairs analysis above, because the label-clustering
factor of a cell type is pooled over the pairs requested.

## Survival

Give the follow-up time as `condition` and the event indicator as `survival` (one value per patient).

```{code-cell} ipython3
s = spicyr.spicy(cells, condition="time", survival="event", subject="patient", r=30, from_="T", to=["tumour", "B"])
s.cell_results[["from", "to", "score_coefficient", "p_value", "hazard_ratio_sd", "hr_p_value"]]
```

The p-value is a score test: each patient's excess is regressed on the martingale residual of a Cox model with no
spatial term, so it does not rely on the Cox model holding. The hazard ratio is per standard deviation of the
patients' excess after shrinkage. (In these simulated data survival is unrelated to the spatial pattern.)

## From AnnData and SpatialData

```{code-cell} ipython3
import anndata as ad

obs = cells.drop(columns=["x", "y"])
obs.index = [f"cell{i}" for i in range(len(obs))]
adata = ad.AnnData(obs=obs, obsm={"spatial": cells[["x", "y"]].to_numpy()})
spicyr.spicy(adata, condition="condition", subject="patient", r=30).top_pairs(3)
```

For a SpatialData object, `spicy()` reads its annotation table (`table_key`). The images come from the table's
region key, and the coordinates from `obsm["spatial"]` or from the centroids of the annotated shapes or labels:

```python
res = spicyr.spicy(sdata, condition="condition", subject="patient", image_id="region", table_key="table")
```
