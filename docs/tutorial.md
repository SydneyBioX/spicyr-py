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

# Introduction to spicyr

spicyr tests whether two cell types sit closer together, or further apart, in one group of patients than in
another, and whether this is associated with patient outcome. This tutorial walks through an analysis of breast
cancer imaging mass cytometry data, from testing every pair of cell types to checking that the results can be
trusted. It is the same analysis, on the same data, as the vignette of the R package spicyR, and gives the same
results.

## Overview

Do T cells gather around tumour cells more in one group of patients than in another? Questions like this are
central to the analysis of spatial omics data, and spicyr answers them for every pair of cell types at once.

For a pair written *from* → *to*, spicyr counts the *to* cells within a radius of each *from* cell. It compares
that count with what we would expect if the cell-type labels were shuffled among the cells in the same image. The
difference is the **excess**: the number of extra *to* cells around each *from* cell, beyond chance. Because the
comparison uses the cells that are actually there, holes, folds and dense regions in the tissue do not create
false signals. spicyr then compares the excess between groups of patients, treating patients, not images or cells,
as the units of the test.

```{figure} _static/spicyR_overview.png
:width: 100%

For each proliferating tumour cell (red), spicyr counts the T cells (blue) within 25 µm and compares the count
with chance. Each patient gets an excess, and the excess is compared between ER-negative and ER-positive tumours.
```

## Installation

```bash
pip install "spicyr[data,plot]"
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spicyr

pd.set_option("display.precision", 4)
pd.set_option("display.width", 120)
```

## The data

We use imaging mass cytometry of breast tumours from the METABRIC cohort (Ali et al., *Nature Cancer* 2020). Each
of the 456 patients with known oestrogen receptor (ER) status contributed one tumour core, and the authors assigned
each cell to one of 22 types. `spicyr.datasets.metabric_ali2020()` downloads the same file that the R package
SpatialDatasets uses, from Bioconductor. Coordinates are in micrometres.

```{code-cell} ipython3
cells = spicyr.datasets.metabric_ali2020()
cells = cells[cells["ER.Status"].isin(["neg", "pos"])].copy()
cells["ER"] = pd.Categorical(np.where(cells["ER.Status"] == "pos", "ER+", "ER-"), categories=["ER-", "ER+"])

print(cells.drop_duplicates("metabricId")["ER"].value_counts().sort_index())
cells["description"].value_counts()
```

```{code-cell} ipython3
two_cores = [cells.loc[cells["ER"] == "ER-", "file_id"].iloc[0], cells.loc[cells["ER"] == "ER+", "file_id"].iloc[0]]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, core in zip(axes, two_cores):
    z = cells[cells["file_id"] == core]
    for t, g in z.groupby("description"):
        ax.scatter(g["x"], g["y"], s=2, label=t)
    ax.set_title(core); ax.set_aspect("equal"); ax.axis("off")
axes[1].legend(markerscale=4, fontsize=6, bbox_to_anchor=(1, 1), loc="upper left", frameon=False)
plt.tight_layout()
```

## Testing every pair of cell types

`spicy()` needs the column of the condition, the column of the patient (`subject`) and a radius. The radius should
reflect the scale at which you expect cells to interact; 25 µm is about two to three cell diameters.

```{code-cell} ipython3
res = spicyr.spicy(cells, condition="ER", subject="metabricId", r=25,
                   image_id="file_id", cell_type="description")
res
```

All 484 ordered pairs are tested in a few seconds. `top_pairs()` lists the most significant. `intercept` is the
excess in the reference group (ER−) and `coefficient` is the change in the excess from ER− to ER+, in extra *to*
cells per *from* cell.

```{code-cell} ipython3
res.top_pairs(8)
```

## Seeing every pair at once

`signif_plot()` shows the whole study. Each circle is a pair: the left half is coloured by the excess in ER−
tumours and the right half by the excess in ER+ tumours, the size reflects the p-value, and a black ring marks
significance.

```{code-cell} ipython3
res.signif_plot(fdr=True, breaks=(-2, 2, 0.5));
```

## Looking at one pair

We focus on T cells around proliferating hormone-receptor-negative tumour cells (`HR- Ki67+` → `T cells`).

```{code-cell} ipython3
res.cell_results.loc[["HR- Ki67+__T cells"], ["excess_ref", "excess_comp", "excess_difference", "p_value", "p_adj"]]
```

In ER− tumours these tumour cells have about as many T cells nearby as chance would give, while in ER+ tumours they
have more. `box_plot()` shows the excess for each patient.

```{code-cell} ipython3
ax = res.box_plot("HR- Ki67+", "T cells")
ax.set_ylim(-2, 4);
```

`bind()` returns these per-image values as a table, ready for any further analysis.

```{code-cell} ipython3
res.bind("HR- Ki67+__T cells").head()
```

## Is it abundance or attraction?

A cell type that is simply more common will be found more often around any other cell. Many of the most
significant pairs above count `HR+ CK7-` tumour cells, which are much more common in ER+ tumours. To separate a
change in arrangement from a change in abundance, spicyr also reports the difference after adjusting for how common
the *to* type is in each image (the `adjusted_*` columns).

```{code-cell} ipython3
tab = res.cell_results.sort_values("p_value")
tab[["from", "to", "excess_difference", "p_adj", "adjusted_difference", "adjusted_p_adj"]].head(8)
```

The pairs that count `HR+ CK7-` cells largely lose their significance once their abundance is taken into account,
while our pair keeps it.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 5))
hr = (tab["to"] == "HR+ CK7-").to_numpy()
for mask, colour, label in [(~hr, "grey", "other"), (hr, "#b3261e", "HR+ CK7- counted")]:
    ax.scatter(-np.log10(tab["p_adj"][mask]), -np.log10(tab["adjusted_p_adj"][mask]), s=12, alpha=0.7,
               color=colour, label=label)
lim = max(-np.log10(tab["p_adj"]).max(), 1)
ax.plot([0, lim], [0, lim], ls="--", c="grey")
ax.set_xlabel("-log10 adjusted p"); ax.set_ylabel("-log10 adjusted p, after adjusting for abundance")
ax.legend(frameon=False);
```

Both results are reported because both can be of interest. In a disease where the change in composition is the
biology, the unadjusted excess may be the question you want to ask.

## Patients with several images

Each patient here has one core. When patients contribute several images, give the patient column as `subject`:
the images of each patient are combined, and patients remain the units of the test. Treating every image as an
independent patient would overstate the evidence.

## Adjusting for clinical covariates

Covariates measured per patient or per image are added with `covariates`. Here we adjust the ER comparison for age
at diagnosis and tumour grade. Patients with a missing covariate are left out of the adjusted comparison only.

```{code-cell} ipython3
cells["Grade"] = pd.Categorical(cells["Grade"])
res_cov = spicyr.spicy(cells, condition="ER", subject="metabricId", r=25,
                       image_id="file_id", cell_type="description",
                       from_="HR- Ki67+", to=["T cells", "B cells"],
                       covariates=["Age.At.Diagnosis", "Grade"])
res_cov.cell_results[["from", "to", "excess_difference", "p_value", "covariate_difference", "covariate_p_value"]]
```

## Which radius?

The scale of an interaction is rarely known in advance. Give several radii and spicyr tests them together,
accounting for the strong correlation between neighbouring radii, and reports the radius with the strongest
evidence.

```{code-cell} ipython3
res_r = spicyr.spicy(cells, condition="ER", subject="metabricId", r=[10, 25, 50, 75],
                     image_id="file_id", cell_type="description",
                     from_="HR- Ki67+", to=["T cells", "B cells", "Macrophages Vim+ Slug-"])
res_r.cell_results[["from", "to", "r", "excess_difference", "p_value"]]
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
for to, g in res_r.radius_results.groupby("to"):
    ax.plot(g["r"], g["excess_difference"], marker="o", label=to)
ax.axhline(0, ls="--", c="grey")
ax.set_xlabel("radius (µm)"); ax.set_ylabel("change in excess, ER+ vs ER-")
ax.legend(title="around HR- Ki67+", frameon=False);
```

## Is co-localisation associated with survival?

With the follow-up time as `condition` and the event indicator as `survival`, spicyr asks whether the excess is
associated with outcome. Here we use relapse-free survival, adjusting for age.

```{code-cell} ipython3
res_s = spicyr.spicy(cells, condition="timeRFS", survival="eventRFS", subject="metabricId", r=25,
                     image_id="file_id", cell_type="description", covariates="Age.At.Diagnosis")
res_s.cell_results.sort_values("p_value")[["from", "to", "p_value", "p_adj", "hazard_ratio_sd"]].head(6)
```

`hazard_ratio_sd` is the hazard ratio for a one standard deviation higher excess. A ratio below one means that
patients whose tumours have more *to* cells around each *from* cell relapse later.

## A check you can run

A test should give few significant results when there is nothing to find. Shuffling the ER labels across patients
removes any real difference, so about 5% of pairs should then have p < 0.05.

```{code-cell} ipython3
rng = np.random.default_rng(2026)
patients = cells.drop_duplicates("metabricId")[["metabricId", "ER"]]
shuffled = []
for i in range(3):
    lab = dict(zip(patients["metabricId"], rng.permutation(patients["ER"].to_numpy())))
    cells["shuffled"] = cells["metabricId"].map(lab)
    s = spicyr.spicy(cells, condition="shuffled", subject="metabricId", r=25,
                     image_id="file_id", cell_type="description", availability=False)
    shuffled.append(100 * (s.cell_results["p_value"] < 0.05).mean())
shuffled
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.bar([1, 2, 3], shuffled, color="grey")
ax.axhline(5, ls="--", c="black")
ax.set_ylim(0, 20); ax.set_xticks([1, 2, 3])
ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
ax.set_xlabel("shuffle of the ER labels"); ax.set_ylabel("pairs with p < 0.05");
```

## How it works

For a pair *from* → *to* and an image, let $O$ be the number of *to* cells within $r$ of the *from* cells. If the
*to* label were given at random to the cells that could carry it, $O$ would have an exact mean and variance, which
spicyr computes without permutations. The excess of an image is

$$\delta = \frac{O - \mathrm{E}(O)}{n},$$

where $n$ is the number of *from* cells: the extra *to* cells per *from* cell. The images of each patient are
combined, weighting each by its information, and patients vary around their group's mean (a frailty model). The
difference between groups is tested with a small-sample cluster-robust variance on Satterthwaite degrees of
freedom, so the test stays accurate with few patients, and the within-image variance is inflated when the *to*
cells cluster among themselves. The full derivation is in the supplementary material of the paper.

## Small studies

With fewer than about ten patients per group, expect few significant pairs, even when an effect is consistent
across patients. This is the correct answer when there is little information, not a failure of the test. For very
small studies, `variance="hartung_knapp"` gives a little more power at the same error rate.

## The R package

spicyr is the Python twin of the Bioconductor package [spicyR](https://bioconductor.org/packages/spicyR): the
same test, the same arguments and the same results. The R package also keeps the image-level test of spicyR
versions before 2.0 (`method = "image"`), which is not yet available in Python.
