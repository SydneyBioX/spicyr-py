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
cancer imaging mass cytometry data, from testing every pair of cell types to checking that the test gives the
expected number of false positives. It is the same analysis, on the same data, as the vignette of the R package
spicyR.

## Overview

Do T cells gather around tumour cells more in one group of patients than in another? spicyr tests this for every
pair of cell types at once.

A pair is written *from* → *to*. For each *from* cell, spicyr counts the *to* cells within a radius, and compares
that count with what we would expect if the *from* cells were a random choice among the cells of the same image
that are not *to* cells. The difference is the **excess**: the number of extra *to* cells around each *from* cell, beyond
chance. Because the comparison uses only the cells that are actually there, empty regions such as holes or air
spaces, and uneven cell density, do not by themselves create a signal (artefacts that affect one cell type more
than others are not removed). spicyr then compares the excess between
groups of patients, treating patients, not images or cells, as the units of the test.

Pairs are directional: T cells → tumour cells asks how many extra tumour cells sit around each T cell, which is a
different question from tumour cells → T cells.

spicyr needs the type and position of every cell, as from imaging mass cytometry, CODEX, MIBI, Xenium, CosMx or
MERSCOPE; it is not designed for spot-based data such as Visium.

```{figure} _static/spicyR_overview.png
:width: 100%
:alt: Left, a tumour cell with a 25 µm circle and the T cells inside it. Right, box plots of the extra T cells per tumour cell, one point per patient, higher in ER-positive than ER-negative tumours.

For each proliferating tumour cell (red), spicyr counts the T cells (blue) within 25 µm and compares the count
with chance. Each patient gets an excess, and the excess is compared between ER-negative and ER-positive tumours.
```

## Installation

spicyr will be released on PyPI together with spicyR 2.0; until then, install it from GitHub (this needs a C++17
compiler).

```bash
pip install "spicyr[data,plot,anndata] @ git+https://github.com/SydneyBioX/spicyr-py"
```

```{code-cell} ipython3
import anndata as ad
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
each cell to one of 22 types. Their labels name tumour cells by marker: HR is hormone receptor, CK cytokeratin, and
Ki67+ marks proliferating cells, so `HR- Ki67+` are proliferating hormone-receptor-negative tumour cells.
`spicyr.datasets.metabric_ali2020()` downloads the same file that the R package SpatialDatasets uses, from
Bioconductor, as a table with one row per cell. We put it in an AnnData object, with coordinates (in micrometres)
in `obsm["spatial"]`.

```{code-cell} ipython3
cells = spicyr.datasets.metabric_ali2020()
cells = cells[cells["ER.Status"].isin(["neg", "pos"])].reset_index(drop=True)
cells["ER"] = pd.Categorical(np.where(cells["ER.Status"] == "pos", "ER+", "ER-"), categories=["ER-", "ER+"])
cells.index = cells.index.astype(str)
adata = ad.AnnData(obs=cells.drop(columns=["x", "y"]), obsm={"spatial": cells[["x", "y"]].to_numpy()})

print(adata.obs.drop_duplicates("metabricId")["ER"].value_counts().sort_index())
adata.obs["description"].value_counts()
```

```{code-cell} ipython3
types = sorted(adata.obs["description"].unique())
palette = dict(zip(types, plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + [[0.2, 0.2, 0.2, 1], [0.6, 0.4, 0.1, 1]]))
size = cells["file_id"].value_counts()                         # show the largest core of each group
two_cores = [size[cells.loc[cells["ER"] == g, "file_id"].unique()].idxmax() for g in ["ER-", "ER+"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, core in zip(axes, two_cores):
    z = cells[cells["file_id"] == core]
    for t, g in z.groupby("description"):
        ax.scatter(g["x"], g["y"], s=2, color=palette[t])
    ax.set_title(f"{core} ({z['ER'].iloc[0]})"); ax.set_aspect("equal"); ax.axis("off")
handles = [plt.Line2D([], [], marker="o", ls="", color=palette[t], label=t) for t in types]
fig.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(0.92, 0.5), frameon=False)
plt.subplots_adjust(right=0.9)
```

## Testing every pair of cell types

`spicy()` needs the column of the condition, the column of the patient (`subject`) and a radius. It accepts an
AnnData object, a SpatialData object or a pandas DataFrame with one row per cell. By default it looks for columns
called `imageID` and `cellType`; here we name the dataset's own columns. The coordinates of an AnnData object are
taken from `obsm["spatial"]`.

The radius should reflect the scale at which you expect cells to interact, and is best chosen before looking at the
results: 10 to 25 µm for contact, 50 to 100 µm for a shared neighbourhood. Here we use 25 µm, about two to three
cell diameters.

```{code-cell} ipython3
res = spicyr.spicy(adata, condition="ER", subject="metabricId", r=25,
                   image_id="file_id", cell_type="description")
res
```

All 484 ordered pairs of cell types are tested, in about ten seconds on one core; run time and memory grow
roughly in proportion to the number of cells and to the radius. `top_pairs()` lists the most
significant. `intercept` is the average excess in ER− patients, and `coefficient` is the difference in average
excess between ER+ and ER− patients (ER+ minus ER−), in extra *to* cells per *from* cell. P-values are adjusted
across all pairs by the Benjamini–Hochberg method.

```{code-cell} ipython3
res.top_pairs(8)
```

The full results are in `res.cell_results`, with one row per pair:

| Column | Meaning |
|---|---|
| `excess_ref`, `excess_comp` | average excess in the reference group (ER−) and the comparison group (ER+) |
| `excess_difference`, `se`, `df` | their difference, its standard error and degrees of freedom |
| `p_value`, `p_adj` | p-value, and Benjamini–Hochberg adjusted p-value across all pairs |
| `tau2` | how much the excess varies between patients within a group |
| `adjusted_difference`, `adjusted_p_value`, `adjusted_p_adj` | the same, after adjusting for how common the *to* type is (see below) |

With more than two groups there is one row per pair and group, each compared with the reference group, in a
column `level`.

## Seeing every pair at once

`signif_plot()` shows the whole study. Read rows as *from* and columns as *to*. Each circle is a pair: the left half
is coloured by the excess in ER− tumours and the right half by the excess in ER+ tumours (red: more *to* cells than
chance, blue: fewer), the size reflects the p-value, and a black ring marks a BH-adjusted p-value below 0.05.

```{code-cell} ipython3
res.signif_plot(fdr=True, breaks=(-2, 2, 0.5));
```

## Looking at one pair

Many of the top pairs have `HR+ CK7-` tumour cells as the *to* type, which, as we will see, mostly reflects how
common those cells are. We focus instead on an immune pair: T cells around proliferating hormone-receptor-negative
tumour cells.

```{code-cell} ipython3
res.cell_results.loc[["HR- Ki67+__T cells"], ["excess_ref", "excess_comp", "excess_difference", "p_value", "p_adj"]]
```

In ER− tumours these tumour cells have no more T cells nearby than chance would give. In ER+ tumours they have about
0.15 extra T cells each, roughly one extra T cell for every seven tumour cells. `box_plot()` shows the excess in each
image (here one image per patient).

```{code-cell} ipython3
above = int((res.bind("HR- Ki67+__T cells")["HR- Ki67+__T cells"] > 4).sum())
ax = res.box_plot("HR- Ki67+", "T cells")
ax.set_ylim(-2, 4)
ax.set_title(f"T cells around HR- Ki67+ ({above} images above 4 not shown)", fontsize=10);
```

This is an association in one cohort; it does not show that the tumour cells attract T cells. `HR- Ki67+` cells are
uncommon in ER+ tumours, and images without any of them carry no information about the pair (`NaN` below).
`bind()` returns the per-image values as a table, for your own plots or models.

```{code-cell} ipython3
res.bind("HR- Ki67+__T cells").head(6)
```

## Is it abundance or attraction?

A cell type that is simply more common will be found more often around any other cell, even if cells are arranged
no differently. `HR+ CK7-` tumour cells are much more common in ER+ tumours. To separate a change in arrangement
from a change in abundance, spicyr also reports each difference after adjusting for how common the *to* type is in
each image (the `adjusted_*` columns).

```{code-cell} ipython3
tab = res.cell_results.sort_values("p_value")
cols = ["from", "to", "excess_difference", "p_adj", "adjusted_difference", "adjusted_p_adj"]
display(tab[cols].head(8))
tab.loc[["HR- Ki67+__T cells"], cols[2:]]
```

Most pairs with `HR+ CK7-` as the *to* type are no longer significant after the adjustment, so their unadjusted
signal may largely reflect abundance. Because abundance differs so much with ER status, the adjusted test also has
less power for these pairs, so a non-significant adjusted result is not evidence of no effect. Our pair stays
significant, and more strongly so: when the *to* type's abundance varies a lot between patients, adjusting for it
can remove noise as well as bias.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 5.5))
group = np.where(tab.index == "HR- Ki67+__T cells", "HR- Ki67+ → T cells",
                 np.where(tab["to"] == "HR+ CK7-", "HR+ CK7- counted", "other"))
for g, colour in [("other", "grey"), ("HR+ CK7- counted", "#b3261e"), ("HR- Ki67+ → T cells", "#1f6fb4")]:
    k = group == g
    ax.scatter(np.minimum(-np.log10(tab["p_adj"][k]), 10), np.minimum(-np.log10(tab["adjusted_p_adj"][k]), 10),
               s=14, alpha=0.8, color=colour, label=g)
ax.plot([0, 10], [0, 10], ls="--", c="grey")
ax.set_xlim(0, 10.3); ax.set_ylim(0, 10.3); ax.set_aspect("equal")
ax.set_xlabel("-log10 adjusted p (values above 10 shown at 10)")
ax.set_ylabel("-log10 adjusted p, after adjusting for abundance")
ax.legend(frameon=False, loc="upper left");
```

spicyr reports both. Report the adjusted result if your question is about arrangement, and the unadjusted one if a
change in composition is part of the biology.

## Patients with several images

Each patient here has one core. When patients contribute several images, give the patient column as `subject`:
the images of each patient are combined, and patients remain the units of the test. Treating every image as an
independent patient would overstate the evidence.

## Adjusting for clinical covariates

Covariates measured per patient or per image are added with `covariates`. Here we adjust the ER comparison for age
at diagnosis and tumour grade. Patients with a missing covariate are left out of the adjusted comparison only.

```{code-cell} ipython3
adata.obs["Grade"] = pd.Categorical(adata.obs["Grade"])
res_cov = spicyr.spicy(adata, condition="ER", subject="metabricId", r=25,
                       image_id="file_id", cell_type="description",
                       covariates=["Age.At.Diagnosis", "Grade"])
res_cov.cell_results.loc[["HR- Ki67+__T cells"],
                         ["excess_difference", "p_value", "covariate_difference", "covariate_p_value"]]
```

The difference between ER+ and ER− patients is much the same after adjusting for age and grade.

Testing every pair keeps the results the same across analyses. If you restrict `from_` and `to`, the numbers for a
pair can shift slightly, because spicyr estimates how much cells of a type cluster among themselves from all the
pairs it tests.

## Which radius?

The scale of an interaction is rarely known in advance. Give several radii and spicyr tests them together, using a
max-T test that accounts for the strong correlation between neighbouring radii (`combine="cauchy"` is an
alternative). This combined test is new, and its calibration is still being checked.

```{code-cell} ipython3
res_r = spicyr.spicy(adata, condition="ER", subject="metabricId", r=[10, 25, 50, 75],
                     image_id="file_id", cell_type="description")
res_r.cell_results.loc[["HR- Ki67+__T cells", "HR- Ki67+__B cells"], ["r", "excess_difference", "p_value"]]
```

`r` is the radius with the strongest evidence, and `p_value` the combined p-value over all radii. The excess grows
with the radius simply because larger circles hold more cells, so compare p-values across radii rather than the
size of the excess. The excess reported at the chosen radius is a little optimistic, because that radius was picked
for its strength.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
rr = res_r.radius_results
for to, g in rr[(rr["from"] == "HR- Ki67+") & rr["to"].isin(["T cells", "B cells"])].groupby("to"):
    ax.plot(g["r"], -np.log10(g["p_value"]), marker="o", label=to)
ax.set_ylim(bottom=0)
ax.set_xlabel("radius (µm)"); ax.set_ylabel("-log10 p at each radius, ER+ vs ER-")
ax.legend(title="around HR- Ki67+", frameon=False);
```

## Is co-localisation associated with survival?

With `survival=(time, event)`, spicyr asks whether the excess is associated with outcome. Here we use relapse-free
survival, adjusting for age.

```{code-cell} ipython3
res_s = spicyr.spicy(adata, survival=("timeRFS", "eventRFS"), subject="metabricId", r=25,
                     image_id="file_id", cell_type="description", covariates="Age.At.Diagnosis")
res_s.cell_results.sort_values("p_value")[["from", "to", "p_value", "p_adj", "hazard_ratio_sd"]].head(6)
```

The p-value comes from a score test that relates each patient's excess to their outcome. `hazard_ratio_sd` is the
hazard ratio for a one standard deviation higher excess, estimated from a Cox model. A ratio below one means that
patients with a higher excess had a lower risk of relapse, after adjusting for age. The hazard ratio is missing
when the excess barely varies between patients. Here the top pairs have the common `HR+ CK7-` and `HR- CK7+`
tumour cells as the *to* type, and may reflect tumour composition, which is itself prognostic, as much as the
arrangement of cells.

## A check you can run

A test should give few significant results when there is nothing to find. Shuffling the ER labels across patients
removes any real difference, so about 5% of pairs should then have p < 0.05, and about 1% p < 0.01.

```{code-cell} ipython3
rng = np.random.default_rng(2026)
patients = adata.obs.drop_duplicates("metabricId")[["metabricId", "ER"]]
rows = []
for i in range(5):
    label = dict(zip(patients["metabricId"], rng.permutation(patients["ER"].to_numpy())))
    adata.obs["shuffled"] = adata.obs["metabricId"].map(label).astype(str)
    s = spicyr.spicy(adata, condition="shuffled", subject="metabricId", r=25,
                     image_id="file_id", cell_type="description",
                     availability=False)   # skip the abundance-adjusted test to save time
    p = s.cell_results["p_value"]
    rows += [(i, "p < 0.05", 100 * (p < 0.05).mean()), (i, "p < 0.01", 100 * (p < 0.01).mean())]
shuffles = pd.DataFrame(rows, columns=["shuffle", "threshold", "percent"])
shuffles.groupby("threshold")["percent"].mean()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 3.5))
for k, th in enumerate(["p < 0.05", "p < 0.01"]):
    v = shuffles.loc[shuffles["threshold"] == th, "percent"]
    ax.scatter(k + rng.uniform(-0.08, 0.08, len(v)), v, color="black", alpha=0.7)
    ax.hlines({"p < 0.05": 5, "p < 0.01": 1}[th], k - 0.25, k + 0.25, color="#b3261e", lw=2)
ax.set_xticks([0, 1], ["p < 0.05", "p < 0.01"]); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 15)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
ax.set_ylabel("pairs below the threshold"); ax.set_title("5 shuffles of the ER labels (red: expected)");
```

Pairs share cells, so the percentage moves by a few points from one shuffle to the next. (The shuffles are random,
so these numbers differ from those in the R vignette.)

## How it works

For a pair *from* → *to* and an image, let $O$ be the number of *to* cells within $r$ of the *from* cells. If the
*from* cells were a random choice among the cells of the image that are not *to* cells, keeping every cell where it
is, $O$ would have an exact mean and variance, which spicyr computes without permutations. The excess of an image is

$$\delta = \frac{O - \mathrm{E}_{\mathrm{RL}}(O)}{n},$$

where $\mathrm{E}_{\mathrm{RL}}(O)$ is that expectation under random labelling and $n$ is the number of *from*
cells.

Images from the same patient are combined, giving more weight to more informative images (usually those with more
*from* cells). Each patient has its
own true excess, which varies around its group's mean by an amount estimated from the data (a frailty, or
random-effects, model). The difference between groups is tested with a small-sample cluster-robust (CR2) variance on
Satterthwaite degrees of freedom, with patients as the clusters. This is designed to keep false positives near the
nominal rate even with modest numbers of patients. When the *from* cells cluster among themselves, the within-image
variance is inflated to match. A paper describing the method is in preparation.

## Small studies

With fewer than about ten patients per group, expect few significant pairs, even when an effect is consistent across
patients: a calibrated test cannot be confident with little information. Look at whether the patients agree in
direction (`box_plot()`). `variance="hartung_knapp"` can give more power in very small studies. It assumes that
patients vary about equally in both groups and can give too many small p-values for rare cell types, so it is not
the default.

## Reporting results

A methods sentence might read: "We used spicyr (version 1.99.0) to test, for every ordered pair of cell types,
whether the number of *to* cells within 25 µm of each *from* cell, relative to random labelling of the cells in each
image, differed between ER+ and ER− patients, with patients as the units of analysis. P-values were adjusted across
pairs by the Benjamini–Hochberg method." Show a per-patient plot (`box_plot()`) and an image of the pair alongside
the p-value. Please cite Canete et al. (2022), *Bioinformatics* 38(11), 3099–3105; a paper describing the
cell-level test is in preparation.

## R

spicyr is the Python version of the Bioconductor package [spicyR](https://bioconductor.org/packages/spicyR). It uses
the same C++ code and gives the same results; the spicyR vignette is this tutorial in R. The names follow Python
conventions:

| R (spicyR) | Python (spicyr) |
|---|---|
| `spicy(spe, condition, subject, imageID =, cellType =, from =, to =)` | `spicy(adata, condition, subject, image_id=, cell_type=, from_=, to=)` |
| `condition = "RFS"` with a `Surv(time, event)` column | `survival=("timeRFS", "eventRFS")` |
| `topPairs(res)`, `res$cellResults` | `res.top_pairs()`, `res.cell_results` |
| `signifPlot(res)`, `spicyBoxPlot(res, from, to)`, `bind(res)` | `res.signif_plot()`, `res.box_plot(from_, to)`, `res.bind()` |

The R package also keeps the image-level test of spicyR versions before 2.0 (`method = "image"`), which is not
available in Python.

## Session information

```{code-cell} ipython3
import importlib.metadata as md
import platform
print("Python", platform.python_version())
for p in ["spicyr", "numpy", "pandas", "anndata", "matplotlib"]:
    print(p, md.version(p))
```
