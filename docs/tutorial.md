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

Do T cells gather around tumour cells more in ER-positive breast cancers than in ER-negative ones? Do patients
whose tumour cells are surrounded by immune cells relapse later? spicyr answers questions like these for every pair
of cell types in an imaging study. This tutorial works through a study of 456 breast tumours: testing every pair,
looking closely at one, finding the images behind the result, and checking that the test behaves when there is
nothing to find.

## Overview

A pair is written `from` → `to`. For each `from` cell, spicyr counts the `to` cells within a radius, and compares
that count with what we would expect if the `from` cells were a random choice among the cells of the same image that
are not `to` cells. The difference is the **excess**: the number of extra `to` cells around each `from` cell,
beyond chance. Because the comparison uses only the cells that are actually there, empty regions such as holes or
air spaces, and uneven cell density, do not by themselves create a signal. spicyr then compares the excess between
groups of patients, treating patients, not images or cells, as the units of the test.

```{figure} _static/spicyR_overview.png
:width: 100%
:alt: Left, a tumour cell with a 25 µm circle and the T cells inside it. Right, box plots of the extra T cells per tumour cell, one point per patient, higher in ER-positive than ER-negative tumours.

For each proliferating tumour cell (red), spicyr counts the T cells (blue) within 25 µm and compares the count
with chance. Each patient gets an excess, and the excess is compared between ER-negative and ER-positive tumours.
```

:::{note}
Pairs are directional. T cells → tumour cells asks how many extra tumour cells sit around each T cell, which is a
different question from tumour cells → T cells.
:::

spicyr needs the type and position of every cell, as from imaging mass cytometry, CODEX, MIBI, Xenium, CosMx or
MERSCOPE. It is not designed for spot-based data such as Visium.

## Installation

spicyr will be on PyPI with its first release. Until then, install it from GitHub (this needs a C++17 compiler):

```bash
pip install "spicyr[data,plot,anndata] @ git+https://github.com/SydneyBioX/spicyr-py"
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spicyr

pd.set_option("display.precision", 4)
pd.set_option("display.width", 120)
```

```{code-cell} ipython3
:tags: [remove-cell]

# for the web page only: plotly figures as plain HTML (plotly.js from its CDN, without MathJax)
import plotly.io as pio
from plotly.io._base_renderers import MimetypeRenderer


class _DocsRenderer(MimetypeRenderer):
    def to_mimebundle(self, fig_dict):
        return {"text/html": pio.to_html(fig_dict, include_plotlyjs="cdn", include_mathjax=False, full_html=False)}


pio.renderers["docs"] = _DocsRenderer()
pio.renderers.default = "docs"
```

## The data

We use imaging mass cytometry of breast tumours from the METABRIC cohort (Ali et al., *Nature Cancer* 2020). Each
of the 456 patients with known oestrogen receptor (ER) status contributed one tumour core, and the authors assigned
each cell to one of 22 types. Their labels name tumour cells by marker: HR is hormone receptor, CK cytokeratin, and
Ki67+ marks proliferating cells, so `HR- Ki67+` are proliferating hormone-receptor-negative tumour cells.

`spicyr.datasets.metabric_ali2020()` downloads the data once (130 MB) and caches it. We ask for an AnnData object:
one observation per cell, with the clinical data in `obs` and the coordinates, in micrometres, in
`obsm["spatial"]`.

```{code-cell} ipython3
adata = spicyr.datasets.metabric_ali2020(as_anndata=True)
adata = adata[adata.obs["ER.Status"].isin(["neg", "pos"])].copy()
adata.obs["ER"] = pd.Categorical(adata.obs["ER.Status"].map({"neg": "ER-", "pos": "ER+"}), categories=["ER-", "ER+"])
adata
```

```{code-cell} ipython3
print(adata.obs.drop_duplicates("metabricId")["ER"].value_counts().sort_index())
adata.obs["description"].value_counts()
```

```{code-cell} ipython3
types = sorted(adata.obs["description"].unique())
palette = dict(zip(types, plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + [[0.2, 0.2, 0.2, 1], [0.6, 0.4, 0.1, 1]]))
size = adata.obs["file_id"].value_counts()                      # show the largest core of each group
two_cores = [size[adata.obs.loc[adata.obs["ER"] == g, "file_id"].unique()].idxmax() for g in ["ER-", "ER+"]]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, core in zip(axes, two_cores):
    sub = adata[adata.obs["file_id"] == core]
    xy = sub.obsm["spatial"]
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c=sub.obs["description"].map(palette).tolist())
    ax.set_title(f"{core} ({sub.obs['ER'].iloc[0]})")
    ax.set_aspect("equal")
    ax.axis("off")
handles = [plt.Line2D([], [], marker="o", ls="", color=palette[t], label=t) for t in types]
fig.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(0.92, 0.5), frameon=False)
plt.subplots_adjust(right=0.9)
```

## Testing every pair of cell types

`spicy()` needs the column of the condition, the column of the patient (`subject`) and a radius. It accepts an
AnnData object, a SpatialData object or a pandas DataFrame with one row per cell. By default it looks for columns
called `imageID` and `cellType`; here we name the dataset's own columns.

:::{tip}
Choose the radius before looking at the results, from the scale at which you expect cells to interact: 10 to 25 µm
for contact, 50 to 100 µm for a shared neighbourhood. Here we use 25 µm, about two to three cell diameters. If you
are unsure, give several radii (see [Which radius?](#which-radius)).
:::

```{code-cell} ipython3
res = spicyr.spicy(adata, condition="ER", subject="metabricId", r=25,
                   image_id="file_id", cell_type="description")
res
```

All 484 ordered pairs of cell types are tested in about ten seconds on one core. Run time and memory grow roughly
in proportion to the number of cells and to the radius. By default each comparison is adjusted for how common the
`to` type is in each image, for reasons we come to in [Why adjust for abundance?](#why-adjust-for-abundance).

`top_pairs()` lists the most significant pairs. `intercept` is the average excess in ER− patients, and
`coefficient` is the difference in average excess between ER+ and ER− patients (ER+ minus ER−), in extra `to` cells
per `from` cell. P-values are adjusted across all pairs by the Benjamini–Hochberg method.

```{code-cell} ipython3
res.top_pairs(8)
```

The full results are in `res.cell_results`, a DataFrame with one row per pair:

| Column | Meaning |
|---|---|
| `excess_ref`, `excess_comp` | average excess in the reference group (ER−) and the comparison group (ER+) |
| `excess_difference`, `se`, `df` | their difference, its standard error and degrees of freedom |
| `p_value`, `p_adj` | p-value, and Benjamini–Hochberg adjusted p-value across all pairs |
| `tau2` | how much the excess varies between patients within a group |
| `adjusted_for`, `abundance_effect` | what the test was adjusted for, and the effect of abundance |
| `unadjusted_difference`, `unadjusted_p_value`, `unadjusted_p_adj` | the same test without the adjustment |

With more than two groups there is one row per pair and group, each compared with the reference group, in a column
`level`.

## Seeing every pair at once

`signif_plot()` shows the whole study. Read rows as `from` and columns as `to`. Each circle is a pair: the left half
is coloured by the excess in ER− tumours and the right half by the excess in ER+ tumours (red: more `to` cells than
chance, blue: fewer), the size reflects the p-value, and a black ring marks a BH-adjusted p-value below 0.05.

```{code-cell} ipython3
res.signif_plot(fdr=True, breaks=(-2, 2, 0.5));
```

## Looking at one pair

We focus on an immune pair: T cells around proliferating hormone-receptor-negative tumour cells.

```{code-cell} ipython3
pair = "HR- Ki67+__T cells"
res.cell_results.loc[[pair], ["excess_ref", "excess_comp", "excess_difference", "p_value", "p_adj"]]
```

In ER− tumours these tumour cells have no more T cells nearby than chance would give. In ER+ tumours they have about
0.23 extra T cells each, roughly one extra T cell for every four tumour cells.

`box_plot()` shows the excess in each image (here one image per patient), with a point per image behind each box.
Points are sized by how much the image contributes to the test.

```{code-cell} ipython3
ax = res.box_plot("HR- Ki67+", "T cells")
ax.set_ylim(-2, 4);
```

With `interactive=True` the plot is a plotly figure. Hover over a point to see which image it is.

```{code-cell} ipython3
res.box_plot("HR- Ki67+", "T cells", interactive=True)
```

:::{important}
This is an association in one cohort; it does not show that the tumour cells attract T cells.
:::

`bind()` returns the per-image values as a DataFrame, for your own plots or models. Images without any `HR- Ki67+`
cells carry no information about the pair (`NaN`).

```{code-cell} ipython3
res.bind(pair).head(6)
```

## Looking at the images

`plot_image()` shows one image: the density of all cells in blue, the `from` cells in gold and the `to` cells in
dark red. With `r`, it draws the circle around each `from` cell inside which `to` cells are counted. We look at
three images found with the interactive box plot.

```{code-cell} ipython3
d = res.bind(pair).set_index("imageID")
d["weight"] = res.image_weights[pair]
examples = ["MB0150_1_155", "MB0132_1_533", "MB0244_1_519"]
d.loc[examples, ["condition", pair, "weight"]]
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, image in zip(axes, examples):
    spicyr.plot_image(adata, image, "HR- Ki67+", "T cells", r=25, image_id="file_id", cell_type="description", ax=ax)
    ax.set_title(f"{image} ({d.loc[image, 'condition']}), excess {d.loc[image, pair]:.2f}")
plt.tight_layout()
```

In the ER+ image on the left, the T cells run along the band of tumour cells at the bottom: 1.5 extra T cells per
tumour cell. In the ER− image in the middle, the T cells are concentrated top left, apart from most of the tumour
cells, and each tumour cell has slightly fewer T cells nearby than chance (−0.22). The ER+ image on the right has
the largest excess of all, 8.3, but it comes from just two tumour cells that happen to sit in a cluster of T cells.

:::{tip}
Point size matters. The right-hand image is the highest point in the box plot, but its weight is close to zero: an
excess estimated from two cells says little. Weights also level off. Once an image has a few dozen `from` cells,
more cells add little, because patients differ from one another more than repeated counts within a patient do.
The weights are in `res.image_weights`.
:::

## Why adjust for abundance?

A cell type that is simply more common will be found more often around any other cell, even if cells are arranged
no differently. Here the `HR+ CK7-` tumour cells are much more common in ER+ tumours. Without adjustment, almost
every pair with `HR+ CK7-` as the `to` type looks strongly different between ER+ and ER− patients.

By default, spicyr adjusts each comparison for the log of the `to` type's share of all cells in each image. The
excess is then compared between groups at the same abundance, and a difference in arrangement is not confused with
a difference in composition. The unadjusted test is kept in the `unadjusted_*` columns.

```{code-cell} ipython3
tab = res.cell_results.sort_values("unadjusted_p_value")
tab[["from", "to", "unadjusted_difference", "unadjusted_p_adj", "excess_difference", "p_adj"]].head(8)
```

Most pairs with `HR+ CK7-` as the `to` type are no longer significant after the adjustment, so their unadjusted
signal largely reflects abundance. Because abundance differs so much with ER status, the adjusted test also has less
power for these pairs, so a non-significant adjusted result is not evidence of no effect.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 5.5))
group = np.where(tab.index == pair, "HR- Ki67+ → T cells",
                 np.where(tab["to"] == "HR+ CK7-", "HR+ CK7- counted", "other"))
for g, colour in [("other", "grey"), ("HR+ CK7- counted", "#b3261e"), ("HR- Ki67+ → T cells", "#1f6fb4")]:
    k = group == g
    ax.scatter(np.minimum(-np.log10(tab.loc[k, "unadjusted_p_adj"]), 10), np.minimum(-np.log10(tab.loc[k, "p_adj"]), 10),
               s=14, alpha=0.8, color=colour, label=g)
ax.plot([0, 10], [0, 10], ls="--", c="grey")
ax.set(xlim=(0, 10.3), ylim=(0, 10.3), aspect="equal",
       xlabel="-log10 adjusted p, without adjustment for abundance\n(values above 10 shown at 10)",
       ylabel="-log10 adjusted p (the default test)")
ax.legend(frameon=False, loc="upper left");
```

Our pair is more significant after the adjustment, not less (`p_value` against `unadjusted_p_value`). Each image
has its own share of T cells, and tumours with many T cells have more T cells near any cell. Adjusting for it
removes this noise. `abundance_effect` is the change in excess for each unit of log share: positive here, as
expected.

```{code-cell} ipython3
res.cell_results.loc[[pair], ["excess_difference", "p_value", "abundance_effect", "abundance_p_value",
                              "unadjusted_difference", "unadjusted_p_value"]]
```

:::{note}
Use the default (adjusted) result when your question is about arrangement. If a change in composition is part of
the biology you are asking about, use the unadjusted columns, or `spicy(..., adjust_abundance=False)`.
:::

## Patients with several images

Each patient here has one core. When patients contribute several images, give the patient column as `subject`: the
images of each patient are combined, and patients remain the units of the test.

:::{warning}
Treating every image as an independent patient overstates the evidence. Always give `subject` when patients have
several images.
:::

## Adjusting for clinical covariates

Covariates measured per patient or per image are added with `covariates`. Here we also adjust the ER comparison for
age at diagnosis and tumour grade.

```{code-cell} ipython3
adata.obs["Grade"] = pd.Categorical(adata.obs["Grade"])
res_cov = spicyr.spicy(adata, condition="ER", subject="metabricId", r=25,
                       image_id="file_id", cell_type="description",
                       covariates=["Age.At.Diagnosis", "Grade"])
res_cov
```

The main columns (`excess_difference`, `p_value`, `p_adj`) are now the ER comparison adjusted for abundance, age
and grade. Each covariate also has its own effect and p-value. A categorical covariate has one per level after the
first: `Grade2` and `Grade3` compare grades 2 and 3 with grade 1.

```{code-cell} ipython3
cols = ["excess_difference", "p_value", "Age.At.Diagnosis_effect", "Age.At.Diagnosis_p_value",
        "Grade2_effect", "Grade2_p_value", "Grade3_effect", "Grade3_p_value"]
res_cov.cell_results.loc[[pair], cols]
```

The difference between ER+ and ER− patients is much the same after adjusting for age and grade, and neither has a
clear effect of its own on this pair. Patients with a missing covariate are left out of the adjusted test.

:::{note}
A pair that cannot be adjusted, for example because its cell types appear in images of only one grade, is reported
unadjusted, and its `adjusted_for` column says why. The printed summary above counts these pairs.
:::

Testing every pair keeps the results the same across analyses. If you restrict `from_` and `to`, the numbers for a
pair can shift slightly, because spicyr estimates how much cells of a type cluster among themselves from all the
pairs it tests.

## Which radius?

The scale of an interaction is rarely known in advance. Give several radii and spicyr tests them together, using a
max-T test that accounts for the strong correlation between neighbouring radii (`combine="cauchy"` is an
alternative).

```{code-cell} ipython3
res_r = spicyr.spicy(adata, condition="ER", subject="metabricId", r=[10, 25, 50, 75],
                     image_id="file_id", cell_type="description")
res_r.cell_results.loc[[pair, "HR- Ki67+__B cells"], ["r", "excess_difference", "p_value"]]
```

`r` is the radius with the strongest evidence, and `p_value` the combined p-value over all radii.

:::{tip}
The excess grows with the radius simply because larger circles hold more cells, so compare p-values across radii
rather than the size of the excess. The excess at the chosen radius is a little optimistic, because that radius was
picked for its strength. This combined test is new, and its calibration is still being checked.
:::

```{code-cell} ipython3
rr = res_r.radius_results
fig, ax = plt.subplots(figsize=(6, 4))
for to, g in rr[(rr["from"] == "HR- Ki67+") & rr["to"].isin(["T cells", "B cells"])].groupby("to"):
    ax.plot(g["r"], -np.log10(g["p_value"]), marker="o", label=to)
ax.set(ylim=(0, None), xlabel="radius (µm)", ylabel="-log10 p at each radius, ER+ vs ER-")
ax.legend(title="around HR- Ki67+", frameon=False);
```

## Is co-localisation associated with survival?

With `survival=(time, event)`, spicyr asks whether a patient's excess is associated with their outcome. Here we use
relapse-free survival, adjusting for age.

```{code-cell} ipython3
res_s = spicyr.spicy(adata, survival=("timeRFS", "eventRFS"), subject="metabricId", r=25,
                     image_id="file_id", cell_type="description", covariates="Age.At.Diagnosis")
res_s
```

```{code-cell} ipython3
cols = ["from", "to", "p_value", "p_adj", "hazard_ratio_sd", "unadjusted_p_value", "unadjusted_p_adj"]
res_s.cell_results.sort_values("p_value")[cols].head(6)
```

The p-value comes from a score test that relates each patient's excess to their outcome. `hazard_ratio_sd` is the
hazard ratio for a one standard deviation higher excess, from a Cox model; below one, patients with a higher excess
had a lower risk of relapse. It is missing when the excess barely varies between patients.

No pair is significant after adjusting for multiple testing. Without the abundance adjustment, three pairs are,
all with tumour cell types as the `to` type. Tumour composition is itself prognostic, so those three may reflect
composition as much as the arrangement of cells.

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
                     image_id="file_id", cell_type="description")
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
ax.set_xticks([0, 1], ["p < 0.05", "p < 0.01"])
ax.set(xlim=(-0.6, 1.6), ylim=(0, 15), ylabel="pairs below the threshold",
       title="5 shuffles of the ER labels (red: expected)")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
```

Pairs share cells, so the percentage moves by a few points from one shuffle to the next.

:::{tip}
Run this check on your own data before trusting a discovery. It takes a few minutes and shows whether the test is
calibrated for your study design.
:::

## How it works

For a pair `from` → `to` and an image, let $O$ be the number of `to` cells within $r$ of the `from` cells. If the
`from` cells were a random choice among the cells of the image that are not `to` cells, keeping every cell where it
is, $O$ would have an exact mean and variance, which spicyr computes without permutations. The excess of an image is

$$\delta = \frac{O - \mathrm{E}_{\mathrm{RL}}(O)}{n},$$

where $\mathrm{E}_{\mathrm{RL}}(O)$ is that expectation under random labelling and $n$ is the number of `from`
cells.

Images from the same patient are combined, giving more weight to more informative images (usually those with more
`from` cells). Each patient has its own true excess, which varies around its group's mean by an amount estimated
from the data (a frailty, or random-effects, model). The difference between groups, adjusted for the log share of
the `to` type in each image and any covariates, is tested with a small-sample cluster-robust (CR2) variance on
Satterthwaite degrees of freedom, with patients as the clusters. This is designed to keep false positives near the
nominal rate even with modest numbers of patients. When the `from` cells cluster among themselves, the
within-image variance is inflated to match. A paper describing the method is in preparation.

## Small studies

With fewer than about ten patients per group, expect few significant pairs, even when an effect is consistent across
patients: a calibrated test cannot be confident with little information. Look at whether the patients agree in
direction with `box_plot()`.

:::{note}
`variance="hartung_knapp"` can give more power in very small studies. It assumes that patients vary about equally
in both groups and can give too many small p-values for rare cell types, so it is not the default.
:::

## Reporting results

A methods sentence might read: "We used spicyr (version 1.99.0) to test, for every ordered pair of cell types,
whether the number of `to` cells within 25 µm of each `from` cell, relative to random labelling of the cells in each
image, differed between ER+ and ER− patients, adjusting for the abundance of the `to` type in each image, with
patients as the units of analysis. P-values were adjusted across pairs by the Benjamini–Hochberg method." Show a
per-patient plot (`box_plot()`) and an image of the pair (`plot_image()`) alongside the p-value. Please cite
Canete et al. (2022), *Bioinformatics* 38(11), 3099–3105; a paper describing the cell-level test is in preparation.

## spicyR in R

The same test is in the Bioconductor package [spicyR](https://github.com/SydneyBioX/spicyR), for
`SpatialExperiment` objects, with the same results. spicyR also has the original image-level test of spicyR before
version 2.0.

## Session information

```{code-cell} ipython3
import importlib.metadata as md
import platform

print("Python", platform.python_version())
for p in ["spicyr", "numpy", "pandas", "anndata", "matplotlib", "plotly"]:
    print(p, md.version(p))
```
