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

A pair is written `from` → `to` and asks whether `to` cells are placed near `from` cells more than other cells
are. For each `to` cell, spicyr checks whether there is a `from` cell within a radius, and compares the share of
`to` cells that have one with what we would expect if the `to` cells were a random choice among the cells of the
same image that are not `from` cells. The effect is the **extra fraction of `to` cells placed next to `from`
cells**, beyond chance: 0.2 means as if a fifth of the `to` cells had been moved next to `from` cells and the rest
left where chance would put them. For a pair whose `to` cells avoid the `from` cells (fewer have one nearby than
chance, over all images), the effect is measured the other way: -0.2 means as if a fifth of the `to` cells that
chance would put next to a `from` cell had been moved away. The `side` column says which scale a pair uses. Each
`to` cell counts once, however many `from` cells are beside it, so more
numerous or more tightly packed `from` cells do not inflate the effect. Because the comparison uses only the cells
that are actually there, empty regions such as holes or air spaces, and uneven cell density, do not by themselves
create a signal (artefacts that affect one cell type more than others are not removed). spicyr then compares the
effect
between groups of patients, treating patients, not images or cells, as the units of the test.

```{figure} _static/spicyR_overview.png
:width: 100%
:alt: Left, tumour cells marked by whether a T cell lies within 25 µm, with the circle drawn around two of them. Right, box plots of the extra fraction of tumour cells next to T cells, one point per patient, higher in ER-positive than ER-negative tumours.

The pair T cells → proliferating tumour cells. For each proliferating tumour cell (red, the `to` type), spicyr
checks whether a T cell (blue, the `from` type) lies within 25 µm, and compares the share of tumour cells that have
one with chance. Each patient gets an effect, and the effect is compared between ER-negative and ER-positive tumours.
```

:::{note}
Pairs are directional. Tumour cells → T cells asks whether T cells are placed near tumour cells (the extra fraction
of T cells with a tumour cell nearby), which is a different question from T cells → tumour cells (are tumour cells
placed near T cells?).
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
`obsm["spatial"]`. The ER status of each patient is in `ER.Status`, `neg` or `pos`; we keep the patients whose ER
status is known.

```{code-cell} ipython3
adata = spicyr.datasets.metabric_ali2020(as_anndata=True)
adata = adata[adata.obs["ER.Status"].notna()].copy()  # the 456 patients with known ER status

# spicyr looks for columns called imageID and cellType by default. Copying the dataset's columns to these names is
# not needed (you can give image_id="file_id", cell_type="description" to each call instead); it only keeps the
# calls below short.
adata.obs["imageID"] = adata.obs["file_id"]
adata.obs["cellType"] = adata.obs["description"]
adata
```

```{code-cell} ipython3
print(adata.obs.drop_duplicates("metabricId")["ER.Status"].value_counts())  # patients by ER status
adata.obs["cellType"].value_counts()
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, core in zip(axes, ["MB0869_1_366", "MB0501_1_222"], strict=True):  # an ER- and an ER+ core
    spicyr.plot_image(adata, core, ax=ax)
plt.tight_layout()
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
res = spicyr.spicy(adata, condition="ER.Status", subject="metabricId", r=25)
res
```

All 484 ordered pairs of cell types are tested in a few seconds on one core. Run time and memory grow roughly in
proportion to the number of cells and to the radius.

`top_pairs()` lists the most significant pairs. `intercept` is the average effect in ER− patients, and
`coefficient` is the difference in average effect between ER+ and ER− patients (ER+ minus ER−), as a fraction of
the `to` cells. P-values are adjusted across all pairs by the Benjamini–Hochberg method.

```{code-cell} ipython3
res.top_pairs(8)
```

```{code-cell} ipython3
:tags: [remove-cell]

# the text below and in "Tissue compartments" describes these pairs
top = res.top_pairs(8)
assert ((top["from"] == "HR+ CK7-") | (top["to"] == "HR+ CK7-")).sum() >= 5 and top["from"].iloc[0] == "HR+ CK7-"
```

Most of these eight pairs have `HR+ CK7-` tumour cells as the `from` or the `to` type, the most significant as the
`from` type; we come back to them in [Tissue compartments](#tissue-compartments).

The full results are in `res.cell_results`, a DataFrame with one row per pair:

| Column | Meaning |
|---|---|
| `excess_ref`, `excess_comp` | average effect in the reference group (ER−) and the comparison group (ER+) |
| `excess_difference`, `se`, `df` | their difference, its standard error and degrees of freedom |
| `p_value`, `p_adj` | p-value, and Benjamini–Hochberg adjusted p-value across all pairs |
| `tau2` | how much the effect varies between patients within a group |
| `adjusted_for`, `<covariate>_effect` | with covariates, what the test was adjusted for and the effect of each covariate |
| `unadjusted_difference`, `unadjusted_p_value`, `unadjusted_p_adj` | with covariates, the same test without them |

With more than two groups there is one row per pair and group, each compared with the reference group, in a column
`level`.

## Seeing every pair at once

`signif_plot()` shows the whole study. Read rows as `from` and columns as `to`. Each circle is a pair: the left half
is coloured by the effect in ER− tumours and the right half by the effect in ER+ tumours (red: more `to` cells next
to `from` cells than chance, blue: fewer), the size reflects the p-value, and a black ring marks a BH-adjusted p-value below 0.05.

```{code-cell} ipython3
res.signif_plot(fdr=True, breaks=(-0.5, 0.5, 0.1));
```

## Looking at one pair

We focus on an immune pair, T cells → `HR- Ki67+`: are proliferating hormone-receptor-negative tumour cells placed
near T cells? Its effect is the extra fraction of these tumour cells with a T cell within 25 µm.

```{code-cell} ipython3
pair = "T cells__HR- Ki67+"
res.cell_results.loc[[pair], ["excess_ref", "excess_comp", "excess_difference", "p_value", "p_adj"]]
```

In ER− tumours these tumour cells have a T cell nearby barely more often than chance would give (0.02). In ER+
tumours the effect is 0.12, as if about one in nine of them had been placed next to T cells.

`box_plot()` shows the effect in each image (here one image per patient), with a point per image behind each box.
Points are sized by how much the image contributes to the test.

```{code-cell} ipython3
res.box_plot("T cells", "HR- Ki67+");
```

With `interactive=True` the plot is a plotly figure. Hover over a point to see which image it is.

```{code-cell} ipython3
res.box_plot("T cells", "HR- Ki67+", interactive=True)
```

:::{important}
This is an association in one cohort; it does not show that T cells attract the tumour cells, or the reverse.
:::

`bind()` returns the per-image values as a DataFrame, for your own plots or models. Images without any `HR- Ki67+`
cells carry no information about the pair (`NaN`).

```{code-cell} ipython3
res.bind(pair).head(6)
```

## Looking at the images

`plot_image()` shows one image: the density of all cells in blue, the `from` cells in gold and the `to` cells in
dark red. With `r`, it draws the circle around each `to` cell inside which spicyr looks for `from` cells. We look
at three images found with the interactive box plot.

```{code-cell} ipython3
d = res.bind(pair).set_index("imageID")
d["weight"] = res.image_weights[pair]
examples = ["MB0150_1_155", "MB0132_1_533", "MB0244_1_519"]
d.loc[examples, ["condition", pair, "weight"]]
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, image in zip(axes, examples):
    spicyr.plot_image(adata, image, "T cells", "HR- Ki67+", r=25, ax=ax)
    ax.set_title(f"{image} ({d.loc[image, 'condition']}), effect {d.loc[image, pair]:.2f}")
plt.tight_layout()
```

In the ER+ image on the left, the T cells run along the band of tumour cells at the bottom: an effect of 0.68, as
if two-thirds of the tumour cells had been placed next to T cells. In the ER− image in the middle, the T cells are
concentrated top left, apart from most of the tumour cells, and slightly fewer tumour cells have a T cell nearby
than chance would give (−0.08). The ER+ image on the right reaches the largest possible value, 1, but only because
its two tumour cells both happen to sit in a cluster of T cells.

:::{tip}
Point size matters. The right-hand image sits at the top of the box plot, but its weight is close to zero: an
effect estimated from two cells says little. Weights also level off. Once an image has a few dozen `to` cells,
more cells add little, because the differences between patients outweigh the noise within an image.
The weights are in `res.image_weights`.
:::

## Tissue compartments

Many of the most significant pairs, the first among them, have `HR+ CK7-` tumour cells as the `from` type. These
cells make up about a quarter of the cells in a typical ER+ core and almost none in ER− cores. In ER+ tumours they make up much of the tumour tissue
and the other tumour cells sit among them, so far more of those cells have an `HR+ CK7-` cell nearby than if they
had been placed at random among all the other cells, stroma included.

```{code-cell} ipython3
tab = res.cell_results[res.cell_results["from"] == "HR+ CK7-"]
tab.sort_values("p_value")[["to", "excess_ref", "excess_comp", "p_adj"]].head(6)
```

In ER+ tumours the effect for proliferating `HR+ CK7- Ki67+` cells is 0.60, against 0.07 in ER− tumours, while
fibroblasts and T cells are placed away from `HR+ CK7-` cells (where `HR+ CK7-` cells are rare, as in ER− cores, an
effect cannot fall far below zero). These are real differences in arrangement, but they
describe the structure of the tissue (tumour cells sit with tumour cells) more than an interaction between
particular cell types.

:::{note}
To ask about arrangement within a compartment, for example whether proliferating tumour cells sit closer to
`HR+ CK7-` cells than other tumour cells do, compare with random labelling among the tumour cells only. The
Kontextual test in the R package Statial does this.
:::

## Allocation or count?

By default spicyr asks whether each `to` cell has a `from` cell nearby. With `effect="count"` it asks how many: the
effect is then the number of extra `from` cells within the radius of each `to` cell.

```{code-cell} ipython3
res_count = spicyr.spicy(adata, condition="ER.Status", subject="metabricId", r=25, effect="count")
res_count.cell_results.loc[[pair], ["excess_ref", "excess_comp", "excess_difference", "p_value", "p_adj"]]
```

In ER+ tumours each proliferating tumour cell has about 0.16 extra T cells within 25 µm, against 0.01 in ER−
tumours. The count also reflects how many T cells surround each tumour cell, but it grows with how tightly the
`from` cells are packed: if T cells form denser clusters in one group, each tumour cell next to a cluster counts more
T cells. The default effect counts each tumour cell once, however many T cells are beside it, so denser clusters do
not inflate it.

:::{note}
Use the default when your question is whether `to` cells are placed next to `from` cells. Use `effect="count"` when
the number of `from` cells around each `to` cell is the question, keeping in mind that it also rises when the `from`
cells are packed more tightly. `adjust_abundance=True` adds the log share of the `from` type in each image to the
model, so that groups are compared at the same abundance; it does not account for how tightly the `from` cells are
packed, and when that share differs strongly between groups, as for `HR+ CK7-` here, it leaves little power.
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
res_cov = spicyr.spicy(adata, condition="ER.Status", subject="metabricId", r=25,
                       covariates=["Age.At.Diagnosis", "Grade"])
res_cov
```

The main columns (`excess_difference`, `p_value`, `p_adj`) are now the ER comparison adjusted for age and grade.
Each covariate also has its own effect and p-value. A categorical covariate has one per level after the first:
`Grade2` and `Grade3` compare grades 2 and 3 with grade 1.

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

Restricting `from_` and `to` does not change the results for the pairs that are tested.

## Which radius?

The scale of an interaction is rarely known in advance. Give several radii and spicyr tests them together, using a
max-T test that accounts for the strong correlation between neighbouring radii. The exact max-T handles up to six
radii; for a finer grid, such as every 2 µm up to 50 µm, `combine="davies"` uses Davies' bound for the maximum over a
smooth range of radii, which stays close to the exact max-T when neighbouring radii are highly correlated.
`combine="cauchy"` is another alternative.

```{code-cell} ipython3
res_r = spicyr.spicy(adata, condition="ER.Status", subject="metabricId", r=[10, 25, 50, 75])
res_r.cell_results.loc[[pair, "B cells__HR- Ki67+"], ["r", "excess_difference", "p_value"]]
```

`r` is the radius with the strongest evidence, and `p_value` the combined p-value over all radii.

:::{tip}
The effect depends on the radius: at a larger radius more cells have a `from` cell nearby by chance, and the effect
describes placement at that scale. Where most cells have a `from` cell nearby by chance, there is little room for
an effect and it is noisy. Compare p-values across radii rather than the size of the effect. The effect at
the chosen radius is a little optimistic, because that radius was picked for its strength.
:::

```{code-cell} ipython3
rr = res_r.radius_results
fig, ax = plt.subplots(figsize=(6, 4))
for from_, g in rr[(rr["to"] == "HR- Ki67+") & rr["from"].isin(["T cells", "B cells"])].groupby("from"):
    ax.plot(g["r"], -np.log10(g["p_value"]), marker="o", label=from_)
ax.set(ylim=(0, None), xlabel="radius (µm)", ylabel="-log10 p at each radius, ER+ vs ER-")
ax.legend(title="from (to: HR- Ki67+)", frameon=False);
```

## Is co-localisation associated with survival?

With `survival=(time, event)`, spicyr asks whether a patient's effect is associated with their outcome. Here we use
relapse-free survival, adjusting for age.

```{code-cell} ipython3
res_s = spicyr.spicy(adata, survival=("timeRFS", "eventRFS"), subject="metabricId", r=25, covariates="Age.At.Diagnosis")
res_s
```

```{code-cell} ipython3
cols = ["from", "to", "p_value", "p_adj", "hazard_ratio_sd"]
res_s.cell_results.sort_values("p_value")[cols].head(6)
```

The p-value comes from a score test that relates each patient's effect to their outcome. `hazard_ratio_sd` is the
hazard ratio for a one standard deviation higher effect, from a Cox model; below one, patients with a higher effect
had a lower risk of relapse. It is missing when the effect barely varies between patients.

Two pairs are significant after adjusting for multiple testing, both with `HR+ CK7-` tumour cells as the `from`
type: patients in whom more of the other tumour cells sat next to `HR+ CK7-` cells relapsed later. These cells are
typical of ER+ tumours, and ER status is itself related to relapse, so we add it to the covariates.

```{code-cell} ipython3
res_s_er = spicyr.spicy(adata, survival=("timeRFS", "eventRFS"), subject="metabricId", r=25, covariates=["Age.At.Diagnosis", "ER.Status"])
res_s_er
```

No pair remains significant: these two pairs say little about relapse beyond the tumour's ER status.

## A check you can run

A test should give few significant results when there is nothing to find. Shuffling the ER labels across patients
removes any real difference, so about 5% of pairs should then have p < 0.05, and about 1% p < 0.01.
`calibration_check()` shuffles the condition and runs `spicy()` with the same arguments, ten times by default; here
five, to keep this page quick to build.

```{code-cell} ipython3
shuffles = spicyr.calibration_check(adata, condition="ER.Status", subject="metabricId", r=25, n_shuffles=5, rng=2026)
shuffles.groupby("threshold")["percent"].mean()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 3.5))
for k, (th, v) in enumerate(shuffles.groupby("threshold", sort=False)["percent"]):
    ax.scatter(np.full(len(v), k), v, color="black", alpha=0.7)
    ax.hlines(100 * th, k - 0.25, k + 0.25, color="#b3261e", lw=2)  # expected
ax.set_xticks([0, 1], [f"p < {th:g}" for th in shuffles["threshold"].unique()])
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

For a pair `from` → `to` and an image, let $O$ be the number of `to` cells with at least one `from` cell within $r$,
out of $n$ `to` cells. If the `to` cells were a random choice among the cells of the image that are not `from`
cells, keeping every cell where it is, $O$ would have an exact mean and variance, which spicyr computes without
permutations. The effect of an image is

$$\delta = \frac{O - \mathrm{E}_{\mathrm{RL}}(O)}{n - \mathrm{E}_{\mathrm{RL}}(O)},$$

where $\mathrm{E}_{\mathrm{RL}}(O)$ is that expectation under random labelling. If a fraction $f$ of the `to`
cells were placed next to `from` cells and the rest at random, $\delta$ would estimate $f$. At radius $r$, $O/n$ is
the cross-type nearest-neighbour distribution function $G$ of spatial statistics, and $\delta$ compares it with
random labelling, much as the J function compares $G$ with the empty-space function (van Lieshout and Baddeley
1996). Negative values mean
fewer `to` cells next to `from` cells than chance; they are not a fraction, and the lowest possible value,
$-\mathrm{E}_{\mathrm{RL}}(O)/(n - \mathrm{E}_{\mathrm{RL}}(O))$, is far below zero only where `from` cells are
common. With `effect="count"`, $O$ is instead the sum, over the `to` cells, of the number of `from` cells within $r$
of each, and the denominator is $n$, an analogue of Ripley's K function.

Images from the same patient are combined, giving more weight to more informative images (usually those with more
`to` cells), up to a limit: a patient's images can differ in their true effect, by an amount estimated from the
spread between each patient's images (DerSimonian and Laird), so one large image cannot outweigh the patient's other
images. Each patient has its own true effect, which varies around its group's mean by an amount estimated
from the data (a frailty, or random-effects, model). The difference between groups, adjusted for any covariates, is
tested on Satterthwaite degrees of freedom with patients as the clusters, using the small-sample cluster-robust (CR2)
variance. The groups share one dispersion: the model-based variance is scaled by the ratio
of each group's CR2 variance to its model-based variance, averaged over the groups, so a small group's own spread does
not set the standard error alone. A radius where a group's effective patients fall below the smaller of three and
half its patients at that radius, because a few patients carry nearly all of its weight, is not tested. This is designed to keep false positives near the nominal
rate even with modest numbers of patients. A paper describing the method is in preparation.

## Small studies

With fewer than about ten patients per group, expect few significant pairs, even when an effect is consistent across
patients: a calibrated test cannot be confident with little information. A group needs at least three
patients, and in practice a few more, since patients rarely carry equal weight. Look at whether the patients agree in
direction with `box_plot()`.

:::{note}
`variance="hartung_knapp"` can give more power in very small studies, but it can give too many small p-values for rare
cell types, so it is not the default.
:::

## Very large data sets: SpatialData stores

Some imaging studies hold more cells than fit in memory. `spicy()` also reads a SpatialData Zarr store, or an
AnnData Zarr store, from disk: give the path of the store as `cells`. Every neighbour count is a sum within one
image, so spicyr loads the image and cell-type columns once as integer codes and reads the coordinates one batch of
whole images at a time. Memory then depends on the size of an image and on the number of threads (`cores`), not on
the size of the study, and the results are the same as for the same cells in memory.

`write_cell_store()` writes cells as a SpatialData store. An AnnData object is written whole: its `obs` as the
table's `obs`, with the image column as its region key, and `X`, `layers`, `var`, `obsm` (with the coordinates in
`"spatial"`) and `uns` as anndata writes them; the cells of each image also become a points element. We write the
whole data set, so that the results can be compared with those above.

```{code-cell} ipython3
import tempfile
from pathlib import Path

path = spicyr.write_cell_store(adata, Path(tempfile.mkdtemp()) / "metabric.zarr")
```

`spicy()` reads the store with the same arguments as the object in memory, and gives the same results:

```{code-cell} ipython3
res_store = spicyr.spicy(str(path), condition="ER.Status", subject="metabricId", r=25)
pd.testing.assert_frame_equal(res_store.cell_results, res.cell_results, check_exact=True)
res_store.cell_results.equals(res.cell_results)
```

Every estimate, standard error and p-value is identical, not just close. The spatialdata package reads the same
store as a SpatialData object, which `spicy()` analyses in the same way:

```python
import spatialdata as sd

sdata = sd.read_zarr(path)
res = spicyr.spicy(sdata, condition="ER.Status", subject="metabricId", r=25)
```

A few arguments matter for stores. `table_key` chooses the cell table of a SpatialData store or object (by default
`"table"`). `image_info` supplies the condition, the patient, covariates or the survival columns when they are not
columns of the cell table: give a DataFrame (or an AnnData object) whose `image_id` column, or else its index,
holds the image IDs, or the key of a table in the store. Every image needs a row in it. `spatial_coords` names
either two `obs` columns or one `obsm` array, whose first two columns are used; by default `obsm["spatial"]`.

:::{tip}
Store the cells grouped by image. When the rows of each image sit together, as they do when the table is sorted by
its region column before it is written, each batch of images is read as one slice of rows. Otherwise each batch
needs a pass over the image codes. Categorical image and cell-type columns also keep the store small.
:::

In a benchmark on a synthetic store of one billion cells in 500 images, `spicy()` ran on one thread in under 4 GB of
memory.

## Reporting results

A methods sentence might read: "We used spicyr (version 1.99.15) to test, for every ordered pair of cell types,
whether the fraction of `to` cells with at least one `from` cell within 25 µm, relative to random labelling of the
cells in each image, differed between ER+ and ER− patients, with patients as the units of analysis. P-values were adjusted across pairs by the Benjamini–Hochberg method." Show a
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
for p in ["spicyr", "numpy", "pandas", "anndata", "zarr", "matplotlib", "plotly"]:
    print(p, md.version(p))
```
