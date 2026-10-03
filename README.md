# spicyr

[![tests](https://github.com/SydneyBioX/spicyr-py/actions/workflows/test.yml/badge.svg)](https://github.com/SydneyBioX/spicyr-py/actions/workflows/test.yml)
[![docs](https://github.com/SydneyBioX/spicyr-py/actions/workflows/docs.yml/badge.svg)](https://sydneybiox.github.io/spicyr-py)
![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![licence](https://img.shields.io/badge/licence-GPL%20(%E2%89%A52)-lightgrey)

**Test whether cell types co-localise differently between groups of patients.**

Do T cells gather around tumour cells more in one group of patients than in another? spicyr tests this for every
pair of cell types in imaging and spatial transcriptomics data, comparing groups of patients or relating
co-localisation to survival. For each image it asks what fraction of the cells of one type have a cell of another
type within a radius, and compares that with what you would expect if the cells had been labelled at random, using
the cells actually present. Holes, air spaces and uneven cell density therefore do not by themselves create a signal.
Patients, not images or cells, are the units of the test. It needs the type and position of every cell (for
example imaging mass cytometry, CODEX, MIBI, Xenium or CosMx), not spot-based data.

![Left, tumour cells marked by whether a T cell lies within 25 µm, with the circle drawn around two of them. Right, box plots of the extra fraction of tumour cells next to T cells, one point per patient, higher in ER-positive than ER-negative tumours.](docs/_static/spicyR_overview.png)

## Quick start

```python
import spicyr

adata = spicyr.datasets.metabric_ali2020(as_anndata=True)      # breast cancer imaging mass cytometry
adata = adata[adata.obs["ER.Status"].isin(["neg", "pos"])].copy()
res = spicyr.spicy(adata, condition="ER.Status", subject="metabricId", r=25,   # reference group: "neg"
                   image_id="file_id", cell_type="description")
res.top_pairs()                                        # the most significant pairs
res.signif_plot()                                      # every pair at a glance
res.box_plot("T cells", "HR- Ki67+", interactive=True)  # proliferating tumour cells next to T cells, per image
```

For your own data, `spicy()` accepts an AnnData object, a SpatialData object or a pandas DataFrame. It looks for
the columns `imageID` and `cellType` (name yours with `image_id=` and `cell_type=`) and takes coordinates from
`obsm["spatial"]`, from the SpatialData annotation table, or from columns `x` and `y`. A pair `from` → `to` asks
whether `to` cells are placed near `from` cells more than other cells are: the extra fraction of `to` cells with a
`from` cell nearby (`effect="count"` gives the number of extra `from` cells around each `to` cell instead).

## What you get

- A table with one row per pair of cell types: the extra fraction of `to` cells next to `from` cells in each
  group, the difference, a p-value and an FDR-adjusted p-value. Each `to` cell counts once, however many `from`
  cells are beside it, so more numerous or more tightly packed `from` cells do not inflate the effect.
- A plot of every pair at once, the per-image values behind any pair (interactive, to find the images worth
  looking at), and a plot of any image.
- The same test with covariates, several radii, more than two groups, or a survival outcome.

## Installation

```bash
pip install "spicyr[plot,data,anndata] @ git+https://github.com/SydneyBioX/spicyr-py"
```

Until spicyr is on PyPI this builds from source and needs a C++17 compiler. `plot` adds matplotlib and plotly,
`data` the example data sets, `anndata` and `spatialdata` those inputs.

## Learn more

- [Introduction to spicyr](https://sydneybiox.github.io/spicyr-py/tutorial.html): a full analysis of breast cancer
  imaging mass cytometry data.
- [Reference](https://sydneybiox.github.io/spicyr-py/api.html).
- [spicyR](https://github.com/SydneyBioX/spicyR), the same test in R, for Bioconductor's `SpatialExperiment`.

## Citation

Canete NP, Iyengar SS, Ormerod JT, Baharlou H, Harman AN, Patrick E (2022). spicyR: spatial analysis of in situ
cytometry data in R. *Bioinformatics* 38(11), 3099–3105.
[doi:10.1093/bioinformatics/btac268](https://doi.org/10.1093/bioinformatics/btac268). A paper describing the
cell-level test is in preparation.

## Contact

Questions and bug reports: [GitHub issues](https://github.com/SydneyBioX/spicyr-py/issues) or
[ellis.patrick@sydney.edu.au](mailto:ellis.patrick@sydney.edu.au). For developers:
[CONTRIBUTING](CONTRIBUTING.md).

spicyr 1.99.5 is a development version. Authors: Ellis Patrick, Sadiq Dohadwalla, Elijah Willie and Nicolas Canete.
Licence: GPL (>= 2).
