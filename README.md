# spicyr

[![tests](https://github.com/SydneyBioX/spicyr-py/actions/workflows/test.yml/badge.svg)](https://github.com/SydneyBioX/spicyr-py/actions/workflows/test.yml)
[![docs](https://github.com/SydneyBioX/spicyr-py/actions/workflows/docs.yml/badge.svg)](https://sydneybiox.github.io/spicyr-py)
![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![licence](https://img.shields.io/badge/licence-GPL%20(%E2%89%A52)-lightgrey)

**Test whether cell types co-localise differently between groups of patients.**

Do T cells gather around tumour cells more in one group of patients than in another? spicyr tests this for every
pair of cell types in imaging and spatial transcriptomics data, comparing groups of patients or relating
co-localisation to survival. For each image it counts the cells of one type within a radius of each cell of another,
and compares that count with what you would expect if the cells had been labelled at random, using the cells
actually present. Holes, air spaces and uneven cell density therefore do not by themselves create a signal.
Patients, not images or cells, are the units of the test. It needs the type and position of every cell (for
example imaging mass cytometry, CODEX, MIBI, Xenium or CosMx), not spot-based data.

spicyr is the Python version of the R package [spicyR 2.0](https://github.com/SydneyBioX/spicyR/tree/spicyR2) and
works with SpatialData, AnnData and pandas objects.

![Left, a tumour cell with a 25 µm circle and the T cells inside it. Right, box plots of the extra T cells per tumour cell, one point per patient, higher in ER-positive than ER-negative tumours.](docs/_static/spicyR_overview.png)

## Quick start

```python
import spicyr

cells = spicyr.datasets.metabric_ali2020()                      # breast cancer imaging mass cytometry
cells = cells[cells["ER.Status"].isin(["neg", "pos"])]
res = spicyr.spicy(cells, condition="ER.Status", subject="metabricId", r=25,   # reference group: "neg"
                   image_id="file_id", cell_type="description")
res.top_pairs()                         # the most significant pairs
res.signif_plot()                       # every pair at a glance
res.box_plot("HR- Ki67+", "T cells")    # T cells around proliferating tumour cells, per image
```

For your own data, `spicy()` looks for the columns `imageID` and `cellType` (name yours with `image_id=` and
`cell_type=`) and takes coordinates from `obsm["spatial"]` for AnnData, from the annotation table for SpatialData,
or from columns `x` and `y` of a DataFrame. A pair *from* → *to* asks how many extra *to* cells sit around each
*from* cell.

## What you get

- A table with one row per pair of cell types: the number of extra neighbours per cell in each group, the
  difference, a p-value and an FDR-adjusted p-value, and the same after adjusting for how common each cell type is.
- A plot of every pair at once, and the per-image values behind any pair.
- The same test with covariates, several radii, more than two groups, or a survival outcome.

## Installation

```bash
pip install "spicyr[plot,data,anndata] @ git+https://github.com/SydneyBioX/spicyr-py"
```

Until wheels are on PyPI this builds from source and needs a C++17 compiler. `plot` adds matplotlib, `data` the
example data sets, `anndata` and `spatialdata` those inputs.

## Learn more

- [Introduction to spicyr](https://sydneybiox.github.io/spicyr-py/tutorial.html): a full analysis of breast cancer
  imaging mass cytometry data, the same as the spicyR vignette.
- [Reference](https://sydneybiox.github.io/spicyr-py/api.html).
- [spicyR](https://github.com/SydneyBioX/spicyR), the R package.

## Citation

Canete NP, Iyengar SS, Ormerod JT, Baharlou H, Harman AN, Patrick E (2022). spicyR: spatial analysis of in situ
cytometry data in R. *Bioinformatics* 38(11), 3099–3105.
[doi:10.1093/bioinformatics/btac268](https://doi.org/10.1093/bioinformatics/btac268). A paper describing the
cell-level test is in preparation.

## Contact

Questions and bug reports: [GitHub issues](https://github.com/SydneyBioX/spicyr-py/issues) or
[ellis.patrick@sydney.edu.au](mailto:ellis.patrick@sydney.edu.au). For developers:
[CONTRIBUTING](CONTRIBUTING.md).

spicyr 1.99.0 is a development version, released together with spicyR 2.0. Licence: GPL (>= 2).
