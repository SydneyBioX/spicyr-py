# spicyr

**Test whether cell types co-localise differently between groups of patients.**

Do T cells gather around tumour cells more in one group of patients than in another? spicyr answers questions
like this for every pair of cell types in imaging and spatial transcriptomics data, comparing groups of patients
or relating co-localisation to survival. For each image it counts how many cells of one type lie within a radius of
each cell of another, and compares that count with what random labelling of the same cells would give, so holes,
folds and dense regions in the tissue are not mistaken for biology. Patients, not images or cells, are the units of
the test. spicyr is the Python twin of the Bioconductor package
[spicyR](https://bioconductor.org/packages/spicyR) and works with SpatialData, AnnData and pandas objects.

![T cells around proliferating tumour cells in breast cancer, compared between ER-negative and ER-positive patients](docs/_static/spicyR_overview.png)

## Quick start

```python
import spicyr

res = spicyr.spicy(adata, condition="response", subject="patient", r=25)
res.top_pairs()               # the most significant pairs
res.signif_plot()             # every pair at a glance
res.box_plot("CD8 T cells", "Tumour")
```

`adata` can be an AnnData object (cell types and the image of each cell in `.obs`, coordinates in
`.obsm["spatial"]`), a SpatialData object or a pandas DataFrame with one row per cell.

## What you get

- A table with one row per pair of cell types: the number of extra neighbours per cell in each group, the
  difference, a p-value and an FDR-adjusted p-value.
- A plot of every pair at once, and the per-patient values behind any pair.
- The same test with covariates, several radii, more than two groups, or a survival outcome.

## Installation

```bash
pip install spicyr                     # once released
pip install git+https://github.com/SydneyBioX/spicyr-py
```

## Learn more

- [Introduction to spicyr](https://sydneybiox.github.io/spicyr-py/tutorial.html): a full analysis of breast cancer
  imaging mass cytometry data, the same as the spicyR vignette.
- [Function reference](https://sydneybiox.github.io/spicyr-py/api.html).
- [spicyR](https://github.com/SydneyBioX/spicyR), the R package.

## Citation

Canete NP, Iyengar SS, Ormerod JT, Baharlou H, Harman AN, Patrick E (2022). spicyR: spatial analysis of in situ
cytometry data in R. *Bioinformatics* 38(11), 3099–3105.
[doi:10.1093/bioinformatics/btac268](https://doi.org/10.1093/bioinformatics/btac268)

## Contact

Questions and bug reports: [GitHub issues](https://github.com/SydneyBioX/spicyr-py/issues) or
[ellis.patrick@sydney.edu.au](mailto:ellis.patrick@sydney.edu.au). Developers: see [CONTRIBUTING](CONTRIBUTING.md).

spicyr 1.99.0 is a pre-release, developed alongside spicyR 2.0; a paper describing the method is in preparation.
Licence: GPL (>= 2).
