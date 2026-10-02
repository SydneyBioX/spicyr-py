# spicyr

**Calibrated tests for changes in cell-type co-localisation between groups of patients.**

`spicyr` is the Python twin of the Bioconductor package [spicyR](https://bioconductor.org/packages/spicyR):
the same C++ core, the same arguments and the same results (they agree to about 1e-13 on a shared set of test
cases). It works directly with **SpatialData**, **AnnData** and **pandas** objects.

> Status: pre-release (1.99.0), developed alongside spicyR 2.0. The statistical methods are under review.

## What it tests

For every ordered pair of cell types *from* → *to*, spicyR Cell asks whether the number of *to* cells within
radius *r* of each *from* cell differs between conditions (or is associated with survival).

- **The effect is the excess:** the number of extra *to* cells per *from* cell, beyond what random labelling
  of the observed cells would give. Tissue shape, holes and cell density are conditioned on, not modelled.
- **Patients are the units.** Images are combined within patients and patients within conditions by a frailty
  GEE, and the difference is tested with a small-sample (CR2) cluster-robust variance on Satterthwaite degrees of
  freedom. Cells and images are never treated as independent replicates.
- **Calibrated by design.** It is built to keep its nominal error rate with few patients; in our benchmarks
  (condition labels shuffled across patients) it did, on every real dataset tried. These results are under
  review for publication.
- **Abundance guard.** Each result also gives the difference at equal availability of the *to* type, so that a
  change in abundance is not read as a change in attraction.
- **More:** several conditions, covariates, k nearest neighbours, several radii (max-T with the sandwich
  correlation, or Cauchy), and survival outcomes (score test and hazard ratio).

## Installation

```bash
pip install spicyr                 # once released
pip install "spicyr[spatialdata,plot]"
```

From source (needs a C++17 compiler; Eigen is downloaded automatically):

```bash
pip install git+https://github.com/SydneyBioX/spicyr-py
```

## Quick start

```python
import spicyr

# a SpatialData object: the table's region key gives the images; coordinates from obsm["spatial"]
# or the centroids of the annotated shapes
res = spicyr.spicy(sdata, condition="group", subject="patient", r=50, image_id="region")

# an AnnData object (coordinates in obsm["spatial"]) or a pandas DataFrame work the same way
res = spicyr.spicy(adata, condition="group", subject="patient", image_id="sample", cell_type="cell_type")

res.top_pairs(10)            # the most significant pairs
res.cell_results             # the full table: excess per group, difference, SE, df, p, BH, adjusted difference
res.box_plot("tumour", "T")  # per-image excess by condition
res.signif_plot()            # every pair at a glance

# survival: time and event columns (one value per patient)
surv = spicyr.spicy(adata, condition="os_time", survival="os_event", subject="patient", r=50)

# several radii, combined by max-T
multi = spicyr.spicy(adata, condition="group", subject="patient", r=[10, 25, 50])
```

`from` is a Python keyword, so the argument is `from_` (or `**{"from": "tumour"}`).

## Twins

The C++ core in `src/core` is a verbatim copy of spicyR's (`tools/sync_core.sh`; `tests/test_core_sync.py`
checks the checksums). `tests/shared_cases` holds inputs and the results computed by the R package
(`make_cases.R`); `tests/test_shared_cases.py` requires the Python package to reproduce them.

## Citation

Canete NP, Iyengar SS, Ormerod JT, Baharlou H, Harman AN, Patrick E (2022). spicyR: spatial analysis of in situ
cytometry data in R. *Bioinformatics* 38(11), 3099-3105. The spicyR Cell method: manuscript in preparation.

## Licence

GPL (>= 2), as spicyR.
