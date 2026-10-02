# spicyr

**Calibrated tests for changes in cell-type co-localisation between groups of patients.**

`spicyr` is the Python twin of the Bioconductor package
[spicyR](https://bioconductor.org/packages/spicyR). Both run the same C++ core and give the same results. It
works directly with [SpatialData](https://spatialdata.scverse.org), [AnnData](https://anndata.readthedocs.io)
and pandas objects.

For every ordered pair of cell types *from* → *to*, `spicy()` asks whether the number of *to* cells within a
radius of each *from* cell changes between conditions, with **patients as the units**:

- the comparison is with random labelling of the observed cells, so tissue shape and density are conditioned on;
- images are combined within patients by a frailty GEE, and the difference is tested with a small-sample
  cluster-robust (CR2) variance on Satterthwaite degrees of freedom;
- every result also gives the difference at equal availability of the counted type.

```{toctree}
:maxdepth: 2

tutorial
api
```
