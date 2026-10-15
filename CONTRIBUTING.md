# Contributing to spicyr

spicyr is the Python twin of the Bioconductor package [spicyR](https://github.com/SydneyBioX/spicyR). The two
share one C++ core, so that they give the same results.

## The shared core

- `src/core` is a verbatim copy of `src/core` in spicyR. Never edit it here: change it in spicyR, then run
  `tools/sync_core.sh /path/to/spicyR`, which copies it and records its checksums in `src/core/CHECKSUMS`.
  `tests/test_core_sync.py` fails if the copy has been edited.
- `src/bindings.cpp` (pybind11) mirrors spicyR's R bindings, and `src/spicyr/_cell.py` and `_api.py` mirror
  spicyR's `R/cell.R` and `R/spicy_main.R` line for line. No statistics live in the Python code.
- The cell method's loop over images lives in the core (`src/core/src/stream.cpp`, `spicyglm/stream.hpp`), shared
  with spicyR: the passes over the image and type codes (`scan_codes`, `constant_within`, `rows_of_images`) and the
  neighbour sums of a batch of whole images (`rows_totals`), each image in a Dataset of its own, the images shared
  among threads. The bindings only describe NumPy arrays to it; reading the cells (pandas, AnnData, zarr in
  `_source.py`) and choosing the batches (`_cell.cell_stream()`) stay in Python.

## Keeping the twins identical

`tests/shared_cases/make_cases.R` runs spicyR on a set of small cases (two and three conditions, covariates with
missing values, k nearest neighbours, several radii, the Hartung-Knapp variance and survival) and writes the inputs
and results to `tests/shared_cases/`. `tests/test_shared_cases.py` requires spicyr to reproduce every number
(they agree to about 1e-13). Regenerate the cases after any change to the core or to either front end.

## Scaling tests

`tools/make_synthetic_store.py` writes a synthetic SpatialData store of any size (uniform cells in square images,
rows grouped by image, chunked by rows), one chunk at a time, for timing `spicy()` on data that do not fit in memory.

## Building

```bash
pip install -e ".[test,data,plot]"      # needs a C++17 compiler; Eigen is downloaded if not found
pytest -q
pip install ".[doc]" && sphinx-build -b html docs docs/_build/html   # the tutorial is executed
```
