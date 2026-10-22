"""``write_cell_store()``: cells as the cell table of a SpatialData Zarr store, written a chunk of rows at a time.

The twin of spicyR's ``writeCellStore()``. Each table is an AnnData group in the encoding anndata gives it, in a Zarr
v3 store. Every array of the length of the table is cut into chunks of ``chunk_rows`` rows; each chunk is taken
from the input, compressed with zstd and written before the next is made, so no whole column is ever copied.
"""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ZARR_MISSING = 'writing a store needs zarr (version 3 or later): pip install "spicyr[store]"'
_ARROW_MISSING = 'writing the cells as points needs pyarrow: pip install "spicyr[store]" (or points=False)'


def _zarr():
    try:
        import zarr
    except ImportError as e:
        raise ImportError(_ZARR_MISSING) from e
    if int(zarr.__version__.split(".")[0]) < 3:
        raise ImportError(_ZARR_MISSING)
    return zarr


def _chunks(n, ch):
    return [(a, min(a + ch, n)) for a in range(0, n, ch)]


def _uint_type(hi):
    """The smallest unsigned integer type holding 0..hi."""
    for t in (np.uint8, np.uint16, np.uint32):
        if hi <= np.iinfo(t).max:
            return t
    return np.uint64


class _Writer:
    def __init__(self, zarr, chunk_rows):
        from zarr.codecs import ZstdCodec

        self.zarr = zarr
        self.ch = int(chunk_rows)
        self.codec = ZstdCodec(level=0, checksum=False)

    def group(self, parent, name, attrs=None):
        return parent.create_group(name, attributes=attrs or {})

    def rows(self, parent, name, n, dtype, get, ncol=None, encoding=("array", "0.2.0")):
        """An array of n rows (and `ncol` columns), written a chunk of rows at a time: get(a, b) gives rows a..b-1."""
        string = dtype is str
        ch = max(1, min(self.ch, n))
        fill = "" if string else (False if np.dtype(dtype) == np.bool_ else 0)
        attrs = {"encoding-type": encoding[0], "encoding-version": encoding[1]} if encoding else {}
        arr = parent.create_array(
            name,
            shape=(n,) if ncol is None else (n, ncol),
            chunks=(ch,) if ncol is None else (ch, ncol),
            dtype=dtype,
            compressors=self.codec,
            fill_value=fill,
            attributes=attrs,
            config={"write_empty_chunks": True},
        )
        for a, b in _chunks(n, ch):
            arr[a:b] = get(a, b)
        return arr

    def string_scalar(self, parent, name, s):
        """A zero-dimensional string array (anndata's encoding of a string in uns)."""
        arr = parent.create_array(
            name,
            shape=(),
            dtype=str,
            compressors=self.codec,
            fill_value="",
            attributes={"encoding-type": "string", "encoding-version": "0.2.0"},
        )
        arr[...] = s

    def strings(self, parent, name, values, encoding=("string-array", "0.2.0")):
        """A short array of strings in one chunk (categories, regions)."""
        v = np.asarray([str(x) for x in values], dtype=object)
        return self.rows(parent, name, len(v), str, lambda a, b: v[a:b], encoding=encoding)

    def column(self, parent, name, col, n):
        """One obs column, as anndata encodes it.

        A categorical as codes and categories, a column with missing values as a nullable array (values and mask),
        any other column as an array.
        """
        dtype = col.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            g = self.group(
                parent,
                name,
                {"encoding-type": "categorical", "encoding-version": "0.2.0", "ordered": bool(dtype.ordered)},
            )
            cats = dtype.categories
            if cats.dtype.kind in "biuf":
                v = cats.to_numpy()
                self.rows(g, "categories", len(v), v.dtype, lambda a, b: v[a:b])
            else:
                self.strings(g, "categories", cats)
            codes = col.array.codes  # the stored codes, not a copy
            self.rows(g, "codes", n, codes.dtype, lambda a, b: codes[a:b])
            return
        if isinstance(dtype, pd.StringDtype) or (isinstance(dtype, np.dtype) and dtype.kind == "O"):
            self._strings_column(parent, name, col, n)
            return
        if isinstance(dtype, pd.BooleanDtype) or (
            isinstance(dtype, pd.api.extensions.ExtensionDtype) and pd.api.types.is_integer_dtype(dtype)
        ):
            arr = col.array
            if isinstance(dtype, pd.BooleanDtype):
                kind, vtype, fill, enc, extra = "nullable-boolean", np.bool_, False, ("array", "0.2.0"), {}
            else:
                kind, vtype, fill, enc, extra = "nullable-integer", dtype.numpy_dtype, 0, ("array", "0.2.0"), {}
            self._nullable(parent, name, n, arr, kind, vtype, fill, enc, extra)
            return
        if dtype.kind in "biuf":
            v = col.to_numpy()
            self.rows(parent, name, n, v.dtype, lambda a, b: v[a:b])
            return
        raise ValueError(f"column '{name}' is not categorical, strings, numbers or logical (its type is {dtype}).")

    def _strings_column(self, parent, name, col, n):
        """Strings: a string array, or a nullable string array when some are missing."""
        v = col.array if isinstance(col.dtype, pd.StringDtype) else col.to_numpy()
        na = False
        for a, b in _chunks(n, self.ch):
            s = v[a:b]
            if pd.api.types.infer_dtype(s, skipna=True) not in ("string", "empty"):
                raise ValueError(f"column '{name}' holds values that are not strings, numbers or logical.")
            na = na or bool(pd.isna(s).any())
        if not na:
            self.rows(
                parent, name, n, str, lambda a, b: np.asarray(v[a:b], dtype=object), encoding=("string-array", "0.2.0")
            )
        else:
            self._nullable(
                parent, name, n, v, "nullable-string-array", str, "", ("string-array", "0.2.0"), {"na-value": "NA"}
            )

    def region_column(self, parent, name, col, n, regions):
        """The image column as a categorical whose categories are the regions (SpatialData's region key must be one)."""
        g = self.group(parent, name, {"encoding-type": "categorical", "encoding-version": "0.2.0", "ordered": False})
        self.strings(g, "categories", regions)
        idx = pd.Index(regions)
        t = np.int8 if len(regions) < 128 else np.int16 if len(regions) < 32768 else np.int32
        v = col.to_numpy()

        def codes(a, b):
            s = pd.Series(v[a:b])
            c = idx.get_indexer(s.astype(str)).astype(t)
            c[s.isna().to_numpy()] = -1
            return c

        self.rows(g, "codes", n, t, codes)

    def _nullable(self, parent, name, n, values, kind, vtype, fill, encoding, extra):
        g = self.group(parent, name, {"encoding-type": kind, "encoding-version": "0.1.0", **extra})

        def vals(a, b):
            s = values[a:b]
            m = np.asarray(pd.isna(s), dtype=bool)
            if vtype is str:
                out = np.asarray(s, dtype=object)
                out[m] = fill
                return out
            return s.to_numpy(dtype=vtype, na_value=fill)

        self.rows(g, "values", n, vtype, vals, encoding=encoding)
        self.rows(g, "mask", n, np.bool_, lambda a, b: np.asarray(pd.isna(values[a:b]), dtype=bool))

    def anndata(
        self,
        tables,
        key,
        obs,
        cols,
        n,
        index,
        instance=False,
        spatial=None,
        spatial_dtype=None,
        sd_attrs=None,
        attrs=None,
    ):
        """An AnnData table of n rows.

        obs holds the columns `cols` of `obs` (and, with `instance`, 0-based instance IDs as "instance_id"), its index
        given a chunk at a time by index(a, b); obsm "spatial" by spatial(a, b); uns "spatialdata_attrs" from
        `sd_attrs`; `attrs` are added to the table's attributes.
        """
        tab = self.group(tables, key, {"encoding-type": "anndata", "encoding-version": "0.1.0", **(attrs or {})})
        order = [*cols, *(["instance_id"] if instance else [])]
        op = self.group(
            tab,
            "obs",
            {"encoding-type": "dataframe", "encoding-version": "0.2.0", "_index": "_index", "column-order": order},
        )
        self.rows(op, "_index", n, str, index, encoding=("string-array", "0.2.0"))
        for c in cols:  # one column at a time: obs[c] shares the frame's data
            if (
                sd_attrs is not None
                and c == sd_attrs["region_key"]
                and not isinstance(obs[c].dtype, pd.CategoricalDtype)
            ):
                self.region_column(op, c, obs[c], n, sd_attrs["region"])
            else:
                self.column(op, c, obs[c], n)
        if instance:
            # instance IDs in 16 bits at least (spatialdata refuses an 8-bit instance key)
            t = _uint_type(max(n - 1, 256))
            self.rows(op, "instance_id", n, t, lambda a, b: np.arange(a, b, dtype=t))
        vp = self.group(
            tab,
            "var",
            {"encoding-type": "dataframe", "encoding-version": "0.2.0", "_index": "_index", "column-order": []},
        )
        self.rows(vp, "_index", 0, str, None, encoding=("string-array", "0.2.0"))
        groups = {
            g: self.group(tab, g, {"encoding-type": "dict", "encoding-version": "0.1.0"})
            for g in ("layers", "obsm", "obsp", "uns", "varm", "varp")
        }
        if spatial is not None:
            self.rows(groups["obsm"], "spatial", n, spatial_dtype, spatial, ncol=2)
        if sd_attrs is not None:
            sp = self.group(groups["uns"], "spatialdata_attrs", {"encoding-type": "dict", "encoding-version": "0.1.0"})
            self.strings(sp, "region", sd_attrs["region"])
            for k in ("region_key", "instance_key"):
                self.string_scalar(sp, k, sd_attrs[k])


def _index_strings(index):
    """The row names a chunk at a time, as strings (a default RangeIndex without making all the strings at once)."""
    if isinstance(index, pd.RangeIndex):
        return lambda a, b: np.asarray(index[a:b].astype(str), dtype=object)
    return lambda a, b: np.asarray([str(x) for x in index[a:b]], dtype=object)


def _names(cols, what):
    if not all(isinstance(c, str) for c in cols) or len(set(cols)) != len(cols):
        raise ValueError(f"the column names of `{what}` must be distinct strings.")
    return cols


def _regions(img, n, ch):
    """The images in order of first appearance, a chunk of rows at a time."""
    seen = {}
    if isinstance(img.dtype, pd.CategoricalDtype):
        codes, cats = img.array.codes, img.dtype.categories
        for a, b in _chunks(n, ch):
            for c in pd.unique(codes[a:b]):
                if c >= 0:
                    seen.setdefault(c, None)
        return [str(cats[c]) for c in seen]
    v = img.to_numpy()
    for a, b in _chunks(n, ch):
        for s in pd.unique(v[a:b]):
            if not pd.isna(s):
                seen.setdefault(str(s), None)
    return list(seen)


def _write_kwargs(value, ch):
    """Chunks for anndata's write_elem(): rows of a dense matrix (about `ch` values per chunk), or `ch` values."""
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return {"chunks": (max(1, min(len(value), (2 * ch) // max(1, value.shape[1]))), value.shape[1])}
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return {"chunks": (max(1, min(len(value), ch)),)}
    if hasattr(value, "tocsr"):  # a sparse matrix: its data and indices arrays
        return {"chunks": (ch,)}
    return {}


def _write_anndata_parts(table, adata, ch, spatial_written):
    """X, layers, var, obsm, varm, obsp, varp and uns of an AnnData object, with anndata's own writer.

    The table's obs (with the region key and instance IDs) is spicyr's; everything else is written as anndata
    writes it, chunked. `spatial_written`: whether obsm "spatial" was written from obs columns.
    """
    from anndata.io import write_elem

    del table["var"]
    write_elem(table, "var", adata.var)
    if adata.X is not None:
        write_elem(table, "X", adata.X, dataset_kwargs=_write_kwargs(adata.X, ch))
    for group, mapping in (
        ("layers", adata.layers),
        ("obsm", adata.obsm),
        ("varm", adata.varm),
        ("obsp", adata.obsp),
        ("varp", adata.varp),
    ):
        for k, v in mapping.items():
            if group == "obsm" and k == "spatial" and spatial_written:
                raise ValueError("obsm has a 'spatial' array as well as the coordinate columns of obs.")
            write_elem(table[group], k, v, dataset_kwargs=_write_kwargs(v, ch))
    for k, v in adata.uns.items():
        if k != "spatialdata_attrs":  # the table's own, written by spicyr
            write_elem(table["uns"], k, v)


def _valid_name(x):
    """A name spatialdata accepts for an element: letters, digits, "_", "-" and ".", not "." or "..", nor "__..."."""
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", x)) and x not in (".", "..") and not x.startswith("__")


def _write_points(root, img, regions, x, y, n, ch):
    """The cells as points: one points element per image.

    Each element holds x, y and the instance ID that links each point to its row of the table, in the image's own
    coordinate system. Each chunk of rows is sorted by image and each image's rows appended to that image's parquet file, so every
    element is one file (spatialdataR reads only part.0.parquet) and memory stays that of a chunk.
    """
    bad = [r for r in regions if not _valid_name(r)]
    if bad:
        warnings.warn(
            f"no points written: images whose IDs are not valid SpatialData element names: {bad[:5]}", stacklevel=3
        )
        return []
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(_ARROW_MISSING) from e
    schema = pa.schema([("x", pa.float64()), ("y", pa.float64()), ("instance_id", pa.int64())])
    points = root.create_group("points")
    files = {}
    try:
        for a, b in _chunks(n, ch):
            codes = pd.Categorical(np.asarray(img[a:b]).astype(object).astype(str), categories=regions).codes
            order = np.argsort(codes, kind="stable")
            sorted_codes = codes[order]
            for k in np.unique(sorted_codes[sorted_codes >= 0]):
                lo, hi = np.searchsorted(sorted_codes, [k, k + 1])
                rows = order[lo:hi]
                t = pa.table(
                    {
                        "x": np.asarray(x[a:b], float)[rows],
                        "y": np.asarray(y[a:b], float)[rows],
                        "instance_id": (a + rows).astype(np.int64),
                    },
                    schema=schema,
                )
                if k not in files:
                    name = regions[k]
                    system = {
                        "name": name,
                        "axes": [
                            {"name": "x", "type": "space", "unit": "unit"},
                            {"name": "y", "type": "space", "unit": "unit"},
                        ],
                    }
                    points.create_group(
                        name,
                        attributes={
                            "encoding-type": "ngff:points",
                            "axes": ["x", "y"],
                            "coordinateTransformations": [
                                {"type": "identity", "input": {**system, "name": "xy"}, "output": system}
                            ],
                            "spatialdata_attrs": {"instance_key": "instance_id", "version": "0.2"},
                        },
                    )
                    d = Path(points.store.root) / "points" / name / "points.parquet"
                    d.mkdir(parents=True, exist_ok=True)
                    files[k] = pq.ParquetWriter(d / "part.0.parquet", schema)
                files[k].write_table(t)
    finally:
        for f in files.values():
            f.close()
    return [regions[k] for k in sorted(files)]


def write_cell_store(
    cells,
    path,
    image_id="imageID",
    spatial_coords=("x", "y"),
    table_key="table",
    image_info=None,
    info_key="images",
    chunk_rows=2**20,
    points=True,
):
    """Write cells as a SpatialData Zarr store.

    The store holds the cells as an AnnData table under ``tables/`` (Zarr v3), which :func:`spicy` reads a block of
    rows at a time and ``spatialdata.read_zarr()`` reads as a SpatialData object. The coordinates become the
    ``obsm`` array ``"spatial"`` and the other columns the table's ``obs`` (categorical columns stay categorical).
    An AnnData object is written whole: ``X``, ``layers``, ``var``, ``obsm``, ``varm``, ``obsp``, ``varp`` and
    ``uns`` as anndata writes them (chunked), and its ``obs`` as described here. The cells of each image also become
    a points element named after the image (x, y and the ``"instance_id"`` that links each point to its row of the
    table), so the table annotates elements of the store.
    The image column is the table's region key: a categorical column whose categories are the images in order of
    first appearance, unless it is categorical already. A column ``"instance_id"`` numbers the cells from 0
    (unsigned integers of 16 bits at least). A table with one row per image can be added as a second table. Every
    array of the length of the table is written in chunks of ``chunk_rows`` rows compressed with zstd, one chunk at
    a time, so writing needs little memory beyond that of ``cells``, however many cells it has.

    Parameters
    ----------
    cells
        A pandas DataFrame with one row per cell, or an AnnData object (cells as observations; its ``obs`` is
        written, with the coordinates from the obs columns ``spatial_coords`` or else from ``obsm["spatial"]``).
        Columns are categorical, strings, integers, floats or booleans (pandas' nullable types included).
    path
        The directory of the store (replaced if it exists).
    image_id
        The column of the image.
    spatial_coords
        The two columns of the coordinates. They are written as float64 (float32 coordinates stay float32).
    table_key
        The name of the cell table.
    image_info
        Optional: a DataFrame (or an AnnData object, its ``obs``) with one row per image, written as the table
        ``info_key``. Its ``image_id`` column, or else its index, gives the image IDs.
    info_key
        The name of the per-image table.
    chunk_rows
        Rows per chunk of every array of the length of a table.
    points
        Whether to write the cells as points, one points element per image (needs pyarrow). Image IDs must then be
        valid element names (letters, digits, ``_``, ``-`` and ``.``); otherwise no points are written, with a warning.

    Returns
    -------
    The path of the store.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> cells = pd.DataFrame({"imageID": pd.Categorical(np.repeat(["a", "b"], 50)),
    ...                       "cellType": pd.Categorical(np.tile(["T", "B"], 50)),
    ...                       "x": rng.uniform(0, 100, 100), "y": rng.uniform(0, 100, 100)})
    >>> path = write_cell_store(cells, "cells.zarr")  # doctest: +SKIP
    """
    zarr = _zarr()
    adata = None
    if type(cells).__name__ == "AnnData":
        from ._input import _anndata_coords

        adata = cells
        obs = cells.obs
        x, y = _anndata_coords(cells, spatial_coords)
        index = cells.obs_names
    elif isinstance(cells, pd.DataFrame):
        obs = cells
        miss = [c for c in spatial_coords if c not in obs.columns]
        if miss:
            raise ValueError(f"columns not found in `cells`: {', '.join(miss)}")
        x, y = obs[spatial_coords[0]], obs[spatial_coords[1]]
        index = obs.index
    else:
        raise TypeError("`cells` must be a pandas DataFrame or an AnnData object.")
    if image_id not in obs.columns:
        raise ValueError(f"column '{image_id}' not found in `cells`.")
    if len(spatial_coords) != 2:
        raise ValueError("`spatial_coords` must name two columns.")
    x, y = np.asarray(x), np.asarray(y)  # views of the columns (or of obsm["spatial"]), not copies
    xy_type = np.float32 if x.dtype == np.float32 and y.dtype == np.float32 else np.float64
    obs_cols = _names([c for c in obs.columns if c not in spatial_coords], "cells")
    if "instance_id" in obs_cols:
        raise ValueError("`cells` has a column 'instance_id'; the store numbers the cells in a column of that name.")
    info_index = None
    if image_info is not None:
        if type(image_info).__name__ == "AnnData":
            image_info = image_info.obs
        if not isinstance(image_info, pd.DataFrame):
            raise TypeError("`image_info` must be a DataFrame with one row per image.")
        if image_id in image_info.columns:
            info_index = pd.Index(image_info[image_id].astype(str))
            image_info = image_info.drop(columns=[image_id])
        else:
            info_index = image_info.index.astype(str)
    path = Path(os.fspath(path))
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    n = len(obs)
    w = _Writer(zarr, chunk_rows)
    root = zarr.open_group(str(path), mode="w", zarr_format=3, attributes={"spatialdata_attrs": {"version": "0.2"}})
    tables = w.group(root, "tables")
    regions = _regions(obs[image_id], n, w.ch)
    sd_attrs = {"region": regions, "region_key": image_id, "instance_key": "instance_id"}

    def spatial(a, b):
        out = np.empty((b - a, 2), dtype=xy_type)
        out[:, 0] = x[a:b]
        out[:, 1] = y[a:b]
        return out

    # an AnnData object's obsm "spatial", when the coordinates come from it, is written whole by anndata below
    from_obs = adata is None or all(c in obs.columns for c in spatial_coords)
    w.anndata(
        tables,
        table_key,
        obs,
        obs_cols,
        n,
        _index_strings(index),
        instance=True,
        spatial=spatial if from_obs else None,
        spatial_dtype=xy_type,
        sd_attrs=sd_attrs,
        attrs={"spatialdata-encoding-type": "ngff:regions_table", "version": "0.2", **sd_attrs},
    )
    if adata is not None:
        _write_anndata_parts(tables[table_key], adata, w.ch, spatial_written=from_obs)
    if points:
        _write_points(root, obs[image_id], regions, x, y, n, w.ch)
    if image_info is not None:
        w.anndata(
            tables,
            info_key,
            image_info,
            _names(list(image_info.columns), "image_info"),
            len(image_info),
            _index_strings(info_index),
            attrs={"spatialdata-encoding-type": "ngff:regions_table", "version": "0.2"},
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata", category=UserWarning)
        # the points' parquet files are part of a SpatialData store, not of its Zarr hierarchy
        warnings.filterwarnings("ignore", message="Object at .*points.parquet is not recognized")
        zarr.consolidate_metadata(str(path))
    return path
