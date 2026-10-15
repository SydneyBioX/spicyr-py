"""The cells of spicy()'s cell method as columns: image and cell-type codes, per-image values and coordinates.

The cell method reads the cells image by image (``_cell.cell_stream()``), so it needs the whole table only as two
columns of integer codes (image and cell type). The coordinates are read a batch of whole images at a time, and the
condition, subject and covariates once per image. A source keeps the cells where they are: the columns of a
DataFrame or AnnData object in memory, or the arrays of an AnnData table in a zarr store (a SpatialData store or an
AnnData ``.zarr``), which are never loaded whole.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import _core


def as_str(col: pd.Series) -> pd.Series:
    """A column as strings, the way R's as.character() writes it (2.0 is "2"), so that labels match spicyR's."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col.cat.rename_categories([_r_str_value(c) for c in col.cat.categories]).astype(str).where(col.notna())
    if pd.api.types.is_float_dtype(col):
        return col.map(_r_str_value).where(col.notna())
    return col.astype(str).where(col.notna())


def _r_str_value(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "NaN" if np.isnan(v) else f"{v:.15g}"
    return str(v)


def _small_codes(codes: np.ndarray) -> np.ndarray:
    """Codes in the narrowest signed integer type that holds them (-1 is missing)."""
    hi = int(codes.max()) if len(codes) else 0
    for t in (np.int8, np.int16, np.int32):
        if hi <= np.iinfo(t).max:
            return np.ascontiguousarray(codes, dtype=t)
    return np.ascontiguousarray(codes, dtype=np.int64)


def _canonical(codes: np.ndarray, labels: list):
    """Merge codes whose labels are equal strings, so that every label has one code."""
    uniq = list(dict.fromkeys(labels))
    if len(uniq) == len(labels):
        return codes, labels
    pos = {s: i for i, s in enumerate(uniq)}
    remap = np.array([pos[s] for s in labels] + [-1])  # -1 stays -1
    return _small_codes(remap[codes]), uniq


def label_codes(values) -> tuple[np.ndarray, list]:
    """Integer codes (-1 missing) and the label (as R's as.character()) of every code, for a column of IDs.

    The labels are those of ``as_str()`` applied to the column's values (as numpy gives them), so a categorical
    column, whose codes are used directly, is labelled as its values would be.
    """
    if isinstance(values, pd.Categorical | pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
        cat = pd.Categorical(values)
        codes, uniques = np.asarray(cat.codes), cat.categories.to_numpy()
    else:
        codes, uniques = pd.factorize(np.asarray(values), use_na_sentinel=True)
    labels = list(as_str(pd.Series(uniques, dtype=uniques.dtype if uniques.dtype.kind != "U" else object)))
    return _canonical(_small_codes(codes), labels)


class FrameSource:
    """Cells in memory: the columns of a DataFrame (or an AnnData object's obs) and two coordinate arrays.

    Nothing is copied or sorted: the image and cell-type columns become integer codes, and a batch's coordinates are
    taken from the arrays as it is read.
    """

    def __init__(self, obs: pd.DataFrame, image_col: str, type_col: str, x, y, aliases=None):
        self.obs = obs
        self.n = len(obs)
        self.image_codes, self.image_labels = label_codes(obs[image_col])
        self.type_codes, self.type_labels = label_codes(obs[type_col])
        self._x, self._y = (_float_column(v) for v in (x, y))
        self.aliases = dict(aliases or {})

    def _col(self, name):
        return self.aliases.get(name, name)

    def has(self, name) -> bool:
        return self._col(name) in self.obs.columns

    def coords(self, rows):
        """The x and y of the cells at `rows` (a slice or row numbers), float32 or float64, contiguous."""
        return np.ascontiguousarray(self._x[rows]), np.ascontiguousarray(self._y[rows])

    def at_rows(self, name, rows) -> pd.Series:
        return self.obs[self._col(name)].iloc[rows].reset_index(drop=True)

    def column(self, name) -> pd.Series:
        return self.obs[self._col(name)]

    def constant_within(self, name, first_row_by_code) -> bool:
        """Whether the column takes its image's first value in every cell (the same value; False means "unknown")."""
        col = self.obs[self._col(name)]
        first = np.asarray(first_row_by_code, dtype=np.int64)
        if isinstance(col.dtype, pd.CategoricalDtype):
            v = np.ascontiguousarray(col.cat.codes.to_numpy())
        else:
            v = col.to_numpy()
            if v.dtype.kind not in "biuf":
                return bool(np.all(v == v[first[self.image_codes]]))
            v = np.ascontiguousarray(v)
        return bool(_core.constant_within(v, self.image_codes, np.maximum(first, 0)))


def _float_column(v):
    """A coordinate column as float32 or float64 (as stored when it is one of those, else converted once)."""
    a = v.to_numpy() if isinstance(v, pd.Series) else np.asarray(v)
    if a.dtype not in (np.float32, np.float64) or not a.dtype.isnative:
        a = a.astype(float)
    return a


class ZarrSource:
    """The cells of an AnnData table in a zarr store (a SpatialData store, or an AnnData ``.zarr``), read lazily.

    The image and cell-type columns are read once as integer codes (categorical columns use their stored codes);
    coordinates (``obsm["spatial"]``, or two obs columns) are read for a batch of rows at a time, and other columns
    once per image (with a pass over the column, chunk by chunk, to check that it is constant within images).
    """

    def __init__(self, store, table_key="table", image_id="imageID", cell_type="cellType", spatial_coords=("x", "y")):
        import zarr

        # the arrays' own metadata (consolidated metadata can be stale after a store is edited in place)
        root = zarr.open_group(os.fspath(store), mode="r", use_consolidated=False)
        if "tables" in root:
            if table_key not in root["tables"]:
                raise ValueError(f"no table '{table_key}' in {store}; tables are: {list(root['tables'].keys())}")
            table = root["tables"][table_key]
        elif "obs" in root:
            table = root
        else:
            raise ValueError(f"{store} is neither a SpatialData store nor an AnnData zarr store.")
        self.path = Path(os.fspath(store))
        self.root = root
        self.table = table
        self._obs = table["obs"]
        attrs = dict(table.attrs)
        rk = attrs.get("region_key")
        if rk is None and "uns" in table and "spatialdata_attrs" in table["uns"]:
            sa = table["uns"]["spatialdata_attrs"]
            rk = _scalar(sa["region_key"]) if "region_key" in sa else None
        self.region_key = rk
        self.aliases = {}
        self._cache = {}
        if image_id not in self._obs:
            if rk is None or rk not in self._obs:
                raise ValueError(f"`image_id` ('{image_id}') is not a table column and the table has no region_key.")
            self.aliases[image_id] = rk
        if cell_type not in self._obs:
            raise ValueError(f"column '{cell_type}' not found; columns are: {self.columns}")
        self.image_codes, self.image_labels = self._codes(self._col(image_id))
        self.type_codes, self.type_labels = self._codes(cell_type)
        self.n = len(self.image_codes)
        if "obsm" in table and "spatial" in table["obsm"]:
            self._xy = table["obsm"]["spatial"]
            self._xcol = self._ycol = None
        elif spatial_coords[0] in self._obs and spatial_coords[1] in self._obs:
            self._xy = None
            self._xcol, self._ycol = self._obs[spatial_coords[0]], self._obs[spatial_coords[1]]
        else:
            raise ValueError(f"no coordinates: neither obsm['spatial'] nor obs columns {tuple(spatial_coords)} found.")

    @property
    def columns(self) -> list:
        return sorted(self._obs.keys())

    def _col(self, name):
        return self.aliases.get(name, name)

    def has(self, name) -> bool:
        return self._col(name) in self._obs

    def _is_categorical(self, name) -> bool:
        return self._obs[name].attrs.get("encoding-type") == "categorical"

    def _other(self, name):
        """A column in another anndata encoding (e.g. nullable integers), read whole once; None for arrays."""
        import zarr

        if isinstance(self._obs[name], zarr.Array) or self._is_categorical(name):
            return None
        if name not in self._cache:
            import anndata as ad

            self._cache[name] = pd.Series(ad.io.read_elem(self._obs[name]))
        return self._cache[name]

    def _categories(self, name) -> pd.Categorical:
        g = self._obs[name]
        cats = _strings(g["categories"][...])
        return pd.CategoricalDtype(cats, ordered=bool(g.attrs.get("ordered", False)))

    def _codes(self, name):
        if self._is_categorical(name):
            g = self._obs[name]
            codes = np.ascontiguousarray(g["codes"][...])
            dtype = self._categories(name)
            if codes.dtype.kind != "i" or not codes.dtype.isnative:
                codes = codes.astype(np.int64)
            labels = list(as_str(pd.Series(dtype.categories.to_numpy())))
            return _canonical(_small_codes(codes) if codes.dtype.itemsize > 4 else codes, labels)
        other = self._other(name)
        return label_codes(other if other is not None else _strings(self._obs[name][...]))

    def coords(self, rows):
        """The x and y of the cells at `rows` (a slice or ascending row numbers), as stored (float32 or float64)."""
        if self._xy is not None:
            if isinstance(rows, slice):
                xy = self._xy[rows.start : rows.stop, :2]
            else:
                xy = self._xy.get_orthogonal_selection((rows, slice(0, 2)))
            return _float_column(np.ascontiguousarray(xy[:, 0])), _float_column(np.ascontiguousarray(xy[:, 1]))
        out = []
        for a in (self._xcol, self._ycol):
            v = a[rows.start : rows.stop] if isinstance(rows, slice) else a.get_coordinate_selection(rows)
            out.append(np.ascontiguousarray(_float_column(v)))
        return tuple(out)

    def at_rows(self, name, rows) -> pd.Series:
        name = self._col(name)
        rows = np.asarray(rows, dtype=np.int64)
        if self._other(name) is not None:
            return self._other(name).iloc[rows].reset_index(drop=True)
        if self._is_categorical(name):
            codes = self._obs[name]["codes"].get_coordinate_selection(rows) if len(rows) else np.zeros(0, int)
            return pd.Series(pd.Categorical.from_codes(np.asarray(codes, dtype=np.int64), dtype=self._categories(name)))
        v = self._obs[name].get_coordinate_selection(rows) if len(rows) else np.zeros(0)
        return pd.Series(_strings(v))

    def column(self, name) -> pd.Series:
        """The whole column (used only when a per-image column is not constant within images)."""
        name = self._col(name)
        if self._other(name) is not None:
            return self._other(name)
        if self._is_categorical(name):
            codes = self._obs[name]["codes"][...]
            return pd.Series(pd.Categorical.from_codes(np.asarray(codes, dtype=np.int64), dtype=self._categories(name)))
        return pd.Series(_strings(self._obs[name][...]))

    def constant_within(self, name, first_row_by_code) -> bool:
        """Whether the column takes its image's first value in every cell, read chunk by chunk."""
        name = self._col(name)
        if self._other(name) is not None:
            return False  # the caller then checks the whole column
        arr = self._obs[name]["codes"] if self._is_categorical(name) else self._obs[name]
        first = np.asarray(first_row_by_code, dtype=np.int64)
        ok = first >= 0
        fv = np.zeros(len(first), dtype=object if arr.dtype.kind not in "biuf" else arr.dtype)
        if ok.any():
            fv[ok] = _strings(arr.get_coordinate_selection(first[ok]))
        step = max(int(arr.chunks[0]), 1 << 20)
        for lo in range(0, self.n, step):
            v = _strings(arr[lo : lo + step])
            f = fv[self.image_codes[lo : lo + step]]
            if v.dtype.kind in "biuf":
                if not np.array_equal(v.view(np.uint8), np.ascontiguousarray(f).view(np.uint8)):
                    return False
            elif not bool(np.all(v == f)):
                return False
        return True


def _scalar(a):
    v = a[...]
    v = v.item() if hasattr(v, "item") else v
    return str(v)


def _strings(a):
    """A zarr array's values with strings as Python objects (zarr gives numpy's StringDType)."""
    a = np.asarray(a)
    if a.dtype.kind in "TUS" or a.dtype.kind == "O":
        return a.astype(object)
    return a


def _is_store(cells) -> bool:
    if isinstance(cells, str | os.PathLike):
        return Path(os.fspath(cells)).is_dir()
    return False


def cell_source(cells, image_id="imageID", cell_type="cellType", spatial_coords=("x", "y"), table_key="table"):
    """The cells of any accepted input as a source for the cell method.

    A DataFrame, an AnnData or a SpatialData object (in memory), or the path of a SpatialData store or of an AnnData
    ``.zarr`` store, read lazily.
    """
    from ._input import _anndata_coords, _spatialdata_obs_coords

    if _is_store(cells):
        return ZarrSource(cells, table_key, image_id, cell_type, spatial_coords)
    if isinstance(cells, str | os.PathLike):
        raise ValueError(f"{cells} is not a zarr store (a directory).")
    kind = type(cells).__name__
    aliases = {}
    if kind == "SpatialData":
        obs, x, y, aliases = _spatialdata_obs_coords(cells, image_id, spatial_coords, table_key)
    elif kind == "AnnData":
        obs = cells.obs
        x, y = _anndata_coords(cells, spatial_coords)
    elif isinstance(cells, pd.DataFrame):
        obs = cells
        miss = [c for c in spatial_coords if c not in obs.columns]
        if miss:
            raise ValueError(f"column '{miss[0]}' not found; columns are: {list(obs.columns)}")
        x, y = obs[spatial_coords[0]], obs[spatial_coords[1]]
    else:
        raise TypeError(
            "`cells` must be a pandas DataFrame, an AnnData or a SpatialData object, or the path of a zarr store."
        )
    image_col = aliases.get(image_id, image_id)
    for col in (image_col, cell_type):
        if col not in obs.columns:
            raise ValueError(f"column '{col}' not found; columns are: {list(obs.columns)}")
    return FrameSource(obs, image_col, cell_type, x, y, aliases)
