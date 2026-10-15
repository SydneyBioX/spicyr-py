"""Inputs: a pandas DataFrame, an AnnData object or a SpatialData object (or store), turned into one cell table.

The table has the standard columns imageID, cellType, x and y (as spicyR's ``.format_data()``).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _anndata_coords(adata, spatial_coords, spatial_key="spatial"):
    """The x and y of an AnnData object's cells: obs columns `spatial_coords`, else obsm[spatial_key] (not copied)."""
    obs = adata.obs
    if spatial_coords[0] in obs.columns and spatial_coords[1] in obs.columns:
        return obs[spatial_coords[0]], obs[spatial_coords[1]]
    if spatial_key in adata.obsm:
        xy = np.asarray(adata.obsm[spatial_key])
        return xy[:, 0], xy[:, 1]
    raise ValueError(f"no coordinates: neither obs columns {spatial_coords} nor obsm['{spatial_key}'] found.")


def _from_anndata(adata, image_id, cell_type, spatial_coords, spatial_key="spatial"):
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    x, y = _anndata_coords(adata, spatial_coords, spatial_key)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    obs = obs.drop(columns=[c for c in spatial_coords if c in obs.columns])
    obs[spatial_coords[0]] = x
    obs[spatial_coords[1]] = y
    return obs


def _spatialdata_obs_coords(sdata, image_id, spatial_coords, table_key="table", coordinate_system=None):
    """The annotation table's obs (not copied), the cells' x and y, and the image column if `image_id` is not one.

    Images are the table's regions (its region_key column) unless `image_id` names another obs column;
    coordinates come from obsm["spatial"], else from obs columns `spatial_coords`, else from the centroids of the
    annotated shapes or labels.
    """
    table = sdata.tables[table_key] if hasattr(sdata, "tables") else sdata.table
    attrs = table.uns.get("spatialdata_attrs", {})
    obs = table.obs
    aliases = {}
    if image_id not in obs.columns:
        rk = attrs.get("region_key")
        if rk is None or rk not in obs.columns:
            raise ValueError(f"`image_id` ('{image_id}') is not a table column and the table has no region_key.")
        aliases[image_id] = rk
    if "spatial" in table.obsm:
        xy = np.asarray(table.obsm["spatial"])
        return obs, xy[:, 0], xy[:, 1], aliases
    if spatial_coords[0] in obs.columns and spatial_coords[1] in obs.columns:
        return obs, obs[spatial_coords[0]], obs[spatial_coords[1]], aliases
    import spatialdata as sd

    rk, ik = attrs.get("region_key"), attrs.get("instance_key")
    xy = np.full((obs.shape[0], 2), np.nan)
    region = obs[rk].astype(str)
    for reg in region.unique():
        element = sdata[reg]
        cs = coordinate_system or next(iter(sd.transformations.get_transformation(element, get_all=True)))
        cen = sd.get_centroids(element, coordinate_system=cs).compute()
        k = (region == reg).to_numpy()
        pos = cen.loc[obs.loc[k, ik].to_numpy()]
        xy[k, 0] = pos["x"].to_numpy()
        xy[k, 1] = pos["y"].to_numpy()
    return obs, xy[:, 0], xy[:, 1], aliases


def _from_spatialdata(sdata, image_id, cell_type, spatial_coords, table_key="table", coordinate_system=None):
    """The annotation table of a SpatialData object as a cell table (see _spatialdata_obs_coords)."""
    obs, x, y, aliases = _spatialdata_obs_coords(sdata, image_id, spatial_coords, table_key, coordinate_system)
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    if image_id in aliases:
        obs[image_id] = obs[aliases[image_id]].astype(str)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    obs[spatial_coords[0]] = x
    obs[spatial_coords[1]] = y
    return obs


def format_data(cells, image_id="imageID", cell_type="cellType", spatial_coords=("x", "y"), table_key="table"):
    """One row per cell with columns imageID, cellType, x, y, and every other column kept.

    Rows are ordered by image in order of first appearance, as spicyR's ``.format_data()``.
    """
    if isinstance(cells, str | os.PathLike):
        import spatialdata as sd

        cells = sd.read_zarr(os.fspath(cells))
    kind = type(cells).__name__
    if kind == "SpatialData":
        df = _from_spatialdata(cells, image_id, cell_type, spatial_coords, table_key)
    elif kind == "AnnData":
        df = _from_anndata(cells, image_id, cell_type, spatial_coords)
    elif isinstance(cells, pd.DataFrame):
        df = cells.copy()
    else:
        raise TypeError("`cells` must be a pandas DataFrame, an AnnData or a SpatialData object.")
    for col in (image_id, cell_type, *spatial_coords):
        if col not in df.columns:
            raise ValueError(f"column '{col}' not found; columns are: {list(df.columns)}")
    needed = pd.DataFrame(
        {
            "imageID": df[image_id].to_numpy(),
            "cellType": df[cell_type].to_numpy(),
            "x": df[spatial_coords[0]].to_numpy(float),
            "y": df[spatial_coords[1]].to_numpy(float),
        },
        index=df.index,
    )
    rest = df.drop(
        columns=[c for c in ("imageID", "cellType", "x", "y", image_id, cell_type, *spatial_coords) if c in df.columns]
    )
    out = pd.concat([rest, needed], axis=1)
    order = pd.Categorical(out["imageID"].astype(str), categories=list(dict.fromkeys(out["imageID"].astype(str)))).codes
    out = out.iloc[np.argsort(order, kind="stable")].reset_index(drop=True)
    return out
