"""Inputs: a pandas DataFrame, an AnnData object or a SpatialData object, turned into one cell table.

The table has the standard columns imageID, cellType, x and y (as spicyR's ``.format_data()``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _from_anndata(adata, image_id, cell_type, spatial_coords, spatial_key="spatial"):
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    if spatial_coords[0] in obs.columns and spatial_coords[1] in obs.columns:
        xy = obs[list(spatial_coords)].to_numpy(float)
    elif spatial_key in adata.obsm:
        xy = np.asarray(adata.obsm[spatial_key])[:, :2].astype(float)
    else:
        raise ValueError(f"no coordinates: neither obs columns {spatial_coords} nor obsm['{spatial_key}'] found.")
    obs = obs.drop(columns=[c for c in spatial_coords if c in obs.columns])
    obs[spatial_coords[0]] = xy[:, 0]
    obs[spatial_coords[1]] = xy[:, 1]
    return obs


def _from_spatialdata(sdata, image_id, cell_type, spatial_coords, table_key="table", coordinate_system=None):
    """The annotation table of a SpatialData object.

    Images are the table's regions (its region_key column) unless `image_id` names another obs column;
    coordinates come from obsm["spatial"] or, failing that, from the centroids of the annotated shapes or labels.
    """
    table = sdata.tables[table_key] if hasattr(sdata, "tables") else sdata.table
    attrs = table.uns.get("spatialdata_attrs", {})
    obs = table.obs.copy()
    obs.index = obs.index.astype(str)
    if image_id not in obs.columns:
        rk = attrs.get("region_key")
        if rk is None or rk not in obs.columns:
            raise ValueError(f"`image_id` ('{image_id}') is not a table column and the table has no region_key.")
        obs[image_id] = obs[rk].astype(str)
    if "spatial" in table.obsm:
        xy = np.asarray(table.obsm["spatial"])[:, :2].astype(float)
    else:
        import spatialdata as sd

        rk, ik = attrs.get("region_key"), attrs.get("instance_key")
        xy = np.full((obs.shape[0], 2), np.nan)
        for region in obs[rk].astype(str).unique():
            element = sdata[region]
            cs = coordinate_system or next(iter(sd.transformations.get_transformation(element, get_all=True)))
            cen = sd.get_centroids(element, coordinate_system=cs).compute()
            k = (obs[rk].astype(str) == region).to_numpy()
            pos = cen.loc[obs.loc[k, ik].to_numpy()]
            xy[k, 0] = pos["x"].to_numpy()
            xy[k, 1] = pos["y"].to_numpy()
    obs[spatial_coords[0]] = xy[:, 0]
    obs[spatial_coords[1]] = xy[:, 1]
    return obs


def format_data(cells, image_id="imageID", cell_type="cellType", spatial_coords=("x", "y"), table_key="table"):
    """One row per cell with columns imageID, cellType, x, y, and every other column kept.

    Rows are ordered by image in order of first appearance, as spicyR's ``.format_data()``.
    """
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
