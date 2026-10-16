"""write_cell_store() writes a SpatialData store that spicy(), anndata and spatialdata read back as the input."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr

zarr = pytest.importorskip("zarr", minversion="3")
ad = pytest.importorskip("anndata")

CELLS = pd.read_csv(Path(__file__).parent / "shared_cases" / "cells.csv")
ARGS = {"condition": "condition", "subject": "patient", "r": 30, "covariates": ["age"]}
PER_IMAGE = ["condition", "patient", "time", "event", "age"]


def same(a, b):
    pd.testing.assert_frame_equal(a.cell_results, b.cell_results, check_exact=True)
    assert a.image_ids == b.image_ids
    for k in a.pairwise_assoc:
        np.testing.assert_array_equal(a.pairwise_assoc[k], b.pairwise_assoc[k])


def same_obs(read, obs):
    """Equal frames; strings compared as Python objects (anndata reads them as object or as pandas strings)."""

    def norm(d):
        return d.apply(lambda c: c.astype(object) if pd.api.types.is_string_dtype(c.dtype) else c)

    pd.testing.assert_frame_equal(norm(read), norm(obs), check_exact=True)


def read_table(path, key="table"):
    return ad.io.read_elem(zarr.open_group(str(path), mode="r")["tables"][key])


@pytest.mark.parametrize("chunk_rows", [700, 2**20])  # 700: several chunks, the last one partial
def test_spicy_reads_the_store(tmp_path, chunk_rows):
    ref = spicyr.spicy(CELLS, **ARGS)
    path = spicyr.write_cell_store(CELLS, tmp_path / "cells.zarr", chunk_rows=chunk_rows)
    same(spicyr.spicy(path, **ARGS), ref)
    # the condition and patient from the per-image table of the store
    info = CELLS.groupby("imageID")[PER_IMAGE].first()
    lean = CELLS.drop(columns=PER_IMAGE).assign(age=CELLS["age"])
    path = spicyr.write_cell_store(lean, tmp_path / "lean.zarr", image_info=info, chunk_rows=chunk_rows)
    same(spicyr.spicy(path, image_info="images", **ARGS), ref)
    surv = {"survival": ("time", "event"), "subject": "patient", "r": 30, "from_": "tumour", "covariates": "age"}
    same(spicyr.spicy(path, image_info="images", **surv), spicyr.spicy(CELLS, **surv))


def test_layout_and_round_trip(tmp_path):
    n = len(CELLS)
    cells = CELLS.assign(
        imageID=pd.Categorical(CELLS["imageID"]),
        cellType=pd.Categorical(CELLS["cellType"]),
        flag=np.arange(n) % 3 == 0,
        count=pd.array(np.where(np.arange(n) % 5 == 0, None, np.arange(n) % 9), dtype="Int64"),
        ok=pd.array(np.where(np.arange(n) % 7 == 0, None, np.arange(n) % 2 == 0), dtype="boolean"),
        note=pd.array(np.where(np.arange(n) % 11 == 0, None, "n"), dtype="string"),
    )
    info = cells.groupby("imageID", observed=True)[["patient", "condition"]].first()
    path = spicyr.write_cell_store(cells, tmp_path / "cells.zarr", image_info=info, chunk_rows=700)
    t = read_table(path)
    obs = cells.drop(columns=["x", "y"]).set_axis(cells.index.astype(str))
    same_obs(t.obs.drop(columns="instance_id"), obs)
    np.testing.assert_array_equal(t.obs["instance_id"], np.arange(n))
    np.testing.assert_array_equal(t.obsm["spatial"], cells[["x", "y"]].to_numpy())
    assert t.n_vars == 0
    sa = t.uns["spatialdata_attrs"]
    assert sa["region_key"] == "imageID" and sa["instance_key"] == "instance_id"
    assert list(sa["region"]) == list(dict.fromkeys(CELLS["imageID"]))
    same_obs(read_table(path, "images").obs, info.set_axis(info.index.astype(str).rename(None)))
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root["tables/table"].attrs)
    assert attrs["spatialdata-encoding-type"] == "ngff:regions_table" and attrs["region_key"] == "imageID"
    assert dict(root["tables/images"].attrs)["spatialdata-encoding-type"] == "ngff:regions_table"
    for a in ("obs/_index", "obs/cellType/codes", "obs/count/values", "obs/note/mask", "obs/instance_id"):
        arr = root[f"tables/table/{a}"]
        assert arr.shape == (n,) and arr.chunks == (700,)
    assert root["tables/table/obsm/spatial"].chunks == (700, 2)
    assert root.metadata.consolidated_metadata is not None


def test_anndata_input(tmp_path):
    obs = CELLS.drop(columns=["x", "y"]).copy()
    obs.index = [f"c{i}" for i in range(len(obs))]
    a = ad.AnnData(obs=obs, obsm={"spatial": CELLS[["x", "y"]].to_numpy(np.float32)})
    path = spicyr.write_cell_store(a, tmp_path / "cells.zarr", chunk_rows=1000)
    t = read_table(path)
    assert list(t.obs_names) == list(obs.index)
    assert t.obsm["spatial"].dtype == np.float32
    same(spicyr.spicy(path, **ARGS), spicyr.spicy(a, **ARGS))


def test_spatialdata_reads_the_store(tmp_path):
    sd = pytest.importorskip("spatialdata")
    cells = CELLS.assign(cellType=pd.Categorical(CELLS["cellType"]))  # image IDs as strings
    info = cells.groupby("imageID")[["patient"]].first()
    path = spicyr.write_cell_store(cells, tmp_path / "cells.zarr", image_info=info, chunk_rows=700)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the regions are not elements of the store
        sdata = sd.read_zarr(path)
    t = sdata.tables["table"]
    # the region key is categorical, with the images in order of first appearance
    images = pd.Categorical(cells["imageID"], categories=list(dict.fromkeys(cells["imageID"])))
    obs = cells.drop(columns=["x", "y"]).assign(imageID=images).set_axis(cells.index.astype(str))
    same_obs(t.obs.drop(columns="instance_id"), obs)
    np.testing.assert_array_equal(t.obsm["spatial"], cells[["x", "y"]].to_numpy())
    same(spicyr.spicy(sdata, **ARGS), spicyr.spicy(cells, **ARGS))


def test_errors(tmp_path):
    with pytest.raises(ValueError, match="columns not found"):
        spicyr.write_cell_store(CELLS.drop(columns="y"), tmp_path / "a.zarr")
    with pytest.raises(ValueError, match="'nope' not found"):
        spicyr.write_cell_store(CELLS, tmp_path / "a.zarr", image_id="nope")
    with pytest.raises(ValueError, match="not strings"):
        spicyr.write_cell_store(CELLS.assign(bad=[{}] * len(CELLS)), tmp_path / "a.zarr")
    with pytest.raises(TypeError):
        spicyr.write_cell_store(CELLS.to_numpy(), tmp_path / "a.zarr")
