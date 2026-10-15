"""spicy() reads a SpatialData (or AnnData) zarr store lazily, image by image, with the results of the in-memory path."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr
from spicyr import _cell

sd = pytest.importorskip("spatialdata")
ad = pytest.importorskip("anndata")

CELLS = pd.read_csv(Path(__file__).parent / "shared_cases" / "cells.csv")
ARGS = {"condition": "condition", "subject": "patient", "r": 30, "from_": ["tumour", "T"], "to": ["T", "B", "macro"]}


def write_store(path, cells, coords="obsm", dtype=np.float64, per_image=False):
    """A SpatialData store of `cells`: a table annotating one region per image, and a per-image table."""
    from spatialdata.models import TableModel

    obs = cells.drop(columns=["x", "y"]).copy()
    obs["region"] = pd.Categorical(obs["imageID"])
    obs["cellType"] = pd.Categorical(obs["cellType"])
    obs["instance_id"] = np.arange(len(obs))
    obs.index = [f"c{i}" for i in range(len(obs))]
    img = cells.groupby("imageID", sort=True)[["condition", "patient"]].first()
    if per_image:
        obs = obs.drop(columns=["condition", "patient"])
    xy = cells[["x", "y"]].to_numpy().astype(dtype)
    if coords == "obsm":
        a = ad.AnnData(obs=obs, obsm={"spatial": xy})
    else:
        obs["x"], obs["y"] = xy[:, 0], xy[:, 1]
        a = ad.AnnData(obs=obs)
    regions = list(obs["region"].cat.categories)
    table = TableModel.parse(a, region=regions, region_key="region", instance_key="instance_id")
    sdata = sd.SpatialData(tables={"table": table, "images": TableModel.parse(ad.AnnData(obs=img))})
    with pytest.warns(UserWarning):  # the regions are not elements of the object
        sdata.write(path)
    return path


def same(a, b):
    pd.testing.assert_frame_equal(a.cell_results, b.cell_results, check_exact=True)
    assert a.image_ids == b.image_ids and a.condition == b.condition and a.subject == b.subject
    for k in a.pairwise_assoc:
        np.testing.assert_array_equal(a.pairwise_assoc[k], b.pairwise_assoc[k])


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    d = tmp_path_factory.mktemp("stores")
    shuffled = CELLS.iloc[np.random.default_rng(1).permutation(len(CELLS))].reset_index(drop=True)
    return {
        "grouped": write_store(d / "grouped.zarr", CELLS),
        "shuffled": write_store(d / "shuffled.zarr", shuffled),
        "obs_xy": write_store(d / "obs_xy.zarr", CELLS, coords="obs"),
        "float32": write_store(d / "float32.zarr", CELLS, dtype=np.float32),
        "per_image": write_store(d / "per_image.zarr", CELLS, per_image=True),
        "shuffled_cells": shuffled,
    }


@pytest.mark.parametrize("kind", ["grouped", "shuffled", "obs_xy"])
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"effect": "count"},
        {"k": 8},
        {"r": [10, 20, 40]},
        {"r": [10, 20, 40], "combine": "davies"},
        {"covariates": ["age", "batch"]},
    ],
)
def test_store_equals_memory(stores, kind, extra):
    args = {**ARGS, **extra}
    cells = stores["shuffled_cells"] if kind == "shuffled" else CELLS
    same(spicyr.spicy(stores[kind], image_id="region", **args), spicyr.spicy(cells, **args))


def test_store_float32_and_spatialdata_object(stores):
    c32 = CELLS.assign(x=CELLS["x"].astype(np.float32), y=CELLS["y"].astype(np.float32))
    ref = spicyr.spicy(c32, **ARGS)
    same(spicyr.spicy(stores["float32"], image_id="region", **ARGS), ref)
    same(spicyr.spicy(str(stores["float32"]), **ARGS), ref)  # images from the region_key by default
    same(spicyr.spicy(sd.read_zarr(stores["float32"]), image_id="region", **ARGS), ref)


def test_survival_from_store(stores):
    args = {"survival": ("time", "event"), "subject": "patient", "r": 30, "from_": "tumour", "covariates": "age"}
    same(spicyr.spicy(stores["grouped"], **args), spicyr.spicy(CELLS, **args))


def test_per_image_table(stores):
    """Condition and subject from a table with one row per image (a table of the store, or a DataFrame)."""
    ref = spicyr.spicy(CELLS, **ARGS)
    same(spicyr.spicy(stores["per_image"], images="images", **ARGS), ref)
    img = CELLS.groupby("imageID")[["condition", "patient"]].first()
    same(spicyr.spicy(CELLS.drop(columns=["condition", "patient"]), images=img, **ARGS), ref)
    with pytest.raises(ValueError, match="missing from `images`"):
        spicyr.spicy(CELLS, images=img.iloc[1:], **ARGS)


def test_batches_and_threads(stores, monkeypatch):
    """Batches of a few images, read and summarised on several threads, give the same numbers."""
    ref = spicyr.spicy(CELLS, **ARGS)
    monkeypatch.setattr(_cell, "BATCH_CELLS", 3000)
    for kind, cells in (("grouped", CELLS), ("shuffled", stores["shuffled_cells"])):
        for cores in (1, 3):
            same(spicyr.spicy(stores[kind], image_id="region", cores=cores, **ARGS), ref)
            same(spicyr.spicy(cells, cores=cores, **ARGS), ref)
    knn = {**ARGS, "k": 8}
    same(spicyr.spicy(stores["shuffled"], image_id="region", cores=3, **knn), spicyr.spicy(CELLS, **knn))


def test_condition_must_be_constant(tmp_path):
    bad = CELLS.copy()
    bad.loc[bad.index[0], "condition"] = "B" if bad["condition"].iloc[0] == "A" else "A"
    path = write_store(tmp_path / "bad.zarr", bad)
    with pytest.raises(ValueError, match="constant within each image"):
        spicyr.spicy(path, image_id="region", **ARGS)


def test_anndata_zarr(tmp_path):
    obs = CELLS.drop(columns=["x", "y"]).copy()
    obs.index = [f"c{i}" for i in range(len(obs))]
    a = ad.AnnData(obs=obs, obsm={"spatial": CELLS[["x", "y"]].to_numpy()})
    a.write_zarr(tmp_path / "cells.zarr")
    same(spicyr.spicy(tmp_path / "cells.zarr", **ARGS), spicyr.spicy(CELLS, **ARGS))


def test_other_encodings(tmp_path):
    """Columns in other anndata encodings (nullable integers, plain strings) are read whole."""
    cells = CELLS.assign(
        patient_n=pd.array(pd.factorize(CELLS["patient"])[0], dtype="Int64"), cellType=CELLS["cellType"].astype(object)
    )
    obs = cells.drop(columns=["x", "y"]).copy()
    obs.index = [f"c{i}" for i in range(len(obs))]
    a = ad.AnnData(obs=obs, obsm={"spatial": cells[["x", "y"]].to_numpy()})
    with ad.settings.override(allow_write_nullable_strings=True):
        a.write_zarr(tmp_path / "cells.zarr")
    args = {**ARGS, "subject": "patient_n"}
    same(spicyr.spicy(tmp_path / "cells.zarr", **args), spicyr.spicy(cells, **args))
