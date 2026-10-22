"""write_cell_store() writes an AnnData object whole, and the cells of each image as points."""

import warnings

import numpy as np
import pandas as pd
import pytest

import spicyr

zarr = pytest.importorskip("zarr", minversion="3")
ad = pytest.importorskip("anndata")
pytest.importorskip("pyarrow")
sp = pytest.importorskip("scipy.sparse")


def full_adata(n=300, p=7):
    rng = np.random.default_rng(1)
    counts = sp.random(n, p, density=0.3, format="csr", random_state=1, data_rvs=lambda k: rng.poisson(5, k) + 1.0)
    obs = pd.DataFrame(
        {
            "imageID": pd.Categorical(np.repeat([f"i{k}" for k in range(1, 7)], n // 6)),
            "cellType": pd.Categorical(rng.choice(["T", "B", "tumour"], n)),
            "score": pd.array([None if k % 11 == 0 else k for k in range(n)], dtype="Int64"),
        },
        index=[f"cell{k}" for k in range(n)],
    )
    var = pd.DataFrame({"symbol": [f"g{j}" for j in range(p)]}, index=[f"G{j}" for j in range(p)])
    a = ad.AnnData(X=counts, obs=obs, var=var, layers={"logcounts": np.log1p(counts.toarray())})
    a.obsm["spatial"] = rng.uniform(0, 500, (n, 2))
    a.obsm["PCA"] = rng.normal(size=(n, 3))
    a.uns["note"] = "from a test"
    return a


def test_anndata_written_whole(tmp_path):
    a = full_adata()
    path = spicyr.write_cell_store(a, tmp_path / "cells.zarr", chunk_rows=64)  # several chunks
    t = ad.read_zarr(path / "tables" / "table")
    assert (t.X != a.X).nnz == 0
    np.testing.assert_array_equal(t.layers["logcounts"], a.layers["logcounts"])
    assert list(t.var_names) == list(a.var_names) and list(t.var["symbol"]) == list(a.var["symbol"])
    assert list(t.obs_names) == list(a.obs_names)
    np.testing.assert_array_equal(t.obsm["spatial"], a.obsm["spatial"])
    np.testing.assert_array_equal(t.obsm["PCA"], a.obsm["PCA"])
    assert t.uns["note"] == "from a test"
    assert int(t.obs["score"].isna().sum()) == int(a.obs["score"].isna().sum())
    np.testing.assert_array_equal(t.obs["instance_id"].to_numpy(), np.arange(a.n_obs))
    # chunked as asked: a sparse X's data and a dense layer in chunks of rows
    g = zarr.open_group(str(path / "tables" / "table"), mode="r")
    assert g["X/data"].chunks == (64,)
    assert g["layers/logcounts"].chunks[0] < a.n_obs


def test_points_one_file_per_image_read_by_spatialdata(tmp_path):
    sd = pytest.importorskip("spatialdata")
    a = full_adata()
    order = np.random.default_rng(2).permutation(a.n_obs)  # images not grouped: their rows across many chunks
    a = a[order].copy()
    path = spicyr.write_cell_store(a, tmp_path / "cells.zarr", chunk_rows=50)
    for im in a.obs["imageID"].cat.categories:
        assert [f.name for f in (path / "points" / im / "points.parquet").iterdir()] == ["part.0.parquet"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # e.g. a table annotating an element that is not there
        s = sd.read_zarr(path)
    xy = a.obsm["spatial"]
    for im in a.obs["imageID"].cat.categories:
        pts = s.points[im].compute().sort_values("instance_id")
        rows = np.flatnonzero(a.obs["imageID"].to_numpy() == im)
        np.testing.assert_array_equal(pts["instance_id"].to_numpy(), rows)
        np.testing.assert_array_equal(pts["x"].to_numpy(), xy[rows, 0])


def test_dataframe_points_and_options(tmp_path):
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"imageID": rng.choice(["a", "b", "c"], 500), "x": rng.random(500), "y": rng.random(500)})
    path = spicyr.write_cell_store(d, tmp_path / "d.zarr", chunk_rows=40)
    assert sorted(p.name for p in (path / "points").iterdir() if p.is_dir()) == ["a", "b", "c"]
    assert not (spicyr.write_cell_store(d, tmp_path / "np.zarr", points=False) / "points").exists()
    d["imageID"] = d["imageID"].replace("a", "a b")
    with pytest.warns(UserWarning, match="valid SpatialData element names"):
        p2 = spicyr.write_cell_store(d, tmp_path / "bad.zarr")
    assert not (p2 / "points").exists()


def test_spatial_twice_is_an_error(tmp_path):
    a = full_adata()
    a.obs["x"], a.obs["y"] = a.obsm["spatial"][:, 0], a.obsm["spatial"][:, 1]
    with pytest.raises(ValueError, match="obsm has a 'spatial' array"):
        spicyr.write_cell_store(a, tmp_path / "c.zarr")
