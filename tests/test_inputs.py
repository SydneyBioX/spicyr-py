"""spicy() gives the same results from a DataFrame, an AnnData object and a SpatialData object."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr

CELLS = pd.read_csv(Path(__file__).parent / "shared_cases" / "cells.csv")
ARGS = {"condition": "condition", "subject": "patient", "r": 30, "from_": "tumour", "to": ["T", "B"]}


def ref():
    return spicyr.spicy(CELLS, **ARGS).cell_results


def test_anndata_obsm():
    ad = pytest.importorskip("anndata")
    obs = CELLS.drop(columns=["x", "y"]).copy()
    obs.index = [f"c{i}" for i in range(len(obs))]
    a = ad.AnnData(obs=obs, obsm={"spatial": CELLS[["x", "y"]].to_numpy()})
    got = spicyr.spicy(a, **ARGS).cell_results
    pd.testing.assert_frame_equal(got, ref())


def test_spatialdata_table_and_centroids():
    sd = pytest.importorskip("spatialdata")
    ad = pytest.importorskip("anndata")
    import geopandas as gpd
    from shapely.geometry import Point
    from spatialdata.models import ShapesModel, TableModel

    obs = CELLS.drop(columns=["x", "y"]).copy()
    obs["region"] = obs["imageID"].astype("category")
    obs["instance_id"] = obs.groupby("imageID").cumcount()
    obs.index = [f"c{i}" for i in range(len(obs))]
    shapes = {}
    for img, z in CELLS.assign(instance_id=obs["instance_id"].to_numpy()).groupby("imageID"):
        g = gpd.GeoDataFrame(
            {"radius": np.full(len(z), 2.0)},
            geometry=[Point(x, y) for x, y in zip(z["x"], z["y"], strict=True)],
            index=z["instance_id"].to_numpy(),
        )
        shapes[img] = ShapesModel.parse(g)
    table = TableModel.parse(ad.AnnData(obs=obs), region=list(shapes), region_key="region", instance_key="instance_id")
    sdata = sd.SpatialData(shapes=shapes, tables={"table": table})
    # images from the table's region_key; coordinates from the shapes' centroids
    got = spicyr.spicy(sdata, image_id="region", **ARGS).cell_results
    pd.testing.assert_frame_equal(got, ref(), check_exact=False, rtol=1e-12)


def test_from_keyword_alias():
    a = spicyr.spicy(CELLS, condition="condition", subject="patient", r=30, to="T", **{"from": "tumour"}).cell_results
    b = spicyr.spicy(CELLS, condition="condition", subject="patient", r=30, from_="tumour", to="T").cell_results
    pd.testing.assert_frame_equal(a, b)


def test_accessors():
    res = spicyr.spicy(CELLS, condition="condition", subject="patient", r=30)
    tp = res.top_pairs(n=5)
    assert len(tp) == 5 and {"from", "to", "coefficient", "p.value"} <= set(tp.columns)
    assert res.bind().shape == (CELLS["imageID"].nunique(), 3 + len(res.cell_results))
    assert "tumour__T" in res.cell_results.index
    assert res.cell_results.loc["tumour__T", "excess_difference"] > 0  # T cells were moved next to tumour cells
    pytest.importorskip("matplotlib")
    res.box_plot("tumour", "T")
    res.signif_plot()
    spicyr.plot_image(CELLS, CELLS["imageID"].iloc[0], "tumour", "T", r=30)
    w = res.image_weights["tumour__T"]
    sums = pd.Series(w).groupby(np.array(res.condition)).sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=1e-10)
    pytest.importorskip("plotly")
    fig = res.box_plot("tumour", "T", interactive=True)
    assert any(tr.type == "scatter" and "image:" in tr.text[0] for tr in fig.data)


def test_numeric_levels_are_named_as_in_r():
    # a float column made categorical: levels "1", "2", as R's factor(), not "1.0", "2.0"
    cells = CELLS.copy()
    pat = sorted(cells["patient"].unique())
    cells["grade"] = pd.Categorical(cells["patient"].map({p: float(1 + i % 3) for i, p in enumerate(pat)}))
    res = spicyr.spicy(
        cells, condition="condition", subject="patient", r=30, from_="tumour", to="T", covariates="grade"
    )
    assert {"grade2_effect", "grade3_p_value"} <= set(res.cell_results.columns)
