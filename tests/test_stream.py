"""The neighbour sums taken image by image (batches of whole images, cells in input order) equal those of the
whole-data Dataset (every cell, sorted by image)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr
from spicyr import _cell

CELLS = pd.read_csv(Path(__file__).parent / "shared_cases" / "cells.csv")


def variants():
    rng = np.random.default_rng(7)
    shuffled = CELLS.iloc[rng.permutation(len(CELLS))].reset_index(drop=True)
    cat = shuffled.assign(
        imageID=pd.Categorical(shuffled["imageID"], categories=["unused", *sorted(CELLS["imageID"].unique())]),
        cellType=pd.Categorical(shuffled["cellType"], categories=["none", *sorted(CELLS["cellType"].unique())]),
    )
    missing = shuffled.copy()
    missing.loc[rng.choice(len(missing), 50, replace=False), "cellType"] = np.nan
    ints = CELLS.assign(imageID=pd.factorize(CELLS["imageID"])[0] + 1)
    f32 = shuffled.assign(x=shuffled["x"].astype(np.float32), y=shuffled["y"].astype(np.float32))
    return {
        "sorted": CELLS,
        "shuffled": shuffled,
        "categorical": cat,
        "missing_type": missing,
        "int_ids": ints,
        "float32": f32,
    }


@pytest.mark.parametrize("name", list(variants()))
@pytest.mark.parametrize("batch", [None, 4000])
def test_stream_equals_dataset(name, batch):
    cells = variants()[name]
    ctx = _cell.cell_context(cells, "condition", "patient", dataset=True)
    d = ctx.data
    for k, radii in ((None, [25.0]), (None, [5.0, 12.5, 30.0]), (6, None)):
        for allocation in (True, False):
            got = _cell.cell_stream(ctx, radii=radii, k=k, allocation=allocation, n_threads=3, batch_cells=batch)
            knn = k is not None
            if radii is not None and len(radii) > 1:
                exp = d.multi_radius_totals(np.asarray(radii), allocation)
            else:
                d.build_knn(k) if knn else d.build_radius_index(radii[0])
                exp = [
                    (d.pair_neighbour_any_totals(knn), d.self_any_expected(knn))
                    if allocation
                    else (d.pair_neighbour_totals(knn), d.pair_neighbour_out_sq_totals(knn))
                ]
            for (a, b), (ea, eb) in zip(got, exp, strict=True):
                np.testing.assert_array_equal(a, ea)
                np.testing.assert_array_equal(b, eb)


def test_layout_matches_sorted_counts():
    cells = variants()["missing_type"]
    ctx = _cell.cell_context(cells, "condition", "patient")
    assert ctx.image_labels == sorted(cells["imageID"].unique())
    tab = pd.crosstab(cells["imageID"], cells["cellType"].fillna("<NA>"))
    labels = ["<NA>" if isinstance(t, float) else t for t in ctx.type_labels]
    np.testing.assert_array_equal(ctx.counts, tab.loc[ctx.image_labels, labels].to_numpy(float))
    assert ctx.image_rows is None and _cell.cell_context(CELLS, "condition", "patient").image_rows is not None


@pytest.mark.parametrize("name", ["shuffled", "categorical", "int_ids", "float32"])
def test_spicy_batches(name, monkeypatch):
    """spicy() gives the same table whatever the batches and threads."""
    cells = variants()[name]
    args = {"condition": "condition", "subject": "patient", "r": [10, 30], "from_": "tumour"}
    ref = spicyr.spicy(cells, **args)
    monkeypatch.setattr(_cell, "BATCH_CELLS", 2500)
    for cores in (1, 4):
        pd.testing.assert_frame_equal(
            spicyr.spicy(cells, cores=cores, **args).cell_results, ref.cell_results, check_exact=True
        )
