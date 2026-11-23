"""The direction of a pair: from_ -> to asks whether `to` cells are placed near `from_` cells (the `to` cells are
relabelled, and the excess is extra `from_` cells per `to` cell), as in spicyR >= 1.99.2."""

import numpy as np
import pytest

import spicyr


def test_to_is_the_relabelled_centre_type():
    # in condition B about a third of the T cells were moved next to tumour cells: T cells are placed near tumour
    # cells, so tumour -> T has more tumour cells around each T cell in B
    cells = spicyr.datasets.example_cells()
    res = spicyr.spicy(cells, condition="condition", subject="patient", r=15, from_="tumour", to="T")
    row = res.cell_results.loc["tumour__T"]
    assert row["excess_difference"] > 0
    assert row["p_value"] < 0.01


def test_allocation_is_the_extra_fraction_under_its_earlier_name():
    import pandas as pd

    cells = spicyr.datasets.example_cells()
    kw = {"condition": "condition", "subject": "patient", "r": 15, "from_": "tumour", "to": "T"}
    a = spicyr.spicy(cells, **kw)
    b = spicyr.spicy(cells, effect="fraction", **kw)
    d = spicyr.spicy(cells, effect="allocation", **kw)
    assert a.effect == b.effect == d.effect == "fraction"
    pd.testing.assert_frame_equal(a.cell_results, b.cell_results)
    pd.testing.assert_frame_equal(d.cell_results, b.cell_results)
    with pytest.raises(ValueError, match="effect must be"):
        spicyr.spicy(cells, effect="share", **kw)
    # a result that records the earlier name is plotted as the extra fraction
    import matplotlib

    matplotlib.use("Agg")
    d.effect = "allocation"
    assert d.box_plot("tumour", "T").get_ylabel().startswith("Extra fraction of T")
    # internal helpers called with the earlier name build the fraction graph and rows, not the count ones
    from spicyr import _cell

    ctx = _cell.cell_context(cells, "condition", "patient")
    ga = _cell.cell_graph(ctx, [("tumour", "T")], r=15, label_clustering=False, effect="allocation")
    gf = _cell.cell_graph(ctx, [("tumour", "T")], r=15, label_clustering=False, effect="fraction")
    assert ga.any is not None and ga.totals is None
    ra = _cell.cell_rows(ctx, ga, "tumour", "T")
    rf = _cell.cell_rows(ctx, gf, "tumour", "T")
    assert ra.keys() == rf.keys()
    for key in rf:
        np.testing.assert_array_equal(np.asarray(ra[key], dtype=object), np.asarray(rf[key], dtype=object))


def test_plot_image_circles_are_around_to_cells():
    import matplotlib

    matplotlib.use("Agg")
    cells = spicyr.datasets.example_cells()
    image = cells["imageID"].iloc[0]
    ax = spicyr.plot_image(cells, image, "tumour", "T", r=10)
    n_to = int(np.sum((cells["imageID"] == image) & (cells["cellType"] == "T")))
    circles = [c for c in ax.collections if type(c).__name__ == "PatchCollection"]
    assert len(circles) == 1 and len(circles[0].get_paths()) == n_to
