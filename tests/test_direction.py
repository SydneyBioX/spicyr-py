"""The direction of a pair: from_ -> to asks whether `to` cells are placed near `from_` cells (the `to` cells are
relabelled, and the excess is extra `from_` cells per `to` cell), as in spicyR >= 1.99.2."""

import numpy as np

import spicyr


def test_to_is_the_relabelled_centre_type():
    # in condition B about a third of the T cells were moved next to tumour cells: T cells are placed near tumour
    # cells, so tumour -> T has more tumour cells around each T cell in B
    cells = spicyr.datasets.example_cells()
    res = spicyr.spicy(cells, condition="condition", subject="patient", r=15, from_="tumour", to="T")
    row = res.cell_results.loc["tumour__T"]
    assert row["excess_difference"] > 0
    assert row["p_value"] < 0.01


def test_plot_image_circles_are_around_to_cells():
    import matplotlib

    matplotlib.use("Agg")
    cells = spicyr.datasets.example_cells()
    image = cells["imageID"].iloc[0]
    ax = spicyr.plot_image(cells, image, "tumour", "T", r=10)
    n_to = int(np.sum((cells["imageID"] == image) & (cells["cellType"] == "T")))
    circles = [c for c in ax.collections if type(c).__name__ == "PatchCollection"]
    assert len(circles) == 1 and len(circles[0].get_paths()) == n_to
