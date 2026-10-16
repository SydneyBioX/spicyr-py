"""Every pair at every radius in one core call (pair_tests) against each pair's test in turn (cell_pair_test)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr
from spicyr import _api

CELLS = pd.read_csv(Path(__file__).parent / "shared_cases" / "cells.csv")
BASE = {"condition": "condition", "subject": "patient"}


def pair_by_pair(monkeypatch):
    def run(ctx, graphs, pairs, variance, adjust, Z_extra, extra_names, radii, combine, n_threads=1):
        return _api._pair_by_pair(ctx, graphs, pairs, variance, adjust, Z_extra, extra_names, radii, combine)

    monkeypatch.setattr(_api, "_pairs", run)


def in_one_call(monkeypatch):
    def run(ctx, graphs, pairs, variance, adjust, Z_extra, extra_names, radii, combine):
        return _api._pairs(ctx, graphs, pairs, variance, adjust, Z_extra, extra_names, radii, combine, 2)

    monkeypatch.setattr(_api, "_pair_by_pair", run)


def same(a, b):
    pd.testing.assert_frame_equal(a.cell_results, b.cell_results, check_exact=True)
    if a.radius_results is None:
        assert b.radius_results is None
    else:
        pd.testing.assert_frame_equal(a.radius_results, b.radius_results, check_exact=True)
    for d in ("pairwise_assoc", "image_weights"):
        x, y = getattr(a, d), getattr(b, d)
        assert list(x) == list(y)
        for k in x:
            np.testing.assert_array_equal(x[k], y[k])


@pytest.mark.parametrize(
    "args",
    [
        {"r": 30},
        {"r": 30, "effect": "count"},
        {"k": 8},
        {"r": [10, 20, 30, 40], "variance": "hartung_knapp"},
        {"r": [10, 20, 30, 40], "effect": "count"},
        {"r": [5, 10, 15, 20, 25, 30, 35, 40], "combine": "davies"},
        {"r": [10, 20, 30], "combine": "cauchy", "from_": ["T", "B"], "to": "tumour", "cores": 3},
    ],
)
def test_identical_to_each_pair_in_turn(monkeypatch, args):
    args = {**BASE, **args}
    new = spicyr.spicy(CELLS, **args)
    with monkeypatch.context() as m:
        pair_by_pair(m)
        old = spicyr.spicy(CELLS, **args)
    same(new, old)


@pytest.mark.parametrize(
    "args",
    [
        {"r": 30, "covariates": ["age"], "adjust_abundance": True},
        {"r": 30, "adjust_abundance": True, "effect": "count"},
        {"r": [10, 20, 30], "covariates": ["age_na", "batch"]},
        {"r": [10, 20, 30], "combine": "cauchy", "adjust_abundance": True},
    ],
)
def test_adjusted_design_in_the_core(monkeypatch, args):
    # the core centres the design's columns with a long double sum (as R's colMeans), numpy pairwise: the same
    # tables up to rounding
    args = {**BASE, **args}
    old = spicyr.spicy(CELLS, **args)
    with monkeypatch.context() as m:
        in_one_call(m)
        new = spicyr.spicy(CELLS, **args)
    a, b = old.cell_results, new.cell_results
    assert list(a.columns) == list(b.columns) and list(a.index) == list(b.index)
    assert "adjusted_for" in a.columns
    for c in a.columns:
        if a[c].dtype.kind == "f":
            np.testing.assert_allclose(b[c], a[c], rtol=1e-9, atol=1e-12)
        else:
            assert a[c].tolist() == b[c].tolist()
    for k in old.image_weights:
        np.testing.assert_allclose(new.image_weights[k], old.image_weights[k], rtol=1e-9, atol=1e-12)
