"""Every pair at every radius in one core call (pair_tests): equal to each pair's test in turn (cell_pair_test), and
with a design to adjust for, exactly spicyR's results (tests/shared_cases/exact_*, written by make_exact_cases.R)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr
from spicyr import _api, _cell

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "cells.csv")
EXACT = json.loads((HERE / "exact_cases.json").read_text())
R_TO_PY = {"from": "from_", "adjustAbundance": "adjust_abundance"}
BASE = {"condition": "condition", "subject": "patient"}
# The core sums the design's columns in long double, as R does. The references were written on x86-64 Linux
# (80-bit long double); where long double is double (macOS on arm64, MSVC) the sums round differently.
EXTENDED = np.finfo(np.longdouble).nmant > 52


def equal(x, y, err_msg):
    if EXTENDED:
        np.testing.assert_array_equal(x, y, err_msg=err_msg)
    else:
        np.testing.assert_allclose(x, y, rtol=1e-7, atol=1e-10, equal_nan=True, err_msg=err_msg)  # test_shared_cases


def each_pair(ctx, graphs, pairs, variance, adjust, Z_extra, extra_names, radii, combine, n_threads=1):
    """One radius: cell_pair_test for each pair, then cell_table."""
    g = graphs[0]
    g.image_heterogeneity = True
    fits = [_cell.cell_pair_test(ctx, g, f, t, True, variance, adjust, Z_extra, extra_names) for f, t in pairs]
    return {"table": _cell.cell_table(fits, ctx, adjust or Z_extra is not None), "fits": fits}


@pytest.mark.parametrize(
    "args",
    [
        {"r": 30},
        {"r": 30, "effect": "count"},
        {"k": 8},
        {"r": 20, "variance": "hartung_knapp", "cores": 3},
        {"r": 30, "from_": ["T", "B"], "to": "tumour"},
    ],
)
def test_identical_to_each_pair_in_turn(monkeypatch, args):
    args = {**BASE, **args}
    new = spicyr.spicy(CELLS, **args)
    with monkeypatch.context() as m:
        m.setattr(_api, "_pairs", each_pair)
        old = spicyr.spicy(CELLS, **args)
    pd.testing.assert_frame_equal(new.cell_results, old.cell_results, check_exact=True)
    for d in ("pairwise_assoc", "image_weights"):
        x, y = getattr(new, d), getattr(old, d)
        assert list(x) == list(y)
        for k in x:
            np.testing.assert_array_equal(x[k], y[k])


def full(path):
    """An R table written at 17 significant digits, read back exactly."""
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=["NA"])


@pytest.mark.parametrize("name", sorted(EXACT))
def test_adjusted_exactly_as_r(name):
    kw = {R_TO_PY.get(k, k): (v if not isinstance(v, list) or len(v) > 1 else v[0]) for k, v in EXACT[name].items()}
    res = spicyr.spicy(CELLS, **kw)
    exp = full(HERE / f"exact_{name}.csv").set_index("Unnamed: 0")
    got = res.cell_results
    assert sorted(got.index) == sorted(exp.index)
    got = got.loc[exp.index]
    assert list(got.columns) == list(exp.columns)
    for col in exp.columns:
        if exp[col].dtype.kind in "fi":
            equal(got[col].to_numpy(float), exp[col].to_numpy(float), f"{name}: {col}")
        else:
            assert got[col].astype(str).tolist() == exp[col].astype(str).tolist(), f"{name}: {col}"
    radii = HERE / f"exact_{name}_radii.csv"
    if radii.exists():
        er = full(radii)
        gr = res.radius_results.reset_index(drop=True)
        m = er.merge(gr, on=["r", "from", "to"], suffixes=("_r", "_py"))
        assert len(m) == len(er) == len(gr)
        for col in ("p_value", "excess_difference", "se", "unadjusted_p_value"):
            if f"{col}_r" in m:
                equal(m[f"{col}_py"].to_numpy(float), m[f"{col}_r"].to_numpy(float), f"{name}: radii {col}")
