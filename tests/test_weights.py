"""Cell weights (spicy(cell_weights=)): one-hot weights give the cell types' results, a vector of cell weights equals
the one-hot matrix scaled by it, and spicyr reproduces spicyR on the shared cases (tests/shared_cases/make_weights_cases.R)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import spicyr

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "cells.csv")
WEIGHTS = pd.read_csv(HERE / "weights.csv")
CASES = json.loads((HERE / "weights_cases.json").read_text())
R_TO_PY = {"from": "from_", "adjustAbundance": "adjust_abundance"}


def one_hot(cells):
    return pd.get_dummies(cells["cellType"]).astype(float)


def compare(got, exp, tol=1e-12, rtol=0.0):
    assert sorted(got.index) == sorted(exp.index)
    assert list(got.columns) == list(exp.columns)
    got = got.loc[exp.index]
    worst = 0.0
    for col in exp.columns:
        if exp[col].dtype.kind in "fi":
            a, b = got[col].to_numpy(float), exp[col].to_numpy(float)
            np.testing.assert_allclose(a, b, rtol=rtol, atol=tol, equal_nan=True, err_msg=col)
            ok = np.isfinite(a) & np.isfinite(b)
            worst = max(worst, float(np.max(np.abs(a[ok] - b[ok]), initial=0.0)))
        else:
            assert (got[col].astype(str).to_numpy() == exp[col].astype(str).to_numpy()).all(), col
    return worst


@pytest.mark.parametrize("effect", ["allocation", "count"])
@pytest.mark.parametrize("adjust", [False, True])
def test_one_hot_weights_give_the_cell_types(effect, adjust):
    kw = {"condition": "condition", "subject": "patient", "r": 30, "effect": effect, "adjust_abundance": adjust}
    hard = spicyr.spicy(CELLS, **kw).cell_results
    soft = spicyr.spicy(CELLS, cell_weights=one_hot(CELLS), **kw).cell_results
    # self-pairs are not tested with weights, so the BH adjustment runs over the other pairs
    hard = hard[hard["from"] != hard["to"]].copy()
    for p in ("p_adj", "unadjusted_p_adj"):
        if p in hard.columns:
            hard[p] = spicyr._cell.p_adjust_bh(hard[p.replace("p_adj", "p_value")])
    compare(soft, hard, tol=1e-12)


def test_one_hot_weights_give_the_cell_types_with_three_levels_and_survival():
    w = one_hot(CELLS)
    for kw in (
        {"condition": "stage", "subject": "patient", "r": 30, "from_": "tumour", "to": ["T", "B"], "covariates": "age"},
        {"survival": ("time", "event"), "subject": "patient", "r": 30, "from_": "tumour", "to": ["T", "B"]},
    ):
        hard = spicyr.spicy(CELLS, adjust_abundance=True, **kw).cell_results
        soft = spicyr.spicy(CELLS, adjust_abundance=True, cell_weights=w, **kw).cell_results
        compare(soft, hard, tol=1e-12)


def test_array_and_sparse_weights_match_a_data_frame():
    kw = {"condition": "condition", "subject": "patient", "r": 30, "from_": "tumour", "to": ["T", "lymphocyte"]}
    ref = spicyr.spicy(CELLS, cell_weights=WEIGHTS, **kw).cell_results
    for m in (WEIGHTS.to_numpy(), sp.csr_matrix(WEIGHTS.to_numpy())):
        got = spicyr.spicy(CELLS, cell_weights=m, weight_labels=list(WEIGHTS.columns), **kw).cell_results
        pd.testing.assert_frame_equal(got, ref)


def test_weights_follow_the_rows_of_the_cells():
    kw = {"condition": "condition", "subject": "patient", "r": 30, "from_": ["tumour", "B"], "to": ["T", "lymphocyte"]}
    ref = spicyr.spicy(CELLS, cell_weights=WEIGHTS, **kw).cell_results
    perm = np.random.default_rng(5).permutation(len(CELLS))
    got = spicyr.spicy(
        CELLS.iloc[perm].reset_index(drop=True), cell_weights=WEIGHTS.iloc[perm].reset_index(drop=True), **kw
    ).cell_results
    compare(got, ref, tol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("effect", ["allocation", "count"])
def test_a_vector_of_cell_weights_is_the_one_hot_matrix_scaled_by_it(effect):
    kw = {"condition": "condition", "subject": "patient", "r": 30, "effect": effect, "adjust_abundance": True}
    q = np.round(np.random.default_rng(7).uniform(0.2, 1, len(CELLS)), 3)
    q[::17] = 0.0
    ref = spicyr.spicy(CELLS, cell_weights=one_hot(CELLS) * q[:, None], **kw).cell_results
    for v in (q, list(q), pd.Series(q, index=CELLS.index[::-1])):
        got = spicyr.spicy(CELLS, cell_weights=v, **kw).cell_results
        compare(got, ref, tol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_matches_r(name):
    kw = {R_TO_PY.get(k, k): (v if not isinstance(v, list) or len(v) > 1 else v[0]) for k, v in CASES[name].items()}
    if kw["condition"] == "os":
        kw.pop("condition")
        kw["survival"] = ("time", "event")
    got = spicyr.spicy(CELLS, cell_weights=WEIGHTS, **kw).cell_results
    exp = pd.read_csv(HERE / f"{name}.csv", index_col=0)
    compare(got, exp, tol=1e-10, rtol=1e-7)


def test_weights_are_checked():
    kw = {"condition": "condition", "subject": "patient", "r": 30}
    with pytest.raises(ValueError, match="rows"):
        spicyr.spicy(CELLS, cell_weights=WEIGHTS.iloc[:10], **kw)
    with pytest.raises(ValueError, match="values but there are"):
        spicyr.spicy(CELLS, cell_weights=np.ones(len(CELLS) - 1), **kw)
    with pytest.raises(ValueError, match="between 0 and 1"):
        spicyr.spicy(CELLS, cell_weights=np.full(len(CELLS), 1.5), **kw)
    with pytest.raises(ValueError, match="between 0 and 1"):
        spicyr.spicy(CELLS, cell_weights=-WEIGHTS, **kw)
    with pytest.raises(ValueError, match="between 0 and 1"):
        spicyr.spicy(CELLS, cell_weights=2 * WEIGHTS, **kw)
    with pytest.raises(ValueError, match="labels"):
        spicyr.spicy(CELLS, cell_weights=WEIGHTS.to_numpy(), **kw)
    with pytest.raises(ValueError, match="not a column"):
        spicyr.spicy(CELLS, cell_weights=WEIGHTS, from_="nope", **kw)
    with pytest.raises(ValueError, match="one radius"):
        spicyr.spicy(CELLS, cell_weights=WEIGHTS, **dict(kw, r=[10, 30]))
    with pytest.raises(ValueError, match="different labels"):
        spicyr.spicy(CELLS, cell_weights=WEIGHTS, from_="T", to="T", **kw)
