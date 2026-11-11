"""The twins agree: spicyr reproduces spicyR's results on the shared cases (tests/shared_cases, written by
make_cases.R with the R package)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "cells.csv")
CASES = json.loads((HERE / "cases.json").read_text())
R_TO_PY = {"from": "from_", "adjustAbundance": "adjust_abundance"}


def run(args):
    kw = {R_TO_PY.get(k, k): (v if not isinstance(v, list) or len(v) > 1 else v[0]) for k, v in args.items()}
    surv = {"os": "time", "os_na": "time_na", "os_few": "time_few"}
    if kw["condition"] in surv:
        kw.update(condition=surv[kw["condition"]], survival="event")
    return spicyr.spicy(CELLS, **kw)


def same_table(got, exp, name):
    """Every column of R's table, numbers to the shared tolerance, the rest as text."""
    assert sorted(got.index) == sorted(exp.index), name
    got = got.loc[exp.index]
    for col in exp.columns:
        if exp[col].dtype.kind in "fi":
            np.testing.assert_allclose(
                got[col].to_numpy(float),
                exp[col].to_numpy(float),
                rtol=1e-7,
                atol=1e-10,
                equal_nan=True,
                err_msg=f"{name}: {col}",
            )
        else:
            g = got[col].astype(object).where(got[col].notna(), np.nan).astype(str).to_numpy()
            assert (g == exp[col].astype(str).to_numpy()).all(), f"{name}: {col}"


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_matches_r(name):
    res = run(CASES[name])
    exp = pd.read_csv(HERE / f"{name}.csv", index_col=0)
    same_table(res.cell_results, exp, name)
    follow = HERE / f"{name}_followups.csv"
    if follow.exists():
        ef = pd.read_csv(follow, index_col=0)
        assert list(res.follow_up_results.columns) == list(ef.columns), name
        same_table(res.follow_up_results, ef, f"{name} follow-ups")
    else:
        assert res.follow_up_results is None
    radii = HERE / f"{name}_radii.csv"
    if radii.exists():
        er = pd.read_csv(radii)
        gr = res.radius_results.reset_index(drop=True)
        key = ["r", "from", "to"]
        m = er.merge(gr, on=key, suffixes=("_r", "_py"))
        assert len(m) == len(er)
        np.testing.assert_allclose(m["p_value_py"], m["p_value_r"], rtol=1e-7, atol=1e-10)


def test_variance():
    """CR2 for every study size; Hartung-Knapp (m - 2 df) only on request; the removed arguments are refused."""
    pts = CELLS[["patient", "condition"]].drop_duplicates()
    keep = pts.groupby("condition")["patient"].apply(lambda s: list(s)[:4]).explode()
    small = CELLS[CELLS["patient"].isin(keep)]
    rd = spicyr.spicy(small, condition="condition", subject="patient", r=20, from_="tumour", to="T")
    assert rd.variance == "cr2"
    rh = spicyr.spicy(
        small, condition="condition", subject="patient", r=20, from_="tumour", to="T", variance="hartung_knapp"
    )
    assert rh.variance == "hartung_knapp"
    np.testing.assert_allclose(rh.cell_results["df"], 6)
    with pytest.raises(ValueError):
        spicyr.spicy(small, condition="condition", subject="patient", variance="auto")
    with pytest.raises(TypeError, match="not an argument"):
        spicyr.spicy(small, condition="condition", subject="patient", label_clustering=True)


def test_maxt_any_number_of_radii():
    """max-T takes any number of radii (it stopped at 6 up to 1.99.39) and reports whether its p-value converged."""
    res = spicyr.spicy(CELLS, condition="condition", subject="patient", r=list(range(5, 45, 5)), from_="tumour", to="T",
                       combine="maxT")
    assert res.cell_results["maxT_converged"].tolist() == [True]
    res = spicyr.spicy(
        CELLS, condition="condition", subject="patient", r=[10, 20], from_="tumour", to="T", combine="cauchy"
    )
    assert "maxT_converged" not in res.cell_results.columns


def test_survival_one_radius_and_alpha():
    """A survival outcome takes one radius; alpha is a number between 0 and 1."""
    with pytest.raises(ValueError, match="one radius"):
        spicyr.spicy(CELLS, survival=("time", "event"), subject="patient", r=[10, 20], from_="tumour", to="T")
    for bad in (0, 1, -0.1, float("nan"), "0.05", True):
        with pytest.raises(ValueError, match="alpha"):
            spicyr.spicy(CELLS, condition="grp3", subject="patient", r=30, from_="tumour", to="T", alpha=bad)


def test_max_abs_tail():
    """The core's max-T tail: P(max |Z_k| >= c) for independent Z is 1 - (1 - 2 Phi(-c))^K, to about 0.1%."""
    from math import erfc, sqrt

    from spicyr import _core

    for K in (3, 6, 12):
        c = 3.0
        p, err, _, conv = _core.max_abs_tail(c, np.eye(K).ravel())
        exact = 1 - (1 - erfc(c / sqrt(2))) ** K
        assert conv and abs(p - exact) <= 2e-3 * exact and err <= 1.5e-3 * exact


def test_radii_one_search():
    """Several radii from one neighbour search equal the single-radius graphs."""
    from spicyr import _cell

    ctx = _cell.cell_context(CELLS, "condition", "patient")
    radii = [5.0, 12.5, 20.0, 40.0]
    for eff, names in (("allocation", ("any", "self_expected")), ("count", ("totals", "sq"))):
        gs = _cell.cell_graphs_radii(ctx, radii, eff)
        for r, g in zip(radii, gs, strict=True):
            g1 = _cell.cell_graph(ctx, [("tumour", "T")], r=r, label_clustering=False, effect=eff)
            for nm in names:
                np.testing.assert_array_equal(getattr(g, nm), getattr(g1, nm))


def test_df_guard():
    """A test on fewer than 1 Satterthwaite df, or infinite df, is not done (too_few_df; as spicyR's test-survival.R)."""
    from spicyr import _cell, _core

    rules = _core.test_rules(True)
    assert rules["min_df"] == 1
    assert _cell._too_few_df(0.7, rules) and _cell._too_few_df(np.inf, rules) and _cell._too_few_df(np.nan, rules)
    assert not _cell._too_few_df(1, rules)
    # four patients, one with nearly all the weight and the one death: the score test's df are 0.03
    rows = {
        "img": np.arange(4, dtype=np.int32),
        "O": np.array([-0.986191, -0.394855, 1.30852, 0.95476]),
        "E": np.zeros(4),
        "n": np.ones(4),
        "v": np.array([0.000196998, 71.9433, 0.00535398, 8.92664]),
    }
    M = np.array([4.89172, -0.127562, -0.0564688, -0.051067])
    unit = np.arange(4, dtype=np.int32)
    d = _core.design_test(rows, unit, 4, np.column_stack([np.ones(4), M]), np.array([0.0, 1.0]), -1)
    assert d["ok"] and d["df"] < 1
    z = _core.survival_test(
        rows, unit, 4, M, np.arange(1.0, 5.0), np.array([1, 0, 0, 0], dtype=np.int32), np.zeros(0), 0, np.zeros(0), 0
    )
    assert not z["ok"] and z["reason"] == "too_few_df"


def test_davies_is_the_default_for_several_radii():
    """combine="davies" is the default (1.99.41); its bound is never below the exact max-T p-value."""
    kw = dict(condition="condition", subject="patient", r=[10, 20, 40], from_="tumour", to="T")
    dflt = spicyr.spicy(CELLS, **kw).cell_results
    dv = spicyr.spicy(CELLS, combine="davies", **kw).cell_results
    mt = spicyr.spicy(CELLS, combine="maxT", **kw).cell_results
    pd.testing.assert_frame_equal(dflt, dv)
    assert dv["p_value"].iloc[0] >= mt["p_value"].iloc[0] * (1 - 2e-3)
