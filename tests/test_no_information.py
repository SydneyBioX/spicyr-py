"""A radius where a condition has no informative patient: tumour -> T avoids, and at r = 4 no T cell of condition A
has a tumour cell within r, so every image of A is at the bound -1 and A's CR2 variance would be exactly 0. The radius
is not tested and is left out of the combination over radii; the count effect has no bound and is unaffected. The
cells and spicyR's results are written by spicyR (noinfo_dev/make_noinfo_case.R, spicyR 1.99.27)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr
from spicyr import _cell, _core

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "noinfo_cells.csv")
CELLS3 = pd.read_csv(HERE / "noinfo_cells3.csv")
EXPECTED = json.loads((HERE / "noinfo_expected.json").read_text())
R_TO_PY = {"adjustAbundance": "adjust_abundance"}
# spicyR's results were written on x86-64 Linux; where long double is double (macOS arm64, MSVC) sums in the core round
# differently, so there they are compared as the shared cases are
EXTENDED = np.finfo(np.longdouble).nmant > 52


def same(got, exp, msg=""):
    if EXTENDED:
        np.testing.assert_array_equal(got, exp, err_msg=msg)
    else:
        np.testing.assert_allclose(got, exp, rtol=1e-7, atol=1e-10, equal_nan=True, err_msg=msg)


def run(cells=CELLS, **kw):
    return spicyr.spicy(cells, condition="condition", from_="tumour", to="T", **kw)


def run_case(name):
    kw = {R_TO_PY.get(k, k): v for k, v in EXPECTED[name]["args"].items()}
    if isinstance(kw["r"], list):
        kw["r"] = [float(v) for v in kw["r"]]
    else:
        kw["r"] = float(kw["r"])
    return run(**kw)


def as_list(v):
    return v if isinstance(v, list) else [v]


def num(v):
    return np.array([np.nan if x is None else float(x) for x in as_list(v)])


def test_the_rows_of_condition_a_are_all_at_minus_one():
    ctx = _cell.cell_context(CELLS, "condition", None)
    g = _cell.cell_graph(ctx, [("tumour", "T")], r=4.0, label_clustering=False)
    g.image_heterogeneity = True
    rows = _cell.cell_rows(ctx, g, "tumour", "T")
    assert rows["side"] == "avoid"
    assert (rows["O"][rows["group"] == 0] == -1).all()
    assert list(_core.informative_units(rows, rows["unit"], rows["group"], 2)) == [0, 8]
    u = _core.excess_test(rows, rows["unit"], rows["group"], len(ctx.unit_labels), True, "cr2")
    assert u["ok"] and u["se"] > 0 and u["coef_ref"] == -1


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_python_equals_r(name):
    exp = EXPECTED[name]
    res = run_case(name)
    cr = res.cell_results
    for col in ("p_value", "unadjusted_p_value", "r", "excess_difference", "se", "df"):
        if exp[col] is None:
            assert col not in cr.columns
            continue
        same(cr[col].to_numpy(float), num(exp[col]), f"{name}: {col}")
    if exp["adjusted_for"] is not None:
        assert list(cr["adjusted_for"]) == as_list(exp["adjusted_for"])
    if exp["radius_p"] is not None:
        rr = res.radius_results
        np.testing.assert_array_equal(rr["r"].to_numpy(float), num(exp["radius_r"]))
        same(rr["p_value"].to_numpy(float), num(exp["radius_p"]), f"{name}: radius p")
        if exp["radius_adjusted_for"] is not None:
            assert list(rr["adjusted_for"]) == as_list(exp["radius_adjusted_for"])


def test_one_radius_not_tested():
    x = run(r=4.0).cell_results
    assert x["p_value"].isna().all()
    assert x["excess_ref"].iloc[0] == -1
    x = run(r=4.0, covariates=["age"], adjust_abundance=True).cell_results
    assert list(x["adjusted_for"]) == ["none (no_information)"]
    assert x["p_value"].isna().all() and x["unadjusted_p_value"].isna().all()
    assert run(r=4.8).cell_results["p_value"].notna().all()


@pytest.mark.parametrize("combine", ["davies", "maxT", "cauchy"])
@pytest.mark.parametrize("adjust", [False, True])
def test_left_out_of_the_combination(combine, adjust):
    a = run(r=[4.0, 4.8, 5.2], combine=combine, adjust_abundance=adjust)
    b = run(r=[4.8, 5.2], combine=combine, adjust_abundance=adjust)
    rr = a.radius_results
    assert rr["p_value"][rr["r"] == 4].isna().all() and rr["p_value"][rr["r"] != 4].notna().all()
    if adjust:
        assert list(rr["adjusted_for"][rr["r"] == 4]) == ["none (no_information)"]
        assert a.cell_results["unadjusted_p_value"].iloc[0] == b.cell_results["unadjusted_p_value"].iloc[0]
    assert a.cell_results["p_value"].iloc[0] == b.cell_results["p_value"].iloc[0]
    assert a.cell_results["r"].iloc[0] == b.cell_results["r"].iloc[0]


@pytest.mark.parametrize("r", [4.0, 4.8])
def test_count_is_unaffected(r):
    ctx = _cell.cell_context(CELLS, "condition", None)
    g = _cell.cell_graph(ctx, [("tumour", "T")], r=r, label_clustering=False, effect="count")
    g.image_heterogeneity = True
    rows = _cell.cell_rows(ctx, g, "tumour", "T")
    u = _core.excess_test(rows, rows["unit"], rows["group"], len(ctx.unit_labels), True, "cr2")
    x = run(r=r, effect="count").cell_results
    assert x["p_value"].iloc[0] == u["p"] and np.isfinite(u["p"])


def test_more_than_two_conditions():
    with pytest.raises(ValueError, match="no pair could be tested"):
        run(CELLS3, r=4.0)
    assert len(run(CELLS3, r=4.8).cell_results)
