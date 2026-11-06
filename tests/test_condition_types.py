"""A continuous condition (a number with more than two values) and the overall test of more than two conditions.

The numbers are spicyR's (tests/shared_cases: continuous*, three_levels*); here the behaviour around them.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr
from spicyr import _cell, _core

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "cells.csv")


def fit(cells=CELLS, condition="age", **kw):
    kw = {"subject": "patient", "r": 30, "from_": "tumour", "to": "T", **kw}
    return spicyr.spicy(cells, condition=condition, **kw)


def test_continuous_is_the_slope_of_the_design():
    res = fit()
    tab = res.cell_results
    assert res.continuous
    assert {"excess_mean", "excess_slope", "se", "df", "p_value", "p_adj"} <= set(tab.columns)
    assert "level" not in tab.columns and "excess_difference" not in tab.columns
    # the design test with the intercept and the centred condition, on the pair's rows
    ctx = _cell.cell_context(CELLS, "age", "patient")
    assert ctx.continuous
    g = _cell.cell_graph(ctx, [("tumour", "T")], r=30, label_clustering=False, effect="allocation")
    g.image_heterogeneity = True
    rows = _cell.cell_rows(ctx, g, "tumour", "T")
    x = ctx.x[rows["img"]]
    Z = np.column_stack([np.ones(len(x)), x - x.mean()])
    d = _core.design_tests(rows, rows["unit"], len(ctx.unit_labels), Z, np.array([[0.0, 1.0]]), 1, -1.0, False)[0]
    got = tab.iloc[0][["excess_mean", "excess_slope", "se", "df", "p_value"]].to_numpy(float)
    np.testing.assert_allclose(got, [d["theta"][0], d["estimate"], d["se"], d["df"], d["p"]], rtol=1e-10)


def test_continuous_units_do_not_change_the_test():
    c = CELLS.copy()
    c["age10"] = (c["age"] - 50) / 10
    a, b = fit().cell_results, fit(c, "age10").cell_results
    np.testing.assert_allclose(b["excess_slope"], 10 * a["excess_slope"], rtol=1e-8)
    for col in ("excess_mean", "p_value", "df"):
        np.testing.assert_allclose(b[col], a[col], rtol=1e-8)


def test_two_values_and_categories_are_groups():
    c = CELLS.copy()
    c["two"] = np.where(c["condition"] == "A", 0.0, 1.0)
    two = fit(c, "two").cell_results
    ref = fit(c, "condition").cell_results
    np.testing.assert_allclose(two["p_value"], ref["p_value"], rtol=1e-12)
    c["agec"] = pd.Categorical(c["age"])
    assert not _cell.cell_context(c, "agec", "patient").continuous
    c["flag"] = c["condition"] == "A"
    assert not _cell.cell_context(c, "flag", "patient").continuous


def test_continuous_refusals():
    with pytest.raises(ValueError, match="continuous"):
        fit(ref=50)
    with pytest.raises(ValueError, match="give one `r`"):
        fit(r=[20, 30])
    bad = CELLS.copy()
    bad.loc[bad["imageID"] == bad["imageID"].iloc[0], "age"] = 99.0
    with pytest.raises(ValueError, match="single value|constant within each image"):
        fit(bad)


def test_continuous_missing_is_removed():
    na = CELLS.copy()
    first = na["patient"].iloc[0]
    na.loc[na["patient"] == first, "age"] = np.nan
    a = fit(na).cell_results
    b = fit(CELLS[CELLS["patient"] != first]).cell_results
    np.testing.assert_allclose(a["p_value"], b["p_value"], rtol=1e-10)


def test_joint_test_of_one_contrast_is_the_t_test():
    ctx = _cell.cell_context(CELLS, "stage", "patient")
    g = _cell.cell_graph(ctx, [("tumour", "T")], r=30, label_clustering=False, effect="allocation")
    g.image_heterogeneity = True
    rows = _cell.cell_rows(ctx, g, "tumour", "T")
    Z = (rows["group"][:, None] == np.arange(3)[None, :]).astype(float)
    C = _cell._contrasts(3, 3)
    m = len(ctx.unit_labels)
    for hk in (False, True):
        t = _core.design_tests(rows, rows["unit"], m, Z, C[1:], 1, 0.01, hk)[0]
        j = _core.design_joint_test(rows, rows["unit"], Z, C[1:], 1, 0.01, hk)
        np.testing.assert_allclose(
            [j["statistic"], j["df2"], j["p"]], [(t["estimate"] / t["se"]) ** 2, t["df"], t["p"]], rtol=1e-10
        )
    hk = _core.design_joint_test(rows, rows["unit"], Z, C, 2, 0.01, True)
    cr = _core.design_joint_test(rows, rows["unit"], Z, C, 2, 0.01, False)
    assert hk["wald"] <= cr["wald"] + 1e-12  # Hartung-Knapp floored at CR2


def test_overall_test_columns():
    res = spicyr.spicy(CELLS, condition="stage", subject="patient", r=30, from_=["tumour", "T"], to=["T", "B"])
    t = res.cell_results
    assert (t.groupby(["from", "to"])["overall_p_value"].nunique(dropna=False) == 1).all()
    assert (t["overall_df1"].dropna() == 2).all()
    first = (t["level"] == t["level"].iloc[0]).to_numpy()
    np.testing.assert_allclose(t.loc[first, "overall_p_adj"], _cell.p_adjust_bh(t.loc[first, "overall_p_value"]))
    assert {"overall_p_value", "overall_p_adj"} <= set(res.top_pairs().columns)
    assert "Overall test" in repr(res)


def test_continuous_accessors_and_plots():
    import matplotlib

    matplotlib.use("Agg")
    res = fit(from_=["tumour", "T"], to=["T", "B"])
    assert "slope on the condition" in repr(res) and "condition from" in repr(res)
    assert list(res.coefficient.columns) == ["(Intercept)", "condition"]
    assert "excess_slope" in res.top_pairs().columns
    assert res.box_plot("tumour", "T") is not None
    assert res.signif_plot() is not None
    with pytest.raises(ValueError, match="continuous"):
        res.signif_plot(comparison_group="B")


def test_calibration_shuffles_a_continuous_condition():
    cal = spicyr.calibration_check(
        CELLS, condition="age", subject="patient", n_shuffles=2, r=30, from_="tumour", to="T", rng=1
    )
    assert len(cal) == 4
