"""col_test() and get_prop() reproduce spicyR's colTest() and getProp() (cases written by make_col_test_cases.R)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "cells.csv")
# survival::coxph() 3.8.6 (coef, se, Wald p) on the patient proportions where spicyR's own Cox fit gives up
RARE_STROMA = {
    "rare": (-76.4232388010477, 82.1834258639778, 0.352417374597857),
    "stroma": (-67.5529070188653, 48.2631000401898, 0.161609096270749),
}


def expected(name):
    return pd.read_csv(HERE / f"col_test_{name}.csv", index_col=0)


def same(got, exp):
    assert list(got.columns) == list(exp.columns)
    assert sorted(got.index) == sorted(exp.index)
    # rows are ordered by p-value; rows with equal p-values may come in another order
    np.testing.assert_array_equal(got["pval"].to_numpy(), exp["pval"].to_numpy())
    got = got.loc[exp.index]
    for col in exp.columns:
        if col == "cluster":
            assert list(got[col]) == list(exp[col])
        else:
            np.testing.assert_allclose(got[col].to_numpy(float), exp[col].to_numpy(float), rtol=1e-12, err_msg=col)


def test_get_prop():
    got = spicyr.get_prop(CELLS, feature="cellType", image_id="patient")
    exp = expected("prop_patient")
    assert list(got.columns) == list(exp.columns)
    assert list(got.index) == list(exp.index)
    np.testing.assert_allclose(got.to_numpy(), exp.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize(
    ("name", "image_id", "type"),
    [("wilcox_patient", "patient", "wilcox"), ("wilcox_image", "imageID", "wilcox"), ("ttest_image", "imageID", None)],
)
def test_cells(name, image_id, type):
    got = spicyr.col_test(CELLS, condition="condition", feature="cellType", image_id=image_id, type=type)
    same(got, expected(name))


def test_wilcox_with_ties():
    props = spicyr.get_prop(CELLS, feature="cellType", image_id="imageID").round(2)
    cond = CELLS.drop_duplicates("imageID").set_index("imageID")["condition"]
    same(spicyr.col_test(props, cond.loc[props.index], type="wilcox"), expected("wilcox_ties"))


def test_survival():
    props = spicyr.get_prop(CELLS, feature="cellType", image_id="patient")
    surv = CELLS.drop_duplicates("patient").set_index("patient")[["time", "event"]]
    same(spicyr.col_test(props, surv.loc[props.index]), expected("survival"))


def test_anndata_matches_dataframe():
    ad = pytest.importorskip("anndata")
    obs = CELLS.copy()
    obs.index = [f"c{i}" for i in range(len(obs))]
    a = ad.AnnData(obs=obs)
    got = spicyr.col_test(a, condition="condition", feature="cellType", image_id="patient", type="wilcox")
    pd.testing.assert_frame_equal(got, spicyr.col_test(obs, "condition", "wilcox", "cellType", "patient"))


def test_wilcox_normal_approximation():
    big = pd.read_csv(HERE / "col_test_big_input.csv")
    same(spicyr.col_test(big[["a", "b", "c"]], big["group"], type="wilcox"), expected("wilcox_big"))


def test_coxph_fallback_unrounded():
    props = spicyr.get_prop(CELLS, feature="cellType", image_id="patient")
    surv = CELLS.drop_duplicates("patient").set_index("patient").loc[props.index]
    from spicyr._coltest import _coxph

    for col, coef in RARE_STROMA.items():
        got = _coxph(props[col].to_numpy(), surv["time"].to_numpy(), surv["event"].to_numpy())
        np.testing.assert_allclose(got, coef, rtol=1e-6)
