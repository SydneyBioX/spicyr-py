"""Images whose condition is missing are left out of every test.

With one image per patient, the result equals that with those images removed.
"""

import numpy as np
import pandas as pd
import pytest

import spicyr


def _data():
    rng = np.random.default_rng(2)
    n = 20
    im = pd.DataFrame(
        {"cond": pd.Categorical(np.tile(["a", "b"], n // 2)), "cv": rng.normal(size=n)},
        index=[f"i{i:02d}" for i in range(n)],
    )
    im["cond3"] = pd.Categorical(np.resize(["x", "y", "z"], n))
    im.loc[im.index[:2], ["cond", "cond3"]] = np.nan
    cells = pd.concat(
        [
            pd.DataFrame(
                {
                    "imageID": i,
                    "cellType": rng.choice(["A", "B"], 500),
                    "x": rng.uniform(0, 500, 500),
                    "y": rng.uniform(0, 500, 500),
                }
            )
            for i in im.index
        ]
    )
    return cells, im


def _full_and_sub(condition, **kw):
    cells, im = _data()
    keep = im[condition].notna()
    full = spicyr.spicy(cells, condition=condition, image_info=im, r=50, **kw)
    sub = spicyr.spicy(
        cells[cells["imageID"].isin(im.index[keep])], condition=condition, image_info=im[keep], r=50, **kw
    )
    by = [c for c in ("from", "to", "level") if c in full.cell_results]
    a = full.cell_results.sort_values(by).reset_index(drop=True)
    b = sub.cell_results.sort_values(by).reset_index(drop=True)
    return a, b


def _same(a, b, cols):
    assert len(a) == len(b) > 0
    for c in cols:
        np.testing.assert_allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-12, err_msg=c)


@pytest.mark.parametrize("effect", ["count", "allocation"])
def test_missing_condition_left_out_of_unadjusted_test(effect):
    a, b = _full_and_sub("cond", effect=effect)
    _same(a, b, ["excess_ref", "excess_comp", "excess_difference", "se", "df", "p_value", "tau2"])


@pytest.mark.parametrize("effect", ["count", "allocation"])
def test_missing_condition_left_out_of_covariate_design(effect):
    a, b = _full_and_sub("cond", covariates=["cv"], effect=effect)
    assert (a["adjusted_for"] == "covariates").all()
    _same(
        a,
        b,
        [
            "excess_difference",
            "se",
            "df",
            "p_value",
            "cv_effect",
            "unadjusted_difference",
            "unadjusted_se",
            "unadjusted_df",
            "unadjusted_p_value",
        ],
    )


@pytest.mark.parametrize("effect", ["count", "allocation"])
def test_missing_condition_left_out_of_three_conditions(effect):
    a, b = _full_and_sub("cond3", effect=effect)
    assert (a["level"] == b["level"]).all()
    _same(a, b, ["excess_ref", "excess_difference", "se", "df", "p_value", "tau2"])
