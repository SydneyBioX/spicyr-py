"""Images whose condition is missing are left out of every test, the covariate design included."""

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
    im.loc[im.index[:2], "cond"] = np.nan
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


@pytest.mark.parametrize("effect", ["count", "allocation"])
def test_missing_condition_left_out(effect):
    cells, im = _data()
    keep = im["cond"].notna()
    full = spicyr.spicy(cells, condition="cond", covariates=["cv"], image_info=im, r=50, effect=effect)
    sub = spicyr.spicy(
        cells[cells["imageID"].isin(im.index[keep])],
        condition="cond",
        covariates=["cv"],
        image_info=im[keep],
        r=50,
        effect=effect,
    )
    a = full.cell_results.sort_values(["from", "to"]).reset_index(drop=True)
    b = sub.cell_results.sort_values(["from", "to"]).reset_index(drop=True)
    assert (a["adjusted_for"] == "covariates").all()
    np.testing.assert_allclose(a["p_value"], b["p_value"])
    np.testing.assert_allclose(a["excess_difference"], b["excess_difference"])
