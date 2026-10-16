"""Images whose condition is missing are left out of every step: the choice of the side, the between-image variance
and every test.

The result equals that with those images removed. Patients have two images; two patients have no condition.
"""

import numpy as np
import pandas as pd
import pytest

import spicyr


def _data(n_pat=18):
    rng = np.random.default_rng(2)
    n = 2 * n_pat
    pat = np.repeat(np.arange(n_pat), 2)
    im = pd.DataFrame(
        {
            "patient": [f"p{p:02d}" for p in pat],
            "cv": rng.normal(size=n),
            "cond": pd.Categorical(np.resize(["a", "b"], n_pat)[pat]),
            "cond3": pd.Categorical(np.resize(["x", "y", "z"], n_pat)[pat]),
        },
        index=[f"i{i:02d}" for i in range(n)],
    )
    im.loc[im.index[[0, 1, 6, 7]], ["cond", "cond3"]] = np.nan
    parts = []
    for i in im.index:
        m = int(rng.integers(300, 501))
        parts.append(
            pd.DataFrame(
                {
                    "imageID": i,
                    "cellType": rng.choice(["A", "B", "C"], m, p=[0.5, 0.3, 0.2]),
                    "x": rng.uniform(0, 400, m),
                    "y": rng.uniform(0, 400, m),
                }
            )
        )
    return pd.concat(parts, ignore_index=True), im


def _full_and_sub(condition, r=20.0, **kw):
    cells, im = _data()
    keep = im[condition].notna()
    full = spicyr.spicy(cells, condition=condition, image_info=im, r=r, **kw)
    sub = spicyr.spicy(
        cells[cells["imageID"].isin(im.index[keep])], condition=condition, image_info=im[keep], r=r, **kw
    )

    def key(t):
        by = [c for c in ("from", "to", "level", "r") if c in t]
        return t.sort_values(by).reset_index(drop=True)

    out = [key(full.cell_results), key(sub.cell_results)]
    if full.radius_results is not None:
        out += [key(full.radius_results), key(sub.radius_results)]
    return out


def _same(a, b):
    assert len(a) == len(b) > 0
    assert list(a.columns) == list(b.columns)
    for c in a.columns:
        if a[c].dtype.kind in "fi":
            np.testing.assert_allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-9, atol=1e-12, err_msg=c)
        else:
            assert list(a[c]) == list(b[c]), c


def _all_same(tables):
    for a, b in zip(tables[::2], tables[1::2], strict=True):
        _same(a, b)


@pytest.mark.parametrize("effect", ["count", "allocation"])
def test_missing_condition_left_out_as_if_removed(effect):
    _all_same(_full_and_sub("cond", effect=effect))
    _all_same(_full_and_sub("cond", subject="patient", effect=effect))
    _all_same(_full_and_sub("cond", subject="patient", r=[10.0, 20.0, 30.0], combine="davies", effect=effect))


@pytest.mark.parametrize("effect", ["count", "allocation"])
def test_missing_condition_left_out_of_adjusted_design(effect):
    t = _full_and_sub("cond", subject="patient", covariates=["cv"], effect=effect)
    assert (t[0]["adjusted_for"] == "covariates").any()
    _all_same(t)
    t = _full_and_sub("cond", subject="patient", adjust_abundance=True, effect=effect)
    assert (t[0]["adjusted_for"] == "abundance").any()
    _all_same(t)
    _all_same(
        _full_and_sub(
            "cond",
            subject="patient",
            adjust_abundance=True,
            covariates=["cv"],
            r=[10.0, 20.0, 30.0],
            combine="maxT",
            effect=effect,
        )
    )


@pytest.mark.parametrize("effect", ["count", "allocation"])
def test_missing_condition_left_out_of_three_conditions(effect):
    _all_same(_full_and_sub("cond3", subject="patient", effect=effect))
    _all_same(_full_and_sub("cond3", subject="patient", covariates=["cv"], effect=effect))
