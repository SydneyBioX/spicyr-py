"""The accessors of the results: top_pairs() and bind() give spicyR's topPairs() and bind() tables on the shared
cases (tests/shared_cases/make_accessor_cases.R), radius_plot(), plot_image() with several images, the summary and
plot of calibration_check(), and the printed summary."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "cells.csv")
SPEC = json.loads((HERE / "accessor_cases.json").read_text())
R_TO_PY = {"from": "from_", "adjustAbundance": "adjust_abundance"}

_fits = {}


def fit(name):
    if name not in _fits:
        kw = {
            R_TO_PY.get(k, k): (v if not isinstance(v, list) or len(v) > 1 else v[0])
            for k, v in SPEC["fits"][name].items()
        }
        if kw["condition"] == "os":
            kw.pop("condition")
            kw["survival"] = ("time", "event")
        _fits[name] = spicyr.spicy(CELLS, **kw)
    return _fits[name]


@pytest.mark.parametrize("name", sorted(SPEC["cases"]))
def test_accessor_matches_r(name):
    case = SPEC["cases"][name]
    res = fit(case["fit"])
    args = {R_TO_PY.get(k, k): v for k, v in (case["args"] or {}).items()}  # an empty R list is []
    got = res.top_pairs(**args) if case["call"] == "topPairs" else res.bind(**args)
    exp = pd.read_csv(HERE / f"accessor_{name}.csv", index_col=0, keep_default_na=False, na_values=["NA"])
    assert list(got.columns) == list(exp.columns)
    assert list(got.index.astype(str)) == list(exp.index.astype(str))
    for col in exp.columns:
        if exp[col].dtype.kind in "fi":
            np.testing.assert_allclose(
                got[col].to_numpy(float), exp[col].to_numpy(float), rtol=1e-7, atol=1e-10, equal_nan=True, err_msg=col
            )
        else:
            assert (got[col].astype(str).to_numpy() == exp[col].astype(str).to_numpy()).all(), col


def test_top_pairs():
    res = fit("two_groups")
    every = res.top_pairs(n=None)
    assert len(every) == len(res.cell_results) and not every["p_value"].dropna().is_monotonic_decreasing
    assert every["p_value"].dropna().is_monotonic_increasing
    pd.testing.assert_frame_equal(
        res.top_pairs(n=None, full=True), res.cell_results.sort_values("p_value", kind="stable")
    )
    assert len(res.top_pairs()) == 10
    # every pair below a cutoff, not only the first ten; n still caps it when given
    below = int((res.cell_results["p_adj"] <= 0.9).sum())
    assert below > 10 and len(res.top_pairs(cutoff=0.9)) == below and len(res.top_pairs(cutoff=0.9, n=3)) == 3
    # chosen pairs; "from" as a keyword; a cell type with no pair is an error
    assert list(res.top_pairs(from_="tumour", to="T").index) == ["tumour__T"]
    assert (res.top_pairs(n=None, **{"from": "tumour"})["from"] == "tumour").all()
    with pytest.raises(ValueError, match="no pair with from = 'nothere'"):
        res.top_pairs(from_="nothere")
    with pytest.raises(ValueError, match="no pair with to = 'x'"):
        res.top_pairs(to=["T", "x"])
    with pytest.raises(ValueError, match="fdr"):
        res.top_pairs(adj="bonferroni")
    three = fit("three_levels")
    assert set(three.top_pairs(coef="conditionII", n=None)["level"]) == {"II"}
    with pytest.raises(ValueError, match="coef"):
        three.top_pairs(coef="IV")


def test_bind():
    res = fit("two_groups")
    every = res.bind()
    assert list(every.index) == res.image_ids and every.shape == (len(res.image_ids), 3 + len(res.cell_results))
    assert list(res.bind("tumour__T").columns) == ["imageID", "condition", "subject", "tumour__T"]
    one = res.bind(from_="tumour", to="T")
    np.testing.assert_array_equal(one["effect"], res.pairwise_assoc["tumour__T"])
    np.testing.assert_array_equal(one["weight"], res.image_weights["tumour__T"])
    assert one.loc[res.image_ids[3], "imageID"] == res.image_ids[3]
    with pytest.raises(KeyError, match="not found"):
        res.bind(from_="tumour", to="nothere")
    with pytest.raises(ValueError, match="one `from_`"):
        res.bind(from_=["tumour", "T"], to="T")


def test_radius_plot():
    import matplotlib

    matplotlib.use("Agg")
    res = fit("radii")
    ax = res.radius_plot(from_=["tumour", "T"], to="T")
    lines = ax.get_lines()
    assert [ln.get_label() for ln in lines] == ["tumour → T", "T → T"]
    rr = res.radius_results
    exp = rr[(rr["from"] == "tumour") & (rr["to"] == "T")].sort_values("r")
    np.testing.assert_allclose(lines[0].get_xdata(), exp["r"])
    np.testing.assert_allclose(lines[0].get_ydata(), -np.log10(exp["p_value"]))
    assert ax.get_ylabel() == "-log10 p at each radius, B vs A" and ax.get_ylim()[0] == 0
    with pytest.raises(ValueError, match="several radii"):
        fit("two_groups").radius_plot(from_="tumour", to="T")
    with pytest.raises(ValueError, match="no pair with to"):
        res.radius_plot(from_="tumour", to="macro")


def test_plot_image_several_images():
    import matplotlib

    matplotlib.use("Agg")
    ims = ["A02_1", "A01_1", "B01_1"]
    fig = spicyr.plot_image(CELLS, ims, "tumour", "T", r=20)
    axes = [a for a in fig.axes if a.get_visible()]
    assert [a.get_title() for a in axes] == ims  # in the order given
    assert len({a.get_xlim() for a in axes}) == 1 and len({a.get_ylim() for a in axes}) == 1  # one scale
    sel = CELLS[CELLS["imageID"].isin(ims)]
    assert axes[0].get_xlim()[0] <= sel["x"].min() and axes[0].get_xlim()[1] >= sel["x"].max()
    # one shared density scale, and the from cells of each image in its own panel
    assert len({im.get_clim() for a in axes for im in a.get_images()}) == 1
    for a, im in zip(axes, ims, strict=True):
        z = CELLS[(CELLS["imageID"] == im) & (CELLS["cellType"] == "tumour")]
        assert len(a.collections[1].get_offsets()) == len(z)
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["tumour", "T"]
    types = spicyr.plot_image(CELLS, ims[:2])
    assert sorted(t.get_text() for t in types.legends[0].get_texts()) == sorted(
        CELLS.loc[CELLS["imageID"].isin(ims[:2]), "cellType"].unique()
    )
    one = spicyr.plot_image(CELLS, [ims[0]])  # a list of one image: the image alone, as before
    assert one.get_title() == ims[0]
    with pytest.raises(ValueError, match="nothere"):
        spicyr.plot_image(CELLS, ["A01_1", "nothere"])


def test_calibration_summary_and_plot():
    import matplotlib

    matplotlib.use("Agg")
    cells = spicyr.datasets.example_cells()
    out = spicyr.calibration_check(cells, condition="condition", subject="patient", r=30, n_shuffles=3, rng=1)
    assert isinstance(out, pd.DataFrame) and isinstance(out, spicyr.SpicyCalibration)
    assert isinstance(out[out["shuffle"] == 1], spicyr.SpicyCalibration)
    s = out.summary()
    assert list(s.columns) == ["threshold", "expected", "percent"]
    assert s["threshold"].tolist() == [0.05, 0.01] and s["expected"].tolist() == pytest.approx([5, 1])
    exp = [out.loc[out["threshold"] == t, "percent"].mean() for t in (0.05, 0.01)]
    assert s["percent"].tolist() == pytest.approx(exp)
    ax = out.plot()
    assert ax.get_title() == "3 shuffles of the condition (red: expected)"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0.01", "0.05"]
    assert ax.get_ylim() == (0, max(15, out["percent"].max()))


def test_printed_summary():
    text = repr(fit("two_groups"))
    assert "Units: 14 patients (A 7, B 7) with 28 images" in text
    assert text.endswith("See top_pairs().")
    assert "Units: 28 images (I 8, II 10, III 10); no subject given: each image is a patient" in repr(
        fit("three_levels")
    )
    assert "Units: 14 patients with 28 images" in repr(fit("survival"))
