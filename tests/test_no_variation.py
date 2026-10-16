"""A radius where the pair's response has no variation (no `to` cell has a `from` neighbour, or every one has): the
unadjusted test is 0 / 0, no adjusted design is fitted there, and nothing of it enters the combination over radii."""

import numpy as np
import pandas as pd
import pytest

import spicyr
from spicyr import _core


def nv_cells(n_img=16, seed=3):
    """tumour -> T: no T cell within 2 of a tumour cell (B cells are), every T cell within 6 of one (far cells are not);
    at r = 4 the fraction differs by condition."""
    rng = np.random.default_rng(seed)
    grid = np.array([(x, y) for y in range(10, 100, 20) for x in range(10, 100, 20)], dtype=float)
    out = []
    for i in range(n_img):
        nt = int(rng.integers(12, 26))
        tum = grid[rng.choice(len(grid), nt, replace=False)]

        def near(n, lo, hi, tum=tum):
            k = rng.integers(0, len(tum), n)
            a = rng.uniform(0, 2 * np.pi, n)
            d = rng.uniform(lo, hi, n)
            return tum[k] + np.column_stack([d * np.cos(a), d * np.sin(a)])

        grp = "A" if i < n_img // 2 else "B"
        nT = int(rng.integers(30, 51))
        nc = round(nT * (rng.uniform(0.2, 0.5) if grp == "A" else rng.uniform(0.5, 0.8)))
        nB, nF = int(rng.integers(20, 41)), int(rng.integers(30, 51))
        xy = np.vstack(
            [tum, near(nc, 2.5, 3.5), near(nT - nc, 4.5, 5.5), near(nB, 0.2, 1), rng.uniform(300, 400, (nF, 2))]
        )
        types = ["tumour"] * nt + ["T"] * nT + ["B"] * nB + ["far"] * nF
        out.append(
            pd.DataFrame(
                {
                    "imageID": f"img{i:02d}",
                    "condition": grp,
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "cellType": types,
                    "age": 40.0 + (i * 7) % 13,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


CELLS = nv_cells()


def run(cells, r, combine="davies", **kw):
    return spicyr.spicy(
        cells,
        condition="condition",
        r=r,
        from_="tumour",
        to="T",
        covariates=["age"],
        adjust_abundance=True,
        combine=combine,
        **kw,
    )


@pytest.mark.parametrize("combine", ["davies", "maxT", "cauchy"])
def test_no_adjusted_design_at_empty_or_saturated_radius(combine):
    res = run(CELLS, [2.0, 4.0, 10.0], combine)
    rr = res.radius_results
    assert list(rr["adjusted_for"]) == ["none (no_variation)", "abundance+covariates", "none (no_variation)"]
    assert rr["p_value"][rr["r"] != 4].isna().all()
    # nothing of those radii enters the combination: the radius chosen is r = 4
    cr = res.cell_results
    assert list(cr["r"]) == [4.0]
    assert cr["p_value_best_radius"].iloc[0] == rr["p_value"][rr["r"] == 4].iloc[0]


def test_combination_equals_that_without_the_radius():
    res = run(CELLS, [2.0, 3.0, 4.0, 10.0])
    ok = run(CELLS, [3.0, 4.0])
    assert res.cell_results["p_value"].iloc[0] == ok.cell_results["p_value"].iloc[0]
    assert res.cell_results["unadjusted_p_value"].iloc[0] == ok.cell_results["unadjusted_p_value"].iloc[0]


def test_perturbed_covariate():
    res = run(CELLS, [2.0, 3.0, 4.0, 10.0])
    img = CELLS["imageID"].unique()
    for s in (1, 2, 3):
        rng = np.random.default_rng(s)
        age = pd.Series((40.0 + (np.arange(len(img)) * 7) % 13) * (1 + 1e-10 * rng.standard_normal(len(img))), img)
        cp = CELLS.assign(age=CELLS["imageID"].map(age))
        rp = run(cp, [2.0, 3.0, 4.0, 10.0])
        assert list(rp.radius_results["adjusted_for"]) == list(res.radius_results["adjusted_for"])
        np.testing.assert_allclose(rp.cell_results["p_value"], res.cell_results["p_value"], rtol=1e-8)


@pytest.mark.parametrize("r", [2.0, 10.0])
def test_one_radius(r):
    x = run(CELLS, r).cell_results
    assert list(x["adjusted_for"]) == ["none (no_variation)"]
    assert x["p_value"].isna().all()


def test_design_test_reports_no_variation():
    rng = np.random.default_rng(1)
    n = 12
    E, nn = rng.uniform(1, 3, n), rng.uniform(1, 3, n)
    rows = {"img": np.arange(n, dtype=np.int32), "O": E - nn, "E": E, "n": nn, "v": rng.uniform(0.5, 1, n)}
    g = np.repeat([0, 1], n // 2)
    Z = np.column_stack([g == 0, g == 1, rng.standard_normal(n)]).astype(float)
    C = np.array([[-1.0, 1.0, 0.0]])
    unit = np.arange(n, dtype=np.int32)
    a = _core.design_tests(rows, unit, n, Z, C, 1, 0.0, False)
    assert not a[0]["ok"] and a[0]["reason"] == "no_variation"
    rows["O"] = rows["O"] + rng.normal(0, 1e-6, n)  # a little variation is tested
    assert _core.design_tests(rows, unit, n, Z, C, 1, 0.0, False)[0]["ok"]


@pytest.mark.parametrize("r", [2.0, 10.0])
def test_levels_and_survival_not_tested(r):
    cells = nv_cells(24)
    img = sorted(cells["imageID"].unique())
    cells["c3"] = cells["imageID"].map({im: "ABC"[(j + 1) % 3] for j, im in enumerate(img)})
    with pytest.raises(ValueError, match="no pair could be tested"):
        spicyr.spicy(cells, condition="c3", r=r, from_="tumour", to="T", adjust_abundance=True)
    rng = np.random.default_rng(2)
    cells["time"] = cells["imageID"].map(dict(zip(img, rng.exponential(1, len(img)), strict=True)))
    cells["event"] = cells["imageID"].map(dict(zip(img, (rng.random(len(img)) < 0.7).astype(int), strict=True)))
    with pytest.raises(ValueError, match="no pair could be tested"):
        spicyr.spicy(cells, condition="time", survival="event", r=r, from_="tumour", to="T", adjust_abundance=True)
    # at r = 4 the pair varies and is tested
    assert len(spicyr.spicy(cells, condition="c3", r=4.0, from_="tumour", to="T", adjust_abundance=True).cell_results)
