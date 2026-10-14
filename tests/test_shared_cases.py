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
    if kw["condition"] == "os":
        kw.update(condition="time", survival="event")
    return spicyr.spicy(CELLS, **kw)


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_matches_r(name):
    res = run(CASES[name])
    exp = pd.read_csv(HERE / f"{name}.csv", index_col=0)
    got = res.cell_results
    assert sorted(got.index) == sorted(exp.index)
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
            assert (got[col].astype(str).to_numpy() == exp[col].astype(str).to_numpy()).all(), f"{name}: {col}"
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


def test_maxt_radius_limit():
    """The exact max-T takes up to six radii; Davies takes a fine grid."""
    with pytest.raises(ValueError, match="davies"):
        spicyr.spicy(CELLS, condition="condition", subject="patient", r=list(range(5, 45, 5)), from_="tumour", to="T")


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
