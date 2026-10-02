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
R_TO_PY = {"labelClustering": "label_clustering", "from": "from_", "adjustAbundance": "adjust_abundance"}


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
