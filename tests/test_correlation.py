"""The correlation between the tests of a result (SpicyResults.test_correlation), rebuilt from pairwise_assoc,
image_weights and condition_information: spicyr reproduces spicyR's testCorrelation() on the shared cases, and the
rebuilt tests are the core's."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spicyr
from spicyr import _cell, _core

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "cells.csv")
WEIGHTS = pd.read_csv(HERE / "weights.csv")
CASES = json.loads((HERE / "cases.json").read_text())
WCASES = json.loads((HERE / "weights_cases.json").read_text())
R_TO_PY = {"from": "from_", "adjustAbundance": "adjust_abundance"}
SURV = {"os": "time", "os_na": "time_na", "os_few": "time_few"}


def run(args, weights=False):
    kw = {R_TO_PY.get(k, k): (v if not isinstance(v, list) or len(v) > 1 else v[0]) for k, v in args.items()}
    if kw["condition"] in SURV:
        kw.update(condition=SURV[kw["condition"]], survival="event")
    return spicyr.spicy(CELLS, cell_weights=WEIGHTS if weights else None, **kw)


def close(got, exp, name):
    assert list(got.index) == list(exp.index) and list(got.columns) == list(map(str, exp.columns)), name
    np.testing.assert_allclose(
        got.to_numpy(float), exp.to_numpy(float), rtol=1e-7, atol=1e-10, equal_nan=True, err_msg=name
    )


@pytest.mark.parametrize("name,weights", [(n, False) for n in sorted(CASES)] + [(n, True) for n in sorted(WCASES)])
def test_case_matches_r(name, weights):
    """condition_information and both correlations as spicyR gives them; none where spicyR has none."""
    res = run((WCASES if weights else CASES)[name], weights)
    info = HERE / f"{name}_information.csv"
    if not info.exists():
        assert res.condition_information is None
        with pytest.raises(ValueError, match="two conditions"):
            res.test_correlation()
        return
    close(res.condition_information, pd.read_csv(info, index_col=0), f"{name}: information")
    cr2 = HERE / f"{name}_correlation_cr2.csv"
    if not cr2.exists():
        with pytest.raises(ValueError):
            res.test_correlation()
        return
    close(res.test_correlation(), pd.read_csv(cr2, index_col=0), f"{name}: cr2")
    perm = pd.read_csv(HERE / f"{name}_correlation_permutation.csv", index_col=0)
    close(res.test_correlation("permutation", n_permutations=200), perm, f"{name}: permutation")


def core_tests(res):
    """Every pair's two-condition test through the internal steps, keeping the core's per-patient quantities."""
    ctx = _cell.cell_context(CELLS, "condition", "patient")
    tab = res.cell_results
    pairs = list(zip(tab["from"], tab["to"], strict=True))
    g = _cell.cell_graph(ctx, pairs, r=30, label_clustering=False)
    g.image_heterogeneity = True
    return ctx, [_cell.cell_pair_test(ctx, g, f, t, True, "cr2", False) for f, t in pairs]


def test_cr2_is_the_cross_product_of_the_core_influences():
    res = spicyr.spicy(CELLS, condition="condition", subject="patient", r=30)
    tab = res.cell_results
    tested = tab["p_value"].notna().to_numpy()
    ctx, fits = core_tests(res)
    psi = np.array([np.asarray(o["test"]["influence"], dtype=float) for o in fits])
    # the squares of a test's influences sum to its se^2
    np.testing.assert_allclose(np.sqrt((psi**2).sum(axis=1))[tested], tab["se"].to_numpy()[tested], rtol=1e-12)
    u = psi[tested] / np.sqrt((psi[tested] ** 2).sum(axis=1))[:, None]
    R0 = u @ u.T
    np.fill_diagonal(R0, 1)
    R = res.test_correlation()
    np.testing.assert_allclose(R.to_numpy()[np.ix_(tested, tested)], R0, rtol=0, atol=1e-12)
    assert R.to_numpy()[~tested].size == 0 or np.isnan(R.to_numpy()[~tested]).all()
    assert (R.index == tab.index).all() and (R.columns == tab.index).all()
    np.testing.assert_array_equal(R.to_numpy(), R.to_numpy().T)
    assert (np.diag(R.to_numpy())[tested] == 1).all()
    # condition_information: each patient's weight J / (1 + tau2 J)
    unit = pd.Series(res.subject)
    for k in np.flatnonzero(tested)[:5]:
        o = fits[k]["test"]
        W = o["unit_info"] / (1 + o["tau2"] * o["unit_info"])
        key = tab.index[k]
        w = res.image_weights[key] * res.condition_information.loc[key, res.condition].to_numpy()
        got = pd.Series(w).groupby(unit).sum().reindex(ctx.unit_labels)
        np.testing.assert_allclose(got.to_numpy(), W, rtol=1e-12)


def test_permutation_recomputes_the_tests_exactly():
    res = spicyr.spicy(CELLS, condition="condition", subject="patient", r=30)
    tab = res.cell_results
    tested = tab["p_value"].notna().to_numpy()
    codes, units = pd.factorize(pd.Series(res.subject))
    group = np.array([res.levels.index(c) for c in res.condition], dtype=np.int32)
    obs = group[[np.flatnonzero(codes == u)[0] for u in range(len(units))]]
    y = np.array([res.pairwise_assoc[k] for k in tab.index])
    w = np.array([res.image_weights[k] for k in tab.index])
    y[~tested], w[~tested] = np.nan, np.nan
    _, _, _, z = _core.test_correlation(
        y,
        w,
        codes.astype(np.int32),
        group,
        len(units),
        res.condition_information.loc[tab.index].to_numpy(),
        2,
        np.concatenate([obs, obs]).astype(np.int32),
    )
    from scipy.stats import norm

    z0 = -norm.ppf(tab["p_value"].to_numpy() / 2) * np.sign(tab["excess_difference"].to_numpy())
    np.testing.assert_allclose(z[tested, 0], z0[tested], rtol=1e-10)
    R1 = res.test_correlation("permutation", n_permutations=100)
    pd.testing.assert_frame_equal(R1, res.test_correlation("permutation", n_permutations=100, cores=2))


def test_refused_where_the_tests_cannot_be_rebuilt():
    kw = {"condition": "condition", "subject": "patient", "from_": "tumour"}
    with pytest.raises(ValueError, match="unadjusted"):
        spicyr.spicy(CELLS, r=30, adjust_abundance=True, **kw).test_correlation()
    with pytest.raises(ValueError, match="one radius"):
        spicyr.spicy(CELLS, r=[20, 30], **kw).test_correlation()
    with pytest.raises(ValueError, match="cr2"):
        spicyr.spicy(CELLS, r=30, variance="hartung_knapp", **kw).test_correlation()
    res = spicyr.spicy(CELLS, r=30, **kw)
    with pytest.raises(ValueError, match="n_permutations"):
        res.test_correlation("permutation", n_permutations=1)
    res.test_correlation(n_permutations=None)  # not used by "cr2"
    with pytest.raises(ValueError, match="cores"):
        res.test_correlation(cores=0)
    with pytest.raises(ValueError, match="method"):
        res.test_correlation("pearson")
    bad = spicyr.spicy(CELLS, r=30, **kw)
    bad.image_weights = spicyr.spicy(CELLS, r=20, **kw).image_weights
    with pytest.raises(ValueError, match="do not give the tests"):
        bad.test_correlation()


def test_missing_condition_and_subject():
    """Patients with a missing condition take no part; an image with a missing subject is a patient, as in spicy()."""
    cm = CELLS.copy()
    cm.loc[cm["patient"].isin(["A02", "B03"]), "condition"] = np.nan
    res = spicyr.spicy(cm, condition="condition", subject="patient", r=30, from_="tumour")
    tested = res.cell_results["p_value"].notna().to_numpy()
    for m in ("cr2", "permutation"):
        R = res.test_correlation(m, n_permutations=100).to_numpy()
        assert (np.diag(R)[tested] == 1).all()
        np.testing.assert_array_equal(R, R.T)
    cs = CELLS.copy()
    cs.loc[cs["patient"] == "A02", "patient"] = np.nan
    res = spicyr.spicy(cs, condition="condition", subject="patient", r=30, from_="tumour")
    R = res.test_correlation().to_numpy()
    assert (np.diag(R)[res.cell_results["p_value"].notna().to_numpy()] == 1).all()
