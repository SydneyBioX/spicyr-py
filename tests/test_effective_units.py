"""The condition test's two rules: at least min(3, m_g / 2) effective patients per condition (m_g its patients with
rows at the radius), and one dispersion shared by the conditions.

One patient per image; image 1 (condition A) has 30 times the cells of the others at the same density, so at r = 5,
where the patients do not differ (tau2 = 0), it carries nearly all of A's weight and A has under 2 effective patients.
In every image a patient-specific fraction of the T cells sits 8-10 units from a tumour cell: from r = 10 the patients
differ, tau2 > 0 caps each patient's weight, and A has about 7. The cells and spicyR's results are written by spicyR
(phipool_dev/make_phipool_case.R, spicyR 1.99.29).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import spicyr
from spicyr import _cell, _core

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "neff_cells.csv")
CELLS3 = pd.read_csv(HERE / "neff_cells3.csv")
EXPECTED = json.loads((HERE / "neff_expected.json").read_text())
R_TO_PY = {"adjustAbundance": "adjust_abundance"}
# spicyR's results were written on x86-64 Linux; where long double is double (macOS arm64, MSVC) sums in the core round
# differently, so there they are compared as the shared cases are
EXTENDED = np.finfo(np.longdouble).nmant > 52


def same(got, exp, msg=""):
    if EXTENDED:
        np.testing.assert_array_equal(got, exp, err_msg=msg)
    else:
        np.testing.assert_allclose(got, exp, rtol=1e-7, atol=1e-10, equal_nan=True, err_msg=msg)


def run(cells=CELLS, **kw):
    return spicyr.spicy(cells, condition="condition", from_="tumour", to="T", **kw)


def as_list(v):
    return v if isinstance(v, list) else [v]


def num(v):
    return np.array([np.nan if x is None else float(x) for x in as_list(v)])


def rows_at(r, effect="allocation", cells=CELLS):
    ctx = _cell.cell_context(cells, "condition", None)
    g = _cell.cell_graph(ctx, [("tumour", "T")], r=r, label_clustering=False, effect=effect)
    g.image_heterogeneity = True
    return ctx, _cell.cell_rows(ctx, g, "tumour", "T")


def per_unit(rows):
    """Per patient: information J, score s and condition, from the image rows."""
    u = rows["unit"]
    J = np.bincount(u, rows["n"] ** 2 / rows["v"])
    s = np.bincount(u, rows["n"] * (rows["O"] - rows["E"]) / rows["v"])
    g = np.full(len(J), -1)
    g[u] = rows["group"]
    keep = np.bincount(u, minlength=len(J)) > 0
    return J[keep], s[keep], g[keep]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_python_equals_r(name):
    exp = EXPECTED[name]
    kw = {R_TO_PY.get(k, k): v for k, v in exp["args"].items()}
    kw["r"] = [float(v) for v in kw["r"]] if isinstance(kw["r"], list) else float(kw["r"])
    if "covariates" in kw:
        kw["covariates"] = as_list(kw["covariates"])
    res = run(**kw)
    cr = res.cell_results
    for col in ("p_value", "unadjusted_p_value", "r", "excess_difference", "se", "df"):
        if exp[col] is None:
            assert col not in cr.columns
            continue
        same(cr[col].to_numpy(float), num(exp[col]), f"{name}: {col}")
    if exp["adjusted_for"] is not None:
        assert list(cr["adjusted_for"]) == as_list(exp["adjusted_for"])
    if exp["radius_p"] is not None:
        rr = res.radius_results
        np.testing.assert_array_equal(rr["r"].to_numpy(float), num(exp["radius_r"]))
        same(rr["p_value"].to_numpy(float), num(exp["radius_p"]), f"{name}: radius p")
        if exp["radius_adjusted_for"] is not None:
            assert list(rr["adjusted_for"]) == as_list(exp["radius_adjusted_for"])


def test_rules():
    for allocation in (True, False):
        a = _core.test_rules(allocation)
        assert a["pooled_dispersion"] and a["min_effective_units"] and a["min_n_eff"] == 3


@pytest.mark.parametrize("r", [5.0, 12.0])
def test_effective_patients_by_hand(r):
    ctx, rows = rows_at(r)
    u = _core.excess_test(rows, rows["unit"], rows["group"], len(ctx.unit_labels), True, "cr2", True)
    J, _, g = per_unit(rows)
    w = J / (1 + u["tau2"] * J)
    kish = [w[g == k].sum() ** 2 / (w[g == k] ** 2).sum() for k in (0, 1)]
    np.testing.assert_allclose(u["n_eff"], kish, rtol=1e-12)
    np.testing.assert_allclose(_core.effective_units(rows, rows["unit"], rows["group"], 2, u["tau2"]), kish, rtol=1e-12)
    np.testing.assert_array_equal(u["n_units"], [8, 8])
    few = min(kish) < 3
    assert few == (r == 5.0)
    x = run(r=r).cell_results
    assert x["p_value"].isna().all() == few
    x = run(r=r, covariates=["age"]).cell_results
    assert list(x["adjusted_for"]) == ["none (few_effective_units)" if few else "covariates"]
    if few:
        assert x["p_value"].isna().all() and x["unadjusted_p_value"].isna().all()


@pytest.mark.parametrize("combine", ["davies", "maxT", "cauchy"])
@pytest.mark.parametrize("adjust", [False, True])
def test_left_out_of_the_combination(combine, adjust):
    a = run(r=[5.0, 12.0, 15.0], combine=combine, adjust_abundance=adjust)
    b = run(r=[12.0, 15.0], combine=combine, adjust_abundance=adjust)
    rr = a.radius_results
    assert rr["p_value"][rr["r"] == 5].isna().all() and rr["p_value"][rr["r"] != 5].notna().all()
    if adjust:
        assert list(rr["adjusted_for"][rr["r"] == 5]) == ["none (few_effective_units)"]
        assert a.cell_results["unadjusted_p_value"].iloc[0] == b.cell_results["unadjusted_p_value"].iloc[0]
    assert a.cell_results["p_value"].iloc[0] == b.cell_results["p_value"].iloc[0]
    assert a.cell_results["r"].iloc[0] == b.cell_results["r"].iloc[0]


def test_count_has_the_same_rules():
    rules = _core.test_rules(False)
    ctx, rows = rows_at(5.0, "count")
    u = _core.excess_test(
        rows, rows["unit"], rows["group"], len(ctx.unit_labels), True, "cr2", rules["pooled_dispersion"]
    )
    x = run(r=5.0, effect="count").cell_results
    assert x["p_value"].isna().iloc[0] == (
        rules["min_effective_units"] and (u["n_eff"] < np.minimum(3, u["n_units"] / 2)).any()
    )
    assert x["se"].iloc[0] == u["se"]


def test_more_than_two_conditions():
    for r, tested in ((5.0, False), (12.0, True)):
        ctx = _cell.cell_context(CELLS3, "condition", None)
        g = _cell.cell_graph(ctx, [("tumour", "T")], r=r, label_clustering=False)
        g.image_heterogeneity = True
        o = _cell.cell_pair_test(ctx, g, "tumour", "T", True, "cr2", False)
        assert o["ok"] == tested
        if not tested:
            assert o["reason"] == "few_effective_units"


def test_pooled_dispersion_by_hand():
    ctx, rows = rows_at(12.0)
    m = len(ctx.unit_labels)
    u0 = _core.excess_test(rows, rows["unit"], rows["group"], m, True, "cr2", False)  # CR2: tau2 and df
    J, s, g = per_unit(rows)
    beta, V, EV, mg = [], [], [], []
    for k in (0, 1):
        i = g == k
        a = 1 / (1 + u0["tau2"] * J[i])
        B = (a * J[i]).sum()
        b = (a * s[i]).sum() / B
        r = s[i] - b * J[i]
        h = a * J[i] / B
        beta.append(b)
        V.append((a**2 * r**2 / (B**2 * (1 - h))).sum())
        EV.append(1 / B)
        mg.append(i.sum())
    np.testing.assert_allclose(u0["se"] ** 2, sum(V), rtol=1e-10)
    phi = sum(mk * v / e for mk, v, e in zip(mg, V, EV, strict=True)) / sum(mg)
    se = np.sqrt(phi * sum(EV))
    dif = beta[1] - beta[0]
    x = run(r=12.0).cell_results.iloc[0]
    np.testing.assert_allclose(
        [x["excess_difference"], x["se"], x["df"], x["p_value"]],
        [dif, se, u0["df"], 2 * stats.t.sf(abs(dif / se), u0["df"])],
        rtol=1e-10,
    )
    # Hartung-Knapp keeps its own variance
    hk = run(r=12.0, variance="hartung_knapp").cell_results.iloc[0]
    uh = _core.excess_test(rows, rows["unit"], rows["group"], m, True, "hartung_knapp", False)
    np.testing.assert_allclose([hk["se"], hk["df"], hk["p_value"]], [uh["se"], uh["df"], uh["p"]], rtol=1e-12)


def test_pooled_dispersion_adjusted_by_hand():
    """The adjusted design (one image per patient): each condition's coefficient's CR2 / model variance ratio, pooled
    by patients, scales the model variance of every contrast."""
    ctx, rows = rows_at(12.0)
    xa = run(r=12.0, covariates=["age"]).cell_results.iloc[0]
    x = run(r=12.0).cell_results.iloc[0]
    tau2 = xa["tau2"]
    age = CELLS.groupby("imageID")["age"].first().reindex(np.asarray(ctx.image_labels)[rows["img"]]).to_numpy(float)
    g = rows["group"]
    Z = np.column_stack([g == 0, g == 1, age - age.mean()]).astype(float)
    w = 1 / (tau2 * rows["n"] ** 2 + rows["v"])  # one row per patient: V = tau2 n^2 + v
    X = np.sqrt(w)[:, None] * rows["n"][:, None] * Z
    y = np.sqrt(w) * (rows["O"] - rows["E"])
    B = np.linalg.inv(X.T @ X)
    theta = B @ X.T @ y
    e = y - X @ theta
    h = np.einsum("ij,jk,ik->i", X, B, X)

    def cr2(c):
        return ((X @ (B @ c)) * e / np.sqrt(1 - h)) ** 2

    phi = sum((g == k).sum() * cr2(np.eye(3)[k]).sum() / B[k, k] for k in (0, 1)) / len(g)
    c = np.array([-1.0, 1.0, 0.0])
    se = np.sqrt(phi * c @ B @ c)
    est = c @ theta
    np.testing.assert_allclose([xa["excess_difference"], xa["se"]], [est, se], rtol=1e-8)
    np.testing.assert_allclose(xa["age_effect"], theta[2], rtol=1e-8)
    assert xa["unadjusted_p_value"] == pytest.approx(x["p_value"], rel=1e-12)


def test_the_bound_is_min_3_half_the_patients():
    rules = _core.test_rules(True)
    assert not _cell._few_effective([1.5, 3], [3, 8], rules) and _cell._few_effective([1.49, 3], [3, 8], rules)
    assert not _cell._few_effective([2.5, 3], [5, 6], rules) and _cell._few_effective([2.9, 3], [6, 6], rules)
    assert _cell._few_effective([3, 2.9], [8, 9], rules)


def test_a_condition_of_3_patients_is_tested():
    cells = CELLS[CELLS["imageID"].isin([f"img{i:02d}" for i in range(2, 8)])]
    ctx, rows = rows_at(12.0, cells=cells)
    u = _core.excess_test(rows, rows["unit"], rows["group"], len(ctx.unit_labels), True, "cr2", True)
    np.testing.assert_array_equal(u["n_units"], [3, 3])
    assert ((u["n_eff"] >= 1.5) & (u["n_eff"] < 3)).all()
    x = run(cells, r=12.0).cell_results
    assert x["p_value"].notna().all()
    np.testing.assert_allclose(x["se"].iloc[0], u["se"], rtol=1e-12)
