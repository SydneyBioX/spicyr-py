"""The skewness correction of the two-condition CR2 t (excess_skewness, skew_corrected_t in the core): a plain-Python
reference of kappa from the units' weights and residuals, Hall's transformation, and when it is (not) applied. As
spicyR's tests/testthat/test-skewness.R and helper-skewness.R."""

import numpy as np
from scipy import stats

from spicyr import _core


def ref_shrunk_skewness(z, h, g=None):
    """The skewness of the two-condition test, shrunk by its own uncertainty, from the units' standardised residuals z
    and leverages h (from the definitions): g = (sum z^3 / sum (1 - h)(1 - 2h)) / (sum z^2 / sum (1 - h))^1.5; its
    delta-method variance V = mean(IF^2) / n, IF = t^3 - 3t - g - 1.5 g (t^2 - 1), t = z / sigma, n = sum (1 - h) + 1;
    shrunk g^3 / (g^2 + V); 0 when a condition (g, if given) has fewer than 3 units."""
    z, h = np.asarray(z, float), np.asarray(h, float)
    if g is not None:
        _, counts = np.unique(g, return_counts=True)
        if len(counts) < 2 or (counts < 3).any():
            return 0.0
    d3 = ((1 - h) * (1 - 2 * h)).sum()
    if not d3 > 0:
        return 0.0
    s2 = (z**2).sum() / (1 - h).sum()
    g1 = ((z**3).sum() / d3) / s2**1.5
    t = z / np.sqrt(s2)
    IF = t**3 - 3 * t - g1 - 1.5 * g1 * (t**2 - 1)
    V = (IF**2).mean() / ((1 - h).sum() + 1)
    return g1**3 / (g1**2 + V)


def ref_hall(t, k):
    """Hall's transformation of t for skewness kappa."""
    return t + k * t**2 / 3 + k**2 * t**3 / 27 + k / 6


def p_uncorrected(u):
    return 2 * stats.t.sf(abs(u["difference"] / u["se"]), u["df"])


def sk_rows(m0, m1, seed, skew=True, images=2):
    rng = np.random.default_rng(seed)
    m = m0 + m1
    unit = np.repeat(np.arange(m), images)
    N = len(unit)
    n = np.round(rng.uniform(20, 200, N))
    v = n * rng.uniform(0.5, 2, N)
    eff = (rng.exponential(size=m) if skew else rng.normal(size=m))[unit]
    O = n * (0.1 * eff + rng.normal(0, 0.05, N))
    return {
        "img": np.arange(N, dtype=np.int32),
        "O": O,
        "E": np.zeros(N),
        "n": n,
        "v": v,
        "unit": unit.astype(np.int32),
        "group": (unit >= m0).astype(np.int32),
    }


def excess(rows, group=None, frailty=True, variance="cr2", pooled=False):
    g = rows["group"] if group is None else group
    return _core.excess_test(rows, rows["unit"], g, int(rows["unit"].max()) + 1, frailty, variance, pooled)


def sk_kappa_ref(rows, u):
    """kappa from what the test returns: unit summaries (ybar = s / J), information J and tau2 give W = J / (1 + tau2 J)."""
    ids = np.unique(rows["unit"])
    g = np.array([rows["group"][rows["unit"] == i][0] for i in ids])
    J = np.asarray(u["unit_info"])[ids]
    ybar = np.asarray(u["unit_summary"])[ids]
    W = J / (1 + u["tau2"] * J)
    S = np.array([W[g == k].sum() for k in (0, 1)])
    beta = np.array([(W * ybar)[g == k].sum() for k in (0, 1)]) / S
    e = ybar - beta[g]
    h = W / S[g]
    z = np.sqrt(W) * e
    w15 = np.array([(W**1.5)[g == k].sum() for k in (0, 1)])
    return ref_shrunk_skewness(z, h, g) * (w15[1] / S[1] ** 3 - w15[0] / S[0] ** 3) / (1 / S[0] + 1 / S[1]) ** 1.5


def test_kappa_and_corrected_p_match_reference():
    for s in range(1, 5):
        rows = sk_rows(6 + 3 * s, 30, s)
        u = excess(rows)
        assert u["ok"]
        k = sk_kappa_ref(rows, u)
        np.testing.assert_allclose(u["skewness"], k, rtol=1e-10)
        t = ref_hall(u["difference"] / u["se"], k)
        np.testing.assert_allclose(u["t"], t, rtol=1e-10)
        np.testing.assert_allclose(u["p"], 2 * stats.t.sf(abs(t), u["df"]), rtol=1e-10)
        np.testing.assert_allclose((np.asarray(u["influence"]) ** 2).sum(), u["se"] ** 2, rtol=1e-10)  # se unchanged


def test_small_second_condition_gives_positive_kappa_and_swapping_flips_it():
    rows = sk_rows(40, 8, 11)
    u = excess(rows)
    w = excess(rows, group=(1 - rows["group"]).astype(np.int32))
    assert u["skewness"] > 0
    np.testing.assert_allclose(w["skewness"], -u["skewness"], rtol=1e-12)
    np.testing.assert_allclose(w["p"], u["p"], rtol=1e-12)


def test_equal_conditions_with_equal_weights_give_no_correction():
    m = 10
    unit = np.arange(2 * m, dtype=np.int32)
    rng = np.random.default_rng(3)
    rows = {
        "img": unit,
        "O": rng.exponential(size=2 * m),
        "E": np.zeros(2 * m),
        "n": np.ones(2 * m),
        "v": np.ones(2 * m),
        "unit": unit,
        "group": (unit >= m).astype(np.int32),
    }
    u = excess(rows, frailty=False)
    assert u["skewness"] == 0
    np.testing.assert_allclose(u["p"], p_uncorrected(u), rtol=1e-14)


def test_hartung_knapp_and_pooled_dispersion_are_not_corrected():
    rows = sk_rows(30, 8, 5)
    for u in (excess(rows, variance="hartung_knapp"), excess(rows, pooled=True)):
        assert u["skewness"] == 0
        np.testing.assert_allclose(u["p"], p_uncorrected(u), rtol=1e-12)


def test_two_units_per_condition_give_no_correction():
    rows = {
        "img": np.arange(4, dtype=np.int32),
        "O": np.array([1.0, 3, 2, 9]),
        "E": np.zeros(4),
        "n": np.ones(4),
        "v": np.ones(4),
        "unit": np.arange(4, dtype=np.int32),
        "group": np.array([0, 0, 1, 1], dtype=np.int32),
    }
    assert excess(rows, frailty=False)["skewness"] == 0


def test_two_units_with_nearly_equal_weights_give_no_unbounded_kappa():
    for eps in (0.1, 0.01, 0.001):
        n = np.array([1, 1 + eps, 9, 9 + eps])
        rows = {
            "img": np.arange(4, dtype=np.int32),
            "O": np.array([0.3, -0.1, 0.05, -0.02]) * n,
            "E": np.zeros(4),
            "n": n,
            "v": n.copy(),
            "unit": np.arange(4, dtype=np.int32),
            "group": np.array([0, 0, 1, 1], dtype=np.int32),
        }
        u = excess(rows, frailty=False)
        assert u["skewness"] == 0
        np.testing.assert_allclose(u["p"], p_uncorrected(u), rtol=1e-12)


def test_hall_transformation_is_monotone_and_odd():
    t = np.arange(-4000, 4001) / 100
    for k in (-1, -0.2, 0, 0.05, 0.3, 2):
        assert (np.diff(ref_hall(t, k)) >= 0).all()
        np.testing.assert_allclose(ref_hall(-t, -k), -ref_hall(t, k), rtol=1e-12, atol=1e-12)


def test_skewness_is_shrunk_by_its_own_uncertainty():
    # (that the package uses this shrinkage is checked against sk_kappa_ref above)
    rng = np.random.default_rng(12)
    # normal residuals: the delta-method variance is about 6 / n
    z = rng.normal(size=100_000)
    t = z / np.sqrt((z**2).mean())
    g = (t**3).mean()
    V = ((t**3 - 3 * t - g - 1.5 * g * (t**2 - 1)) ** 2).mean()
    np.testing.assert_allclose(V, 6, rtol=0.15)
    # the same skewed sample, replicated: the raw skewness is the same, its variance falls, so less shrinkage
    x = rng.exponential(size=12)
    f = []
    for r in (1, 4, 16, 256):
        z = np.tile(x - x.mean(), r)
        h = np.full(len(z), 1 / len(z))
        raw = ((z**3).sum() / ((1 - h) * (1 - 2 * h)).sum()) / ((z**2).sum() / (1 - h).sum()) ** 1.5
        f.append(ref_shrunk_skewness(z, h) / raw)
    assert (np.diff(f) > 0).all()
    assert f[3] > 0.95
