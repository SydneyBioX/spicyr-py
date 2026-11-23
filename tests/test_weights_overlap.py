"""Cell weights where the labels share cells (a label with itself, a group and one of its subtypes): random labelling
of all the image's cells, each cell keeping its pair of weights, with exact moments, as in spicyR 1.99.60
(tests/testthat/test-cellWeights.R). The core's moments against enumeration on tiny images, several pairs at once
against one at a time, and the pairs that keep the shuffle of the `to` weights against spicyr 1.99.44."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import spicyr
from spicyr import _core

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "cells.csv")
WEIGHTS = pd.read_csv(HERE / "weights.csv")


def arrangements(cats):
    """All distinct arrangements of a multiset of category codes (one row each)."""
    cats = list(cats)
    if len(cats) <= 1:
        return [cats]
    out = []
    for v in dict.fromkeys(cats):
        rest = list(cats)
        rest.remove(v)
        out += [[v, *a] for a in arrangements(rest)]
    return out


def neighbours(x, y, r):
    """Each cell's neighbours within r, itself excluded."""
    return [
        np.flatnonzero(((x - x[i]) ** 2 + (y - y[i]) ** 2 <= r**2) & (np.arange(len(x)) != i)) for i in range(len(x))
    ]


def overlap_count(nb, F, T):
    """The number of TO cells with a FROM cell other than themselves within r (0/1 vectors F and T)."""
    return sum(bool(T[i]) and bool(np.any(F[nb[i]])) for i in range(len(nb)))


def one_image(x, y, r):
    d = _core.Dataset(x, y, np.zeros(len(x), dtype=np.int32), np.array([0, len(x)], dtype=np.int32), 1)
    d.build_radius_index(r)
    return d


def test_the_random_labelling_moments_of_labels_that_share_cells_are_exact():
    rng = np.random.default_rng(1)

    def check(N, r, nF, nT, nFT, self=None):
        self = (nF == nT and nFT == nT) if self is None else self
        x, y = rng.uniform(size=N), rng.uniform(size=N)
        d = one_image(x, y, r)
        nb = neighbours(x, y, r)
        # 1 = FROM and TO, 2 = FROM only, 3 = TO only, 4 = neither
        A = np.array(arrangements([1] * nFT + [2] * (nF - nFT) + [3] * (nT - nFT) + [4] * (N - nF - nT + nFT)))
        O = np.array([overlap_count(nb, lab <= 2, np.isin(lab, (1, 3))) for lab in A], dtype=float)
        E, V = d.overlap_moments([nF], [nT], [nFT], [int(self)], 0, 50)
        return [E[0] - O.mean(), V[0] - (np.mean(O**2) - O.mean() ** 2)]

    # self-pairs, a group whose cells are all of one subtype (the same counts, not a self-pair), a group and its
    # subtype both ways, partial and no overlap, a single FROM cell, every cell labelled
    d = [
        check(9, 0.4, 3, 3, 3),
        check(10, 0.6, 4, 4, 4),
        check(9, 0.4, 3, 3, 3, self=False),
        check(9, 0.5, 2, 5, 2),
        check(9, 0.5, 5, 2, 2),
        check(10, 0.35, 3, 6, 3),
        check(9, 0.5, 3, 4, 1),
        check(9, 0.5, 2, 3, 0),
        check(10, 0.5, 1, 5, 1),
        check(8, 0.5, 8, 3, 3),
        check(8, 0.5, 3, 8, 3),
    ]
    assert np.max(np.abs(d)) < 1e-9


def test_soft_and_count_moments_under_permutation_of_each_cells_pair_of_weights_are_exact():
    from itertools import permutations

    def check(N, r, a, b, seed):
        rng = np.random.default_rng(seed)
        x, y = rng.uniform(size=N), rng.uniform(size=N)
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = one_image(x, y, r)
        nb = neighbours(x, y, r)
        P = np.array(list(permutations(range(N))))
        out = []
        for alloc in (True, False):

            def stat(pi, alloc=alloc):
                A, B = a[pi], b[pi]
                h = np.array([1 - np.prod(1 - A[n]) if alloc else np.sum(A[n]) for n in nb])
                return np.sum(B * h), np.sum(h)

            S = np.array([stat(pi) for pi in P])
            m = d.overlap_soft_rows(a, b, 1, alloc)
            obs = stat(np.arange(N))
            if alloc:
                out += [m[0] - obs[0], m[1] - S[:, 0].mean(), m[2] - np.mean((S[:, 0] - S[:, 0].mean()) ** 2)]
            else:  # the count effect: O - p L, with E = p L
                p = m[1] / obs[1]
                D = S[:, 0] - p * S[:, 1]
                out += [m[0] - obs[0], D.mean(), m[2] - np.mean((D - D.mean()) ** 2)]
            assert m[3] == pytest.approx(b.sum(), abs=1e-12)
        return out

    rng = np.random.default_rng(10)
    w = rng.uniform(size=7)
    d = [
        check(7, 0.5, rng.uniform(size=7), rng.uniform(size=7), 1),  # soft, partly shared
        check(7, 0.6, w, w, 2),  # a soft label with itself
        check(7, 0.5, [1, 1, 0.7, 0.2, 0, 0, 0.4], [0.5, 1, 0, 0.3, 0, 0.8, 0.4], 3),  # some `from` weights 1
        check(7, 0.45, [1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], 4),  # 0/1 group -> subtype
        check(7, 0.9, rng.uniform(size=7), rng.uniform(size=7), 6),  # every cell a neighbour
    ]
    assert np.max(np.abs(d)) < 1e-12


def test_with_0_1_weights_the_general_moments_are_those_of_the_counts_and_pairs_go_one_at_a_time():
    rng = np.random.default_rng(3)
    N = 400
    x, y = rng.uniform(0, 200, N), rng.uniform(0, 200, N)
    d = one_image(x, y, 20)
    F = rng.binomial(1, 0.3, N).astype(float)
    T = F * rng.binomial(1, 0.5, N)
    for a, b, self in ((F, T, False), (T, F, False), (F, F, True)):
        s = d.overlap_soft_rows(a, b, 1, True)
        E, V = d.overlap_moments([a.sum()], [b.sum()], [(a * b).sum()], [int(self)], 0, 50)
        np.testing.assert_allclose(s[1:3], [E[0], V[0]], rtol=1e-9)
    # several pairs in one call, with images of 0, 1 and 400 cells: as one pair at a time
    d2 = _core.Dataset(
        np.r_[x, 5, x[:3]],
        np.r_[y, 5, y[:3]],
        np.zeros(N + 4, dtype=np.int32),
        np.array([0, N, N, N + 1, N + 4], dtype=np.int32),
        1,
    )
    d2.build_radius_index(20)
    a = np.r_[F * rng.uniform(size=N), 1, 0.5, 1, 0]
    B = np.stack([np.r_[T, 1, 1, 0, 1], np.r_[rng.uniform(size=N), 0.3, 0.2, 0.2, 0.9]])
    for alloc in (True, False):
        both = d2.overlap_soft_rows(a, B.ravel(), 2, alloc).reshape(4, 2, 4)
        for q in range(2):
            one = d2.overlap_soft_rows(a, B[q], 1, alloc).reshape(4, 4)
            np.testing.assert_array_equal(both[:, q, :], one)
    # weighted_score_sums: the sum of b h over all the cells, h without the cell itself
    ptr = np.arange(N + 1, dtype=np.int32)
    nb = neighbours(x, y, 20)
    for alloc in (True, False):
        h = np.array([1 - np.prod(1 - a[:N][n]) if alloc else np.sum(a[:N][n]) for n in nb])
        got = d.weighted_score_sums(a[:N], ptr, np.zeros(N, dtype=np.int32), B[1, :N], 1, alloc)
        assert got[0] == pytest.approx(np.sum(B[1, :N] * h), rel=1e-12)


def test_a_group_and_its_subtypes_each_images_excess_counts_every_cell_around_the_exact_mean():
    # (a seed where no image's room n lands on the guard's boundary n = 1, which rounding decides)
    rng = np.random.default_rng(4)
    imgs = []
    for i in range(1, 13):
        imgs.append(
            pd.DataFrame(
                {
                    "x": rng.uniform(0, 100, 9),
                    "y": rng.uniform(0, 100, 9),
                    "cellType": rng.choice(["a", "b", "c", "c"], 9),
                    "imageID": f"i{i:02d}",
                    "patient": f"p{(i + 1) // 2:02d}",
                    "condition": "x" if i <= 6 else "y",
                }
            )
        )
    d = pd.concat(imgs, ignore_index=True)
    M = pd.DataFrame({t: (d["cellType"] == t).astype(float) for t in ("a", "b", "c")})
    M["G"] = M["a"] + M["b"]
    r = 40
    res = spicyr.spicy(
        d, condition="condition", subject="patient", r=r, cell_weights=M, from_=["G", "a"], to=["G", "a"]
    )

    def lab(L, label):
        return np.isin(L, ("a", "b")) if label == "G" else L == label

    for f, t in (("G", "a"), ("a", "G"), ("G", "G"), ("a", "a")):
        rows = []
        for z in imgs:
            Mi = M[(d["imageID"] == z["imageID"].iloc[0]).to_numpy()]
            nb = neighbours(z["x"].to_numpy(), z["y"].to_numpy(), r)
            A = np.array(arrangements(z["cellType"]))  # random labelling of the image's cells
            O0 = np.array([overlap_count(nb, lab(L, f), lab(L, t)) for L in A], dtype=float)
            O = overlap_count(nb, Mi[f].to_numpy() == 1, Mi[t].to_numpy() == 1)
            rows.append([O, O0.mean(), np.mean(O0**2) - O0.mean() ** 2, Mi[t].sum()])
        O, E, V, nT = np.array(rows).T
        keep = (nT > 0) & (V > 1e-9)
        avoid = np.sum(O[keep] - E[keep]) < 0
        n = E if avoid else nT - E
        keep &= n >= 1
        y = np.where(keep, (O - E) / np.where(keep, n, 1), np.nan)
        np.testing.assert_allclose(res.pairwise_assoc[f"{f}__{t}"], y, rtol=0, atol=1e-9, equal_nan=True)


def soft_inputs():
    """Soft weights of the shared cells: a dense matrix of probabilities, a confidence for each cell's own type, and
    the dense matrix with the B cells' rows set to 0/1 (as tests/shared_cases/reference_1.99.44_soft.csv)."""
    lev = sorted(CELLS["cellType"].unique())
    onehot = pd.get_dummies(CELLS["cellType"])[lev].astype(float).to_numpy()
    rng = np.random.default_rng(8)
    dense = onehot * 3 + rng.uniform(size=onehot.shape)
    dense = dense / dense.sum(axis=1, keepdims=True)
    q = rng.uniform(0.5, 1, len(CELLS))
    mixed = dense.copy()
    isb = (CELLS["cellType"] == "B").to_numpy()
    mixed[isb] = onehot[isb]
    return {"dense": pd.DataFrame(dense, columns=lev), "q": q, "mixed": pd.DataFrame(mixed, columns=lev)}


def test_soft_pairs_whose_to_cells_all_are_candidates_give_the_results_of_spicyr_1_99_44():
    # reference_1.99.44_soft.csv: spicy() of spicyr 1.99.44 with these weights (no self-pairs; every `to` cell has
    # `from` weight below 1)
    ref = pd.read_csv(HERE / "reference_1.99.44_soft.csv", index_col=0, keep_default_na=False, na_values=[""])
    inputs = soft_inputs()
    for case, exp in ref.groupby("case", sort=False):
        name, effect = case.split(" ")
        now = spicyr.spicy(
            CELLS,
            condition="condition",
            subject="patient",
            r=30,
            from_=["tumour", "T", "B"],
            to=["T", "B", "macro"],
            cell_weights=inputs[name],
            effect=effect,
        ).cell_results
        assert (now["from"] == now["to"]).sum() == 2
        now = now[now["from"] != now["to"]]
        assert list(now.index) == list(exp.index), case
        # at the shared cases' tolerance (libm differs between platforms); p_adj is over every tested pair, and the
        # self-pairs are now among them; the file has the columns of both effects (count has no side)
        for col in [c for c in exp.columns.drop(["case", "p_adj"]) if c in now.columns or exp[c].notna().any()]:
            if col == "p_value":
                # the skewness correction of the two-condition CR2 t changes p-values only: each equals the reference
                # as it is or as the uncorrected p of the same row
                p, ref = now[col].to_numpy(float), exp[col].to_numpy(float)
                t = (now["excess_difference"] / now["se"]).to_numpy(float)
                p_unc = 2 * stats.t.sf(np.abs(t), now["df"].to_numpy(float))
                assert np.array_equal(np.isnan(p), np.isnan(ref)), case
                close = np.isclose(p, ref, rtol=1e-7, atol=1e-10) | np.isclose(p_unc, ref, rtol=1e-7, atol=1e-10)
                assert (close | np.isnan(ref)).all(), case
            elif exp[col].dtype.kind in "fi":
                np.testing.assert_allclose(
                    now[col].to_numpy(float),
                    exp[col].to_numpy(float),
                    rtol=1e-7,
                    atol=1e-10,
                    equal_nan=True,
                    err_msg=f"{case} {col}",
                )
            else:
                assert (now[col].fillna("").astype(str) == exp[col].fillna("").astype(str)).all(), f"{case} {col}"


@pytest.mark.parametrize("effect", ["fraction", "count"])
def test_several_pairs_in_one_call_equal_one_pair_at_a_time(effect):
    w = WEIGHTS.copy()
    w["lymphocyte01"] = CELLS["cellType"].isin(["T", "B"]).astype(float).to_numpy()
    labels = ["T", "lymphocyte", "lymphocyte01", "tumour"]
    kw = {"condition": "condition", "subject": "patient", "r": 30, "effect": effect, "cell_weights": w}
    both = spicyr.spicy(CELLS, from_=labels, to=labels, **kw).cell_results
    assert len(both) == 16
    for key, row in both.iterrows():
        one = spicyr.spicy(CELLS, from_=row["from"], to=row["to"], **kw).cell_results.loc[key]
        for col in ("excess_difference", "se", "df", "p_value", "tau2"):
            assert one[col] == pytest.approx(row[col], rel=1e-12, abs=1e-15, nan_ok=True), f"{key} {col}"


def test_soft_self_pairs_a_soft_subtype_of_a_0_1_group_and_the_count_effect_are_tested():
    w = soft_inputs()["dense"]
    w["lymphocyte"] = CELLS["cellType"].isin(["T", "B"]).astype(float).to_numpy()
    for effect in ("fraction", "count"):
        b = spicyr.spicy(
            CELLS,
            condition="condition",
            subject="patient",
            r=30,
            from_=["T", "lymphocyte"],
            to=["T", "lymphocyte"],
            cell_weights=w,
            effect=effect,
        ).cell_results
        assert list(b.index) == ["T__T", "T__lymphocyte", "lymphocyte__T", "lymphocyte__lymphocyte"]
        assert np.all(np.isfinite(b["p_value"])), effect
