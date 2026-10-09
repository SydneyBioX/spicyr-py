"""Proportions per image and simple tests on their columns, as spicyR's ``getProp()`` and ``colTest()``.

The Wilcoxon rank-sum and Welch t-tests follow R's ``wilcox.test()`` (R >= 4.6) and ``t.test()``: the exact
permutation distribution of the ranks (midranks with ties) when both groups have fewer than 50 values, otherwise
the normal approximation with continuity and tie corrections.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import _core
from ._cell import _r_str_value, as_str, p_adjust_bh


def _r_sorted(values) -> list[str]:
    # the order of R's sort() in an English locale for plain labels: case-insensitive, lower case first on ties
    return sorted(values, key=lambda s: (s.casefold(), s.swapcase()))


def _levels(col: pd.Series, drop: bool = True) -> list[str]:
    """Levels as R's factor() makes them.

    The categories of a categorical (those present, with ``drop``), else the sorted values.
    """
    s = as_str(col)
    if isinstance(col.dtype, pd.CategoricalDtype):
        present = set(s.dropna())
        return [c for c in (_r_str_value(c) for c in col.cat.categories) if not drop or c in present]
    return _r_sorted(set(s.dropna()))


def _obs(cells) -> pd.DataFrame:
    if type(cells).__name__ == "AnnData":
        return cells.obs
    if isinstance(cells, pd.DataFrame):
        return cells
    raise TypeError("`cells` must be a pandas DataFrame or an AnnData object.")


def get_prop(cells, feature="cellType", image_id="imageID") -> pd.DataFrame:
    """The proportion of each level of ``feature`` in each image, as spicyR's ``getProp()``.

    Parameters
    ----------
    cells
        A pandas DataFrame or an AnnData object with one row per cell.
    feature
        The column whose proportions are computed, e.g. cell types or regions.
    image_id
        The column of the units: images, or patients to pool the images of each patient.

    Returns
    -------
    A DataFrame with one row per unit and one column per level of ``feature``; each row sums to one.

    """
    obs = _obs(cells)
    for col in (image_id, feature):
        if col not in obs.columns:
            raise ValueError(f"column '{col}' not found")
    units, levels = as_str(obs[image_id]), as_str(obs[feature])
    tab = pd.crosstab(units, levels).reindex(
        index=_levels(obs[image_id], drop=False), columns=_levels(obs[feature], drop=False), fill_value=0
    )
    tab.index.name = tab.columns.name = None
    return tab.div(tab.sum(axis=1), axis=0)


def _signif(x, digits=2):
    x = np.asarray(x, dtype=float)
    return np.array([float(f"{v:.{digits}g}") if np.isfinite(v) else v for v in x.ravel()]).reshape(x.shape)


def _pnorm(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _sum_distribution(scores: np.ndarray, m: int) -> np.ndarray:
    """P(sum = w), w = 0, 1, ..., of m of the integer ``scores`` drawn without replacement (R's dpermdist2)."""
    total = int(scores.sum())
    counts = np.zeros((m + 1, total + 1))
    counts[0, 0] = 1.0
    for i, sc in enumerate(scores, start=1):
        for j in range(min(i, m), 0, -1):
            counts[j, sc:] += counts[j - 1, : total + 1 - sc]
    return counts[m] / counts[m].sum()


def _wilcox(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The statistic and two-sided p-value of R's ``wilcox.test(x, y)`` (R >= 4.6)."""
    nx, ny = len(x), len(y)
    r = pd.Series(np.r_[x, y]).rank().to_numpy()
    w = r[:nx].sum() - nx * (nx + 1) / 2
    if nx < 50 and ny < 50:
        # exact: the permutation distribution of the (mid)ranks, on a grid of halves when ranks are halves
        f = 1 if np.all(r == np.floor(r)) else 2
        d = _sum_distribution(np.sort((f * r).astype(int)), nx)
        s = (np.arange(len(d)) / f) - nx * (nx + 1) / 2
        lower = d[s < w + 1e-8].sum()
        upper = 1 - d[s < w - 0.25 + 1e-8].sum()
        return w, min(2 * lower, 2 * upper, 1.0)
    _, nties = np.unique(r, return_counts=True)
    z = w - nx * ny / 2
    sd = math.sqrt((nx * ny / 12) * ((nx + ny + 1) - np.sum(nties**3 - nties) / ((nx + ny) * (nx + ny - 1))))
    z = (z - np.sign(z) * 0.5) / sd
    p = _pnorm(z)
    return w, 2 * min(p, 1 - p)


def _ttest(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    nx, ny = len(x), len(y)
    mx, my = x.mean(), y.mean()
    sx2, sy2 = x.var(ddof=1) / nx, y.var(ddof=1) / ny
    se = math.sqrt(sx2 + sy2)
    if se < 10 * np.finfo(float).eps * max(abs(mx), abs(my)):
        raise ValueError("data are essentially constant")
    df = se**4 / (sx2**2 / (nx - 1) + sy2**2 / (ny - 1))
    t = (mx - my) / se
    return mx, my, t, _core.pt_two_sided(t, df)


def _efron(x: np.ndarray, time: np.ndarray, event: np.ndarray, beta: float) -> tuple[float, float, float]:
    """Cox partial log-likelihood (Efron ties), score and information for one covariate."""
    eta = beta * x
    risk = np.exp(eta)
    loglik = u = imat = 0.0
    for t in np.unique(time[event == 1]):
        at = time >= t
        dead = (time == t) & (event == 1)
        d = int(dead.sum())
        s0, s1, s2 = risk[at].sum(), (risk * x)[at].sum(), (risk * x * x)[at].sum()
        d0, d1, d2 = risk[dead].sum(), (risk * x)[dead].sum(), (risk * x * x)[dead].sum()
        loglik += eta[dead].sum()
        u += x[dead].sum()
        for k in range(d):
            f = k / d
            den, a, c = s0 - f * d0, s1 - f * d1, s2 - f * d2
            loglik -= math.log(den)
            u -= a / den
            imat += c / den - (a / den) ** 2
    return loglik, u, imat


def _coxph(x: np.ndarray, time: np.ndarray, event: np.ndarray, iter_max=20, eps=1e-9) -> tuple[float, float, float]:
    """One-covariate ``survival::coxph()`` (its coxfit6 Newton-Raphson with step halving; Efron ties)."""
    x = x - x.mean()  # coxph centres (and scales) the covariate, which leaves the iterations unchanged
    beta = 0.0
    loglik, u, imat = _efron(x, time, event, beta)
    if not math.isfinite(loglik) or imat <= 0:
        return np.nan, np.nan, np.nan
    best, newbeta, halving = loglik, beta + u / imat, 0
    for _ in range(iter_max):
        newlk, u, imat = _efron(x, time, event, newbeta)
        finite = math.isfinite(newlk) and math.isfinite(u) and math.isfinite(imat)
        if finite and abs(1 - best / newlk) <= eps:
            beta = newbeta
            break
        if not finite or newlk < best:
            halving += 1
            newbeta = (newbeta + halving * beta) / (halving + 1)
        else:
            halving, best = 0, newlk
            beta, newbeta = newbeta, newbeta + u / imat
    else:
        _, u, imat = _efron(x, time, event, beta)
    se = math.sqrt(1 / imat) if imat > 0 else np.nan
    return beta, se, 2 * (1 - _pnorm(abs(beta / se)))


def _cox(values: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[float, float, float]:
    """As spicyR's colCoxTests(): the core's Cox fit, or coxph() where it fails."""
    ok = np.isfinite(time) & np.isfinite(event) & np.isfinite(values)
    fit = _core.cox_fit(time[ok], event[ok].astype(np.int32), values[ok], 1)
    if fit["ok"] and np.isfinite(fit["se"][0]):
        return fit["beta"][0], fit["se"][0], fit["p"][0]
    try:
        return _coxph(values[ok], time[ok], event[ok].astype(int))
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.nan, np.nan, np.nan


def col_test(df, condition, type=None, feature=None, image_id="imageID") -> pd.DataFrame:
    """Test each column of a table, such as proportions per image, as spicyR's ``colTest()``.

    Each column is tested for a difference between two groups (Wilcoxon or Welch t-test) or for an association
    with survival (Cox model).

    Parameters
    ----------
    df
        A DataFrame with one row per image (or patient) and one column per feature, such as the output of
        :func:`get_prop`. With ``feature`` given, ``df`` is instead a table of cells (a DataFrame or an AnnData
        object), and the proportions of ``feature`` in each ``image_id`` are computed first.
    condition
        The group of each row of ``df`` (a sequence or Series in its row order), or two columns ``(time,
        event)`` for survival. With ``feature``, the name of the cells' group column.
    type
        ``"wilcox"``, ``"ttest"`` or ``"survival"``; by default ``"ttest"`` for one condition and
        ``"survival"`` for two columns.
    feature
        The column of the cells whose proportions are tested, e.g. cell types or regions.
    image_id
        With ``feature``: the units, images or patients (to pool the images of each patient).

    Returns
    -------
    A DataFrame with one row per column of ``df``, ordered by p-value: the group means (t-test), the test
    statistic (``tval.W``, ``tval.t``) or the Cox coefficient and its standard error (survival), the p-value
    (``pval``), the Benjamini-Hochberg adjusted p-value (``adjPval``) and the column name (``cluster``), all
    rounded to two significant digits as in R.

    """
    if feature is not None:
        obs = _obs(df)
        if type is None:
            type = "ttest" if isinstance(condition, str) else "survival"
        if type == "survival":
            raise ValueError("for survival, pass the proportions from get_prop() and a (time, event) table.")
        units = obs[[image_id, condition]].drop_duplicates()
        group = pd.Series(units[condition].to_numpy(), index=as_str(units[image_id]).to_numpy())
        group = group[~group.index.duplicated()]
        df = get_prop(obs, feature=feature, image_id=image_id)
        condition = group.loc[df.index]
    else:
        cond = condition.to_numpy() if isinstance(condition, (pd.Series, pd.DataFrame)) else np.asarray(condition)
        if len(cond) != len(df):
            raise ValueError("Number of oberservation does not match between df and outcome")
        if type is None:
            type = "ttest" if cond.ndim == 1 or cond.shape[1] == 1 else "survival"
        if not isinstance(condition, (pd.Series, pd.DataFrame)):
            condition = pd.DataFrame(cond) if cond.ndim == 2 else pd.Series(cond)

    values = df.to_numpy(float)
    if type == "survival":
        cond = np.asarray(condition, dtype=float)
        rows = [_cox(values[:, j], cond[:, 0], cond[:, 1]) for j in range(values.shape[1])]
        test = pd.DataFrame(_signif(np.array(rows)), index=df.columns, columns=["coef", "se.coef", "pval"])
    else:
        g = pd.Series(condition).reset_index(drop=True)
        levels = _levels(g)
        g = as_str(g).to_numpy()
        if len(levels) != 2:
            raise ValueError("grouping factor must have exactly 2 levels")
        rows = []
        for j in range(values.shape[1]):
            v = values[:, j]
            x = v[(g == levels[0]) & np.isfinite(v)]
            y = v[(g == levels[1]) & np.isfinite(v)]
            rows.append(_wilcox(x, y) if type == "wilcox" else _ttest(x, y))
        cols = (
            ["tval.W", "pval"]
            if type == "wilcox"
            else [f"mean in group {levels[0]}", f"mean in group {levels[1]}", "tval.t", "pval"]
        )
        test = pd.DataFrame(_signif(np.array(rows)), index=df.columns, columns=cols)
    test["adjPval"] = _signif(p_adjust_bh(test["pval"].to_numpy()))
    test["cluster"] = test.index
    return test.iloc[np.argsort(test["pval"].fillna(np.inf).to_numpy(), kind="stable")]
