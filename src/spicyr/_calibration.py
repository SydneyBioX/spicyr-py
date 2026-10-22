"""Checking that :func:`spicyr.spicy` is calibrated on a data set, by shuffling the condition across patients."""

import numpy as np
import pandas as pd

from ._api import spicy

__all__ = ["calibration_check"]

_COLUMN = "shuffledCondition"


def calibration_check(
    cells,
    condition=None,
    subject=None,
    survival=None,
    n_shuffles=10,
    thresholds=(0.05, 0.01),
    rng=None,
    image_id="imageID",
    **kwargs,
):
    """Check that :func:`spicyr.spicy` is calibrated on your data by shuffling the condition.

    Shuffles the condition across patients (or across images, without ``subject``), which removes any real
    difference between the groups while keeping everything else about the study, and runs :func:`spicyr.spicy` on
    each shuffle. If the test is calibrated, about 5% of pairs have p < 0.05 and about 1% p < 0.01 in each shuffle.
    Pairs share cells, so the percentage moves by a few points from one shuffle to the next; look at the average over
    shuffles. Patients without a condition keep it missing.

    Parameters
    ----------
    cells
        A pandas DataFrame or AnnData object, as for :func:`spicyr.spicy` (not a store).
    condition
        The column of the condition to shuffle.
    subject
        The column of the patient. Each patient keeps one condition, moved as a whole; without it, the condition is
        shuffled across images.
    survival
        Instead of ``condition``: a survival outcome ``(time_column, event_column)``, whose time and event are
        shuffled together.
    n_shuffles
        The number of shuffles.
    thresholds
        The p-value thresholds to report.
    rng
        A :class:`numpy.random.Generator`, or a seed for one.
    image_id
        The column of the image.
    **kwargs
        Other arguments of :func:`spicyr.spicy`, such as ``r``, ``cell_type`` or ``covariates``.

    Returns
    -------
    A DataFrame with one row per shuffle and threshold: ``shuffle``, ``threshold`` and ``percent``, the percentage
    of pairs whose p-value is below the threshold.

    Examples
    --------
    >>> import spicyr
    >>> cells = spicyr.datasets.example_cells()
    >>> spicyr.calibration_check(cells, condition="condition", subject="patient", r=30, n_shuffles=2, rng=1)  # doctest: +SKIP
    """
    if (condition is None) == (survival is None):
        raise ValueError("give `condition` or `survival`, not both.")
    if isinstance(cells, pd.DataFrame):
        obs = cells
    elif type(cells).__name__ == "AnnData":
        obs = cells.obs
    else:
        raise TypeError("calibration_check() needs the cells in memory: a pandas DataFrame or AnnData object.")
    cols = [condition] if condition is not None else list(survival)
    unit = subject if subject is not None else image_id
    miss = [c for c in [*cols, unit] if c not in obs.columns]
    if miss:
        raise ValueError(f"columns not found: {', '.join(miss)}")
    rng = np.random.default_rng(rng)
    # one condition per patient (or image), shuffled across those that have one; a missing condition stays missing
    u = obs[unit].astype(str).to_numpy()
    first = ~pd.Series(u).duplicated().to_numpy()
    units = u[first]
    labels = obs.loc[first, cols].reset_index(drop=True)
    has = np.flatnonzero(labels.notna().all(axis=1).to_numpy())
    at = pd.Index(units).get_indexer(u)
    new = [f"{_COLUMN}{k}" for k in range(len(cols))]
    rows = []
    try:
        for i in range(n_shuffles):
            shuffled = labels.copy()
            shuffled.iloc[has] = labels.iloc[has[rng.permutation(len(has))]].to_numpy()
            for c, n in zip(cols, new, strict=True):
                v = shuffled[c].to_numpy()[at]
                dt = labels[c].dtype
                # a categorical condition keeps its categories, and so its reference level
                obs[n] = pd.Categorical(v, categories=dt.categories, ordered=dt.ordered) if dt == "category" else v
            if condition is not None:
                res = spicy(cells, condition=new[0], subject=subject, image_id=image_id, **kwargs)
            else:
                res = spicy(cells, survival=tuple(new), subject=subject, image_id=image_id, **kwargs)
            p = res.cell_results["p_value"].dropna()  # the pairs tested
            rows += [(i + 1, t, 100 * float((p < t).mean())) for t in thresholds]
    finally:
        obs.drop(columns=[n for n in new if n in obs.columns], inplace=True)
    return pd.DataFrame(rows, columns=["shuffle", "threshold", "percent"])
