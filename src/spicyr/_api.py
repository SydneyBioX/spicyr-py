"""``spicy()``: test for changes in the co-localisation of cell types between conditions.

The mirror of spicyR's ``R/spicy_main.R``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import _cell, _core
from ._results import SpicyResults
from ._source import cell_source


def spicy(
    cells,
    condition=None,
    subject=None,
    covariates=None,
    image_id="imageID",
    cell_type="cellType",
    spatial_coords=("x", "y"),
    r=None,
    from_=None,
    to=None,
    method="cell",
    effect="allocation",
    k=None,
    combine="maxT",
    adjust_abundance=False,
    variance="cr2",
    ref=None,
    cores=1,
    survival=None,
    table_key="table",
    image_info=None,
    cell_weights=None,
    weight_labels=None,
    **kwargs,
) -> SpicyResults:
    """Test whether the co-localisation of cell types differs between conditions, or is associated with survival.

    Every ordered pair of cell types ``from_ -> to`` is tested: are ``to`` cells placed near ``from`` cells more than
    other cells are?

    In each image, the share of ``to`` cells with at least one ``from`` cell within ``r`` is compared with its exact
    expectation q if the ``to`` cells were a random choice among the cells that are not ``from`` cells (random
    labelling of the observed cells). The effect (``effect="allocation"``, the default) is the **fraction of** ``to``
    **cells placed next to (or kept away from)** ``from`` **cells**. For a pair that attracts (more ``to`` cells next to
    ``from`` cells than q over all images together) it is (observed share - q) / (1 - q): if a fraction f of the ``to``
    cells were moved next to ``from`` cells, the effect is f. For a pair that avoids it is (observed share - q) / q: if
    a fraction f of the ``to`` cells that would have a ``from`` cell nearby were moved away, the effect is -f. Either
    way it does not depend on how many ``from`` cells there are or how densely they are packed. The side is chosen
    once per pair from all images with a condition, without the conditions, and is reported in the ``side`` column.
    ``effect="count"`` gives the number of extra ``from`` cells within ``r`` of each ``to`` cell
    instead; it also reflects how many ``from`` cells surround a ``to`` cell (depth of infiltration), but it grows
    with how densely the ``from`` cells are packed. Images are combined within patients (``subject``) and patients
    within conditions by a frailty GEE with a second random effect for the images of a patient (each image is
    weighted by 1 / (its sampling variance + the between-image variance), so one large image cannot outweigh the
    patient's other images), and the difference is tested with a CR2 variance on Satterthwaite degrees of freedom,
    with **patients as the units**. The difference is adjusted for any ``covariates`` and, with
    ``adjust_abundance=True``, for the log share of the ``from`` type in each image; the unadjusted test is then
    reported alongside (``unadjusted_*`` columns). When nearly every cell has a ``from`` cell within ``r`` (q close to
    1), there is little room for attraction and an attracting pair's images carry little information; a smaller ``r``
    is more informative.

    The allocation effect is bounded: an image of an avoiding pair with no ``to`` cell next to a ``from`` cell has the
    effect -1, and an image of an attracting pair with every ``to`` cell next to one has +1. A patient is informative
    at a radius when at least one of its images is not at -1 or +1. A radius where one condition has no informative
    patient is not tested: that condition's patients all have the same effect, so its spread is exactly 0 and the
    standard error would rest on the other condition alone. Its p-value is NaN (with adjustment, ``adjusted_for`` is
    ``"none (no_information)"``), and it is left out of the combination over radii. With more than two conditions, a
    pair where any condition has no informative patient is not tested.

    The standard error of the difference is the sum of each condition's own CR2 variance, on Satterthwaite degrees
    of freedom; with adjustment, the CR2 variance of the adjusted design. ``variance="hartung_knapp"`` keeps its own
    variance.

    For a self-pair (``from_`` = ``to``) every cell can carry the label, so random labelling moves both the cells
    counted and their neighbours. Its variance is the exact random-labelling variance of the self-pair (for the radius
    graph and for the k nearest neighbours), not the variance of a pair of two types.

    With the allocation effect, an image whose room on the pair's side (the ``to`` cells without a ``from`` neighbour
    expected by chance for an attracting pair, those with one for an avoiding pair) is below one cell is not used:
    its effect could not show the pair's side, and could be large in the other direction. The side is chosen first,
    from every image.

    Every requested pair has a row in ``cell_results``: ``not_tested`` gives why a pair (or, in ``radius_results``, a
    radius) was not tested (``"one_patient_per_group"``, ``"no_information"``, ``"few_effective_units"`` or
    ``"no_variation"``; None when it was), and its p-value is then NaN and not counted in the BH adjustment.

    A patient's weight within its condition grows with the information of its images, so one or two patients can
    carry nearly all of a condition's weight. A radius where a condition's effective patients (Kish's
    (sum w)^2 / sum w^2 of the patients' weights w) fall below the smaller of 3 and half the number of its patients
    that contribute at that radius is not tested: its p-value is NaN (with adjustment, ``adjusted_for`` is
    ``"none (few_effective_units)"``), and it is left out of the combination over radii. A condition of 3 patients
    with equal weights can therefore be tested. With more than two conditions, a pair where any condition falls below
    this is not tested.

    Images with a missing condition are left out of every step: the choice of the side, the between-image variance
    and the tests.

    ``from`` is spelled ``from_`` because ``from`` is a Python keyword (``**{"from": ...}`` also works).

    Parameters
    ----------
    cells
        A pandas DataFrame (one row per cell), an AnnData object (cells as observations; coordinates in
        ``obsm["spatial"]`` or obs columns), a SpatialData object (its annotation table ``table_key``), or the path
        of a SpatialData store (``.zarr``) or of an AnnData zarr store. A store is read lazily: the image and
        cell-type columns are loaded once as integer codes, the coordinates (``obsm["spatial"]``, or the obs columns
        ``spatial_coords``) a batch of whole images at a time, and the condition, subject and covariates once per
        image, so the data never need to fit in memory. Reading is fastest when the rows are grouped by image (as
        spatialdata writes a table sorted by region); categorical image and cell-type columns keep the codes small.
    condition
        Column of the image-level condition (two or more groups).
    subject
        Column of the patient of each image. Images of one patient are combined; by default each image is a
        patient.
    covariates
        Image- or patient-level columns to adjust for. The effect of each is reported as ``<column>_effect`` and
        ``<column>_p_value`` (one column per level after the first for a categorical covariate).
    r
        Radius, or several radii combined by ``combine`` ("maxT", up to 6 radii; "davies", Davies' bound for the
        same maximum, for many radii; or "cauchy"). Default 50. The max-T p-value is the multivariate normal
        probability of the largest absolute z, computed by the quasi-Monte Carlo method of Genz and Bretz with an
        absolute error of about 1e-4 (by quadrature for two radii); the same inputs always give the same value.
    from_, to
        Cell types to test (all ordered pairs by default).
    method
        "cell" (spicyR Cell). The image-level method of spicyR is not yet available in Python.
    effect
        "allocation" (default): the fraction of ``to`` cells moved next to (or away from) ``from`` cells within ``r``;
        "count": the number of extra ``from`` cells within ``r`` of each ``to`` cell. With ``k``, "within ``r``"
        means among the cell's ``k`` nearest neighbours.
    k
        Use the ``k`` nearest neighbours instead of a radius.
    adjust_abundance
        Adjust the test for the log share of the ``from`` type in each image (default ``False``). Its effect is
        reported as ``abundance_effect``. It does not separate more ``from`` cells from more densely packed ones,
        and it removes real effects when the share tracks the condition.
    variance
        "cr2" (default: each condition's CR2 variance, on Satterthwaite df, see above) or "hartung_knapp" (the model-based variance, scaled by the Pearson statistic when that
        exceeds its df and never below CR2, on m - 2 df). The variance used is in ``.variance``
        of the result.
    survival
        A survival outcome: ``(time_column, event_column)``, one value per patient (the R package's
        ``Surv(time, event)``). Leave ``condition`` empty. (Older form: ``condition=time_column,
        survival=event_column``.)
    image_id, cell_type, spatial_coords
        Columns of the image, the cell type and the coordinates. For SpatialData, images are the table's regions
        (its region_key column) unless ``image_id`` names another column.
    cores
        Threads. Batches of whole images are read and summarised in parallel; results do not depend on it.
    table_key
        The annotation table of a SpatialData object or store.
    image_info
        A table with one row per image, for ``condition``, ``subject``, covariates or the survival columns that are
        not columns of the cell table: a DataFrame, an AnnData object (its obs), or the key of a table of the
        SpatialData object or store. Its ``image_id`` column, or else its index, gives the image IDs. Every image of
        the cells needs a row, and every row an image.
    cell_weights
        Optional cell weights in [0, 1], in the order of the cells: a 1-D array with one weight per cell for its own
        cell type (for example a quality score), or a cells x labels matrix of soft cell type membership (for example
        a classifier's confidence; a DataFrame, whose columns are the labels, or a numpy array or scipy sparse matrix
        with ``weight_labels``). A 1-D array (numpy array, list or pandas Series, taken by position) is the same as
        the cells x cell types matrix with each cell's weight in its own type's column. With a matrix, ``from_`` and
        ``to`` name labels, and labels may overlap (a group and its subtypes). A ``to`` cell counts with its weight,
        and its score is the chance that it has a ``from`` neighbour within ``r`` if labels were drawn from the
        weights (allocation), or the summed ``from`` weight of its neighbours (count). The null shuffles the ``to``
        weights over the cells whose ``from`` weight is below 1, whose exact mean and variance replace those of
        random labelling; 0/1 weights give the results of the cell types. The abundance share is the ``from``
        label's summed weight over the image's cells. One radius, cells in memory; pairs of a label with itself are
        not tested.
    weight_labels
        The labels of the columns of ``cell_weights`` when it is a 2-D numpy array or sparse matrix.
    """
    if "from" in kwargs:
        from_ = kwargs.pop("from")
    gone = [a for a in ("frailty", "label_clustering", "image_heterogeneity") if a in kwargs]
    if gone:
        raise TypeError(
            f"`{gone[0]}` is not an argument of spicy(): the cell method always uses the patient frailty and the "
            "image random effect, without label-clustering inflation."
        )
    if kwargs:
        raise TypeError(f"unexpected arguments: {list(kwargs)}")
    if method != "cell":
        raise NotImplementedError(
            "method='image' (the original spicyR test) is not yet available in Python; use spicyR in R."
        )
    if combine not in ("maxT", "davies", "cauchy"):
        raise ValueError("combine must be 'maxT', 'davies' or 'cauchy'.")
    if variance not in ("cr2", "hartung_knapp"):
        raise ValueError("variance must be 'cr2' or 'hartung_knapp'.")
    if effect not in ("allocation", "count"):
        raise ValueError("effect must be 'allocation' or 'count'.")
    if r is None and k is None:
        r = 50
    weighted = cell_weights is not None
    if weighted and (k is not None or len(np.atleast_1d(r)) != 1 or isinstance(cells, str | os.PathLike)):
        raise ValueError("`cell_weights` needs the cells in memory and one radius `r` (no `k`).")
    if k is not None:
        r = None
    src = cell_source(cells, image_id, cell_type, spatial_coords, table_key)
    if isinstance(image_info, str) and type(cells).__name__ == "SpatialData":
        image_info = cells.tables[image_info].obs
    is_surv = survival is not None
    if is_surv and not isinstance(survival, str):
        if len(survival) != 2:
            raise ValueError("survival must be (time_column, event_column).")
        condition, survival = survival[0], survival[1]
    if condition is None:
        raise ValueError("give `condition` (the column of the groups) or `survival` (time and event columns).")
    ctx = _cell.cell_context(
        src,
        None if is_surv else condition,
        subject,
        image_info=image_info,
        survival=is_surv,
        ref=ref,
        image_id=image_id,
        dataset=weighted,
    )
    types = ctx.type_labels
    if weighted:
        memb, types = _cell.membership(
            cell_weights, src.n, weight_labels, src.type_codes, ctx.type_labels, src.type_labels
        )
    bad = [
        t
        for t in ([from_] if isinstance(from_, str) else (from_ or [])) + ([to] if isinstance(to, str) else (to or []))
        if t not in types
    ]
    if bad:
        matrix = weighted and not _cell.is_weight_vector(cell_weights)
        raise ValueError(f"not a column of `cell_weights`: {bad}" if matrix else f"cell type not found: {bad}")
    # from -> to: are `to` cells placed next to `from` cells (allocation: the extra fraction of `to` cells with a `from`
    # cell within r; count: extra `from` cells within r of each `to` cell), beyond the `to` cells being a random subset
    # of the cells that are not `from` cells. This is the core's own (counted, centre) order, so pairs pass through.
    pairs = list(_cell.enumerate_pairs(from_, to, types))
    covariates = None if covariates is None else ([covariates] if isinstance(covariates, str) else list(covariates))
    miss = [c for c in covariates or [] if not ctx.has(c)]
    if miss:
        raise ValueError(f"covariates not found: {miss}")
    pheno = ctx.pheno(
        [c for c in (condition, subject, *(covariates or []), *([survival] if is_surv else [])) if c is not None]
    )
    if is_surv:
        pheno[".time"] = pheno[condition].astype(float)
        pheno[".event"] = pheno[survival].astype(float)
    Z_extra, extra_names = None, []
    if covariates is not None:
        Z_extra, extra_names = _model_matrix(pheno, covariates)  # NaN rows where a covariate is missing
        Z_extra = Z_extra - np.nanmean(Z_extra, axis=0)

    radii = [np.nan] if k is not None else sorted(set(np.atleast_1d(r).astype(float)))
    if len(radii) > 1 and not is_surv and len(ctx.levels) > 2:
        raise ValueError("several radii are supported for two conditions; give one `r`.")
    if combine == "maxT" and len(radii) > 6 and not is_surv:
        raise ValueError('combine="maxT" supports up to 6 radii; use combine="davies" for more.')
    if weighted:
        sv = _survival_setup(ctx, pheno, covariates) if is_surv else None
        res = _cell.cell_weighted(
            ctx,
            memb,
            types,
            pairs,
            radii[0],
            effect,
            variance,
            adjust_abundance,
            Z_extra,
            extra_names,
            (lambda c, rows, f, t: _survival_test(c, rows, f, t, sv, adjust_abundance, covariates))
            if is_surv
            else None,
        )
        res["table"] = (
            _survival_table(res["fits"], adjust_abundance, covariates)
            if is_surv
            else _cell.cell_table(res["fits"], ctx, adjust_abundance or Z_extra is not None)
        )
    elif is_surv:
        res = _survival(
            ctx,
            pairs,
            radii,
            k,
            pheno,
            covariates,
            cores,
            adjust_abundance,
            effect,
        )
    else:
        # several radii: one neighbour search at the largest radius gives every radius's graph
        graphs = (
            _cell.cell_graphs_radii(ctx, radii, effect, n_threads=cores)
            if len(radii) > 1
            else [
                _cell.cell_graph(
                    ctx,
                    pairs,
                    r=None if np.isnan(radii[0]) else radii[0],
                    k=k,
                    label_clustering=False,
                    n_threads=cores,
                    effect=effect,
                )
            ]
        )
        if len(ctx.levels) > 2:
            res = _levels(ctx, graphs[0], pairs, variance, adjust_abundance, Z_extra, extra_names)
        else:
            res = _pairs(ctx, graphs, pairs, variance, adjust_abundance, Z_extra, extra_names, radii, combine, cores)
    out = _results(res, ctx, pheno, condition, subject, is_surv, radii, k, effect)
    out.variance = None if is_surv else variance
    return out


def _model_matrix(pheno, covariates):
    """R's model.matrix(~ covariates)[, -1] and its column names.

    Treatment contrasts (levels sorted, the first dropped); a column is named by the covariate, followed by the
    level for a categorical one.
    """
    cols, names = [], []
    for c in covariates:
        v = pheno[c]
        if pd.api.types.is_numeric_dtype(v) and not isinstance(v.dtype, pd.CategoricalDtype):
            cols.append(v.to_numpy(float)[:, None])
            names.append(c)
        else:
            lev = _cell.condition_levels(v)
            d = np.column_stack([(_cell.as_str(v) == l).to_numpy(float) for l in lev[1:]])
            d[v.isna().to_numpy()] = np.nan
            cols.append(d)
            names += [f"{c}{l}" for l in lev[1:]]
    return np.column_stack(cols), names


def _levels(ctx, g, pairs, variance, adjust, Z_extra, extra_names):
    """More than two conditions (one radius): each pair's design in turn, then the table."""
    g.image_heterogeneity = True
    fits = [_cell.cell_pair_test(ctx, g, p[0], p[1], True, variance, adjust, Z_extra, extra_names) for p in pairs]
    return {"table": _cell.cell_table(fits, ctx, adjust or Z_extra is not None), "fits": fits}


def _pairs(ctx, graphs, pairs, variance, adjust, Z_extra, extra_names, radii, combine, n_threads=1):
    """Two conditions: every pair at every radius in one call to the core, then the tables.

    The core's pair_tests gives, per pair and radius, the image rows, the image random effect, the frailty test and
    the adjusted design, as cell_pair_test() does for one, and with several radii their combination. Several radii:
    per-radius tables, and one row per pair with the combined p-value (the main test and, when adjusted, the
    unadjusted test); the other columns are those of the radius chosen by max-T. The image rows are those of the
    radius nearest the median.
    """
    allocation = graphs[0].effect == "allocation"
    K = len(graphs)
    keep = int(np.argmin(np.abs(np.array(radii) - np.median(radii)))) if K > 1 else 0
    from_ = [p[0] for p in pairs]
    to = [p[1] for p in pairs]
    x = _core.pair_tests(
        [g.any if allocation else g.totals for g in graphs],
        [g.self_expected if allocation else g.sq for g in graphs],
        [g.self_var for g in graphs],
        allocation,
        graphs[0].knn,
        ctx.counts,
        len(ctx.type_labels),
        graphs[0].psi,
        np.array([ctx.type_labels.index(f) for f in from_], dtype=np.int32),
        np.array([ctx.type_labels.index(t) for t in to], dtype=np.int32),
        ctx.image_unit.astype(np.int32),
        ctx.image_group.astype(np.int32),
        len(ctx.unit_labels),
        True,
        True,
        variance,
        adjust,
        np.zeros(0) if Z_extra is None else Z_extra,
        0 if Z_extra is None else Z_extra.shape[1],
        keep,
        combine if K > 1 else "none",
        int(n_threads),
    )
    adjusted = adjust or Z_extra is not None
    names = None if Z_extra is None else list(extra_names)
    # the effect columns of a radius where no pair's design was fitted (e.g. none with variation): those of the first
    # radius where one was, so that every radius has the same columns
    ens = [_pairs_enames(x, k, adjusted, names) for k in range(K)]
    en0 = next((e for e in ens if e is not None), None)
    tabs = [
        _pairs_table(x, k, from_, to, allocation, adjusted, names, en0 if ens[k] is None else ens[k]) for k in range(K)
    ]
    fits = [
        {"from": from_[i], "to": to[i], "rows": x["rows"][i], "unadjusted": {"image_weight": x["image_weight"][i]}}
        for i in range(len(pairs))
    ]
    if K == 1:
        return {"table": tabs[0], "fits": fits}
    long = pd.concat([t.assign(r=rr) for t, rr in zip(tabs, radii, strict=True) if t is not None])
    long = long[["r"] + [c for c in long.columns if c != "r"]]
    # the rows of `long`: radius by radius, every pair; one row per pair, that of its chosen radius. A pair tested at
    # no radius keeps a row of NaNs, with the reasons it was not tested at each radius.
    P = len(pairs)
    best = np.asarray(x["best"])
    none = best < 0
    tab = long.iloc[np.where(none, keep, best) * P + np.arange(P)].copy()
    tab["p_value"] = x["comb_p"]
    tab["p_value_best_radius"] = x["comb_p_best"]
    if adjusted:
        tab["unadjusted_p_value"] = x["comb_u_p"]
    if none.any():
        num = [c for c in tab.columns if pd.api.types.is_numeric_dtype(tab[c])]
        tab.loc[none, num] = np.nan
        if allocation:
            tab.loc[none, "side"] = None
        reasons = []
        for j in np.flatnonzero(none):
            w = [x["not_tested"][j * K + k] for k in range(K)]
            w = list(dict.fromkeys(v for v in w if v))
            reasons.append("; ".join(w) if w else None)
        tab.loc[none, "not_tested"] = reasons
    tab["p_adj"] = _cell.p_adjust_bh(tab["p_value"])
    if adjusted:
        tab["unadjusted_p_adj"] = _cell.p_adjust_bh(tab["unadjusted_p_value"])
    tab = tab.drop(columns=[c for c in ("unadjusted_difference", "unadjusted_se", "unadjusted_df") if c in tab.columns])
    tab.index = tab["from"] + "__" + tab["to"]
    return {"table": _cell.order_columns(tab), "fits": fits, "radius_table": long}


def _added(ab, extra_names):
    """The added columns of each adjusted fit (the abundance first), named as cell_pair_test() names its effects."""
    extra = [] if extra_names is None else extra_names
    return [(["abundance"] if a else []) + extra for a in ab]


def _pairs_enames(x, k, adjusted, extra_names):
    """The added columns of the pairs adjusted at radius k, as named in their fits; None when none was."""
    s = np.flatnonzero(x["ok"][:, k] == 1)
    if not len(s) or not adjusted:
        return None
    design = x["design"][s, k]
    added = _added(x["abundance"][s, k] == 1, extra_names)
    return list(dict.fromkeys(nm for j in np.flatnonzero(design == 1) for nm in added[j])) or None


def _pairs_table(x, k, from_, to, allocation, adjusted, extra_names, enames=None):
    """The table of one radius from the core's pair_tests (the columns and rows cell_table() gives for the same fits).

    extra_names: the covariates' columns, None without covariates. enames: the effect columns (default: those of the
    pairs adjusted at this radius).
    """
    # every requested pair, tested or not (not_tested: why not; None when it was)
    s = np.arange(len(from_))
    if not len(s):
        return None
    K = x["ok"].shape[1]

    def at(key):
        return x[key][s, k]

    tab = {
        "from": [from_[i] for i in s],
        "to": [to[i] for i in s],
        "excess_ref": at("coef_ref"),
        "excess_comp": at("coef_comp"),
        "excess_difference": at("difference"),
        "se": at("se"),
        "df": at("df"),
        "p_value": at("p"),
        "tau2": at("tau2"),
    }
    if allocation:
        tab["side"] = [("attract" if v > 0 else "avoid") if o == 1 else None for v, o in zip(at("side"), at("ok"))]
    tab["tau2_image"] = at("tau2_image")
    tab["not_tested"] = [x["not_tested"][i * K + k] or None for i in s]
    if adjusted:
        design, ab = at("design"), at("abundance") == 1
        added = _added(ab, extra_names)
        label = [
            "+".join(
                (["abundance"] if "abundance" in nm[:1] else []) + (["covariates"] if extra_names is not None else [])
            )
            for nm in added
        ]
        tab["adjusted_for"] = [
            label[j] if d == 1 else f"none ({x['design_reason'][i * K + k]})" if d == 2 else "none"
            for j, (i, d) in enumerate(zip(s, design, strict=True))
        ]
        if enames is None:
            enames = _pairs_enames(x, k, adjusted, extra_names) or []
        for nm in enames:
            e = np.full(len(s), np.nan)
            ep = np.full(len(s), np.nan)
            for j in np.flatnonzero(design == 1):
                if nm in added[j]:
                    q = len(added[j]) - 1 - added[j][::-1].index(nm)  # its last place (as a dict of the names)
                    c = q if ab[j] else q + 1  # the core's index: 0 the abundance, 1 + c covariate c
                    e[j] = x["effect"][s[j], k, c]
                    ep[j] = x["effect_p"][s[j], k, c]
            tab[f"{nm}_effect"] = e
            tab[f"{nm}_p_value"] = ep
        tab["unadjusted_difference"] = at("u_difference")
        tab["unadjusted_se"] = at("u_se")
        tab["unadjusted_df"] = at("u_df")
        tab["unadjusted_p_value"] = at("u_p")
    tab = pd.DataFrame(tab)
    tab["p_adj"] = _cell.p_adjust_bh(tab["p_value"])
    if adjusted:
        tab["unadjusted_p_adj"] = _cell.p_adjust_bh(tab["unadjusted_p_value"])
    tab.index = tab["from"] + "__" + tab["to"]
    return _cell.order_columns(tab)


def _survival(
    ctx,
    pairs,
    radii,
    k,
    pheno,
    covariates,
    cores,
    adjust,
    effect="allocation",
):
    """Survival: score test and the hazard ratio of the shrunken excess, per pair."""
    sv = _survival_setup(ctx, pheno, covariates)
    rr = radii[0]
    g = _cell.cell_graph(
        ctx,
        pairs,
        r=None if np.isnan(rr) else rr,
        k=k,
        label_clustering=False,
        n_threads=cores,
        effect=effect,
    )
    g.image_heterogeneity = True
    fits = [_survival_test(ctx, _cell.cell_rows(ctx, g, f, t), f, t, sv, adjust, covariates) for f, t in pairs]
    return {"table": _survival_table(fits, adjust, covariates), "fits": fits}


def _survival_setup(ctx, pheno, covariates):
    """The null Cox model (covariates only) and its martingale residuals, one per patient."""
    unit_first = np.array([int(np.argmax(ctx.image_unit == u)) for u in range(len(ctx.unit_labels))])
    time = pheno[".time"].to_numpy(float)[unit_first]
    event = pheno[".event"].to_numpy(float)[unit_first]
    if (pd.Series(pheno[".time"].to_numpy()).groupby(ctx.image_unit).nunique() > 1).any():
        raise ValueError("the survival outcome must be constant within each subject.")
    W = (
        np.zeros((len(time), 0))
        if covariates is None
        else _model_matrix(pheno.iloc[unit_first].reset_index(drop=True), covariates)[0]
    )
    ok_u = np.isfinite(time) & np.isfinite(event) & np.all(np.isfinite(W), axis=1)
    null = _core.cox_fit(time[ok_u], event[ok_u].astype(np.int32), W[ok_u].ravel(), W.shape[1])
    if not null["ok"]:
        raise RuntimeError("the null Cox model did not converge.")
    M = np.full(len(time), np.nan)
    M[ok_u] = null["martingale"]
    ev = np.where(np.isfinite(event), event, 0).astype(np.int32)
    return {"M": M, "time": time, "event": ev}


def _survival_test(ctx, rows, f, t, sv, adjust, covariates):
    """The survival test of one pair on given image rows."""
    m = len(ctx.unit_labels)
    u = _core.survival_test(rows, rows["unit"], m, sv["M"], sv["time"], sv["event"], np.zeros(0))
    x = _cell._share(ctx, rows, f) if adjust else np.zeros(0)
    # the adjusted test only where the unadjusted one was done (not where the excess has no variation: no_variation)
    shared = not (u["ok"] and len(x) > 1 and np.var(x, ddof=1) > 0)
    s = u if shared else _core.survival_test(rows, rows["unit"], m, sv["M"], sv["time"], sv["event"], x)
    parts = ([] if shared else ["abundance"]) + ([] if covariates is None else ["covariates"])
    return {
        "from": f,
        "to": t,
        "ok": s["ok"],
        "reason": s["reason"],
        "rows": rows,
        "surv": s,
        "unadjusted_surv": u,
        "adjusted_for": "+".join(parts) or "none",
    }


def _survival_table(fits, adjust, covariates):
    rows = []
    for o in fits:
        s, u = o["surv"], o["unadjusted_surv"]
        if not s["ok"]:
            continue
        rows.append(
            _cell.add_side(
                {
                    "from": o["from"],
                    "to": o["to"],
                    "score_coefficient": s["score_coef"],
                    "score_se": s["score_se"],
                    "score_df": s["score_df"],
                    "p_value": s["score_p"],
                    "hazard_ratio_sd": s["hr_sd"],
                    "log_hr_sd": s["log_hr_sd"],
                    "log_hr_se": s["hr_se"],
                    "hr_p_value": s["hr_p"],
                    "log_hr_per_unit": s["log_hr_unit"],
                    "tau2": s["tau2"],
                    "adjusted_for": o["adjusted_for"],
                    "unadjusted_p_value": u["score_p"] if u["ok"] else np.nan,
                    "unadjusted_hazard_ratio_sd": u["hr_sd"] if u["ok"] else np.nan,
                },
                o,
            )
        )
    tab = pd.DataFrame(rows) if rows else None
    if tab is not None:
        tab["p_adj"] = _cell.p_adjust_bh(tab["p_value"])
        tab["unadjusted_p_adj"] = _cell.p_adjust_bh(tab["unadjusted_p_value"])
        # the unadjusted columns are the test without the abundance adjustment (covariates enter both null Cox models)
        if not adjust:
            tab = tab.drop(columns=[c for c in tab.columns if c.startswith("unadjusted_")])
        if not adjust and covariates is None:
            tab = tab.drop(columns=["adjusted_for"])
        tab.index = tab["from"] + "__" + tab["to"]
    return tab


def _results(res, ctx, pheno, condition, subject, survival, radii, k, effect="count") -> SpicyResults:
    tab = res["table"]
    if tab is None:
        raise ValueError(
            "no pair could be tested (each condition needs at least two patients with both cell types, "
            "and enough effective patients)."
        )
    pa = _cell.cell_image_excess(res["fits"], ctx)
    w = _cell.cell_image_weight(res["fits"], ctx)
    return SpicyResults(
        cell_results=tab,
        levels=ctx.levels,
        survival=survival,
        radius_results=res.get("radius_table"),
        image_ids=list(ctx.image_labels),
        condition=(None if survival else [ctx.levels[g] if g >= 0 else None for g in ctx.image_group]),
        subject=(None if subject is None else _cell.as_str(pheno[subject]).tolist()),
        pairwise_assoc=pa,
        image_weights=w,
        r=None if k is not None else radii,
        k=k,
        effect=effect,
    )
