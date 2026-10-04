"""``spicy()``: test for changes in the co-localisation of cell types between conditions.

The mirror of spicyR's ``R/spicy_main.R``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import _cell, _core
from ._input import format_data
from ._results import SpicyResults


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
    variance="auto",
    frailty=True,
    label_clustering=True,
    ref=None,
    cores=1,
    survival=None,
    table_key="table",
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
    once per pair from all images, without the conditions, and is reported in the ``side`` column.
    ``effect="count"`` gives the number of extra ``from`` cells within ``r`` of each ``to`` cell
    instead; it also reflects how many ``from`` cells surround a ``to`` cell (depth of infiltration), but it grows
    with how densely the ``from`` cells are packed. Images are combined within patients (``subject``) and patients
    within conditions by a frailty GEE, and the difference is tested with a CR2 variance on Satterthwaite degrees of
    freedom (Hartung-Knapp on m - 2 df when a condition has at most 5 patients), with **patients as the units**. The difference is adjusted for any ``covariates`` and, with
    ``adjust_abundance=True``, for the log share of the ``from`` type in each image; the unadjusted test is then
    reported alongside (``unadjusted_*`` columns). When nearly every cell has a ``from`` cell within ``r`` (q close to
    1), there is little room for attraction and an attracting pair's images carry little information; a smaller ``r``
    is more informative.

    ``from`` is spelled ``from_`` because ``from`` is a Python keyword (``**{"from": ...}`` also works).

    Parameters
    ----------
    cells
        A pandas DataFrame (one row per cell), an AnnData object (cells as observations; coordinates in
        ``obsm["spatial"]`` or obs columns) or a SpatialData object (its annotation table ``table_key``).
    condition
        Column of the image-level condition (two or more groups).
    subject
        Column of the patient of each image. Images of one patient are combined; by default each image is a
        patient.
    covariates
        Image- or patient-level columns to adjust for. The effect of each is reported as ``<column>_effect`` and
        ``<column>_p_value`` (one column per level after the first for a categorical covariate).
    r
        Radius, or several radii combined by ``combine`` ("maxT" or "cauchy"). Default 50.
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
        "auto" (default: "hartung_knapp" when a condition has at most 5 patients, "cr2" otherwise), "cr2" (CR2 on
        Satterthwaite df) or "hartung_knapp" (for very few patients: the model-based variance floored at CR2, on
        m - 2 df). The variance used is in ``.variance`` of the result.
    survival
        A survival outcome: ``(time_column, event_column)``, one value per patient (the R package's
        ``Surv(time, event)``). Leave ``condition`` empty. (Older form: ``condition=time_column,
        survival=event_column``.)
    """
    if "from" in kwargs:
        from_ = kwargs.pop("from")
    if kwargs:
        raise TypeError(f"unexpected arguments: {list(kwargs)}")
    if method != "cell":
        raise NotImplementedError(
            "method='image' (the original spicyR test) is not yet available in Python; use spicyR in R."
        )
    if combine not in ("maxT", "cauchy"):
        raise ValueError("combine must be 'maxT' or 'cauchy'.")
    if variance not in ("auto", "cr2", "hartung_knapp"):
        raise ValueError("variance must be 'auto', 'cr2' or 'hartung_knapp'.")
    if effect not in ("allocation", "count"):
        raise ValueError("effect must be 'allocation' or 'count'.")
    if r is None and k is None:
        r = 50
    if k is not None:
        r = None
    df = format_data(cells, image_id, cell_type, spatial_coords, table_key)
    if subject is not None and subject not in df.columns:
        raise ValueError("`subject` column not found.")
    is_surv = survival is not None
    if is_surv and not isinstance(survival, str):
        if len(survival) != 2:
            raise ValueError("survival must be (time_column, event_column).")
        condition, survival = survival[0], survival[1]
    if condition is None:
        raise ValueError("give `condition` (the column of the groups) or `survival` (time and event columns).")
    types = list(dict.fromkeys(_cell.as_str(df["cellType"])))
    bad = [
        t
        for t in ([from_] if isinstance(from_, str) else (from_ or [])) + ([to] if isinstance(to, str) else (to or []))
        if t not in types
    ]
    if bad:
        raise ValueError(f"cell type not found: {bad}")
    # from -> to: are `to` cells placed next to `from` cells (allocation: the extra fraction of `to` cells with a `from`
    # cell within r; count: extra `from` cells within r of each `to` cell), beyond the `to` cells being a random subset
    # of the cells that are not `from` cells. This is the core's own (counted, centre) order, so pairs pass through.
    pairs = list(_cell.enumerate_pairs(from_, to, types))
    if is_surv:
        df[".time"] = df[condition].astype(float)
        df[".event"] = df[survival].astype(float)
    ctx = _cell.cell_context(df, None if is_surv else condition, subject, ref=ref, survival=is_surv)
    pheno = ctx.df.iloc[ctx.first].reset_index(drop=True)
    if variance == "auto":
        # Hartung-Knapp (m - 2 df) when a condition has at most 5 patients, where CR2's Satterthwaite df are very
        # small; CR2 otherwise (Hartung-Knapp's m - 2 df overstate the information of rare pairs in larger cohorts)
        m_min = np.inf if is_surv else pd.Series(ctx.image_unit).groupby(ctx.image_group).nunique().min()
        variance = "hartung_knapp" if m_min <= 5 else "cr2"
        if variance == "hartung_knapp":
            warnings.warn(
                f'variance="auto": a condition has {m_min} patients; using the Hartung-Knapp variance on m - 2 df.',
                stacklevel=2,
            )

    Z_extra, extra_names = None, []
    if covariates is not None:
        covariates = [covariates] if isinstance(covariates, str) else list(covariates)
        miss = [c for c in covariates if c not in pheno.columns]
        if miss:
            raise ValueError(f"covariates not found: {miss}")
        Z_extra, extra_names = _model_matrix(pheno, covariates)  # NaN rows where a covariate is missing
        Z_extra = Z_extra - np.nanmean(Z_extra, axis=0)

    radii = [np.nan] if k is not None else sorted(set(np.atleast_1d(r).astype(float)))
    if len(radii) > 1 and not is_surv and len(ctx.levels) > 2:
        raise ValueError("several radii are supported for two conditions; give one `r`.")
    if is_surv:
        res = _survival(ctx, pairs, radii, k, pheno, covariates, label_clustering, cores, adjust_abundance, effect)
    else:
        per_r = []
        for rr in radii:
            g = _cell.cell_graph(
                ctx,
                pairs,
                r=None if np.isnan(rr) else rr,
                k=k,
                label_clustering=label_clustering,
                n_threads=cores,
                effect=effect,
            )
            per_r.append(
                [
                    _cell.cell_pair_test(ctx, g, p[0], p[1], frailty, variance, adjust_abundance, Z_extra, extra_names)
                    for p in pairs
                ]
            )
        res = _combine(per_r, radii, ctx, adjust_abundance or covariates is not None, combine)
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


def _combine(per_r, radii, ctx, adjusted, combine):
    """Several radii: per-radius tables, and one row per pair with the combined p-value.

    The p-value is combined for the main test and, when adjusted, for the unadjusted test. The other columns are
    those of the radius chosen by max-T.
    """
    tabs = [_cell.cell_table(fs, ctx, adjusted) for fs in per_r]
    if len(radii) == 1:
        return {"table": tabs[0], "fits": per_r[0]}
    long = pd.concat([t.assign(r=rr) for t, rr in zip(tabs, radii, strict=True) if t is not None])
    long = long[["r"] + [c for c in long.columns if c != "r"]]
    keys = long[["from", "to"]].drop_duplicates()

    def comb(tests):
        ok = [i for i, z in enumerate(tests) if z is not None]
        if not ok:
            return None
        tt = np.array([tests[i]["difference"] / tests[i]["se"] for i in ok])
        df_ = np.array([tests[i]["df"] for i in ok])
        pv = np.array([tests[i]["p"] for i in ok])
        if combine == "maxT":
            p, best = _core.max_t([list(tests[i]["influence"]) for i in ok], tt, df_)
        else:
            best = int(np.argmin(pv))
            p = _core.cauchy_combine(pv)
        return ok[best], p, pv[best]

    rows = []
    for f, t in keys.itertuples(index=False):
        fit_at = []
        for fs in per_r:
            o = next(z for z in fs if z["from"] == f and z["to"] == t)
            fit_at.append(o if o["ok"] else None)
        c = comb([None if o is None else o["test"] for o in fit_at])
        if c is None:
            continue
        best, p, p_best = c
        row = long[(long["from"] == f) & (long["to"] == t) & (long["r"] == radii[best])].iloc[0].to_dict()
        row.update(p_value=p, p_value_best_radius=p_best)
        if adjusted:
            u = comb([None if o is None else o["unadjusted"] for o in fit_at])
            row["unadjusted_p_value"] = np.nan if u is None else u[1]
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["p_adj"] = _cell.p_adjust_bh(tab["p_value"])
    if adjusted:
        tab["unadjusted_p_adj"] = _cell.p_adjust_bh(tab["unadjusted_p_value"])
    tab = tab.drop(columns=[c for c in ("unadjusted_difference", "unadjusted_se", "unadjusted_df") if c in tab.columns])
    tab.index = tab["from"] + "__" + tab["to"]
    mid = int(np.argmin(np.abs(np.array(radii) - np.median(radii))))
    return {"table": _cell.order_columns(tab), "fits": per_r[mid], "radius_table": long}


def _survival(ctx, pairs, radii, k, pheno, covariates, label_clustering, cores, adjust, effect="allocation"):
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
    rr = radii[0]
    g = _cell.cell_graph(
        ctx,
        pairs,
        r=None if np.isnan(rr) else rr,
        k=k,
        label_clustering=label_clustering,
        n_threads=cores,
        effect=effect,
    )
    fits, rows = [], []
    ev = np.where(np.isfinite(event), event, 0).astype(np.int32)
    for f, t in pairs:
        rws = _cell.cell_rows(ctx, g, f, t)
        m = len(ctx.unit_labels)
        u = _core.survival_test(rws, rws["unit"], m, M, time, ev, np.zeros(0))
        x = _cell._share(ctx, rws, f) if adjust else np.zeros(0)
        shared = not (len(x) > 1 and np.var(x, ddof=1) > 0)
        s = u if shared else _core.survival_test(rws, rws["unit"], m, M, time, ev, x)
        parts = ([] if shared else ["abundance"]) + ([] if covariates is None else ["covariates"])
        fits.append({"from": f, "to": t, "ok": s["ok"], "reason": s["reason"], "rows": rws, "surv": s})
        if s["ok"]:
            rows.append(
                _cell.add_side(
                {
                    "from": f,
                    "to": t,
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
                    "adjusted_for": "+".join(parts) or "none",
                    "unadjusted_p_value": u["score_p"] if u["ok"] else np.nan,
                    "unadjusted_hazard_ratio_sd": u["hr_sd"] if u["ok"] else np.nan,
                },
                fits[-1],
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
    return {"table": tab, "fits": fits}


def _results(res, ctx, pheno, condition, subject, survival, radii, k, effect="count") -> SpicyResults:
    tab = res["table"]
    if tab is None:
        raise ValueError("no pair could be tested (each condition needs at least two patients with both cell types).")
    pa = _cell.cell_image_excess(res["fits"], ctx)
    w = _cell.cell_image_weight(res["fits"], ctx)
    return SpicyResults(
        cell_results=tab,
        levels=ctx.levels,
        survival=survival,
        radius_results=res.get("radius_table"),
        image_ids=list(ctx.image_labels),
        condition=(None if survival else [ctx.levels[g] for g in ctx.image_group]),
        subject=(None if subject is None else _cell.as_str(pheno[subject]).tolist()),
        pairwise_assoc=pa,
        image_weights=w,
        r=None if k is not None else radii,
        k=k,
        effect=effect,
    )
