"""``spicy()``: test for changes in the co-localisation of cell types between conditions (mirror of spicyR's
``R/spicy_main.R``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _cell, _core
from ._input import format_data
from ._results import SpicyResults


def spicy(cells, condition, subject=None, covariates=None, image_id="imageID", cell_type="cellType",
          spatial_coords=("x", "y"), r=None, from_=None, to=None, method="cell", k=None, combine="maxT",
          availability=True, variance="cr2", frailty=True, label_clustering=True, ref=None, cores=1,
          survival=None, table_key="table", **kwargs) -> SpicyResults:
    """Test, for every ordered pair of cell types ``from_ -> to``, whether their co-localisation differs between
    conditions, or is associated with survival.

    The effect is the **excess**: the number of extra ``to`` cells within ``r`` of each ``from`` cell, beyond
    random labelling of the observed cells. Images are combined within patients (``subject``) and patients within
    conditions by a frailty GEE, and the difference is tested with a CR2 variance on Satterthwaite degrees of
    freedom, with **patients as the units**. The results also give the difference at equal availability of the
    ``to`` type (``adjusted_*`` columns).

    This is the Python twin of ``spicyR::spicy()``: the same C++ core, the same arguments (``from`` is spelled
    ``from_`` because ``from`` is a Python keyword; ``**{"from": ...}`` also works) and the same results.

    Parameters
    ----------
    cells
        A pandas DataFrame (one row per cell), an AnnData object (cells as observations; coordinates in
        ``obsm["spatial"]`` or obs columns) or a SpatialData object (its annotation table ``table_key``).
    condition
        Column of the image-level condition (two or more groups). For survival, leave ``condition`` as the
        name of the time column and pass ``survival=<event column>``.
    subject
        Column of the patient of each image. Images of one patient are combined; by default each image is a
        patient.
    covariates
        Image- or patient-level columns to adjust for.
    r
        Radius, or several radii combined by ``combine`` ("maxT" or "cauchy"). Default 50.
    from_, to
        Cell types to test (all ordered pairs by default).
    method
        "cell" (spicyR Cell). The image-level method of spicyR is not yet available in Python.
    k
        Use the ``k`` nearest neighbours instead of a radius.
    availability
        Also report the difference at equal availability of the ``to`` type.
    variance
        "cr2" (default) or "hartung_knapp" (for very few patients).
    survival
        Column of the event indicator (1 = event); ``condition`` is then the follow-up time.
    """
    if "from" in kwargs:
        from_ = kwargs.pop("from")
    if kwargs:
        raise TypeError(f"unexpected arguments: {list(kwargs)}")
    if method != "cell":
        raise NotImplementedError("method='image' (the original spicyR test) is not yet available in Python; use spicyR in R.")
    if combine not in ("maxT", "cauchy"):
        raise ValueError("combine must be 'maxT' or 'cauchy'.")
    if variance not in ("cr2", "hartung_knapp"):
        raise ValueError("variance must be 'cr2' or 'hartung_knapp'.")
    if r is None and k is None:
        r = 50
    if k is not None:
        r = None
    df = format_data(cells, image_id, cell_type, spatial_coords, table_key)
    if subject is not None and subject not in df.columns:
        raise ValueError("`subject` column not found.")
    is_surv = survival is not None
    types = list(dict.fromkeys(df["cellType"].astype(str)))
    bad = [t for t in ([from_] if isinstance(from_, str) else (from_ or [])) + ([to] if isinstance(to, str) else (to or []))
           if t not in types]
    if bad:
        raise ValueError(f"cell type not found: {bad}")
    # from -> to: extra `to` cells around each `from` cell; the core works in (counted, centre) order
    pairs = [(t, f) for f, t in _cell.enumerate_pairs(from_, to, types)]
    if is_surv:
        df[".time"] = df[condition].astype(float)
        df[".event"] = df[survival].astype(float)
    ctx = _cell.cell_context(df, None if is_surv else condition, subject, ref=ref, survival=is_surv)
    pheno = ctx.df.iloc[ctx.first].reset_index(drop=True)

    Z_extra = None
    if covariates is not None:
        covariates = [covariates] if isinstance(covariates, str) else list(covariates)
        miss = [c for c in covariates if c not in pheno.columns]
        if miss:
            raise ValueError(f"covariates not found: {miss}")
        Z_extra = _model_matrix(pheno, covariates)          # NaN rows where a covariate is missing
        Z_extra = Z_extra - np.nanmean(Z_extra, axis=0)

    radii = [np.nan] if k is not None else sorted(set(np.atleast_1d(r).astype(float)))
    if len(radii) > 1 and not is_surv and len(ctx.levels) > 2:
        raise ValueError("several radii are supported for two conditions; give one `r`.")
    if is_surv:
        res = _survival(ctx, pairs, radii, k, pheno, covariates, label_clustering, cores)
    else:
        per_r = []
        for rr in radii:
            g = _cell.cell_graph(ctx, pairs, r=None if np.isnan(rr) else rr, k=k, label_clustering=label_clustering,
                                 n_threads=cores)
            per_r.append([_cell.cell_pair_test(ctx, g, p[0], p[1], frailty, variance, availability, Z_extra) for p in pairs])
        res = _combine(per_r, radii, ctx, availability, covariates, combine)
    return _results(res, ctx, pheno, condition, subject, is_surv, radii, k)


def _model_matrix(pheno, covariates) -> np.ndarray:
    """R's model.matrix(~ covariates)[, -1] with treatment contrasts (levels sorted, the first dropped)."""
    cols = []
    for c in covariates:
        v = pheno[c]
        if pd.api.types.is_numeric_dtype(v) and not isinstance(v.dtype, pd.CategoricalDtype):
            cols.append(v.to_numpy(float)[:, None])
        else:
            lev = _cell.condition_levels(v)
            d = np.column_stack([(v.astype(str) == l).to_numpy(float) for l in lev[1:]])
            d[v.isna().to_numpy()] = np.nan
            cols.append(d)
    return np.column_stack(cols)


def _combine(per_r, radii, ctx, availability, covariates, combine):
    tabs = [_cell.cell_table(fs, ctx, availability, covariates) for fs in per_r]
    if len(radii) == 1:
        return {"table": tabs[0], "fits": per_r[0]}
    long = pd.concat([t.assign(r=rr) for t, rr in zip(tabs, radii) if t is not None])
    long = long[["r"] + [c for c in long.columns if c != "r"]]
    keys = long[["from", "to"]].drop_duplicates()
    rows = []
    for f, t in keys.itertuples(index=False):
        tests = []
        for fs in per_r:
            o = next(z for z in fs if z["from"] == f and z["to"] == t)
            tests.append(o["test"] if o["ok"] else None)
        ok = [i for i, z in enumerate(tests) if z is not None]
        if not ok:
            continue
        tt = np.array([tests[i]["difference"] / tests[i]["se"] for i in ok])
        df_ = np.array([tests[i]["df"] for i in ok])
        pv = np.array([tests[i]["p"] for i in ok])
        if combine == "maxT":
            infl = [list(tests[i]["influence"]) for i in ok]
            p, best = _core.max_t(infl, tt, df_)
        else:
            best = int(np.argmin(pv))
            p = _core.cauchy_combine(pv)
        z = tests[ok[best]]
        rows.append({"from": f, "to": t, "r": radii[ok[best]], "excess_ref": z["coef_ref"], "excess_comp": z["coef_comp"],
                     "excess_difference": z["difference"], "se": z["se"], "df": z["df"], "p_value": p, "tau2": z["tau2"],
                     "p_value_best_radius": pv[best]})
    tab = pd.DataFrame(rows)
    tab["p_adj"] = _cell.p_adjust_bh(tab["p_value"])
    if availability:
        key = long["from"] + "__" + long["to"]
        adj = long.groupby(key, sort=False)["adjusted_p_value"].apply(
            lambda p: _core.cauchy_combine(p[np.isfinite(p)].to_numpy()))
        tab["adjusted_p_value"] = adj.reindex(tab["from"] + "__" + tab["to"]).to_numpy()
        tab["adjusted_p_adj"] = _cell.p_adjust_bh(tab["adjusted_p_value"])
    tab.index = tab["from"] + "__" + tab["to"]
    mid = int(np.argmin(np.abs(np.array(radii) - np.median(radii))))
    return {"table": tab, "fits": per_r[mid], "radius_table": long}


def _survival(ctx, pairs, radii, k, pheno, covariates, label_clustering, cores):
    unit_first = np.array([int(np.argmax(ctx.image_unit == u)) for u in range(len(ctx.unit_labels))])
    time = pheno[".time"].to_numpy(float)[unit_first]
    event = pheno[".event"].to_numpy(float)[unit_first]
    if (pd.Series(pheno[".time"].to_numpy()).groupby(ctx.image_unit).nunique() > 1).any():
        raise ValueError("the survival outcome must be constant within each subject.")
    W = np.zeros((len(time), 0)) if covariates is None else _model_matrix(pheno.iloc[unit_first].reset_index(drop=True), covariates)
    ok_u = np.isfinite(time) & np.isfinite(event) & np.all(np.isfinite(W), axis=1)
    null = _core.cox_fit(time[ok_u], event[ok_u].astype(np.int32), W[ok_u].ravel(), W.shape[1])
    if not null["ok"]:
        raise RuntimeError("the null Cox model did not converge.")
    M = np.full(len(time), np.nan)
    M[ok_u] = null["martingale"]
    rr = radii[0]
    g = _cell.cell_graph(ctx, pairs, r=None if np.isnan(rr) else rr, k=k, label_clustering=label_clustering, n_threads=cores)
    fits, rows = [], []
    ev = np.where(np.isfinite(event), event, 0).astype(np.int32)
    for f, t in pairs:
        rws = _cell.cell_rows(ctx, g, f, t)
        s = _core.survival_test(rws, rws["unit"], len(ctx.unit_labels), M, time, ev)
        fits.append({"from": f, "to": t, "ok": s["ok"], "reason": s["reason"], "rows": rws, "surv": s})
        if s["ok"]:
            rows.append({"from": f, "to": t, "score_coefficient": s["score_coef"], "score_se": s["score_se"],
                         "score_df": s["score_df"], "p_value": s["score_p"], "hazard_ratio_sd": s["hr_sd"],
                         "log_hr_sd": s["log_hr_sd"], "log_hr_se": s["hr_se"], "hr_p_value": s["hr_p"],
                         "log_hr_per_cell": s["log_hr_unit"], "tau2": s["tau2"]})
    tab = pd.DataFrame(rows) if rows else None
    if tab is not None:
        tab["p_adj"] = _cell.p_adjust_bh(tab["p_value"])
        tab.index = tab["from"] + "__" + tab["to"]
    return {"table": tab, "fits": fits}


def _swap(d: pd.DataFrame | None):
    if d is None:
        return None
    d = d.copy()
    d["from"], d["to"] = d["to"].to_numpy(), d["from"].to_numpy()
    if "level" in d.columns:
        d.index = d["from"] + "__" + d["to"] + "__" + d["level"]
    elif not (d["from"] + "__" + d["to"]).duplicated().any():
        d.index = d["from"] + "__" + d["to"]
    else:
        d = d.reset_index(drop=True)
    return d


def _results(res, ctx, pheno, condition, subject, survival, radii, k) -> SpicyResults:
    tab = _swap(res["table"])
    if tab is None:
        raise ValueError("no pair could be tested (each condition needs at least two patients with both cell types).")
    pa = _cell.cell_image_excess(res["fits"], ctx)
    pa = {"__".join(reversed(key.split("__"))): v for key, v in pa.items()}
    return SpicyResults(cell_results=tab, levels=ctx.levels, survival=survival,
                        radius_results=_swap(res.get("radius_table")), image_ids=list(ctx.image_labels),
                        condition=(None if survival else [ctx.levels[g] for g in ctx.image_group]),
                        subject=(None if subject is None else pheno[subject].astype(str).tolist()),
                        pairwise_assoc=pa, r=None if k is not None else radii, k=k)
