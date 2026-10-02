"""spicyR Cell: the Python side of the cell-level analysis.

A line-for-line mirror of spicyR's ``R/cell.R``. Everything numeric happens in the shared C++ core; these
functions only prepare inputs, call it and assemble tables, in the same order as the R package so that the
two give the same numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import _core


def p_adjust_bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, as R's ``p.adjust(p, "BH")`` (NaN kept in place)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if n == 0:
        return out
    q = p[ok]
    order = np.argsort(-q, kind="stable")
    ranks = np.arange(n, 0, -1)
    adj = np.minimum.accumulate(n / ranks * q[order])
    res = np.empty(n)
    res[order] = np.minimum(1.0, adj)
    out[ok] = res
    return out


def condition_levels(col: pd.Series) -> list[str]:
    """Levels present in a condition column: categories in order if categorical, else sorted (C locale)."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        present = set(col.dropna().astype(str))
        return [str(c) for c in col.cat.categories if str(c) in present]
    return sorted(set(col.dropna().astype(str)))


def enumerate_pairs(from_, to, all_types):
    """Every ordered (from, to) pair, from-major, as spicyR's enumerate_pairs(family = "binomial")."""
    if isinstance(from_, str) and isinstance(to, str):
        return [(from_, to)]
    ft = list(dict.fromkeys(all_types if from_ is None else ([from_] if isinstance(from_, str) else from_)))
    tt = list(dict.fromkeys(all_types if to is None else ([to] if isinstance(to, str) else to)))
    return [(f, t) for f in ft for t in tt]


@dataclass
class Context:
    df: pd.DataFrame
    image_labels: list
    image_codes: np.ndarray
    n_images: int
    unit_labels: list
    image_unit: np.ndarray
    first: np.ndarray
    type_labels: list
    counts: np.ndarray
    data: _core.Dataset
    levels: list | None = None
    image_group: np.ndarray | None = None


def cell_context(cells: pd.DataFrame, condition, subject, ref=None, survival=False) -> Context:
    image_chr = cells["imageID"].astype(str).to_numpy()
    image_labels = sorted(set(image_chr))
    lab_index = {s: i for i, s in enumerate(image_labels)}
    image_codes = np.array([lab_index[s] for s in image_chr], dtype=int)
    ord_ = np.argsort(image_codes, kind="stable")
    df = cells.iloc[ord_].reset_index(drop=True)
    image_codes = image_codes[ord_]
    n_images = len(image_labels)
    first = np.searchsorted(image_codes, np.arange(n_images))

    if subject is None or cells[subject].astype(str).nunique() == n_images:
        unit_labels = list(image_labels)
        image_unit = np.arange(n_images)
    else:
        sub = df[subject].astype(str).to_numpy()[first]
        unit_labels = list(dict.fromkeys(sub))
        ui = {s: i for i, s in enumerate(unit_labels)}
        image_unit = np.array([ui[s] for s in sub], dtype=int)

    levels = image_group = None
    if condition is not None and not survival:
        levels = condition_levels(cells[condition])
        if len(levels) < 2:
            raise ValueError(f"`condition` needs at least two levels; found {len(levels)}.")
        if ref is not None:
            if str(ref) not in levels:
                raise ValueError("`ref` is not a level of `condition`.")
            levels = [str(ref)] + [l for l in levels if l != str(ref)]
        cond = df[condition].astype(str).to_numpy()
        per_image = pd.Series(cond).groupby(image_codes).nunique()
        if (per_image > 1).any():
            raise ValueError(f"'{condition}' must be constant within each image.")
        li = {l: i for i, l in enumerate(levels)}
        image_group = np.array([li[c] for c in cond[first]], dtype=int)
        if (pd.Series(image_group).groupby(image_unit).nunique() > 1).any():
            raise ValueError(f"each subject must belong to a single '{condition}' level.")

    type_labels = list(dict.fromkeys(cells["cellType"].astype(str)))
    ti = {t: i for i, t in enumerate(type_labels)}
    type_codes = np.array([ti[t] for t in df["cellType"].astype(str)], dtype=np.int32)
    counts = np.zeros((n_images, len(type_labels)))
    np.add.at(counts, (image_codes, type_codes), 1.0)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(image_codes, minlength=n_images))]).astype(np.int32)
    data = _core.Dataset(df["x"].to_numpy(float), df["y"].to_numpy(float), type_codes, offsets, len(type_labels))
    return Context(df, image_labels, image_codes, n_images, unit_labels, image_unit, first, type_labels, counts,
                   data, levels, image_group)


@dataclass
class Graph:
    knn: bool
    totals: np.ndarray
    sq: np.ndarray
    psi: np.ndarray


def cell_graph(ctx: Context, pairs, r=None, k=None, label_clustering=True, window="convex", n_threads=1) -> Graph:
    knn = k is not None
    T = len(ctx.type_labels)
    if knn:
        ctx.data.build_knn(int(k), int(n_threads))
        scale = float(np.median(np.sqrt(k / (math.pi * ctx.counts.sum(axis=1) / ctx.data.image_areas(window)))))
        if label_clustering:
            ctx.data.build_radius_index(scale)
        h = 2 * scale
    else:
        ctx.data.build_radius_index(float(r))
        h = 2 * float(r)
    totals = ctx.data.pair_neighbour_totals(knn)
    sq = ctx.data.pair_neighbour_out_sq_totals(knn)
    if label_clustering:
        f = np.array([ctx.type_labels.index(p[0]) for p in pairs], dtype=np.int32)
        t = np.array([ctx.type_labels.index(p[1]) for p in pairs], dtype=np.int32)
        psi = _core.label_clustering_factor(ctx.data, f, t, ctx.counts, T, knn, h)
    else:
        psi = np.zeros(0)
    return Graph(knn, totals, sq, psi)


def cell_rows(ctx: Context, g: Graph, f: str, t: str) -> dict:
    rows = _core.excess_image_rows(g.totals, g.sq, ctx.counts, len(ctx.type_labels), ctx.type_labels.index(f),
                                   ctx.type_labels.index(t), g.knn, g.psi)
    rows["unit"] = ctx.image_unit[rows["img"]].astype(np.int32)
    if ctx.image_group is not None:
        rows["group"] = ctx.image_group[rows["img"]].astype(np.int32)
    return rows


def _subset_rows(rows, keep) -> dict:
    return {k: v[keep] for k, v in rows.items()}


def _share(ctx, rows, f):
    i = rows["img"]
    return np.log(np.maximum(ctx.counts[i, ctx.type_labels.index(f)], 0.5) / ctx.counts.sum(axis=1)[i])


def cell_pair_test(ctx, g, f, t, frailty, variance, availability, Z_extra=None) -> dict:
    rows = cell_rows(ctx, g, f, t)
    if len(ctx.levels) > 2:
        return cell_pair_test_levels(ctx, rows, f, t, frailty, availability, Z_extra)
    m = len(ctx.unit_labels)
    r = _core.excess_test(rows, rows["unit"], rows["group"], m, frailty, variance)
    out = {"from": f, "to": t, "ok": r["ok"], "reason": r["reason"], "rows": rows, "test": r}
    if not r["ok"]:
        return out
    if Z_extra is not None:
        # images with a missing covariate are left out of the covariate model
        ze = Z_extra[rows["img"], :]
        cc = np.all(np.isfinite(ze), axis=1)
        rc = _subset_rows(rows, cc)
        ze = ze[cc]
        Z = np.column_stack([rc["group"] == 0, rc["group"] == 1, ze]).astype(float)
        patient_level = all(pd.Series(ze[:, j]).groupby(rc["unit"]).nunique().eq(1).all() for j in range(ze.shape[1]))
        cvec = np.concatenate([[-1.0, 1.0], np.zeros(ze.shape[1])])
        out["covariate"] = _core.design_test(rc, rc["unit"], m, Z, cvec, -1.0 if patient_level else r["tau2"])
    if availability:
        share = _share(ctx, rows, f)
        out["availability"] = (_core.availability_test(rows, rows["unit"], rows["group"], m, share, r["tau2"])
                               if np.var(share, ddof=1) > 0 else {"ok": False, "reason": "share_constant"})
    return out


def cell_pair_test_levels(ctx, rows, f, t, frailty, availability, Z_extra) -> dict:
    G = len(ctx.levels)
    m = len(ctx.unit_labels)
    out = {"from": f, "to": t, "ok": False, "rows": rows, "levels": ctx.levels}
    per_level = pd.Series(rows["unit"]).groupby(rows["group"]).nunique().reindex(range(G))
    if per_level.isna().any() or (per_level < 2).any():
        out["reason"] = "one_patient_per_group"
        return out
    Zg = (rows["group"][:, None] == np.arange(G)[None, :]).astype(float)

    def fit(Z, tau2, rw=rows):
        q = Z.shape[1]
        tau = tau2
        res = {}
        for l in range(1, G):
            cvec = np.zeros(q)
            cvec[0], cvec[l] = -1.0, 1.0
            d = _core.design_test(rw, rw["unit"], m, Z, cvec, tau)
            if not d["ok"]:
                return None
            tau = d["tau2"]
            res[ctx.levels[l]] = d
        return res

    d = fit(Zg, -1.0 if frailty else 0.0)
    if d is None:
        out["reason"] = "design_not_full_rank"
        return out
    first = next(iter(d.values()))
    out.update(ok=True, levels_test=d, test={"coef_ref": first["theta"][0], "tau2": first["tau2"]})
    if Z_extra is not None:
        cc = np.all(np.isfinite(Z_extra[rows["img"], :]), axis=1)
        rc = _subset_rows(rows, cc)
        out["covariate_levels"] = fit(np.column_stack([Zg[cc], Z_extra[rc["img"], :]]), first["tau2"], rc)
    if availability:
        share = _share(ctx, rows, f)
        if np.var(share, ddof=1) > 0:
            out["availability_levels"] = fit(np.column_stack([Zg, share - share.mean()]), first["tau2"])
    return out


def _num(z, name):
    return np.nan if (z is None or not z.get("ok", False)) else z[name]


def cell_table(fits, ctx, availability, covariates) -> pd.DataFrame | None:
    if len(ctx.levels) > 2:
        return cell_table_levels(fits, ctx, availability, covariates)
    rows = []
    for o in fits:
        if not o["ok"]:
            continue
        x = o["test"]
        row = {"from": o["from"], "to": o["to"], "excess_ref": x["coef_ref"], "excess_comp": x["coef_comp"],
               "excess_difference": x["difference"], "se": x["se"], "df": x["df"], "p_value": x["p"], "tau2": x["tau2"]}
        if availability:
            a = o.get("availability")
            row.update(adjusted_difference=_num(a, "estimate"), adjusted_se=_num(a, "se"), adjusted_df=_num(a, "df"),
                       adjusted_p_value=_num(a, "p"))
        if covariates is not None:
            c = o.get("covariate")
            row.update(covariate_difference=_num(c, "estimate"), covariate_se=_num(c, "se"),
                       covariate_df=_num(c, "df"), covariate_p_value=_num(c, "p"))
        rows.append(row)
    if not rows:
        return None
    tab = pd.DataFrame(rows)
    tab["p_adj"] = p_adjust_bh(tab["p_value"])
    if availability:
        tab["adjusted_p_adj"] = p_adjust_bh(tab["adjusted_p_value"])
    if covariates is not None:
        tab["covariate_p_adj"] = p_adjust_bh(tab["covariate_p_value"])
    tab.index = tab["from"] + "__" + tab["to"]
    return tab


def cell_table_levels(fits, ctx, availability, covariates) -> pd.DataFrame | None:
    def num(z, l, nm):
        return np.nan if (z is None or l not in z) else z[l][nm]

    rows = []
    for o in fits:
        if not o["ok"]:
            continue
        for l, d in o["levels_test"].items():
            row = {"from": o["from"], "to": o["to"], "level": l, "excess_ref": d["theta"][0],
                   "excess_difference": d["estimate"], "se": d["se"], "df": d["df"], "p_value": d["p"], "tau2": d["tau2"]}
            if availability:
                row["adjusted_difference"] = num(o.get("availability_levels"), l, "estimate")
                row["adjusted_p_value"] = num(o.get("availability_levels"), l, "p")
            if covariates is not None:
                row["covariate_difference"] = num(o.get("covariate_levels"), l, "estimate")
                row["covariate_p_value"] = num(o.get("covariate_levels"), l, "p")
            rows.append(row)
    if not rows:
        return None
    tab = pd.DataFrame(rows)
    tab["p_adj"] = np.nan
    if availability:
        tab["adjusted_p_adj"] = np.nan
    for l in tab["level"].unique():
        k = (tab["level"] == l).to_numpy()
        tab.loc[k, "p_adj"] = p_adjust_bh(tab.loc[k, "p_value"])
        if availability:
            tab.loc[k, "adjusted_p_adj"] = p_adjust_bh(tab.loc[k, "adjusted_p_value"])
    tab.index = tab["from"] + "__" + tab["to"] + "__" + tab["level"]
    return tab


def cell_image_excess(fits, ctx) -> dict:
    out = {}
    for o in fits:
        v = np.full(ctx.n_images, np.nan)
        rows = o.get("rows")
        if rows is not None and len(rows["img"]):
            v[rows["img"]] = (rows["O"] - rows["E"]) / rows["n"]
        out[f"{o['from']}__{o['to']}"] = v
    return out
