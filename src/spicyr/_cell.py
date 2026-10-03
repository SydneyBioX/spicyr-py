"""spicyR Cell: the Python side of the cell-level analysis.

A line-for-line mirror of spicyR's ``R/cell.R``. Everything numeric happens in the shared C++ core; these
functions only prepare inputs, call it and assemble tables, in the same order as the R package so that the
two give the same numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


def _r_str_value(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "NaN" if np.isnan(v) else f"{v:.15g}"
    return str(v)


def as_str(col: pd.Series) -> pd.Series:
    """A column as strings, the way R's as.character() writes it (2.0 is "2"), so that labels match spicyR's."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col.cat.rename_categories([_r_str_value(c) for c in col.cat.categories]).astype(str).where(col.notna())
    if pd.api.types.is_float_dtype(col):
        return col.map(_r_str_value).where(col.notna())
    return col.astype(str).where(col.notna())


def condition_levels(col: pd.Series) -> list[str]:
    """Levels present in a condition column: categories in order if categorical, else sorted (C locale)."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        present = set(as_str(col).dropna())
        return [c for c in (_r_str_value(c) for c in col.cat.categories) if c in present]
    return sorted(set(as_str(col).dropna()))


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
    image_chr = as_str(cells["imageID"]).to_numpy()
    image_labels = sorted(set(image_chr))
    lab_index = {s: i for i, s in enumerate(image_labels)}
    image_codes = np.array([lab_index[s] for s in image_chr], dtype=int)
    ord_ = np.argsort(image_codes, kind="stable")
    df = cells.iloc[ord_].reset_index(drop=True)
    image_codes = image_codes[ord_]
    n_images = len(image_labels)
    first = np.searchsorted(image_codes, np.arange(n_images))

    if subject is None or as_str(cells[subject]).nunique() == n_images:
        unit_labels = list(image_labels)
        image_unit = np.arange(n_images)
    else:
        sub = as_str(df[subject]).to_numpy()[first]
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
        cond = as_str(df[condition]).to_numpy()
        per_image = pd.Series(cond).groupby(image_codes).nunique()
        if (per_image > 1).any():
            raise ValueError(f"'{condition}' must be constant within each image.")
        li = {l: i for i, l in enumerate(levels)}
        image_group = np.array([li[c] for c in cond[first]], dtype=int)
        if (pd.Series(image_group).groupby(image_unit).nunique() > 1).any():
            raise ValueError(f"each subject must belong to a single '{condition}' level.")

    type_labels = list(dict.fromkeys(as_str(cells["cellType"])))
    ti = {t: i for i, t in enumerate(type_labels)}
    type_codes = np.array([ti[t] for t in as_str(df["cellType"])], dtype=np.int32)
    counts = np.zeros((n_images, len(type_labels)))
    np.add.at(counts, (image_codes, type_codes), 1.0)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(image_codes, minlength=n_images))]).astype(np.int32)
    data = _core.Dataset(df["x"].to_numpy(float), df["y"].to_numpy(float), type_codes, offsets, len(type_labels))
    return Context(
        df,
        image_labels,
        image_codes,
        n_images,
        unit_labels,
        image_unit,
        first,
        type_labels,
        counts,
        data,
        levels,
        image_group,
    )


@dataclass
class Graph:
    knn: bool
    effect: str
    psi: np.ndarray
    totals: np.ndarray | None = None
    sq: np.ndarray | None = None
    any: np.ndarray | None = None
    self_expected: np.ndarray | None = None


def cell_graph(
    ctx: Context, pairs, r=None, k=None, label_clustering=True, window="convex", n_threads=1, effect="allocation"
) -> Graph:
    """Neighbour sums and the label-clustering factor at one radius (or k).

    effect "allocation": for every cell, whether it has any cell of each type among its neighbours; "count": how many.
    """
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
    allocation = effect == "allocation"
    if allocation:
        g = Graph(knn, effect, np.zeros(0), any=ctx.data.pair_neighbour_any_totals(knn),
                  self_expected=ctx.data.self_any_expected(knn))
    else:
        g = Graph(knn, effect, np.zeros(0), totals=ctx.data.pair_neighbour_totals(knn),
                  sq=ctx.data.pair_neighbour_out_sq_totals(knn))
    if label_clustering:
        # psi of a `to` type is the median over every counted type (not only the requested pairs), so a pair's result
        # does not depend on which other pairs were asked for
        tos = list(dict.fromkeys(p[1] for p in pairs))
        f = np.array([ctx.type_labels.index(a) for t in tos for a in ctx.type_labels], dtype=np.int32)
        t = np.array([ctx.type_labels.index(t) for t in tos for _ in ctx.type_labels], dtype=np.int32)
        g.psi = _core.label_clustering_factor(ctx.data, f, t, ctx.counts, T, knn, h, allocation)
    return g


def cell_rows(ctx: Context, g: Graph, f: str, t: str) -> dict:
    T = len(ctx.type_labels)
    fc, tc = ctx.type_labels.index(f), ctx.type_labels.index(t)
    if g.effect == "allocation":
        rows = _core.allocation_image_rows(g.any, g.self_expected, ctx.counts, T, fc, tc, g.psi)
    else:
        rows = _core.excess_image_rows(g.totals, g.sq, ctx.counts, T, fc, tc, g.knn, g.psi)
    rows["unit"] = ctx.image_unit[rows["img"]].astype(np.int32)
    if ctx.image_group is not None:
        rows["group"] = ctx.image_group[rows["img"]].astype(np.int32)
    if g.effect == "allocation":
        # the pair's side, chosen in the core from all images (attraction scales by 1 - q, avoidance by q)
        rows["side"] = "avoid" if float(np.sum(rows["O"] - rows["E"])) < 0 else "attract"
    return rows


def _subset_rows(rows, keep) -> dict:
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in rows.items()}


def add_side(row: dict, o: dict) -> dict:
    """Add the allocation side of a fit to its table row."""
    rows = o.get("rows")
    if rows is not None and "side" in rows:
        row["side"] = rows["side"]
    return row


def _share(ctx, rows, f):
    """The log share of the counted type in each image row (the abundance covariate)."""
    i = rows["img"]
    return np.log(np.maximum(ctx.counts[i, ctx.type_labels.index(f)], 0.5) / ctx.counts.sum(axis=1)[i])


def _design(ctx, rows, f, adjust, Z_extra, extra_names):
    """The design of the main test.

    One indicator per condition, then the centred log share of the counted type (adjust) and the centred
    covariates. Images with a missing covariate are left out. None when there is nothing to adjust for.
    """
    cols, names = [], []
    if adjust:
        s = _share(ctx, rows, f)
        if len(s) > 1 and np.var(s, ddof=1) > 0:
            cols.append(s[:, None])
            names.append("abundance")
    keep = np.ones(len(rows["img"]), dtype=bool)
    if Z_extra is not None:
        ze = Z_extra[rows["img"], :]
        keep = np.all(np.isfinite(ze), axis=1)
        cols.append(ze)
        names += list(extra_names)
    if not cols:
        return None
    X = np.column_stack(cols)[keep]
    X = X - X.mean(axis=0)
    rw = _subset_rows(rows, keep)
    G = len(ctx.levels)
    Zg = (rw["group"][:, None] == np.arange(G)[None, :]).astype(float)
    parts = (["abundance"] if "abundance" in names[:1] else []) + (["covariates"] if Z_extra is not None else [])
    return {"rows": rw, "Z": np.column_stack([Zg, X]), "extra": names, "adjusted_for": "+".join(parts)}


def _design_tau2(d, G, frailty, tau2):
    """tau2 of the adjusted design.

    Re-estimated when every added column is constant within patients, else held at the unadjusted value
    (new_methods.pdf, Remark 3).
    """
    if not frailty:
        return 0.0
    X = d["Z"][:, G:]
    unit = d["rows"]["unit"]
    patient_level = all(pd.Series(X[:, j]).groupby(unit).nunique().eq(1).all() for j in range(X.shape[1]))
    return -1.0 if patient_level else float(tau2)


def _contrasts(G, q):
    """The level differences, then one row per added column (its effect)."""
    C = np.zeros((G - 1 + q - G, q))
    for l in range(1, G):
        C[l - 1, 0], C[l - 1, l] = -1.0, 1.0
    for j in range(q - G):
        C[G - 1 + j, G + j] = 1.0
    return C


def _design_tests(rows, m, Z, C, tau2, hk):
    return _core.design_tests(rows, rows["unit"], m, Z, C, C.shape[0], tau2, hk)


def cell_pair_test(ctx, g, f, t, frailty, variance, adjust, Z_extra=None, extra_names=()) -> dict:
    rows = cell_rows(ctx, g, f, t)
    if len(ctx.levels) > 2:
        return cell_pair_test_levels(ctx, rows, f, t, frailty, variance, adjust, Z_extra, extra_names)
    m = len(ctx.unit_labels)
    r = _core.excess_test(rows, rows["unit"], rows["group"], m, frailty, variance)
    out = {
        "from": f,
        "to": t,
        "ok": r["ok"],
        "reason": r["reason"],
        "rows": rows,
        "test": r,
        "unadjusted": r,
        "adjusted_for": "none",
    }
    if not r["ok"]:
        return out
    d = _design(ctx, rows, f, adjust, Z_extra, extra_names)
    if d is None:
        return out
    a = _design_tests(
        d["rows"],
        m,
        d["Z"],
        _contrasts(2, d["Z"].shape[1]),
        _design_tau2(d, 2, frailty, r["tau2"]),
        variance == "hartung_knapp",
    )
    # a design that is not of full rank (e.g. a covariate confounded with the condition): the unadjusted test
    if not a[0]["ok"]:
        out["adjusted_for"] = f"none ({a[0]['reason']})"
        return out
    x = a[0]
    out["test"] = {
        "coef_ref": x["theta"][0],
        "coef_comp": x["theta"][1],
        "difference": x["estimate"],
        "se": x["se"],
        "df": x["df"],
        "p": x["p"],
        "tau2": x["tau2"],
        "influence": x["influence"],
    }
    out["effects"] = dict(zip(d["extra"], a[1:], strict=True))
    out["adjusted_for"] = d["adjusted_for"]
    return out


def cell_pair_test_levels(ctx, rows, f, t, frailty, variance, adjust, Z_extra, extra_names) -> dict:
    """More than two conditions: each level tested against the reference (the first level).

    One design with an indicator per level, with the same adjustments as for two conditions.
    """
    G = len(ctx.levels)
    m = len(ctx.unit_labels)
    hk = variance == "hartung_knapp"
    out = {"from": f, "to": t, "ok": False, "rows": rows, "levels": ctx.levels, "adjusted_for": "none"}
    per_level = pd.Series(rows["unit"]).groupby(rows["group"]).nunique().reindex(range(G))
    if per_level.isna().any() or (per_level < 2).any():
        out["reason"] = "one_patient_per_group"
        return out
    Zg = (rows["group"][:, None] == np.arange(G)[None, :]).astype(float)
    u = _design_tests(rows, m, Zg, _contrasts(G, G), -1.0 if frailty else 0.0, hk)
    if not u[0]["ok"]:
        out["reason"] = "design_not_full_rank"
        return out
    un = dict(zip(ctx.levels[1:], u, strict=True))
    out.update(ok=True, levels_test=un, unadjusted_levels=un, test={"coef_ref": u[0]["theta"][0], "tau2": u[0]["tau2"]})
    d = _design(ctx, rows, f, adjust, Z_extra, extra_names)
    if d is None:
        return out
    a = _design_tests(
        d["rows"], m, d["Z"], _contrasts(G, d["Z"].shape[1]), _design_tau2(d, G, frailty, u[0]["tau2"]), hk
    )
    if not a[0]["ok"]:
        out["adjusted_for"] = f"none ({a[0]['reason']})"
        return out
    out["levels_test"] = dict(zip(ctx.levels[1:], a[: G - 1], strict=True))
    out["effects"] = dict(zip(d["extra"], a[G - 1 :], strict=True))
    out["test"] = {"coef_ref": a[0]["theta"][0], "tau2": a[0]["tau2"]}
    out["adjusted_for"] = d["adjusted_for"]
    return out


def _effect_names(fits):
    return list(dict.fromkeys(nm for o in fits for nm in o.get("effects", {})))


def _add_effects(row, o, enames):
    for nm in enames:
        e = o.get("effects", {}).get(nm)
        ok = e is not None and e["ok"]
        row[f"{nm}_effect"] = e["estimate"] if ok else np.nan
        row[f"{nm}_p_value"] = e["p"] if ok else np.nan
    return row


FIRST_COLUMNS = [
    "from",
    "to",
    "level",
    "r",
    "side",
    "excess_ref",
    "excess_comp",
    "excess_difference",
    "se",
    "df",
    "p_value",
    "p_adj",
    "p_value_best_radius",
    "tau2",
    "adjusted_for",
]


def order_columns(tab):
    """The main test first, then what it was adjusted for and the effects, then the unadjusted test."""
    first = [c for c in FIRST_COLUMNS if c in tab.columns]
    un = [c for c in tab.columns if c.startswith("unadjusted_")]
    return tab[first + [c for c in tab.columns if c not in first and c not in un] + un]


def cell_table(fits, ctx, adjusted) -> pd.DataFrame | None:
    if len(ctx.levels) > 2:
        return cell_table_levels(fits, ctx, adjusted)
    ok = [o for o in fits if o["ok"]]
    enames = _effect_names(ok)
    rows = []
    for o in ok:
        x = o["test"]
        row = {
            "from": o["from"],
            "to": o["to"],
            "excess_ref": x["coef_ref"],
            "excess_comp": x["coef_comp"],
            "excess_difference": x["difference"],
            "se": x["se"],
            "df": x["df"],
            "p_value": x["p"],
            "tau2": x["tau2"],
        }
        add_side(row, o)
        if adjusted:
            row["adjusted_for"] = o["adjusted_for"]
            _add_effects(row, o, enames)
            u = o["unadjusted"]
            row.update(
                unadjusted_difference=u["difference"],
                unadjusted_se=u["se"],
                unadjusted_df=u["df"],
                unadjusted_p_value=u["p"],
            )
        rows.append(row)
    if not rows:
        return None
    tab = pd.DataFrame(rows)
    tab["p_adj"] = p_adjust_bh(tab["p_value"])
    if adjusted:
        tab["unadjusted_p_adj"] = p_adjust_bh(tab["unadjusted_p_value"])
    tab.index = tab["from"] + "__" + tab["to"]
    return order_columns(tab)


def cell_table_levels(fits, ctx, adjusted) -> pd.DataFrame | None:
    """More than two conditions: one row per pair and level (contrast with the reference level)."""
    ok = [o for o in fits if o["ok"]]
    enames = _effect_names(ok)
    rows = []
    for o in ok:
        for l, d in o["levels_test"].items():
            row = {
                "from": o["from"],
                "to": o["to"],
                "level": l,
                "excess_ref": d["theta"][0],
                "excess_difference": d["estimate"],
                "se": d["se"],
                "df": d["df"],
                "p_value": d["p"],
                "tau2": d["tau2"],
            }
            add_side(row, o)
            if adjusted:
                row["adjusted_for"] = o["adjusted_for"]
                _add_effects(row, o, enames)
                u = o["unadjusted_levels"][l]
                row.update(
                    unadjusted_difference=u["estimate"],
                    unadjusted_se=u["se"],
                    unadjusted_df=u["df"],
                    unadjusted_p_value=u["p"],
                )
            rows.append(row)
    if not rows:
        return None
    tab = pd.DataFrame(rows)
    tab["p_adj"] = np.nan
    if adjusted:
        tab["unadjusted_p_adj"] = np.nan
    for l in tab["level"].unique():
        k = (tab["level"] == l).to_numpy()
        tab.loc[k, "p_adj"] = p_adjust_bh(tab.loc[k, "p_value"])
        if adjusted:
            tab.loc[k, "unadjusted_p_adj"] = p_adjust_bh(tab.loc[k, "unadjusted_p_value"])
    tab.index = tab["from"] + "__" + tab["to"] + "__" + tab["level"]
    return order_columns(tab)


def cell_image_excess(fits, ctx) -> dict:
    out = {}
    for o in fits:
        v = np.full(ctx.n_images, np.nan)
        rows = o.get("rows")
        if rows is not None and len(rows["img"]):
            v[rows["img"]] = (rows["O"] - rows["E"]) / rows["n"]
        out[f"{o['from']}__{o['to']}"] = v
    return out


def cell_image_weight(fits, ctx) -> dict:
    """Per-image weight of every pair.

    The image's share of its condition's information in the frailty model (sums to 1 within each condition);
    NaN where the pair was not tested.
    """
    out = {}
    for o in fits:
        v = np.full(ctx.n_images, np.nan)
        w = (o.get("unadjusted") or {}).get("image_weight")
        if w is not None and len(w):
            v[o["rows"]["img"]] = w
        out[f"{o['from']}__{o['to']}"] = v
    return out
