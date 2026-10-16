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

from . import _core, _source


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


# as_str() lives with the cell sources (labels of image and cell-type codes); kept here for the callers of _cell
as_str = _source.as_str
_r_str_value = _source._r_str_value


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


# cells per batch of whole images in cell_stream(), per thread: the coordinates read at a time stay small
BATCH_CELLS = 1 << 21


@dataclass
class Context:
    source: object  # the cells (a _source.FrameSource or ZarrSource): codes, per-image columns, coordinates
    image_labels: list
    n_images: int
    unit_labels: list
    image_unit: np.ndarray
    first: np.ndarray  # first row of each image (sorted order)
    type_labels: list
    counts: np.ndarray
    image_map: np.ndarray  # image (sorted order) of each image code; -1 if absent
    type_map: np.ndarray  # cell type of each type code, the last entry for a missing type
    image_rows: np.ndarray | None  # per image (sorted order): its rows [start, end) when its cells are contiguous
    image_order: np.ndarray  # images in order of their first row
    data: _core.Dataset | None = None
    levels: list | None = None
    image_group: np.ndarray | None = None
    images: pd.DataFrame | None = None  # one row per image (index: image IDs) for columns not in the cell table

    def has(self, name) -> bool:
        return self.source.has(name) or (self.images is not None and name in self.images.columns)

    def per_image(self, name) -> pd.Series:
        """A column's value in each image (its first cell), or from the per-image table."""
        if self.source.has(name):
            return self.source.at_rows(name, self.first)
        return self.images[name].reset_index(drop=True)

    def pheno(self, names) -> pd.DataFrame:
        """One row per image (sorted order) with the columns `names`."""
        return pd.DataFrame({c: self.per_image(c) for c in dict.fromkeys(names)}, index=range(self.n_images))


def _image_table(image_info, image_labels, source=None, image_id=None):
    """The per-image table, one row per image in the analysis's order.

    Image IDs are its ``image_id`` column, or else its index. Every image needs a row, and every row an image.
    """
    if image_info is None:
        return None
    if isinstance(image_info, str):
        tables = getattr(source, "root", None)
        if tables is not None and "tables" in tables and image_info in tables["tables"]:
            import anndata as ad

            image_info = ad.io.read_elem(tables["tables"][image_info]["obs"])
        else:
            raise ValueError(f"no per-image table '{image_info}'.")
    if type(image_info).__name__ == "AnnData":
        image_info = image_info.obs
    if not isinstance(image_info, pd.DataFrame):
        raise TypeError("`image_info` must be a DataFrame with one row per image (index: the image IDs).")
    ids = image_info[image_id] if image_id is not None and image_id in image_info.columns else image_info.index
    key = as_str(pd.Series(np.asarray(ids)))
    if key.duplicated().any():
        raise ValueError("`image_info` has more than one row for an image.")
    unknown = [k for k in key if k not in set(image_labels)]
    if unknown:
        raise ValueError(f"images in `image_info` not found among the cells: {unknown[:5]}")
    pos = pd.Index(key.to_numpy())
    idx = pos.get_indexer(image_labels)
    if (idx < 0).any():
        miss = [image_labels[i] for i in np.flatnonzero(idx < 0)[:5]]
        raise ValueError(f"images missing from `image_info`: {miss}")
    return image_info.iloc[idx].reset_index(drop=True)


def _values(ctx, name, first_row_code):
    """A per-image column at the images' first rows, and the column over all cells when it varies within images.

    Returns (per-image values, all values): all values is the per-image values when the column is constant within
    images (then the values present are those of the images' first rows), so the whole column is read only otherwise.
    """
    if not ctx.source.has(name):
        v = ctx.per_image(name)
        return v, v, None
    v = ctx.per_image(name)
    if ctx.source.constant_within(name, first_row_code):
        return v, v, None
    full = ctx.source.column(name)
    return v, full, ctx.image_map[ctx.source.image_codes]


def cell_context(
    cells, condition, subject, ref=None, survival=False, dataset=False, image_info=None, by_image=True, image_id=None
) -> Context:
    """Image-level layout shared by every analysis of a data set.

    Integer codes for images (in the C-locale order of their IDs), types and units (patients), the per-image
    condition, and the cells per image and type. The cells stay in their source, in input order, with no sorted
    copy: one pass gives the cells and the first row of every (image, type), and the neighbour sums are taken image
    by image (cell_stream()). The whole-data Dataset (ctx.data, for the label-clustering factor) is built only with
    ``dataset=True``. Cell types are in order of first appearance; by_image=True: with the cells sorted stably by
    image in order of first appearance (as format_data() sorts them), else in row order. `cells` is a source
    (_source.cell_source()) or a DataFrame with columns imageID, cellType, x and y.
    """
    src = (
        cells
        if hasattr(cells, "image_codes")
        else _source.FrameSource(cells, "imageID", "cellType", cells["x"], cells["y"])
    )
    n_icodes, n_tcodes = len(src.image_labels), len(src.type_labels)
    lay = _core.cell_layout(src.image_codes, n_icodes, src.type_codes, n_tcodes)
    cnt, frs = lay["counts"], lay["first"]
    n_code = cnt.sum(axis=1)
    present = np.flatnonzero(n_code > 0)
    image_labels = sorted(src.image_labels[c] for c in present)  # C-locale (code point) order, as spicyR
    n_images = len(image_labels)
    code_pos = {s: c for c, s in enumerate(src.image_labels)}
    code_of = np.array([code_pos[s] for s in image_labels], dtype=np.int64)
    image_map = np.full(n_icodes, -1, dtype=np.int32)
    image_map[code_of] = np.arange(n_images, dtype=np.int32)
    first_row_code = np.where(frs >= 0, frs, np.iinfo(np.int64).max).min(axis=1)
    first_row_code[n_code == 0] = -1
    first = first_row_code[code_of]
    last = lay["last"][code_of]
    n_img = n_code[code_of]
    contiguous = bool(np.all(last - first + 1 == n_img))
    image_rows = np.column_stack([first, last + 1]) if contiguous else None
    image_order = np.argsort(first, kind="stable")  # images in order of first appearance

    ctx = Context(
        source=src,
        image_labels=image_labels,
        n_images=n_images,
        unit_labels=list(image_labels),
        image_unit=np.arange(n_images),
        first=first,
        type_labels=[],
        counts=np.zeros((n_images, 0)),
        image_map=image_map,
        type_map=np.zeros(0, np.int32),
        image_rows=image_rows,
        image_order=image_order,
    )
    ctx.images = _image_table(image_info, image_labels, src, image_id)

    # units: patients if `subject` maps several images to one patient, else images
    if subject is not None:
        if not ctx.has(subject):
            raise ValueError("`subject` column not found.")
        sv, sv_all, _ = _values(ctx, subject, first_row_code)
        if as_str(sv_all).nunique() != n_images:
            sub = as_str(sv).to_numpy()
            ctx.unit_labels = list(dict.fromkeys(sub))
            ui = {s: i for i, s in enumerate(ctx.unit_labels)}
            ctx.image_unit = np.array([ui[s] for s in sub], dtype=int)

    if condition is not None and not survival:
        cv, cv_all, img_of_row = _values(ctx, condition, first_row_code)
        levels = condition_levels(cv_all)
        if len(levels) < 2:
            raise ValueError(f"`condition` needs at least two levels; found {len(levels)}.")
        if ref is not None:
            if str(ref) not in levels:
                raise ValueError("`ref` is not a level of `condition`.")
            levels = [str(ref)] + [l for l in levels if l != str(ref)]
        if img_of_row is not None:
            per_image = pd.Series(as_str(cv_all).to_numpy()).groupby(img_of_row).nunique()
            if (per_image > 1).any():
                raise ValueError(f"'{condition}' must be constant within each image.")
        li = {l: i for i, l in enumerate(levels)}
        ctx.levels = levels
        # an image with a missing condition gets -1: the tests leave it out
        ctx.image_group = np.array([li.get(c, -1) for c in as_str(cv).to_numpy()], dtype=int)
        if (pd.Series(ctx.image_group).groupby(ctx.image_unit).nunique() > 1).any():
            raise ValueError(f"each subject must belong to a single '{condition}' level.")

    # cell types present (the last column of the layout is a missing type), by first appearance
    tcol = np.flatnonzero(cnt.sum(axis=0) > 0)
    if by_image:
        rank = np.empty(n_icodes, dtype=np.int64)
        rank[code_of[image_order]] = np.arange(n_images)
        keys = []
        for t in tcol:
            codes_t = present[cnt[present, t] > 0]
            c = codes_t[np.argmin(rank[codes_t])]
            keys.append((rank[c], frs[c, t]))
    else:
        keys = [(frs[cnt[:, t] > 0, t].min(),) for t in tcol]
    tcol = tcol[sorted(range(len(tcol)), key=lambda i: keys[i])]
    ctx.type_labels = [src.type_labels[t] if t < n_tcodes else np.nan for t in tcol]
    ctx.type_map = np.full(n_tcodes + 1, -1, dtype=np.int32)
    ctx.type_map[tcol] = np.arange(len(tcol), dtype=np.int32)
    ctx.counts = cnt[np.ix_(code_of, tcol)].astype(float)
    if dataset:
        order = np.argsort(image_map[src.image_codes], kind="stable")  # cells sorted (stably) by image
        x, y = src.coords(order)
        tc = src.type_codes[order].astype(np.int64)
        tc[tc < 0] = n_tcodes
        offsets = np.concatenate([[0], np.cumsum(ctx.counts.sum(axis=1))]).astype(np.int32)
        ctx.data = _core.Dataset(
            np.asarray(x, float), np.asarray(y, float), ctx.type_map[tc], offsets, len(ctx.type_labels)
        )
    return ctx


def _batches(ctx, budget):
    """Batches of whole images of about `budget` cells (one image at least).

    Contiguous images (each image's cells in one run of rows) are taken in row order and read as one slice of rows;
    otherwise images are taken in sorted order and their rows found with one pass over the image codes.
    """
    n_img = ctx.counts.sum(axis=1)
    out, cur, n = [], [], 0
    order = ctx.image_order if ctx.image_rows is not None else np.arange(ctx.n_images)
    for i in order:
        if cur and n + n_img[i] > budget:
            out.append(cur)
            cur, n = [], 0
        cur.append(int(i))
        n += n_img[i]
    if cur:
        out.append(cur)
    return out


def _batch_rows(ctx, images):
    if ctx.image_rows is not None:
        return slice(int(ctx.image_rows[images[0], 0]), int(ctx.image_rows[images[-1], 1]))
    keep = np.zeros(len(ctx.image_map), dtype=bool)
    keep[np.isin(ctx.image_map, images)] = True
    return _core.rows_of_images(ctx.source.image_codes, keep)


def cell_stream(ctx: Context, radii=None, k=None, allocation=True, n_threads=1, batch_cells=None) -> list:
    """The neighbour totals of every radius (or of k), image by image from the cells' source.

    One element per radius (one for kNN): (any, self_expected) or (totals, sq), exactly as the whole-data Dataset
    gives them. Batches of whole images (about `batch_cells` cells per thread) are read one after another, the next
    while the core sums the current one; in the core each image runs in a Dataset of its own, the images shared among
    `n_threads` threads, so the results do not depend on the threads.
    """
    T = len(ctx.type_labels)
    knn = k is not None
    rad = np.zeros(0) if knn else np.asarray(np.atleast_1d(radii), float)
    K = 1 if knn else len(rad)
    out_a = [np.zeros(ctx.n_images * T * T) for _ in range(K)]
    out_b = [np.zeros(ctx.n_images * T * (1 if allocation else T)) for _ in range(K)]
    n_threads = max(1, int(n_threads))
    batches = _batches(ctx, (batch_cells or BATCH_CELLS) * n_threads)
    src = ctx.source

    def read(images):
        rows = _batch_rows(ctx, images)
        x, y = src.coords(rows)
        return x, y, np.ascontiguousarray(src.image_codes[rows]), np.ascontiguousarray(src.type_codes[rows])

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(1) as reader:
        ahead = reader.submit(read, batches[0])
        for i in range(len(batches)):
            x, y, im, ty = ahead.result()
            if i + 1 < len(batches):
                ahead = reader.submit(read, batches[i + 1])
            _core.stream_batch(
                x,
                y,
                im,
                ty,
                ctx.image_map,
                ctx.type_map,
                T,
                rad,
                int(k) if knn else 0,
                allocation,
                out_a,
                out_b,
                n_threads,
            )
            del x, y, im, ty
    return list(zip(out_a, out_b, strict=True))


@dataclass
class Graph:
    knn: bool
    effect: str
    psi: np.ndarray
    totals: np.ndarray | None = None
    sq: np.ndarray | None = None
    any: np.ndarray | None = None
    self_expected: np.ndarray | None = None
    image_heterogeneity: bool = False


def _graph(knn, effect, a, b):
    if effect == "allocation":
        return Graph(knn, effect, np.zeros(0), any=a, self_expected=b)
    return Graph(knn, effect, np.zeros(0), totals=a, sq=b)


def cell_graphs_radii(ctx: Context, radii, effect: str = "allocation", n_threads=1) -> list:
    """Several radii from one neighbour search, image by image.

    One Graph per radius, each identical to cell_graph(ctx, pairs, r=radius, label_clustering=False, effect=effect).
    """
    return [
        _graph(False, effect, a, b)
        for a, b in cell_stream(ctx, radii=radii, allocation=effect == "allocation", n_threads=n_threads)
    ]


def cell_graph(
    ctx: Context, pairs, r=None, k=None, label_clustering=True, window="convex", n_threads=1, effect="allocation"
) -> Graph:
    """Neighbour sums and the label-clustering factor at one radius (or k).

    effect "allocation": for every cell, whether it has any cell of each type among its neighbours; "count": how many.
    Without the label-clustering factor (as spicy()) the sums are taken image by image (cell_stream()); the factor
    needs the whole-data Dataset (ctx.data, from cell_context(dataset=True)).
    """
    knn = k is not None
    allocation = effect == "allocation"
    if not label_clustering:
        a, b = cell_stream(ctx, radii=None if knn else r, k=k, allocation=allocation, n_threads=n_threads)[0]
        return _graph(knn, effect, a, b)
    if ctx.data is None:
        raise ValueError("the label-clustering factor needs cell_context(dataset=True).")
    T = len(ctx.type_labels)
    if knn:
        ctx.data.build_knn(int(k), int(n_threads))
        scale = float(np.median(np.sqrt(k / (math.pi * ctx.counts.sum(axis=1) / ctx.data.image_areas(window)))))
        ctx.data.build_radius_index(scale)
        h = 2 * scale
    else:
        ctx.data.build_radius_index(float(r))
        h = 2 * float(r)
    if allocation:
        g = _graph(knn, effect, ctx.data.pair_neighbour_any_totals(knn), ctx.data.self_any_expected(knn))
    else:
        g = _graph(knn, effect, ctx.data.pair_neighbour_totals(knn), ctx.data.pair_neighbour_out_sq_totals(knn))
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
    side = None
    if g.effect == "allocation":
        # the pair's side, chosen in the core from all images (attraction scales by 1 - q, avoidance by q)
        side = "avoid" if float(np.sum(rows["O"] - rows["E"])) < 0 else "attract"
    # images within a patient: each row becomes its effect (O - E) / n with variance v / n^2 + tau2_image
    if g.image_heterogeneity and len(rows["O"]):
        h = _core.image_heterogeneity_rows(rows, rows["unit"])
        for key in ("unit", "group"):
            if key in rows:
                h[key] = rows[key]
        rows = h
    if side is not None:
        rows["side"] = side
    return rows


def _subset_rows(rows, keep) -> dict:
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in rows.items()}


def add_side(row: dict, o: dict) -> dict:
    """Add the allocation side and the between-image variance of a fit to its table row."""
    rows = o.get("rows")
    if rows is not None and "side" in rows:
        row["side"] = rows["side"]
    if rows is not None and "tau2_image" in rows:
        row["tau2_image"] = rows["tau2_image"]
    return row


def _share(ctx, rows, f):
    """The log share of the counted type in each image row (the abundance covariate)."""
    i = rows["img"]
    return np.log(np.maximum(ctx.counts[i, ctx.type_labels.index(f)], 0.5) / ctx.counts.sum(axis=1)[i])


def _design(ctx, rows, f, adjust, Z_extra, extra_names):
    """The design of the main test.

    One indicator per condition, then the centred log share of the counted type (adjust) and the centred
    covariates. Images with a missing condition or covariate are left out, as the unadjusted test leaves out
    images with a missing condition. None when there is nothing to adjust for.
    """
    cols, names = [], []
    keep = rows["group"] >= 0 if "group" in rows else np.ones(len(rows["img"]), dtype=bool)
    if adjust:
        s = _share(ctx, rows, f)
        if keep.sum() > 1 and np.var(s[keep], ddof=1) > 0:
            cols.append(s[:, None])
            names.append("abundance")
    if Z_extra is not None:
        ze = Z_extra[rows["img"], :]
        keep = keep & np.all(np.isfinite(ze), axis=1)
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
    # no variation (the unadjusted test is 0 / 0): an adjusted design would only fit rounding noise; not tested
    if not r["se"] > 0:
        out["test"] = dict(r, p=np.nan)
        out["adjusted_for"] = "none (no_variation)"
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
    # not of full rank, or no variation (the condition means fit the response exactly; the design test checks it, so no
    # adjusted design is fitted then either)
    if not u[0]["ok"]:
        out["reason"] = "no_variation" if u[0]["reason"] == "no_variation" else "design_not_full_rank"
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
    "tau2_image",
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
