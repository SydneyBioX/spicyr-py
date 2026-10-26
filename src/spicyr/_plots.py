"""Plots of the data: images with the cells of a pair (spicyR's ``plotImage()``)."""

from __future__ import annotations

import numpy as np

from ._input import format_data


def _density(x, y, bins=100):
    """The density of all cells: a 2-D histogram smoothed with a Gaussian kernel (bandwidth by Scott's rule).

    Returns the density per unit area (it integrates to one) and the extent of the grid.
    """
    xr, yr = (x.min(), x.max()), (y.min(), y.max())
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    sd = len(x) ** (-1 / 6)  # Scott's rule, per axis, in units of sd
    for axis, (lo, hi), v in ((0, xr, x), (1, yr, y)):
        s = max(sd * v.std() / ((hi - lo) / bins), 0.5)
        k = np.exp(-0.5 * (np.arange(-int(4 * s), int(4 * s) + 1) / s) ** 2)
        k = k / k.sum()
        h = np.apply_along_axis(np.convolve, axis, h, k, mode="same")
    area = max((xe[1] - xe[0]) * (ye[1] - ye[0]), np.finfo(float).tiny)
    return h / (len(x) * area), (xe[0], xe[-1], ye[0], ye[-1])


def plot_image(
    cells,
    image,
    from_=None,
    to=None,
    r=None,
    image_id="imageID",
    cell_type="cellType",
    spatial_coords=("x", "y"),
    table_key="table",
    ax=None,
    **kwargs,
):
    """Plot images, showing their cell types or the ``from_`` and ``to`` cells of a pair.

    Without ``from_`` and ``to``, every cell is shown, coloured by its cell type, with the same colour for a cell type
    in every image of ``cells``. With them, the density of all cells is shown in blue, with the ``from_`` cells
    (gold) and ``to`` cells (dark red) on top.
    With ``r``, a circle of radius ``r`` is drawn around each ``to`` cell: the ``from_`` cells inside the circles
    are those that :func:`spicyr.spicy` counts. Several images are drawn side by side, in the order given, on the
    same scale.

    Parameters
    ----------
    cells
        A pandas DataFrame, AnnData or SpatialData object, as for :func:`spicyr.spicy`.
    image
        The image to plot (a value of the ``image_id`` column), or a list of several.
    from_, to
        The two cell types (``from`` also works as a keyword). Optional, together.
    r
        Optional radius of the circles around the ``to`` cells, in the units of the coordinates.
    ax
        One image: the matplotlib axes to draw on.

    Returns
    -------
    One image: the matplotlib axes. Several: the matplotlib figure, with a panel per image.
    """
    import matplotlib.pyplot as plt

    if "from" in kwargs:
        from_ = kwargs.pop("from")
    if kwargs:
        raise TypeError(f"unexpected arguments: {list(kwargs)}")
    images = [image] if isinstance(image, str) or np.ndim(image) == 0 else list(dict.fromkeys(image))
    df = format_data(cells, image_id, cell_type, spatial_coords, table_key)
    ids = df["imageID"].astype(str)
    miss = [str(i) for i in images if str(i) not in set(ids)]
    if miss:
        raise ValueError(f"image not found in the {image_id!r} column: {', '.join(miss)}")
    if (from_ is None) != (to is None):
        raise ValueError("give both `from_` and `to`, or neither.")
    if from_ is None and r is not None:
        raise ValueError("`r` draws circles around the `to` cells: give `from_` and `to`.")
    if from_ is not None:
        types = set(df["cellType"].astype(str))
        bad = [t for t in (from_, to) if t not in types]
        if bad:
            raise ValueError(f"cell type not found: {bad}")
    if len(images) == 1:
        z = df[ids == str(images[0])]
        ax = ax or plt.figure(figsize=(5.5, 5)).gca()
        if from_ is None:
            _cell_types(ax, z, _palette(df))
        else:
            h, extent = _density(z["x"].to_numpy(float), z["y"].to_numpy(float))
            _pair(ax, z, h, extent, from_, to, r, h.max(), counts=True)
        ax.set_title(str(images[0]))
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            markerscale=3 if from_ is None else 2,
            fontsize=7 if from_ is None else None,
        )
        return ax
    if ax is not None:
        raise ValueError("`ax` draws one image: leave it out to plot several.")
    # several images: a panel each, as R's facet_wrap() lays them out, sharing the axes and the density scale
    k = len(images)
    nrow = 1 if k <= 3 else 2 if k <= 6 else 3 if k <= 12 else int(np.ceil(k / np.ceil(np.sqrt(k))))
    ncol = int(np.ceil(k / nrow))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(4.5 * ncol + 2, 4.2 * nrow),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    zs = [df[ids == str(i)] for i in images]
    if from_ is None:
        palette = _palette(df)
        for a, z in zip(axes.flat, zs, strict=False):
            _cell_types(a, z, palette)
    else:
        dens = [_density(z["x"].to_numpy(float), z["y"].to_numpy(float)) for z in zs]
        vmax = max(h.max() for h, _ in dens)
        for a, z, (h, extent) in zip(axes.flat, zs, dens, strict=False):
            _pair(a, z, h, extent, from_, to, r, vmax, counts=False)
    sel = df[ids.isin([str(i) for i in images])]
    pad = 0.02 * max(np.ptp(sel["x"].to_numpy(float)), np.ptp(sel["y"].to_numpy(float)))
    for a, im in zip(axes.flat, images, strict=False):
        a.set_title(str(im))
    for a in axes.flat[k:]:
        a.set_visible(False)
    axes[0, 0].set_xlim(sel["x"].min() - pad, sel["x"].max() + pad)
    axes[0, 0].set_ylim(sel["y"].min() - pad, sel["y"].max() + pad)
    # one legend for the figure, every cell type (or the pair) once
    handles = {}
    for a in axes.flat[:k]:
        for hd, lb in zip(*a.get_legend_handles_labels(), strict=True):
            handles.setdefault(lb, hd)
    if from_ is None:
        order = [t for t in palette if t in handles]
        fig.legend(
            [handles[t] for t in order], order, loc="outside right upper", frameon=False, markerscale=3, fontsize=7
        )
    else:
        fig.legend(list(handles.values()), list(handles), loc="outside right upper", frameon=False, markerscale=2)
    return fig


def _palette(df):
    """A colour for every cell type of ``df``, so that a cell type has the same colour in every image."""
    import matplotlib.pyplot as plt

    types = sorted(df["cellType"].astype(str).unique())
    colours = [*plt.cm.tab20.colors, *plt.cm.tab20b.colors, *plt.cm.tab20c.colors]
    return {t: colours[k % len(colours)] for k, t in enumerate(types)}


def _pair(ax, z, h, extent, from_, to, r, vmax, counts):
    """The density of the cells of image ``z`` in blue, its ``from_`` and ``to`` cells, and circles of radius ``r``."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    ax.imshow(h.T, origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=vmax, aspect="equal", zorder=0)
    ct = z["cellType"].astype(str)
    f, t = z[ct == from_], z[ct == to]
    if r is not None and len(t):
        circles = PatchCollection(
            [Circle((a, b), r) for a, b in zip(t["x"], t["y"], strict=True)],
            facecolor="none",
            edgecolor="#850f07",
            linewidth=0.5,
            alpha=0.7,
            zorder=1,
        )
        ax.add_collection(circles)
    ax.scatter(f["x"], f["y"], s=6, color="#d6b11c", label=f"{from_} ({len(f)})" if counts else from_, zorder=2)
    ax.scatter(t["x"], t["y"], s=6, color="#850f07", label=f"{to} ({len(t)})" if counts else to, zorder=3)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)


def _cell_types(ax, z, palette):
    """Every cell of image ``z``, coloured by cell type."""
    ct = z["cellType"].astype(str)
    for t, colour in palette.items():
        w = (ct == t).to_numpy()
        if w.any():
            ax.scatter(z["x"].to_numpy()[w], z["y"].to_numpy()[w], s=2, color=colour, label=t)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
