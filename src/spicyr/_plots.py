"""Plots of the data: one image with the cells of a pair (spicyR's ``plotImage()``)."""

from __future__ import annotations

import numpy as np

from ._input import format_data


def _density(ax, x, y, bins=100):
    """The density of all cells: a 2-D histogram smoothed with a Gaussian kernel (bandwidth by Scott's rule)."""
    xr, yr = (x.min(), x.max()), (y.min(), y.max())
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    sd = len(x) ** (-1 / 6)  # Scott's rule, per axis, in units of sd
    for axis, (lo, hi), v in ((0, xr, x), (1, yr, y)):
        s = max(sd * v.std() / ((hi - lo) / bins), 0.5)
        k = np.exp(-0.5 * (np.arange(-int(4 * s), int(4 * s) + 1) / s) ** 2)
        k = k / k.sum()
        h = np.apply_along_axis(np.convolve, axis, h, k, mode="same")
    ax.imshow(h.T, origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap="Blues", aspect="equal", zorder=0)


def plot_image(
    cells,
    image,
    from_,
    to,
    r=None,
    image_id="imageID",
    cell_type="cellType",
    spatial_coords=("x", "y"),
    table_key="table",
    ax=None,
    **kwargs,
):
    """Plot one image, showing the ``from_`` and ``to`` cells of a pair.

    The density of all cells is shown in blue, with the ``from_`` cells (gold) and ``to`` cells (dark red) on top.
    With ``r``, a circle of radius ``r`` is drawn around each ``to`` cell: the ``from_`` cells inside the circles
    are those that :func:`spicyr.spicy` counts.

    Parameters
    ----------
    cells
        A pandas DataFrame, AnnData or SpatialData object, as for :func:`spicyr.spicy`.
    image
        The image to plot (a value of the ``image_id`` column).
    from_, to
        The two cell types (``from`` also works as a keyword).
    r
        Optional radius of the circles around the ``to`` cells, in the units of the coordinates.

    Returns
    -------
    The matplotlib axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    if "from" in kwargs:
        from_ = kwargs.pop("from")
    if kwargs:
        raise TypeError(f"unexpected arguments: {list(kwargs)}")
    df = format_data(cells, image_id, cell_type, spatial_coords, table_key)
    z = df[df["imageID"].astype(str) == str(image)]
    if z.empty:
        raise ValueError(f"image {image!r} not found in the {image_id!r} column.")
    types = set(df["cellType"].astype(str))
    bad = [t for t in (from_, to) if t not in types]
    if bad:
        raise ValueError(f"cell type not found: {bad}")
    ax = ax or plt.figure(figsize=(5.5, 5)).gca()
    _density(ax, z["x"].to_numpy(float), z["y"].to_numpy(float))
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
    ax.scatter(f["x"], f["y"], s=6, color="#d6b11c", label=f"{from_} ({len(f)})", zorder=2)
    ax.scatter(t["x"], t["y"], s=6, color="#850f07", label=f"{to} ({len(t)})", zorder=3)
    ax.set_aspect("equal")
    ax.set_title(str(image))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1), markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
