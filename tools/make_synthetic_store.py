"""Write a large synthetic SpatialData store for timing spicy() on data that do not fit in memory.

    python tools/make_synthetic_store.py OUT.zarr --images 500 --cells 2000000 [--types 10] [--chunk 1048576]

Cells are uniform in square images at 0.003 cells per square micron (side sqrt(cells / 0.003)), cell types t01, t02,
... equally likely, one image per patient, the first half of the images in condition "a" and the rest in "b". The
table "table" annotates one region per image (obs "region", categorical i001, i002, ...; rows grouped by region in
that order) with obs "cell_type", "condition" and "subject" (categorical), "cell_id" (the instance key, the row
number) and float32 coordinates in obsm["spatial"]; the table "images" has one row per image (index: the region)
with its condition and subject. Arrays are chunked by rows (`--chunk`) and written one chunk at a time, so memory
stays at a few chunks whatever the size. The regions are not elements of the store (it holds only the tables).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np


def _template(path: Path, images: list[str], types: list[str]):
    """Write a store with one cell per image through spatialdata, for the metadata of every element."""
    import warnings

    import anndata as ad
    import pandas as pd
    import spatialdata as sd
    from spatialdata.models import TableModel

    m = len(images)
    cond = ["a" if i < m / 2 else "b" for i in range(m)]
    obs = pd.DataFrame(
        {
            "region": pd.Categorical(images, categories=images),
            "cell_type": pd.Categorical([types[i % len(types)] for i in range(m)], categories=types),
            "condition": pd.Categorical(cond, categories=["a", "b"]),
            "subject": pd.Categorical(images, categories=images),
            "cell_id": np.arange(m, dtype=np.int64),
        },
        index=[str(i) for i in range(m)],
    )
    table = TableModel.parse(
        ad.AnnData(obs=obs, obsm={"spatial": np.zeros((m, 2), np.float32)}),
        region=images,
        region_key="region",
        instance_key="cell_id",
    )
    per_image = pd.DataFrame(
        {
            "condition": pd.Categorical(cond, categories=["a", "b"]),
            "subject": pd.Categorical(images, categories=images),
        },
        index=images,
    )
    sdata = sd.SpatialData(tables={"table": table, "images": TableModel.parse(ad.AnnData(obs=per_image))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the regions are not elements of the store
        sdata.write(path)


def _replace(group, name, shape, chunks, dtype, attrs):
    """Replace an array of the template by an empty one of the full size, with the same codec as spatialdata's."""
    from zarr.codecs import ZstdCodec

    if name in group:
        del group[name]
    return group.create_array(
        name, shape=shape, chunks=chunks, dtype=dtype, compressors=ZstdCodec(level=0), fill_value=0, attributes=attrs
    )


def write_store(path, n_images, cells, n_types=10, chunk=1 << 20, density=0.003, seed=91):
    """Write the store (see the module docstring) and return its path."""
    import zarr

    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    images = [f"i{i + 1:03d}" for i in range(n_images)]
    types = [f"t{j + 1:02d}" for j in range(n_types)]
    _template(path, images, types)
    t = zarr.open_group(path / "tables" / "table", mode="a")
    obs, obsm = t["obs"], t["obsm"]
    N = n_images * cells
    arr = {"encoding-type": "array", "encoding-version": "0.2.0"}
    code_t = np.int8 if n_images < 128 else (np.int16 if n_images < 32768 else np.int32)
    out = {
        "region": _replace(obs["region"], "codes", (N,), (chunk,), code_t, arr),
        "subject": _replace(obs["subject"], "codes", (N,), (chunk,), code_t, arr),
        "cell_type": _replace(obs["cell_type"], "codes", (N,), (chunk,), np.int8, arr),
        "condition": _replace(obs["condition"], "codes", (N,), (chunk,), np.int8, arr),
        "cell_id": _replace(obs, "cell_id", (N,), (chunk,), np.int64, arr),
        "_index": _replace(obs, "_index", (N,), (chunk,), np.int64, arr),
        "spatial": _replace(obsm, "spatial", (N, 2), (chunk, 2), np.float32, arr),
    }
    side = np.sqrt(cells / density)
    buf_xy = np.empty((chunk, 2), np.float32)
    buf_t = np.empty(chunk, np.int8)
    buf_i = np.empty(chunk, code_t)
    filled, row = 0, 0

    def flush(n):
        nonlocal row
        sl = slice(row, row + n)
        out["spatial"][sl, :] = buf_xy[:n]
        out["cell_type"][sl] = buf_t[:n]
        out["region"][sl] = buf_i[:n]
        out["subject"][sl] = buf_i[:n]
        out["condition"][sl] = (buf_i[:n] >= n_images / 2).astype(np.int8)
        ids = np.arange(row, row + n, dtype=np.int64)
        out["cell_id"][sl] = ids
        out["_index"][sl] = ids
        row += n

    t0 = time.time()
    for i in range(n_images):
        rng = np.random.default_rng([seed, i])
        x = rng.uniform(0, side, cells).astype(np.float32)
        y = rng.uniform(0, side, cells).astype(np.float32)
        ty = rng.integers(0, n_types, cells).astype(np.int8)
        done = 0
        while done < cells:
            n = min(chunk - filled, cells - done)
            buf_xy[filled : filled + n, 0] = x[done : done + n]
            buf_xy[filled : filled + n, 1] = y[done : done + n]
            buf_t[filled : filled + n] = ty[done : done + n]
            buf_i[filled : filled + n] = i
            filled += n
            done += n
            if filled == chunk:
                flush(chunk)
                filled = 0
        if (i + 1) % max(1, n_images // 20) == 0:
            print(f"{time.strftime('%H:%M:%S')} image {i + 1}/{n_images} ({time.time() - t0:.0f} s)", flush=True)
    if filled:
        flush(filled)
    assert row == N
    zarr.consolidate_metadata(path)  # the template's consolidated metadata has the old shapes
    meta = {"images": n_images, "cells_per_image": cells, "types": n_types, "chunk": chunk, "seed": seed}
    (path / "synthetic.json").write_text(json.dumps(meta))
    return path


def main():
    """Parse the command line and write the store."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--images", type=int, required=True)
    p.add_argument("--cells", type=int, required=True, help="cells per image")
    p.add_argument("--types", type=int, default=10)
    p.add_argument("--chunk", type=int, default=1 << 20, help="rows per chunk")
    p.add_argument("--seed", type=int, default=91)
    a = p.parse_args()
    write_store(a.out, a.images, a.cells, a.types, a.chunk, seed=a.seed)


if __name__ == "__main__":
    main()
