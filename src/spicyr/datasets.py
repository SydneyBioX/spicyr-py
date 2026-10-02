"""Example data.

``metabric_ali2020()`` downloads the same file that the Bioconductor package SpatialDatasets serves to R
(``spe_Ali_2020``, ExperimentHub EH9586), so the R and Python tutorials analyse identical data.
"""

from __future__ import annotations

import os
import urllib.request
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

_EH_URL = "https://experimenthub.bioconductor.org/fetch/9652"   # SpatialDatasets: spe_Ali_2020 (EH9586)
_EH_SIZE = 130297491
_NA_INT = -2147483648


def example_cells() -> pd.DataFrame:
    """A small simulated data set: 14 patients (7 per condition, A and B), two images each, six cell types.

    In condition B, about a third of the T cells were moved next to tumour cells. Both directions change;
    ``T -> tumour`` (tumour cells counted around each T cell, the recruited type at the centre) is the direction
    whose excess does not depend on how common tumour cells are.
    """
    return pd.read_csv(files("spicyr") / "data" / "example_cells.csv.gz")


def _cache_dir() -> Path:
    d = Path(os.environ.get("SPICYR_CACHE", Path.home() / ".cache" / "spicyr"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _vector(x):
    """An R vector or factor from rds2py's parse tree, as a numpy array or pandas Categorical."""
    data, attrs = x.get("data"), x.get("attributes", {}) or {}
    if "levels" in attrs:
        codes = np.asarray(data)
        return pd.Categorical.from_codes(np.where(codes < 1, -1, codes - 1), attrs["levels"]["data"])
    arr = np.asarray(data)
    if arr.dtype.kind == "i" and (arr == _NA_INT).any():
        arr = np.where(arr == _NA_INT, np.nan, arr)
    if arr.dtype == object:
        arr = np.array([None if v is None else v for v in arr], dtype=object)
    return arr


def metabric_ali2020(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Imaging mass cytometry of 483 breast tumours from the METABRIC cohort (Ali et al., Nature Cancer 2020).

    One row per cell (433,001 cells): the core (``file_id``), the patient (``metabricId``), the cell type
    assigned by the authors (``description``), the coordinates in micrometres (``Location_Center_X``,
    ``Location_Center_Y``) and the patient's clinical data, including ER status (``ER.Status``) and relapse-free
    survival (``timeRFS``, ``eventRFS``).

    The file (130 MB) is downloaded once from Bioconductor's ExperimentHub, where the R package SpatialDatasets
    gets it, and cached in ``~/.cache/spicyr`` (or ``$SPICYR_CACHE``). Reading it needs the ``rds2py`` package
    (``pip install "spicyr[data]"``).
    """
    try:
        from rds2py import parse_rds
    except ImportError as e:  # pragma: no cover
        raise ImportError('metabric_ali2020() needs rds2py: pip install "spicyr[data]"') from e
    if path is None:
        path = _cache_dir() / "spe_Ali_2020.rds"
        if not path.exists() or path.stat().st_size != _EH_SIZE:
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(_EH_URL, tmp)
            tmp.replace(path)
    d = parse_rds(str(path))
    ld = d["attributes"]["colData"]["attributes"]["listData"]
    names = ld["attributes"]["names"]["data"]
    df = pd.DataFrame({n: _vector(c) for n, c in zip(names, ld["data"])})
    ic = d["attributes"]["int_colData"]["attributes"]["listData"]
    sc = ic["data"][ic["attributes"]["names"]["data"].index("spatialCoords")]
    nrow, ncol = sc["attributes"]["dim"]["data"]
    xy = np.asarray(sc["data"]).reshape(ncol, nrow).T
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    return df
