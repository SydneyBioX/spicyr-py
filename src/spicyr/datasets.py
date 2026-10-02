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

# ExperimentHub files of the Bioconductor package SpatialDatasets: name -> (fetch id, size in bytes)
_EH = {
    "spe_Ali_2020": (9652, 130297491),  # EH9586
    "spe_Schurch_2020": (9651, 82302858),
}  # EH9585
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
    return arr


def _parse_spe(rds: str, out: str) -> None:
    """Write the colData and spatialCoords of a SpatialExperiment RDS file to `out` as a pickled DataFrame.

    Runs in a separate process (see _spatial_experiment): rds2py changes the importing process.
    """
    from rds2py import parse_rds

    d = parse_rds(rds)
    ld = d["attributes"]["colData"]["attributes"]["listData"]
    names = ld["attributes"]["names"]["data"]
    df = pd.DataFrame({n: _vector(c) for n, c in zip(names, ld["data"], strict=True)})
    ic = d["attributes"]["int_colData"]["attributes"]["listData"]
    sc = ic["data"][ic["attributes"]["names"]["data"].index("spatialCoords")]
    nrow, ncol = sc["attributes"]["dim"]["data"]
    xy = np.asarray(sc["data"]).reshape(ncol, nrow).T
    df = df.drop(columns=[c for c in ("x", "y") if c in df.columns])
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    df.to_pickle(out)


def _spatial_experiment(name: str, path=None) -> pd.DataFrame:
    """The cells of a SpatialDatasets object, downloaded from ExperimentHub and cached as a table."""
    import subprocess
    import sys

    fetch, size = _EH[name]
    table = _cache_dir() / f"{name}.pkl"
    if path is None and table.exists():
        return pd.read_pickle(table)
    if path is None:
        path = _cache_dir() / f"{name}.rds"
        if not path.exists() or path.stat().st_size != size:
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(f"https://experimenthub.bioconductor.org/fetch/{fetch}", tmp)
            tmp.replace(path)
    out = table if path == _cache_dir() / f"{name}.rds" else Path(str(path) + ".pkl")
    code = f"from spicyr.datasets import _parse_spe; _parse_spe({str(path)!r}, {str(out)!r})"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        if "No module named 'rds2py'" in r.stderr:
            raise ImportError('the example data sets need rds2py: pip install "spicyr[data]"')
        raise RuntimeError(f"could not read {path}:\n{r.stderr[-2000:]}")
    return pd.read_pickle(out)


def _as_anndata(df):
    import anndata as ad

    df = df.reset_index(drop=True)
    df.index = df.index.astype(str)
    return ad.AnnData(obs=df.drop(columns=["x", "y"]), obsm={"spatial": df[["x", "y"]].to_numpy(float)})


def metabric_ali2020(path: str | os.PathLike | None = None, as_anndata: bool = False):
    """Imaging mass cytometry of 483 breast tumours from the METABRIC cohort (Ali et al., Nature Cancer 2020).

    One row per cell (433,001 cells): the core (``file_id``), the patient (``metabricId``), the cell type
    assigned by the authors (``description``), the coordinates in micrometres (``x``, ``y``) and the patient's
    clinical data, including ER status (``ER.Status``) and relapse-free survival (``timeRFS``, ``eventRFS``).

    The file (130 MB) is downloaded once from Bioconductor's ExperimentHub, where the R package SpatialDatasets
    gets it (``spe_Ali_2020``), and cached in ``~/.cache/spicyr`` (or ``$SPICYR_CACHE``). Reading it needs the
    ``rds2py`` package (``pip install "spicyr[data]"``).

    With ``as_anndata=True`` the cells are returned as an AnnData object: the table as ``obs`` and the
    coordinates in ``obsm["spatial"]`` (needs ``anndata``).
    """
    df = _spatial_experiment("spe_Ali_2020", path)
    return _as_anndata(df) if as_anndata else df


def schurch2020(path: str | os.PathLike | None = None, as_anndata: bool = False):
    """CODEX imaging of colorectal cancer (Schürch et al., Cell 2020): 35 patients, four tissue cores each.

    One row per cell (258,385 cells) with the core (``imageID``), the patient (``patient``), the cell type
    (``cellType``), the coordinates (``x``, ``y``) and the patient group (``group``: 1 = Crohn's-like reaction,
    2 = diffuse inflammatory infiltration). Downloaded from ExperimentHub as ``spe_Schurch_2020`` of the R package
    SpatialDatasets (82 MB), and cached as for :func:`metabric_ali2020`. ``as_anndata`` as for
    :func:`metabric_ali2020`.
    """
    df = _spatial_experiment("spe_Schurch_2020", path)
    return _as_anndata(df) if as_anndata else df
