"""Example data."""

from importlib.resources import files

import pandas as pd


def example_cells() -> pd.DataFrame:
    """A small simulated data set: 14 patients (7 per condition, A and B), two images each, six cell types.

    In condition B, about a third of the T cells were moved next to tumour cells. Both directions change;
    ``T -> tumour`` (tumour cells counted around each T cell, the recruited type at the centre) is the direction
    whose excess does not depend on how common tumour cells are. Columns: x, y, cellType, imageID, patient, condition, and per
    patient age, stage (I/II/III), time and event; batch varies by image.
    """
    return pd.read_csv(files("spicyr") / "data" / "example_cells.csv.gz")
