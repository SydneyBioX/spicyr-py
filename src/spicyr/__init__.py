"""spicyr: test whether cell types co-localise differently between groups of patients.

``spicy()`` works with AnnData, SpatialData and pandas objects; ``write_cell_store()`` writes cells as a SpatialData
Zarr store that ``spicy()`` reads a block of rows at a time.
"""

from . import datasets
from ._api import spicy
from ._coltest import col_test, get_prop
from ._input import format_data
from ._plots import plot_image
from ._results import SpicyResults
from ._write import write_cell_store

__version__ = "1.99.19"
__all__ = [
    "spicy",
    "col_test",
    "get_prop",
    "format_data",
    "plot_image",
    "SpicyResults",
    "write_cell_store",
    "datasets",
    "__version__",
]
