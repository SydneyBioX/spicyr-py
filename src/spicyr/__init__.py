"""spicyr: calibrated tests for changes in cell-type co-localisation between groups of patients.

The Python twin of the Bioconductor package spicyR: the same C++ core, the same arguments and the same
results. ``spicy()`` works with pandas DataFrames, AnnData and SpatialData objects.
"""

from ._api import spicy
from ._input import format_data
from ._results import SpicyResults

__version__ = "1.99.0"
__all__ = ["spicy", "format_data", "SpicyResults", "__version__"]
