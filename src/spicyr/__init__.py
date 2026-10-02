"""spicyr: test whether cell types co-localise differently between groups of patients.

``spicy()`` works with AnnData, SpatialData and pandas objects.
"""

from ._api import spicy
from ._input import format_data
from ._results import SpicyResults
from ._plots import plot_image
from . import datasets

__version__ = "1.99.0"
__all__ = ["spicy", "format_data", "plot_image", "SpicyResults", "datasets", "__version__"]
