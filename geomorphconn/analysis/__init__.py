"""
Spatio-temporal analysis module for sediment connectivity.

This subpackage provides tools for analyzing dynamic patterns in
connectivity time series, including Connectivity Response Unit (CRU)
classification for identifying emerging, persistent, and changing
connectivity zones over time.

References:
    Singh et al. (2017). Assessment of connectivity in a water-stressed wetland.
        Earth Surf. Process. Landf. 42(11): 1982-1996.
    Singh et al. (2018). Evaluating dynamic hydrological connectivity of a
        floodplain wetland. Sci. Total Environ. 651: 2473-2488.
"""

from .cru_dynamics import classify_dynamic_crus
from .disconnectivity import (
    build_disconnectivity_hierarchy,
    compute_node_comparison_metrics,
)
from .utils import export_cru_geotiff, generate_arcgis_legend_dict, generate_qgis_legend_dict

__all__ = [
    "classify_dynamic_crus",
    "build_disconnectivity_hierarchy",
    "compute_node_comparison_metrics",
    "export_cru_geotiff",
    "generate_arcgis_legend_dict",
    "generate_qgis_legend_dict",
]
