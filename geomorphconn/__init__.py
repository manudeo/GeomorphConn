"""
geomorphconn — Hydrologically-weighted Index of Connectivity.

Implements the Index of Connectivity (IC) of Cavalli et al. (2013) as a
Landlab component, extended with a composable weight pipeline supporting
rainfall, NDVI C-factor, RUSLE land-cover C-factor, DEM surface roughness,
and fully custom weights. The original reference implementation of the IC
is SedInConnect (Crema & Cavalli, 2018).

Quick start::

    from landlab import RasterModelGrid
    import numpy as np
    from geomorphconn import ConnectivityIndex

    grid = RasterModelGrid((30, 30), xy_spacing=30.0)
    z = grid.add_zeros("topographic__elevation", at="node")
    z += np.random.default_rng(0).random(grid.number_of_nodes) * 50

    ic = ConnectivityIndex(grid)   # default: NDVI=0, uniform rainfall
    ic.run_one_step()
    print(ic.as_2d().shape)        # (30, 30)

"""

from .components import ConnectivityIndex
from .api import run_connectivity_from_rasters
from .weights import (
    CustomWeight,
    LandCoverWeight,
    NDVIWeight,
    RainfallWeight,
    SurfaceRoughnessWeight,
    WeightBuilder,
    preset_landcover_only,
    preset_rainfall_landcover,
    preset_rainfall_ndvi,
    preset_rainfall_ndvi_roughness,
    preset_roughness_only,
)
from .utils import coarsen_rasters

__version__ = "0.1.0"
__all__ = [
    "ConnectivityIndex",
    "run_connectivity_from_rasters",
    "WeightBuilder",
    "RainfallWeight",
    "NDVIWeight",
    "SurfaceRoughnessWeight",
    "LandCoverWeight",
    "CustomWeight",
    "preset_rainfall_ndvi",
    "preset_roughness_only",
    "preset_landcover_only",
    "preset_rainfall_landcover",
    "preset_rainfall_ndvi_roughness",
    "coarsen_rasters",
]
