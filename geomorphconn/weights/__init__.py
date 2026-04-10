"""
GeomorphConn.weights
===============
Composable weight-factor pipeline for the IC.

Import the main classes directly::

    from geomorphconn.weights import (
        WeightBuilder,
        RainfallWeight,
        NDVIWeight,
        SurfaceRoughnessWeight,
        LandCoverWeight,
        CustomWeight,
    )

Preset factory functions are also available for common configurations::

    from geomorphconn.weights import (
        preset_rainfall_ndvi,           # Dubey, Singh & Jain (submitted)
        preset_roughness_only,          # TRI from DEM only
        preset_landcover_only,          # RUSLE C-factor from land-cover map
        preset_rainfall_landcover,      # rainfall + land-cover C-factor
        preset_rainfall_ndvi_roughness, # all three combined
    )
"""

from .builder import WeightBuilder
from .components import (
    CustomWeight,
    LandCoverWeight,
    NDVIWeight,
    RainfallWeight,
    SurfaceRoughnessWeight,
)

__all__ = [
    "WeightBuilder",
    # components
    "RainfallWeight",
    "NDVIWeight",
    "SurfaceRoughnessWeight",
    "LandCoverWeight",
    "CustomWeight",
    # presets
    "preset_rainfall_ndvi",
    "preset_roughness_only",
    "preset_landcover_only",
    "preset_rainfall_landcover",
    "preset_rainfall_ndvi_roughness",
]


# ---------------------------------------------------------------------------
# Preset factory functions
# ---------------------------------------------------------------------------


def preset_rainfall_ndvi(
    rainfall,
    ndvi,
    combine: str = "mean",
    w_min: float = 0.005,
) -> WeightBuilder:
    """
    Default weight from Dubey, Singh & Jain (submitted): ``W = (RF_norm + C_NDVI) / 2``.

    Parameters
    ----------
    rainfall : array_like
        Rainfall at each node (any units; only the spatial range matters).
    ndvi : array_like
        NDVI at each node, values in ``[−1, 1]``.
    combine : str, optional
        Combination mode passed to :class:`WeightBuilder`.  Default ``'mean'``.
    w_min : float, optional
        Lower clamp.  Default ``0.005``.
    """
    return (
        WeightBuilder(combine=combine, w_min=w_min)
        .add(RainfallWeight(rainfall, w_min=w_min))
        .add(NDVIWeight(ndvi, w_min=w_min))
    )


def preset_roughness_only(
    grid,
    invert: bool = False,
    w_min: float = 0.005,
) -> WeightBuilder:
    """
    DEM-only weight using the Terrain Ruggedness Index (TRI).

    Useful when no satellite-derived data is available.

    Parameters
    ----------
    grid : RasterModelGrid
        Landlab grid with ``'topographic__elevation'``.
    invert : bool, optional
        If *True*, rough cells → low W (hydraulic-impedance interpretation).
        Default *False* (rough cells → high W; source-area interpretation).
    w_min : float, optional
        Lower clamp.  Default ``0.005``.
    """
    return WeightBuilder(w_min=w_min).add(
        SurfaceRoughnessWeight(grid, w_min=w_min, invert=invert)
    )


def preset_landcover_only(
    landcover,
    c_factor_table=None,
    w_min: float = 0.005,
) -> WeightBuilder:
    """
    RUSLE C-factor weight from a land-cover classification map.

    Follows the spirit of Borselli et al. (2008), who used land-use C-factors
    as the sole weight in the original IC formulation.

    Parameters
    ----------
    landcover : array_like of int
        Integer land-cover class codes.
    c_factor_table : dict[int, float], optional
        Custom ``{code: c_factor}`` mapping.  Defaults to ESA WorldCover.
    w_min : float, optional
        Lower clamp.  Default ``0.005``.
    """
    return WeightBuilder(w_min=w_min).add(
        LandCoverWeight(landcover, c_factor_table=c_factor_table, w_min=w_min)
    )


def preset_rainfall_landcover(
    rainfall,
    landcover,
    c_factor_table=None,
    combine: str = "mean",
    w_min: float = 0.005,
) -> WeightBuilder:
    """
    Rainfall + RUSLE C-factor from land cover.

    Extends the Borselli (2008) approach by adding a spatially variable
    rainfall term.

    Parameters
    ----------
    rainfall : array_like
        Rainfall at each node.
    landcover : array_like of int
        Integer land-cover class codes.
    c_factor_table : dict[int, float], optional
        Custom lookup table.  Defaults to ESA WorldCover.
    combine : str, optional
        Combination mode.  Default ``'mean'``.
    w_min : float, optional
        Lower clamp.  Default ``0.005``.
    """
    return (
        WeightBuilder(combine=combine, w_min=w_min)
        .add(RainfallWeight(rainfall, w_min=w_min))
        .add(LandCoverWeight(landcover, c_factor_table=c_factor_table, w_min=w_min))
    )


def preset_rainfall_ndvi_roughness(
    rainfall,
    ndvi,
    grid,
    combine: str = "mean",
    roughness_invert: bool = False,
    w_min: float = 0.005,
) -> WeightBuilder:
    """
    Full three-component weight: rainfall + NDVI C-factor + DEM roughness.

    Parameters
    ----------
    rainfall : array_like
        Rainfall at each node.
    ndvi : array_like
        NDVI at each node.
    grid : RasterModelGrid
        Landlab grid with ``'topographic__elevation'``.
    combine : str, optional
        Combination mode.  Default ``'mean'``.
    roughness_invert : bool, optional
        Sign convention for TRI.  Default *False*.
    w_min : float, optional
        Lower clamp.  Default ``0.005``.
    """
    return (
        WeightBuilder(combine=combine, w_min=w_min)
        .add(RainfallWeight(rainfall, w_min=w_min))
        .add(NDVIWeight(ndvi, w_min=w_min))
        .add(SurfaceRoughnessWeight(grid, w_min=w_min, invert=roughness_invert))
    )
