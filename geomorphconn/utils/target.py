"""
target.py
=========
Utilities for rasterizing vector target features onto a Landlab grid.

``rasterize_targets`` converts any geopandas-readable polyline or polygon
(river network, lake boundary, channel centreline, etc.) to a 1-D array of
Landlab node IDs for use with
``ConnectivityIndex(target_nodes=target_nodes)``.

This replicates the ArcGIS IC-target workflow:
  1. Rasterize the shapefile at the DEM cell size.
  2. Identify all DEM cells that overlap the geometry.
  3. Return those cells as Landlab node IDs (bottom-row-first).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

GeoDataFrameLike = Any


def rasterize_targets(
    source: Union[str, Path, GeoDataFrameLike],
    grid,
    dem_transform=None,
    dem_crs=None,
    all_touched: bool = True,
    buffer_m: float = 0.0,
) -> np.ndarray:
    """
    Rasterize a vector target onto a Landlab grid and return node IDs.

    Parameters
    ----------
    source : str, Path, or GeoDataFrame
        Path to any geopandas-readable file, or an already-loaded
        ``GeoDataFrame``.
    grid : RasterModelGrid
        The Landlab grid on which IC will be computed.
    dem_transform : Affine, optional
        Rasterio ``Affine`` transform for spatial registration.  Pass the
        ``profile['transform']`` from rasterio or :class:`~GeomorphConn.gee.GEEFetcher`.
        If *None*, a unit-origin transform is assumed (suitable for grids
        without a projected CRS).
    dem_crs : str or CRS, optional
        CRS of the grid used to reproject the vector.  Defaults to
        ``'EPSG:4326'``.
    all_touched : bool, optional
        Include any cell whose bounding box is touched by the geometry
        (ArcGIS *PolygonToRaster* equivalent).  Default *True*.
    buffer_m : float, optional
        Buffer (metres) applied to the geometry before rasterization.
        Useful to ensure narrow lines cover at least one cell.
        If ``0.0`` and line geometries are detected, a small automatic
        buffer of half a grid cell is applied to line targets.
        Default ``0.0``.

    Returns
    -------
    ndarray of int, shape (k,)
        Node IDs of cells overlapping the target.

    Raises
    ------
    ImportError
        If ``geopandas`` or ``rasterio`` are not installed.
    ValueError
        If *source* has no CRS, or if no target cells are found.
    """
    try:
        import geopandas as gpd  # pyright: ignore[reportMissingModuleSource]
        from rasterio.features import rasterize as rio_rasterize
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise ImportError(
            "geopandas and rasterio are required for rasterize_targets. "
            "Install with: pip install \"geomorphconn[target]\" "
            "or reinstall the GUI extras with: pip install \"geomorphconn[gui]\""
        ) from exc

    nrows = grid.number_of_node_rows
    ncols = grid.number_of_node_columns
    dx = float(grid.dx)

    # Load vector
    if isinstance(source, (str, Path)):
        gdf = gpd.read_file(source)
    else:
        gdf = source.copy()

    # Reproject to DEM CRS
    target_crs = dem_crs if dem_crs is not None else "EPSG:4326"
    if gdf.crs is None:
        raise ValueError("Target GeoDataFrame has no CRS set.")
    gdf = gdf.to_crs(target_crs)

    # Optional/automatic buffer.
    geom_types = set(gdf.geometry.geom_type.dropna().tolist())
    line_like = {"LineString", "MultiLineString"}
    if buffer_m > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(buffer_m)
    elif geom_types & line_like:
        auto_buffer = dx * 0.5
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(
            lambda geom: geom.buffer(auto_buffer)
            if geom is not None and geom.geom_type in line_like
            else geom
        )

    # Build transform if not provided
    if dem_transform is None:
        dem_transform = from_origin(0.0, nrows * dx, dx, dx)

    # Rasterize
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        raise ValueError("No valid geometries found in the target source.")

    raster = np.asarray(
        rio_rasterize(
            shapes,
            out_shape=(nrows, ncols),
            transform=dem_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=all_touched,
        )
    )

    # GeoTIFF row 0 = north; Landlab row 0 = south → flipud
    target_nodes = np.where(np.flipud(raster).ravel() == 1)[0].astype(np.int64)

    if len(target_nodes) == 0:
        raise RuntimeError(
            "No target nodes found after rasterization. "
            "Check that the vector overlaps the DEM extent and that the "
            "CRS is correct."
        )

    return target_nodes


def nodes_from_geodataframe(
    gdf,
    grid,
    dem_transform=None,
    dem_crs=None,
    all_touched: bool = True,
    buffer_m: float = 0.0,
) -> np.ndarray:
    """
    Convenience wrapper around :func:`rasterize_targets` for an already-loaded
    ``GeoDataFrame``.  Accepts the same keyword arguments.
    """
    return rasterize_targets(
        gdf,
        grid,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        all_touched=all_touched,
        buffer_m=buffer_m,
    )
