"""High-level raster/xarray API wrappers for ConnectivityIndex.

This module provides a convenience entrypoint that mirrors CLI/GUI behavior:
- accepts DEM/NDVI/rainfall/weight as GeoTIFF paths or xarray DataArrays
- verifies DEM CRS and, if geographic, reprojects DEM to estimated local UTM
- aligns all optional rasters to DEM using rioxarray.reproject_match
- builds RasterModelGrid and runs ConnectivityIndex
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from landlab import RasterModelGrid
from rasterio.enums import Resampling

from .components import ConnectivityIndex
from .weights import NDVIWeight, RainfallWeight, WeightBuilder


def _require_rioxarray():
    try:
        import rioxarray as rxr
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "run_connectivity_from_rasters requires rioxarray. Install with: pip install rioxarray"
        ) from exc
    return rxr


def _require_xarray():
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "run_connectivity_from_rasters requires xarray. Install with: pip install xarray"
        ) from exc
    return xr


def _as_dataarray(src: Any, name: str):
    """Convert a path/DataArray input to an xarray.DataArray."""
    rxr = _require_rioxarray()

    if hasattr(src, "rio") and hasattr(src, "values"):
        return src.squeeze(drop=True), None

    if isinstance(src, (str, Path)):
        obj: Any = rxr.open_rasterio(src, masked=True)
        da = obj.squeeze(drop=True)
        return da, da

    raise TypeError(
        f"{name} must be an xarray.DataArray or raster path (str/Path). Got {type(src).__name__}."
    )


def _to_nan_array(da) -> np.ndarray:
    arr = np.asarray(da.values, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, np.nan)


def _array_to_geoda(template_da, array2d: np.ndarray, name: str):
    """Return a georeferenced DataArray using *template_da* spatial metadata."""
    out = template_da.copy(data=np.asarray(array2d, dtype=np.float64))
    out.name = name
    return out


def _is_projected(crs_obj: Any) -> bool:
    if crs_obj is None:
        return False
    try:
        return bool(crs_obj.is_projected)
    except Exception:
        return False


def _normalize_weight_spec(weight: Any, ndvi: Any, rainfall: Any) -> tuple[Any, Any, Any]:
    """Resolve user-facing weight shortcut forms.

    Supported forms:
    - weight=<precomputed weight raster/dataarray>
    - weight=<ndvi raster/dataarray> (single item interpreted as NDVI)
    - weight=[ndvi, rainfall]
    - weight={"ndvi": ..., "rainfall": ...}
    """
    if weight is None:
        return None, ndvi, rainfall

    if isinstance(weight, dict):
        ndvi_val = weight.get("ndvi", ndvi)
        rainfall_val = weight.get("rainfall", rainfall)
        precomputed = weight.get("weight", None)
        return precomputed, ndvi_val, rainfall_val

    if isinstance(weight, (list, tuple)):
        if len(weight) == 0:
            raise ValueError("weight list/tuple cannot be empty")
        if len(weight) == 1:
            return None, weight[0], rainfall
        if len(weight) == 2:
            return None, weight[0], weight[1]
        raise ValueError("weight list/tuple supports at most two items: [ndvi] or [ndvi, rainfall]")

    # Scalar path/DataArray: treat as precomputed W unless ndvi/rainfall also supplied.
    if ndvi is None and rainfall is None:
        return weight, ndvi, rainfall

    # If user passed ndvi/rainfall separately, keep weight as precomputed W.
    return weight, ndvi, rainfall


def run_connectivity_from_rasters(
    dem: Any,
    *,
    weight: Any = None,
    ndvi: Any = None,
    rainfall: Any = None,
    flow_director: str = "DINF",
    fill_sinks: bool = False,
    depression_finder: str | None = "DepressionFinderAndRouter",
    use_aspect_weighting: bool = False,
    w_min: float = 0.005,
    w_max: float = 1.0,
    weight_combine: str = "mean",
    auto_project_to_utm: bool = True,
    ndvi_resampling: Resampling = Resampling.bilinear,
    rainfall_resampling: Resampling = Resampling.bilinear,
    weight_resampling: Resampling = Resampling.bilinear,
) -> dict[str, Any]:
    """Run ConnectivityIndex from raster/xarray inputs.

    Parameters
    ----------
    dem : DataArray or path
        DEM raster (required). If geographic and ``auto_project_to_utm=True``,
        it is reprojected to ``dem.rio.estimate_utm_crs()`` first.
    weight : optional
        Precomputed weight raster OR shorthand inputs:
        - ``weight=ndvi_input``
        - ``weight=[ndvi_input, rainfall_input]``
        - ``weight={"ndvi": ..., "rainfall": ...}``
    ndvi, rainfall : optional
        Optional NDVI/rainfall inputs when not using ``weight`` shorthand.

    Returns
    -------
    dict
        ``{"dataset", "inputs", "grid", "component", "profile"}``
        where ``dataset`` is a georeferenced xarray Dataset containing 2-D
        output layers (IC, Dup, Ddn, W, S, Wmean, Smean, ACCfinal), and
        ``inputs`` is a georeferenced xarray Dataset of aligned input rasters.
    """
    opened = []

    try:
        xr = _require_xarray()
        precomputed_weight, ndvi_in, rainfall_in = _normalize_weight_spec(weight, ndvi, rainfall)

        dem_da, dem_opened = _as_dataarray(dem, "dem")
        if dem_opened is not None:
            opened.append(dem_opened)

        if dem_da.rio.crs is None:
            raise ValueError("DEM must have a valid CRS in the raster metadata.")

        if auto_project_to_utm and not _is_projected(dem_da.rio.crs):
            utm_crs = dem_da.rio.estimate_utm_crs()
            if utm_crs is None:
                raise ValueError("Could not estimate UTM CRS from DEM extent.")
            dem_da = dem_da.rio.reproject(utm_crs)

        dem_arr = _to_nan_array(dem_da)
        ndvi_da = None
        rainfall_da = None
        weight_da = None

        ndvi_arr = None
        if ndvi_in is not None:
            ndvi_da, ndvi_opened = _as_dataarray(ndvi_in, "ndvi")
            if ndvi_opened is not None:
                opened.append(ndvi_opened)
            ndvi_da = ndvi_da.rio.reproject_match(dem_da, resampling=ndvi_resampling)
            ndvi_arr = _to_nan_array(ndvi_da)

        rainfall_arr = None
        if rainfall_in is not None:
            rainfall_da, rainfall_opened = _as_dataarray(rainfall_in, "rainfall")
            if rainfall_opened is not None:
                opened.append(rainfall_opened)
            rainfall_da = rainfall_da.rio.reproject_match(dem_da, resampling=rainfall_resampling)
            rainfall_arr = _to_nan_array(rainfall_da)

        weight_arr = None
        if precomputed_weight is not None:
            weight_da, weight_opened = _as_dataarray(precomputed_weight, "weight")
            if weight_opened is not None:
                opened.append(weight_opened)
            weight_da = weight_da.rio.reproject_match(dem_da, resampling=weight_resampling)
            weight_arr = _to_nan_array(weight_da)

        res = dem_da.rio.resolution()
        dx = float(abs(res[0]))
        dy = float(abs(res[1]))
        if not np.isclose(dx, dy):
            raise ValueError("Non-square pixels are not supported.")

        grid = RasterModelGrid(dem_arr.shape, xy_spacing=dx)
        grid.add_field("topographic__elevation", np.flipud(dem_arr).ravel(), at="node")

        if weight_arr is not None:
            weight_input = np.flipud(weight_arr).ravel()
        else:
            wb = WeightBuilder(combine=weight_combine, w_min=w_min, w_max=w_max)
            if rainfall_arr is not None:
                wb.add(RainfallWeight(np.flipud(rainfall_arr).ravel(), w_min=w_min))
            if ndvi_arr is not None:
                wb.add(NDVIWeight(np.flipud(ndvi_arr).ravel(), w_min=w_min))
            if rainfall_arr is None and ndvi_arr is None:
                raise ValueError(
                    "No weight input was provided. Supply precomputed weight, ndvi, rainfall, "
                    "or weight=[ndvi] / weight=[ndvi, rainfall]."
                )
            weight_input = wb

        ic = ConnectivityIndex(
            grid,
            flow_director=flow_director,
            weight=weight_input,
            fill_sinks=fill_sinks,
            depression_finder=depression_finder,
            use_aspect_weighting=use_aspect_weighting,
            w_min=w_min,
            w_max=w_max,
        )
        ic.run_one_step()

        fields = {
            "IC": "connectivity_index__IC",
            "Dup": "connectivity_index__Dup",
            "Ddn": "connectivity_index__Ddn",
            "W": "connectivity_index__W",
            "S": "connectivity_index__S",
            "Wmean": "connectivity_index__Wmean",
            "Smean": "connectivity_index__Smean",
            "ACCfinal": "connectivity_index__ACCfinal",
        }
        layers = {
            key: np.flipud(grid.at_node[field].reshape(dem_arr.shape))
            for key, field in fields.items()
        }
        layers_xr = {
            key: _array_to_geoda(dem_da, arr, key)
            for key, arr in layers.items()
        }
        dataset = xr.Dataset(layers_xr)
        dataset.attrs.update(
            {
                "flow_director": flow_director,
                "fill_sinks": bool(fill_sinks),
                "depression_finder": depression_finder if depression_finder is not None else "none",
                "use_aspect_weighting": bool(use_aspect_weighting),
                "w_min": float(w_min),
                "w_max": float(w_max),
                "weight_combine": weight_combine,
            }
        )

        input_vars = {"dem": dem_da}
        if ndvi_da is not None:
            input_vars["ndvi"] = ndvi_da
        if rainfall_da is not None:
            input_vars["rainfall"] = rainfall_da
        if weight_da is not None:
            input_vars["weight"] = weight_da
        inputs = xr.Dataset(input_vars)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": int(dem_da.rio.width),
            "height": int(dem_da.rio.height),
            "count": 1,
            "crs": dem_da.rio.crs,
            "transform": dem_da.rio.transform(),
            "nodata": -9999.0,
        }

        return {
            "dataset": dataset,
            "inputs": inputs,
            "grid": grid,
            "component": ic,
            "profile": profile,
        }
    finally:
        for da in opened:
            try:
                da.close()
            except Exception:
                pass
