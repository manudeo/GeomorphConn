"""
components.py
=============
Individual weight-factor components for the GeomorphConn IC weight pipeline.

Each class exposes a single ``.compute() -> np.ndarray`` method that returns
a float64 array of shape ``(n_nodes,)`` with values in ``[0, 1]`` after
clamping.  Components are combined by :class:`~GeomorphConn.weights.WeightBuilder`.

Available components
--------------------
``RainfallWeight``
    Normalises a rainfall raster to ``[0, 1]``.

``NDVIWeight``
    NDVI-derived RUSLE C-factor proxy: ``C = (1 − NDVI) / 2``.

``SurfaceRoughnessWeight``
    Cavalli-style roughness from DEM residuals and local standard deviation.

``LandCoverWeight``
    Maps integer land-cover codes to RUSLE C-factor values via a lookup table.
    Built-in tables for ESA WorldCover, CORINE, and MODIS IGBP are provided in
    :mod:`~GeomorphConn.weights.tables`.

``CustomWeight``
    Pass-through: accepts any pre-computed array; only clamping is applied.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clamp(arr: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """Return a float64 copy of *arr* clamped to ``[w_min, w_max]``."""
    return np.clip(arr.astype(np.float64), w_min, w_max)


def _minmax_norm(arr: np.ndarray, fallback: float = 0.5) -> np.ndarray:
    """Min-max normalise to ``[0, 1]``; return *fallback* if range is zero."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.full_like(arr, fallback, dtype=np.float64)


def _validate_odd_window(value: int, name: str) -> int:
    """Validate that a moving-window size is a positive odd integer."""
    v = int(value)
    if v < 1 or (v % 2) == 0:
        raise ValueError(f"{name} must be a positive odd integer (got {value}).")
    return v


def _box_mean_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast centered moving-window mean using an integral image."""
    pad = window // 2
    arr_pad = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
    ii = np.cumsum(np.cumsum(arr_pad, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
    s = ii[window:, window:] - ii[:-window, window:] - ii[window:, :-window] + ii[:-window, :-window]
    return s / float(window * window)


def compute_surface_roughness_weight_2d(
    elev_2d: np.ndarray,
    detrend_window: int = 3,
    std_window: int = 3,
    w_min: float = 0.005,
) -> np.ndarray:
    """Compute Cavalli roughness-derived weight from a 2-D DEM.

    Uses xarray rolling windows when available, with a NumPy fallback.
    Returns a 2-D weight array in geographic row/column order.
    """
    detrend_window = _validate_odd_window(detrend_window, "detrend_window")
    std_window = _validate_odd_window(std_window, "std_window")

    elev = np.asarray(elev_2d, dtype=np.float64)
    if elev.ndim != 2:
        raise ValueError("elev_2d must be a 2-D array")

    try:
        import xarray as xr

        da = xr.DataArray(elev, dims=("y", "x"))
        z_avg = da.rolling(y=detrend_window, x=detrend_window, center=True, min_periods=1).mean()
        residual = da - z_avg
        ri = residual.rolling(y=std_window, x=std_window, center=True, min_periods=1).std(ddof=0)
        ri_arr = np.asarray(ri.values, dtype=np.float64)
    except Exception:
        z_avg = _box_mean_2d(elev, detrend_window)
        residual = elev - z_avg
        r_mean = _box_mean_2d(residual, std_window)
        r2_mean = _box_mean_2d(residual * residual, std_window)
        variance = np.maximum(0.0, r2_mean - (r_mean * r_mean))
        ri_arr = np.sqrt(variance)

    valid = ri_arr[np.isfinite(ri_arr)]
    ri_max = float(np.max(valid)) if valid.size > 0 else 0.0
    if ri_max > 0.0:
        w = 1.0 - (ri_arr / ri_max)
    else:
        w = np.ones_like(ri_arr, dtype=np.float64)

    floor = max(float(w_min), 0.001)
    return _clamp(w, floor, 1.0)


# ---------------------------------------------------------------------------
# RainfallWeight
# ---------------------------------------------------------------------------

class RainfallWeight:
    """
    Normalised rainfall weight.

    Applies min-max normalisation so that the driest cell → 0 and the
    wettest cell → 1.  Only the spatial *gradient* of rainfall matters; the
    absolute units are irrelevant.

    Parameters
    ----------
    rainfall : array_like, shape (n_nodes,)
        Rainfall values at each node (mm, mm/yr, or any consistent unit).
    w_min : float, optional
        Lower clamp applied after normalisation.  Default ``0.005``.

    Examples
    --------
    >>> import numpy as np
    >>> from GeomorphConn.weights import RainfallWeight
    >>> w = RainfallWeight(np.array([500.0, 1000.0, 1500.0]), w_min=0.0).compute()
    >>> w.tolist()
    [0.0, 0.5, 1.0]
    """

    name = "rainfall"

    def __init__(self, rainfall: np.ndarray, w_min: float = 0.005) -> None:
        self._rf = np.asarray(rainfall, dtype=np.float64).ravel()
        self._w_min = float(w_min)

    def compute(self) -> np.ndarray:
        """Return normalised rainfall weight, shape ``(n_nodes,)``."""
        normed = _minmax_norm(self._rf)
        return _clamp(normed, self._w_min, 1.0)


# ---------------------------------------------------------------------------
# NDVIWeight
# ---------------------------------------------------------------------------

class NDVIWeight:
    """
    NDVI-derived RUSLE C-factor proxy.

    Converts NDVI to an approximate C-factor following the relationship used
    in Cavalli et al. (2013) and Borselli et al. (2008):

    .. math::

        C = \\frac{1 - \\text{NDVI}}{2}

    This is a first-order approximation: dense vegetation (NDVI → 1) gives
    C → 0 (well-protected surface); bare/degraded soil (NDVI → 0) gives
    C → 0.5; non-vegetated water or snow (NDVI → −1) gives C → 1.  The
    formula is not a formal RUSLE regression but is widely applied in
    geomorphic connectivity studies because NDVI is globally available at
    high resolution from Landsat and Sentinel-2.

    For a physically more rigorous C-factor derived from a classified land
    cover map and the RUSLE lookup tables, use :class:`LandCoverWeight`.

    Parameters
    ----------
    ndvi : array_like, shape (n_nodes,)
        NDVI values in ``[−1, 1]``.
    w_min : float, optional
        Lower clamp.  Default ``0.005``.

    References
    ----------
    Cavalli, M. et al. (2013). Geomorphology, 188, 31–41.
    Borselli, L. et al. (2008). Catena, 75(3), 268–277.
    Crema, S. & Cavalli, M. (2018). Computers & Geosciences, 111, 39–45.
    """

    name = "ndvi"

    def __init__(self, ndvi: np.ndarray, w_min: float = 0.005) -> None:
        self._ndvi = np.asarray(ndvi, dtype=np.float64).ravel()
        self._w_min = float(w_min)

    def compute(self) -> np.ndarray:
        """Return NDVI-based C-factor, shape ``(n_nodes,)``."""
        c = (1.0 - self._ndvi) / 2.0
        return _clamp(c, self._w_min, 1.0)


# ---------------------------------------------------------------------------
# SurfaceRoughnessWeight
# ---------------------------------------------------------------------------

class SurfaceRoughnessWeight:
    """
    Surface roughness weight derived from DEM using the Cavalli and Marchi (2008)
    residual-roughness workflow.

    .. math::

        z_{avg} = \\text{mean}(z, w_1), \\quad r = z - z_{avg}

    .. math::

        RI = \\text{std}(r, w_2)

    If used as IC weight, RI is normalised as:

    .. math::

        W = 1 - RI / RI_{max}

    Then :math:`W` is clamped to ``[max(w_min, 0.001), 1]`` to avoid zeros in
    the IC denominator.

    **Physical interpretation**

    High roughness represents high impedance to sediment transfer. Therefore:
    rough cells receive *low* W.

    Parameters
    ----------
    grid : RasterModelGrid
        Landlab grid with ``'topographic__elevation'`` at nodes.
    detrend_window : int, optional
        Moving-window size for local-mean detrending (operation 1).
        Must be a positive odd integer. Default ``3``.
    std_window : int, optional
        Moving-window size for local standard deviation of the residual
        (operation 2). Must be a positive odd integer. Default ``3``.
    w_min : float, optional
        Lower clamp.  Default ``0.005``.

    References
    ----------
    Cavalli, M. and Marchi, L. (2008). Characterisation of the surface
        morphology of an alpine alluvial fan using airborne LiDAR.
        Natural Hazards and Earth System Sciences, 8, 323-333.
        https://doi.org/10.5194/nhess-8-323-2008
    """

    name = "roughness"

    def __init__(
        self,
        grid,
        detrend_window: int = 3,
        std_window: int = 3,
        w_min: float = 0.005,
    ) -> None:
        self._grid = grid
        self._detrend_window = _validate_odd_window(detrend_window, "detrend_window")
        self._std_window = _validate_odd_window(std_window, "std_window")
        self._w_min = float(w_min)

    def compute(self) -> np.ndarray:
        """Return Cavalli roughness-based weight, shape ``(n_nodes,)``."""
        nrows = self._grid.number_of_node_rows
        ncols = self._grid.number_of_node_columns
        elev = self._grid.at_node["topographic__elevation"].reshape(nrows, ncols)
        w = compute_surface_roughness_weight_2d(
            elev,
            detrend_window=self._detrend_window,
            std_window=self._std_window,
            w_min=self._w_min,
        )
        return np.asarray(w, dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# LandCoverWeight
# ---------------------------------------------------------------------------

class LandCoverWeight:
    """
    RUSLE C-factor weight derived from a land-cover classification map.

    Maps integer class codes to C-factor values using a lookup table sourced
    from RUSLE literature (Renard et al. 1997; Wischmeier & Smith 1978).
    Built-in tables are provided for ESA WorldCover 10 m, CORINE Land Cover
    2018, and MODIS IGBP (see :mod:`~GeomorphConn.weights.tables`).

    C-factors are normalised to ``[0, 1]`` across the full range present in
    the supplied lookup table before clamping, so that the relative ordering
    of cover types is preserved even when raw C-factor values span a very
    small range.  Set ``normalise=False`` to use raw C-factor values directly.

    Parameters
    ----------
    landcover : array_like of int, shape (n_nodes,)
        Integer land-cover class codes matching the keys of *c_factor_table*.
        Nodata pixels should use *nodata_code* (default ``-1``).
    c_factor_table : dict[int, float], optional
        ``{class_code: c_factor}`` mapping.  Defaults to
        :data:`~GeomorphConn.weights.tables.WORLDCOVER_C_FACTOR`.
        Any code absent from the table is assigned *fallback_c*.
    nodata_code : int, optional
        Code used for nodata / unclassified pixels.  Default ``-1``.
    fallback_c : float, optional
        C-factor for codes absent from *c_factor_table*.  Default ``0.2``.
    w_min : float, optional
        Lower clamp.  Default ``0.005``.
    normalise : bool, optional
        If *True* (default), normalise C-factors to ``[0, 1]`` using the
        table's own min–max range.  If *False*, use raw values directly (they
        must already be in ``[0, 1]``).

    References
    ----------
    Renard, K.G. et al. (1997). USDA Agricultural Handbook 703.
    Wischmeier, W.H. & Smith, D.D. (1978). USDA Agricultural Handbook 537.
    Borselli, L. et al. (2008). Catena, 75(3), 268–277.

    Examples
    --------
    >>> import numpy as np
    >>> from GeomorphConn.weights import LandCoverWeight
    >>> from GeomorphConn.weights.tables import WORLDCOVER_C_FACTOR
    >>> lc = np.array([10, 40, 60, 80])   # tree, crop, bare, water
    >>> w  = LandCoverWeight(lc, WORLDCOVER_C_FACTOR, normalise=False).compute()
    >>> # bare (0.6) > crop (0.2) > tree (0.003) > water (0.0) → clamped
    """

    name = "landcover"

    def __init__(
        self,
        landcover: np.ndarray,
        c_factor_table: Optional[dict[int, float]] = None,
        nodata_code: int = -1,
        fallback_c: float = 0.2,
        w_min: float = 0.005,
        normalise: bool = True,
    ) -> None:
        from .tables import DEFAULT_C_FACTOR_TABLE

        self._lc = np.asarray(landcover, dtype=np.int32).ravel()
        self._table = (
            c_factor_table if c_factor_table is not None else DEFAULT_C_FACTOR_TABLE
        )
        self._nodata = int(nodata_code)
        self._fallback = float(fallback_c)
        self._w_min = float(w_min)
        self._normalise = bool(normalise)

    def compute(self) -> np.ndarray:
        """Return RUSLE C-factor weight, shape ``(n_nodes,)``."""
        # self._fallback is the user-supplied value for codes absent from the
        # table.  We look up each code independently; unknown codes → _fallback.
        c_vals = np.array(
            [self._table.get(int(code), self._fallback) for code in self._lc],
            dtype=np.float64,
        )
        if self._normalise:
            table_vals = np.array(list(self._table.values()), dtype=np.float64)
            c_min, c_max = float(table_vals.min()), float(table_vals.max())
            if c_max > c_min:
                c_vals = (c_vals - c_min) / (c_max - c_min)
        return _clamp(c_vals, self._w_min, 1.0)

    @classmethod
    def from_worldcover(
        cls,
        landcover: np.ndarray,
        **kwargs,
    ) -> "LandCoverWeight":
        """Construct using the built-in ESA WorldCover C-factor table."""
        from .tables import WORLDCOVER_C_FACTOR

        return cls(landcover, WORLDCOVER_C_FACTOR, **kwargs)

    @classmethod
    def from_corine(
        cls,
        landcover: np.ndarray,
        **kwargs,
    ) -> "LandCoverWeight":
        """Construct using the built-in CORINE 2018 C-factor table."""
        from .tables import CORINE_C_FACTOR

        return cls(landcover, CORINE_C_FACTOR, **kwargs)

    @classmethod
    def from_modis_igbp(
        cls,
        landcover: np.ndarray,
        **kwargs,
    ) -> "LandCoverWeight":
        """Construct using the built-in MODIS IGBP C-factor table."""
        from .tables import MODIS_IGBP_C_FACTOR

        return cls(landcover, MODIS_IGBP_C_FACTOR, **kwargs)


# ---------------------------------------------------------------------------
# CustomWeight
# ---------------------------------------------------------------------------

class CustomWeight:
    """
    Pass-through weight: accept any pre-computed array and apply clamping only.

    Use this when you have externally computed W values — e.g., from a
    physics-based erosion model, a published spatial dataset, or a formula
    that combines several factors in a way not covered by the other components.

    Parameters
    ----------
    values : array_like, shape (n_nodes,)
        Pre-computed weight values.  Should ideally be in ``[0, 1]`` before
        clamping, but values outside this range are silently clipped.
    w_min : float, optional
        Lower clamp.  Default ``0.005``.
    w_max : float, optional
        Upper clamp.  Default ``1.0``.

    Examples
    --------
    >>> import numpy as np
    >>> from GeomorphConn.weights import CustomWeight
    >>> CustomWeight(np.array([0.0, 0.5, 1.5])).compute()
    array([0.005, 0.5  , 1.   ])
    """

    name = "custom"

    def __init__(
        self,
        values: np.ndarray,
        w_min: float = 0.005,
        w_max: float = 1.0,
    ) -> None:
        self._vals = np.asarray(values, dtype=np.float64).ravel()
        self._w_min = float(w_min)
        self._w_max = float(w_max)

    def compute(self) -> np.ndarray:
        """Return clamped custom weight, shape ``(n_nodes,)``."""
        return _clamp(self._vals, self._w_min, self._w_max)
