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
    Terrain Ruggedness Index (TRI; Riley et al. 1999) normalised from the DEM.

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
    Surface roughness weight derived from the DEM via the Terrain Ruggedness
    Index (TRI; Riley et al. 1999).

    .. math::

        \\text{TRI}(i) = \\sqrt{\\sum_{j \\in \\mathcal{N}(i)} (z_j - z_i)^2}

    where :math:`\\mathcal{N}(i)` is the 3×3 Moore neighbourhood.  TRI is then
    min-max normalised to ``[0, 1]``.

    **Sign convention (``invert`` parameter)**

    The physical meaning of roughness in IC analysis depends on context:

    * ``invert=False`` (default) — rough cells receive *high* W, reflecting
      higher sediment availability and detachment potential (source-area
      interpretation; e.g., talus, badlands).
    * ``invert=True`` — rough cells receive *low* W, reflecting higher
      hydraulic roughness that impedes downslope transfer (impedance
      interpretation; follows Cavalli et al. 2013 Appendix roughness proxy).

    Parameters
    ----------
    grid : RasterModelGrid
        Landlab grid with ``'topographic__elevation'`` at nodes.
    w_min : float, optional
        Lower clamp.  Default ``0.005``.
    invert : bool, optional
        If *True*, return ``1 − TRI_norm`` (rough → low W).  Default *False*.

    References
    ----------
    Riley, S.J., DeGloria, S.D., & Elliot, R. (1999). A terrain ruggedness
        index that quantifies topographic heterogeneity. Intermountain Journal
        of Sciences, 5(1–4), 23–27.
    """

    name = "roughness"

    def __init__(
        self,
        grid,
        w_min: float = 0.005,
        invert: bool = False,
    ) -> None:
        self._grid = grid
        self._w_min = float(w_min)
        self._invert = bool(invert)

    def compute(self) -> np.ndarray:
        """Return TRI-based roughness weight, shape ``(n_nodes,)``."""
        nrows = self._grid.number_of_node_rows
        ncols = self._grid.number_of_node_columns
        elev = self._grid.at_node["topographic__elevation"].reshape(nrows, ncols)

        tri = np.zeros((nrows, ncols), dtype=np.float64)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r0 = max(0, -dr)
                r1 = nrows - max(0, dr)
                c0 = max(0, -dc)
                c1 = ncols - max(0, dc)
                rn0 = max(0, dr)
                rn1 = nrows - max(0, -dr)
                cn0 = max(0, dc)
                cn1 = ncols - max(0, -dc)
                diff = elev[r0:r1, c0:c1] - elev[rn0:rn1, cn0:cn1]
                tri[r0:r1, c0:c1] += diff ** 2

        tri_flat = np.sqrt(tri).ravel()
        normed = _minmax_norm(tri_flat)
        if self._invert:
            normed = 1.0 - normed
        return _clamp(normed, self._w_min, 1.0)


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
