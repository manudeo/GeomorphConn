# -*- coding: utf-8 -*-
"""
connectivity_index.py
=====================
Landlab component implementing the hydrologically-weighted Index of
Connectivity (IC) after Cavalli et al. (2013), extended with an NDVI-based
C-factor and normalised rainfall as weighting factors (Dubey, Singh & Jain; submitted).

Flow routing uses Landlab's built-in FlowDirector classes (D8, D-infinity,
and MFD). The downstream flow-path length (D_dn) always uses D8 steepest
descent to remain consistent with the ArcGIS reference implementation.

**Unit convention:** All spatial inputs (elevation, grid spacing) and outputs
(D_up, D_dn, drainage area) are assumed to be in SI units (meters).  Slope
is always dimensionless (rise/run) regardless of the approximation used.

References
----------
Cavalli, M., Trevisani, S., Comiti, F., & Marchi, L. (2013).
    Geomorphology, 188, 31–41. https://doi.org/10.1016/j.geomorph.2012.05.010

Crema, S. & Cavalli, M. (2018). SedInConnect: a stand-alone, free and
    open source tool for the assessment of sediment connectivity.
    Computers & Geosciences, 111, 39–45. https://doi.org/10.1016/j.cageo.2017.10.009

Singh, M., Cavalli, M. & Crema, S. (2026). GeomorphConn: A Python package for
    hydrologically-weighted sediment connectivity. JOSS.
"""

from __future__ import annotations

import warnings

import numpy as np
from landlab import Component

# ── Optional numba JIT ────────────────────────────────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:  # pragma: no cover
    _NUMBA = False

# ── Flow-director name aliases ────────────────────────────────────────────────
_LL_DIRECTORS = {
    "D8"                    : "FlowDirectorSteepest",
    "FlowDirectorSteepest"  : "FlowDirectorSteepest",
    "DINF"                  : "FlowDirectorDINF",
    "FlowDirectorDINF"      : "FlowDirectorDINF",
    "MFD"                   : "FlowDirectorMFD",
    "FlowDirectorMFD"       : "FlowDirectorMFD",
}


def _import_ll_director(name: str):
    canonical = _LL_DIRECTORS.get(name)
    if canonical is None:
        raise ValueError(
            f"Unknown flow director '{name}'. "
            f"Valid options: {sorted(_LL_DIRECTORS)}"
        )
    from landlab.components import (
        FlowDirectorDINF,
        FlowDirectorMFD,
        FlowDirectorSteepest,
    )
    return {
        "FlowDirectorSteepest": FlowDirectorSteepest,
        "FlowDirectorDINF": FlowDirectorDINF,
        "FlowDirectorMFD": FlowDirectorMFD,
    }[canonical]


# ═════════════════════════════════════════════════════════════════════════════
#  Accumulation kernels (Python + optional numba JIT)
# ═════════════════════════════════════════════════════════════════════════════

def _acc_d8_py(weight, receivers, node_order):
    """Single-receiver weighted upstream accumulation (D8)."""
    acc = np.zeros(len(weight), dtype=np.float64)
    for node in node_order:
        rec = int(receivers[node])
        if rec != node:
            acc[rec] += acc[node] + weight[node]
    return acc


def _acc_mfd_py(weight, receivers, proportions, node_order):
    """Multi-receiver weighted upstream accumulation (D-inf / MFD)."""
    acc   = np.zeros(len(weight), dtype=np.float64)
    n_rec = receivers.shape[1]
    for i in range(len(node_order)):
        node = int(node_order[i])
        val  = float(acc[node]) + float(weight[node])
        for k in range(n_rec):
            rec  = int(receivers[node, k])
            prop = float(proportions[node, k])
            # guard: Landlab uses -1 as sentinel for unused receiver slots
            if rec >= 0 and rec != node and prop > 0.0:
                acc[rec] += val * prop
    return acc


def _ddn_d8_py(local_contrib, inv_WS, receivers, node_order):
    """
    Downstream weighted flow-path length (D_dn), D8 only.
    Reverse-topological traversal: outlets first.

    Terminal nodes (outlets / target sinks) get D_dn = inv_WS[node], matching
    the Borselli (2008) ArcGIS recipe step:
      ``(([X] == 0) * [inv_CS]) + [X]``
    which ensures channel/outlet pixels carry their own pixel impedance.
    """
    ddn = local_contrib.copy()
    for node in node_order[::-1]:
        rec = int(receivers[node])
        if rec == node:                 # outlet / imposed sink
            ddn[node] = inv_WS[node]
        else:
            ddn[node] = local_contrib[node] + ddn[rec]
    return ddn


if _NUMBA:
    @_njit(cache=True)
    def _acc_d8_jit(weight, receivers, node_order):
        acc = np.zeros(len(weight))
        for i in range(len(node_order)):
            node = node_order[i]
            rec  = receivers[node]
            if rec != node:
                acc[rec] += acc[node] + weight[node]
        return acc

    @_njit(cache=True)
    def _acc_mfd_jit(weight, receivers, proportions, node_order):
        acc   = np.zeros(len(weight))
        n_rec = receivers.shape[1]
        for i in range(len(node_order)):
            node = node_order[i]
            val  = acc[node] + weight[node]
            for k in range(n_rec):
                rec  = receivers[node, k]
                prop = proportions[node, k]
                if rec != node and prop > 0.0:
                    acc[rec] += val * prop
        return acc

    @_njit(cache=True)
    def _ddn_d8_jit(local_contrib, inv_WS, receivers, node_order):
        ddn = local_contrib.copy()
        n   = len(node_order)
        for i in range(n - 1, -1, -1):
            node = node_order[i]
            rec  = receivers[node]
            if rec == node:
                ddn[node] = inv_WS[node]
            else:
                ddn[node] = local_contrib[node] + ddn[rec]
        return ddn

    # Wrappers with dtype enforcement
    def _acc_d8(w, rec, order):
        return _acc_d8_jit(
            w.astype(np.float64), rec.astype(np.int64), order.astype(np.int64)
        )

    def _acc_mfd(w, rec, prop, order):
        return _acc_mfd_jit(w.astype(np.float64), rec.astype(np.int64),
                            prop.astype(np.float64), order.astype(np.int64))

    def _ddn_d8(lc, iws, rec, order):
        return _ddn_d8_jit(lc.astype(np.float64), iws.astype(np.float64),
                           rec.astype(np.int64), order.astype(np.int64))
else:
    _acc_d8  = _acc_d8_py
    _acc_mfd = _acc_mfd_py
    _ddn_d8  = _ddn_d8_py


# ═════════════════════════════════════════════════════════════════════════════
#  Main Component
# ═════════════════════════════════════════════════════════════════════════════

class ConnectivityIndex(Component):
    """
    Hydrologically-weighted Index of Connectivity (IC) Landlab component.

    Extends the topographic IC of Cavalli et al. (2013) with NDVI-based
    C-factor and normalised rainfall weighting. The original reference
    implementation of the IC is SedInConnect (Crema & Cavalli, 2018).

    **Core equations** (SI units):

        RF_norm = (RF − RF_min) / (RF_max − RF_min)    [dimensionless]
        C       = (1 − NDVI) / 2                        [dimensionless]
        W       = clip( (RF_norm + C) / 2,  w_min, 1 ) [dimensionless]
        
        S       = tan(θ)  or  θ° / 100                 [dimensionless, rise/run]
        D_up    = W̄ · S̄ · √A                           [m]
        D_dn    = Σ d_i / (W_i · S_i)                  [m]
        IC      = log₁₀( D_up / D_dn )                  [dimensionless]

    where A is in m², d_i is in m, and all weights and slopes are dimensionless.
    The product W̄·S̄·√A consistently yields meters despite all factors being
    dimensionless because √(m²) contributes the length dimension.

    Parameters
    ----------
    grid : RasterModelGrid
        Landlab grid with ``'topographic__elevation'`` set at nodes (in meters).
        Grid spacing (xy_spacing or dx/dy) must also be in meters.  All
        output distances (D_up, D_dn) will be in meters.
    flow_director : str, optional
        Flow algorithm for **upstream** accumulation.
        ``'D8'``, ``'DINF'``, ``'MFD'`` (aliases accepted).
        Default: ``'DINF'``.
    weight : WeightBuilder or array_like or None, optional
        Controls how the spatial weight W is constructed.  Three forms:

        * **None** (default) — uses *ndvi* and *rainfall* kwargs (legacy
          interface, backward compatible).  Equivalent to
          ``preset_rainfall_ndvi(rainfall, ndvi)``.
        * **WeightBuilder** — a fully configured weight pipeline built from
          one or more components (rainfall, NDVI, land-cover C-factor,
          surface roughness, or custom).  See :mod:`~GeomorphConn.weights` for
          details and preset helpers.
        * **array_like** — a pre-computed W array of shape ``(n_nodes,)``.
          Only clamping is applied.  Use when you have externally computed W.

        This is the **main tuning switch** for the IC.  Examples::

            # Default (NDVI + rainfall)
            ic = ConnectivityIndex(grid, ndvi=ndvi_arr, rainfall=rf_arr)

            # Land-cover C-factor only (Borselli 2008 spirit)
            from geomorphconn.weights import preset_landcover_only
            ic = ConnectivityIndex(grid,
                weight=preset_landcover_only(lc_arr, WORLDCOVER_C_FACTOR))

            # Roughness-only (DEM-only, no satellite data)
            from geomorphconn.weights import preset_roughness_only
            ic = ConnectivityIndex(grid, weight=preset_roughness_only(grid))

            # Full three-component pipeline
            from geomorphconn.weights import preset_rainfall_ndvi_roughness
            ic = ConnectivityIndex(grid,
                weight=preset_rainfall_ndvi_roughness(rf, ndvi, grid))

            # Custom combination
            from geomorphconn.weights import WeightBuilder, RainfallWeight, LandCoverWeight
            wb = (WeightBuilder(combine="geometric_mean")
                  .add(RainfallWeight(rf))
                  .add(LandCoverWeight.from_worldcover(lc)))
            ic = ConnectivityIndex(grid, weight=wb)

            # Fully manual W (your own formula)
            my_W = (rf_norm * 0.7 + ndvi_c * 0.3)
            ic = ConnectivityIndex(grid, weight=my_W)

    ndvi : array_like or str, optional
        NDVI at each node, values in [−1, 1].  Used only when *weight* is
        *None*.  Default: 0 (bare soil).
    rainfall : array_like or str, optional
        Rainfall at each node (any units).  Used only when *weight* is *None*.
        Default: spatially uniform.
    slope : array_like or str or None, optional
        Pre-computed slope array (dimensionless, rise/run) at each node.
        Use this to supply slopes from an external tool (TauDEM, ArcGIS) for
        comparison or consistency with prior analyses.

        * **None** (default): Slope is computed internally via Landlab's D8
          steepest descent algorithm. This is the default and most reproducible.
        * **array_like**: Must have shape (n_nodes,) with values in tan(θ) form
          (dimensionless rise/run). Custom slopes bypass Landlab's calculation.
        * **str**: Name of an existing grid field (e.g., pass ``'slope'`` if you
          pre-loaded TauDEM slopes into the grid).

        **Note on algorithmic differences & TauDEM consistency:**
        
        Results will differ systematically between TauDEM (SedInConnect; Crema &
        Cavalli 2018), ArcGIS, and Landlab due to different slope calculation
        and flow-routing algorithms. Typical differences: 5–10% IC bias on
        complex terrain, up to 20% on very steep slopes.
        
        * **TauDEM** (SedInConnect reference): Uses 3×3 aspect-weighted planar
          slope calculation; D-infinity flow model (Tarboton 1997).
        * **ArcGIS**: Uses D8 flow; steepest-descent slope similar to Landlab.
        * **Landlab** (this code, default): D8/DINF/MFD (upstream); D8 steepest
          descent (downstream); priority-flood pit-filling.
        
        To match TauDEM results, compute slopes externally via TauDEM CLI
        tools (``PitFill``, ``DInfFlowDir`` or ``D8FlowDir``) and pass them via
        this parameter. Similarly, to match ArcGIS, export ArcGIS slopes
        (though Landlab D8 slopes are usually similar to ArcGIS).

    target_nodes : array_like of int, optional
        Node IDs of target features (river/lake).  If *None*, IC is
        computed toward the natural basin outlet.  Provide via
        :func:`~GeomorphConn.utils.rasterize_targets`.
    stream_threshold : int or None, optional
        Automatically define the channel network from D8 flow accumulation.
        Every node whose upstream cell count >= *stream_threshold* is treated
        as a target (channel) node, matching the Borselli (2008) ArcGIS recipe
        step ``[ACCMASK] <= threshold`` (RIVERMASK).

        * E.g. ``stream_threshold=1000``: pixels draining >= 1000 cells → channel.
        * Typical range for 30 m grids: 500–2000 for first-order streams.
        * If *target_nodes* is also given, the two sets are merged (union).
        * Default *None* — no automatic channel detection.
    fill_sinks : bool, optional
        If *True*, fill depressions explicitly with
        :class:`landlab.components.SinkFillerBarnes` before flow routing.
        This modifies the input DEM and best matches the ArcGIS-style workflow
        of Fill -> FlowDirection -> FlowAccumulation. Default is *False*,
        which routes directly on the input DEM without modification.
    use_aspect_weighting : bool, optional
        If *True*, apply an additional aspect-alignment weighting to
        multi-receiver upstream routing (DINF/MFD) during D_up accumulation.
        This can improve consistency with TauDEM-style D-infinity area
        partitioning on complex terrain. Default is *False*.

        Notes:
        - Only affects D_up/Wmean/Smean when using multi-receiver routing.
        - Has no effect for D8 (single receiver).
        - D_dn remains D8-based, unchanged.
    w_min : float, optional
        Lower clamp for W and S (prevents ÷0).  Default 0.005.
    w_max : float, optional
        Upper clamp for W.  Default 1.0.
    use_degree_approx : bool, optional
        Slope factor computation convention (applies to internally computed slopes only).
        Both forms are dimensionless (rise/run):

        * *True* (default): ``S = θ° / 100`` where θ° is steepest-descent angle
          in degrees.  Matches ArcGIS and is suitable for low-gradient terrain
          (θ < 10°, where θ° ≈ 100·tan(θ)).
        * *False*: ``S = tan(θ)`` (physically exact, rise/run).  Use for steep
          terrain (θ > 10°) where the approximation breaks down.

        **Important:** If *slope* parameter is provided, it is assumed to be
        in tan(θ) form and is used directly without conversion.

    Output grid fields
    ------------------
    ``connectivity_index__IC``
        Index of Connectivity = log₁₀(D_up / D_dn) [dimensionless].
    ``connectivity_index__Dup``
        Upstream component D_up = W̄·S̄·√A [m]. Represents potential for
        sediment delivery from contributing area.
    ``connectivity_index__Ddn``
        Downstream component = Σ d_i / (W_i·S_i) [m]. Represents impedance
        of downslope pathway (D8 steepest descent).
    ``connectivity_index__W``
        Hydrological weight W [dimensionless, range 0.005–1.0].
    ``connectivity_index__S``
        Slope factor S [dimensionless, range 0.005–1.0].  Both formulations
        (θ°/100 or tan(θ)) are dimensionless and represent rise/run.
    ``connectivity_index__Wmean``
        Area-weighted mean W over upstream contributing area [dimensionless].
    ``connectivity_index__Smean``
        Area-weighted mean S over upstream contributing area [dimensionless].

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from geomorphconn import ConnectivityIndex
    >>> grid = RasterModelGrid((15, 15), xy_spacing=30.0)
    >>> z = grid.add_zeros("topographic__elevation", at="node")
    >>> z += np.random.default_rng(0).random(grid.number_of_nodes) * 30
    >>> ndvi = np.full(grid.number_of_nodes, 0.4)
    >>> rf   = np.full(grid.number_of_nodes, 800.0)
    >>> ic = ConnectivityIndex(grid, ndvi=ndvi, rainfall=rf)
    >>> ic.run_one_step()
    >>> np.isfinite(ic.IC).any()
    True
    """

    _name = "ConnectivityIndex"
    _unit_agnostic = False

    _info = {
        "topographic__elevation": {
            "dtype": float, "intent": "in", "optional": False,
            "units": "m", "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "connectivity_index__IC": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "-", "mapping": "node",
            "doc": "Index of Connectivity IC = log10(D_up / D_dn)",
        },
        "connectivity_index__Dup": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "m", "mapping": "node",
            "doc": "Upstream IC component D_up = W_mean * S_mean * sqrt(A)",
        },
        "connectivity_index__Ddn": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "m", "mapping": "node",
            "doc": "Downstream IC component: weighted path length to target",
        },
        "connectivity_index__W": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "-", "mapping": "node",
            "doc": "Hydrological weight W = (RF_norm + C_factor) / 2",
        },
        "connectivity_index__S": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "-", "mapping": "node",
            "doc": "Clamped slope factor S",
        },
        "connectivity_index__Wmean": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "-", "mapping": "node",
            "doc": "Area-weighted mean W over upstream contributing area",
        },
        "connectivity_index__Smean": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "-", "mapping": "node",
            "doc": "Area-weighted mean S over upstream contributing area",
        },
    }

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        grid,
        flow_director: str = "DINF",
        weight=None,
        ndvi=None,
        rainfall=None,
        slope=None,
        target_nodes=None,
        stream_threshold=None,
        fill_sinks: bool = False,
        use_aspect_weighting: bool = False,
        w_min: float = 0.005,
        w_max: float = 1.0,
        use_degree_approx: bool = True,
    ):
        super().__init__(grid)

        self._fd_name        = flow_director
        # Validate director name eagerly so errors surface at construction time
        _import_ll_director(flow_director)
        self._w_min          = float(w_min)
        self._w_max          = float(w_max)
        self._use_deg_approx = bool(use_degree_approx)
        self._use_aspect_weighting = bool(use_aspect_weighting)
        self._fill_sinks = bool(fill_sinks)
        self._stream_threshold = int(stream_threshold) if stream_threshold is not None else None
        n = grid.number_of_nodes

        # ── Slope (optional user-provided override) ────────────────────
        if slope is not None:
            self._slope_array = self._coerce_field(slope, n, None, "slope")
        else:
            self._slope_array = None

        # ── Target nodes ──────────────────────────────────────────────
        self._target_nodes = (
            np.asarray(target_nodes, dtype=np.int64)
            if target_nodes is not None else None
        )

        # ── Weight configuration ───────────────────────────────────────
        # Three modes:
        #   1. weight=WeightBuilder  → use builder directly
        #   2. weight=array_like     → pre-computed W, only clamp applied
        #   3. weight=None           → legacy ndvi= / rainfall= interface
        if weight is not None:
            from ..weights import WeightBuilder
            if isinstance(weight, WeightBuilder):
                self._weight_builder = weight
                self._weight_array   = None
            else:
                # Assume pre-computed array
                arr = np.asarray(weight, dtype=np.float64).ravel()
                if len(arr) != n:
                    raise ValueError(
                        f"'weight' array length {len(arr)} != n_nodes {n}"
                    )
                self._weight_builder = None
                self._weight_array   = arr
            # Warn if ndvi/rainfall also supplied — they will be ignored
            if ndvi is not None or rainfall is not None:
                warnings.warn(
                    "Both 'weight' and ('ndvi' / 'rainfall') were supplied. "
                    "'ndvi' and 'rainfall' are ignored when 'weight' is set.",
                    UserWarning, stacklevel=2,
                )
        else:
            # Legacy interface: build default WeightBuilder from ndvi + rainfall
            self._weight_builder = None
            self._weight_array   = None
            # Store for _build_default_weight()
            self._ndvi     = self._coerce_field(ndvi,     n, 0.0, "ndvi")
            self._rainfall = self._coerce_field(rainfall, n, 1.0, "rainfall")

        self._weight_mode = (
            "builder"    if self._weight_builder is not None else
            "precomputed" if (weight is not None and self._weight_builder is None)
            else "legacy"
        )

        if not _NUMBA:
            warnings.warn(
                "numba not installed — IC loops will use pure Python and "
                "may be slow for grids > ~500×500.  "
                "Speed up with: pip install numba",
                UserWarning, stacklevel=2,
            )

        self.initialize_output_fields()

    # ── Public helpers ────────────────────────────────────────────────────────

    def update_weight(self, weight):
        """
        Replace the weight configuration without rebuilding the component.

        Parameters
        ----------
        weight : WeightBuilder, array_like, or None
            New weight specification.  Same forms as the constructor *weight*
            parameter.  Pass *None* to revert to the legacy ndvi/rainfall mode.
        """
        from ..weights import WeightBuilder
        n = self._grid.number_of_nodes
        if weight is None:
            self._weight_builder = None
            self._weight_array   = None
            self._weight_mode    = "legacy"
        elif isinstance(weight, WeightBuilder):
            self._weight_builder = weight
            self._weight_array   = None
            self._weight_mode    = "builder"
        else:
            arr = np.asarray(weight, dtype=np.float64).ravel()
            if len(arr) != n:
                raise ValueError(
                    f"'weight' array length {len(arr)} != n_nodes {n}"
                )
            self._weight_builder = None
            self._weight_array   = arr
            self._weight_mode    = "precomputed"

    def update_ndvi(self, ndvi):
        """
        Replace NDVI values (legacy interface).

        Only has effect when *weight* was not set (legacy ndvi/rainfall mode).
        To update a WeightBuilder pipeline, call :meth:`update_weight` with a
        new builder.
        """
        if self._weight_mode != "legacy":
            warnings.warn(
                "update_ndvi() has no effect when 'weight' is set. "
                "Use update_weight() instead.",
                UserWarning, stacklevel=2,
            )
            return
        self._ndvi = self._coerce_field(
            ndvi, self._grid.number_of_nodes, default_val=0.0, name="ndvi"
        )

    def update_rainfall(self, rainfall):
        """
        Replace rainfall values (legacy interface).

        Only has effect when *weight* was not set (legacy ndvi/rainfall mode).
        """
        if self._weight_mode != "legacy":
            warnings.warn(
                "update_rainfall() has no effect when 'weight' is set. "
                "Use update_weight() instead.",
                UserWarning, stacklevel=2,
            )
            return
        self._rainfall = self._coerce_field(
            rainfall, self._grid.number_of_nodes, default_val=1.0, name="rainfall"
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_one_step(self):
        """
        Compute IC for the current grid state.

        All seven output fields are updated in-place.
        """
        # 1. Fill + route
        routing = self._run_routing()
        
        # 2. Resolve slope (custom or Landlab-computed)
        if self._slope_array is None:
            # Landlab-computed slope: apply degree approximation if requested
            slope_tan = routing["slope_tan"]
            if self._use_deg_approx:
                # Convert tan(θ) to θ°/100 for ArcGIS compatibility
                slope_tan = np.degrees(np.arctan(np.clip(slope_tan, 0.0, None))) / 100.0
        else:
            # Custom slope: assumed to be tan(θ) already, use directly
            slope_tan = self._slope_array.copy()

        # 3. W and S (S will be clamped in _compute_W_S)
        W, S = self._compute_W_S(slope_tan)

        # 3. D_up via upstream accumulation
        Dup, Wmean, Smean = self._compute_Dup(W, S, routing)

        # 4. D_dn via D8 downstream path
        Ddn = self._compute_Ddn(W, S, routing)

        # 5. IC
        with np.errstate(divide="ignore", invalid="ignore"):
            IC = np.where(
                (Dup > 0) & (Ddn > 0),
                np.log10(Dup / Ddn),
                np.nan,
            )

        # In target mode, target/channel cells are masked to NaN in outputs.
        eff_targets = routing.get("effective_target_nodes")
        if eff_targets is not None:
            IC[eff_targets]    = np.nan
            Dup[eff_targets]   = np.nan
            Ddn[eff_targets]   = np.nan
            Wmean[eff_targets] = np.nan
            Smean[eff_targets] = np.nan

        # 6. Write fields
        g = self._grid
        g.at_node["connectivity_index__IC"][:]    = IC
        g.at_node["connectivity_index__Dup"][:]   = Dup
        g.at_node["connectivity_index__Ddn"][:]   = Ddn
        g.at_node["connectivity_index__W"][:]     = W
        g.at_node["connectivity_index__S"][:]     = S
        g.at_node["connectivity_index__Wmean"][:] = Wmean
        g.at_node["connectivity_index__Smean"][:] = Smean

    # ── Routing ───────────────────────────────────────────────────────────────

    def _run_routing(self) -> dict:
        """
        Run Landlab routing and return a dict with keys:

        ``d8_receivers``     int64 (n,)   — D8 receiver per node
        ``slope_tan``        float64 (n,) — |tan(steepest slope)|
        ``receivers``        int64 (n,) or (n, k) — primary direction receivers
        ``proportions``      float64 (n, k) or None — receiver proportions (MFD/DINF)
        ``node_order``       int64 (n,)   — headwaters→outlets topological order
        ``d8_node_order``    int64 (n,)   — D8 headwaters→outlets topological order
        ``drainage_area``    float64 (n,) — m²
        """
        return self._run_landlab_routing()

    def _run_landlab_routing(self) -> dict:
        """
        Two-stage Landlab routing:

        Stage 0  Optionally fill depressions explicitly using SinkFillerBarnes
                 (ArcGIS-like workflow), producing a filled working DEM.

        Stage 1  Run D8 FlowAccumulator on the working DEM to harvest D8
                 receivers, slope, and node ordering.

        Stage 2  If the primary director is not D8, copy the *filled* elevation
                 onto a fresh grid and run the primary FlowAccumulator there.
                 A fresh grid avoids the field-shape conflict
                 (D8 sets fields as (n,) whereas DINF/MFD use (n,k)).

        D_dn always uses D8 receivers.  Slope is always taken from the D8 stage
        so it is a 1-D (n,) array regardless of the primary director.
        """
        from landlab import RasterModelGrid as _RMG
        from landlab.components import (
            FlowAccumulator,
            FlowDirectorSteepest,
            SinkFillerBarnes,
        )

        grid  = self._grid
        FDCls = _import_ll_director(self._fd_name)
        is_d8 = FDCls is FlowDirectorSteepest

        # Build a working grid so we can fill sinks without mutating user DEM.
        g_work = _RMG(grid.shape, xy_spacing=float(grid.dx))
        g_work.status_at_node[:] = grid.status_at_node
        g_work.add_field(
            "topographic__elevation",
            grid.at_node["topographic__elevation"].copy(),
            at="node",
        )

        # ── Stage 0: explicit sink filling (ArcGIS-like) ──────────────
        if self._fill_sinks:
            sf = SinkFillerBarnes(
                g_work,
                surface="topographic__elevation",
                method="D8",
                fill_flat=False,
            )
            sf.run_one_step()

        # ── Stage 1: D8 (pit-filling + slope + D8 receivers) ──────────
        fa_d8 = FlowAccumulator(g_work, "topographic__elevation",
                                flow_director="FlowDirectorSteepest")
        fa_d8.run_one_step()

        d8_recv   = g_work.at_node["flow__receiver_node"].copy().astype(np.int64)
        # topographic__steepest_slope is (n,) after D8
        slope_tan = np.abs(g_work.at_node["topographic__steepest_slope"].copy().ravel())
        d8_order  = g_work.at_node["flow__upstream_node_order"].copy().astype(np.int64)

        # ── Resolve effective target nodes ─────────────────────────────
        # Start with any vector-supplied target nodes.
        eff_targets = self._target_nodes  # may be None

        # Add stream-threshold channel nodes (Borselli RIVERMASK equivalent).
        if self._stream_threshold is not None:
            cell_area = float(grid.dx) * float(grid.dy)
            cell_count = g_work.at_node["drainage_area"] / cell_area
            stream_nodes = np.where(cell_count >= self._stream_threshold)[0].astype(np.int64)
            if eff_targets is not None:
                eff_targets = np.unique(np.concatenate([eff_targets, stream_nodes]))
            elif len(stream_nodes) > 0:
                eff_targets = stream_nodes

        # Impose effective targets on D8 receiver array (self-loop = local sink).
        if eff_targets is not None:
            d8_recv[eff_targets] = eff_targets

        if is_d8:
            # D8 is also the primary — everything comes from Stage 1.
            return {
                "d8_receivers"  : d8_recv,
                "slope_tan"     : slope_tan,
                "receivers"     : d8_recv.copy(),
                "proportions"   : None,
                "node_order"    : d8_order,
                "d8_node_order" : d8_order,
                "drainage_area" : g_work.at_node["drainage_area"].copy(),
                "effective_target_nodes": eff_targets,
            }

        # ── Stage 2: primary director on a fresh grid ──────────────────
        # Use the (already pit-filled) elevation from Stage 1.
        g2 = _RMG(grid.shape, xy_spacing=float(grid.dx))
        g2.status_at_node[:] = grid.status_at_node
        g2.add_field("topographic__elevation",
                     g_work.at_node["topographic__elevation"].copy(),
                     at="node")

        fa2 = FlowAccumulator(g2, "topographic__elevation",
                              flow_director=FDCls)
        fa2.run_one_step()

        rec2   = g2.at_node["flow__receiver_node"].copy().astype(np.int64)
        prop2  = g2.at_node["flow__receiver_proportions"].copy().astype(np.float64)
        order2 = g2.at_node["flow__upstream_node_order"].copy().astype(np.int64)
        da2    = g2.at_node["drainage_area"].copy()

        # Impose targets
        if eff_targets is not None:
            rec2[eff_targets, :]  = eff_targets[:, None]
            prop2[eff_targets, :] = 0.0

        return {
            "d8_receivers"  : d8_recv,
            "slope_tan"     : slope_tan,
            "receivers"     : rec2,
            "proportions"   : prop2,
            "node_order"    : order2,
            "d8_node_order" : d8_order,
            "drainage_area" : da2,
            "effective_target_nodes": eff_targets,
        }

    # ── Weights ───────────────────────────────────────────────────────────────

    def _compute_W_S(self, slope_tan: np.ndarray):
        """
        Compute the weight array W and slope factor S.

        W is resolved from one of three modes set at construction time:
        - 'builder'     : call WeightBuilder.build()
        - 'precomputed' : use stored array (only clamp)
        - 'legacy'      : legacy NDVI + rainfall formula
        """
        n = self._grid.number_of_nodes

        if self._weight_mode == "builder":
            W = self._weight_builder.build(n_nodes=n)
            W = np.clip(W.astype(np.float64), self._w_min, self._w_max)

        elif self._weight_mode == "precomputed":
            W = np.clip(
                self._weight_array.astype(np.float64), self._w_min, self._w_max
            )

        else:  # legacy: ndvi + rainfall
            rf      = self._rainfall.astype(np.float64)
            rf_min, rf_max = rf.min(), rf.max()
            RF_norm = (rf - rf_min) / (rf_max - rf_min) if rf_max > rf_min \
                      else np.full_like(rf, 0.5)
            C = (1.0 - self._ndvi.astype(np.float64)) / 2.0
            W = np.clip((RF_norm + C) / 2.0, self._w_min, self._w_max)

        # Slope factor S — independent of weight mode
        # slope_tan is already in the correct form (tan(θ) or θ°/100) from run_one_step().
        # Just clip to valid range [w_min, w_max].
        S = np.clip(slope_tan, self._w_min, self._w_max)

        return W.astype(np.float64), S.astype(np.float64)

    # ── D_up ──────────────────────────────────────────────────────────────────

    def _compute_Dup(self, W, S, routing):
        cell_area  = float(self._grid.dx) * float(self._grid.dy)
        n_nodes    = self._grid.number_of_nodes
        node_order = routing["node_order"]
        receivers  = routing["receivers"]
        proportions= routing["proportions"]
        DA         = routing["drainage_area"]
        eff_targets = routing.get("effective_target_nodes")

        if proportions is not None:
            eff_props = proportions
            if self._use_aspect_weighting:
                eff_props = self._build_aspect_weighted_proportions(
                    receivers,
                    proportions,
                    routing["d8_receivers"],
                )

            AccW = _acc_mfd(W, receivers, eff_props, node_order)
            AccS = _acc_mfd(S, receivers, eff_props, node_order)

            if self._use_aspect_weighting or eff_targets is not None:
                # Recompute contributing-cell count under modified partitioning.
                AccA = _acc_mfd(np.ones(n_nodes, dtype=np.float64), receivers, eff_props, node_order)
                ACC_final = np.maximum(AccA + 1.0, 1.0)
            else:
                ACC_final = np.maximum(DA / cell_area, 1.0)
        else:
            AccW = _acc_d8(W, receivers, node_order)
            AccS = _acc_d8(S, receivers, node_order)
            if eff_targets is not None:
                # In target mode, drainage_area from Landlab is computed before
                # target outlets are imposed. Recompute contributing count from
                # the modified receiver graph so each target acts as terminal.
                AccA = _acc_d8(np.ones(n_nodes, dtype=np.float64), receivers, node_order)
                ACC_final = np.maximum(AccA + 1.0, 1.0)
            else:
                ACC_final = np.maximum(DA / cell_area, 1.0)

        Wmean = (AccW + W) / ACC_final
        Smean = (AccS + S) / ACC_final
        A     = ACC_final * cell_area
        Dup   = Wmean * Smean * np.sqrt(A)
        return Dup, Wmean, Smean

    def _build_aspect_weighted_proportions(self, receivers, proportions, d8_receivers):
        """
        Build aspect-weighted receiver proportions for multi-flow routing.

        Each outgoing proportion is multiplied by max(cos(delta), 0), where
        delta is the angle between the D8 steepest-descent direction and the
        edge direction to that receiver. Rows are then renormalized to sum to 1.
        """
        g = self._grid
        n, k = receivers.shape
        out = proportions.copy().astype(np.float64)

        d8_recv = np.asarray(d8_receivers, dtype=np.int64)
        xy = g.xy_of_node

        # D8 reference direction vector per node.
        d8_dx = xy[d8_recv, 0] - xy[:, 0]
        d8_dy = xy[d8_recv, 1] - xy[:, 1]
        d8_norm = np.sqrt(d8_dx * d8_dx + d8_dy * d8_dy)

        valid_d8 = (d8_recv != np.arange(n, dtype=np.int64)) & (d8_norm > 0.0)

        for j in range(k):
            rec = receivers[:, j]
            dx = np.zeros(n, dtype=np.float64)
            dy = np.zeros(n, dtype=np.float64)

            valid_rec = rec >= 0
            if np.any(valid_rec):
                dx[valid_rec] = xy[rec[valid_rec], 0] - xy[valid_rec, 0]
                dy[valid_rec] = xy[rec[valid_rec], 1] - xy[valid_rec, 1]

            r_norm = np.sqrt(dx * dx + dy * dy)
            good = valid_d8 & valid_rec & (r_norm > 0.0)

            cos_delta = np.zeros(n, dtype=np.float64)
            cos_delta[good] = (
                (d8_dx[good] * dx[good] + d8_dy[good] * dy[good])
                / (d8_norm[good] * r_norm[good])
            )
            cos_delta = np.clip(cos_delta, 0.0, 1.0)
            out[:, j] *= cos_delta

        row_sum = out.sum(axis=1)
        nz = row_sum > 0.0
        out[nz, :] = out[nz, :] / row_sum[nz, None]

        # Fallback to original proportions when aspect weights collapse to zero.
        out[~nz, :] = proportions[~nz, :]
        return out

    # ── D_dn ──────────────────────────────────────────────────────────────────

    def _compute_Ddn(self, W, S, routing):
        grid     = self._grid
        recv     = routing["d8_receivers"].copy()
        order    = routing["d8_node_order"]

        xy   = grid.xy_of_node
        dx_v = xy[recv, 0] - xy[:, 0]
        dy_v = xy[recv, 1] - xy[:, 1]
        dist = np.sqrt(dx_v**2 + dy_v**2)
        dist[recv == np.arange(len(recv), dtype=np.int64)] = 0.0

        inv_WS = 1.0 / (W * S)
        return _ddn_d8(dist * inv_WS, inv_WS, recv, order)

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def IC(self) -> np.ndarray:
        """IC at all nodes (1-D)."""
        return self._grid.at_node["connectivity_index__IC"]

    @property
    def Dup(self) -> np.ndarray:
        return self._grid.at_node["connectivity_index__Dup"]

    @property
    def Ddn(self) -> np.ndarray:
        return self._grid.at_node["connectivity_index__Ddn"]

    @property
    def W(self) -> np.ndarray:
        return self._grid.at_node["connectivity_index__W"]

    @property
    def S(self) -> np.ndarray:
        return self._grid.at_node["connectivity_index__S"]

    def as_2d(self, field: str = "connectivity_index__IC",
              geo_order: bool = True) -> np.ndarray:
        """
        Return a grid field as a 2-D array.

        Parameters
        ----------
        field : str
            Name of an ``at_node`` field.  Default: IC.
        geo_order : bool
            If *True* (default), flip rows so that row 0 is the north edge
            (GeoTIFF convention).  Landlab stores row 0 at the south.
        """
        arr = self._grid.at_node[field].reshape(self._grid.shape)
        return np.flipud(arr) if geo_order else arr

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _coerce_field(self, value, n, default_val, name):
        """Coerce value to numpy array; allow grid field names."""
        if value is None:
            if default_val is None:
                return None
            return np.full(n, default_val, dtype=np.float64)
        if isinstance(value, str):
            return self._grid.at_node[value].astype(np.float64)
        arr = np.asarray(value, dtype=np.float64).ravel()
        if len(arr) != n:
            raise ValueError(f"'{name}' length {len(arr)} != n_nodes {n}")
        return arr
