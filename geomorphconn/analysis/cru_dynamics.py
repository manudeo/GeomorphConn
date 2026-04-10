"""
Connectivity Response Unit (CRU) classification for spatio-temporal connectivity analysis.

This module implements dynamic CRU classification using cumulative trend analysis
and temporal windowing to identify emerging, intensifying, diminishing, persistent,
sporadic, and new connectivity hotspots (and their symmetric coldspot equivalents).

The CRU concept, developed by Singh et al. (2017), defines landscape units that
exhibit similar connectivity response patterns. Dynamic CRU classification extends
this to the spatio-temporal domain, tracking how these zones evolve through time.

References:
    Singh et al. (2017). Assessment of connectivity in a water-stressed wetland.
        Earth Surf. Process. Landf. 42(11): 1982-1996.
    Singh et al. (2018). Evaluating dynamic hydrological connectivity of a
        floodplain wetland. Sci. Total Environ. 651: 2473-2488.
"""

from typing import Optional, Dict
import xarray as xr


# CRU Classification enum-like constants (human-readable labels)
CRU_CLASSES = {
    -6: "New Coldspot",
    -5: "Sporadic Coldspot",
    -4: "Persistent Coldspot",
    -3: "Diminishing Coldspot",
    -2: "Intensifying Coldspot",
    -1: "Emerging Coldspot",
    0: "No Pattern",
    1: "Emerging Hotspot",
    2: "Intensifying Hotspot",
    3: "Diminishing Hotspot",
    4: "Persistent Hotspot",
    5: "Sporadic Hotspot",
    6: "New Hotspot",
}


def classify_dynamic_crus(
    connectivity_timeseries: xr.DataArray,
    recent_window: int = 2,
    early_window: Optional[int] = None,
    emergence_threshold: float = 0.5,
    persistence_fraction: float = 0.8,
    attribution_tags: Optional[Dict[str, str]] = None,
) -> xr.DataArray:
    """
    Classify spatio-temporal connectivity patterns into dynamic CRU categories.

    This function implements the dynamic Connectivity Response Unit (CRU)
    classification methodology from Singh et al. (2017-2019). It combines
    cumulative trend analysis with temporal windowing to classify each pixel
    into one of 13 CRU categories representing distinct patterns of connectivity
    change over time.

    **Methodology:**
    1. Compute cumulative sum across time dimension to capture overall trend
    2. Calculate mean connectivity in "recent" (final N timesteps) window
    3. Calculate mean connectivity in "early" (initial N timesteps) window
    4. Apply nested decision logic to classify based on trends vs. thresholds

    **Output Classification (codes −6 to +6):**
    
    Hotspot (positive) classes:
    - 1: Emerging Hotspot — recently became connected (recent > 0.5, early ≤ 0.5)
    - 2: Intensifying Hotspot — connection strength increasing significantly
    - 3: Diminishing Hotspot — connection strength decreasing significantly
    - 4: Persistent Hotspot — connected >80% of timesteps, stable trend
    - 5: Sporadic Hotspot — intermittent connection pattern
    - 6: New Hotspot — first-ever appearance in final timestep
    
    Coldspot (negative) classes:
    - −1 to −6: Symmetric equivalents for low-connectivity zones
    
    Neutral:
    - 0: No significant pattern detected

    Parameters
    ----------
    connectivity_timeseries : xr.DataArray
        Time series of connectivity metrics, shape (time, y, x).
        Values typically range [−1, 0, 1] from hotspot detection
        (−1 = coldspot, 0 = neutral, 1 = hotspot), or can be continuous
        connectivity index values (IC).
        Expected dims: ('time', 'y', 'x') or similar spatial names.
        Required coordinate: time dimension name (inferred from data).

    recent_window : int, optional
        Number of final timesteps to average for "recent trend" signal.
        Default: 2 (recommend 2–3 for 10–20 timestep series).

    early_window : int or None, optional
        Number of initial timesteps to average for "early trend" signal.
        If None (default), set equal to recent_window.

    emergence_threshold : float, optional
        Threshold (on [−1, 1] scale) for classifying "emerging" zones.
        Default: 0.5. Interpret as: |recent_avg| > threshold signals emergence.

    persistence_fraction : float, optional
        Fraction of timesteps (0–1) required for "persistent" classification.
        Default: 0.8 (80% of time must show connectivity pattern).

    attribution_tags : dict or None, optional
        Metadata tags to attach to output DataArray (e.g., DOI, methodology version).
        Default: None (metadata auto-populated from function parameters).

    Returns
    -------
    xr.DataArray
        Classified CRU map, shape (y, x), dtype=int8.
        Values: [−6, +6] corresponding to CRU_CLASSES dict.
        
        Attributes:
        - 'recent_window': recent_window parameter used
        - 'early_window': early_window parameter used
        - 'emergence_threshold': threshold used
        - 'persistence_fraction': persistence threshold used
        - 'total_timesteps': number of timesteps in input
        - 'cru_classes': dict mapping codes to class names
        - Any user-provided attribution_tags

    Raises
    ------
    ValueError
        If connectivity_timeseries lacks time dimension or has <2 timesteps.
    TypeError
        If input not an xarray.DataArray.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from geomorphconn.analysis import classify_dynamic_crus
    
    # Synthetic 10-timestep hotspot series (time, y, x)
    >>> time = np.arange(10)
    >>> y = np.arange(50)
    >>> x = np.arange(50)
    >>> hotspots = xr.DataArray(
    ...     np.random.choice([-1, 0, 1], size=(10, 50, 50)),
    ...     coords={'time': time, 'y': y, 'x': x},
    ...     dims=['time', 'y', 'x'],
    ...     name='hotspots'
    ... )
    
    # Classify into dynamic CRUs
    >>> crus = classify_dynamic_crus(
    ...     hotspots,
    ...     recent_window=2,
    ...     emergence_threshold=0.5
    ... )
    >>> crus.shape
    (50, 50)
    >>> crus.attrs['total_timesteps']
    10
    
    # Visualize
    >>> import matplotlib.pyplot as plt
    >>> crus.plot(cmap='RdYlBu', vmin=-6, vmax=6)
    >>> plt.colorbar(label='CRU Class')
    """
    # Input validation
    if not isinstance(connectivity_timeseries, xr.DataArray):
        raise TypeError(f"Expected xarray.DataArray, got {type(connectivity_timeseries)}")
    
    # Infer time dimension (common names: 'time', 'temporal', etc.)
    time_dim = None
    for dim in connectivity_timeseries.dims:
        if dim.lower() in ('time', 'temporal', 't'):
            time_dim = dim
            break
    if time_dim is None:
        raise ValueError(
            f"Could not identify time dimension in {connectivity_timeseries.dims}. "
            "Expected one of: 'time', 'temporal', 't'."
        )
    
    n_timesteps = connectivity_timeseries.sizes[time_dim]
    if n_timesteps < 2:
        raise ValueError(
            f"Input must have >=2 timesteps for trend analysis. Got {n_timesteps}."
        )
    
    # Set early_window default
    if early_window is None:
        early_window = recent_window
    
    # Validate window sizes
    if recent_window > n_timesteps or early_window > n_timesteps:
        raise ValueError(
            f"Window sizes ({recent_window}, {early_window}) exceed "
            f"number of timesteps ({n_timesteps})."
        )
    
    # Compute trend signals
    trend_signal = _compute_cumsum_trend(connectivity_timeseries, time_dim)
    recent_signal = _compute_recent_mean(connectivity_timeseries, time_dim, recent_window)
    early_signal = _compute_early_mean(connectivity_timeseries, time_dim, early_window)
    hot_count = (connectivity_timeseries > 0).sum(dim=time_dim)
    cold_count = (connectivity_timeseries < 0).sum(dim=time_dim)
    
    # Apply classification decision tree
    classified = _apply_cru_classification(
        trend=trend_signal,
        recent=recent_signal,
        early=early_signal,
        hot_count=hot_count,
        cold_count=cold_count,
        total_steps=n_timesteps,
        emergence_threshold=emergence_threshold,
        persistence_fraction=persistence_fraction,
    )
    
    # Attach metadata
    classified.attrs.update({
        'recent_window': recent_window,
        'early_window': early_window,
        'emergence_threshold': emergence_threshold,
        'persistence_fraction': persistence_fraction,
        'total_timesteps': n_timesteps,
        'cru_classes': CRU_CLASSES,
    })
    
    if attribution_tags:
        classified.attrs.update(attribution_tags)
    
    return classified.astype('int8')


def _compute_cumsum_trend(da: xr.DataArray, time_dim: str) -> xr.DataArray:
    """Net trend signal over time dimension (2D, sign-preserving)."""
    return da.sum(dim=time_dim)


def _compute_recent_mean(
    da: xr.DataArray, time_dim: str, window: int
) -> xr.DataArray:
    """Mean of final N timesteps."""
    return da.isel({time_dim: slice(-window, None)}).mean(dim=time_dim)


def _compute_early_mean(
    da: xr.DataArray, time_dim: str, window: int
) -> xr.DataArray:
    """Mean of first N timesteps."""
    return da.isel({time_dim: slice(0, window)}).mean(dim=time_dim)


def _apply_cru_classification(
    trend: xr.DataArray,
    recent: xr.DataArray,
    early: xr.DataArray,
    hot_count: xr.DataArray,
    cold_count: xr.DataArray,
    total_steps: int,
    emergence_threshold: float,
    persistence_fraction: float,
) -> xr.DataArray:
    """
    Vectorized nested xr.where decision tree for CRU classification.
    
    Decision logic (evaluated in priority order):
    1. Check for Emerging: recent > thresh AND early ≤ thresh → 1 or −1
    2. Check for Intensifying: recent > early + thresh → 2 or −2
    3. Check for Diminishing: recent < early − thresh → 3 or −3
    4. Check for Persistent: overall_count > persistence_frac * steps → 4 or −4
    5. Check for Sporadic: recent > small_thresh → 5 or −5
    6. Check for New: never seen before, only now → 6 or −6
    7. Default: 0 (no pattern)
    """
    
    # Persistence threshold (count-based)
    persistence_count_threshold = persistence_fraction * total_steps
    
    # Initialize with 0 (no pattern)
    result = xr.zeros_like(trend, dtype='int8')
    
    # Helper: hot vs cold based on sign of net trend
    is_hot = trend > 0
    is_cold = trend < 0

    # Persistent first so long-duration clusters are not mislabeled as emerging.
    persistent_hot = xr.where(
        (hot_count >= persistence_count_threshold) & (result == 0),
        4, 0
    )
    persistent_cold = xr.where(
        (cold_count >= persistence_count_threshold) & (result == 0),
        -4, 0
    )
    result = xr.where(persistent_hot > 0, persistent_hot, result)
    result = xr.where(persistent_cold < 0, persistent_cold, result)

    # New (strict): single occurrence and only in the recent window.
    new_hot = xr.where(
        (recent == 1) & (early <= 0) & (hot_count == 1) & (result == 0),
        6, 0
    )
    new_cold = xr.where(
        (recent == -1) & (early >= 0) & (cold_count == 1) & (result == 0),
        -6, 0
    )
    result = xr.where(new_hot > 0, new_hot, result)
    result = xr.where(new_cold < 0, new_cold, result)
    
    # Emerging: recent high, early low
    emerging_hot = xr.where(
        (recent >= emergence_threshold) & (early <= emergence_threshold),
        1, 0
    )
    emerging_cold = xr.where(
        (recent <= -emergence_threshold) & (early >= -emergence_threshold),
        -1, 0
    )
    result = xr.where((emerging_hot > 0) & (result == 0), emerging_hot, result)
    result = xr.where((emerging_cold < 0) & (result == 0), emerging_cold, result)
    
    # Intensifying: recent significantly > early
    intensifying_hot = xr.where(
        (is_hot) & (recent > early + emergence_threshold) & (result == 0),
        2, 0
    )
    intensifying_cold = xr.where(
        (is_cold) & (recent < early - emergence_threshold) & (result == 0),
        -2, 0
    )
    result = xr.where((intensifying_hot > 0) & (result == 0), intensifying_hot, result)
    result = xr.where((intensifying_cold < 0) & (result == 0), intensifying_cold, result)
    
    # Diminishing: recent significantly < early
    diminishing_hot = xr.where(
        (is_hot) & (recent < early - emergence_threshold) & (result == 0),
        3, 0
    )
    diminishing_cold = xr.where(
        (is_cold) & (recent > early + emergence_threshold) & (result == 0),
        -3, 0
    )
    result = xr.where((diminishing_hot > 0) & (result == 0), diminishing_hot, result)
    result = xr.where((diminishing_cold < 0) & (result == 0), diminishing_cold, result)
    
    # Sporadic: intermittent activity
    sporadic_hot = xr.where(
        (is_hot) & (recent > 0.2) & (result == 0),
        5, 0
    )
    sporadic_cold = xr.where(
        (is_cold) & (recent < -0.2) & (result == 0),
        -5, 0
    )
    result = xr.where((sporadic_hot > 0) & (result == 0), sporadic_hot, result)
    result = xr.where((sporadic_cold < 0) & (result == 0), sporadic_cold, result)
    
    # Default unclassified to 0 (no pattern)
    result = result.where(result != 0, 0)
    
    return result
