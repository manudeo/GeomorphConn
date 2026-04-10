"""
builder.py
==========
``WeightBuilder`` — composable W pipeline for GeomorphConn.

A :class:`WeightBuilder` stacks one or more weight components and combines
them into a single W array.  It is the primary interface for controlling the
IC weight function in :class:`~GeomorphConn.components.ConnectivityIndex`.

Quick examples
--------------
**Rainfall + NDVI (Dubey, Singh & Jain; submitted)**::

    from geomorphconn.weights import WeightBuilder, RainfallWeight, NDVIWeight
    wb = (
        WeightBuilder()
        .add(RainfallWeight(rainfall_arr))
        .add(NDVIWeight(ndvi_arr))
    )
    ic = ConnectivityIndex(grid, weight=wb)

**Geometric mean of three components**::

    wb = (
        WeightBuilder(combine="geometric_mean")
        .add(RainfallWeight(rf))
        .add(NDVIWeight(ndvi))
        .add(SurfaceRoughnessWeight(grid))
    )

**WorldCover C-factor only (Borselli 2008 spirit)**::

    from geomorphconn.weights import WeightBuilder, LandCoverWeight
    wb = WeightBuilder().add(LandCoverWeight.from_worldcover(lc_arr))

**Fully custom formula**::

    wb = WeightBuilder(
        combine=lambda arrays: np.max(np.stack(arrays), axis=0)
    )
    wb.add(RainfallWeight(rf)).add(NDVIWeight(ndvi))
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
_CombineSpec = Union[str, Callable[[List[np.ndarray]], np.ndarray]]

_BUILTIN_MODES = frozenset(
    {
        "mean",
        "arithmetic_mean",
        "geometric_mean",
        "product",
        "max",
        "min",
        "weighted_mean",
    }
)


# ---------------------------------------------------------------------------
# WeightBuilder
# ---------------------------------------------------------------------------


class WeightBuilder:
    """
    Composable pipeline that combines weight components into a single W array.

    Parameters
    ----------
    combine : str or callable, optional
        How to merge multiple components into a single W.  Built-in modes:

        ``'mean'`` / ``'arithmetic_mean'`` (default)
            Simple arithmetic mean (equal contribution from all components).
        ``'geometric_mean'``
            Geometric mean — penalises any single very-low component more
            strongly than arithmetic mean.
        ``'product'``
            Element-wise product.  Use when components act as multiplicative
            gates (each must be high for W to be high).
        ``'max'``
            Dominated by the highest-risk component at each cell.
        ``'min'``
            Dominated by the most restrictive component at each cell.
        ``'weighted_mean'``
            Weighted arithmetic mean; each component's ``component_weight``
            (set via :meth:`add`) is used.  Defaults to equal weights.

        Alternatively pass any callable with signature
        ``f(list[ndarray]) -> ndarray`` to implement a custom combination.

    w_min : float, optional
        Final lower clamp applied to the combined W.  Default ``0.005``.
    w_max : float, optional
        Final upper clamp applied to the combined W.  Default ``1.0``.
    """

    def __init__(
        self,
        combine: _CombineSpec = "mean",
        w_min: float = 0.005,
        w_max: float = 1.0,
    ) -> None:
        self._combine = combine
        self._w_min = float(w_min)
        self._w_max = float(w_max)
        # List of (component, component_weight) pairs
        self._components: List[Tuple[object, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, component, component_weight: float = 1.0) -> "WeightBuilder":
        """
        Register a weight component.

        Parameters
        ----------
        component
            Any object with a ``.compute() -> ndarray`` method.  See
            :mod:`~GeomorphConn.weights.components`.
        component_weight : float, optional
            Relative importance used only in ``'weighted_mean'`` mode.
            Default ``1.0``.

        Returns
        -------
        WeightBuilder
            *self*, enabling method chaining.
        """
        if not hasattr(component, "compute"):
            raise TypeError(
                f"component must have a .compute() method, got {type(component)}"
            )
        self._components.append((component, float(component_weight)))
        return self

    def build(self, n_nodes: Optional[int] = None) -> np.ndarray:
        """
        Compute and combine all registered components into a single W array.

        Parameters
        ----------
        n_nodes : int, optional
            Expected node count; validated against each component's output.

        Returns
        -------
        ndarray, shape (n_nodes,)
            Combined weight in ``[w_min, w_max]``.

        Raises
        ------
        ValueError
            If no components have been registered, or a component produces an
            array of the wrong length.
        """
        if not self._components:
            raise ValueError(
                "WeightBuilder has no components. "
                "Call .add(SomeWeightComponent(...)) at least once."
            )

        arrays: List[np.ndarray] = []
        cweights: List[float] = []
        for comp, cw in self._components:
            arr = comp.compute().astype(np.float64).ravel()
            if n_nodes is not None and len(arr) != n_nodes:
                name = getattr(comp, "name", type(comp).__name__)
                raise ValueError(
                    f"Component '{name}' returned array of length {len(arr)}, "
                    f"expected {n_nodes}."
                )
            arrays.append(arr)
            cweights.append(cw)

        combined = self._combine_arrays(arrays, cweights)
        return np.clip(combined, self._w_min, self._w_max)

    def describe(self) -> str:
        """Return a human-readable description of the pipeline."""
        mode = (
            self._combine
            if isinstance(self._combine, str)
            else getattr(self._combine, "__name__", repr(self._combine))
        )
        lines = [
            f"WeightBuilder  combine='{mode}'  "
            f"w_min={self._w_min}  w_max={self._w_max}"
        ]
        for comp, cw in self._components:
            name = getattr(comp, "name", type(comp).__name__)
            lines.append(f"  [{name}]  component_weight={cw}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _combine_arrays(
        self,
        arrays: List[np.ndarray],
        weights: List[float],
    ) -> np.ndarray:
        if callable(self._combine) and not isinstance(self._combine, str):
            return self._combine(arrays)

        if len(arrays) == 1:
            return arrays[0]

        mode = self._combine.lower().replace("-", "_")
        stack = np.stack(arrays, axis=0)  # (n_components, n_nodes)

        if mode in ("mean", "arithmetic_mean"):
            return stack.mean(axis=0)

        if mode == "geometric_mean":
            log_mean = np.log(np.clip(stack, 1e-10, None)).mean(axis=0)
            return np.exp(log_mean)

        if mode == "product":
            result = np.ones(stack.shape[1], dtype=np.float64)
            for arr in arrays:
                result *= arr
            return result

        if mode == "max":
            return stack.max(axis=0)

        if mode == "min":
            return stack.min(axis=0)

        if mode == "weighted_mean":
            w = np.array(weights, dtype=np.float64)
            w_sum = w.sum()
            if w_sum < 1e-12:
                raise ValueError("component_weight values sum to zero.")
            w /= w_sum
            return (stack * w[:, np.newaxis]).sum(axis=0)

        raise ValueError(
            f"Unknown combine mode '{self._combine}'. "
            f"Choose from {sorted(_BUILTIN_MODES)} or pass a callable."
        )
