"""Unit tests for geomorphconn.analysis.disconnectivity."""

import numpy as np
import pytest
import xarray as xr

from geomorphconn.analysis import (
    build_disconnectivity_hierarchy,
    compute_node_comparison_metrics,
)


def _make_map(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values.astype(float),
        coords={"y": np.arange(values.shape[0]), "x": np.arange(values.shape[1])},
        dims=["y", "x"],
    )


def test_hierarchy_builds_nodes_and_links():
    data = np.array(
        [
            [0.1, 0.2, 0.9, 0.95],
            [0.1, 0.3, 0.85, 0.9],
            [0.2, 0.2, 0.1, 0.1],
            [0.8, 0.82, 0.1, 0.1],
        ]
    )
    ic = _make_map(data)

    hierarchy = build_disconnectivity_hierarchy(ic, quantiles=[0.5, 0.75, 0.9])

    assert len(hierarchy["thresholds"]) >= 1
    assert len(hierarchy["nodes"]) >= 1
    assert hierarchy["shape"] == data.shape

    # If multiple levels exist, links should exist for nested components.
    if len(hierarchy["thresholds"]) > 1:
        assert isinstance(hierarchy["links"], list)


def test_metrics_zero_for_identical_reference_and_comparison():
    data = np.array(
        [
            [0.1, 0.2, 0.9, 0.95],
            [0.1, 0.3, 0.85, 0.9],
            [0.2, 0.2, 0.1, 0.1],
            [0.8, 0.82, 0.1, 0.1],
        ]
    )
    ref = _make_map(data)
    cmp_same = _make_map(data.copy())

    hierarchy = build_disconnectivity_hierarchy(ref, quantiles=[0.6, 0.8])
    rows = compute_node_comparison_metrics(hierarchy, ref, {"same": cmp_same})

    assert len(rows) > 0
    for row in rows:
        assert np.isclose(row["rmse_vs_same"], 0.0)
        assert np.isclose(row["bias_vs_same"], 0.0)
        assert np.isclose(row["same_sign_vs_same"], 1.0)


def test_invalid_shape_raises():
    ref = _make_map(np.ones((4, 4)))
    bad = xr.DataArray(np.ones((4, 4, 2)), dims=["y", "x", "z"])

    raised = False
    try:
        build_disconnectivity_hierarchy(bad)
    except ValueError:
        raised = True
    assert raised


def test_hierarchy_validation_and_empty_cases():
    with pytest.raises(TypeError):
        build_disconnectivity_hierarchy(np.zeros((3, 3)))

    ic = _make_map(np.full((3, 3), np.nan))
    hierarchy = build_disconnectivity_hierarchy(ic, quantiles=[0.5])
    assert hierarchy["nodes"] == []
    assert hierarchy["links"] == []
    assert hierarchy["thresholds"] == []


def test_quantile_validation_errors():
    ic = _make_map(np.ones((3, 3)))
    with pytest.raises(ValueError, match="at least one"):
        build_disconnectivity_hierarchy(ic, quantiles=[])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_disconnectivity_hierarchy(ic, quantiles=[-0.1, 0.5])


def test_metrics_shape_mismatch_raises():
    ref = _make_map(np.ones((4, 4)))
    hierarchy = build_disconnectivity_hierarchy(ref, quantiles=[0.5])

    bad_ref = xr.DataArray(np.ones((3, 3)), dims=["y", "x"])
    bad = xr.DataArray(np.ones((4, 4, 2)), dims=["y", "x", "z"])
    with pytest.raises(ValueError, match="does not match"):
        compute_node_comparison_metrics({**hierarchy, "shape": (5, 5)}, ref, {})

    with pytest.raises(ValueError, match="expected"):
        compute_node_comparison_metrics(hierarchy, ref, {"bad": bad_ref})

    hierarchy = build_disconnectivity_hierarchy(ref, quantiles=[0.5])

    raised = False
    try:
        compute_node_comparison_metrics(hierarchy, ref, {"bad": bad})
    except ValueError:
        raised = True
    assert raised
