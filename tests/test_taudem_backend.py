from __future__ import annotations

import numpy as np

from geomorphconn.backends.taudem import _windows_path_to_wsl
from geomorphconn.backends.taudem import check_taudem_installation
from geomorphconn.backends.taudem import _dominant_outlet_mask
from geomorphconn.weights.components import compute_surface_roughness_weight_2d


def test_windows_path_to_wsl_converts_drive_prefix():
    out = _windows_path_to_wsl(r"C:\Program Files\TauDEM\TauDEM5Exe")
    assert out == "/mnt/c/Program Files/TauDEM/TauDEM5Exe"


def test_windows_path_to_wsl_leaves_non_windows_paths_unchanged():
    out = _windows_path_to_wsl("/mnt/c/Program Files/TauDEM/TauDEM5Exe")
    assert out == "/mnt/c/Program Files/TauDEM/TauDEM5Exe"


def test_check_taudem_installation_report_shape():
    report = check_taudem_installation()
    assert "executables" in report
    assert "ok" in report
    assert "missing" in report
    assert "PitRemove" in report["executables"]


def test_compute_surface_roughness_weight_2d_shape_and_range():
    dem = np.array(
        [
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 11.0, 12.0, 10.0],
            [10.0, 12.0, 13.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float64,
    )
    w = compute_surface_roughness_weight_2d(dem, detrend_window=3, std_window=3, w_min=0.005)
    assert w.shape == dem.shape
    assert np.nanmin(w) >= 0.005
    assert np.nanmax(w) <= 1.0


def test_compute_surface_roughness_weight_2d_flat_dem_returns_ones():
    dem = np.full((5, 5), 100.0, dtype=np.float64)
    w = compute_surface_roughness_weight_2d(dem, detrend_window=3, std_window=3, w_min=0.005)
    assert np.allclose(w, 1.0)


def test_dominant_outlet_mask_ignores_invalid_outlets():
    # Graph with two outlets: node 3 (valid) and node 4 (invalid).
    # 0 -> 1 -> 2 -> 3(outlet), and 4(outlet invalid)
    recv = np.array([1, 2, 3, 3, 4], dtype=np.int64)
    valid = np.array([True, True, True, True, False], dtype=bool)
    metric = np.array([1.0, 2.0, 3.0, 10.0, 1e9], dtype=np.float64)

    mask = _dominant_outlet_mask(recv, metric, valid)

    assert mask is not None
    assert mask.dtype == bool
    assert np.array_equal(mask, np.array([True, True, True, True, False], dtype=bool))


def test_dominant_outlet_mask_respects_outlet_metric_among_valid_outlets():
    # Two valid outlet basins: node 3 has higher metric than node 4.
    # Basin A: 0 -> 1 -> 3(outlet)
    # Basin B: 2 -> 4(outlet)
    recv = np.array([1, 3, 4, 3, 4], dtype=np.int64)
    valid = np.array([True, True, True, True, True], dtype=bool)
    metric = np.array([0.0, 0.0, 0.0, 100.0, 10.0], dtype=np.float64)

    mask = _dominant_outlet_mask(recv, metric, valid)

    assert mask is not None
    # Dominant outlet should be node 3, so only nodes draining to 3 remain.
    assert np.array_equal(mask, np.array([True, True, False, True, False], dtype=bool))


def test_dominant_outlet_mask_falls_back_to_largest_basin_when_metric_missing():
    # Two valid outlet basins with all-NaN outlet metrics.
    # Basin A (larger): 0->1->3(outlet), 5->3
    # Basin B (smaller): 2->4(outlet)
    recv = np.array([1, 3, 4, 3, 4, 3], dtype=np.int64)
    valid = np.array([True, True, True, True, True, True], dtype=bool)
    metric = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

    mask = _dominant_outlet_mask(recv, metric, valid)

    assert mask is not None
    # Expect larger basin draining to outlet 3.
    assert np.array_equal(mask, np.array([True, True, False, True, False, True], dtype=bool))