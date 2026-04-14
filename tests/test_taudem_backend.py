from __future__ import annotations

import numpy as np

from geomorphconn.backends.taudem import _windows_path_to_wsl
from geomorphconn.backends.taudem import check_taudem_installation
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