from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geomorphconn.cli import main
import geomorphconn.cli as cli


def _write_tif(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, float(arr.shape[0]), 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


def test_cli_run_writes_outputs(tmp_path):
    dem = np.array(
        [
            [10, 11, 12, 13, 14],
            [9, 10, 11, 12, 13],
            [8, 9, 10, 11, 12],
            [7, 8, 9, 10, 11],
            [6, 7, 8, 9, 10],
        ],
        dtype=np.float64,
    )
    ndvi = np.full_like(dem, 0.4)
    rainfall = np.full_like(dem, 800.0)

    dem_p = tmp_path / "dem.tif"
    ndvi_p = tmp_path / "ndvi.tif"
    rf_p = tmp_path / "rain.tif"
    _write_tif(dem_p, dem)
    _write_tif(ndvi_p, ndvi)
    _write_tif(rf_p, rainfall)

    out_dir = tmp_path / "out"
    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--ndvi",
            str(ndvi_p),
            "--rainfall",
            str(rf_p),
            "--outputs",
            "IC",
            "Dup",
            "--out-dir",
            str(out_dir),
            "--prefix",
            "test_",
        ]
    )

    assert rc == 0
    assert (out_dir / "test_IC.tif").exists()
    assert (out_dir / "test_Dup.tif").exists()


def test_cli_shape_mismatch_autoreproject_default_succeeds(tmp_path):
    dem = np.ones((5, 5), dtype=np.float64)
    ndvi = np.ones((4, 5), dtype=np.float64)
    rainfall = np.ones((5, 5), dtype=np.float64)

    dem_p = tmp_path / "dem.tif"
    ndvi_p = tmp_path / "ndvi.tif"
    rf_p = tmp_path / "rain.tif"
    _write_tif(dem_p, dem)
    _write_tif(ndvi_p, ndvi)
    _write_tif(rf_p, rainfall)

    out_dir = tmp_path / "out_default_autoreproject"
    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--ndvi",
            str(ndvi_p),
            "--rainfall",
            str(rf_p),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "ic_IC.tif").exists()


def test_cli_shape_mismatch_auto_reproject_succeeds(tmp_path):
    dem = np.ones((5, 5), dtype=np.float64) * 100.0
    ndvi = np.ones((3, 3), dtype=np.float64) * 0.4
    rainfall = np.ones((4, 4), dtype=np.float64) * 700.0

    dem_p = tmp_path / "dem.tif"
    ndvi_p = tmp_path / "ndvi.tif"
    rf_p = tmp_path / "rain.tif"
    _write_tif(dem_p, dem)
    _write_tif(ndvi_p, ndvi)
    _write_tif(rf_p, rainfall)

    out_dir = tmp_path / "out_reproject"
    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--ndvi",
            str(ndvi_p),
            "--rainfall",
            str(rf_p),
            "--reference-grid",
            "dem",
            "--outputs",
            "IC",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert (out_dir / "ic_IC.tif").exists()


def test_cli_shape_mismatch_no_reproject_fails(tmp_path):
    dem = np.ones((5, 5), dtype=np.float64)
    ndvi = np.ones((3, 3), dtype=np.float64)
    rainfall = np.ones((5, 5), dtype=np.float64)

    dem_p = tmp_path / "dem.tif"
    ndvi_p = tmp_path / "ndvi.tif"
    rf_p = tmp_path / "rain.tif"
    _write_tif(dem_p, dem)
    _write_tif(ndvi_p, ndvi)
    _write_tif(rf_p, rainfall)

    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--ndvi",
            str(ndvi_p),
            "--rainfall",
            str(rf_p),
            "--no-auto-reproject",
        ]
    )
    assert rc == 2


def test_cli_main_basin_mask_shape_mismatch_no_reproject_fails(tmp_path):
    dem = np.ones((5, 5), dtype=np.float64)
    ndvi = np.ones((5, 5), dtype=np.float64)
    rainfall = np.ones((5, 5), dtype=np.float64)
    mask = np.ones((4, 5), dtype=np.float64)

    dem_p = tmp_path / "dem.tif"
    ndvi_p = tmp_path / "ndvi.tif"
    rf_p = tmp_path / "rain.tif"
    mask_p = tmp_path / "mask.tif"
    _write_tif(dem_p, dem)
    _write_tif(ndvi_p, ndvi)
    _write_tif(rf_p, rainfall)
    _write_tif(mask_p, mask)

    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--ndvi",
            str(ndvi_p),
            "--rainfall",
            str(rf_p),
            "--main-basin-mask",
            str(mask_p),
            "--no-auto-reproject",
        ]
    )
    assert rc == 2


def test_cli_main_basin_mask_runs_and_writes_masked_output(tmp_path):
    dem = np.array(
        [
            [10, 11, 12, 13, 14],
            [9, 10, 11, 12, 13],
            [8, 9, 10, 11, 12],
            [7, 8, 9, 10, 11],
            [6, 7, 8, 9, 10],
        ],
        dtype=np.float64,
    )
    ndvi = np.full_like(dem, 0.4)
    rainfall = np.full_like(dem, 800.0)
    mask = np.zeros_like(dem)
    mask[:, :3] = 1.0

    dem_p = tmp_path / "dem.tif"
    ndvi_p = tmp_path / "ndvi.tif"
    rf_p = tmp_path / "rain.tif"
    mask_p = tmp_path / "mask.tif"
    _write_tif(dem_p, dem)
    _write_tif(ndvi_p, ndvi)
    _write_tif(rf_p, rainfall)
    _write_tif(mask_p, mask)

    out_dir = tmp_path / "out_mask"
    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--ndvi",
            str(ndvi_p),
            "--rainfall",
            str(rf_p),
            "--main-basin-mask",
            str(mask_p),
            "--outputs",
            "IC",
            "--out-dir",
            str(out_dir),
            "--prefix",
            "masked_",
        ]
    )

    assert rc == 0
    ic_path = out_dir / "masked_IC.tif"
    assert ic_path.exists()
    with rasterio.open(ic_path) as src:
        data = src.read(1)
        nodata = src.nodata
    assert np.any(data == nodata)


def test_cli_welcome_subcommand_returns_zero():
    rc = main(["welcome"])
    assert rc == 0


def test_cli_version_option_exits_cleanly():
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_run_no_welcome_flag(tmp_path):
    dem = np.ones((5, 5), dtype=np.float64) * 100.0
    ndvi = np.full((5, 5), 0.4, dtype=np.float64)
    rainfall = np.full((5, 5), 800.0, dtype=np.float64)

    dem_p = tmp_path / "dem.tif"
    ndvi_p = tmp_path / "ndvi.tif"
    rf_p = tmp_path / "rain.tif"
    _write_tif(dem_p, dem)
    _write_tif(ndvi_p, ndvi)
    _write_tif(rf_p, rainfall)

    out_dir = tmp_path / "out_no_welcome"
    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--ndvi",
            str(ndvi_p),
            "--rainfall",
            str(rf_p),
            "--no-show-welcome",
            "--outputs",
            "IC",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0


def test_cli_roughness_only_needs_dem_only(tmp_path):
    dem = np.array(
        [
            [100, 101, 102, 103, 104],
            [99, 100, 101, 102, 103],
            [98, 99, 100, 101, 102],
            [97, 98, 99, 100, 101],
            [96, 97, 98, 99, 100],
        ],
        dtype=np.float64,
    )
    dem_p = tmp_path / "dem.tif"
    _write_tif(dem_p, dem)

    out_dir = tmp_path / "out_roughness"
    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--weight-factors",
            "roughness",
            "--outputs",
            "IC",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "ic_IC.tif").exists()


def test_cli_roughness_even_detrend_window_fails(tmp_path):
    dem = np.ones((5, 5), dtype=np.float64)
    dem_p = tmp_path / "dem.tif"
    _write_tif(dem_p, dem)

    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--weight-factors",
            "roughness",
            "--roughness-detrend-window",
            "4",
        ]
    )
    assert rc == 2


def test_cli_roughness_even_std_window_fails(tmp_path):
    dem = np.ones((5, 5), dtype=np.float64)
    dem_p = tmp_path / "dem.tif"
    _write_tif(dem_p, dem)

    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--weight-factors",
            "roughness",
            "--roughness-std-window",
            "6",
        ]
    )
    assert rc == 2


def test_cli_missing_selected_factor_input_fails(tmp_path):
    dem = np.ones((5, 5), dtype=np.float64)
    dem_p = tmp_path / "dem.tif"
    _write_tif(dem_p, dem)

    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--weight-factors",
            "ndvi",
        ]
    )
    assert rc == 2


def test_cli_user_supplied_weight_raster_mode(tmp_path):
    dem = np.array(
        [
            [10, 11, 12, 13, 14],
            [9, 10, 11, 12, 13],
            [8, 9, 10, 11, 12],
            [7, 8, 9, 10, 11],
            [6, 7, 8, 9, 10],
        ],
        dtype=np.float64,
    )
    weight = np.full_like(dem, 0.35)

    dem_p = tmp_path / "dem.tif"
    w_p = tmp_path / "weight.tif"
    _write_tif(dem_p, dem)
    _write_tif(w_p, weight)

    out_dir = tmp_path / "out_weight_mode"
    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--weight-raster",
            str(w_p),
            "--outputs",
            "IC",
            "W",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "ic_IC.tif").exists()
    assert (out_dir / "ic_W.tif").exists()


def test_cli_user_weight_no_reproject_shape_mismatch_fails(tmp_path):
    dem = np.ones((5, 5), dtype=np.float64)
    weight = np.ones((4, 4), dtype=np.float64) * 0.4

    dem_p = tmp_path / "dem.tif"
    w_p = tmp_path / "weight.tif"
    _write_tif(dem_p, dem)
    _write_tif(w_p, weight)

    rc = main(
        [
            "run",
            "--dem",
            str(dem_p),
            "--weight-raster",
            str(w_p),
            "--no-auto-reproject",
        ]
    )
    assert rc == 2


def test_cli_taudem_check_ok(monkeypatch, capsys):
    def _fake_report(_bin_dir):
        return {
            "platform": "Windows-11",
            "is_wsl": False,
            "requested_bin_dir": None,
            "search_dirs": [r"C:\\Program Files\\TauDEM\\TauDEM5Exe"],
            "executables": {
                "mpiexec": "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe",
                "PitRemove": "C:/Program Files/TauDEM/TauDEM5Exe/PitRemove.exe",
                "D8FlowDir": "C:/Program Files/TauDEM/TauDEM5Exe/D8FlowDir.exe",
                "DinfFlowDir": "C:/Program Files/TauDEM/TauDEM5Exe/DinfFlowDir.exe",
                "AreaD8": "C:/Program Files/TauDEM/TauDEM5Exe/AreaD8.exe",
                "AreaDinf": "C:/Program Files/TauDEM/TauDEM5Exe/AreaDinf.exe",
            },
            "missing": [],
            "ok": True,
        }

    monkeypatch.setattr("geomorphconn.cli.check_taudem_installation", _fake_report)
    rc = main(["taudem-check"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "TauDEM self-check: OK" in out


def test_cli_taudem_check_failure_shows_wsl_note(monkeypatch, capsys):
    def _fake_report(_bin_dir):
        return {
            "platform": "Linux-WSL2",
            "is_wsl": True,
            "requested_bin_dir": None,
            "search_dirs": ["/mnt/c/Program Files/TauDEM/TauDEM5Exe"],
            "executables": {
                "mpiexec": None,
                "PitRemove": None,
                "D8FlowDir": None,
                "DinfFlowDir": None,
                "AreaD8": None,
                "AreaDinf": None,
            },
            "missing": ["mpiexec", "PitRemove"],
            "ok": False,
        }

    monkeypatch.setattr("geomorphconn.cli.check_taudem_installation", _fake_report)
    rc = main(["taudem-check"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "TauDEM self-check: FAILED" in out
    assert "WSL note" in out


def test_cli_gui_streamlit_missing_returns_2(monkeypatch):
    def _raise_file_not_found(_cmd):
        raise FileNotFoundError("streamlit missing")

    monkeypatch.setattr("geomorphconn.cli.subprocess.call", _raise_file_not_found)
    rc = main(["gui", "--backend", "streamlit", "--no-show-welcome"])
    assert rc == 2


def test_make_connectivity_index_suppresses_duplicate_depression_warning(monkeypatch):
    class _DummyCI:
        def __init__(self, *args, **kwargs):
            import warnings

            warnings.warn(
                "Using DepressionFinderAndRouter: typically better depression handling and routing quality, but runtime may increase (especially on high-resolution DEMs).",
                UserWarning,
                stacklevel=2,
            )

    monkeypatch.setattr(cli, "ConnectivityIndex", _DummyCI)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        cli._make_connectivity_index(object(), depression_finder="DepressionFinderAndRouter")

    assert len(recorded) == 0
