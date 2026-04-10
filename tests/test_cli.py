from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geomorphconn.cli import main


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
