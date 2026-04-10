from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geomorphconn.cli import main


def _write_tif(path: Path, arr: np.ndarray, pixel_size: float = 1.0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, float(arr.shape[0]), pixel_size, pixel_size),
        nodata=-9999.0,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


def run_smoke() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

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
        ndvi = np.full((3, 3), 0.4, dtype=np.float64)
        rainfall = np.full((4, 4), 700.0, dtype=np.float64)

        dem_p = root / "dem.tif"
        ndvi_p = root / "ndvi.tif"
        rain_p = root / "rainfall.tif"
        out_dir = root / "out"

        _write_tif(dem_p, dem, pixel_size=1.0)
        _write_tif(ndvi_p, ndvi, pixel_size=2.0)
        _write_tif(rain_p, rainfall, pixel_size=1.5)

        rc = main(
            [
                "run",
                "--dem",
                str(dem_p),
                "--ndvi",
                str(ndvi_p),
                "--rainfall",
                str(rain_p),
                "--auto-reproject",
                "--reference-grid",
                "dem",
                "--outputs",
                "IC",
                "Dup",
                "--out-dir",
                str(out_dir),
                "--prefix",
                "smoke_",
            ]
        )
        if rc != 0:
            return rc

        ic_out = out_dir / "smoke_IC.tif"
        dup_out = out_dir / "smoke_Dup.tif"
        if not ic_out.exists() or not dup_out.exists():
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke())
