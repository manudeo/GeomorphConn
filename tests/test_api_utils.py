from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin

import geomorphconn.api as api
from geomorphconn.utils.preprocess import coarsen_rasters
from geomorphconn.utils.target import nodes_from_geodataframe, rasterize_targets


def _make_da(shape=(4, 4), value=100.0, crs="EPSG:32632", xres=30.0, yres=30.0):
    arr = np.full(shape, value, dtype=np.float64)
    da = xr.DataArray(arr, dims=("y", "x"))
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(from_origin(500000.0, 4600000.0, xres, yres))
    return da


class _DummyConnectivityIndex:
    def __init__(self, grid, **kwargs):
        self.grid = grid
        self.kwargs = kwargs
        n = grid.number_of_nodes
        outputs = {
            "connectivity_index__IC": 1.0,
            "connectivity_index__Dup": 2.0,
            "connectivity_index__Ddn": 3.0,
            "connectivity_index__W": 0.5,
            "connectivity_index__S": 0.5,
            "connectivity_index__Wmean": 0.5,
            "connectivity_index__Smean": 0.5,
            "connectivity_index__ACCfinal": 4.0,
        }
        for field, val in outputs.items():
            if field not in grid.at_node:
                grid.add_field(field, np.full(n, val, dtype=np.float64), at="node")

    def run_one_step(self):
        return None


def test_normalize_weight_spec_variants():
    pre, ndvi, rain = api._normalize_weight_spec(None, "n", "r")
    assert (pre, ndvi, rain) == (None, "n", "r")

    pre, ndvi, rain = api._normalize_weight_spec({"ndvi": "n2", "rainfall": "r2"}, "n", "r")
    assert (pre, ndvi, rain) == (None, "n2", "r2")

    pre, ndvi, rain = api._normalize_weight_spec(["n3"], None, "r")
    assert (pre, ndvi, rain) == (None, "n3", "r")

    pre, ndvi, rain = api._normalize_weight_spec(["n4", "r4"], None, None)
    assert (pre, ndvi, rain) == (None, "n4", "r4")

    with pytest.raises(ValueError, match="cannot be empty"):
        api._normalize_weight_spec([], None, None)

    with pytest.raises(ValueError, match="at most two items"):
        api._normalize_weight_spec([1, 2, 3], None, None)


def test_is_projected_handles_bad_crs_object():
    class _BadCRS:
        @property
        def is_projected(self):
            raise RuntimeError("boom")

    assert api._is_projected(None) is False
    assert api._is_projected(_BadCRS()) is False


def test_as_dataarray_type_validation():
    with pytest.raises(TypeError, match="must be an xarray.DataArray or raster path"):
        api._as_dataarray(123, "dem")


def test_run_connectivity_validation_errors(monkeypatch):
    monkeypatch.setattr(api, "ConnectivityIndex", _DummyConnectivityIndex)
    dem = _make_da()

    with pytest.raises(ValueError, match="ic_mode must be 'outlet' or 'target'"):
        api.run_connectivity_from_rasters(dem=dem, weight=_make_da(), ic_mode="invalid")

    with pytest.raises(ValueError, match="Set ic_mode='target'"):
        api.run_connectivity_from_rasters(dem=dem, weight=_make_da(), ic_mode="outlet", stream_threshold=10)

    with pytest.raises(ValueError, match="requires one of"):
        api.run_connectivity_from_rasters(dem=dem, weight=_make_da(), ic_mode="target")

    with pytest.raises(ValueError, match="either stream_threshold or target_vector"):
        api.run_connectivity_from_rasters(
            dem=dem,
            weight=_make_da(),
            ic_mode="target",
            stream_threshold=10,
            target_vector="dummy.shp",
        )


def test_run_connectivity_no_weight_input_raises(monkeypatch):
    monkeypatch.setattr(api, "ConnectivityIndex", _DummyConnectivityIndex)
    dem = _make_da()
    with pytest.raises(ValueError, match="No weight input was provided"):
        api.run_connectivity_from_rasters(dem=dem, ic_mode="outlet")


def test_run_connectivity_non_square_pixels_raises(monkeypatch):
    monkeypatch.setattr(api, "ConnectivityIndex", _DummyConnectivityIndex)
    dem = _make_da(xres=30.0, yres=60.0)
    with pytest.raises(ValueError, match="Non-square pixels"):
        api.run_connectivity_from_rasters(dem=dem, weight=_make_da())


def test_run_connectivity_target_nodes_validation(monkeypatch):
    monkeypatch.setattr(api, "ConnectivityIndex", _DummyConnectivityIndex)
    dem = _make_da(shape=(3, 3))

    with pytest.raises(ValueError, match="must not be empty"):
        api.run_connectivity_from_rasters(dem=dem, weight=_make_da(shape=(3, 3)), ic_mode="target", target_nodes=[])

    with pytest.raises(ValueError, match="outside valid node range"):
        api.run_connectivity_from_rasters(
            dem=dem,
            weight=_make_da(shape=(3, 3)),
            ic_mode="target",
            target_nodes=[-1, 999],
        )


def test_run_connectivity_happy_path_precomputed_weight(monkeypatch):
    monkeypatch.setattr(api, "ConnectivityIndex", _DummyConnectivityIndex)

    dem = _make_da(shape=(4, 4), value=200.0)
    w = _make_da(shape=(4, 4), value=0.6)
    out = api.run_connectivity_from_rasters(dem=dem, weight=w, ic_mode="outlet")

    assert set(out.keys()) == {"dataset", "inputs", "grid", "component", "profile"}
    assert "IC" in out["dataset"]
    assert "weight" in out["inputs"]
    assert out["dataset"].attrs["ic_mode"] == "outlet"
    assert out["profile"]["width"] == 4
    assert out["profile"]["height"] == 4


def test_run_connectivity_weight_shorthand_ndvi_rainfall(monkeypatch):
    monkeypatch.setattr(api, "ConnectivityIndex", _DummyConnectivityIndex)

    dem = _make_da(shape=(4, 4), value=200.0)
    ndvi = _make_da(shape=(4, 4), value=0.2)
    rain = _make_da(shape=(4, 4), value=900.0)

    out = api.run_connectivity_from_rasters(
        dem=dem,
        weight=[ndvi, rain],
        ic_mode="outlet",
        weight_combine="mean",
    )
    assert "ndvi" in out["inputs"]
    assert "rainfall" in out["inputs"]


def test_run_connectivity_target_vector_branch(monkeypatch):
    monkeypatch.setattr(api, "ConnectivityIndex", _DummyConnectivityIndex)

    import geomorphconn.utils as utils_mod

    def _fake_rasterize_targets(**kwargs):
        return np.array([0, 1, 2], dtype=np.int64)

    monkeypatch.setattr(utils_mod, "rasterize_targets", _fake_rasterize_targets)

    dem = _make_da(shape=(4, 4))
    out = api.run_connectivity_from_rasters(
        dem=dem,
        weight=_make_da(shape=(4, 4), value=0.7),
        ic_mode="target",
        target_vector=object(),
        target_all_touched=False,
        target_buffer_m=5.0,
    )
    assert out["dataset"].attrs["ic_mode"] == "target"
    assert out["dataset"].attrs["target_vector_used"] is True
    assert out["dataset"].attrs["target_nodes_count"] == 3


def test_run_connectivity_taudem_backend_dispatch(monkeypatch):
    dem = _make_da(shape=(3, 3), value=120.0)
    w = _make_da(shape=(3, 3), value=0.7)

    called = {}

    def _fake_taudem(**kwargs):
        called.update(kwargs)
        arr = np.ones((3, 3), dtype=np.float64)
        return {
            "layers": {
                "IC": arr,
                "Dup": arr,
                "Ddn": arr,
                "W": arr,
                "S": arr,
                "Wmean": arr,
                "Smean": arr,
                "ACCfinal": arr,
            }
        }

    monkeypatch.setattr(api, "run_connectivity_taudem_arrays", _fake_taudem)

    out = api.run_connectivity_from_rasters(
        dem=dem,
        weight=w,
        compute_backend="taudem",
        flow_director="D8",
        taudem_n_procs=4,
        taudem_bin_dir="C:/fake/taudem",
    )

    assert out["dataset"].attrs["compute_backend"] == "taudem"
    assert out["component"] is None
    assert called["taudem_n_procs"] == 4
    assert called["taudem_bin_dir"] == "C:/fake/taudem"


def test_coarsen_rasters_basic_and_factor_one():
    arr = np.arange(16, dtype=float).reshape(4, 4)
    profile = {
        "transform": from_origin(0.0, 4.0, 1.0, 1.0),
        "width": 4,
        "height": 4,
    }

    unchanged, unchanged_profile = coarsen_rasters({"dem": arr}, 1, profile)
    assert np.array_equal(unchanged["dem"], arr)
    assert unchanged_profile["width"] == 4

    out, out_profile = coarsen_rasters({"dem": arr, "ndvi": None}, 2, profile)
    assert out["dem"].shape == (2, 2)
    assert out["ndvi"] is None
    assert out_profile["width"] == 2
    assert out_profile["height"] == 2


def test_coarsen_rasters_too_large_factor_raises():
    arr = np.ones((2, 2), dtype=float)
    profile = {
        "transform": from_origin(0.0, 2.0, 1.0, 1.0),
        "width": 2,
        "height": 2,
    }
    with pytest.raises(ValueError, match="too large"):
        coarsen_rasters({"dem": arr}, 3, profile)


def test_nodes_from_geodataframe_delegates(monkeypatch):
    import geomorphconn.utils.target as target_mod

    called = {}

    def _fake_rasterize_targets(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return np.array([10, 11], dtype=np.int64)

    monkeypatch.setattr(target_mod, "rasterize_targets", _fake_rasterize_targets)
    out = nodes_from_geodataframe("dummy_gdf", "dummy_grid", all_touched=False, buffer_m=2.0)
    assert np.array_equal(out, np.array([10, 11], dtype=np.int64))
    assert called["kwargs"]["all_touched"] is False
    assert called["kwargs"]["buffer_m"] == 2.0


def test_rasterize_targets_import_error_branch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name in {"geopandas", "rasterio.features", "rasterio.transform"}:
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(ImportError, match="geopandas and rasterio are required"):
        rasterize_targets(source="dummy", grid=types.SimpleNamespace(number_of_node_rows=2, number_of_node_columns=2, dx=1.0))


def test_require_helpers_import_error_branches(monkeypatch):
    monkeypatch.setitem(sys.modules, "rioxarray", None)
    monkeypatch.setitem(sys.modules, "xarray", None)

    with pytest.raises(RuntimeError, match="requires rioxarray"):
        api._require_rioxarray()
    with pytest.raises(RuntimeError, match="requires xarray"):
        api._require_xarray()