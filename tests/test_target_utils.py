from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from geomorphconn.utils.target import rasterize_targets


class _FakeGeomTypeSeries(list):
    def dropna(self):
        return _FakeGeomTypeSeries([v for v in self if v is not None])

    def tolist(self):
        return list(self)


class _FakeGeom:
    def __init__(self, geom_type):
        self.geom_type = geom_type
        self.buffer_calls = []

    def buffer(self, distance):
        self.buffer_calls.append(distance)
        return _FakeGeom("Polygon")


class _FakeGeometryColumn(list):
    @property
    def geom_type(self):
        return _FakeGeomTypeSeries([g.geom_type if g is not None else None for g in self])

    def buffer(self, distance):
        return _FakeGeometryColumn([g.buffer(distance) if g is not None else None for g in self])

    def apply(self, func):
        return _FakeGeometryColumn([func(g) for g in self])


class _FakeGDF:
    def __init__(self, geometries, crs="EPSG:4326"):
        self._geometry = _FakeGeometryColumn(geometries)
        self.crs = crs
        self.last_to_crs = None

    @property
    def geometry(self):
        return self._geometry

    def copy(self):
        out = _FakeGDF(list(self._geometry), crs=self.crs)
        out.last_to_crs = self.last_to_crs
        return out

    def to_crs(self, crs):
        self.last_to_crs = crs
        return self

    def __setitem__(self, key, value):
        if key == "geometry":
            self._geometry = _FakeGeometryColumn(value)
            return
        raise KeyError(key)


def _install_fake_modules(monkeypatch, gdf, raster, from_origin_return=("T",)):
    gpd_mod = types.ModuleType("geopandas")
    gpd_mod.read_file = lambda path: gdf

    calls = {}

    def _fake_rasterize(shapes, out_shape, transform, fill, dtype, all_touched):
        calls["shapes"] = shapes
        calls["out_shape"] = out_shape
        calls["transform"] = transform
        calls["all_touched"] = all_touched
        return raster

    features_mod = types.ModuleType("rasterio.features")
    features_mod.rasterize = _fake_rasterize

    transform_mod = types.ModuleType("rasterio.transform")
    transform_mod.from_origin = lambda x0, y0, dx, dy: (
        from_origin_return,
        x0,
        y0,
        dx,
        dy,
    )

    rasterio_mod = types.ModuleType("rasterio")
    rasterio_mod.features = features_mod
    rasterio_mod.transform = transform_mod

    monkeypatch.setitem(sys.modules, "geopandas", gpd_mod)
    monkeypatch.setitem(sys.modules, "rasterio", rasterio_mod)
    monkeypatch.setitem(sys.modules, "rasterio.features", features_mod)
    monkeypatch.setitem(sys.modules, "rasterio.transform", transform_mod)

    return calls


def test_rasterize_targets_path_input_auto_buffers_lines(monkeypatch):
    gdf = _FakeGDF([_FakeGeom("LineString")], crs="EPSG:4326")
    raster = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    calls = _install_fake_modules(monkeypatch, gdf=gdf, raster=raster)

    grid = types.SimpleNamespace(number_of_node_rows=2, number_of_node_columns=3, dx=10.0)
    out = rasterize_targets(
        source="dummy.shp",
        grid=grid,
        dem_transform="MY_TRANSFORM",
        dem_crs="EPSG:32632",
        all_touched=False,
        buffer_m=0.0,
    )

    assert np.array_equal(out, np.array([1, 3], dtype=np.int64))
    assert gdf.last_to_crs == "EPSG:32632"
    assert calls["out_shape"] == (2, 3)
    assert calls["transform"] == "MY_TRANSFORM"
    assert calls["all_touched"] is False


def test_rasterize_targets_uses_explicit_buffer(monkeypatch):
    line = _FakeGeom("LineString")
    gdf = _FakeGDF([line], crs="EPSG:4326")
    raster = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    _install_fake_modules(monkeypatch, gdf=gdf, raster=raster)

    grid = types.SimpleNamespace(number_of_node_rows=2, number_of_node_columns=2, dx=30.0)
    out = rasterize_targets(source=gdf, grid=grid, dem_transform="T", buffer_m=12.5)

    assert out.dtype == np.int64
    assert len(out) == 1
    assert line.buffer_calls == [12.5]


def test_rasterize_targets_requires_crs(monkeypatch):
    gdf = _FakeGDF([_FakeGeom("Polygon")], crs=None)
    raster = np.array([[1]], dtype=np.uint8)
    _install_fake_modules(monkeypatch, gdf=gdf, raster=raster)

    grid = types.SimpleNamespace(number_of_node_rows=1, number_of_node_columns=1, dx=1.0)
    with pytest.raises(ValueError, match="has no CRS"):
        rasterize_targets(source=gdf, grid=grid)


def test_rasterize_targets_raises_on_empty_geometries(monkeypatch):
    gdf = _FakeGDF([None], crs="EPSG:4326")
    raster = np.array([[0, 0]], dtype=np.uint8)
    _install_fake_modules(monkeypatch, gdf=gdf, raster=raster)

    grid = types.SimpleNamespace(number_of_node_rows=1, number_of_node_columns=2, dx=1.0)
    with pytest.raises(ValueError, match="No valid geometries"):
        rasterize_targets(source=gdf, grid=grid, dem_transform="T")


def test_rasterize_targets_raises_when_no_target_nodes(monkeypatch):
    gdf = _FakeGDF([_FakeGeom("Polygon")], crs="EPSG:4326")
    raster = np.zeros((2, 2), dtype=np.uint8)
    _install_fake_modules(monkeypatch, gdf=gdf, raster=raster)

    grid = types.SimpleNamespace(number_of_node_rows=2, number_of_node_columns=2, dx=5.0)
    with pytest.raises(RuntimeError, match="No target nodes found"):
        rasterize_targets(source=gdf, grid=grid, dem_transform="T")


def test_rasterize_targets_builds_default_transform(monkeypatch):
    gdf = _FakeGDF([_FakeGeom("Polygon")], crs="EPSG:4326")
    raster = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    calls = _install_fake_modules(monkeypatch, gdf=gdf, raster=raster, from_origin_return="AUTO")

    grid = types.SimpleNamespace(number_of_node_rows=2, number_of_node_columns=2, dx=20.0)
    out = rasterize_targets(source=gdf, grid=grid, dem_transform=None)

    assert len(out) == 1
    # from_origin should be called with northing = nrows * dx
    assert calls["transform"][0] == "AUTO"
    assert calls["transform"][1:] == (0.0, 40.0, 20.0, 20.0)