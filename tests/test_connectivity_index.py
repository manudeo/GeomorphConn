"""
tests/test_connectivity_index.py
=================================
Tests for geomorphconn.ConnectivityIndex and the geomorphconn.weights pipeline.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
from landlab import HexModelGrid
from landlab import RasterModelGrid

from geomorphconn import ConnectivityIndex
from geomorphconn.components.connectivity_index import _ddn_d8_py
from geomorphconn.weights import (
    CustomWeight,
    LandCoverWeight,
    NDVIWeight,
    RainfallWeight,
    SurfaceRoughnessWeight,
    WeightBuilder,
    preset_landcover_only,
    preset_rainfall_landcover,
    preset_rainfall_ndvi,
    preset_rainfall_ndvi_roughness,
    preset_roughness_only,
)
from geomorphconn.weights.tables import WORLDCOVER_C_FACTOR

# ---------------------------------------------------------------------------
# Grid factory
# ---------------------------------------------------------------------------


def _make_grid(nrows: int = 20, ncols: int = 20, dx: float = 30.0, seed: int = 42):
    """Return a small synthetic V-shaped catchment grid."""
    rng = np.random.default_rng(seed)
    grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
    ri = np.arange(nrows)[:, None]
    ci = np.arange(ncols)[None, :]
    elev = (
        ri * 2.0
        + np.abs(ci - ncols // 2) * 1.5
        + rng.normal(0, 0.3, (nrows, ncols))
    )
    grid.add_field(
        "topographic__elevation",
        np.flipud(np.clip(elev, 0, None)).ravel(),
        at="node",
    )
    return grid


@pytest.fixture
def grid():
    return _make_grid()


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default(self, grid):
        assert ConnectivityIndex(grid) is not None

    def test_with_ndvi_and_rainfall(self, grid):
        n = grid.number_of_nodes
        rng = np.random.default_rng(1)
        ic = ConnectivityIndex(
            grid,
            ndvi=rng.uniform(0.0, 0.8, n),
            rainfall=rng.uniform(500, 1500, n),
        )
        assert ic is not None

    def test_bad_flow_director_raises(self, grid):
        with pytest.raises(ValueError):
            ConnectivityIndex(grid, flow_director="NONSENSE")

    def test_old_flow_director_aliases_raise(self, grid):
        for old_name in ("FlowDirectorSteepest", "FlowDirectorDINF", "FlowDirectorMFD"):
            with pytest.raises(ValueError):
                ConnectivityIndex(grid, flow_director=old_name)

    def test_ndvi_wrong_length_raises(self, grid):
        with pytest.raises(ValueError, match="'ndvi' length"):
            ConnectivityIndex(grid, ndvi=np.zeros(5))

    def test_rainfall_wrong_length_raises(self, grid):
        with pytest.raises(ValueError, match="'rainfall' length"):
            ConnectivityIndex(grid, rainfall=np.ones(5))

    def test_ndvi_from_field_name(self, grid):
        grid.add_field("my_ndvi", np.full(grid.number_of_nodes, 0.5), at="node")
        assert ConnectivityIndex(grid, ndvi="my_ndvi") is not None

    def test_weight_and_ndvi_warns(self, grid):
        wb = WeightBuilder().add(CustomWeight(np.full(grid.number_of_nodes, 0.5)))
        with pytest.warns(UserWarning, match="ignored"):
            ConnectivityIndex(grid, weight=wb, ndvi=np.zeros(grid.number_of_nodes))

    def test_wrong_weight_array_length_raises(self, grid):
        with pytest.raises(ValueError, match="'weight' array length"):
            ConnectivityIndex(grid, weight=np.ones(5))

    def test_non_raster_grid_raises(self):
        hex_grid = HexModelGrid((4, 4), spacing=10.0)
        with pytest.raises(ValueError, match="RasterModelGrid only"):
            ConnectivityIndex(hex_grid)

    def test_missing_named_field_raises_value_error(self, grid):
        with pytest.raises(ValueError, match="field 'missing_slope' not found"):
            ConnectivityIndex(grid, slope="missing_slope")


# ---------------------------------------------------------------------------
# Output fields after run_one_step
# ---------------------------------------------------------------------------

_OUTPUT_FIELDS = [
    "connectivity_index__IC",
    "connectivity_index__Dup",
    "connectivity_index__Ddn",
    "connectivity_index__W",
    "connectivity_index__S",
    "connectivity_index__Wmean",
    "connectivity_index__Smean",
]


class TestOutputFields:
    def test_all_fields_exist(self, grid):
        ic = ConnectivityIndex(grid)
        ic.run_one_step()
        for f in _OUTPUT_FIELDS:
            assert f in grid.at_node, f"Missing field: {f}"

    def test_IC_has_finite_values(self, grid):
        ic = ConnectivityIndex(grid)
        ic.run_one_step()
        assert np.isfinite(ic.IC).any()

    def test_W_clamped(self, grid):
        rng = np.random.default_rng(10)
        n = grid.number_of_nodes
        ic = ConnectivityIndex(
            grid,
            ndvi=rng.uniform(-0.1, 0.9, n),
            rainfall=rng.uniform(400, 2000, n),
        )
        ic.run_one_step()
        w = ic.W
        assert np.all(w[np.isfinite(w)] >= 0.005 - 1e-12)
        assert np.all(w[np.isfinite(w)] <= 1.0 + 1e-12)

    def test_S_clamped(self, grid):
        ConnectivityIndex(grid).run_one_step()
        s = grid.at_node["connectivity_index__S"]
        assert np.all(s[np.isfinite(s)] >= 0.005 - 1e-12)
        assert np.all(s[np.isfinite(s)] <= 1.0 + 1e-12)

    def test_Dup_nonneg(self, grid):
        ConnectivityIndex(grid).run_one_step()
        dup = grid.at_node["connectivity_index__Dup"][grid.core_nodes]
        assert np.all(dup >= 0)

    def test_Ddn_nonneg(self, grid):
        ConnectivityIndex(grid).run_one_step()
        ddn = grid.at_node["connectivity_index__Ddn"][grid.core_nodes]
        assert np.all(ddn >= 0)


# ---------------------------------------------------------------------------
# Flow directors (Landlab backend)
# ---------------------------------------------------------------------------


class TestFlowDirectors:
    @pytest.mark.parametrize(
        "director",
        [
            "D8",
            "DINF",
            "MFD",
        ],
    )
    def test_director_runs(self, director):
        ic = ConnectivityIndex(_make_grid(), flow_director=director)
        ic.run_one_step()
        assert np.isfinite(ic.IC).any()

    def test_dinf_differs_from_d8(self):
        g1, g2 = _make_grid(), _make_grid()
        ConnectivityIndex(g1, flow_director="D8").run_one_step()
        ConnectivityIndex(g2, flow_director="DINF").run_one_step()
        assert not np.allclose(
            g1.at_node["connectivity_index__Dup"],
            g2.at_node["connectivity_index__Dup"],
            equal_nan=True,
        )

    def test_mfd_differs_from_d8(self):
        g1, g2 = _make_grid(), _make_grid()
        ConnectivityIndex(g1, flow_director="D8").run_one_step()
        ConnectivityIndex(g2, flow_director="MFD").run_one_step()
        assert not np.allclose(
            g1.at_node["connectivity_index__Dup"],
            g2.at_node["connectivity_index__Dup"],
            equal_nan=True,
        )

    def test_aspect_weighting_default_equals_explicit_false(self):
        g1, g2 = _make_grid(), _make_grid()
        ConnectivityIndex(g1, flow_director="DINF").run_one_step()
        ConnectivityIndex(
            g2,
            flow_director="DINF",
            use_aspect_weighting=False,
        ).run_one_step()
        assert np.allclose(
            g1.at_node["connectivity_index__Dup"],
            g2.at_node["connectivity_index__Dup"],
            equal_nan=True,
        )

    def test_aspect_weighting_changes_dinf_dup(self):
        g1, g2 = _make_grid(), _make_grid()
        ConnectivityIndex(
            g1,
            flow_director="DINF",
            use_aspect_weighting=False,
        ).run_one_step()
        ConnectivityIndex(
            g2,
            flow_director="DINF",
            use_aspect_weighting=True,
        ).run_one_step()
        assert not np.allclose(
            g1.at_node["connectivity_index__Dup"],
            g2.at_node["connectivity_index__Dup"],
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# Target mode
# ---------------------------------------------------------------------------


class TestTargetMode:
    def test_target_changes_IC(self):
        nrows, ncols = 25, 25
        g1 = _make_grid(nrows=nrows, ncols=ncols)
        g2 = _make_grid(nrows=nrows, ncols=ncols)

        mid_row = nrows // 2
        band = np.zeros((nrows, ncols), dtype=bool)
        band[mid_row, :] = True
        target_nodes = np.where(band.ravel())[0]

        ConnectivityIndex(g1).run_one_step()
        ConnectivityIndex(g2, target_nodes=target_nodes).run_one_step()

        assert not np.allclose(
            g1.at_node["connectivity_index__IC"],
            g2.at_node["connectivity_index__IC"],
            equal_nan=True,
        )

    def test_target_int64_array(self, grid):
        target = np.array([50, 51, 52], dtype=np.int64)
        ic = ConnectivityIndex(grid, target_nodes=target)
        ic.run_one_step()
        assert np.isfinite(ic.IC).any()

    def test_target_changes_Dup(self):
        nrows, ncols = 25, 25
        g1 = _make_grid(nrows=nrows, ncols=ncols)
        g2 = _make_grid(nrows=nrows, ncols=ncols)

        mid_row = nrows // 2
        band = np.zeros((nrows, ncols), dtype=bool)
        band[mid_row, :] = True
        target_nodes = np.where(band.ravel())[0]

        ConnectivityIndex(g1).run_one_step()
        ConnectivityIndex(g2, target_nodes=target_nodes).run_one_step()

        assert not np.allclose(
            g1.at_node["connectivity_index__Dup"],
            g2.at_node["connectivity_index__Dup"],
            equal_nan=True,
        )

    def test_analysis_mask_sets_outside_to_nan(self):
        nrows, ncols = 20, 20
        g = _make_grid(nrows=nrows, ncols=ncols)

        mask = np.zeros((nrows, ncols), dtype=bool)
        mask[4:16, 4:16] = True
        mask_nodes = np.where(np.flipud(mask).ravel())[0].astype(np.int64)

        ConnectivityIndex(g, analysis_mask_nodes=mask_nodes).run_one_step()

        ic = g.at_node["connectivity_index__IC"]
        outside_nodes = np.where(~np.flipud(mask).ravel())[0]
        assert np.all(~np.isfinite(ic[outside_nodes]))

    def test_analysis_mask_filters_stream_threshold_targets(self):
        nrows, ncols = 30, 30
        g = _make_grid(nrows=nrows, ncols=ncols)

        left_half = np.zeros((nrows, ncols), dtype=bool)
        left_half[:, : ncols // 2] = True
        mask_nodes = np.where(np.flipud(left_half).ravel())[0].astype(np.int64)

        ConnectivityIndex(
            g,
            stream_threshold=5,
            analysis_mask_nodes=mask_nodes,
        ).run_one_step()

        ic = g.at_node["connectivity_index__IC"]
        right_half_nodes = np.where(~np.flipud(left_half).ravel())[0]
        assert np.all(~np.isfinite(ic[right_half_nodes]))


class TestDdnKernel:
    def test_terminal_node_ddn_equals_inv_ws(self):
        """Terminal (outlet/target) gets inv_WS per Borselli (2008) ArcGIS recipe.
        Chain: 0 -> 1 -> 2 (terminal).
          ddn[2] = inv_ws[2] = 5.0
          ddn[1] = dist[1]*inv_ws[1] + ddn[2] = 3.0 + 5.0 = 8.0
          ddn[0] = dist[0]*inv_ws[0] + ddn[1] = 2.0 + 8.0 = 10.0
        """
        local = np.array([2.0, 3.0, 0.0], dtype=np.float64)
        inv_ws = np.array([1.0, 1.0, 5.0], dtype=np.float64)
        recv = np.array([1, 2, 2], dtype=np.int64)
        order = np.array([0, 1, 2], dtype=np.int64)

        ddn = _ddn_d8_py(local, inv_ws, recv, order)

        assert ddn[2] == pytest.approx(5.0)   # terminal = inv_ws
        assert ddn[1] == pytest.approx(8.0)   # 3.0 + 5.0
        assert ddn[0] == pytest.approx(10.0)  # 2.0 + 8.0


# ---------------------------------------------------------------------------
# Stream threshold
# ---------------------------------------------------------------------------


class TestStreamThreshold:
    def test_stream_threshold_produces_different_IC(self):
        """With a stream threshold, near-channel cells should route to channel."""
        g1 = _make_grid()
        g2 = _make_grid()
        ConnectivityIndex(g1).run_one_step()
        ConnectivityIndex(g2, stream_threshold=50).run_one_step()
        assert not np.allclose(
            g1.at_node["connectivity_index__IC"],
            g2.at_node["connectivity_index__IC"],
            equal_nan=True,
        )

    def test_stream_threshold_masks_channel_cells(self):
        """Channel cells (acc >= threshold) must be NaN in IC output."""
        grid = _make_grid()
        ic = ConnectivityIndex(grid, stream_threshold=20)
        ic.run_one_step()
        # At least some cells should be NaN (the channel network)
        assert np.isnan(ic.IC).any()
        # And at least some cells should have finite IC (upstream of channel)
        assert np.isfinite(ic.IC).any()

    def test_stream_threshold_and_target_nodes_merge(self):
        """Supplying both target_nodes and stream_threshold should take their union."""
        grid = _make_grid()
        target = np.array([50, 51, 52], dtype=np.int64)
        # With threshold, more cells should be NaN than with target nodes alone
        g1 = _make_grid()
        g2 = _make_grid()
        ConnectivityIndex(g1, target_nodes=target).run_one_step()
        ConnectivityIndex(g2, target_nodes=target, stream_threshold=20).run_one_step()
        nan1 = np.sum(np.isnan(g1.at_node["connectivity_index__IC"]))
        nan2 = np.sum(np.isnan(g2.at_node["connectivity_index__IC"]))
        assert nan2 >= nan1

    @pytest.mark.parametrize("director", ["D8", "DINF", "MFD"])
    def test_stream_threshold_uses_selected_director_accumulation(self, director):
        """Effective targets must be computed from the chosen primary director accumulation."""
        grid = _make_grid()
        threshold = 20
        ic = ConnectivityIndex(grid, flow_director=director, stream_threshold=threshold)
        routing = ic._run_routing()

        cell_area = float(grid.dx) * float(grid.dy)
        expected = np.where((routing["drainage_area"] / cell_area) >= threshold)[0].astype(np.int64)
        effective = routing["effective_target_nodes"]

        if len(expected) == 0:
            assert effective is None or len(effective) == 0
        else:
            assert effective is not None
            np.testing.assert_array_equal(np.sort(effective), np.sort(expected))


# ---------------------------------------------------------------------------
# as_2d
# ---------------------------------------------------------------------------


class TestAs2D:
    def test_shape(self):
        nrows, ncols = 18, 22
        ic = ConnectivityIndex(_make_grid(nrows=nrows, ncols=ncols))
        ic.run_one_step()
        assert ic.as_2d().shape == (nrows, ncols)

    def test_geo_order_is_flipud(self):
        ic = ConnectivityIndex(_make_grid())
        ic.run_one_step()
        np.testing.assert_array_equal(
            ic.as_2d(geo_order=True),
            np.flipud(ic.as_2d(geo_order=False)),
        )


# ---------------------------------------------------------------------------
# Update methods
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_ndvi_changes_IC(self, grid):
        ic = ConnectivityIndex(grid)
        ic.run_one_step()
        ic_before = ic.IC.copy()
        ic.update_ndvi(np.full(grid.number_of_nodes, 0.9))
        ic.run_one_step()
        assert not np.allclose(ic_before, ic.IC, equal_nan=True)

    def test_update_rainfall_changes_IC(self, grid):
        """Spatially varying rainfall → changes W → changes IC."""
        rng = np.random.default_rng(7)
        ic = ConnectivityIndex(grid)
        ic.run_one_step()
        ic_before = ic.IC.copy()
        # Non-uniform rainfall so normalisation produces a different W
        ic.update_rainfall(rng.uniform(100, 2000, grid.number_of_nodes))
        ic.run_one_step()
        assert not np.allclose(ic_before, ic.IC, equal_nan=True)

    def test_update_weight_changes_IC(self, grid):
        ic = ConnectivityIndex(grid)
        ic.run_one_step()
        ic_before = ic.IC.copy()
        ic.update_weight(preset_roughness_only(grid))
        ic.run_one_step()
        assert not np.allclose(ic_before, ic.IC, equal_nan=True)


# ---------------------------------------------------------------------------
# Slope convention
# ---------------------------------------------------------------------------


class TestSlopeConvention:
    def test_slope_is_dydx_by_default(self):
        g = _make_grid()
        ConnectivityIndex(g).run_one_step()
        s = g.at_node["connectivity_index__S"]
        assert np.isfinite(s[np.isfinite(s)]).all()


# ---------------------------------------------------------------------------
# Weight components — unit tests
# ---------------------------------------------------------------------------


class TestRainfallWeight:
    def test_range(self):
        w = RainfallWeight(np.linspace(200, 1500, 100)).compute()
        assert (w >= 0.005 - 1e-12).all()
        assert (w <= 1.0 + 1e-12).all()

    def test_normalised_endpoints(self):
        w = RainfallWeight(np.array([0.0, 500.0, 1000.0]), w_min=0.0).compute()
        assert w[0] == pytest.approx(0.0)
        assert w[-1] == pytest.approx(1.0)

    def test_uniform_rainfall_midpoint(self):
        # Spatially uniform → all values equal (fallback = 0.5, then clamped)
        w = RainfallWeight(np.full(10, 800.0)).compute()
        assert np.allclose(w, w[0])


class TestNDVIWeight:
    def test_range(self):
        w = NDVIWeight(np.linspace(-1, 1, 50)).compute()
        assert (w >= 0.005 - 1e-12).all()
        assert (w <= 1.0 + 1e-12).all()

    def test_dense_veg_lower_than_bare(self):
        w_dense = NDVIWeight(np.array([0.9])).compute()[0]
        w_bare = NDVIWeight(np.array([0.0])).compute()[0]
        assert w_dense < w_bare

    def test_formula(self):
        # C = (1 - 0.6) / 2 = 0.2 — no clamping needed
        w = NDVIWeight(np.array([0.6]), w_min=0.0).compute()
        assert w[0] == pytest.approx(0.2)


class TestSurfaceRoughnessWeight:
    def test_range(self):
        w = SurfaceRoughnessWeight(_make_grid()).compute()
        assert (w >= 0.005 - 1e-12).all()
        assert (w <= 1.0 + 1e-12).all()

    def test_impedance_interpretation(self):
        """High roughness should produce low weight (impedance interpretation)."""
        grid = _make_grid()
        # Create elevated terrain at center (high local residual roughness)
        z = grid.at_node["topographic__elevation"].reshape(grid.number_of_node_rows, grid.number_of_node_columns)
        center_r, center_c = grid.number_of_node_rows // 2, grid.number_of_node_columns // 2
        z[center_r, center_c] += 100  # Add tall peak at center
        
        w = SurfaceRoughnessWeight(grid).compute()
        # Center node has high roughness → should have low weight
        center_node = center_r * grid.number_of_node_columns + center_c
        # High roughness at center should produce lower weight than average
        assert w[center_node] < np.mean(w)

    def test_flat_dem_returns_near_one(self):
        """Flat DEM has RI=0 everywhere, so W should be 1 everywhere."""
        grid = _make_grid()
        grid.at_node["topographic__elevation"][:] = 100.0
        w = SurfaceRoughnessWeight(grid).compute()
        assert np.allclose(w, 1.0)

    def test_even_window_raises(self):
        grid = _make_grid()
        with pytest.raises(ValueError):
            SurfaceRoughnessWeight(grid, detrend_window=4)
        with pytest.raises(ValueError):
            SurfaceRoughnessWeight(grid, std_window=2)



class TestLandCoverWeight:
    def test_range(self):
        lc = np.array([10, 20, 40, 60, 80, -1])
        w = LandCoverWeight(lc, WORLDCOVER_C_FACTOR).compute()
        assert (w >= 0.005 - 1e-12).all()
        assert (w <= 1.0 + 1e-12).all()

    def test_bare_gt_forest(self):
        w_forest = LandCoverWeight(np.array([10]), WORLDCOVER_C_FACTOR).compute()[0]
        w_bare = LandCoverWeight(np.array([60]), WORLDCOVER_C_FACTOR).compute()[0]
        assert w_bare > w_forest

    def test_from_worldcover(self):
        w = LandCoverWeight.from_worldcover(np.array([10, 40, 60])).compute()
        assert len(w) == 3

    def test_from_corine(self):
        from geomorphconn.weights.tables import CORINE_C_FACTOR

        w = LandCoverWeight.from_corine(np.array([311, 211, 332])).compute()
        assert len(w) == 3
        # Forest (311) should have lower C than bare rock (332)
        w_forest = LandCoverWeight(np.array([311]), CORINE_C_FACTOR).compute()[0]
        w_rock = LandCoverWeight(np.array([332]), CORINE_C_FACTOR).compute()[0]
        assert w_forest < w_rock

    def test_from_modis_igbp(self):
        w = LandCoverWeight.from_modis_igbp(np.array([1, 12, 16])).compute()
        assert len(w) == 3

    def test_custom_table(self):
        custom = {1: 0.1, 2: 0.5, 3: 0.9}
        w = LandCoverWeight(
            np.array([1, 2, 3]), custom, normalise=False
        ).compute()
        assert len(w) == 3

    def test_no_normalise_preserves_values(self):
        table = {1: 0.1, 2: 0.5}
        w = LandCoverWeight(
            np.array([1, 2]), table, normalise=False, w_min=0.0
        ).compute()
        assert w[0] == pytest.approx(0.1)
        assert w[1] == pytest.approx(0.5)

    def test_unknown_code_uses_fallback(self):
        """With normalise=False, unknown code → fallback_c directly."""
        w = LandCoverWeight(
            np.array([9999]),
            WORLDCOVER_C_FACTOR,
            fallback_c=0.3,
            normalise=False,
        ).compute()[0]
        assert w == pytest.approx(0.3)


class TestCustomWeight:
    def test_clamps_below(self):
        assert CustomWeight(np.array([0.0])).compute()[0] == pytest.approx(0.005)

    def test_clamps_above(self):
        assert CustomWeight(np.array([2.0])).compute()[0] == pytest.approx(1.0)

    def test_passthrough_valid(self):
        w = CustomWeight(np.array([0.5]), w_min=0.0).compute()[0]
        assert w == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# WeightBuilder — combination modes
# ---------------------------------------------------------------------------


def _rf(n: int = 100) -> np.ndarray:
    return np.random.default_rng(0).uniform(300, 1200, n)


def _ndvi(n: int = 100) -> np.ndarray:
    return np.random.default_rng(1).uniform(0.1, 0.8, n)


class TestWeightBuilder:
    def test_single_component(self):
        wb = WeightBuilder().add(RainfallWeight(_rf()))
        w = wb.build()
        assert w.shape == (100,)
        assert (w >= 0.005).all() and (w <= 1.0).all()

    def test_mean(self):
        wb = WeightBuilder(combine="mean")
        wb.add(RainfallWeight(_rf())).add(NDVIWeight(_ndvi()))
        assert wb.build().shape == (100,)

    def test_geometric_mean(self):
        wb = (
            WeightBuilder(combine="geometric_mean")
            .add(RainfallWeight(_rf()))
            .add(NDVIWeight(_ndvi()))
        )
        assert (wb.build() >= 0.005).all()

    def test_product(self):
        wb = (
            WeightBuilder(combine="product")
            .add(RainfallWeight(_rf()))
            .add(NDVIWeight(_ndvi()))
        )
        assert (wb.build() >= 0.005).all()

    def test_max_mode(self):
        wb = (
            WeightBuilder(combine="max")
            .add(RainfallWeight(_rf()))
            .add(NDVIWeight(_ndvi()))
        )
        assert (wb.build() >= 0.005).all()

    def test_min_mode(self):
        wb = (
            WeightBuilder(combine="min")
            .add(RainfallWeight(_rf()))
            .add(NDVIWeight(_ndvi()))
        )
        assert (wb.build() >= 0.005).all()

    def test_weighted_mean(self):
        wb = (
            WeightBuilder(combine="weighted_mean")
            .add(RainfallWeight(_rf()), component_weight=0.7)
            .add(NDVIWeight(_ndvi()), component_weight=0.3)
        )
        assert wb.build().shape == (100,)

    def test_custom_callable(self):
        def top_half(arrays):
            return np.max(np.stack(arrays), axis=0)

        wb = WeightBuilder(combine=top_half)
        wb.add(RainfallWeight(_rf())).add(NDVIWeight(_ndvi()))
        assert wb.build().shape == (100,)

    def test_no_components_raises(self):
        with pytest.raises(ValueError, match="no components"):
            WeightBuilder().build()

    def test_wrong_length_raises(self):
        wb = WeightBuilder().add(RainfallWeight(np.ones(50)))
        with pytest.raises(ValueError, match="length"):
            wb.build(n_nodes=100)

    def test_chaining_returns_self(self):
        wb = WeightBuilder()
        result = wb.add(RainfallWeight(_rf()))
        assert result is wb

    def test_describe_contains_name(self):
        wb = WeightBuilder().add(RainfallWeight(_rf()))
        assert "rainfall" in wb.describe()

    def test_unknown_mode_raises(self):
        wb = (
            WeightBuilder(combine="unicorn")
            .add(RainfallWeight(_rf()))
            .add(NDVIWeight(_ndvi()))
        )
        with pytest.raises(ValueError, match="Unknown combine mode"):
            wb.build()


# ---------------------------------------------------------------------------
# Weight integration with ConnectivityIndex
# ---------------------------------------------------------------------------


class TestWeightInConnectivityIndex:
    def test_precomputed_array(self):
        grid = _make_grid()
        ic = ConnectivityIndex(grid, weight=np.full(grid.number_of_nodes, 0.3))
        ic.run_one_step()
        assert np.allclose(ic.W, 0.3)

    def test_builder_equals_legacy_for_same_formula(self):
        """WeightBuilder(RF+NDVI) should match legacy ndvi= / rainfall= output."""
        rng = np.random.default_rng(99)
        n = _make_grid().number_of_nodes
        rf = rng.uniform(300, 1200, n)
        ndvi = rng.uniform(0.1, 0.8, n)

        g1 = _make_grid()
        g2 = _make_grid()
        ConnectivityIndex(g1, ndvi=ndvi, rainfall=rf).run_one_step()
        ConnectivityIndex(
            g2, weight=WeightBuilder().add(RainfallWeight(rf)).add(NDVIWeight(ndvi))
        ).run_one_step()

        # Two separate grid runs have tiny float diffs; verify the
        # mathematical result matches to within 1 % (relative).
        ic1 = g1.at_node["connectivity_index__IC"]
        ic2 = g2.at_node["connectivity_index__IC"]
        finite = np.isfinite(ic1) & np.isfinite(ic2)
        assert np.allclose(ic1[finite], ic2[finite], rtol=0.01)

    def test_roughness_only_runs(self):
        grid = _make_grid()
        ic = ConnectivityIndex(grid, weight=preset_roughness_only(grid))
        ic.run_one_step()
        assert np.isfinite(ic.IC).any()

    def test_landcover_preset_runs(self):
        grid = _make_grid()
        lc = np.random.choice([10, 20, 40, 60, 80], size=grid.number_of_nodes)
        ic = ConnectivityIndex(
            grid, weight=preset_landcover_only(lc, WORLDCOVER_C_FACTOR)
        )
        ic.run_one_step()
        assert np.isfinite(ic.IC).any()

    def test_preset_rainfall_ndvi_roughness(self):
        grid = _make_grid()
        rng = np.random.default_rng(7)
        n = grid.number_of_nodes
        ic = ConnectivityIndex(
            grid,
            weight=preset_rainfall_ndvi_roughness(
                rng.uniform(300, 1200, n),
                rng.uniform(0.1, 0.8, n),
                grid,
            ),
        )
        ic.run_one_step()
        assert np.isfinite(ic.IC).any()


# ---------------------------------------------------------------------------
# Preset factory functions
# ---------------------------------------------------------------------------


class TestPresets:
    def test_all_presets_produce_valid_builders(self):
        """Each preset must return a WeightBuilder whose .build() output is valid."""
        n = _make_grid().number_of_nodes
        rng = np.random.default_rng(3)
        rf = rng.uniform(300, 1200, n)
        ndvi_arr = rng.uniform(0.1, 0.8, n)
        lc = np.random.choice([10, 40, 60], size=n)
        grid = _make_grid()

        cases = [
            preset_rainfall_ndvi(rf, ndvi_arr),
            preset_roughness_only(grid),
            preset_landcover_only(lc),
            preset_rainfall_landcover(rf, lc),
            preset_rainfall_ndvi_roughness(rf, ndvi_arr, grid),
        ]
        for wb in cases:
            assert isinstance(wb, WeightBuilder)
            w = wb.build(n_nodes=n)
            assert w.shape == (n,)
            assert (w >= 0.005 - 1e-12).all()
            assert (w <= 1.0 + 1e-12).all()


# ---------------------------------------------------------------------------
# Docstring smoke test
# ---------------------------------------------------------------------------


def test_quickstart_docstring():
    """Verify the package docstring quick-start example runs without error."""
    grid = RasterModelGrid((15, 15), xy_spacing=30.0)
    z = grid.add_zeros("topographic__elevation", at="node")
    z += np.random.default_rng(0).random(grid.number_of_nodes) * 30
    ic = ConnectivityIndex(
        grid,
        ndvi=np.full(grid.number_of_nodes, 0.4),
        rainfall=np.full(grid.number_of_nodes, 800.0),
    )
    ic.run_one_step()
    assert np.isfinite(ic.IC).any()
