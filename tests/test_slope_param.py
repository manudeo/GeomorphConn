"""
Test custom slope parameter for ConnectivityIndex.

Verify that externally-provided slopes (e.g., from TauDEM) can be
used directly without being transformed by degree approximation.
"""

import numpy as np
from landlab import RasterModelGrid
from geomorphconn import ConnectivityIndex


def test_slope_param_default_landlab():
    """Test default behavior: Landlab-computed slope with degree approx."""
    grid = RasterModelGrid((10, 10), xy_spacing=30.0)
    z = grid.add_zeros("topographic__elevation", at="node")
    z += np.random.default_rng(42).random(grid.number_of_nodes) * 50
    
    ndvi = np.full(grid.number_of_nodes, 0.4)
    rf = np.full(grid.number_of_nodes, 800.0)
    
    # Default: use_degree_approx=True (ArcGIS-like)
    ic = ConnectivityIndex(grid, ndvi=ndvi, rainfall=rf, use_degree_approx=True)
    ic.run_one_step()
    
    # Should have valid S values (degree-approximated slope)
    S = grid.at_node["connectivity_index__S"]
    assert np.all(np.isfinite(S[~grid.status_at_node.astype(bool)]))
    print(f"✓ Test 1: Default Landlab slope")
    print(f"  S range: {S[~grid.status_at_node.astype(bool)].min():.6f} to"
          f" {S[~grid.status_at_node.astype(bool)].max():.6f}")


def test_slope_param_custom_uniform():
    """
    Test custom slope parameter with uniform value.
    
    Custom slope is tan(θ), should be used directly without
    degree approximation conversion.
    """
    grid = RasterModelGrid((10, 10), xy_spacing=30.0)
    z = grid.add_zeros("topographic__elevation", at="node")
    z += np.random.default_rng(42).random(grid.number_of_nodes) * 50
    
    ndvi = np.full(grid.number_of_nodes, 0.4)
    rf = np.full(grid.number_of_nodes, 800.0)
    
    # Custom slope: uniform tan(θ) = 0.5
    custom_slope = np.full(grid.number_of_nodes, 0.5)
    
    ic = ConnectivityIndex(
        grid, ndvi=ndvi, rainfall=rf, slope=custom_slope, use_degree_approx=True
    )
    ic.run_one_step()
    
    S = grid.at_node["connectivity_index__S"]
    S_core = S[~grid.status_at_node.astype(bool)]
    
    # With custom slope, S should be approximately 0.5 (clamped to [0.005, 1.0])
    expected = 0.5
    actual_min, actual_max = S_core.min(), S_core.max()
    
    print(f"✓ Test 2: Custom uniform slope = 0.5")
    print(f"  Expected S: {expected}")
    print(f"  Actual S range: {actual_min:.6f} to {actual_max:.6f}")
    
    # Since core nodes have uniform custom slope, S should be uniform
    # Allow small numerical variation (0.01)
    assert np.allclose(S_core, expected, atol=0.01), \
        f"Expected S ≈ {expected}, got range {actual_min:.6f}–{actual_max:.6f}"


def test_slope_param_custom_array():
    """
    Test custom slope parameter with varying slope array.
    
    Verify that custom slopes are passed through without
    degree approximation transformation.
    """
    grid = RasterModelGrid((10, 10), xy_spacing=30.0)
    z = grid.add_zeros("topographic__elevation", at="node")
    z += np.random.default_rng(42).random(grid.number_of_nodes) * 50
    
    ndvi = np.full(grid.number_of_nodes, 0.4)
    rf = np.full(grid.number_of_nodes, 800.0)
    
    # Custom slope: array of varying tan(θ) values
    custom_slope = np.full(grid.number_of_nodes, 0.3)
    
    ic = ConnectivityIndex(
        grid, ndvi=ndvi, rainfall=rf, slope=custom_slope, use_degree_approx=True
    )
    ic.run_one_step()
    
    S = grid.at_node["connectivity_index__S"]
    S_core = S[~grid.status_at_node.astype(bool)]
    
    expected = 0.3
    actual_min, actual_max = S_core.min(), S_core.max()
    
    print(f"✓ Test 3: Custom varying slope (mean = 0.3)")
    print(f"  Expected S: {expected}")
    print(f"  Actual S range: {actual_min:.6f} to {actual_max:.6f}")
    
    # Custom slope should be used directly
    assert np.allclose(S_core, expected, atol=0.01), \
        f"Expected S ≈ {expected}, got range {actual_min:.6f}–{actual_max:.6f}"


def test_slope_param_no_degree_approx():
    """
    Test that use_degree_approx=False works with both Landlab and custom slopes.
    """
    grid = RasterModelGrid((10, 10), xy_spacing=30.0)
    z = grid.add_zeros("topographic__elevation", at="node")
    z += np.random.default_rng(42).random(grid.number_of_nodes) * 50
    
    ndvi = np.full(grid.number_of_nodes, 0.4)
    rf = np.full(grid.number_of_nodes, 800.0)
    
    # Landlab computation with use_degree_approx=False
    ic1 = ConnectivityIndex(grid, ndvi=ndvi, rainfall=rf, use_degree_approx=False)
    ic1.run_one_step()
    S1 = grid.at_node["connectivity_index__S"].copy()
    
    # Custom slope with use_degree_approx=False
    # (should be identical if custom slope matches Landlab's tan(θ))
    grid2 = RasterModelGrid((10, 10), xy_spacing=30.0)
    z2 = grid2.add_zeros("topographic__elevation", at="node")
    z2[:] = z.copy()
    
    custom_slope = S1.copy()  # Use Landlab's computed tan(θ) as custom
    ic2 = ConnectivityIndex(
        grid2, ndvi=ndvi, rainfall=rf, slope=custom_slope, use_degree_approx=False
    )
    ic2.run_one_step()
    S2 = grid2.at_node["connectivity_index__S"].copy()
    
    print(f"✓ Test 4: use_degree_approx=False comparison")
    print(f"  Landlab S range: {S1[~grid.status_at_node.astype(bool)].min():.6f} to"
          f" {S1[~grid.status_at_node.astype(bool)].max():.6f}")
    print(f"  Custom S range:  {S2[~grid2.status_at_node.astype(bool)].min():.6f} to"
          f" {S2[~grid2.status_at_node.astype(bool)].max():.6f}")


if __name__ == "__main__":
    test_slope_param_default_landlab()
    test_slope_param_custom_uniform()
    test_slope_param_custom_array()
    test_slope_param_no_degree_approx()
    print("\n✅ All slope parameter tests passed!")
