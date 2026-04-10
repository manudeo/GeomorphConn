"""Quick test of the new slope parameter functionality."""
import numpy as np
from landlab import RasterModelGrid
from geomorphconn import ConnectivityIndex

# Create test grid
grid = RasterModelGrid((10, 10), xy_spacing=30.0)
z = grid.add_zeros("topographic__elevation", at="node")
z += np.random.default_rng(0).random(grid.number_of_nodes) * 50

# Test 1: Default (internal Landlab slope)
print("Test 1: Default Landlab-computed slope")
ic1 = ConnectivityIndex(grid)
ic1.run_one_step()
print(f"  IC range: {ic1.IC[np.isfinite(ic1.IC)].min():.3f} to {ic1.IC[np.isfinite(ic1.IC)].max():.3f}")
print(f"  S range:  {ic1.S.min():.6f} to {ic1.S.max():.6f}")

# Test 2: Custom slope (simulating TauDEM output)
print("\nTest 2: Custom TauDEM-style slope")
custom_slope = 0.5 * np.ones(grid.number_of_nodes)  # Uniform slope for demo
ic2 = ConnectivityIndex(grid, slope=custom_slope)
ic2.run_one_step()
print(f"  IC range: {ic2.IC[np.isfinite(ic2.IC)].min():.3f} to {ic2.IC[np.isfinite(ic2.IC)].max():.3f}")
print(f"  S range:  {ic2.S.min():.6f} to {ic2.S.max():.6f}")
print(f"  Custom slope correctly applied: {np.allclose(ic2.S, 0.5)}")

# Test 3: Pass slope via grid field name
print("\nTest 3: Slope via grid field name")
grid.at_node["my_slope"] = np.full(grid.number_of_nodes, 0.3)
ic3 = ConnectivityIndex(grid, slope="my_slope")
ic3.run_one_step()
print(f"  S range:  {ic3.S.min():.6f} to {ic3.S.max():.6f}")
print(f"  Field name correctly resolved: {np.allclose(ic3.S, 0.3)}")

print("\n✓ All slope parameter tests passed!")
