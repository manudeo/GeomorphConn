"""
Empirical sweep - SIMPLIFIED TEST with just ONE configuration to debug.
"""

import numpy as np
import rasterio
from pathlib import Path
from scipy import stats
import sys
import json
from datetime import datetime
import os

# Ensure we can import geomorphconn
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from geomorphconn.components.connectivity_index import ConnectivityIndex
from landlab import RasterModelGrid

def load_raster(path):
    """Load raster and return (data, profile)."""
    with rasterio.open(path) as src:
        data = src.read(1, out_dtype=np.float32)
        profile = src.profile
    return data, profile

def compute_metrics(computed, reference, nodata_mask=None):
    """Compute correlation, RMSE, median abs error, and p95 ratios."""
    if nodata_mask is not None:
        valid = ~nodata_mask
    else:
        valid = np.isfinite(computed) & np.isfinite(reference)
    
    if valid.sum() < 2:
        return None
    
    c_valid = computed[valid]
    r_valid = reference[valid]
    
    # Correlation
    corr, pval = stats.spearmanr(c_valid, r_valid)
    
    # RMSE (normalized by reference std)
    rmse = np.sqrt(np.mean((c_valid - r_valid) ** 2))
    ref_std = np.std(r_valid)
    rmse_norm = rmse / ref_std if ref_std > 0 else np.inf
    
    # Median absolute error
    mae = np.median(np.abs(c_valid - r_valid))
    mae_norm = mae / np.median(np.abs(r_valid)) if np.median(np.abs(r_valid)) > 0 else np.inf
    
    # Percentile ratios
    p95_comp = np.percentile(c_valid, 95)
    p95_ref = np.percentile(r_valid, 95)
    p95_ratio = p95_comp / p95_ref if p95_ref > 0 else 0
    
    return {
        'correlation': float(corr),
        'p_value': float(pval),
        'rmse_normalized': float(rmse_norm),
        'mae_normalized': float(mae_norm),
        'p95_ratio': float(p95_ratio),
        'n_valid': int(valid.sum()),
    }

def main():
    print("=" * 80)
    print("EMPIRICAL SWEEP (SIMPLIFIED): Landlab vs Italy Reference")
    print("=" * 80)
    print()
    
    # Paths
    dem_path = Path(r"e:\SideResearch\softwares\IndexOfConnectivity\DTM_2013_basin_1m\DTM_2013_basin_1mfel.tif")
    ref_dir = Path(r"e:\SideResearch\softwares\IndexOfConnectivity\DTM_2013_basin_1m\temp2")
    
    print(f"Loading DEM from: {dem_path}")
    print(f"Reference directory: {ref_dir}")
    print()
    
    if not dem_path.exists():
        print(f"ERROR: DEM not found")
        return
    
    # Load reference rasters
    print("Loading reference rasters...")
    ref_data = {}
    for layer in ['Dup', 'Ddn', 'ACCfinal']:
        ref_path = ref_dir / f"{layer}.tif"
        if ref_path.exists():
            ref_data[layer], _ = load_raster(str(ref_path))
            print(f"  ✓ {layer}: shape {ref_data[layer].shape}")
        else:
            print(f"  ✗ {layer}: NOT FOUND")
    
    print()
    
    # Load DEM
    print("Loading Italy DEM...")
    dem, dem_profile = load_raster(str(dem_path))
    print(f"  DEM shape: {dem.shape}, dtype: {dem.dtype}")
    print(f"  DEM min/max: {np.nanmin(dem):.1f} / {np.nanmax(dem):.1f}")
    
    # Handle nodata values - replace with NaN so they're ignored
    nodata = dem_profile.get('nodata', None)
    if nodata is not None:
        dem[dem == nodata] = np.nan
        print(f"  Replaced nodata ({nodata}) with NaN")
    
    # Convert to float64 for Landlab
    dem = dem.astype(np.float64)
    print(f"  Converted to float64")
    
    # Create grid
    print("\nCreating RasterModelGrid...")
    rows, cols = dem.shape
    xy_spacing = 1.0  # 1m DEM
    
    print(f"  Grid dimensions: {rows} x {cols}")
    print(f"  XY spacing: {xy_spacing}")
    
    grid = RasterModelGrid((rows, cols), xy_spacing=xy_spacing)
    print(f"  Grid created: {grid.shape}, nodes={grid.number_of_nodes}")
    
    # Convert DEM to correct shape and add to grid
    dem_2d = dem.reshape((rows, cols))
    grid.add_field("topographic__elevation", dem_2d, at="node")
    print(f"  Elevation field added")
    
    print()
    
    # Test ONE configuration: D8 with no fill, no aspect
    print("[1/1] Testing D8 (no fill, no aspect)...")
    
    try:
        ic = ConnectivityIndex(
            grid,
            flow_director='D8',
            fill_sinks=False,
            use_aspect_weighting=False,
        )
        print(f"  Component initialized")
        
        ic.run_one_step()
        print(f"  Component run_one_step() completed")
        
        # Extract outputs
        ic_out = grid.at_node['index_of_connectivity'].reshape(dem.shape)
        dup_out = grid.at_node['distance_upstream'].reshape(dem.shape)
        ddn_out = grid.at_node['distance_downstream'].reshape(dem.shape)
        acc_out = grid.at_node['drainage_area'].reshape(dem.shape)
        
        print(f"  Outputs extracted")
        print(f"    IC: min={np.nanmin(ic_out):.3f}, max={np.nanmax(ic_out):.3f}")
        print(f"    Dup: min={np.nanmin(dup_out):.1f}, max={np.nanmax(dup_out):.1f}")
        print(f"    Ddn: min={np.nanmin(ddn_out):.1f}, max={np.nanmax(ddn_out):.1f}")
        print(f"    ACC: min={np.nanmin(acc_out):.1f}, max={np.nanmax(acc_out):.1f}")
        
        # Compute metrics
        print()
        print("  Metrics vs ArcGIS Reference:")
        for layer, out in [('Dup', dup_out), ('Ddn', ddn_out), ('ACCfinal', acc_out)]:
            if layer in ref_data:
                metrics = compute_metrics(out, ref_data[layer])
                if metrics:
                    print(f"    {layer}: corr={metrics['correlation']:.3f}, p95_ratio={metrics['p95_ratio']:.3f}")
                else:
                    print(f"    {layer}: INVALID")
    
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
