"""
Empirical sweep - WINDOWED approach using a downsampled or cropped subset of the data.
Tests 7 routingconfigurations on a smaller region for faster iteration.
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

def load_raster_windowed(path, window=None):
    """Load raster, optionally from a window, and return (data, profile)."""
    with rasterio.open(path) as src:
        if window is None:
            data = src.read(1, out_dtype=np.float32)
        else:
            data = src.read(1, window=window, out_dtype=np.float32)
        profile = src.profile.copy()
    return data, profile

def compute_metrics(computed, reference):
    """Compute correlation, RMSE, median abs error, and p95 ratios."""
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
    print("EMPIRICAL SWEEP: Landlab Routing Configurations vs Italy Reference")
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
    
    # Use a windowed subset for faster testing
    # Window: (row_offset, col_offset, height, width)
    # Using a 600x600 window in the middle of the 2835x3667 DEM
    window = rasterio.windows.Window(1500, 1200, 600, 600)
    
    # Load reference rasters (windowed)
    print(f"Loading reference rasters (windowed: 600x600)...")
    ref_data = {}
    for layer in ['Dup', 'Ddn', 'ACCfinal']:
        ref_path = ref_dir / f"{layer}.tif"
        if ref_path.exists():
            ref_data[layer], _ = load_raster_windowed(str(ref_path), window=window)
            print(f"  [OK] {layer}: shape {ref_data[layer].shape}")
        else:
            print(f"  [XX] {layer}: NOT FOUND")
    
    print()
    
    # Load DEM (windowed)
    print(f"Loading Italy DEM (windowed)...")
    dem, dem_profile = load_raster_windowed(str(dem_path), window=window)
    print(f"  DEM shape: {dem.shape}, dtype: {dem.dtype}")
    
    # Handle nodata values
    nodata = dem_profile.get('nodata', None)
    if nodata is not None:
        dem[dem == nodata] = np.nan
    
    # Convert to float64 for Landlab
    dem = dem.astype(np.float64)
    
    print()
    
    # Create grid
    print(f"Creating RasterModelGrid...")
    rows, cols = dem.shape
    xy_spacing = 1.0  # 1m DEM
    
    grid = RasterModelGrid((rows, cols), xy_spacing=xy_spacing)
    grid.add_field("topographic__elevation", dem, at="node")
    print(f"  Grid created: {grid.shape}, nodes={grid.number_of_nodes}")
    
    print()
    
    # Define sweep configurations
    configs = [
        {'flow_director': 'D8', 'fill_sinks': False, 'use_aspect_weighting': False, 'label': 'D8'},
        {'flow_director': 'D8', 'fill_sinks': True, 'use_aspect_weighting': False, 'label': 'D8+Fill'},
        {'flow_director': 'DINF', 'fill_sinks': False, 'use_aspect_weighting': False, 'label': 'DINF'},
        {'flow_director': 'DINF', 'fill_sinks': True, 'use_aspect_weighting': False, 'label': 'DINF+Fill'},
        {'flow_director': 'DINF', 'fill_sinks': True, 'use_aspect_weighting': True, 'label': 'DINF+Fill+Aspect'},
        {'flow_director': 'MFD', 'fill_sinks': True, 'use_aspect_weighting': False, 'label': 'MFD+Fill'},
        {'flow_director': 'MFD', 'fill_sinks': True, 'use_aspect_weighting': True, 'label': 'MFD+Fill+Aspect'},
    ]
    
    # Run sweep
    results = []
    
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Testing {config['label']}...")
        
        try:
            # Create fresh grid
            grid_test = RasterModelGrid((rows, cols), xy_spacing=xy_spacing)
            dem_copy = dem.copy()
            grid_test.add_field("topographic__elevation", dem_copy, at="node")
            
            # Initialize and run component
            ic = ConnectivityIndex(
                grid_test,
                flow_director=config['flow_director'],
                fill_sinks=config['fill_sinks'],
                use_aspect_weighting=config['use_aspect_weighting'],
            )
            
            ic.run_one_step()
            
            # Extract outputs (note: field names are prefixed with 'connectivity_index__')
            ic_out = grid_test.at_node['connectivity_index__IC'].reshape(dem.shape)
            dup_out = grid_test.at_node['connectivity_index__Dup'].reshape(dem.shape)
            ddn_out = grid_test.at_node['connectivity_index__Ddn'].reshape(dem.shape)
            acc_out = grid_test.at_node['connectivity_index__ACCfinal'].reshape(dem.shape)
            
            # Compute metrics
            result_row = {'config': config['label']}
            
            for layer, out in [('Dup', dup_out), ('Ddn', ddn_out), ('ACCfinal', acc_out)]:
                if layer in ref_data:
                    metrics = compute_metrics(out, ref_data[layer])
                    if metrics:
                        result_row[f'{layer}_corr'] = metrics['correlation']
                        result_row[f'{layer}_rmse_norm'] = metrics['rmse_normalized']
                        result_row[f'{layer}_mae_norm'] = metrics['mae_normalized']
                        result_row[f'{layer}_p95_ratio'] = metrics['p95_ratio']
                        print(f"    {layer}: corr={metrics['correlation']:.3f}, p95_ratio={metrics['p95_ratio']:.3f}")
                    else:
                        print(f"    {layer}: INVALID")
            
            results.append(result_row)
            print()
        
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Compute composite scores
    print("=" * 80)
    print("RANKING BY COMPOSITE SCORE")
    print("=" * 80)
    print()
    
    # Score: prioritize Ddn and Dup
    for row in results:
        score = 0
        if 'Ddn_corr' in row:
            score += row['Ddn_corr'] * 0.4
        if 'Dup_corr' in row:
            score += row['Dup_corr'] * 0.4
        if 'ACCfinal_corr' in row:
            score += row['ACCfinal_corr'] * 0.2
        
        # Penalize p95 ratios far from 1.0
        p95_penalties = 0
        for layer in ['Dup', 'Ddn', 'ACCfinal']:
            if f'{layer}_p95_ratio' in row:
                p95_penalties += abs(1.0 - row[f'{layer}_p95_ratio']) * 0.05
        
        row['score'] = score - p95_penalties
    
    results_sorted = sorted(results, key=lambda x: x.get('score', -np.inf), reverse=True)
    
    for rank, row in enumerate(results_sorted, 1):
        print(f"{rank}. {row['config']:20s} | Score: {row.get('score', np.nan):7.3f}")
        print(f"   Ddn={row.get('Ddn_corr', np.nan):6.3f}, Dup={row.get('Dup_corr', np.nan):6.3f}, ACC={row.get('ACCfinal_corr', np.nan):6.3f}")
        print()
    
    # Save results
    output_file = Path("sweep_results_windowed.json")
    with open(output_file, 'w') as f:
        json.dump(results_sorted, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
