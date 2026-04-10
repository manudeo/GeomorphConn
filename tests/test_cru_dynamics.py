"""
Unit tests for geomorphconn.analysis.cru_dynamics module.

Tests cover CRU classification logic, metadata preservation, and error handling.
"""

import pytest
import numpy as np
import xarray as xr
from geomorphconn.analysis import classify_dynamic_crus


class TestCRUClassification:
    """Test suite for CRU classification logic."""
    
    def test_emerging_hotspot(self):
        """Pixel that becomes hot recently should classify as Emerging Hotspot (1)."""
        hotspots = xr.DataArray(
            np.array([
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
            ]),
            coords={'time': np.arange(5), 'y': np.arange(2), 'x': np.arange(2)},
            dims=['time', 'y', 'x'],
        )
        
        result = classify_dynamic_crus(hotspots, recent_window=2)
        assert result.values[0, 0] == 1
    
    def test_persistent_hotspot(self):
        """Pixel hot in 80%+ of timesteps should classify as Persistent (4)."""
        data = np.ones((10, 2, 2))
        data[0, :, :] = 0
        data[1, :, :] = 0
        hotspots = xr.DataArray(
            data,
            coords={'time': np.arange(10), 'y': np.arange(2), 'x': np.arange(2)},
            dims=['time', 'y', 'x'],
        )
        
        result = classify_dynamic_crus(hotspots, recent_window=2, persistence_fraction=0.8)
        assert np.all(result.values == 4)
    
    def test_no_pattern(self):
        """All zeros should result in 0 (no pattern) for all cells."""
        hotspots = xr.DataArray(
            np.zeros((5, 3, 3)),
            coords={'time': np.arange(5), 'y': np.arange(3), 'x': np.arange(3)},
            dims=['time', 'y', 'x'],
        )
        
        result = classify_dynamic_crus(hotspots)
        assert np.all(result.values == 0)
    
    def test_constant_hotspot(self):
        """Constant hot signal should be Persistent Hotspot (4)."""
        hotspots = xr.DataArray(
            np.ones((8, 2, 2)),
            coords={'time': np.arange(8), 'y': np.arange(2), 'x': np.arange(2)},
            dims=['time', 'y', 'x'],
        )
        
        result = classify_dynamic_crus(hotspots, recent_window=2, persistence_fraction=0.8)
        assert np.all(result.values == 4)
    
    def test_output_dtype(self):
        """Output should be int8 dtype."""
        hotspots = xr.DataArray(
            np.random.choice([-1, 0, 1], size=(5, 3, 3)),
            coords={'time': np.arange(5), 'y': np.arange(3), 'x': np.arange(3)},
            dims=['time', 'y', 'x'],
        )
        
        result = classify_dynamic_crus(hotspots)
        assert result.dtype == np.int8
    
    def test_output_value_range(self):
        """Output should only contain values in [-6, +6]."""
        hotspots = xr.DataArray(
            np.random.choice([-1, 0, 1], size=(10, 5, 5)),
            coords={'time': np.arange(10), 'y': np.arange(5), 'x': np.arange(5)},
            dims=['time', 'y', 'x'],
        )
        
        result = classify_dynamic_crus(hotspots)
        unique_vals = np.unique(result.values)
        assert np.all(unique_vals >= -6) and np.all(unique_vals <= 6)
    
    def test_metadata_preserved(self):
        """Output DataArray should include all metadata attributes."""
        hotspots = xr.DataArray(
            np.random.choice([-1, 0, 1], size=(5, 3, 3)),
            coords={'time': np.arange(5), 'y': np.arange(3), 'x': np.arange(3)},
            dims=['time', 'y', 'x'],
        )
        
        result = classify_dynamic_crus(
            hotspots,
            recent_window=2,
            early_window=2,
            emergence_threshold=0.5,
            persistence_fraction=0.8,
            attribution_tags={'methodology': 'Singh et al. 2018'}
        )
        
        assert result.attrs['recent_window'] == 2
        assert result.attrs['early_window'] == 2
        assert result.attrs['total_timesteps'] == 5
        assert result.attrs['methodology'] == 'Singh et al. 2018'
        assert 'cru_classes' in result.attrs


class TestInputValidation:
    """Test error handling and input validation."""
    
    def test_non_dataarray_input(self):
        """Should raise TypeError if input is not xarray.DataArray."""
        with pytest.raises(TypeError):
            classify_dynamic_crus(np.array([1, 2, 3]))
    
    def test_missing_time_dimension(self):
        """Should raise ValueError if no time dimension found."""
        with pytest.raises(ValueError, match="Could not identify time dimension"):
            no_time = xr.DataArray(
                np.zeros((3, 3)),
                coords={'y': np.arange(3), 'x': np.arange(3)},
                dims=['y', 'x'],
            )
            classify_dynamic_crus(no_time)
    
    def test_single_timestep(self):
        """Should raise ValueError if only 1 timestep provided."""
        with pytest.raises(ValueError, match=">=2 timesteps"):
            single_step = xr.DataArray(
                np.zeros((1, 3, 3)),
                coords={'time': [0], 'y': np.arange(3), 'x': np.arange(3)},
                dims=['time', 'y', 'x'],
            )
            classify_dynamic_crus(single_step)


class TestEdgeCases:
    """Test boundary and edge case behaviors."""
    
    def test_very_small_grid(self):
        """Should handle minimal grid sizes (1x1)."""
        hotspots = xr.DataArray(
            np.ones((5, 1, 1)),
            coords={'time': np.arange(5), 'y': [0], 'x': [0]},
            dims=['time', 'y', 'x'],
        )
        
        result = classify_dynamic_crus(hotspots)
        assert result.shape == (1, 1)
        assert result.values[0, 0] == 4

    def test_early_window_defaults_to_recent_window(self):
        hotspots = xr.DataArray(
            np.random.choice([-1, 0, 1], size=(6, 2, 2)),
            coords={'time': np.arange(6), 'y': np.arange(2), 'x': np.arange(2)},
            dims=['time', 'y', 'x'],
        )

        result = classify_dynamic_crus(hotspots, recent_window=3, early_window=None)
        assert result.attrs['early_window'] == 3

    def test_window_size_exceeds_timesteps_raises(self):
        hotspots = xr.DataArray(
            np.random.choice([-1, 0, 1], size=(4, 2, 2)),
            coords={'time': np.arange(4), 'y': np.arange(2), 'x': np.arange(2)},
            dims=['time', 'y', 'x'],
        )
        with pytest.raises(ValueError, match="Window sizes"):
            classify_dynamic_crus(hotspots, recent_window=5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
