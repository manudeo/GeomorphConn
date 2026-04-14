# Target Workflow

Target-mode IC routes connectivity toward a stream/river/lake target instead of only the basin outlet.

## High-level API target mode

Choose one target definition method:

- `stream_threshold`
- `target_vector`
- `target_nodes` (advanced precomputed node IDs)

### Stream-threshold target

```python
from geomorphconn import run_connectivity_from_rasters

result = run_connectivity_from_rasters(
    dem="dem.tif",
    weight=["ndvi.tif", "rainfall.tif"],
    ic_mode="target",
    stream_threshold=1000,
    flow_director="DINF",
)
```

### Vector target

```python
result = run_connectivity_from_rasters(
    dem="dem.tif",
    weight=["ndvi.tif", "rainfall.tif"],
    ic_mode="target",
    target_vector="river.shp",
    target_all_touched=True,
    target_buffer_m=0.0,
)
```

### Explicit target nodes

```python
import numpy as np

result = run_connectivity_from_rasters(
    dem="dem.tif",
    weight=["ndvi.tif", "rainfall.tif"],
    ic_mode="target",
    target_nodes=np.array([10, 11, 12], dtype=np.int64),
)
```

## `target_all_touched` meaning

- `True`: any raster cell touched by geometry is included.
- `False`: only center-based rasterization inclusion is used.

Use `True` for thin river lines; use `False` for strict masks.

## Rasterization helper

```python
from geomorphconn.utils import rasterize_targets

nodes = rasterize_targets(
    source="river.shp",
    grid=grid,
    dem_transform=dem_transform,
    dem_crs=dem_crs,
    all_touched=True,
    buffer_m=0.0,
)
```
