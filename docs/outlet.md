# Outlet Workflow

## High-level API (recommended)

```python
from geomorphconn import run_connectivity_from_rasters

result = run_connectivity_from_rasters(
    dem="dem.tif",
    weight=["ndvi.tif", "rainfall.tif"],
    ic_mode="outlet",
    flow_director="DINF",
    fill_sinks=True,
    auto_project_to_utm=True,
)

result["dataset"]["IC"].rio.to_raster("IC_outlet.tif")
```

## Low-level component usage

```python
import numpy as np
import xarray as xr
import rioxarray
from landlab import RasterModelGrid
from geomorphconn.components import ConnectivityIndex

# Load inputs
Dem = xr.load_dataarray("dem.tif").squeeze(drop=True)
Ndvi = xr.load_dataarray("ndvi.tif").squeeze(drop=True)
Rain = xr.load_dataarray("rainfall.tif").squeeze(drop=True)

dem = Dem.values.astype(float)
ndvi = Ndvi.values.astype(float)
rainfall = Rain.values.astype(float)
dx = float(abs(Dem.rio.resolution()[0]))

grid = RasterModelGrid(dem.shape, xy_spacing=dx)
grid.add_field("topographic__elevation", np.flipud(dem).ravel(), at="node")

ic = ConnectivityIndex(
    grid,
    flow_director="DINF",
    ndvi=np.flipud(ndvi).ravel(),
    rainfall=np.flipud(rainfall).ravel(),
)
ic.run_one_step()
```

## GEE inputs

```python
from geomorphconn.gee import GEEFetcher

fetcher = GEEFetcher(
    bounds=(72.5, 28.0, 80.5, 32.0),
    dem_source="COPDEM30",
    rainfall_source="CHIRPS",
    ndvi_source="SENTINEL2",
    start_date="2020-06-01",
    end_date="2020-08-31",
    scale=30,
    gee_project="your-gee-project",
)

dem, ndvi, rainfall, profile = fetcher.fetch()
result = fetcher.fetch(return_xarray=True)
```

## GEE timeseries

```python
ts = fetcher.fetch_timeseries(resampling="monthly")
for p in ts["periods"]:
    print(p["label"], p["start_date"], p["end_date"])
```

Supported `resampling`: `monthly`, `seasonal`, `annual`.
