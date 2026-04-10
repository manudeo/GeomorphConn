# GeomorphConn

[![PyPI version](https://badge.fury.io/py/geomorphconn.svg)](https://badge.fury.io/py/geomorphconn)
[![Tests](https://github.com/manudeo/GeomorphConn/actions/workflows/tests.yml/badge.svg)](https://github.com/manudeo/GeomorphConn/actions)
[![DOI](https://joss.theoj.org/papers/placeholder/badge.svg)](https://joss.theoj.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GeomorphConn** is an open-source Python package that implements the
[Index of Connectivity (IC)](https://doi.org/10.1016/j.geomorph.2012.05.010)
(Cavalli et al., 2013) as a [Landlab](https://landlab.readthedocs.io)-type workflow,
extended with NDVI- and rainfall-based hydrological weights (Dubey et al., submitted). It also provides a
Google Earth Engine (GEE) data-fetching module so that all required inputs — DEM,
NDVI, and rainfall — can be retrieved directly from the cloud for any catchment on
Earth.

---

## Key features

| Feature | Detail |
|---|---|
| **IC toward outlet** | Standard Cavalli et al. (2013) formulation |
| **IC toward target** | Compute IC relative to a river/lake shapefile (any geopandas-readable format) |
| **Hydrological weights** | W = f(RF_norm, NDVI-C-factor) — extends purely topographic/land-cover weighting |
| **Flow direction options** | D8 (steepest), D-infinity, Multiple-Flow-Direction (MFD) via Landlab |
| **GEE data fetching** | DEM: SRTM / CopDEM-30 / MERIT-DEM; Rainfall: CHIRPS / ERA5 / PERSIANN; NDVI: Landsat-8/9 / Sentinel-2 |
| **ArcGIS tools** | Identical workflows provided as an ArcGIS Pro toolbox (`arcgis_tools/`) |
| **Speed** | Optional `numba` JIT compilation for O(N) traversal loops |

---

## Installation

See **[INSTALLATION.md](INSTALLATION.md)** for full instructions including
virtual-environment setup, optional extras, system requirements, and ArcGIS
Pro toolbox setup.

**Quick start (from source):**
```bash
git clone https://github.com/manudeo/GeomorphConn.git
cd GeomorphConn
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -e ".[gui]"
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common install, GUI, alignment,
and target-vector issues.

---

## CLI usage

Show welcome/about information:

```bash
geomorphconn welcome
```

Show installed CLI version:

```bash
geomorphconn --version
```

After installation, use the console command:

```bash
geomorphconn run \
    --dem dem.tif \
    --ndvi ndvi.tif \
    --rainfall rainfall.tif \
    --weight-factors rainfall ndvi roughness \
    --weight-combine mean \
    --flow-director DINF \
    --reference-grid dem \
    --use-aspect-weighting \
    --outputs IC Dup Ddn \
    --out-dir outputs
```

Current CLI mode uses local raster inputs (`--dem`, `--ndvi`, `--rainfall`).
Direct GEE-fetch execution from CLI is planned as future work.

If input rasters differ in resolution/extent/shape, CLI now aligns them internally
using `rioxarray` `reproject_match`, with `--reference-grid` controlling the target
grid (`dem` by default, or `ndvi` / `rainfall`).

`--weight-factors` lets users choose one, two, or all three factors
(`rainfall`, `ndvi`, `roughness`).
Examples:

- Roughness only (DEM-only):

```bash
geomorphconn run --dem dem.tif --weight-factors roughness --outputs IC
```

- Rainfall only:

```bash
geomorphconn run --dem dem.tif --rainfall rainfall.tif --weight-factors rainfall --outputs IC
```

- NDVI + rainfall (legacy-equivalent):

```bash
geomorphconn run --dem dem.tif --ndvi ndvi.tif --rainfall rainfall.tif --weight-factors ndvi rainfall
```

- User-supplied weight raster (bypass internal factor calculation):

```bash
geomorphconn run --dem dem.tif --weight-raster weight_w.tif --outputs IC W
```

Optional target mode:

```bash
geomorphconn run \
    --dem dem.tif \
    --ndvi ndvi.tif \
    --rainfall rainfall.tif \
    --target-vector river.shp \
    --all-touched \
    --target-buffer 5 \
    --outputs IC
```

`--target-vector` accepts any geopandas-readable vector path. For narrow line
targets, a small automatic buffer is applied if `--target-buffer` is left at
`0`, so the target rasterizes robustly on the DEM grid.

---

## GUI options

Current GUI backend:

- Streamlit app launcher:

```bash
geomorphconn gui --backend streamlit
```

This opens a local web UI where you can upload DEM, NDVI, and rainfall GeoTIFFs,
choose flow settings, and run IC interactively.

The GUI opens with a welcome/about panel summarizing capabilities, references,
and authors.

GUI upload limit is configured to **4 GB per file** via project Streamlit config
(`.streamlit/config.toml`, `server.maxUploadSize = 4096`).

GUI includes an **Auto-align rasters** option with the same reference-grid logic
(`dem` default) to handle mismatched raster grids before IC computation.

GUI also exposes:

- explicit `Outlet` vs `Target` IC mode selection
- selectable weight factors (`rainfall`, `ndvi`, `roughness`)
- weight combine mode
- local target vector path input with native file picker; a `.shp` path works directly when sidecar files are present beside it
- optional uploaded target vector input (GeoJSON/GPKG or zipped shapefile) for IC toward target mode
- target buffer and target rasterization controls
- output directory text field
- one-click output directory picker (native folder dialog)
- output name affix with mode selector (suffix/prefix; default is suffix)
- selectable output layers to save (IC, Dup, Ddn, W, S, Wmean, Smean)
- optional user-supplied weight (W) raster mode that bypasses internal
    factor-based weight calculation

Current GUI mode also uses uploaded local rasters.
Direct GEE-driven GUI workflow is planned as future work.

### Option help (CLI and GUI)

Use this quick reference to interpret the main checkbox/toggle options.

- `Use supplied weight raster (W)` (GUI) / `--weight-raster` (CLI): use a precomputed W raster and skip internal NDVI/rainfall/roughness factor computation. In GUI, NDVI and rainfall inputs are hidden in this mode.
- `Use degree approximation` (GUI) / `--use-degree-approx` (CLI): if enabled, use $S = \theta^\circ/100$; if disabled, use $S = \tan(\theta)$.
- `Use aspect weighting` (GUI) / `--use-aspect-weighting` (CLI): enable TauDEM-style partition weighting for multi-receiver upstream accumulation.
- `Auto-align rasters` (GUI) / `--auto-reproject` (CLI): align all rasters to a selected reference grid before computation.
- `Fill sinks before routing (ArcGIS-like)` (GUI): explicitly fill depressions before routing (`Fill -> FlowDirection -> FlowAccumulation`) to better match ArcGIS workflows.
- `Reference grid` (GUI/CLI): choose which raster grid (`dem`, `ndvi`, `rainfall`, or `weight`) is used as the alignment target.
- `Invert roughness (TRI)` (GUI) / `--roughness-invert` (CLI): invert TRI effect so rougher terrain reduces W instead of increasing it.
- `w_min` / `w_max` (GUI): lower and upper clamps for weight scaling; GUI now accepts values to 5 decimal places.
- `IC mode` (GUI): choose `Outlet` for standard basin-outlet IC or `Target` to route IC toward a supplied vector target.
- `Target vector path` (GUI): local vector file path for target mode. A `.shp` path is sufficient when its `.dbf`, `.shx`, and other sidecar files are present in the same folder.
- `Target vector` (GUI upload) / `--target-vector` (CLI): use a vector feature instead of the basin outlet as the routing target. Any geopandas-readable format is supported in CLI; GUI supports direct local vector paths plus uploaded GeoJSON/GPKG and zipped shapefiles.
- `Target buffer` (GUI) / `--target-buffer` (CLI): optional pre-rasterization buffer; if left at `0`, line targets are automatically buffered by about half a cell.
- `Target rasterization: all touched` (GUI) / `--all-touched` (CLI): include any raster cell touched by the target geometry.
- Target masking behavior: in target mode, pixels that belong to the target feature are written as `NaN` in IC outputs (treated as terminal mask cells).
- `Save selected outputs to disk` (GUI): write selected output layers as GeoTIFFs to the chosen output directory.
- `Output affix mode` (GUI): choose whether output name affix is used as suffix (default) or prefix.
- Progress feedback: both CLI and GUI now show stage-based run progress (load/preprocess, compute, save).

GUI output layer meanings:

- `IC`: final Index of Connectivity.
- `Dup`: upstream connectivity component.
- `Ddn`: downstream impedance component.
- `W`: effective weight raster used in the run.
- `S`: slope factor used in the run.
- `Wmean`: mean upstream weight over the contributing area.
- `Smean`: mean upstream slope factor over the contributing area.

### Future TODOs (planned)

- Add `geomorphconn gee fetch ...` command for CLI-native GEE retrieval.
- Add `geomorphconn run --from-gee ...` pipeline (fetch + compute in one command).
- Add GUI controls for bounds/date/source/project to run GEE fetch and IC in one workflow.
- Add a desktop/clickable GUI launcher (double-click icon/shortcut, no terminal required).
- Add optional SCI-inspired mobility mode (rainfall + soil stability + land use + ruggedness; cf. Zingaro et al., 2019, https://doi.org/10.1016/j.scitotenv.2019.03.461) as an advanced workflow, while keeping the default workflow minimal-data.
- Add optional IHC-inspired event mode (runoff/CN and antecedent-rainfall weighting with RS-style impedance; cf. Zanandrea et al., 2021, https://doi.org/10.1016/j.catena.2021.105380) as an advanced workflow, while keeping the default workflow minimal-data.
- Add explicit SCI and IHC citations at each discussion point in README and manuscript so method references are self-explanatory for new readers.
- Support reading all gridded formats supported by xarray where possible, with clear warnings/errors when extra backend dependencies are required.
- Add time-series IC mode in GUI and CLI: accept time-varying grids (e.g., NetCDF); if data are single-time or 2D, treat them as static inputs across all requested timesteps.
- Add SedConnect as a [Landlab](https://landlab.readthedocs.io) component. 

---

## Quick start

### IC toward outlet — from a GeoTIFF
```python
import numpy as np
import rasterio
from landlab import RasterModelGrid
from geomorphconn.components import ConnectivityIndex

# Load inputs
with rasterio.open("dem.tif") as src:
    dem = src.read(1).astype(float)
    dx  = src.res[0]

with rasterio.open("ndvi.tif")     as src: ndvi     = src.read(1).astype(float)
with rasterio.open("rainfall.tif") as src: rainfall = src.read(1).astype(float)

# Build Landlab grid (flipud: GeoTIFF is top-down, Landlab is bottom-up)
grid = RasterModelGrid(dem.shape, xy_spacing=dx)
grid.add_field("topographic__elevation", np.flipud(dem).ravel(), at="node")

# Run IC
ic = ConnectivityIndex(
    grid,
    flow_director="FlowDirectorDINF",   # D8 | FlowDirectorDINF | FlowDirectorMFD
    ndvi=np.flipud(ndvi).ravel(),
    rainfall=np.flipud(rainfall).ravel(),
)
ic.run_one_step()

IC_map = np.flipud(grid.at_node["connectivity_index__IC"].reshape(dem.shape))
```

### Optional: TauDEM-style aspect weighting (opt-in)
```python
ic = ConnectivityIndex(
    grid,
    flow_director="FlowDirectorDINF",
    ndvi=np.flipud(ndvi).ravel(),
    rainfall=np.flipud(rainfall).ravel(),
    use_aspect_weighting=True,  # default is False
)
ic.run_one_step()
```

`use_aspect_weighting` is disabled by default to preserve existing behavior.
When enabled, it only affects multi-receiver upstream accumulation
(`FlowDirectorDINF` / `FlowDirectorMFD`) in `D_up`; `D_dn` remains D8-based.

### IC toward a river shapefile
```python
from geomorphconn.utils import rasterize_targets

target_nodes = rasterize_targets("river.shp", grid, dem_transform=src.transform)

ic = ConnectivityIndex(
    grid,
    ndvi=np.flipud(ndvi).ravel(),
    rainfall=np.flipud(rainfall).ravel(),
    target_nodes=target_nodes,
)
ic.run_one_step()
```

### Fetch all inputs from Google Earth Engine
```python
from geomorphconn.gee import GEEFetcher

fetcher = GEEFetcher(
    bounds=(72.5, 28.0, 80.5, 32.0),     # (lon_min, lat_min, lon_max, lat_max)
    # or: bounds="catchment.shp"
    dem_source="COPDEM30",                # SRTM | COPDEM30 | MERIT
    rainfall_source="CHIRPS",             # CHIRPS | ERA5 | PERSIANN
    ndvi_source="SENTINEL2",              # SENTINEL2 | LANDSAT8 | LANDSAT9
    start_date="2020-06-01",
    end_date="2020-08-31",                # median NDVI over this period
    scale=30,                             # output resolution in metres
    gee_project="your-gee-project",
)
dem, ndvi, rainfall, profile = fetcher.fetch()
```

### Google Earth Engine authentication (local setup)

`GeomorphConn` uses your local Earth Engine credentials. Authenticate once in your Python environment:

```bash
earthengine authenticate
earthengine set_project drylands-aberuni
```

You can also authenticate from Python:

```python
import ee
ee.Authenticate()
ee.Initialize(project="drylands-aberuni")
```

In notebooks, set:

```python
GEE_PROJECT = "drylands-aberuni"
```

---

## Weight scenarios

`GeomorphConn` supports DEM-only and mixed weighting through the `geomorphconn.weights` builders/presets.

| Scenario name | Includes | Typical builder/preset |
|---|---|---|
| `roughness_only` | DEM roughness only | `preset_roughness_only(grid)` |
| `rainfall_only` | Rainfall only | `WeightBuilder().add(RainfallWeight(rainfall_nodes))` |
| `ndvi_only` | NDVI only | `WeightBuilder().add(NDVIWeight(ndvi_nodes))` |
| `ndvi_rainfall` | NDVI + Rainfall | `preset_rainfall_ndvi(rainfall_nodes, ndvi_nodes)` |
| `ndvi_rainfall_roughness` | NDVI + Rainfall + DEM roughness | `preset_rainfall_ndvi_roughness(rainfall_nodes, ndvi_nodes, grid)` |

These five scenarios are demonstrated in:

- `notebooks/01_IC_outlet_GEE_demo.ipynb` (IC toward outlet)
- `notebooks/02_IC_target_demo.ipynb` (IC toward target feature)

---

## Repository structure

```
geomorphconn/
├── geomorphconn/                   # new canonical import path
│   ├── components/
│   │   └── connectivity_index.py   ← Landlab Component (core algorithm)
│   ├── gee/
│   │   └── fetcher.py              ← GEE/xee data fetcher
│   └── utils/
│       └── target.py               ← Target shapefile rasterization
├── notebooks/
│   ├── 01_IC_outlet_GEE_demo.ipynb ← Full workflow: GEE fetch → IC outlet
│   └── 02_IC_target_demo.ipynb     ← IC toward river / lake target
├── arcgis_tools/
│   ├── ConnectivityTools.atbx       ← ArcGIS Pro toolbox (outlet + target)
│   └── README.md
├── paper/
│   ├── paper.md                    ← JOSS manuscript
│   └── paper.bib
└── tests/
    └── test_connectivity_index.py
```

---

## ArcGIS tools

For users without a Python/Jupyter workflow, identical IC calculations are provided
as an ArcGIS Pro toolbox in `arcgis_tools/`. These require ArcGIS Pro
with Spatial Analyst, 3D Analyst, and Image Analyst licences. See
[`arcgis_tools/README.md`](arcgis_tools/README.md) for usage instructions.

---

## Citation

If you use this software, please cite:

> Singh, M., Cavalli, M. & Crema, S. (2026). GeomorphConn: A Python Package for
> Hydrologically-Weighted Index of Connectivity. *Journal of Open Source
> Software*. https://doi.org/10.21105/joss.XXXXX (TO BE SUBMITTED)

And the original IC formulation:

> Cavalli, M., Trevisani, S., Comiti, F., & Marchi, L. (2013). Geomorphometric
> assessment of spatial sediment connectivity in small Alpine catchments.
> *Geomorphology*, 188, 31–41. https://doi.org/10.1016/j.geomorph.2012.05.010

And the original IC software (SedInConnect):

> Crema, S. & Cavalli, M. (2018). SedInConnect: a stand-alone, free and open
> source tool for the assessment of sediment connectivity. *Computers &
> Geosciences*, 111, 39–45. https://doi.org/10.1016/j.cageo.2017.10.009

And the NDVI/rainfall hydrological-weight extension used in GeomorphConn:

> Dubey, A., Singh, M., & Jain, V. (submitted). Hydrologically weighted
> extension of the Index of Connectivity using NDVI and rainfall forcings.

And a related SCI-based connectivity approach:

> Zingaro, M., Refice, A., Giachetta, E., D'Addabbo, A., Lovergine, F., De
> Pasquale, V., Pepe, G., Brandolini, P., Cevasco, A., & Capolongo, D. (2019).
> Sediment mobility and connectivity in a catchment: A new mapping approach.
> *Science of The Total Environment*, 672, 763-775.
> https://doi.org/10.1016/j.scitotenv.2019.03.461

Related connectivity publications led by M. Singh:

> Singh, M., Tandon, S. K., & Sinha, R. (2017). Assessment of connectivity in a
> water-stressed wetland (Kaabar Tal) of Kosi-Gandak interfan, north Bihar
> Plains, India. *Earth Surface Processes and Landforms*, 42(12), 1982-1996.
> https://doi.org/10.1002/esp.4156

> Singh, M., & Sinha, R. (2019). Evaluating dynamic hydrological connectivity of
> a floodplain wetland in North Bihar, India using geostatistical methods.
> *Science of The Total Environment*, 651, 2473-2488.
> https://doi.org/10.1016/j.scitotenv.2018.10.139

> Singh, M., Sinha, R., & Tandon, S. K. (2020). Geomorphic connectivity and its
> application for understanding landscape complexities: a focus on the
> hydro-geomorphic systems of India. *Earth Surface Processes and Landforms*,
> 46(1), 110-130. https://doi.org/10.1002/esp.4945

> Singh, M., Sinha, R., Mishra, A., & Babu, S. (2022). Wetlandscape
> (dis)connectivity and fragmentation in a large wetland (Haiderpur) in west
> Ganga plains, India. *Earth Surface Processes and Landforms*, 47(8), 1872-1887.
> https://doi.org/10.1002/esp.5352

> Singh, M., & Sinha, R. (2022). Integrating hydrological connectivity in a
> process-response framework for restoration and monitoring prioritisation of
> floodplain wetlands in the Ramganga Basin, India. *Water*, 14(21), 3520.
> https://doi.org/10.3390/w14213520

---

## Contributing

Contributions welcome — please open an issue or pull request on GitHub.

---

## Acknowledgements

M. Singh is supported by the Royal Society Newton International Fellowship
(NIF\R1\232344) and was previously an Alexander von Humboldt Research Fellow
at the University of Potsdam. The IC methodology was developed by Marco Cavalli and his colleagues (CNR-IRPI, Italy). 
