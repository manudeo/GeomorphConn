---
title: 'GeomorphConn: A Python Package for Hydrologically-Weighted Index of Connectivity with Google Earth Engine Data Fetching'
tags:
  - Python
  - hydrology
  - geomorphology
  - sediment connectivity
  - Landlab
  - Google Earth Engine
  - remote sensing
  - NDVI
  - rainfall
authors:
  - name: Manudeo Singh
    orcid: 0000-0002-3511-8362
    corresponding: true
    affiliation: "1, 2"
  - name: Marco Cavalli
    orcid: 0000-0001-5937-454X
    affiliation: 3
  - name: Stefano Crema
    orcid: 0000-0001-8828-3129
    affiliation: 3
affiliations:
  - name: >
      Department of Geography and Earth Sciences,
      Aberystwyth University, Aberystwyth, UK
    index: 1
  - name: >
      Institute of Geosciences, University of Potsdam, Potsdam, Germany
      (Alexander von Humboldt Research Fellow)
    index: 2
  - name: >
      Research Institute for Geo-Hydrological Protection (IRPI),
      National Research Council (CNR), Padova, Italy
    index: 3
date: 3 April 2026
bibliography: paper.bib
---

# Summary

Sediment connectivity — the degree to which a landscape facilitates or impedes
the transfer of sediment from source areas to receiving water bodies — is a
foundational concept in catchment geomorphology [@Bracken2015]. The Index of
Connectivity (IC) of @Cavalli2013 provides a spatially explicit, raster-based
quantification of connectivity that has been widely adopted in studies of
sediment budgets, flood routing, and land degradation across diverse landscapes
[@Heckmann2018; @Cantreul2018; @Borselli2008]. This package also builds on
application-oriented connectivity research led by the first author across
floodplain and wetland systems, spanning geomorphic, hydrological, and
management-focused analyses [@Singh2017; @Singh2019; @Singh2020; @Singh2022a;
@Singh2022b].

`GeomorphConn` is an open-source Python package that implements the IC as a
formal [Landlab](https://landlab.readthedocs.io) component [@Hobley2017;
@Barnhart2020], extended with two physically motivated hydrological weighting
terms: (1) normalised rainfall, and (2) an NDVI-based C-factor proxy for
vegetation-mediated erosion resistance. These extensions make the IC explicitly
responsive to hydrological forcing rather than solely to topographic structure
and land cover. A companion Google Earth Engine (GEE) data-fetching module,
built on the `xee` [@xee2023] backend, allows all three required inputs — DEM,
NDVI, and rainfall — to be retrieved at any spatial scale directly from the
cloud for any catchment on Earth. For users who work in ArcGIS Pro,
functionally identical ModelBuilder scripts are provided in the repository. The
package complements empirical connectivity assessments spanning floodplain and
wetland systems [@Singh2020b; @Singh2021a; @Singh2021b], extending the IC
framework to diverse hydromorphic and semi-arid contexts. The software is
available through a Python API, a command-line interface, and a Streamlit GUI
for interactive use.

# Statement of Need

The original IC formulation of @Cavalli2013 weights upstream contributing area
by a combined factor $W = f(\text{land cover}, S)$ that collapses to
slope-weighted area in the absence of land-cover data. While powerful, this
weight does not directly encode the intensity of rainfall or the biophysical
suppression of runoff by vegetation — the two dominant controls on
hillslope-to-channel sediment transfer in most climatic settings
[@Nearing2004; @Wischmeier1978].

Several authors have noted this gap. @Heckmann2018 recommend including climatic
drivers in connectivity analyses; @Persichillo2018 demonstrate that IC-based
susceptibility maps improve significantly when vegetation dynamics are accounted
for; @Borselli2008 explicitly incorporated land-use C-factors. Despite these
recommendations, no open-source tool extending the IC weight function beyond a
single fixed formulation has been published. The closest prior art is
`SedInConnect` [@Crema2018], a stand-alone GUI application that implements the
original @Cavalli2013 formulation but does not expose a programmable weight
pipeline or cloud-based data fetching.

`GeomorphConn` addresses this by making $W$ a **user-configurable composable pipeline**
— the central design contribution of this software. Five weight components are
provided:

- `RainfallWeight` — normalised rainfall; high rainfall → high erosive potential.
- `NDVIWeight` — NDVI-derived C-factor proxy $C = (1 - \text{NDVI})/2$;
  the formulation of @Singh2017; combined rainfall + NDVI weighting after @Dubey2026.
- `LandCoverWeight` — maps integer land-cover class codes to USLE/RUSLE C-factors
  [@Wischmeier1978; @Borselli2008] via a lookup table.  Built-in tables are
  provided for ESA WorldCover 10 m [@WorldCover2021], CORINE Land Cover 2018,
  and MODIS IGBP.
- `SurfaceRoughnessWeight` — Terrain Ruggedness Index (TRI; @Riley1999)
  normalised from the DEM; enables weighting with no satellite data at all.
- `CustomWeight` — accepts any pre-computed array, enabling physics-based
  erosion model outputs or published maps.

Components are combined by `WeightBuilder` using a configurable aggregation
function (arithmetic mean, geometric mean, product, max, min, weighted mean, or
any user-supplied callable), then clamped to $[W_{min}, 1]$.  Five preset factory
functions cover the most common configurations out of the box (Table 2).

Additional contributions include:

- **IC-toward-target mode** — compute IC relative to a river network, reservoir
  boundary, or any geopandas-readable vector rather than the basin outlet.
- **Multiple flow-direction algorithms** — D8, D-infinity [@Tarboton1997], and
  Multiple Flow Direction [@Quinn1991] for upstream accumulation; D8 for the
  downstream path.
- **Integrated GEE data fetching** via `xee` [@xee2023], supporting DEM, NDVI,
  rainfall, and land-cover datasets with no manual download (Table 1).
- **ArcGIS Pro scripts** (`arcgis_tools/`) for practitioners without a Python
  workflow.

The design target of `GeomorphConn` is broad applicability in data-scarce
regions, where users often have only a DEM and, at most, one or two readily
available environmental layers. In this context, a robust minimum-input
connectivity workflow is often more practical than data-intensive event
frameworks. For studies with dense hydro-meteorological and geotechnical inputs
and an explicit event-scale objective, users may consider richer alternatives
such as SCI and IHC-style formulations that include additional mobility,
impedance, and antecedent-condition terms [@Zingaro2019; @Zanandrea2021].


# Mathematics

The IC is defined as [@Cavalli2013]:

$$IC = \log_{10}\left(\frac{D_{up}}{D_{dn}}\right)$$

**Upstream component $D_{up}$** represents the potential of the upslope
contributing area to deliver sediment to a given point:

$$D_{up} = \bar{W} \cdot \bar{S} \cdot \sqrt{A}$$

where $A$ is the upslope contributing area (m²), $\bar{W}$ and $\bar{S}$ are
the area-weighted means of the hydrological weight $W$ and slope factor $S$
over the contributing area, and:

$$\bar{W}(i) = \frac{Acc_W(i) + W_i}{A_i / \delta}, \qquad
  \bar{S}(i) = \frac{Acc_S(i) + S_i}{A_i / \delta}$$

with $\delta$ the cell area and $Acc_W$, $Acc_S$ the weighted upstream flow
accumulation for $W$ and $S$ respectively.

**Downstream component $D_{dn}$** represents the impedance of the downslope
pathway from a cell to the outlet or target:

$$D_{dn}(i) = \sum_{j:\, i \to \text{target}} \frac{d_j}{W_j S_j}$$

where $d_j$ is the physical distance from cell $j$ to its D8 receiver and the
sum proceeds along the steepest-descent path to the outlet or target feature.
At sink/target cells, $D_{dn} = 1/(W_i S_i)$.

**Slope factor $S$** is computed from the steepest-descent gradient and
clamped to $[W_{min}, 1]$ to prevent numerical instabilities. The default
convention $S = \theta°/100$ is faithful to the reference ArcGIS
implementation and is appropriate for low-gradient terrain. The physically
correct $S = \tan\theta$ can be selected via a parameter switch.

The weight $W$ and slope $S$ are used for both upstream accumulation and
downstream path weighting, giving the IC a consistent bivariate physical
interpretation: connectivity is simultaneously controlled by the erosive
potential of rainfall on bare ground ($W$) and the slope-driven transport
capacity ($S$).

Supported GEE datasets used by `GeomorphConn.gee.GEEFetcher` are listed in
Table @tbl:gee. Common `WeightBuilder` presets are summarised in
Table @tbl:presets.

| Dataset type | Source key | Collection / asset ID | Native scale |
|---|---|---|---|
| DEM | `SRTM` | USGS/SRTMGL1_003 | 30 m |
| DEM | `COPDEM30` | COPERNICUS/DEM/GLO30 | 30 m |
| DEM | `MERIT` | MERIT/DEM/v1_0_3 | 90 m |
| Rainfall | `CHIRPS` | UCSB-CHG/CHIRPS/DAILY | 5.5 km |
| Rainfall | `ERA5` | ECMWF/ERA5_LAND/MONTHLY_AGGR | 11 km |
| Rainfall | `PERSIANN` | NOAA/PERSIANN-CDR | 27 km |
| NDVI | `SENTINEL2` | COPERNICUS/S2_SR_HARMONIZED | 10 m |
| NDVI | `LANDSAT8` | LANDSAT/LC08/C02/T1_L2 | 30 m |
| NDVI | `LANDSAT9` | LANDSAT/LC09/C02/T1_L2 | 30 m |

Table: Supported GEE datasets in `GeomorphConn.gee.GEEFetcher`. NDVI is computed
as a cloud-masked median composite over the specified date range.
Rainfall is aggregated as a sum (CHIRPS, PERSIANN) or mean (ERA5).
Land-cover data use the most recent available image. All outputs are
bilinear-resampled to the DEM grid before returning. {#tbl:gee}

| Preset function | Components | Typical use case |
|---|---|---|
| `preset_rainfall_ndvi` | RF + NDVI C-factor | Dubey, Singh & Jain (submitted) |
| `preset_roughness_only` | TRI from DEM | DEM-only; no satellite data |
| `preset_landcover_only` | LC C-factor | Borselli (2008) spirit |
| `preset_rainfall_landcover` | RF + LC C-factor | Hydrology + land cover |
| `preset_rainfall_ndvi_roughness` | RF + NDVI + TRI | Full three-component weight |

Table: Preset `WeightBuilder` factory functions. All return a `WeightBuilder`
that can be further customised via `.add()`. {#tbl:presets}

# Implementation

`GeomorphConn` is structured as a Python package with four submodules.

The `components` submodule contains :class:`ConnectivityIndex`, which inherits
from Landlab's `Component` base class and follows Landlab conventions for field
declaration, unit specification, and single-step execution. The `weight`
parameter of :class:`ConnectivityIndex` accepts a :class:`WeightBuilder`
instance, a pre-computed array, or *None* (for the legacy NDVI + rainfall
interface), providing three pathways for weight specification at different levels
of customisation.

The `weights` submodule implements the composable weight pipeline (Figure 1).
Each weight component class exposes a `.compute()` method returning a 1-D float64
array. :class:`WeightBuilder` stacks any number of components and combines them
using a user-specified aggregation function, then applies final clamping to
$[W_{min}, 1]$. The design deliberately separates the weight computation from the
IC routing, so that weights can be pre-inspected, compared, and updated between
time-steps via `ic.update_weight()`.

The `gee` submodule provides :class:`GEEFetcher` for cloud-native data access via
the `xee` [@xee2023] backend, returning a unified result dictionary that includes
DEM, NDVI, rainfall, and (optionally) integer land-cover classification.

The `utils` submodule provides `rasterize_targets` for converting vector target
features to Landlab node arrays.

`GeomorphConn` is available through three user-facing entry points:

- **Python API** via :class:`ConnectivityIndex` and the composable
  :class:`WeightBuilder` pipeline.
- **CLI** via the `geomorphconn` command (`run`, `gee fetch`, and `gui`
  subcommands).
- **GUI** via the Streamlit frontend for interactive runs.

Minimal usage examples are shown below.

CLI (local rasters):

```bash
geomorphconn run --dem dem.tif --ndvi ndvi.tif --rainfall rain.tif --outputs IC
```

GUI launch:

```bash
geomorphconn gui --backend streamlit
```

Python API (minimum):

```python
import numpy as np
import rioxarray as rxr
from landlab import RasterModelGrid
from geomorphconn.components import ConnectivityIndex

# Read GeoTIFF inputs as xarray DataArray objects
dem_da = rxr.open_rasterio("dem.tif", masked=True).squeeze(drop=True)
ndvi_da = rxr.open_rasterio("ndvi.tif", masked=True).squeeze(drop=True)
rain_da = rxr.open_rasterio("rainfall.tif", masked=True).squeeze(drop=True)

# Convert to 2D numpy arrays (north-up raster convention)
dem2d = np.asarray(dem_da.values, dtype=float)
ndvi2d = np.asarray(ndvi_da.values, dtype=float)
rain2d = np.asarray(rain_da.values, dtype=float)

# Build Landlab grid (south-up node convention)
dx = abs(float(dem_da.rio.transform().a))
grid = RasterModelGrid(dem2d.shape, xy_spacing=dx)
grid.add_field("topographic__elevation", np.flipud(dem2d).ravel(), at="node")

ic = ConnectivityIndex(
  grid,
  ndvi=np.flipud(ndvi2d).ravel(),
  rainfall=np.flipud(rain2d).ravel(),
)
ic.run_one_step()

ic_map = np.flipud(grid.at_node["connectivity_index__IC"].reshape(dem2d.shape))
```

The two computationally intensive operations — weighted upstream flow accumulation
and downstream path-length calculation — are sequential graph traversals over the
Landlab `flow__upstream_node_order` array, $O(N)$ in the number of grid nodes.
Optional `numba` [@Lam2015] JIT compilation accelerates these loops by an order of
magnitude for grids larger than approximately $500 \times 500$ cells.

The package includes a comprehensive `pytest` test suite covering: correct field
initialization, plausible $W$ and $S$ ranges across all five weight components,
all six combination modes of :class:`WeightBuilder`, all five preset factory
functions, backward compatibility of the legacy interface, the outlet vs target
mode difference, D8 vs D-infinity upstream accumulation differences, and update
methods.

# Acknowledgements

M. Singh is supported by the Royal Society Newton International Fellowship
(NIF\R1\232539) and was an Alexander von Humboldt Research Fellow at the
University of Potsdam (hosted by Prof. Bodo Bookhagen). The IC methodology
was originally developed by M. Cavalli and collaborators at CNR-IRPI, Padova.
The `xee` library is developed by the Google Earth Engine team.

# References
