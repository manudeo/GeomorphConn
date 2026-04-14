# ArcGIS Toolboxes

This folder contains native ArcGIS toolbox implementations of the GeomorphConn workflows.

## Included toolbox

- ConnectivityTools.atbx
- GeomorphConn_ArcGIS_v10.8.0.tbx

`ConnectivityTools.atbx` targets ArcGIS Pro.

`GeomorphConn_ArcGIS_v10.8.0.tbx` targets ArcMap 10.8.

## Toolset overview

The toolbox provides four model tools:

- ICoutlet
- ICoutletwithNDVIRFweightCalc
- ICtarget
- ICtargetwithNDVIRFweightCalc
- SurfaceRoughness

## Tool descriptions

- ICoutlet:
	Computes Index of Connectivity (IC) toward the basin outlet using supplied DEM and IC inputs.
- ICoutletwithNDVIRFweightCalc:
	Computes rainfall-normalized and NDVI-derived weighting, then runs outlet-mode IC.
- ICtarget:
	Computes IC toward user-defined target features (for example streams, channels, or other sinks).
- ICtargetwithNDVIRFweightCalc:
	Computes rainfall/NDVI weighting first, then runs target-mode IC.
- SurfaceRoughness:
	Computes DEM-based surface roughness index used in roughness-driven weighting workflows,
	following Cavalli and Marchi (2008).

Roughness reference:

- Cavalli, M. and Marchi, L. (2008). Characterisation of the surface morphology
	of an alpine alluvial fan using airborne LiDAR. *Natural Hazards and Earth
	System Sciences*, 8, 323-333. https://doi.org/10.5194/nhess-8-323-2008

## Requirements

- ArcGIS Pro 3.x (for `.atbx`) or ArcMap 10.8 (for `.tbx`)
- Spatial Analyst extension
- 3D Analyst extension
- Image Analyst extension (ArcGIS Pro workflow)

## Performance tip (ArcGIS Environments)

To speed up calculations, open the tool's **Environments** tab and set **Parallel Processing Factor**.
ArcGIS supports either:

- a process count (for example `4`), or
- a percent of available cores (for example `50%`, `75%`, `100%`).

`0` disables parallel processing, and leaving it blank lets each tool choose its default.

Use a lower value if your workstation is memory-limited or if multiple heavy GIS processes are running.

## Workflow notes

- ArcGIS models rely on ArcGIS Flow Direction and Flow Accumulation tools.
- D8 is used for downstream path-distance calculations in this ArcGIS workflow.
- Weighted tools compute rainfall normalization and NDVI-based C-factor before W.
- The Python package GeomorphConn provides the same IC formulation in a scriptable Landlab workflow.
