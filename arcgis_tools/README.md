# ArcGIS Pro Toolbox

This folder contains the ArcGIS Pro toolbox implementation of the GeomorphConn workflows.

## Included toolbox

- ConnectivityTools.atbx

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
	Computes DEM-based surface roughness index used in roughness-driven weighting workflows.

## Requirements

- ArcGIS Pro 3.x
- Spatial Analyst extension
- 3D Analyst extension
- Image Analyst extension

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
