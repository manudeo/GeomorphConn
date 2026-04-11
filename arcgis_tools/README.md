# ArcGIS Pro Toolbox

This folder contains the latest ArcGIS Pro toolbox build for GeomorphConn-compatible IC workflows.

## Included file

- ConnectivityTools.atbx

## Tools included in the toolbox

- ICoutlet
- ICoutletwithNDVIRFweightCalc
- ICtarget
- ICtargetwithNDVIRFweightCalc

## What each tool does

- ICoutlet: Computes IC toward the basin outlet using provided IC inputs.
- ICoutletwithNDVIRFweightCalc: First computes W from rainfall and NDVI, then computes outlet-mode IC.
- ICtarget: Computes IC toward a user-supplied target feature or target raster.
- ICtargetwithNDVIRFweightCalc: First computes W from rainfall and NDVI, then computes target-mode IC.

## Requirements

- ArcGIS Pro 3.x
- Spatial Analyst extension
- 3D Analyst extension
- Image Analyst extension

## Notes

- ArcGIS models rely on ArcGIS Flow Direction and Flow Accumulation tools.
- D8 is used for downstream path distance in the ArcGIS workflow.
- Weighted tools compute rainfall normalization and NDVI-based C-factor before W.
- The Python package GeomorphConn provides the same IC formulation using Landlab for scriptable workflows.
