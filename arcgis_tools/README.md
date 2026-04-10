# ArcGIS Pro Toolbox

This folder contains an ArcGIS Pro toolbox that mirrors the core GeomorphConn workflows.

## Included toolbox

- ConnectivityTools.atbx

The toolbox exposes four model tools:

- ICoutlet
- ICoutletwithNDVIRFweightCalc
- ICtarget
- ICtargetwithNDVIRFweightCalc

## Requirements

- ArcGIS Pro 3.x
- Spatial Analyst extension
- 3D Analyst extension
- Image Analyst extension

## Notes

- The ArcGIS models use ArcGIS Flow Direction / Flow Accumulation tools.
- D8 is used for downstream path distance.
- Weighted tools compute rainfall normalization and NDVI-based C-factor before W.
- The Python package GeomorphConn implements the same IC formulation in Landlab.
