# CLI and GUI Options Reference

Use this reference for quick interpretation of key options.

- `Use supplied weight raster (W)` (GUI) / `--weight-raster` (CLI): use precomputed W and skip NDVI/rainfall/roughness computation.
- Slope convention (GUI/CLI): slope uses dy/dx (equivalent to ArcGIS/TauDEM `percent_rise / 100`).
- `Use aspect weighting` (GUI) / `--use-aspect-weighting` (CLI): TauDEM-style partition weighting for multi-receiver upstream accumulation.
- `Auto-align rasters` (GUI) / `--auto-reproject` (CLI): align rasters to selected reference grid.
- `Fill sinks before routing (ArcGIS-like)` (GUI) / `--fill-sinks` (CLI): run sink filling before routing; `--no-fill-sinks` disables it.
- `Reference grid` (GUI/CLI): alignment target (`dem`, `ndvi`, `rainfall`, `weight`).
- `Roughness detrend window` and `Roughness std window` (GUI) / `--roughness-detrend-window`, `--roughness-std-window` (CLI): odd moving-window sizes for roughness.
- `w_min` / `w_max` (GUI): weight scaling clamps.
- `IC mode` (GUI): `Outlet` or `Target`.
- Target definition (GUI target mode): stream-threshold or vector file.
- `Target vector path` (GUI) / `--target-vector` (CLI): vector target input.
- `Target buffer` (GUI) / `--target-buffer` (CLI): optional pre-rasterization buffer.
- `Target rasterization: all touched` (GUI) / `--all-touched` (CLI): include any pixel touched by target geometry.
- Target masking behavior: target pixels are written as `NaN` in IC outputs.
- `Save selected outputs to disk` (GUI): export selected layers.
- `Output affix mode` (GUI): suffix or prefix naming.
- Progress feedback: stage-based progress in CLI and GUI.

GUI output layer meanings:

- `IC`: final index.
- `Dup`: upstream connectivity term.
- `Ddn`: downstream impedance term.
- `W`: effective weight raster.
- `S`: slope factor.
- `Wmean`: mean upstream weight.
- `Smean`: mean upstream slope.
