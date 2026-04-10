# Troubleshooting

This page collects common setup and runtime issues for GeomorphConn and the
usual fixes.

## Installation problems

### `geomorphconn` command is not found

Cause:
The package is not installed in the active Python environment, or the virtual
environment is not activated.

Fix:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Then verify:

```powershell
geomorphconn --version
```

### GUI target-vector mode says `geopandas and rasterio are required`

Cause:
The environment was created before target-vector dependencies were included in
the GUI extra, or the vector stack is not installed.

Fix:

```powershell
pip install -U "geomorphconn[gui]"
```

If needed, install the vector packages directly:

```powershell
pip install geopandas shapely pyogrio rasterio
```

## Raster input issues

### Rasters do not have the same shape, extent, or resolution

Cause:
DEM, NDVI, rainfall, or weight rasters are on different grids.

Fix:
- In the GUI, enable `Auto-align rasters` and choose the `Reference grid`.
- In the CLI, use `--auto-reproject` and set `--reference-grid` as needed.

Notes:
- `dem` is usually the safest reference grid.
- Alignment uses `rioxarray.reproject_match`.

### Target vector produces no target nodes

Cause:
The target does not overlap the DEM extent, the CRS is wrong, or the target is
too narrow relative to the raster cell size.

Fix:
- Check that the vector and DEM use compatible CRS definitions.
- Confirm the vector actually overlaps the DEM bounds.
- Increase `Target buffer` in the GUI or `--target-buffer` in the CLI.
- For narrow lines, leaving the buffer at `0` will apply a small automatic
  half-cell buffer.

### Shapefile upload is not recognized in the GUI

Cause:
A shapefile was uploaded without its sidecar files.

Fix:
- Prefer uploading a zipped shapefile.
- Or upload `.shp`, `.dbf`, `.shx`, and `.prj` together.
- GeoJSON and GPKG are also supported.

## Windows-specific issues

### `PermissionError: [WinError 32]` during temporary-file cleanup

Cause:
Windows is still holding a handle to one of the raster files used during
alignment.

Fix:
- Retry the run once.
- Close any external GIS software that may still have the same files open.
- Keep `Auto-align rasters` enabled only when needed.

Notes:
GeomorphConn already performs explicit handle cleanup for temporary alignment
files, but Windows can still be aggressive about file locking.

### PowerShell blocks virtual-environment activation

Cause:
The PowerShell execution policy is preventing the activation script from
running.

Fix:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This only changes policy for the current shell session.

## Output issues

### Target pixels show IC values (should be masked)

Cause:
The run was not in target mode, or an older build was used where target cells
were not explicitly masked in outputs.

Fix:
- In GUI, set `IC mode` to `Target`.
- Update to the latest code/package and rerun.

Expected behavior:
- Cells that belong to the target geometry are written as `NaN` in IC outputs.
- Values around the target boundary are still valid (they represent impedance
   to the target), but cells inside the target mask should be `NaN`.

### Sink/depression artefacts still visible in flow outputs

Cause:
If depressions are not explicitly filled before routing, local pits can create
artefacts in flow direction/accumulation and downstream IC patterns.

Fix:
- In GUI, enable `Fill sinks before routing (ArcGIS-like)`.

Why this matches ArcGIS:
- ArcGIS workflows typically follow `Fill -> FlowDirection -> FlowAccumulation`.
- GeomorphConn now supports the same sequence by explicitly applying
   `SinkFillerBarnes` before routing.

### Output filenames are not what I expected

Cause:
The GUI supports both prefix and suffix naming modes.

Fix:
- Set `Output affix mode` to `suffix` or `prefix` as needed.
- Leave `Output name affix` blank to keep default names.

### No files were written to disk in the GUI

Cause:
Saving was disabled, no output layers were selected, or the output folder is
not writable.

Fix:
- Make sure `Save selected outputs to disk` is enabled.
- Select at least one output layer.
- Confirm the output directory exists or can be created.

## Large DEM / MemoryError

### `MemoryError: Unable to allocate X GiB for an array with shape (...)` in the GUI or CLI

Cause:
Landlab's DINF and MFD flow directors build dense 8-neighbour arrays (shape
`n_nodes × 8`) in one contiguous allocation. For a ~30 M-node DEM (≈5500 × 5500
pixels at 1 m resolution) that is ~1.8 GiB per internal array. Several such arrays
are needed simultaneously. Even with large physical RAM, Windows heap fragmentation
can block a single contiguous allocation of this size.

Dask and Numba do not help here — the allocation happens inside Landlab's own
internals, not in IC computation loops.

Fix — choose one or more:

1. **Increase the DEM coarsen factor in the GUI** (selectbox in the settings
   column). Factor 2 reduces node count to ¼; factor 4 reduces it to 1/16.

2. **Switch the Flow director to D8**. D8 only stores one receiver per node
   and needs far less memory than DINF or MFD for the same grid.

3. **Reduce input resolution before uploading**. Reproject the DEM to a coarser
   resolution in QGIS or with `gdalwarp` before starting the GUI.

4. **Run in WSL2** (Linux). The Linux kernel memory allocator handles large
   contiguous requests more reliably than the Windows heap.

Quick memory estimate for DINF:
- 5 M nodes → ~3.2 GiB peak internal (DINF)
- 10 M nodes → ~6.4 GiB peak
- 30 M nodes → ~19 GiB peak
D8 typically uses ≤10 % of those figures.

---

## Performance and environment notes

### The run feels slow on large rasters

Cause:
IC calculations and raster reprojection are memory- and CPU-intensive.

Fix:
- Install the optional speed dependency:

```powershell
pip install "geomorphconn[fast]"
```

- Use fewer output layers when testing.
- Align rasters in advance if you reuse the same inputs many times.

### Tests or notebooks fail in one environment but not another

Cause:
Different Python environments may have different dependency sets.

Fix:
- Confirm the active interpreter is the project `.venv`.
- Reinstall the project extras needed for your workflow.
- Run:

```powershell
pytest
```

## Still stuck?

When reporting a problem, include:

- your OS
- Python version
- how you installed GeomorphConn
- whether you used CLI or GUI
- the full error message
- whether the inputs were GeoTIFF, GeoJSON, GPKG, or shapefile

That usually makes the issue reproducible much faster.