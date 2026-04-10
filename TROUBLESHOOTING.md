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

### Streamlit GUI loses connection after a few minutes of inactivity

Cause:
Windows' TCP stack aggressively closes idle socket connections, which drops the
WebSocket link between the browser and the Streamlit server. This does not
happen on WSL2/Linux because the Linux kernel keeps idle sockets open longer.

Fix:
The project's `.streamlit/config.toml` already sets `headless = true` and
`enableWebsocketCompression = false` to mitigate this. If you still see
disconnections:

- Refresh the browser tab — Streamlit will reconnect automatically.
- Make sure you are running from the project directory so that
  `.streamlit/config.toml` is picked up.
- Do not use the `streamlit run` command directly; always use
  `geomorphconn gui` so the config file is found correctly.

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

### Target cells are `NaN` in the IC output — is this correct?

Yes, this is expected behaviour. Cells belonging to the target geometry
(river network, reservoir boundary, etc.) represent the sink of the downstream
path; their own IC value is undefined and is written as `NaN`. Values in the
cells immediately adjacent to the target are valid and represent the impedance
to the nearest target cell.

If all cells appear as `NaN`, the target geometry likely does not overlap the
DEM extent — see *Target vector produces no target nodes* above.

### Are depressions filled before routing?

Yes, by default. GeomorphConn applies `SinkFillerBarnes` before flow routing,
replicating the ArcGIS-style `Fill → FlowDirection → FlowAccumulation`
workflow. This is enabled by default (`fill_sinks=True`) and can be toggled via
the `Fill sinks before routing` checkbox in the GUI or the `fill_sinks` parameter
in the Python API.

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

### Which flow director should I use for large catchments?

This is the most important question for large-area runs:

| Flow director | Memory use (relative) | Recommended for |
|---|---|---|
| **D8** | Low (1×) | Any catchment size; safest default |
| DINF | Very high (≈10–20×) | Small to medium areas only (<5 M nodes) |
| MFD | Very high (≈10–20×) | Small to medium areas only (<5 M nodes) |

**Rule of thumb:** if your DEM has more than ~5 million nodes (e.g. anything
larger than roughly 2000 × 2500 pixels), use **D8**. DINF and MFD are
theoretically more realistic for diffuse hillslope flow but become
prohibitively expensive on large grids.

**For very large areas (regional/national scale):** the ArcGIS tools in
`arcgis_tools/` are the recommended route. ArcGIS handles large rasters in
tiled chunks and does not require the entire grid to fit in RAM.

### `MemoryError: Unable to allocate X GiB for an array with shape (...)` in the GUI or CLI

Cause:
Landlab's DINF and MFD flow directors build dense 8-neighbour arrays (shape
`n_nodes × 8`) in one contiguous allocation. For a ~30 M-node DEM (≈5500 × 5500
pixels at 1 m resolution) that is ~1.8 GiB per internal array. Several such
arrays are needed simultaneously. Even with large physical RAM, Windows heap
fragmentation can block a single contiguous allocation of this size.

Dask and Numba do not help here — the allocation happens inside Landlab's own
internals, not in IC computation loops.

Fix — in order of preference:

1. **Switch the Flow director to D8.** D8 stores one receiver per node and
   uses ≤10 % of the memory that DINF or MFD require for the same grid.
   For large catchments D8 is the right choice.

2. **Use the ArcGIS tools** (`arcgis_tools/`). The ModelBuilder scripts handle
   large rasters in tiled blocks and are not memory-limited in the same way.

3. **Increase the DEM coarsen factor in the GUI** (selectbox in the settings
   column). Factor 2 reduces node count to ¼; factor 4 reduces it to 1/16.
   Appropriate when you need DINF/MFD but cannot switch to ArcGIS.

4. **Reduce input resolution before uploading.** Reproject the DEM to a coarser
   resolution in QGIS or with `gdalwarp` before starting the GUI.

5. **Run in WSL2** (Linux). The Linux kernel memory allocator handles large
   contiguous requests more reliably than the Windows heap.

Quick memory estimate for DINF/MFD:
- 1 M nodes → ~0.6 GiB peak
- 5 M nodes → ~3.2 GiB peak
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