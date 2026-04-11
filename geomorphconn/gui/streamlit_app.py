"""Streamlit GUI for GeomorphConn."""

from __future__ import annotations

import importlib
import tempfile
import gc
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from landlab import RasterModelGrid

from geomorphconn import ConnectivityIndex
from geomorphconn.weights import NDVIWeight, RainfallWeight, SurfaceRoughnessWeight, WeightBuilder


_FIELD_MAP = {
    "IC": "connectivity_index__IC",
    "Dup": "connectivity_index__Dup",
    "Ddn": "connectivity_index__Ddn",
    "W": "connectivity_index__W",
    "S": "connectivity_index__S",
    "Wmean": "connectivity_index__Wmean",
    "Smean": "connectivity_index__Smean",
}


def _read_uploaded_raster(uploaded_file):
    with rasterio.MemoryFile(uploaded_file.getvalue()) as memfile:
        with memfile.open() as src:
            arr = src.read(1).astype(np.float64)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            profile = src.profile.copy()
            return arr, profile


def _compute_ic(
    dem,
    ndvi,
    rainfall,
    flow_director,
    use_degree_approx,
    use_aspect_weighting,
    xy_spacing,
    weight_factors,
    weight_combine,
    roughness_invert,
    w_min,
    w_max,
    user_weight,
    target_nodes,
    stream_threshold,
    fill_sinks,
):
    if ndvi is not None and dem.shape != ndvi.shape:
        raise ValueError("DEM and NDVI rasters must have identical shape.")
    if rainfall is not None and dem.shape != rainfall.shape:
        raise ValueError("DEM and rainfall rasters must have identical shape.")
    if user_weight is not None and dem.shape != user_weight.shape:
        raise ValueError("DEM and user-supplied weight raster must have identical shape.")

    grid = RasterModelGrid(dem.shape, xy_spacing=float(xy_spacing))
    grid.add_field("topographic__elevation", np.flipud(dem).ravel(), at="node")

    if user_weight is not None:
        ci_kwargs = {
            "flow_director": flow_director,
            "weight": np.flipud(user_weight).ravel(),
            "target_nodes": target_nodes,
            "stream_threshold": stream_threshold,
            "fill_sinks": fill_sinks,
            "w_min": w_min,
            "w_max": w_max,
            "use_degree_approx": use_degree_approx,
            "use_aspect_weighting": use_aspect_weighting,
        }
        ic = ConnectivityIndex(
            grid,
            **ci_kwargs,
        )
    else:
        wb = WeightBuilder(combine=weight_combine, w_min=w_min, w_max=w_max)
        if "rainfall" in weight_factors:
            wb.add(RainfallWeight(np.flipud(rainfall).ravel(), w_min=w_min))
        if "ndvi" in weight_factors:
            wb.add(NDVIWeight(np.flipud(ndvi).ravel(), w_min=w_min))
        if "roughness" in weight_factors:
            wb.add(SurfaceRoughnessWeight(grid, w_min=w_min, invert=roughness_invert))

        ci_kwargs = {
            "flow_director": flow_director,
            "weight": wb,
            "target_nodes": target_nodes,
            "stream_threshold": stream_threshold,
            "fill_sinks": fill_sinks,
            "w_min": w_min,
            "w_max": w_max,
            "use_degree_approx": use_degree_approx,
            "use_aspect_weighting": use_aspect_weighting,
        }

        ic = ConnectivityIndex(grid, **ci_kwargs)
    ic.run_one_step()

    return {
        key: np.flipud(grid.at_node[field].reshape(dem.shape))
        for key, field in _FIELD_MAP.items()
    }


def _align_uploaded_to_reference(dem_file, ndvi_file, rainfall_file, weight_file, reference_grid: str):
    try:
        import rioxarray as rxr
    except ImportError as exc:
        raise RuntimeError(
            "Automatic raster alignment requires rioxarray. Install with: pip install rioxarray"
        ) from exc

    def _open_da(path: Path):
        obj: Any = rxr.open_rasterio(path, masked=True)
        return obj.squeeze(drop=True)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        p_dem = Path(tmpdir) / "dem.tif"
        p_ndvi = Path(tmpdir) / "ndvi.tif"
        p_rf = Path(tmpdir) / "rainfall.tif"
        p_w = Path(tmpdir) / "weight.tif"

        p_dem.write_bytes(dem_file.getvalue())
        if ndvi_file is not None:
            p_ndvi.write_bytes(ndvi_file.getvalue())
        if rainfall_file is not None:
            p_rf.write_bytes(rainfall_file.getvalue())
        if weight_file is not None:
            p_w.write_bytes(weight_file.getvalue())

        ds_map: dict[str, Any] = {}
        aligned: dict[str, Any] = {}
        ref_ds: Any = None
        try:
            ds_map = {"dem": _open_da(p_dem)}
            if ndvi_file is not None:
                ds_map["ndvi"] = _open_da(p_ndvi)
            if rainfall_file is not None:
                ds_map["rainfall"] = _open_da(p_rf)
            if weight_file is not None:
                ds_map["weight"] = _open_da(p_w)

            if reference_grid not in ds_map:
                raise ValueError(
                    f"Reference grid '{reference_grid}' is unavailable. Available: {sorted(ds_map)}"
                )
            ref_ds = ds_map[reference_grid]
            aligned = {
                k: (v if k == reference_grid else v.rio.reproject_match(ref_ds))
                for k, v in ds_map.items()
            }

            # Force arrays in-memory so all file handles can be closed before tmp cleanup.
            arrays = {k: np.asarray(aligned[k].values, dtype=np.float64).copy() for k in ds_map}
            for key in arrays:
                arrays[key] = np.where(np.isfinite(arrays[key]), arrays[key], np.nan)

            profile = {
                "transform": ref_ds.rio.transform(),
                "crs": ref_ds.rio.crs,
                "width": int(ref_ds.rio.width),
                "height": int(ref_ds.rio.height),
            }
            return arrays, profile
        finally:
            # Close xarray/rioxarray handles explicitly (important on Windows).
            for da in {**aligned, **ds_map}.values():
                try:
                    da.close()
                except Exception:
                    pass
            ref_ds = None
            aligned.clear()
            ds_map.clear()
            gc.collect()


def _write_output_raster(path: Path, array2d: np.ndarray, profile: dict):
    out = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "count": 1,
        "crs": profile["crs"],
        "transform": profile["transform"],
        "nodata": -9999.0,
    }
    data = np.where(np.isfinite(array2d), array2d, -9999.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out) as dst:
        dst.write(data, 1)


def _plot_output_layer(layer: np.ndarray, title: str):
    masked = np.ma.masked_invalid(layer)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(masked, cmap="viridis", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return fig


def _output_suffix_text(raw: str) -> str:
    suffix = raw.strip().strip("_")
    return suffix


def _browse_directory_native(initial_dir: str | None = None) -> str | None:
    """Open a native directory chooser and return selected folder path.

    Returns None if user cancels or if a GUI backend is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=initial_dir or str(Path.cwd()))
        root.destroy()
        if selected:
            return selected
    except Exception:
        return None
    return None


def _browse_file_native(
    title: str,
    filetypes: list[tuple[str, str]],
    initial_dir: str | None = None,
) -> str | None:
    """Open a native file chooser and return the selected file path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir or str(Path.cwd()),
            filetypes=filetypes,
        )
        root.destroy()
        if selected:
            return selected
    except Exception:
        return None
    return None


def _coarsen_rasters(
    arrs: dict,
    factor: int,
    xy_spacing: float,
    save_profile: dict,
):
    """Block-mean coarsen all rasters by *factor* (integer >= 1).

    Uses nanmean so NoData cells don't corrupt neighbours.  The Affine
    transform and width/height in *save_profile* are updated accordingly.
    """
    if factor <= 1:
        return arrs, xy_spacing, save_profile

    from rasterio.transform import Affine

    coarsened: dict = {}
    for k, arr in arrs.items():
        if arr is None:
            coarsened[k] = None
            continue
        r, c = arr.shape
        nr, nc = r // factor, c // factor
        trimmed = arr[: nr * factor, : nc * factor]
        coarsened[k] = np.nanmean(
            trimmed.reshape(nr, factor, nc, factor), axis=(1, 3)
        )

    new_xy = xy_spacing * factor
    old_t = save_profile["transform"]
    new_t = Affine(old_t.a * factor, old_t.b, old_t.c, old_t.d, old_t.e * factor, old_t.f)
    ref_shape = next(v.shape for v in coarsened.values() if v is not None)
    new_profile = {
        **save_profile,
        "transform": new_t,
        "width": ref_shape[1],
        "height": ref_shape[0],
    }
    return coarsened, new_xy, new_profile


def _target_nodes_from_uploaded_vector(
    uploaded_files,
    grid,
    dem_transform,
    dem_crs,
    all_touched: bool,
    buffer_m: float,
):
    if not uploaded_files:
        return None

    from geomorphconn.utils import rasterize_targets

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp = Path(tmpdir)
        source_path: Path | None = None

        if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".zip"):
            zip_path = tmp / uploaded_files[0].name
            zip_path.write_bytes(uploaded_files[0].getvalue())
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            shp_files = list(tmp.rglob("*.shp"))
            if shp_files:
                source_path = shp_files[0]
        else:
            for uploaded in uploaded_files:
                (tmp / uploaded.name).write_bytes(uploaded.getvalue())
            shp_files = list(tmp.rglob("*.shp"))
            if shp_files:
                source_path = shp_files[0]

        if source_path is None:
            candidates = []
            for pattern in ("*.geojson", "*.json", "*.gpkg", "*.gpkg.zip"):
                candidates.extend(tmp.rglob(pattern))
            if candidates:
                source_path = candidates[0]

        if source_path is None:
            raise ValueError(
                "Could not identify a target vector source. Upload a GeoJSON/GPKG or a zipped shapefile."
            )

        return rasterize_targets(
            source_path,
            grid,
            dem_transform=dem_transform,
            dem_crs=dem_crs,
            all_touched=all_touched,
            buffer_m=buffer_m,
        )


def _target_nodes_from_vector_path(
    source_path: str,
    grid,
    dem_transform,
    dem_crs,
    all_touched: bool,
    buffer_m: float,
):
    from geomorphconn.utils import rasterize_targets

    return rasterize_targets(
        source_path,
        grid,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        all_touched=all_touched,
        buffer_m=buffer_m,
    )


def main():
    try:
        st = importlib.import_module("streamlit")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "streamlit is required for the GUI. Install with: pip install streamlit"
        ) from exc

    st.set_page_config(page_title="GeomorphConn GUI", layout="wide")
    st.title("GeomorphConn")
    st.write("Run Index of Connectivity from DEM, NDVI, and rainfall rasters.")

    with st.expander("Welcome / About GeomorphConn", expanded=True):
        st.markdown(
            """
### What this software does
- Computes hydrologically-weighted Index of Connectivity (IC) maps.
- Produces IC components: IC, Dup, Ddn, W, S, Wmean, Smean.
- Supports D8, D-infinity, and MFD routing options.
- Supports optional aspect-weighted upstream accumulation.
- Handles mismatched input raster grids via automatic alignment.

### Core references
- Cavalli et al. (2013), *Geomorphology*, 188, 31-41.
- Crema & Cavalli (2018), *Computers & Geosciences*, 111, 39-45.

### Authors
- Manudeo Singh
- Marco Cavalli
- Stefano Crema
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        target_mode = st.radio(
            "IC mode",
            ["Outlet", "Target"],
            index=0,
            horizontal=True,
            help="Choose whether IC is computed toward the basin outlet or toward a user-supplied target feature.",
        )
        flow_director = st.selectbox(
            "Flow director",
            ["D8", "DINF", "MFD"],
            index=1,
            help="Routing method used for upstream flow accumulation: D8 = single steepest path, DINF = distributed flow by aspect, MFD = multiple-flow-direction partitioning.",
        )
        use_degree_approx = st.checkbox(
            "Use degree approximation",
            value=True,
            help="If enabled: S = slope_degrees/100. If disabled: S = tan(theta).",
        )
        use_supplied_weight = st.checkbox(
            "Use supplied weight raster (W)",
            value=False,
            help="Use a precomputed W raster and skip NDVI/rainfall/roughness factor computation.",
        )
        if use_supplied_weight:
            st.caption("Custom W mode active: NDVI/rainfall/roughness factors are hidden.")
            weight_factors = []
        else:
            weight_factors = st.multiselect(
                "Weight factors",
                ["rainfall", "ndvi", "roughness"],
                default=["rainfall", "ndvi"],
                help="Choose which factors contribute to W. Rainfall adds erosive forcing, NDVI adds vegetation/C-factor influence, and roughness adds DEM-derived terrain resistance/structure.",
            )
        weight_combine = st.selectbox(
            "Weight combine",
            ["mean", "arithmetic_mean", "geometric_mean", "product", "max", "min", "weighted_mean"],
            index=0,
            help="Method used to combine selected weight factors into one W raster. 'mean' is the default balanced option; 'product' is stricter; 'max'/'min' keep only the strongest/weakest factor response.",
        )
    with col2:
        use_aspect_weighting = st.checkbox(
            "Use aspect weighting",
            value=False,
            help="Enable TauDEM-style partition weighting for multi-receiver upstream flow.",
        )
        auto_align = st.checkbox(
            "Auto-align rasters",
            value=True,
            help="Align all rasters to the selected reference grid before IC computation.",
        )
        fill_sinks = st.checkbox(
            "Fill sinks before routing (ArcGIS-like)",
            value=False,
            help="If enabled, depressions are explicitly filled before flow direction/accumulation (Fill -> FlowDirection -> FlowAccumulation). Disable only for diagnostic comparisons.",
        )
        available_refs = ["dem"]
        if use_supplied_weight:
            available_refs.append("weight")
        if ("ndvi" in weight_factors) and not use_supplied_weight:
            available_refs.append("ndvi")
        if ("rainfall" in weight_factors) and not use_supplied_weight:
            available_refs.append("rainfall")
        reference_grid = st.selectbox(
            "Reference grid",
            available_refs,
            index=0,
            help="Raster grid to which other inputs are aligned before computation. Choose the dataset whose resolution/extent you want to preserve.",
        )
        roughness_invert = st.checkbox(
            "Invert roughness (TRI)",
            value=False,
            help="Invert TRI-based weighting so rougher terrain decreases W instead of increasing it.",
        )
        w_min = st.number_input(
            "w_min",
            value=0.005,
            min_value=0.0,
            max_value=1.0,
            step=0.00001,
            format="%.5f",
            help="Lower clamp applied to W (and slope S) to avoid zeros and unstable path calculations.",
        )
        w_max = st.number_input(
            "w_max",
            value=1.0,
            min_value=0.01,
            max_value=10.0,
            step=0.00001,
            format="%.5f",
            help="Upper clamp applied to W after factor combination. Default 1.0 keeps weights in the usual normalized range.",
        )
        coarsen_factor = st.selectbox(
            "DEM coarsen factor",
            [1, 2, 4, 8],
            index=0,
            format_func=lambda x: "1× (original resolution)" if x == 1 else f"{x}× coarser (1/{x} resolution, {x*x}× fewer nodes)",
            help=(
                "Reduce DEM resolution before IC computation. Use 2× or 4× for very large DEMs "
                "(>5 M nodes) to prevent memory errors. DINF builds dense 8-neighbour arrays "
                "(~64 bytes/node); at 30 M nodes that is 1.8 GiB for a single internal array. "
                "D8 needs far less memory than DINF/MFD for the same grid."
            ),
        )

    dem_file = st.file_uploader("DEM GeoTIFF", type=["tif", "tiff"])
    ndvi_file = (
        st.file_uploader("NDVI GeoTIFF", type=["tif", "tiff"])
        if (not use_supplied_weight) and ("ndvi" in weight_factors)
        else None
    )
    rainfall_file = (
        st.file_uploader("Rainfall GeoTIFF", type=["tif", "tiff"])
        if (not use_supplied_weight) and ("rainfall" in weight_factors)
        else None
    )
    weight_file = (
        st.file_uploader("Weight (W) GeoTIFF", type=["tif", "tiff"])
        if use_supplied_weight
        else None
    )

    target_vector_path = None
    target_files = None
    stream_threshold = None
    if target_mode == "Target":
        st.subheader("Target settings")
        stream_threshold = st.number_input(
            "Stream generation threshold (cells)",
            min_value=1,
            value=1000,
            step=100,
            help=(
                "Cells with D8 upstream count >= this threshold are treated as stream/target cells. "
                "This matches the Borselli ArcGIS channel-mask concept and can be used with or without a vector target."
            ),
        )
        if "target_vector_path_value" not in st.session_state:
            st.session_state["target_vector_path_value"] = ""

        tcol1, tcol2 = st.columns([3, 1])
        with tcol1:
            target_vector_path = st.text_input(
                "Target vector path",
                value=st.session_state["target_vector_path_value"],
                help="Path to a local target vector file. A `.shp` path works directly if its sidecar files are in the same folder.",
            )
            st.session_state["target_vector_path_value"] = target_vector_path
        with tcol2:
            st.write("")
            if st.button("Browse vector...", help="Pick a local target vector file"):
                picked = _browse_file_native(
                    "Select target vector",
                    [
                        ("Vector files", "*.shp *.zip *.geojson *.json *.gpkg"),
                        ("Shapefile", "*.shp"),
                        ("Zip archive", "*.zip"),
                        ("GeoJSON", "*.geojson *.json"),
                        ("GeoPackage", "*.gpkg"),
                        ("All files", "*.*"),
                    ],
                    initial_dir=str(Path(target_vector_path).parent)
                    if target_vector_path.strip()
                    else str(Path.cwd()),
                )
                if picked:
                    st.session_state["target_vector_path_value"] = picked
                    st.rerun()

        target_files = st.file_uploader(
            "Or upload target vector (GeoJSON/GPKG or zipped shapefile)",
            type=["geojson", "json", "gpkg", "zip", "shp", "dbf", "shx", "prj", "cpg"],
            accept_multiple_files=True,
            help="Alternative to using a local path. A single `.shp` upload is usually not enough by itself; use the path field for local shapefiles, upload all sidecar files together, or upload a zipped shapefile.",
        )

    st.subheader("Output settings")
    if "output_dir_value" not in st.session_state:
        st.session_state["output_dir_value"] = "outputs_gui"

    bcol1, bcol2 = st.columns([3, 1])
    with bcol1:
        output_dir = st.text_input(
            "Output directory",
            value=st.session_state["output_dir_value"],
            help="Directory where selected output rasters will be saved.",
        )
        st.session_state["output_dir_value"] = output_dir
    with bcol2:
        st.write("")
        if st.button("Browse...", help="Open a native folder picker"):
            picked = _browse_directory_native(st.session_state["output_dir_value"])
            if picked:
                st.session_state["output_dir_value"] = picked
                st.rerun()

    output_suffix_raw = st.text_input(
        "Output name affix",
        value="",
        help="Token used as prefix/suffix in output filenames, e.g. monsoon2020",
    )
    output_affix_mode = st.selectbox(
        "Output affix mode",
        ["suffix", "prefix"],
        index=0,
        help="Choose whether the output name affix is appended as a suffix (default) or prepended as a prefix.",
    )

    output_keys = st.multiselect(
        "Output layers to save",
        list(_FIELD_MAP.keys()),
        default=["IC"],
        help="Choose which result layers are written to disk. IC is the main final index; Dup and Ddn are upstream/downstream components; W and S are the effective weight and slope terms.",
    )
    preview_key = st.selectbox(
        "Preview selected output layer",
        list(_FIELD_MAP.keys()),
        index=0,
        help="Select which computed layer to display in the preview map after the run completes.",
    )
    with st.expander("What the output layers mean"):
        st.markdown(
            """
- `IC`: final Index of Connectivity, computed as the log-ratio of upstream connectivity potential to downstream impedance.
- `Dup`: upstream component, representing the potential contribution from the upslope contributing area.
- `Ddn`: downstream component, representing the impedance/resistance along the downslope path to the outlet or target.
- `W`: effective weight raster used in the computation after combining selected factors or using a supplied W raster.
- `S`: slope factor used in the IC calculation.
- `Wmean`: mean upstream weight over the contributing area.
- `Smean`: mean upstream slope factor over the contributing area.
            """
        )
    target_buffer = 0.0
    target_all_touched = True
    if target_mode == "Target":
        target_buffer = st.number_input(
            "Target buffer (m)",
            value=0.0,
            min_value=0.0,
            step=0.00001,
            format="%.5f",
            help="Optional buffer applied to target vectors before rasterization. If 0, line targets are auto-buffered by about half a cell.",
        )
        target_all_touched = st.checkbox(
            "Target rasterization: all touched",
            value=True,
            help="If enabled, any raster cell touched by the target geometry is included. If disabled, only cells whose centers are covered are included.",
        )
    save_outputs = st.checkbox(
        "Save selected outputs to disk",
        value=True,
        help="Write selected output layers as GeoTIFF files in the output directory.",
    )

    if st.button("Run IC"):
        progress = st.progress(0)
        status = st.empty()
        status.info("Validating inputs...")

        if dem_file is None:
            st.error("Upload DEM file first.")
            return
        if use_supplied_weight and weight_file is None:
            st.error("Supplied-weight mode is enabled; upload Weight (W) file.")
            return
        if ("ndvi" in weight_factors) and (not use_supplied_weight) and ndvi_file is None:
            st.error("NDVI is selected as a weight factor; upload NDVI file.")
            return
        if ("rainfall" in weight_factors) and (not use_supplied_weight) and rainfall_file is None:
            st.error("Rainfall is selected as a weight factor; upload rainfall file.")
            return
        if (not use_supplied_weight) and (not weight_factors):
            st.error("Select at least one weight factor.")
            return
        if save_outputs and not output_keys:
            st.error("Select at least one output layer to save.")
            return
        if target_mode == "Target" and (stream_threshold is None) and not (target_vector_path and target_vector_path.strip()) and not target_files:
            st.error("Target mode requires a stream threshold and/or a target vector (path or upload).")
            return

        progress.progress(15)

        try:
            status.info("Preparing rasters...")
            if auto_align:
                arrays, ref_profile = _align_uploaded_to_reference(
                    dem_file, ndvi_file, rainfall_file, weight_file, reference_grid
                )
                dem = arrays["dem"]
                ndvi = arrays.get("ndvi")
                rainfall = arrays.get("rainfall")
                user_weight = arrays.get("weight")
                xy_spacing = abs(float(ref_profile["transform"].a))
                save_profile = {
                    "transform": ref_profile["transform"],
                    "crs": ref_profile["crs"],
                    "width": int(ref_profile["width"]),
                    "height": int(ref_profile["height"]),
                }
            else:
                dem, dem_profile = _read_uploaded_raster(dem_file)
                ndvi = _read_uploaded_raster(ndvi_file)[0] if ndvi_file is not None else None
                rainfall = (
                    _read_uploaded_raster(rainfall_file)[0]
                    if rainfall_file is not None
                    else None
                )
                user_weight = _read_uploaded_raster(weight_file)[0] if weight_file is not None else None
                xy_spacing = abs(float(dem_profile["transform"].a))
                save_profile = {
                    "transform": dem_profile["transform"],
                    "crs": dem_profile.get("crs"),
                    "width": int(dem.shape[1]),
                    "height": int(dem.shape[0]),
                }

            # DEM size advisory
            n_nodes = dem.size
            if n_nodes > 5_000_000 and flow_director in ("DINF", "MFD"):
                st.warning(
                    f"Large DEM: {n_nodes:,} nodes ({dem.shape[0]}\u00d7{dem.shape[1]}). "
                    f"DINF/MFD allocate ~{8 * 8 * n_nodes / (1024 ** 3):.1f}\u202fGiB for internal "
                    f"8-neighbour arrays. If you hit a MemoryError, increase the Coarsen factor "
                    f"or switch to D8."
                )

            # Optional coarsening
            if coarsen_factor > 1:
                status.info(f"Coarsening rasters {coarsen_factor}\u00d7 ...")
                raster_dict = {"dem": dem, "ndvi": ndvi, "rainfall": rainfall, "weight": user_weight}
                coarsened_dict, xy_spacing, save_profile = _coarsen_rasters(
                    raster_dict, coarsen_factor, xy_spacing, save_profile
                )
                dem = coarsened_dict["dem"]
                ndvi = coarsened_dict["ndvi"]
                rainfall = coarsened_dict["rainfall"]
                user_weight = coarsened_dict["weight"]

            target_nodes = None
            if target_mode == "Target":
                status.info("Rasterizing target vector...")
                target_grid = RasterModelGrid(dem.shape, xy_spacing=float(xy_spacing))
                if target_vector_path and target_vector_path.strip():
                    target_nodes = _target_nodes_from_vector_path(
                        target_vector_path.strip(),
                        target_grid,
                        save_profile["transform"],
                        save_profile.get("crs"),
                        all_touched=target_all_touched,
                        buffer_m=float(target_buffer),
                    )
                else:
                    target_nodes = _target_nodes_from_uploaded_vector(
                        target_files,
                        target_grid,
                        save_profile["transform"],
                        save_profile.get("crs"),
                        all_touched=target_all_touched,
                        buffer_m=float(target_buffer),
                    )

            progress.progress(45)
            status.info("Running IC computation...")
            outputs = _compute_ic(
                dem,
                ndvi,
                rainfall,
                flow_director,
                use_degree_approx,
                use_aspect_weighting,
                xy_spacing,
                weight_factors,
                weight_combine,
                roughness_invert,
                float(w_min),
                float(w_max),
                user_weight,
                target_nodes,
                int(stream_threshold) if stream_threshold is not None else None,
                fill_sinks,
            )
        except MemoryError:  # pragma: no cover
            progress.empty()
            status.empty()
            st.error(
                "Out of memory: unable to allocate internal flow-routing arrays for this DEM. "
                "Try one or more of these:\n"
                "1. Increase **DEM coarsen factor** to 2\u00d7 or 4\u00d7 (reduces nodes by 4\u00d7 or 16\u00d7).\n"
                "2. Switch **Flow director** from DINF/MFD to **D8** \u2014 D8 uses far less memory.\n"
                "3. See **TROUBLESHOOTING.md** \u2192 *Large DEM / MemoryError* for detailed options."
            )
            return
        except Exception as exc:  # pragma: no cover
            progress.empty()
            status.empty()
            st.exception(exc)
            return

        progress.progress(75)
        status.info("Preparing preview and summary...")

        st.success("IC computation complete.")
        ic_map = outputs["IC"]
        st.write(
            {
                "ic_min": float(np.nanmin(ic_map)),
                "ic_max": float(np.nanmax(ic_map)),
                "ic_mean": float(np.nanmean(ic_map)),
            }
        )
        preview_map = outputs[preview_key]
        st.write(
            {
                "preview_layer": preview_key,
                "preview_min": float(np.nanmin(preview_map)),
                "preview_max": float(np.nanmax(preview_map)),
                "preview_mean": float(np.nanmean(preview_map)),
            }
        )
        fig = _plot_output_layer(preview_map, f"GeomorphConn output: {preview_key}")
        st.pyplot(fig, clear_figure=True)

        if save_outputs:
            status.info("Saving output rasters...")
            out_dir = Path(output_dir)
            suffix = _output_suffix_text(output_suffix_raw)
            for key in output_keys:
                stem = f"gui_{key}"
                if suffix:
                    stem = f"{stem}_{suffix}" if output_affix_mode == "suffix" else f"{suffix}_{stem}"
                _write_output_raster(out_dir / f"{stem}.tif", outputs[key], save_profile)
            st.info(f"Saved {len(output_keys)} output layer(s) to: {out_dir}")

        progress.progress(100)
        status.success("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
