"""Command line interface for GeomorphConn."""

from __future__ import annotations

import argparse
import subprocess
import sys
import gc
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from landlab import RasterModelGrid

from geomorphconn import ConnectivityIndex
from geomorphconn import __version__ as _PKG_VERSION
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


def _welcome_text() -> str:
    return (
        "\n"
        f"GeomorphConn v{_PKG_VERSION}\n"
        "===================\n"
        "Hydrologically-weighted Index of Connectivity (IC) software.\n"
        "\n"
        "What it does:\n"
        "- Computes IC maps and components (IC, Dup, Ddn, W, S, Wmean, Smean).\n"
        "- Supports D8, D-infinity, and MFD routing options.\n"
        "- Supports optional aspect-weighted upstream accumulation.\n"
        "- Aligns mismatched raster grids via reproject_match (default ref: DEM).\n"
        "\n"
        "Core references:\n"
        "- Cavalli et al. (2013), Geomorphology, 188, 31-41.\n"
        "- Crema & Cavalli (2018), Computers & Geosciences, 111, 39-45.\n"
        "\n"
        "Authors:\n"
        "- Manudeo Singh\n"
        "- Marco Cavalli\n"
        "- Stefano Crema\n"
    )


def _print_welcome() -> None:
    print(_welcome_text())


def _load_aligned_rasters(
    dem_path: Path,
    ndvi_path: Path | None,
    rainfall_path: Path | None,
    weight_path: Path | None,
    ref_grid: str,
):
    try:
        import rioxarray as rxr
    except ImportError as exc:
        raise RuntimeError(
            "Automatic raster alignment requires rioxarray. Install with: pip install rioxarray"
        ) from exc

    def _open_da(path: Path):
        obj: Any = rxr.open_rasterio(path, masked=True)
        return obj.squeeze(drop=True)

    ds_map: dict[str, Any] = {}
    aligned: dict[str, Any] = {}
    ref_ds: Any = None
    try:
        ds_map = {"dem": _open_da(dem_path)}
        if ndvi_path is not None:
            ds_map["ndvi"] = _open_da(ndvi_path)
        if rainfall_path is not None:
            ds_map["rainfall"] = _open_da(rainfall_path)
        if weight_path is not None:
            ds_map["weight"] = _open_da(weight_path)

        if ref_grid not in ds_map:
            raise ValueError(
                f"reference grid '{ref_grid}' is unavailable. "
                f"Available: {sorted(ds_map)}"
            )

        ref_ds = ds_map[ref_grid]
        for key, ds in ds_map.items():
            aligned[key] = ds if key == ref_grid else ds.rio.reproject_match(ref_ds)

        arrays = {key: np.asarray(aligned[key].values, dtype=np.float64).copy() for key in ds_map}

        for key in arrays:
            arrays[key] = np.where(np.isfinite(arrays[key]), arrays[key], np.nan)

        transform = ref_ds.rio.transform()
        crs = ref_ds.rio.crs
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": int(ref_ds.rio.width),
            "height": int(ref_ds.rio.height),
            "count": 1,
            "crs": crs,
            "transform": transform,
            "nodata": -9999.0,
        }
        return arrays, profile, transform, crs
    finally:
        for da in {**aligned, **ds_map}.values():
            try:
                da.close()
            except Exception:
                pass
        ref_ds = None
        aligned.clear()
        ds_map.clear()
        gc.collect()


def _read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return arr, profile, transform, crs


def _write_raster(path: Path, array2d: np.ndarray, profile: dict):
    out = profile.copy()
    out.update(dtype="float32", count=1, nodata=-9999.0)

    data = array2d.astype(np.float32)
    data = np.where(np.isfinite(data), data, -9999.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out) as dst:
        dst.write(data, 1)


def _run_command(args) -> int:
    def _show_progress(done: int, total: int, message: str) -> None:
        pct = int((done / total) * 100)
        print(f"[{done}/{total}] {pct:3d}% - {message}")

    total_steps = 5
    _show_progress(0, total_steps, "Starting")

    if args.show_welcome:
        _print_welcome()

    dem_path = Path(args.dem)
    use_supplied_weight = args.weight_raster is not None
    factors = set(args.weight_factors)
    need_ndvi = ("ndvi" in factors) and not use_supplied_weight
    need_rainfall = ("rainfall" in factors) and not use_supplied_weight

    if use_supplied_weight and args.weight_factors:
        print("Note: --weight-raster supplied; --weight-factors are ignored.")

    if need_ndvi and not args.ndvi:
        print("Error: --ndvi is required when 'ndvi' is selected in --weight-factors.", file=sys.stderr)
        return 2
    if need_rainfall and not args.rainfall:
        print(
            "Error: --rainfall is required when 'rainfall' is selected in --weight-factors.",
            file=sys.stderr,
        )
        return 2

    ndvi_path = Path(args.ndvi) if args.ndvi else None
    rainfall_path = Path(args.rainfall) if args.rainfall else None
    weight_path = Path(args.weight_raster) if args.weight_raster else None

    _show_progress(1, total_steps, "Loading/preprocessing rasters")

    if args.auto_reproject:
        try:
            arrays, dem_profile, dem_transform, dem_crs = _load_aligned_rasters(
                dem_path, ndvi_path, rainfall_path, weight_path, args.reference_grid
            )
        except Exception as exc:
            print(f"Error during raster alignment: {exc}", file=sys.stderr)
            return 2
        dem = arrays["dem"]
        ndvi = arrays.get("ndvi")
        rainfall = arrays.get("rainfall")
        user_weight = arrays.get("weight")
    else:
        dem, dem_profile, dem_transform, dem_crs = _read_raster(dem_path)
        ndvi = None
        rainfall = None
        user_weight = None
        if need_ndvi:
            if ndvi_path is None:
                print("Error: --ndvi is required when 'ndvi' factor is selected.", file=sys.stderr)
                return 2
            ndvi, _, _, _ = _read_raster(ndvi_path)
            if dem.shape != ndvi.shape:
                print(
                    "Error: DEM and NDVI rasters must have identical shape when --no-auto-reproject is used.",
                    file=sys.stderr,
                )
                return 2
        if need_rainfall:
            if rainfall_path is None:
                print(
                    "Error: --rainfall is required when 'rainfall' factor is selected.",
                    file=sys.stderr,
                )
                return 2
            rainfall, _, _, _ = _read_raster(rainfall_path)
            if dem.shape != rainfall.shape:
                print(
                    "Error: DEM and rainfall rasters must have identical shape when --no-auto-reproject is used.",
                    file=sys.stderr,
                )
                return 2
        if use_supplied_weight:
            if weight_path is None:
                print("Error: --weight-raster path is missing.", file=sys.stderr)
                return 2
            user_weight, _, _, _ = _read_raster(weight_path)
            if dem.shape != user_weight.shape:
                print(
                    "Error: DEM and weight rasters must have identical shape when --no-auto-reproject is used.",
                    file=sys.stderr,
                )
                return 2

    dx = float(abs(dem_transform.a))
    dy = float(abs(dem_transform.e))
    if not np.isclose(dx, dy):
        print("Error: non-square pixels are not supported by this CLI.", file=sys.stderr)
        return 2

    grid = RasterModelGrid(dem.shape, xy_spacing=dx)
    grid.add_field("topographic__elevation", np.flipud(dem).ravel(), at="node")

    target_nodes = None
    if args.target_vector:
        try:
            from geomorphconn.utils import rasterize_targets
        except Exception as exc:
            print(f"Error: target support requires optional dependencies: {exc}", file=sys.stderr)
            return 2

        target_nodes = rasterize_targets(
            args.target_vector,
            grid,
            dem_transform=dem_transform,
            dem_crs=str(dem_crs) if dem_crs is not None else None,
            all_touched=args.all_touched,
            buffer_m=args.target_buffer,
        )

    _show_progress(2, total_steps, "Building IC configuration")

    # ── Memory warning for large grids with DINF/MFD ─────────────────────────
    n_nodes = grid.number_of_nodes
    _LARGE_GRID_THRESHOLD = 5_000_000
    if args.flow_director in ("DINF", "MFD") and n_nodes > _LARGE_GRID_THRESHOLD:
        n_M = n_nodes / 1e6
        est_ram_gb = n_nodes * 8 * 8 / 1e9  # 8 neighbours × 8 bytes × n_nodes
        print(
            f"\nWARNING: Large grid detected — {n_M:.1f} M nodes with {args.flow_director}.\n"
            f"  Estimated peak RAM: ~{est_ram_gb:.0f} GB.\n"
            f"  If this is killed or runs out of memory, rerun with --flow-director D8\n"
            f"  (D8 uses ~10% of the memory of DINF/MFD for the same grid).\n"
            f"  For very large areas, consider the ArcGIS tools in arcgis_tools/.\n",
            flush=True,
        )
    # ─────────────────────────────────────────────────────────────────────────

    if use_supplied_weight:
        if user_weight is None:
            print("Error: weight raster missing after preprocessing.", file=sys.stderr)
            return 2
        ic = ConnectivityIndex(
            grid,
            flow_director=args.flow_director,
            weight=np.flipud(user_weight).ravel(),
            target_nodes=target_nodes,
            stream_threshold=args.stream_threshold,
            w_min=args.w_min,
            w_max=args.w_max,
            use_degree_approx=args.use_degree_approx,
            use_aspect_weighting=args.use_aspect_weighting,
        )
    else:
        weight_builder = WeightBuilder(combine=args.weight_combine, w_min=args.w_min, w_max=args.w_max)
        if need_rainfall:
            if rainfall is None:
                print("Error: rainfall array missing after preprocessing.", file=sys.stderr)
                return 2
            weight_builder.add(RainfallWeight(np.flipud(rainfall).ravel(), w_min=args.w_min))
        if need_ndvi:
            if ndvi is None:
                print("Error: NDVI array missing after preprocessing.", file=sys.stderr)
                return 2
            weight_builder.add(NDVIWeight(np.flipud(ndvi).ravel(), w_min=args.w_min))
        if "roughness" in factors:
            weight_builder.add(
                SurfaceRoughnessWeight(grid, w_min=args.w_min, invert=args.roughness_invert)
            )

        ic = ConnectivityIndex(
            grid,
            flow_director=args.flow_director,
            weight=weight_builder,
            target_nodes=target_nodes,
            stream_threshold=args.stream_threshold,
            w_min=args.w_min,
            w_max=args.w_max,
            use_degree_approx=args.use_degree_approx,
            use_aspect_weighting=args.use_aspect_weighting,
        )

    _show_progress(3, total_steps, "Running IC computation")
    try:
        ic.run_one_step()
    except MemoryError:
        print(
            "\nMemoryError: not enough contiguous RAM to complete the computation.\n"
            "  → Rerun with --flow-director D8 (uses ~10% of DINF/MFD memory).\n"
            "  → Or use the ArcGIS tools in arcgis_tools/ for large areas.\n"
            "  → See TROUBLESHOOTING.md for a full explanation.",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out_dir)
    outputs = args.outputs
    _show_progress(4, total_steps, "Writing outputs")
    for key in outputs:
        field = _FIELD_MAP[key]
        arr = np.flipud(grid.at_node[field].reshape(dem.shape))
        out_path = out_dir / f"{args.prefix}{key}.tif"
        _write_raster(out_path, arr, dem_profile)

    _show_progress(5, total_steps, "Done")
    print(f"Wrote {len(outputs)} raster(s) to {out_dir}")
    return 0


def _gui_command(args) -> int:
    if args.show_welcome:
        _print_welcome()

    if args.backend == "streamlit":
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).parent / "gui" / "streamlit_app.py"),
        ]
        try:
            return subprocess.call(cmd)
        except FileNotFoundError:
            print(
                "Error: streamlit is not installed. Install it with: pip install streamlit",
                file=sys.stderr,
            )
            return 2

    print("Supported GUI backends: streamlit")
    return 0


def _welcome_command(_args) -> int:
    _print_welcome()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geomorphconn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_welcome_text(),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"geomorphconn { _PKG_VERSION }",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Run IC from DEM and selected weight-factor rasters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_welcome_text() + "Run IC computation from local raster files.\n",
    )
    run_p.add_argument("--dem", required=True, help="Path to DEM GeoTIFF")
    run_p.add_argument("--ndvi", required=False, help="Path to NDVI GeoTIFF (required if 'ndvi' factor is selected)")
    run_p.add_argument(
        "--weight-raster",
        required=False,
        help="Optional user-supplied W raster (overrides computed factor combination)",
    )
    run_p.add_argument(
        "--rainfall",
        required=False,
        help="Path to rainfall GeoTIFF (required if 'rainfall' factor is selected)",
    )
    run_p.add_argument(
        "--weight-factors",
        nargs="+",
        choices=["rainfall", "ndvi", "roughness"],
        default=["rainfall", "ndvi"],
        help="Weight factors to include (single, pairwise, or all three)",
    )
    run_p.add_argument(
        "--weight-combine",
        default="mean",
        choices=["mean", "arithmetic_mean", "geometric_mean", "product", "max", "min", "weighted_mean"],
        help="How to combine selected weight factors",
    )
    run_p.add_argument(
        "--roughness-invert",
        action="store_true",
        help="Invert TRI roughness scaling (higher TRI -> lower W instead of higher W)",
    )
    run_p.add_argument("--w-min", type=float, default=0.005, help="Lower clamp for W and S")
    run_p.add_argument("--w-max", type=float, default=1.0, help="Upper clamp for W")
    run_p.add_argument(
        "--auto-reproject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically align raster grids to a reference grid using rioxarray.reproject_match",
    )
    run_p.add_argument(
        "--reference-grid",
        choices=["dem", "ndvi", "rainfall", "weight"],
        default="dem",
        help="Reference raster grid for automatic alignment",
    )
    run_p.add_argument("--target-vector", default=None, help="Optional target vector path")
    run_p.add_argument(
        "--target-buffer",
        type=float,
        default=0.0,
        help="Optional target buffer in meters; if omitted, narrow line targets are auto-buffered by about half a cell",
    )
    run_p.add_argument(
        "--all-touched",
        action="store_true",
        help="When rasterizing target vectors, mark all touched cells instead of only cell centers",
    )
    run_p.add_argument(
        "--stream-threshold",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Auto-define channel network from D8 flow accumulation: cells where "
            "upstream cell count >= N are treated as targets (Borselli 2008 recipe). "
            "Typical values for 30 m grids: 500-2000. "
            "Can be combined with --target-vector (union is taken)."
        ),
    )
    run_p.add_argument("--flow-director", default="DINF", choices=["D8", "DINF", "MFD"], help="Upstream flow director")
    run_p.add_argument(
        "--use-degree-approx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use S = slope_degrees/100 (default); disable to use S = tan(theta)",
    )
    run_p.add_argument(
        "--use-aspect-weighting",
        action="store_true",
        help="Enable TauDEM-style partition weighting for multi-receiver upstream accumulation",
    )
    run_p.add_argument("--out-dir", default="outputs", help="Output directory")
    run_p.add_argument("--prefix", default="ic_", help="Output filename prefix")
    run_p.add_argument(
        "--outputs",
        nargs="+",
        default=["IC"],
        choices=sorted(_FIELD_MAP.keys()),
        help="Output rasters to write",
    )
    run_p.add_argument(
        "--show-welcome",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show or hide welcome/about banner before processing",
    )
    run_p.set_defaults(func=_run_command)

    gui_p = sub.add_parser("gui", help="Launch GUI")
    gui_p.add_argument("--backend", default="streamlit", choices=["streamlit"])
    gui_p.add_argument(
        "--show-welcome",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show or hide welcome/about banner before GUI launch",
    )
    gui_p.set_defaults(func=_gui_command)

    welcome_p = sub.add_parser("welcome", help="Show welcome/about screen")
    welcome_p.set_defaults(func=_welcome_command)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
