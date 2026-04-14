"""Command line interface for GeomorphConn."""

from __future__ import annotations

import argparse
import datetime
import math
import subprocess
import sys
import gc
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from landlab import RasterModelGrid
from rasterio.transform import Affine

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
    "ACCfinal": "connectivity_index__ACCfinal",
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
        "Script author:\n"
        "- Manudeo Singh\n"
        "\n"
        "Method origin credits:\n"
        "- IC methodology: Cavalli et al. (2013)\n"
        "- SedInConnect reference implementation: Crema & Cavalli (2018)\n"
    )


def _print_welcome() -> None:
    print(_welcome_text())


def _load_aligned_rasters(
    dem_path: Path,
    ndvi_path: Path | None,
    rainfall_path: Path | None,
    weight_path: Path | None,
    mask_path: Path | None,
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
        if mask_path is not None:
            ds_map["mask"] = _open_da(mask_path)

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


def _coarsen_rasters(arrs: dict[str, np.ndarray | None], factor: int, profile: dict):
    """Block-mean coarsen all raster arrays by integer factor."""
    if factor <= 1:
        return arrs, profile

    def _block_nanmean(arr: np.ndarray) -> np.ndarray:
        r, c = arr.shape
        nr, nc = r // factor, c // factor
        if nr == 0 or nc == 0:
            raise ValueError(
                f"DEM coarsen factor {factor} is too large for raster shape {r}x{c}. "
                f"Choose a factor no greater than {min(r, c)}."
            )
        trimmed = arr[: nr * factor, : nc * factor]
        blocks = trimmed.reshape(nr, factor, nc, factor)
        valid = np.isfinite(blocks)
        counts = valid.sum(axis=(1, 3))
        sums = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
        out = np.full((nr, nc), np.nan, dtype=np.float64)
        np.divide(sums, counts, out=out, where=counts > 0)
        return out

    coarsened: dict[str, np.ndarray | None] = {}
    for key, arr in arrs.items():
        if arr is None:
            coarsened[key] = None
            continue
        coarsened[key] = _block_nanmean(arr)

    old_t = profile["transform"]
    new_t = Affine(old_t.a * factor, old_t.b, old_t.c, old_t.d, old_t.e * factor, old_t.f)
    ref_shape = next(v.shape for v in coarsened.values() if v is not None)
    new_profile = {
        **profile,
        "transform": new_t,
        "width": ref_shape[1],
        "height": ref_shape[0],
    }
    return coarsened, new_profile


def _write_cli_run_params_txt(
    txt_path: Path,
    *,
    args,
    dem_transform,
    dem_crs,
    dem_shape: tuple,
    all_layers: dict,
) -> None:
    """Write a plain-text run summary (parameters + layer stats) to *txt_path*."""
    lines: list[str] = []
    lines.append(f"GeomorphConn v{_PKG_VERSION} \u2013 Run Summary")
    lines.append("=" * 45)
    lines.append(f"Run date/time     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("--- Input files ---")
    lines.append(f"DEM               : {args.dem}")
    lines.append(f"NDVI              : {args.ndvi or 'N/A'}")
    lines.append(f"Rainfall          : {args.rainfall or 'N/A'}")
    lines.append(f"Weight raster     : {args.weight_raster or 'N/A'}")
    lines.append(f"Main basin mask   : {args.main_basin_mask or 'N/A'}")
    lines.append(f"Main basin only   : {args.main_basin_only}")
    lines.append("")
    lines.append("--- Parameters ---")
    lines.append(f"Flow director     : {args.flow_director}")
    lines.append(f"Fill sinks        : {args.fill_sinks}")
    lines.append(f"Depression finder : {args.depression_finder}")
    if args.weight_raster:
        lines.append("Weight factors    : N/A (supplied weight raster)")
    else:
        lines.append(f"Weight factors    : {', '.join(args.weight_factors)}")
        lines.append(f"Weight combine    : {args.weight_combine}")
        if "roughness" in (args.weight_factors or []):
            lines.append(f"  Roughness detrend window : {args.roughness_detrend_window}")
            lines.append(f"  Roughness std window     : {args.roughness_std_window}")
    lines.append(f"w_min             : {args.w_min}")
    lines.append(f"w_max             : {args.w_max}")
    lines.append(f"Aspect weighting  : {args.use_aspect_weighting}")
    lines.append(f"DEM coarsen       : {args.dem_coarsen_factor}\u00d7")
    lines.append(f"Stream threshold  : {args.stream_threshold or 'N/A'}")
    lines.append(f"Target vector     : {args.target_vector or 'N/A'}")
    lines.append(f"Auto reproject    : {args.auto_reproject}")
    lines.append(f"Reference grid    : {args.reference_grid}")
    lines.append("")
    lines.append("--- Output grid ---")
    lines.append(f"Grid shape        : {dem_shape[0]} \u00d7 {dem_shape[1]} (rows \u00d7 cols)")
    lines.append(
        f"Cell size         : {abs(dem_transform.a):.4f} (x) \u00d7 {abs(dem_transform.e):.4f} (y)"
    )
    lines.append(f"CRS               : {dem_crs or 'unknown'}")
    lines.append(f"Output prefix     : {args.prefix}")
    lines.append(f"Output directory  : {args.out_dir}")
    lines.append("")
    lines.append("--- Output layer statistics ---")
    _hdr = (
        f"{'Layer':<12} {'Valid_cells':>12} {'Min':>10} {'Max':>10}"
        f" {'Mean':>10} {'Std':>10} {'Median':>10}"
    )
    lines.append(_hdr)
    lines.append("-" * len(_hdr))
    for key, arr in all_layers.items():
        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            lines.append(
                f"{key:<12} {valid.size:>12d} {float(np.min(valid)):>10.4f}"
                f" {float(np.max(valid)):>10.4f} {float(np.mean(valid)):>10.4f}"
                f" {float(np.std(valid)):>10.4f} {float(np.median(valid)):>10.4f}"
            )
        else:
            lines.append(
                f"{key:<12} {'0':>12} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
            )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    if ("roughness" in factors) and (not use_supplied_weight):
        if args.roughness_detrend_window < 1 or (args.roughness_detrend_window % 2) == 0:
            print("Error: --roughness-detrend-window must be a positive odd integer.", file=sys.stderr)
            return 2
        if args.roughness_std_window < 1 or (args.roughness_std_window % 2) == 0:
            print("Error: --roughness-std-window must be a positive odd integer.", file=sys.stderr)
            return 2

    ndvi_path = Path(args.ndvi) if args.ndvi else None
    rainfall_path = Path(args.rainfall) if args.rainfall else None
    weight_path = Path(args.weight_raster) if args.weight_raster else None
    mask_path = Path(args.main_basin_mask) if args.main_basin_mask else None

    _show_progress(1, total_steps, "Loading/preprocessing rasters")

    if args.auto_reproject:
        try:
            arrays, dem_profile, dem_transform, dem_crs = _load_aligned_rasters(
                dem_path,
                ndvi_path,
                rainfall_path,
                weight_path,
                mask_path,
                args.reference_grid,
            )
        except Exception as exc:
            print(f"Error during raster alignment: {exc}", file=sys.stderr)
            return 2
        dem = arrays["dem"]
        ndvi = arrays.get("ndvi")
        rainfall = arrays.get("rainfall")
        user_weight = arrays.get("weight")
        mask_arr = arrays.get("mask")
    else:
        dem, dem_profile, dem_transform, dem_crs = _read_raster(dem_path)
        ndvi = None
        rainfall = None
        user_weight = None
        mask_arr = None
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
        if mask_path is not None:
            mask_arr, _, _, _ = _read_raster(mask_path)
            if dem.shape != mask_arr.shape:
                print(
                    "Error: DEM and main basin mask must have identical shape when --no-auto-reproject is used.",
                    file=sys.stderr,
                )
                return 2

    dx = float(abs(dem_transform.a))
    dy = float(abs(dem_transform.e))
    if not np.isclose(dx, dy):
        print("Error: non-square pixels are not supported by this CLI.", file=sys.stderr)
        return 2

    if args.dem_coarsen_factor > 1:
        arr_dict = {
            "dem": dem,
            "ndvi": ndvi,
            "rainfall": rainfall,
            "weight": user_weight,
            "mask": mask_arr,
        }
        try:
            arr_dict, dem_profile = _coarsen_rasters(arr_dict, int(args.dem_coarsen_factor), dem_profile)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        dem = arr_dict["dem"]
        ndvi = arr_dict["ndvi"]
        rainfall = arr_dict["rainfall"]
        user_weight = arr_dict["weight"]
        mask_arr = arr_dict["mask"]
        dem_transform = dem_profile["transform"]
        dx = float(abs(dem_transform.a))

    grid = RasterModelGrid(dem.shape, xy_spacing=dx)
    grid.add_field("topographic__elevation", np.flipud(dem).ravel(), at="node")

    analysis_mask_nodes = None
    if mask_arr is not None:
        mask_bool = np.isfinite(mask_arr) & (mask_arr > 0.5)
        if not np.any(mask_bool):
            print("Error: main basin mask contains no valid (>0.5) cells.", file=sys.stderr)
            return 2
        analysis_mask_nodes = np.where(np.flipud(mask_bool).ravel())[0].astype(np.int64)

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

    if args.depression_finder == "DepressionFinderAndRouter":
        print(
            "Note: DepressionFinderAndRouter is enabled. This usually improves routing quality "
            "in depressed/flat terrain, but can increase runtime (especially for high-resolution DEMs).",
            flush=True,
        )

    if use_supplied_weight:
        if user_weight is None:
            print("Error: weight raster missing after preprocessing.", file=sys.stderr)
            return 2
        ic = ConnectivityIndex(
            grid,
            flow_director=args.flow_director,
            weight=np.flipud(user_weight).ravel(),
            target_nodes=target_nodes,
            analysis_mask_nodes=analysis_mask_nodes,
            main_basin_only=args.main_basin_only,
            stream_threshold=args.stream_threshold,
            fill_sinks=args.fill_sinks,
            depression_finder=None if args.depression_finder == "none" else args.depression_finder,
            w_min=args.w_min,
            w_max=args.w_max,
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
                SurfaceRoughnessWeight(
                    grid,
                    detrend_window=args.roughness_detrend_window,
                    std_window=args.roughness_std_window,
                    w_min=args.w_min,
                )
            )

        ic = ConnectivityIndex(
            grid,
            flow_director=args.flow_director,
            weight=weight_builder,
            target_nodes=target_nodes,
            analysis_mask_nodes=analysis_mask_nodes,
            main_basin_only=args.main_basin_only,
            stream_threshold=args.stream_threshold,
            fill_sinks=args.fill_sinks,
            depression_finder=None if args.depression_finder == "none" else args.depression_finder,
            w_min=args.w_min,
            w_max=args.w_max,
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
    if "all" in outputs:
        outputs = sorted(_FIELD_MAP.keys())
    _show_progress(4, total_steps, "Writing outputs")

    # Collect all 8 output arrays (for PNG and statistics, regardless of --outputs selection)
    all_layers: dict = {
        key: np.flipud(grid.at_node[field].reshape(dem.shape))
        for key, field in _FIELD_MAP.items()
    }

    # Write only the requested TIFs
    for key in outputs:
        out_path = out_dir / f"{args.prefix}{key}.tif"
        _write_raster(out_path, all_layers[key], dem_profile)

    # Write composite preview PNG (all 8 layers in a 4×2 grid)
    try:
        import matplotlib.pyplot as plt

        import matplotlib.colors as mcolors

        _log_keys = {"ACCfinal", "Dup"}

        def _imshow_kw(key: str, arr: np.ndarray) -> dict:
            valid = arr[np.isfinite(arr)]
            if key in _log_keys:
                vmin = float(np.percentile(valid, 2)) if valid.size > 0 else None
                vmax = float(np.percentile(valid, 98)) if valid.size > 0 else None
                vmin = max(vmin, 1e-6) if vmin is not None and vmin <= 0 else vmin
                vmax = max(vmax, (vmin or 1e-6) * 10) if vmax is not None and vmax <= (vmin or 0) else vmax
                try:
                    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                except Exception:
                    norm = None
                return {"norm": norm, "cmap": "viridis"}
            else:
                if valid.size > 0:
                    vmin = float(np.percentile(valid, 2))
                    vmax = float(np.percentile(valid, 98))
                else:
                    vmin = vmax = None
                return {"vmin": vmin, "vmax": vmax, "cmap": "viridis"}

        ncols = 4
        nrows = math.ceil(len(_FIELD_MAP) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        for ax, (key, arr) in zip(np.array(axes).flatten(), all_layers.items()):
            masked = np.ma.masked_invalid(arr)
            kw = _imshow_kw(key, arr)
            im = ax.imshow(masked, origin="upper", **kw)
            ax.set_title(key, fontsize=9)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        fig.suptitle("GeomorphConn \u2013 output layers", fontsize=11)
        fig.tight_layout()
        png_path = out_dir / f"{args.prefix}preview.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved preview PNG  : {png_path}")
    except Exception as exc:
        print(f"Warning: could not save preview PNG: {exc}", file=sys.stderr)

    # Write parameter + statistics summary (.txt with same stem as IC output)
    txt_path = out_dir / f"{args.prefix}IC.txt"
    try:
        _write_cli_run_params_txt(
            txt_path,
            args=args,
            dem_transform=dem_transform,
            dem_crs=dem_crs,
            dem_shape=dem.shape,
            all_layers=all_layers,
        )
        print(f"Saved run summary  : {txt_path}")
    except Exception as exc:
        print(f"Warning: could not write run summary: {exc}", file=sys.stderr)

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
        "--roughness-detrend-window",
        type=int,
        default=3,
        help="Odd moving-window size for DEM local-mean detrending in roughness calculation",
    )
    run_p.add_argument(
        "--roughness-std-window",
        type=int,
        default=3,
        help="Odd moving-window size for residual standard deviation in roughness calculation",
    )
    run_p.add_argument("--w-min", type=float, default=0.005, help="Lower clamp for W and S")
    run_p.add_argument("--w-max", type=float, default=1.0, help="Upper clamp for W")
    run_p.add_argument(
        "--dem-coarsen-factor",
        type=int,
        choices=[1, 2, 4, 8],
        default=1,
        help="Integer coarsening factor for all input rasters before IC computation",
    )
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
        "--main-basin-mask",
        default=None,
        help=(
            "Optional basin-mask raster. Cells > 0 are treated as the analysis domain; "
            "stream-threshold targets are restricted to this mask and outputs outside are NoData."
        ),
    )
    run_p.add_argument(
        "--main-basin-only",
        action="store_true",
        help=(
            "Restrict analysis to the outlet basin footprint inferred from outlet-style D_dn support "
            "(with dominant-outlet fallback). Useful in target mode to exclude neighbouring catchments "
            "without supplying a mask raster."
        ),
    )
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
            "Auto-define channel network from selected flow-director accumulation: cells where "
            "upstream cell count >= N are treated as targets (Borselli 2008 recipe). "
            "Typical values for 30 m grids: 500-2000. "
            "Can be combined with --target-vector (union is taken)."
        ),
    )
    run_p.add_argument("--flow-director", default="DINF", choices=["D8", "DINF", "MFD"], help="Upstream flow director")
    run_p.add_argument(
        "--fill-sinks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply SinkFillerBarnes before routing to remove pits/flats by modifying DEM elevations. "
            "Disabled by default."
        ),
    )
    run_p.add_argument(
        "--depression-finder",
        default="DepressionFinderAndRouter",
        choices=["DepressionFinderAndRouter", "none"],
        help=(
            "Depression-handling method passed to Landlab's FlowAccumulator D8 stage. "
            '"DepressionFinderAndRouter" (default) routes flow through depressions without '
            "modifying the DEM, giving better visual results than --fill-sinks. "
            '"none" disables depression handling entirely.'
        ),
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
        choices=sorted(_FIELD_MAP.keys()) + ["all"],
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
