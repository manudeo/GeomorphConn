from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine

from ..weights import CustomWeight, NDVIWeight, RainfallWeight, WeightBuilder
from ..weights.components import compute_surface_roughness_weight_2d


def _is_wsl() -> bool:
    return ("microsoft" in platform.release().lower()) or ("WSL_DISTRO_NAME" in os.environ)


def _windows_path_to_wsl(path: str | None) -> str | None:
    if not path:
        return None
    p = str(path)
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if not match:
        return p
    drive = match.group(1).lower()
    tail = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def _candidate_bin_dirs(bin_dir: str | None) -> list[str]:
    dirs: list[str] = []
    if bin_dir:
        dirs.append(str(Path(bin_dir)))
        converted = _windows_path_to_wsl(bin_dir)
        if converted and converted not in dirs:
            dirs.append(converted)

    if os.name == "nt":
        dirs.extend(
            [
                r"C:\Program Files\TauDEM\TauDEM5Exe",
                r"C:\Program Files\Microsoft MPI\Bin",
            ]
        )

    if _is_wsl():
        dirs.extend(
            [
                "/mnt/c/Program Files/TauDEM/TauDEM5Exe",
                "/mnt/c/Program Files/Microsoft MPI/Bin",
                "/mnt/c/Program Files/GDAL",
                "/mnt/c/OSGeo4W64/bin",
            ]
        )

    return [d for i, d in enumerate(dirs) if d and d not in dirs[:i]]


def _resolve_executable(candidates: list[str], bin_dir: str | None) -> str | None:
    search_paths = _candidate_bin_dirs(bin_dir)
    for name in candidates:
        for base in search_paths:
            p = Path(base) / name
            if p.exists():
                return str(p)
            if os.name == "nt" and not str(p).lower().endswith(".exe"):
                p_exe = Path(str(p) + ".exe")
                if p_exe.exists():
                    return str(p_exe)
        found = shutil.which(name)
        if found:
            return found
    return None


def check_taudem_installation(taudem_bin_dir: str | None = None) -> dict[str, Any]:
    """Return a TauDEM/MPI availability report for the current environment."""
    exe_map = {
        "mpiexec": ["mpiexec", "mpiexec.exe"],
        "PitRemove": ["PitRemove", "PitRemove.exe", "pitremove"],
        "D8FlowDir": ["D8FlowDir", "D8FlowDir.exe", "d8flowdir"],
        "DinfFlowDir": ["DinfFlowDir", "DinfFlowDir.exe", "dinfflowdir"],
        "AreaD8": ["AreaD8", "AreaD8.exe", "aread8"],
        "AreaDinf": ["AreaDinf", "AreaDinf.exe", "areadinf"],
    }
    found = {key: _resolve_executable(names, taudem_bin_dir) for key, names in exe_map.items()}
    missing = [key for key, value in found.items() if value is None]
    return {
        "platform": platform.platform(),
        "os_name": os.name,
        "is_wsl": _is_wsl(),
        "search_dirs": _candidate_bin_dirs(taudem_bin_dir),
        "requested_bin_dir": taudem_bin_dir,
        "executables": found,
        "missing": missing,
        "ok": len(missing) == 0,
    }


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr or stdout or "no process output"
        raise RuntimeError(f"TauDEM command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")


def _read_float_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr


def _write_float_raster(path: Path, arr: np.ndarray, profile: dict[str, Any], nodata: float = -9999.0) -> None:
    out_profile = {
        **profile,
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "nodata": nodata,
    }
    out = np.asarray(arr, dtype=np.float32).copy()
    out[~np.isfinite(out)] = float(nodata)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)


def _ll_nodes_to_geo_flat_indices(nodes: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
    rows_south = nodes // ncols
    cols = nodes % ncols
    rows_geo = (nrows - 1) - rows_south
    return (rows_geo * ncols + cols).astype(np.int64)


def _receivers_from_taudem_d8(p_arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    nrows, ncols = p_arr.shape
    n = nrows * ncols
    recv = np.arange(n, dtype=np.int64)

    # TauDEM D8 coding:
    # 4 3 2
    # 5 - 1
    # 6 7 8
    offsets = {
        1: (0, 1),
        2: (-1, 1),
        3: (-1, 0),
        4: (-1, -1),
        5: (0, -1),
        6: (1, -1),
        7: (1, 0),
        8: (1, 1),
    }

    flat_valid = valid_mask.ravel()
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c
            if not flat_valid[idx]:
                recv[idx] = idx
                continue
            code = int(p_arr[r, c]) if np.isfinite(p_arr[r, c]) else -1
            if code not in offsets:
                recv[idx] = idx
                continue
            dr, dc = offsets[code]
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                recv[idx] = idx
                continue
            rec_idx = rr * ncols + cc
            if not flat_valid[rec_idx]:
                recv[idx] = idx
                continue
            recv[idx] = rec_idx
    return recv


def _topological_order_d8(receivers: np.ndarray) -> np.ndarray:
    n = receivers.size
    indeg = np.zeros(n, dtype=np.int64)
    idx = np.arange(n, dtype=np.int64)
    m = receivers != idx
    np.add.at(indeg, receivers[m], 1)

    queue = list(np.where(indeg == 0)[0].astype(np.int64))
    order: list[int] = []
    while queue:
        node = int(queue.pop())
        order.append(node)
        rec = int(receivers[node])
        if rec != node:
            indeg[rec] -= 1
            if indeg[rec] == 0:
                queue.append(rec)

    if len(order) < n:
        present = np.zeros(n, dtype=bool)
        present[np.asarray(order, dtype=np.int64)] = True
        remaining = np.where(~present)[0].astype(np.int64).tolist()
        order.extend(remaining)

    return np.asarray(order, dtype=np.int64)


def _acc_d8(weight: np.ndarray, receivers: np.ndarray, node_order: np.ndarray) -> np.ndarray:
    acc = np.zeros(len(weight), dtype=np.float64)
    for node in node_order:
        rec = int(receivers[node])
        if rec != int(node):
            acc[rec] += acc[node] + weight[node]
    return acc


def _ddn_weighted_flow_length_d8(
    dist_to_receiver: np.ndarray,
    inv_ws: np.ndarray,
    receivers: np.ndarray,
    node_order: np.ndarray,
) -> np.ndarray:
    """SedInConnect-style D8 weighted flow-length propagation for D_dn."""
    n = receivers.size
    _ = node_order  # kept for API compatibility

    upstream: list[list[int]] = [[] for _ in range(n)]
    idx = np.arange(n, dtype=np.int64)
    for node in range(n):
        rec = int(receivers[node])
        if rec >= 0 and rec != int(idx[node]):
            upstream[rec].append(node)

    ddn = np.full(n, -1.0, dtype=np.float64)
    terminals = np.where(receivers == idx)[0].astype(np.int64)

    frontier: list[int] = []
    for term in terminals:
        for up in upstream[int(term)]:
            if ddn[up] < 0.0:
                ddn[up] = 0.0
                frontier.append(up)

    count = 1
    while frontier:
        nxt: list[int] = []
        for cur in frontier:
            for up in upstream[int(cur)]:
                if ddn[up] >= 0.0:
                    continue
                if count == 1:
                    ddn[up] = 0.0
                else:
                    seg = float(dist_to_receiver[up]) * (
                        (float(inv_ws[cur]) + float(inv_ws[up])) / 2.0
                    )
                    ddn[up] = float(ddn[cur]) + seg
                nxt.append(up)
        frontier = nxt
        count += 1

    ddn[ddn == 0.0] = 1.0
    ddn[ddn < 0.0] = np.nan
    return ddn


def _dominant_outlet_mask(receivers: np.ndarray, outlet_metric: np.ndarray) -> np.ndarray | None:
    n_nodes = receivers.size
    nodes = np.arange(n_nodes, dtype=np.int64)
    outlets = nodes[receivers == nodes]
    if outlets.size == 0:
        return None
    main_outlet = outlets[np.argmax(outlet_metric[outlets])]

    visited = np.full(n_nodes, -1, dtype=np.int64)

    def _terminal(start: int) -> int:
        trail = []
        cur = int(start)
        while True:
            nxt = int(receivers[cur])
            trail.append(cur)
            if nxt == cur:
                term = cur
                break
            if visited[cur] >= 0:
                term = int(visited[cur])
                break
            cur = nxt
        for t in trail:
            visited[t] = term
        return term

    for i in range(n_nodes):
        if visited[i] < 0:
            _terminal(i)
    return visited == int(main_outlet)


def run_connectivity_taudem_arrays(
    *,
    dem: np.ndarray,
    dem_profile: dict[str, Any],
    flow_director: str = "D8",
    ndvi: np.ndarray | None = None,
    rainfall: np.ndarray | None = None,
    user_weight: np.ndarray | None = None,
    weight_combine: str = "mean",
    w_min: float = 0.005,
    w_max: float = 1.0,
    target_nodes: np.ndarray | None = None,
    analysis_mask_nodes: np.ndarray | None = None,
    main_basin_only: bool = False,
    stream_threshold: int | None = None,
    use_roughness: bool = False,
    roughness_detrend_window: int = 3,
    roughness_std_window: int = 3,
    taudem_n_procs: int | None = None,
    taudem_bin_dir: str | None = None,
) -> dict[str, Any]:
    """Compute IC using TauDEM routing outputs instead of Landlab routing.

    Notes
    -----
    - TauDEM upstream accumulation follows SedInConnect's DINF path.
    - Downstream weighted flow length always follows TauDEM D8 flow directions.
    - ``flow_director='MFD'`` is not available in TauDEM here and falls back to ``'DINF'``.
    """
    requested_fd = str(flow_director).upper()
    if requested_fd not in {"D8", "DINF", "MFD"}:
        raise ValueError("TauDEM backend flow_director must be one of: D8, DINF, MFD")
    primary_fd = requested_fd
    if requested_fd == "MFD":
        warnings.warn(
            "TauDEM backend does not provide MFD here; requested flow_director='MFD' "
            "will be treated as 'DINF'.",
            UserWarning,
            stacklevel=2,
        )
        primary_fd = "DINF"

    if user_weight is None and ndvi is None and rainfall is None and not use_roughness:
        raise ValueError(
            "TauDEM backend requires one of: user_weight, ndvi, rainfall, or roughness."
        )

    nrows, ncols = dem.shape
    n_nodes = nrows * ncols

    transform = dem_profile.get("transform", Affine.identity())
    if not isinstance(transform, Affine):
        transform = Affine(*transform)
    dx = float(abs(transform.a))
    dy = float(abs(transform.e)) if transform.e != 0 else dx
    if not np.isclose(dx, dy):
        raise ValueError("TauDEM backend requires square pixels.")

    valid_mask = np.isfinite(dem)

    # Build W following the same WeightBuilder logic used elsewhere.
    if user_weight is not None:
        W = np.clip(np.asarray(user_weight, dtype=np.float64).ravel(), w_min, w_max)
    else:
        wb = WeightBuilder(combine=weight_combine, w_min=w_min, w_max=w_max)
        if rainfall is not None:
            wb.add(RainfallWeight(np.asarray(rainfall, dtype=np.float64).ravel(), w_min=w_min))
        if ndvi is not None:
            wb.add(NDVIWeight(np.asarray(ndvi, dtype=np.float64).ravel(), w_min=w_min))
        if use_roughness:
            rough_w = compute_surface_roughness_weight_2d(
                dem,
                detrend_window=roughness_detrend_window,
                std_window=roughness_std_window,
                w_min=w_min,
            ).ravel()
            wb.add(CustomWeight(rough_w, w_min=w_min, w_max=w_max))
        W = wb.build(n_nodes=n_nodes).astype(np.float64)

    n_procs = int(taudem_n_procs) if taudem_n_procs is not None else 0
    if n_procs <= 0:
        n_procs = max(1, (os.cpu_count() or 1))

    mpiexec = _resolve_executable(["mpiexec", "mpiexec.exe"], taudem_bin_dir)
    pitremove = _resolve_executable(["PitRemove", "pitremove"], taudem_bin_dir)
    d8flowdir = _resolve_executable(["D8FlowDir", "d8flowdir"], taudem_bin_dir)
    dinfflowdir = _resolve_executable(["DinfFlowDir", "dinfflowdir"], taudem_bin_dir)
    aread8 = _resolve_executable(["AreaD8", "aread8"], taudem_bin_dir)
    areadinf = _resolve_executable(["AreaDinf", "areadinf"], taudem_bin_dir)

    if not (pitremove and d8flowdir and aread8 and dinfflowdir and areadinf):
        report = check_taudem_installation(taudem_bin_dir)
        searched = ", ".join(report["search_dirs"]) if report["search_dirs"] else "PATH only"
        wsl_hint = ""
        if report["is_wsl"]:
            wsl_hint = (
                " Running under WSL: if TauDEM is installed in Windows, try "
                "taudem_bin_dir='/mnt/c/Program Files/TauDEM/TauDEM5Exe'."
            )
        raise RuntimeError(
            "TauDEM executables not found. Ensure PitRemove, D8FlowDir, DinfFlowDir, AreaD8, and AreaDinf are installed "
            f"and visible in PATH (or provide --taudem-bin-dir). Searched: {searched}.{wsl_hint}"
        )

    with tempfile.TemporaryDirectory(prefix="geomorphconn_taudem_") as td:
        tdir = Path(td)
        dem_tif = tdir / "dem.tif"
        fel_tif = tdir / "fel.tif"
        p_tif = tdir / "p.tif"
        sd8_tif = tdir / "sd8.tif"
        ad8_tif = tdir / "ad8.tif"
        ang_tif = tdir / "ang.tif"
        slp_tif = tdir / "slp.tif"
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": nrows,
            "width": ncols,
            "transform": transform,
            "crs": dem_profile.get("crs", None),
            "nodata": -9999.0,
        }

        dem_out = np.asarray(dem, dtype=np.float32).copy()
        dem_out[~np.isfinite(dem_out)] = -9999.0
        with rasterio.open(dem_tif, "w", **profile) as dst:
            dst.write(dem_out, 1)

        def _taudem_cmd(exe: str, args: list[str]) -> list[str]:
            if mpiexec is not None and n_procs > 1:
                return [mpiexec, "-n", str(n_procs), exe] + args
            return [exe] + args

        _run_cmd(_taudem_cmd(pitremove, ["-z", str(dem_tif), "-fel", str(fel_tif)]), tdir)
        _run_cmd(
            _taudem_cmd(
                d8flowdir,
                ["-fel", str(fel_tif), "-p", str(p_tif), "-sd8", str(sd8_tif)],
            ),
            tdir,
        )
        _run_cmd(
            _taudem_cmd(
                dinfflowdir,
                ["-fel", str(fel_tif), "-ang", str(ang_tif), "-slp", str(slp_tif)],
            ),
            tdir,
        )
        _run_cmd(_taudem_cmd(aread8, ["-p", str(p_tif), "-ad8", str(ad8_tif)]), tdir)

        p = _read_float_raster(p_tif)
        sd8 = _read_float_raster(sd8_tif)
        ad8 = _read_float_raster(ad8_tif)
        ang = _read_float_raster(ang_tif)

        sd8_proc = np.asarray(sd8, dtype=np.float64)
        sd8_proc[(sd8_proc >= 0.0) & (sd8_proc < w_min)] = w_min
        sd8_proc[sd8_proc > w_max] = w_max
        sd8_proc[sd8_proc < 0.0] = -1.0
    receivers = _receivers_from_taudem_d8(p, valid_mask)

    # Explicit targets are provided as Landlab node IDs; convert to GeoTIFF order.
    explicit_targets = None
    if target_nodes is not None:
        tn = np.asarray(target_nodes, dtype=np.int64).ravel()
        if np.any(tn < 0) or np.any(tn >= n_nodes):
            raise ValueError("target_nodes contains values outside valid node range")
        explicit_targets = _ll_nodes_to_geo_flat_indices(tn, nrows, ncols)

    effective_targets = None
    if explicit_targets is not None:
        effective_targets = np.unique(explicit_targets)

    if effective_targets is not None and len(effective_targets) > 0:
        receivers = receivers.copy()
        receivers[effective_targets] = effective_targets

    with tempfile.TemporaryDirectory(prefix="geomorphconn_taudem_acc_") as td2:
        tdir2 = Path(td2)
        sca_tif = tdir2 / "sca.tif"
        accw_tif = tdir2 / "accW.tif"
        accs_tif = tdir2 / "accS.tif"
        p_mod_tif = tdir2 / "p_mod.tif"
        ang_mod_tif = tdir2 / "ang_mod.tif"

        _write_float_raster(tdir2 / "weight.tif", np.asarray(W, dtype=np.float64).reshape((nrows, ncols)), profile)
        _write_float_raster(tdir2 / "s.tif", sd8_proc, profile)
        _write_float_raster(tdir2 / "p.tif", p, profile)
        _write_float_raster(tdir2 / "ang.tif", ang, profile)

        ang_run = tdir2 / "ang.tif"
        p_run = tdir2 / "p.tif"

        if effective_targets is not None and len(effective_targets) > 0:
            p_mod = np.asarray(p, dtype=np.float64).copy().reshape(-1)
            p_mod[effective_targets] = -1000.0
            p_mod = p_mod.reshape((nrows, ncols))
            _write_float_raster(p_mod_tif, p_mod, profile)
            p_run = p_mod_tif

            ang_mod = np.asarray(ang, dtype=np.float64).copy().reshape(-1)
            ang_mod[effective_targets] = -1000.0
            ang_mod = ang_mod.reshape((nrows, ncols))
            _write_float_raster(ang_mod_tif, ang_mod, profile)
            ang_run = ang_mod_tif

        # Threshold targets follow selected upstream routing metric.
        if stream_threshold is not None:
            if primary_fd == "D8":
                metric_flat = np.asarray(ad8, dtype=np.float64).ravel()
                stream_targets = np.where(np.isfinite(metric_flat) & (metric_flat >= float(stream_threshold)))[0].astype(np.int64)
            else:
                _run_cmd(_taudem_cmd(areadinf, ["-ang", str(ang_run), "-sca", str(sca_tif), "-nc"]), tdir2)
                sca_for_thresh = _read_float_raster(sca_tif)
                metric_flat = (np.asarray(sca_for_thresh, dtype=np.float64) / dx).ravel()
                stream_targets = np.where(np.isfinite(metric_flat) & (metric_flat >= float(stream_threshold)))[0].astype(np.int64)
            if effective_targets is not None and len(effective_targets) > 0:
                effective_targets = np.unique(np.concatenate([effective_targets, stream_targets]))
            elif len(stream_targets) > 0:
                effective_targets = np.unique(stream_targets)

            if len(stream_targets) > 0:
                receivers = receivers.copy()
                receivers[stream_targets] = stream_targets

                p_mod = np.asarray(p, dtype=np.float64).copy().reshape(-1)
                p_mod[effective_targets] = -1000.0
                p_mod = p_mod.reshape((nrows, ncols))
                _write_float_raster(p_mod_tif, p_mod, profile)
                p_run = p_mod_tif

                ang_mod = np.asarray(ang, dtype=np.float64).copy().reshape(-1)
                ang_mod[effective_targets] = -1000.0
                ang_mod = ang_mod.reshape((nrows, ncols))
                _write_float_raster(ang_mod_tif, ang_mod, profile)
                ang_run = ang_mod_tif

        # Upstream accumulation: DINF path (SedInConnect pattern) unless D8 explicitly requested.
        if primary_fd == "D8":
            _run_cmd(_taudem_cmd(aread8, ["-p", str(p_run), "-ad8", str(sca_tif)]), tdir2)
            sca = _read_float_raster(sca_tif)
            cell_count = np.asarray(sca, dtype=np.float64)

            # Weighted D8 upstream accumulation uses local topological propagation in-memory.
            order = _topological_order_d8(receivers)
            S = np.asarray(sd8_proc, dtype=np.float64).ravel().copy()
            S[(S >= 0.0) & (S < w_min)] = w_min
            S[S > w_max] = w_max
            S[S < 0.0] = -1.0
            valid_flat = valid_mask.ravel()
            ones = valid_flat.astype(np.float64)
            acc_w = _acc_d8(W, receivers, order)
            acc_s = _acc_d8(S, receivers, order)
            acc_a = _acc_d8(ones, receivers, order)
            acc_final = acc_a + ones
        else:
            _run_cmd(_taudem_cmd(areadinf, ["-ang", str(ang_run), "-sca", str(sca_tif), "-nc"]), tdir2)
            _run_cmd(_taudem_cmd(areadinf, ["-ang", str(ang_run), "-sca", str(accw_tif), "-wg", str(tdir2 / 'weight.tif'), "-nc"]), tdir2)
            _run_cmd(_taudem_cmd(areadinf, ["-ang", str(ang_run), "-sca", str(accs_tif), "-wg", str(tdir2 / 's.tif'), "-nc"]), tdir2)

            sca = _read_float_raster(sca_tif)
            accw = _read_float_raster(accw_tif)
            accs = _read_float_raster(accs_tif)

            acc_final = np.asarray(sca, dtype=np.float64).ravel() / dx
            acc_w = np.asarray(accw, dtype=np.float64).ravel()
            acc_s = np.asarray(accs, dtype=np.float64).ravel()

            S = np.asarray(sd8_proc, dtype=np.float64).ravel().copy()
            S[(S >= 0.0) & (S < w_min)] = w_min
            S[S > w_max] = w_max
            S[S < 0.0] = -1.0

    order = _topological_order_d8(receivers)
    valid_flat = valid_mask.ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        wmean = (acc_w + W) / acc_final
        smean = (acc_s + S) / acc_final

        cell_area = dx * dy
        area = acc_final * cell_area
        dup = wmean * smean * np.sqrt(area)

    rows = np.arange(n_nodes, dtype=np.int64) // ncols
    cols = np.arange(n_nodes, dtype=np.int64) % ncols
    rrec = receivers // ncols
    crec = receivers % ncols
    dist = np.sqrt(((rrec - rows) * dy) ** 2 + ((crec - cols) * dx) ** 2)
    dist[receivers == np.arange(n_nodes, dtype=np.int64)] = 0.0

    inv_ws = 1.0 / (W * S)
    ddn = _ddn_weighted_flow_length_d8(dist, inv_ws, receivers, order)

    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.where((dup > 0) & (ddn > 0), np.log10(dup / ddn), np.nan)

    # Boundary and nodata masking to mimic Landlab outputs.
    boundary = np.zeros((nrows, ncols), dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    boundary_flat = boundary.ravel()

    analysis_mask_bool = None
    if analysis_mask_nodes is not None:
        mn = np.asarray(analysis_mask_nodes, dtype=np.int64).ravel()
        if np.any(mn < 0) or np.any(mn >= n_nodes):
            raise ValueError("analysis_mask_nodes contains values outside valid node range")
        mgeo = _ll_nodes_to_geo_flat_indices(mn, nrows, ncols)
        analysis_mask_bool = np.zeros(n_nodes, dtype=bool)
        analysis_mask_bool[mgeo] = True

    if main_basin_only:
        outlet_metric = np.nan_to_num(np.asarray(ad8, dtype=np.float64).ravel(), nan=0.0)
        dom_mask = _dominant_outlet_mask(receivers, outlet_metric)
        if dom_mask is not None:
            analysis_mask_bool = dom_mask if analysis_mask_bool is None else (analysis_mask_bool & dom_mask)

    outside = ~valid_flat | boundary_flat
    if analysis_mask_bool is not None:
        outside = outside | (~analysis_mask_bool)

    if effective_targets is not None and len(effective_targets) > 0:
        outside = outside | np.isin(np.arange(n_nodes, dtype=np.int64), effective_targets)

    for arr in (ic, dup, ddn, W, S, wmean, smean, acc_final):
        arr[outside] = np.nan

    def _as_2d(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64).reshape((nrows, ncols))

    layers = {
        "IC": _as_2d(ic),
        "Dup": _as_2d(dup),
        "Ddn": _as_2d(ddn),
        "W": _as_2d(W),
        "S": _as_2d(S),
        "Wmean": _as_2d(wmean),
        "Smean": _as_2d(smean),
        "ACCfinal": _as_2d(acc_final),
    }

    return {
        "layers": layers,
        "effective_target_nodes_geo": effective_targets,
        "backend": "taudem",
        "routing": primary_fd,
    }
