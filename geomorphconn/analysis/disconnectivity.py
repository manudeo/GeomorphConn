"""
Disconnectivity-style hierarchy analysis for connectivity rasters.

This module adapts ideas from metric disconnectivity graphs (Smeeton et al.,
J. Comput. Chem., 2014) to 2D connectivity maps such as IC rasters.

Core ideas implemented:
- Hierarchical components across increasing IC thresholds
- Parent-child links across adjacent thresholds (maximum-overlap criterion)
- Node-level comparison metrics against alternative methods
    (RMSd-like RMSE, contact-like sign agreement, PCA-like summary)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import xarray as xr
from scipy import ndimage


def _validate_2d_da(name: str, da: xr.DataArray) -> None:
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray.")
    if da.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape {da.shape}.")


def _component_rows(ic: xr.DataArray, labels: np.ndarray, threshold: float) -> List[dict]:
    rows: List[dict] = []
    values = ic.values
    yy, xx = np.indices(values.shape)
    for comp_id in np.unique(labels):
        if comp_id <= 0:
            continue
        mask = labels == comp_id
        comp_vals = values[mask]
        rows.append(
            {
                "threshold": float(threshold),
                "component_id": int(comp_id),
                "node_id": f"T{threshold:.6f}_C{int(comp_id)}",
                "n_pixels": int(mask.sum()),
                "mean_ic": float(np.nanmean(comp_vals)),
                "median_ic": float(np.nanmedian(comp_vals)),
                "std_ic": float(np.nanstd(comp_vals)),
                "cx": float(np.mean(xx[mask])),
                "cy": float(np.mean(yy[mask])),
                "bbox_xmin": int(np.min(xx[mask])),
                "bbox_xmax": int(np.max(xx[mask])),
                "bbox_ymin": int(np.min(yy[mask])),
                "bbox_ymax": int(np.max(yy[mask])),
            }
        )
    return rows


def build_disconnectivity_hierarchy(
    ic_map: xr.DataArray,
    quantiles: Optional[Iterable[float]] = None,
    eight_connected: bool = True,
) -> dict:
    """
    Build threshold hierarchy and parent-child links for a 2D connectivity map.

    Parameters
    ----------
    ic_map : xr.DataArray
        2D connectivity map.
    quantiles : iterable of float, optional
        Quantiles used to generate thresholds from finite values.
    eight_connected : bool, optional
        If True, use 8-neighborhood connectivity, else 4-neighborhood.

    Returns
    -------
    dict
        {
          "nodes": list[dict],
          "links": list[dict],
          "thresholds": list[float],
          "levels": list[{"threshold": float, "labels": np.ndarray}],
          "shape": tuple[int, int],
        }
    """
    _validate_2d_da("ic_map", ic_map)

    if quantiles is None:
        quantiles = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

    q = np.array(list(quantiles), dtype=float)
    if q.size == 0:
        raise ValueError("quantiles must contain at least one value.")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("quantiles must be in [0, 1].")

    values = ic_map.values
    finite = np.isfinite(values)
    if not np.any(finite):
        return {
            "nodes": [],
            "links": [],
            "thresholds": [],
            "levels": [],
            "shape": tuple(values.shape),
        }

    thresholds = [float(np.quantile(values[finite], x)) for x in q]
    # Preserve order but remove duplicates caused by degenerate data.
    thresholds = list(dict.fromkeys(thresholds))

    conn = 2 if eight_connected else 1
    structure = ndimage.generate_binary_structure(2, conn)

    levels: List[dict] = []
    nodes: List[dict] = []

    for threshold in thresholds:
        mask = finite & (values >= threshold)
        labels, _ = ndimage.label(mask, structure=structure)
        levels.append({"threshold": float(threshold), "labels": labels})
        nodes.extend(_component_rows(ic_map, labels, threshold))

    links: List[dict] = []
    for level_idx in range(len(levels) - 1):
        lo = levels[level_idx]
        hi = levels[level_idx + 1]
        lo_labels = lo["labels"]
        hi_labels = hi["labels"]

        for child_comp_id in np.unique(hi_labels):
            if child_comp_id <= 0:
                continue
            child_mask = hi_labels == child_comp_id
            parent_candidates = lo_labels[child_mask]
            parent_candidates = parent_candidates[parent_candidates > 0]
            if parent_candidates.size == 0:
                continue
            vals, counts = np.unique(parent_candidates, return_counts=True)
            parent_comp_id = int(vals[np.argmax(counts)])
            overlap_pixels = int(np.max(counts))
            links.append(
                {
                    "parent_threshold": float(lo["threshold"]),
                    "parent_component_id": parent_comp_id,
                    "parent_node_id": f"T{lo['threshold']:.6f}_C{parent_comp_id}",
                    "child_threshold": float(hi["threshold"]),
                    "child_component_id": int(child_comp_id),
                    "child_node_id": f"T{hi['threshold']:.6f}_C{int(child_comp_id)}",
                    "overlap_pixels": overlap_pixels,
                }
            )

    return {
        "nodes": nodes,
        "links": links,
        "thresholds": thresholds,
        "levels": levels,
        "shape": tuple(values.shape),
    }


def compute_node_comparison_metrics(
    hierarchy: dict,
    reference_map: xr.DataArray,
    comparison_maps: Dict[str, xr.DataArray],
) -> List[dict]:
    """
    Compute node-level Smeeton et al. (2014)-inspired metrics for hierarchy components.

    Metrics per node include:
    - RMSE and bias vs each comparison map (RMSd-like)
    - Sign-agreement fraction vs each comparison map (contact-like)
    - PCA-like summary from all map values inside the node

    Parameters
    ----------
    hierarchy : dict
        Output of build_disconnectivity_hierarchy.
    reference_map : xr.DataArray
        2D reference map (e.g., GeomorphConn IC).
    comparison_maps : dict[str, xr.DataArray]
        Comparison maps with same shape.

    Returns
    -------
    list[dict]
        Node metrics rows keyed by node_id/threshold/component_id.
    """
    _validate_2d_da("reference_map", reference_map)
    ref = reference_map.values

    if hierarchy.get("shape") != tuple(ref.shape):
        raise ValueError(
            f"Hierarchy shape {hierarchy.get('shape')} does not match reference_map {ref.shape}."
        )

    comp_data: Dict[str, np.ndarray] = {}
    for name, da in comparison_maps.items():
        _validate_2d_da(f"comparison_maps['{name}']", da)
        if tuple(da.shape) != tuple(ref.shape):
            raise ValueError(f"Map '{name}' has shape {da.shape}, expected {ref.shape}.")
        comp_data[name] = da.values

    valid = np.isfinite(ref)
    for arr in comp_data.values():
        valid &= np.isfinite(arr)

    ref_sign = np.sign(ref)
    comp_sign = {name: np.sign(arr) for name, arr in comp_data.items()}

    metrics: List[dict] = []

    for level in hierarchy.get("levels", []):
        threshold = float(level["threshold"])
        labels = level["labels"]
        comp_ids = np.unique(labels)
        comp_ids = comp_ids[comp_ids > 0]

        for comp_id in comp_ids:
            comp_mask = (labels == comp_id) & valid
            n = int(comp_mask.sum())
            if n == 0:
                continue

            row = {
                "threshold": threshold,
                "component_id": int(comp_id),
                "node_id": f"T{threshold:.6f}_C{int(comp_id)}",
                "n_valid_pixels": n,
            }

            ref_vals = ref[comp_mask]

            pca_cols = [ref_vals]
            for name, arr in comp_data.items():
                vals = arr[comp_mask]
                diff = ref_vals - vals
                row[f"rmse_vs_{name}"] = float(np.sqrt(np.mean(diff**2)))
                row[f"bias_vs_{name}"] = float(np.mean(diff))
                row[f"same_sign_vs_{name}"] = float(
                    np.mean(ref_sign[comp_mask] == comp_sign[name][comp_mask])
                )
                pca_cols.append(vals)

            # PCA-like summary using SVD (no external dependency).
            X = np.column_stack(pca_cols)
            Xc = X - np.mean(X, axis=0, keepdims=True)
            if Xc.shape[0] > 1 and Xc.shape[1] > 1:
                _, svals, _ = np.linalg.svd(Xc, full_matrices=False)
                var = svals**2
                row["pc1_var_explained"] = float(var[0] / np.sum(var)) if np.sum(var) > 0 else np.nan
            else:
                row["pc1_var_explained"] = np.nan

            metrics.append(row)

    return metrics
