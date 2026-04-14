"""Preprocessing helpers for Python API workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.transform import Affine


def coarsen_rasters(
    arrs: dict[str, np.ndarray | None],
    factor: int,
    profile: dict[str, Any],
) -> tuple[dict[str, np.ndarray | None], dict[str, Any]]:
    """Block-mean coarsen raster arrays by an integer factor.

    Parameters
    ----------
    arrs : dict[str, np.ndarray | None]
        Mapping of raster names to 2D arrays. All non-None arrays should be
        aligned on the same grid.
    factor : int
        Integer coarsening factor. ``1`` returns input unchanged.
    profile : dict
        Raster profile with at least ``transform``, ``width``, and ``height``.

    Returns
    -------
    (coarsened, new_profile) : tuple
        Coarsened arrays and updated profile.
    """
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
