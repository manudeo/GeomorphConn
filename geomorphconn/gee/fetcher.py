# -*- coding: utf-8 -*-
"""
fetcher.py
==========
Google Earth Engine data fetcher for GeomorphConn.

Uses the ``xee`` backend (Earth Engine + xarray) to stream rasters
directly from GEE without local download, then reprojects / resamples
everything to a common grid matching the chosen DEM.

Supported sources
-----------------
DEM
    ``'SRTM'``     — USGS SRTM 1-arc-second (~30 m), global coverage
    ``'COPDEM30'``  — Copernicus DEM GLO-30 (~30 m), global coverage
    ``'MERIT'``     — MERIT DEM (~90 m), global, hydrologically conditioned

Rainfall
    ``'CHIRPS'``    — UCSB CHIRPS v2.0 daily (~5.5 km), 1981–present
    ``'ERA5'``      — ECMWF ERA5 monthly (~27 km), 1979–present
    ``'PERSIANN'``  — PERSIANN-CDR daily (~27 km), 1983–present

NDVI
    ``'SENTINEL2'`` — Sentinel-2 Level-2A SR, 10–20 m, 2017–present
    ``'LANDSAT8'``  — Landsat 8 Collection 2 SR, 30 m, 2013–present
    ``'LANDSAT9'``  — Landsat 9 Collection 2 SR, 30 m, 2021–present

Dependencies
------------
    pip install "GeomorphConn[gee]"
    # i.e.: earthengine-api xee xarray geopandas pyproj rasterio
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


# ── Lazy-import GEE / xee so the core component stays importable without them
def _require_gee():
    try:
        import ee
        return ee
    except ImportError:
        raise ImportError(
            "earthengine-api is required for GEEFetcher. "
            "Install it with: pip install 'GeomorphConn[gee]'"
        )


def _require_xee():
    try:
        import xarray as xr
        import xee  # noqa: F401
        return xr
    except ImportError:
        raise ImportError(
            "xee and xarray are required for GEEFetcher. "
            "Install them with: pip install 'GeomorphConn[gee]'"
        )


# ── Catalogue of GEE collection/image IDs and band names ─────────────────────
_DEM_CATALOGUE = {
    "SRTM": {
        "asset"  : "USGS/SRTMGL1_003",
        "band"   : "elevation",
        "type"   : "image",
        "scale"  : 30,
        "desc"   : "SRTM 1 arc-second (~30 m)",
    },
    "COPDEM30": {
        "asset"  : "COPERNICUS/DEM/GLO30",
        "band"   : "DEM",
        "type"   : "image_collection",  # mosaic needed
        "scale"  : 30,
        "desc"   : "Copernicus DEM GLO-30 (~30 m)",
    },
    "MERIT": {
        "asset"  : "MERIT/DEM/v1_0_3",
        "band"   : "dem",
        "type"   : "image",
        "scale"  : 90,
        "desc"   : "MERIT Hydrologically Conditioned DEM (~90 m)",
    },
}

_RAINFALL_CATALOGUE = {
    "CHIRPS": {
        "asset"  : "UCSB-CHG/CHIRPS/DAILY",
        "band"   : "precipitation",
        "type"   : "image_collection",
        "agg"    : "sum",     # aggregate daily → total over period
        "scale"  : 5566,
        "desc"   : "CHIRPS daily precipitation (~5.5 km)",
    },
    "ERA5": {
        "asset"  : "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "band"   : "total_precipitation_sum",
        "type"   : "image_collection",
        "agg"    : "mean",    # mean monthly over period
        "scale"  : 11132,
        "desc"   : "ERA5-Land monthly total precipitation (~11 km)",
    },
    "PERSIANN": {
        "asset"  : "NOAA/PERSIANN-CDR",
        "band"   : "precipitation",
        "type"   : "image_collection",
        "agg"    : "sum",
        "scale"  : 27830,
        "desc"   : "PERSIANN-CDR daily precipitation (~27 km)",
    },
}

_NDVI_CATALOGUE = {
    "SENTINEL2": {
        "asset"        : "COPERNICUS/S2_SR_HARMONIZED",
        "nir_band"     : "B8",
        "red_band"     : "B4",
        "cloud_band"   : "MSK_CLDPRB",
        "cloud_thresh" : 20,
        "scale"        : 10,
        "desc"         : "Sentinel-2 SR Harmonized (~10 m)",
    },
    "LANDSAT8": {
        "asset"        : "LANDSAT/LC08/C02/T1_L2",
        "nir_band"     : "SR_B5",
        "red_band"     : "SR_B4",
        "cloud_band"   : "QA_PIXEL",
        "cloud_thresh" : None,          # uses bitwise QA
        "scale"        : 30,
        "desc"         : "Landsat 8 Collection 2 SR (~30 m)",
    },
    "LANDSAT9": {
        "asset"        : "LANDSAT/LC09/C02/T1_L2",
        "nir_band"     : "SR_B5",
        "red_band"     : "SR_B4",
        "cloud_band"   : "QA_PIXEL",
        "cloud_thresh" : None,
        "scale"        : 30,
        "desc"         : "Landsat 9 Collection 2 SR (~30 m)",
    },
}


# ── Land-cover datasets ───────────────────────────────────────────────────
_LANDCOVER_CATALOGUE = {
    "WORLDCOVER": {
        "asset"  : "ESA/WorldCover/v200",
        "band"   : "Map",
        "type"   : "image_collection",
        "scale"  : 10,
        "desc"   : "ESA WorldCover 10 m v200 (2021)",
        "c_table": "WORLDCOVER_C_FACTOR",
    },
    "WORLDCOVER_V100": {
        "asset"  : "ESA/WorldCover/v100",
        "band"   : "Map",
        "type"   : "image_collection",
        "scale"  : 10,
        "desc"   : "ESA WorldCover 10 m v100 (2020)",
        "c_table": "WORLDCOVER_C_FACTOR",
    },
    "MODIS_LC": {
        "asset"  : "MODIS/061/MCD12Q1",
        "band"   : "LC_Type1",
        "type"   : "image_collection",
        "scale"  : 500,
        "desc"   : "MODIS MCD12Q1 Land Cover Type 1 IGBP (500 m)",
        "c_table": "MODIS_IGBP_C_FACTOR",
    },
    "CORINE": {
        "asset"  : "COPERNICUS/CORINE/V20/100m",
        "band"   : "landcover",
        "type"   : "image",
        "scale"  : 100,
        "desc"   : "CORINE Land Cover 2018 (100 m, Europe only)",
        "c_table": "CORINE_C_FACTOR",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
class _FetchResult(dict):
    """
    Dict subclass returned by GEEFetcher.fetch().

    Supports both attribute access (``result.dem``) and legacy 4-tuple
    unpacking (``dem, ndvi, rainfall, profile = fetcher.fetch()``).
    """
    def __init__(self, dem, ndvi, rainfall, landcover, profile, **extras):
        super().__init__(dem=dem, ndvi=ndvi, rainfall=rainfall,
                         landcover=landcover, profile=profile, **extras)
        self.dem       = dem
        self.ndvi      = ndvi
        self.rainfall  = rainfall
        self.landcover = landcover
        self.profile   = profile
        for key, value in extras.items():
            setattr(self, key, value)

    def __iter__(self):
        # Support: dem, ndvi, rainfall, profile = fetcher.fetch()
        yield self.dem
        yield self.ndvi
        yield self.rainfall
        yield self.profile


class GEEFetcher:
    """
    Fetch DEM, NDVI, and Rainfall from Google Earth Engine for a given area.

    Parameters
    ----------
    bounds : tuple of float or str or Path
        Either ``(lon_min, lat_min, lon_max, lat_max)`` in WGS-84 decimal
        degrees, or a path to any geopandas-readable file (Shapefile,
        GeoJSON, GeoPackage, …) whose envelope is used as the bounding box.
    dem_source : str, optional
        DEM dataset. One of ``'SRTM'``, ``'COPDEM30'``, ``'MERIT'``.
        Default: ``'COPDEM30'``.
    rainfall_source : str, optional
        Rainfall dataset. One of ``'CHIRPS'``, ``'ERA5'``, ``'PERSIANN'``.
        Default: ``'CHIRPS'``.
    ndvi_source : str, optional
        NDVI dataset. One of ``'SENTINEL2'``, ``'LANDSAT8'``, ``'LANDSAT9'``.
        Default: ``'SENTINEL2'``.
    start_date : str, optional
        ISO date string ``'YYYY-MM-DD'`` for the start of the NDVI/rainfall
        compositing window. Default: ``'2020-01-01'``.
    end_date : str, optional
        ISO date string ``'YYYY-MM-DD'`` for the end of the window.
        Default: ``'2020-12-31'``.
    scale : int or None, optional
        Output raster resolution in metres. If *None*, uses the native scale
        of the chosen DEM source. Default: *None*.
    crs : str, optional
        Output CRS as an EPSG string (e.g. ``'EPSG:32644'``) or
        ``'EPSG:4326'`` for geographic coordinates. Default: ``'EPSG:4326'``.
    gee_project : str, optional
        GEE cloud project ID, required when using ``ee.Initialize(project=...)``.
        If *None*, uses previously authenticated credentials.
    buffer_m : float, optional
        Additional buffer (metres) added around the bounding box before
        fetching. Useful to avoid edge effects. Default: 0.

    Examples
    --------
    >>> fetcher = GEEFetcher(
    ...     bounds=(72.5, 28.0, 80.5, 32.0),
    ...     dem_source="COPDEM30",
    ...     rainfall_source="CHIRPS",
    ...     ndvi_source="SENTINEL2",
    ...     start_date="2020-06-01",
    ...     end_date="2020-08-31",
    ...     scale=30,
    ...     gee_project="my-gee-project",
    ... )
    >>> dem, ndvi, rainfall, profile = fetcher.fetch()
    """

    def __init__(
        self,
        bounds: Union[Tuple[float, float, float, float], str, Path],
        dem_source: str = "COPDEM30",
        rainfall_source: str = "CHIRPS",
        ndvi_source: str = "SENTINEL2",
        landcover_source: Optional[str] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2020-12-31",
        scale: Optional[int] = None,
        crs: str = "EPSG:4326",
        gee_project: Optional[str] = None,
        buffer_m: float = 0.0,
    ):
        # ── Parse bounds ───────────────────────────────────────────────
        self._bbox    = self._parse_bounds(bounds, buffer_m)
        self._start   = start_date
        self._end     = end_date
        self._crs     = crs
        self._project = gee_project

        # Slope calculations assume horizontal spacing in linear units (metres).
        # Geographic CRS (degrees) can distort dy/dx and IC routing metrics.
        try:
            from pyproj import CRS as _CRS

            if not _CRS.from_user_input(self._crs).is_projected:
                warnings.warn(
                    "GEEFetcher is configured with a geographic CRS. "
                    "Connectivity slope/routing calculations are most reliable in a projected CRS "
                    "with metric x/y units (for example a local UTM EPSG).",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception:
            pass

        # ── Validate source keys ───────────────────────────────────────
        dem_key = dem_source.upper()
        rf_key  = rainfall_source.upper()
        nv_key  = ndvi_source.upper()
        if dem_key not in _DEM_CATALOGUE:
            raise ValueError(f"dem_source must be one of {list(_DEM_CATALOGUE)}")
        if rf_key not in _RAINFALL_CATALOGUE:
            raise ValueError(
                f"rainfall_source must be one of {list(_RAINFALL_CATALOGUE)}"
            )
        if nv_key not in _NDVI_CATALOGUE:
            raise ValueError(f"ndvi_source must be one of {list(_NDVI_CATALOGUE)}")

        self._dem_cfg = _DEM_CATALOGUE[dem_key]
        self._rf_cfg  = _RAINFALL_CATALOGUE[rf_key]
        self._nv_cfg  = _NDVI_CATALOGUE[nv_key]
        self._scale   = scale if scale is not None else self._dem_cfg["scale"]

        # ── Optional land-cover source ─────────────────────────────────
        if landcover_source is not None:
            lc_key = landcover_source.upper()
            if lc_key not in _LANDCOVER_CATALOGUE:
                raise ValueError(
                    f"landcover_source must be one of "
                    f"{list(_LANDCOVER_CATALOGUE)} or None"
                )
            self._lc_cfg = _LANDCOVER_CATALOGUE[lc_key]
        else:
            self._lc_cfg = None

        lc_desc = self._lc_cfg["desc"] if self._lc_cfg else "None (not fetched)"
        print(
            f"GEEFetcher configured:\n"
            f"  DEM        : {self._dem_cfg['desc']}\n"
            f"  Rainfall   : {self._rf_cfg['desc']}\n"
            f"  NDVI       : {self._nv_cfg['desc']}\n"
            f"  Land cover : {lc_desc}\n"
            f"  Period     : {start_date} → {end_date}\n"
            f"  Scale      : {self._scale} m  |  CRS: {crs}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self, return_xarray: bool = False) -> dict:
        """
        Fetch all configured datasets and return a results dictionary.

        Returns
        -------
        dict with keys:
            ``dem``       ndarray (nrows, ncols) — elevation in metres
            ``ndvi``      ndarray (nrows, ncols) — median NDVI in [−1, 1]
            ``rainfall``  ndarray (nrows, ncols) — aggregated rainfall
            ``landcover`` ndarray or None — integer LC codes (if landcover_source set)
            ``profile``   dict — rasterio profile for all arrays

            If ``return_xarray=True``, also includes aligned xarray DataArrays:
            ``dem_xr``, ``ndvi_xr``, ``rainfall_xr``, ``landcover_xr``.

        For backward compatibility, the dict also unpacks as a 4-tuple
        ``(dem, ndvi, rainfall, profile)`` via ``__iter__``.

        Examples
        --------
        >>> result = fetcher.fetch()
        >>> dem, ndvi, rainfall, profile = fetcher.fetch()   # legacy unpack
        >>> result["landcover"]  # None if landcover_source not set
        """
        ee  = self._init_gee()
        xr  = _require_xee()

        geometry = ee.Geometry.BBox(*self._bbox)

        print("Fetching DEM …")
        if return_xarray:
            dem_out = self._fetch_dem(ee, xr, geometry, return_xarray=True)
            dem = dem_out[0]
            profile = dem_out[1]
            dem_xr = dem_out[2] if len(dem_out) > 2 else None
        else:
            dem_out = self._fetch_dem(ee, xr, geometry, return_xarray=False)
            dem = dem_out[0]
            profile = dem_out[1]
            dem_xr = None

        print("Fetching NDVI …")
        if return_xarray:
            ndvi, ndvi_xr = self._fetch_ndvi(
                ee,
                xr,
                geometry,
                profile,
                return_xarray=True,
            )
        else:
            ndvi = self._fetch_ndvi(
                ee,
                xr,
                geometry,
                profile,
                return_xarray=False,
            )
            ndvi_xr = None

        print("Fetching Rainfall …")
        if return_xarray:
            rainfall, rainfall_xr = self._fetch_rainfall(
                ee,
                xr,
                geometry,
                profile,
                return_xarray=True,
            )
        else:
            rainfall = self._fetch_rainfall(
                ee,
                xr,
                geometry,
                profile,
                return_xarray=False,
            )
            rainfall_xr = None

        landcover = None
        landcover_xr = None
        if self._lc_cfg is not None:
            print("Fetching Land Cover …")
            if return_xarray:
                landcover, landcover_xr = self._fetch_landcover(
                    ee,
                    xr,
                    geometry,
                    profile,
                    return_xarray=True,
                )
            else:
                landcover = self._fetch_landcover(
                    ee,
                    xr,
                    geometry,
                    profile,
                    return_xarray=False,
                )

        extras = {}
        if return_xarray:
            extras = {
                "dem_xr": dem_xr,
                "ndvi_xr": ndvi_xr,
                "rainfall_xr": rainfall_xr,
                "landcover_xr": landcover_xr,
            }

        return _FetchResult(
            dem=dem,
            ndvi=ndvi,
            rainfall=rainfall,
            landcover=landcover,
            profile=profile,
            **extras,
        )

    def fetch_timeseries(
        self,
        resampling: str = "monthly",
        include_landcover: bool = False,
        max_periods: Optional[int] = None,
        return_xarray: bool = False,
    ) -> dict:
        """
        Fetch DEM once and NDVI/rainfall as a temporal sequence.

        Parameters
        ----------
        resampling : str, optional
            Temporal binning mode for the [start_date, end_date] range.
            Supported: ``'monthly'``, ``'seasonal'``, ``'annual'``.
            Default: ``'monthly'``.
        include_landcover : bool, optional
            If True and ``landcover_source`` is configured, fetch land-cover once
            and include it in the returned dictionary. Default: False.
        max_periods : int or None, optional
            Optional hard limit on number of periods to fetch (useful for quick
            tests). Periods are kept in chronological order. Default: None.

        Returns
        -------
        dict
            ``{\"dem\": ndarray, \"profile\": dict, \"periods\": list, \"landcover\": ...}``

            where ``periods`` is a list of dictionaries with keys:
            ``label``, ``start_date``, ``end_date``, ``ndvi``, ``rainfall``.
        """
        ee, xr, geometry = self._setup()

        print("Fetching DEM (shared across all periods) …")
        if return_xarray:
            dem_out = self._fetch_dem(ee, xr, geometry, return_xarray=True)
            dem = dem_out[0]
            profile = dem_out[1]
            dem_xr = dem_out[2] if len(dem_out) > 2 else None
        else:
            dem_out = self._fetch_dem(ee, xr, geometry, return_xarray=False)
            dem = dem_out[0]
            profile = dem_out[1]
            dem_xr = None

        windows = self._build_time_windows(
            self._start,
            self._end,
            resampling=resampling,
        )
        if max_periods is not None:
            windows = windows[:max_periods]

        periods = []
        for idx, (label, p_start, p_end) in enumerate(windows, start=1):
            print(f"[{idx}/{len(windows)}] Fetching window {label}: {p_start} → {p_end}")
            if return_xarray:
                ndvi, ndvi_xr = self._fetch_ndvi(
                    ee,
                    xr,
                    geometry,
                    profile,
                    start_date=p_start,
                    end_date=p_end,
                    return_xarray=True,
                )
            else:
                ndvi = self._fetch_ndvi(
                    ee,
                    xr,
                    geometry,
                    profile,
                    start_date=p_start,
                    end_date=p_end,
                    return_xarray=False,
                )
                ndvi_xr = None

            if return_xarray:
                rainfall, rainfall_xr = self._fetch_rainfall(
                    ee,
                    xr,
                    geometry,
                    profile,
                    start_date=p_start,
                    end_date=p_end,
                    return_xarray=True,
                )
            else:
                rainfall = self._fetch_rainfall(
                    ee,
                    xr,
                    geometry,
                    profile,
                    start_date=p_start,
                    end_date=p_end,
                    return_xarray=False,
                )
                rainfall_xr = None

            period_item = {
                "label": label,
                "start_date": p_start,
                "end_date": p_end,
                "ndvi": ndvi,
                "rainfall": rainfall,
            }
            if return_xarray:
                period_item["ndvi_xr"] = ndvi_xr
                period_item["rainfall_xr"] = rainfall_xr
            periods.append(period_item)

        out = {
            "dem": dem,
            "profile": profile,
            "periods": periods,
            "resampling": resampling.lower(),
        }
        if return_xarray:
            out["dem_xr"] = dem_xr

        if include_landcover and self._lc_cfg is not None:
            print("Fetching Land Cover (shared across all periods) …")
            if return_xarray:
                out["landcover"], out["landcover_xr"] = self._fetch_landcover(
                    ee,
                    xr,
                    geometry,
                    profile,
                    return_xarray=True,
                )
            else:
                out["landcover"] = self._fetch_landcover(
                    ee,
                    xr,
                    geometry,
                    profile,
                    return_xarray=False,
                )
                out["landcover_xr"] = None
        else:
            out["landcover"] = None
            out["landcover_xr"] = None

        return out

    def fetch_dem(self, return_xarray: bool = False):
        """Fetch DEM only."""
        ee, xr, geometry = self._setup()
        return self._fetch_dem(ee, xr, geometry, return_xarray=return_xarray)

    def fetch_ndvi(self, profile=None, return_xarray: bool = False):
        """Fetch NDVI only."""
        ee, xr, geometry = self._setup()
        if profile is None:
            dem_out = self._fetch_dem(ee, xr, geometry, return_xarray=False)
            profile = dem_out[1]
        return self._fetch_ndvi(
            ee,
            xr,
            geometry,
            profile,
            return_xarray=return_xarray,
        )

    def fetch_rainfall(self, profile=None, return_xarray: bool = False):
        """Fetch Rainfall only."""
        ee, xr, geometry = self._setup()
        if profile is None:
            dem_out = self._fetch_dem(ee, xr, geometry, return_xarray=False)
            profile = dem_out[1]
        return self._fetch_rainfall(
            ee,
            xr,
            geometry,
            profile,
            return_xarray=return_xarray,
        )

    @staticmethod
    def list_sources():
        """Print all available dataset sources."""
        print("DEM sources:        ", list(_DEM_CATALOGUE.keys()))
        print("Rainfall sources:   ", list(_RAINFALL_CATALOGUE.keys()))
        print("NDVI sources:       ", list(_NDVI_CATALOGUE.keys()))
        print("Land-cover sources: ", list(_LANDCOVER_CATALOGUE.keys()))

    def fetch_landcover(self, profile=None, return_xarray: bool = False):
        """Fetch land-cover classification only."""
        ee, xr, geometry = self._setup()
        if self._lc_cfg is None:
            raise ValueError(
                "No landcover_source configured. "
                "Pass landcover_source= to GEEFetcher()."
            )
        if profile is None:
            dem_out = self._fetch_dem(ee, xr, geometry, return_xarray=False)
            profile = dem_out[1]
        return self._fetch_landcover(
            ee,
            xr,
            geometry,
            profile,
            return_xarray=return_xarray,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _setup(self):
        ee = self._init_gee()
        xr = _require_xee()
        geometry = ee.Geometry.BBox(*self._bbox)
        return ee, xr, geometry

    def _init_gee(self):
        """Initialise the GEE API (authenticate if needed)."""
        ee = _require_gee()
        try:
            if self._project:
                ee.Initialize(project=self._project)
            else:
                ee.Initialize()
        except Exception:
            print("GEE not authenticated. Attempting ee.Authenticate() …")
            ee.Authenticate()
            if self._project:
                ee.Initialize(project=self._project)
            else:
                ee.Initialize()
        return ee

    @staticmethod
    def _build_time_windows(start_date: str, end_date: str, resampling: str = "monthly"):
        """Build inclusive temporal windows from ISO date strings."""
        mode = str(resampling).lower()
        if mode not in {"monthly", "seasonal", "annual"}:
            raise ValueError("resampling must be one of: 'monthly', 'seasonal', 'annual'")

        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        if end < start:
            raise ValueError("end_date must be >= start_date")

        windows = []

        if mode == "monthly":
            y, m = start.year, start.month
            while (y < end.year) or (y == end.year and m <= end.month):
                w_start = date(y, m, 1)
                if m == 12:
                    n_y, n_m = y + 1, 1
                else:
                    n_y, n_m = y, m + 1
                w_end = date(n_y, n_m, 1).fromordinal(date(n_y, n_m, 1).toordinal() - 1)
                clip_start = max(w_start, start)
                clip_end = min(w_end, end)
                if clip_start <= clip_end:
                    windows.append((
                        f"{y:04d}-{m:02d}",
                        clip_start.isoformat(),
                        clip_end.isoformat(),
                    ))
                y, m = n_y, n_m

        elif mode == "annual":
            for y in range(start.year, end.year + 1):
                w_start = date(y, 1, 1)
                w_end = date(y, 12, 31)
                clip_start = max(w_start, start)
                clip_end = min(w_end, end)
                if clip_start <= clip_end:
                    windows.append((
                        f"{y:04d}",
                        clip_start.isoformat(),
                        clip_end.isoformat(),
                    ))

        else:  # seasonal
            season_defs = [
                ("DJF", (12, 1, 2)),
                ("MAM", (3, 4, 5)),
                ("JJA", (6, 7, 8)),
                ("SON", (9, 10, 11)),
            ]
            for y in range(start.year - 1, end.year + 1):
                for name, months in season_defs:
                    if name == "DJF":
                        w_start = date(y, 12, 1)
                        w_end = date(y + 1, 2, 28)
                        # handle leap year
                        if (y + 1) % 4 == 0 and ((y + 1) % 100 != 0 or (y + 1) % 400 == 0):
                            w_end = date(y + 1, 2, 29)
                        label_year = y + 1
                    else:
                        w_start = date(y, months[0], 1)
                        last_month = months[-1]
                        if last_month == 12:
                            w_end = date(y, 12, 31)
                        else:
                            next_month = date(y, last_month + 1, 1)
                            w_end = next_month.fromordinal(next_month.toordinal() - 1)
                        label_year = y

                    clip_start = max(w_start, start)
                    clip_end = min(w_end, end)
                    if clip_start <= clip_end:
                        windows.append((
                            f"{label_year:04d}-{name}",
                            clip_start.isoformat(),
                            clip_end.isoformat(),
                        ))

            # keep chronological order and drop duplicates that may arise at range edges
            seen = set()
            uniq = []
            for row in sorted(windows, key=lambda r: (r[1], r[2], r[0])):
                key = (row[0], row[1], row[2])
                if key not in seen:
                    seen.add(key)
                    uniq.append(row)
            windows = uniq

        return windows

    @staticmethod
    def _parse_bounds(bounds, buffer_m):
        """Return (lon_min, lat_min, lon_max, lat_max) in WGS-84."""
        if isinstance(bounds, (str, Path)):
            try:
                import geopandas as gpd
            except ImportError:
                raise ImportError("geopandas is required to read shapefile bounds.")
            gdf    = gpd.read_file(bounds).to_crs("EPSG:4326")
            b      = gdf.total_bounds    # (minx, miny, maxx, maxy)
            bounds = (b[0], b[1], b[2], b[3])

        lon_min, lat_min, lon_max, lat_max = bounds
        if buffer_m > 0:
            # Very rough degree-equivalent buffer (fine for small buffers)
            deg = buffer_m / 111_000
            lon_min -= deg
            lat_min -= deg
            lon_max += deg
            lat_max += deg

        return (lon_min, lat_min, lon_max, lat_max)

    def _xee_open(self, xr, ee, asset_id, band, geometry, scale, is_collection=True):
        """
        Open a GEE asset via xee and return the requested band as a
        numpy array clipped to *geometry*.
        """
        import xee  # noqa

        if is_collection:
            url = f"ee://{asset_id}"
        else:
            # Single image — wrap in a one-image collection for xee
            img = ee.Image(asset_id).select(band)
            url = img

        try:
            ds = xr.open_dataset(
                url if isinstance(url, str) else ee.ImageCollection([url]),
                engine    = "ee",
                geometry  = geometry,
                scale     = scale,
                crs       = self._crs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open GEE asset '{asset_id}' via xee. "
                f"Check your GEE credentials and asset availability.\n"
                f"Original error: {exc}"
            )

        # Select the requested band
        if band in ds:
            arr = ds[band].values
        else:
            available = list(ds.data_vars)
            raise KeyError(
                f"Band '{band}' not found in dataset. Available: {available}"
            )
        return arr

    @staticmethod
    def _xy_dim_names(da):
        """Return best-guess (y_dim, x_dim) names for an xarray.DataArray."""
        dims = list(da.dims)
        dim_map = {d.lower(): d for d in dims}

        y_dim = None
        x_dim = None
        for cand in ("y", "lat", "latitude"):
            if cand in dim_map:
                y_dim = dim_map[cand]
                break
        for cand in ("x", "lon", "longitude"):
            if cand in dim_map:
                x_dim = dim_map[cand]
                break

        if y_dim is None or x_dim is None:
            if len(dims) >= 2:
                y_dim, x_dim = dims[-2], dims[-1]
            else:
                raise ValueError(
                    f"Could not infer X/Y dimensions from DataArray dims={dims}."
                )
        return y_dim, x_dim

    @staticmethod
    def _to_2d_yx(da):
        """Coerce DataArray to 2D (y, x) and normalize to north-up orientation."""
        d2 = da

        # Prefer explicit spatial dims if present, and only drop non-spatial axes.
        try:
            y_dim, x_dim = GEEFetcher._xy_dim_names(d2)
            for dim in list(d2.dims):
                if dim not in (y_dim, x_dim):
                    d2 = d2.isel({dim: 0}, drop=True)

            if y_dim in d2.dims and x_dim in d2.dims:
                if y_dim != "y" or x_dim != "x":
                    d2 = d2.rename({y_dim: "y", x_dim: "x"})
                    y_dim, x_dim = "y", "x"
                if tuple(d2.dims) != (y_dim, x_dim):
                    d2 = d2.transpose(y_dim, x_dim)

                # Normalize orientation so arrays are north-up / west-east:
                # - y should decrease from top row to bottom row
                # - x should increase from left to right
                if y_dim in d2.coords:
                    yv = np.asarray(d2.coords[y_dim].values, dtype=np.float64)
                    if yv.size >= 2 and np.isfinite(yv[0]) and np.isfinite(yv[-1]):
                        if yv[0] < yv[-1]:
                            d2 = d2.isel({y_dim: slice(None, None, -1)})

                if x_dim in d2.coords:
                    xv = np.asarray(d2.coords[x_dim].values, dtype=np.float64)
                    if xv.size >= 2 and np.isfinite(xv[0]) and np.isfinite(xv[-1]):
                        if xv[0] > xv[-1]:
                            d2 = d2.isel({x_dim: slice(None, None, -1)})

                arr = np.asarray(d2.values)
                if arr.ndim == 2:
                    return arr, d2, y_dim, x_dim
        except Exception:
            pass

        # Fallback path for edge cases where xee returns scalar or odd dim layouts.
        arr = np.asarray(d2.squeeze(drop=True).values)
        if arr.ndim == 0:
            warnings.warn(
                "xee returned a scalar raster value; coercing to shape (1, 1).",
                UserWarning,
                stacklevel=2,
            )
            arr = arr.reshape((1, 1))
            return arr, None, None, None

        if arr.ndim == 1:
            warnings.warn(
                "xee returned a 1D raster slice; coercing to shape (1, n).",
                UserWarning,
                stacklevel=2,
            )
            arr = arr.reshape((1, arr.shape[0]))
            return arr, None, None, None

        while arr.ndim > 2:
            arr = arr[0]

        warnings.warn(
            "xee returned non-standard raster dims; using best-effort 2D coercion.",
            UserWarning,
            stacklevel=2,
        )
        return arr, None, None, None

    def _fallback_bounds_in_output_crs(self):
        """Transform WGS84 bbox to output CRS bounds for robust transform fallback."""
        lon_min, lat_min, lon_max, lat_max = self._bbox
        if str(self._crs).upper() == "EPSG:4326":
            return lon_min, lat_min, lon_max, lat_max

        try:
            from pyproj import Transformer

            tr = Transformer.from_crs("EPSG:4326", self._crs, always_xy=True)
            x0, y0 = tr.transform(lon_min, lat_min)
            x1, y1 = tr.transform(lon_max, lat_max)
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
        except Exception:
            warnings.warn(
                "Could not transform bbox to output CRS. Falling back to WGS84 bounds; "
                "this may reduce georeferencing accuracy.",
                UserWarning,
                stacklevel=2,
            )
            return lon_min, lat_min, lon_max, lat_max

    def _transform_from_dataarray(self, da_2d, y_dim, x_dim):
        """Build rasterio transform from xarray coordinates when available."""
        from rasterio.transform import from_bounds

        try:
            x = np.asarray(da_2d.coords[x_dim].values, dtype=np.float64)
            y = np.asarray(da_2d.coords[y_dim].values, dtype=np.float64)
            if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
                raise ValueError("X/Y coordinates are not 1D with at least 2 values")

            dx = float(np.nanmedian(np.abs(np.diff(x))))
            dy = float(np.nanmedian(np.abs(np.diff(y))))
            if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dy) or dy <= 0:
                raise ValueError("Invalid coordinate spacing inferred from xarray coords")

            left = float(np.nanmin(x) - dx / 2.0)
            right = float(np.nanmax(x) + dx / 2.0)
            bottom = float(np.nanmin(y) - dy / 2.0)
            top = float(np.nanmax(y) + dy / 2.0)
            return from_bounds(left, bottom, right, top, len(x), len(y))
        except Exception:
            warnings.warn(
                "Could not infer transform from xee coordinates; using transformed bbox fallback.",
                UserWarning,
                stacklevel=2,
            )
            lon_min, lat_min, lon_max, lat_max = self._fallback_bounds_in_output_crs()
            return from_bounds(lon_min, lat_min, lon_max, lat_max, da_2d.shape[1], da_2d.shape[0])

    @staticmethod
    def _array_to_dataarray(arr, profile, name):
        """Convert a 2D ndarray into an xarray.DataArray on the profile grid."""
        xr = _require_xee()

        tr = profile["transform"]
        width = int(profile["width"])
        height = int(profile["height"])

        # Pixel-center coordinates derived from affine transform.
        x = tr.c + tr.a * (np.arange(width, dtype=np.float64) + 0.5)
        y = tr.f + tr.e * (np.arange(height, dtype=np.float64) + 0.5)

        da = xr.DataArray(
            np.asarray(arr),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name=name,
        )
        da.attrs["crs"] = str(profile.get("crs"))
        da.attrs["transform"] = tuple(tr)
        return da

    def _fetch_dem(self, ee, xr, geometry, return_xarray: bool = False):
        """Fetch DEM and build rasterio profile."""
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        cfg  = self._dem_cfg
        band = cfg["band"]

        if cfg["type"] == "image":
            img  = ee.Image(cfg["asset"]).select(band)
            coll = ee.ImageCollection([img])
        else:
            coll = ee.ImageCollection(cfg["asset"]).select(band).mosaic()
            coll = ee.ImageCollection([coll])

        import xee  # noqa
        ds = xr.open_dataset(
            coll,
            engine   = "ee",
            geometry = geometry,
            scale    = self._scale,
            crs      = self._crs,
        )

        if band not in ds:
            band = list(ds.data_vars)[0]
            warnings.warn(f"DEM band not found, using first available: '{band}'")

        # xee may return dims in either (Y, X) or (X, Y) order; normalize to (Y, X)
        arr, da2d, y_dim, x_dim = self._to_2d_yx(ds[band])
        arr = arr.astype(np.float64)
        arr[arr == ds[band].attrs.get("_FillValue", -9999.0)] = np.nan

        nrows, ncols = arr.shape
        if da2d is not None and y_dim is not None and x_dim is not None:
            transform = self._transform_from_dataarray(da2d, y_dim, x_dim)
        else:
            warnings.warn(
                "DEM transform built from fallback bounds because xee coordinates were not available.",
                UserWarning,
                stacklevel=2,
            )
            left, bottom, right, top = self._fallback_bounds_in_output_crs()
            transform = from_bounds(left, bottom, right, top, ncols, nrows)
        profile   = {
            "driver"   : "GTiff",
            "dtype"    : "float32",
            "width"    : ncols,
            "height"   : nrows,
            "count"    : 1,
            "crs"      : CRS.from_string(self._crs),
            "transform": transform,
            "nodata"   : -9999.0,
        }
        print(f"  DEM shape: {nrows}×{ncols}, range: "
              f"{np.nanmin(arr):.1f}–{np.nanmax(arr):.1f} m")
        if return_xarray:
            return arr, profile, self._array_to_dataarray(arr, profile, "dem")
        return arr, profile

    def _fetch_ndvi(
        self,
        ee,
        xr,
        geometry,
        ref_profile,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        return_xarray: bool = False,
    ):
        """Compute median NDVI from the chosen sensor over the date range."""
        cfg  = self._nv_cfg
        s_date = start_date if start_date is not None else self._start
        e_date = end_date if end_date is not None else self._end
        coll = (
            ee.ImageCollection(cfg["asset"])
            .filterDate(s_date, e_date)
            .filterBounds(geometry)
        )

        if cfg["cloud_thresh"] is None:
            # Landsat: bitwise QA masking
            def _mask_landsat(img):
                qa   = img.select(cfg["cloud_band"])
                mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
                return img.updateMask(mask)
            coll = coll.map(_mask_landsat)
        else:
            # Sentinel-2: cloud probability band
            thresh = cfg["cloud_thresh"]
            def _mask_s2(img):
                return img.updateMask(
                    img.select(cfg["cloud_band"]).lt(thresh)
                )
            coll = coll.map(_mask_s2)

        # Compute NDVI per image then take median
        def _add_ndvi(img):
            nir  = img.select(cfg["nir_band"]).toFloat()
            red  = img.select(cfg["red_band"]).toFloat()
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
            return ndvi

        ndvi_coll = coll.map(_add_ndvi)
        ndvi_img  = ndvi_coll.median().clip(geometry)

        import xee  # noqa
        ds = xr.open_dataset(
            ee.ImageCollection([ndvi_img]),
            engine   = "ee",
            geometry = geometry,
            scale    = self._scale,
            crs      = self._crs,
        )

        band = "NDVI" if "NDVI" in ds else list(ds.data_vars)[0]
        arr, _, _, _ = self._to_2d_yx(ds[band])
        arr = arr.astype(np.float64)

        # Resample to match DEM grid if needed
        arr = self._match_grid(arr, ref_profile)
        arr = np.clip(arr, -1.0, 1.0)
        print(f"  NDVI shape: {arr.shape}, range: "
              f"{np.nanmin(arr):.3f}–{np.nanmax(arr):.3f}")
        if return_xarray:
            return arr, self._array_to_dataarray(arr, ref_profile, "ndvi")
        return arr

    def _fetch_rainfall(
        self,
        ee,
        xr,
        geometry,
        ref_profile,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        return_xarray: bool = False,
    ):
        """Fetch and aggregate rainfall over the date range."""
        cfg  = self._rf_cfg
        s_date = start_date if start_date is not None else self._start
        e_date = end_date if end_date is not None else self._end
        coll = (
            ee.ImageCollection(cfg["asset"])
            .filterDate(s_date, e_date)
            .filterBounds(geometry)
            .select(cfg["band"])
        )

        if cfg["agg"] == "sum":
            rf_img = coll.sum().clip(geometry)
        else:
            rf_img = coll.mean().clip(geometry)

        import xee  # noqa
        ds = xr.open_dataset(
            ee.ImageCollection([rf_img]),
            engine   = "ee",
            geometry = geometry,
            scale    = self._rf_cfg["scale"],   # use native RF scale; resample after
            crs      = self._crs,
        )

        band = cfg["band"] if cfg["band"] in ds else list(ds.data_vars)[0]
        arr, _, _, _ = self._to_2d_yx(ds[band])
        arr = arr.astype(np.float64)
        arr[arr < 0] = np.nan

        # Resample to DEM grid
        arr = self._match_grid(arr, ref_profile)
        print(f"  Rainfall shape: {arr.shape}, range: "
              f"{np.nanmin(arr):.2f}–{np.nanmax(arr):.2f}")
        if return_xarray:
            return arr, self._array_to_dataarray(arr, ref_profile, "rainfall")
        return arr

    def _fetch_landcover(self, ee, xr, geometry, ref_profile, return_xarray: bool = False):
        """Fetch integer land-cover classification and resample to DEM grid."""
        cfg  = self._lc_cfg
        assert cfg is not None
        band = cfg["band"]

        if cfg["type"] == "image":
            img  = ee.Image(cfg["asset"]).select(band)
            coll = ee.ImageCollection([img])
        else:
            # Take the most recent image in the collection
            coll = (ee.ImageCollection(cfg["asset"])
                    .filterBounds(geometry)
                    .sort("system:time_start", False)
                    .limit(1)
                    .select(band))

        import xee  # noqa
        ds = xr.open_dataset(
            coll,
            engine   = "ee",
            geometry = geometry,
            scale    = cfg["scale"],
            crs      = self._crs,
        )

        b   = band if band in ds else list(ds.data_vars)[0]
        arr, _, _, _ = self._to_2d_yx(ds[b])
        arr = arr.astype(np.int32)

        # Nearest-neighbour resample to DEM grid (preserve integer codes)
        arr_f = self._match_grid(
            arr.astype(np.float32), ref_profile, resampling="nearest"
        )
        arr_i = np.round(arr_f).astype(np.int32)
        arr_i[np.isnan(arr_f)] = -1   # nodata → -1

        print(f"  Land cover shape: {arr_i.shape}, "
              f"unique codes: {sorted(set(arr_i.ravel().tolist()))[:10]}…")
        if return_xarray:
            return arr_i, self._array_to_dataarray(arr_i, ref_profile, "landcover")
        return arr_i

    @staticmethod
    def _match_grid(arr, ref_profile, resampling: str = "bilinear"):
        """
        Resample *arr* to match *ref_profile* grid dimensions using
        rasterio's in-memory warp.
        """
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject

        ref_h = ref_profile["height"]
        ref_w = ref_profile["width"]

        if arr.shape == (ref_h, ref_w):
            return arr

        tr = ref_profile["transform"]
        lon_min = tr.c
        lat_min = tr.f - ref_profile["height"] * abs(tr.e)
        lon_max = tr.c + ref_profile["width"] * tr.a
        lat_max = tr.f

        src_tr = from_bounds(
            lon_min, lat_min, lon_max, lat_max, arr.shape[1], arr.shape[0]
        )
        dst_arr = np.empty((ref_h, ref_w), dtype=np.float64)
        resampling_mode = {
            "bilinear": Resampling.bilinear,
            "nearest": Resampling.nearest,
        }.get(resampling)
        if resampling_mode is None:
            raise ValueError(
                f"Unsupported resampling='{resampling}'. Use 'bilinear' or 'nearest'."
            )

        reproject(
            source       = arr,
            destination  = dst_arr,
            src_transform= src_tr,
            src_crs      = ref_profile["crs"],
            dst_transform= ref_profile["transform"],
            dst_crs      = ref_profile["crs"],
            resampling   = resampling_mode,
            src_nodata   = np.nan,
            dst_nodata   = np.nan,
        )
        return dst_arr
