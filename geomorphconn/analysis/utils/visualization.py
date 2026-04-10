"""
Visualization utilities for CRU classification maps.

Provides color palettes and legend generation for dynamic CRU
classification results, with ArcGIS compatibility for future
toolbox integration.
"""

import json
from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr


def get_cru_colormap() -> mcolors.ListedColormap:
    """
    Generate perceptually distinct colormap for 13-class CRU classification.
    
    Maps integer codes [−6, +6] to distinct colors:
    - Negative (−6 to −1): Cool tones for coldspots (blue/cyan/green)
    - Zero (0): White/light gray for no pattern
    - Positive (+1 to +6): Warm tones for hotspots (red/orange/yellow)
    
    Returns
    -------
    matplotlib.colors.ListedColormap
        21-color colormap (for codes in range [−6, 6]).
    """
    # NOTE: Index 0 maps to code −6, index 6 to code 0, index 12 to code +6
    # We'll create a 13-entry colormap and map indices [0..12] ↔ codes [−6..+6]
    
    colors = [
        '#1a1a4d',  # −6 (New Coldspot, very dark blue)
        '#0047ab',  # −5 (Sporadic Coldspot, royal blue)
        '#0080ff',  # −4 (Persistent Coldspot, bright blue)
        '#00d4ff',  # −3 (Diminishing Coldspot, cyan)
        '#00ff80',  # −2 (Intensifying Coldspot, light green)
        '#80ff00',  # −1 (Emerging Coldspot, green-yellow)
        '#ffffff',  # 0 (No Pattern, white)
        '#ffff00',  # +1 (Emerging Hotspot, yellow)
        '#ff8000',  # +2 (Intensifying Hotspot, orange)
        '#ff4000',  # +3 (Diminishing Hotspot, deep orange)
        '#ff0000',  # +4 (Persistent Hotspot, red)
        '#8b0000',  # +5 (Sporadic Hotspot, dark red)
        '#4d0000',  # +6 (New Hotspot, very dark red)
    ]
    
    cmap = mcolors.ListedColormap(colors)
    return cmap


def get_cru_norm() -> mcolors.Normalize:
    """
    Normalization for CRU classification codes [−6, +6] → [0, 1] range.
    
    Returns
    -------
    matplotlib.colors.Normalize
        Normalized mapper from code range −6..+6 to 0..1.
    """
    return mcolors.Normalize(vmin=-6, vmax=6)


def plot_cru_map(
    cru_data: xr.DataArray,
    title: str = "Dynamic CRU Classification",
    figsize: Tuple[int, int] = (12, 8),
    cbar_label: str = "CRU Class",
) -> plt.Figure:
    """
    Convenience function to plot CRU classification map with colormap.
    
    Parameters
    ----------
    cru_data : xr.DataArray
        Output from classify_dynamic_crus(), shape (y, x), values [−6, +6].
    title : str, optional
        Figure title. Default: "Dynamic CRU Classification".
    figsize : tuple, optional
        Figure size (width, height). Default: (12, 8).
    cbar_label : str, optional
        Colorbar label. Default: "CRU Class".
    
    Returns
    -------
    matplotlib.pyplot.Figure
        Figure object (for further customization, saving, etc.).
    """
    from geomorphconn.analysis.cru_dynamics import CRU_CLASSES
    
    cmap = get_cru_colormap()
    norm = get_cru_norm()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cru_data.values, cmap=cmap, norm=norm, origin='upper')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    # Colorbar with tick labels
    cbar = plt.colorbar(im, ax=ax, label=cbar_label)
    tick_codes = list(range(-6, 7))
    cbar.set_ticks(tick_codes)
    cbar.set_ticklabels([CRU_CLASSES.get(code, f"{code}") for code in tick_codes], fontsize=9)
    
    return fig


def generate_arcgis_legend_dict() -> dict:
    """
    Generate ArcGIS-compatible legend dictionary for CRU visualization.
    
    Useful for future integration with ArcGIS Pro toolboxes.
    
    Returns
    -------
    dict
        Legend mapping: {code: {"className": str, "description": str, "color": hex_str}}
    """
    from geomorphconn.analysis.cru_dynamics import CRU_CLASSES
    
    # Color hex codes (from get_cru_colormap)
    color_codes = [
        '#1a1a4d',  # −6
        '#0047ab',  # −5
        '#0080ff',  # −4
        '#00d4ff',  # −3
        '#00ff80',  # −2
        '#80ff00',  # −1
        '#ffffff',  # 0
        '#ffff00',  # +1
        '#ff8000',  # +2
        '#ff4000',  # +3
        '#ff0000',  # +4
        '#8b0000',  # +5
        '#4d0000',  # +6
    ]
    
    descriptions = {
        1: "Recently became connected. High priority for intervention.",
        2: "Connectivity increasing. Trend of intensification detected.",
        3: "Connectivity decreasing. Positive restoration trajectory.",
        4: "Consistently connected. Stable pattern over time.",
        5: "Intermittent connectivity. On-and-off pattern.",
        6: "First-ever connection. Novel feature, needs monitoring.",
        -1: "Recently became disconnected. Improved isolation.",
        -2: "Disconnection increasing. Fragmentation trend.",
        -3: "Disconnection decreasing. Re-connection trend.",
        -4: "Persistently disconnected. Stable isolation.",
        -5: "Intermittent disconnection. Variable isolation.",
        -6: "First-time disconnection. Novel isolation feature.",
        0: "No significant pattern. Background/noise.",
    }
    
    legend = {}
    for code, class_name in CRU_CLASSES.items():
        idx = code + 6  # Map [−6, +6] → [0, 12]
        legend[code] = {
            "className": class_name,
            "description": descriptions.get(code, "Unknown"),
            "color": color_codes[idx],
        }
    
    return legend


def generate_qgis_legend_dict() -> dict:
    """
    Generate QGIS-friendly legend metadata for CRU classes.

    Returns
    -------
    dict
        Mapping with value, label and RGBA values compatible with QGIS palette workflows.
    """
    arc_legend = generate_arcgis_legend_dict()
    items = []
    for code in range(-6, 7):
        hex_color = arc_legend[code]["color"].lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        items.append(
            {
                "value": code,
                "label": arc_legend[code]["className"],
                "description": arc_legend[code]["description"],
                "color_rgba": [r, g, b, 255],
            }
        )
    return {"colormap": "CRU_CLASSES", "items": items}


def export_cru_geotiff(
    cru_data: xr.DataArray,
    output_tif: str,
    create_qgis_sidecars: bool = True,
    create_arcgis_sidecars: bool = True,
) -> Dict[str, str]:
    """
    Export classified CRU map as GeoTIFF with embedded palette and GIS sidecars.

    Parameters
    ----------
    cru_data : xr.DataArray
        2D CRU classification map (values in [-6, 6]).
    output_tif : str
        Output GeoTIFF path.
    create_qgis_sidecars : bool, optional
        If True, writes QGIS-compatible legend JSON and TXT palette sidecars.
    create_arcgis_sidecars : bool, optional
        If True, writes ArcGIS-compatible legend JSON sidecar.

    Returns
    -------
    dict
        Paths of generated files.
    """
    import rasterio

    output_path = Path(output_tif)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cru_data.ndim != 2:
        raise ValueError(f"Expected 2D CRU map, got shape {cru_data.shape}")

    # Write raster first.
    cru_data.astype("int16").rio.to_raster(str(output_path))

    arc_legend = generate_arcgis_legend_dict()
    qgis_legend = generate_qgis_legend_dict()

    # Build RasterIO color table where class value maps to RGBA.
    colormap = {}
    for item in qgis_legend["items"]:
        value = int(item["value"])
        r, g, b, a = item["color_rgba"]
        colormap[value] = (r, g, b, a)

    # Embed colormap + metadata into GeoTIFF.
    with rasterio.open(output_path, "r+") as dst:
        dst.write_colormap(1, colormap)
        dst.update_tags(
            1,
            CRU_CLASSES=json.dumps({k: v["className"] for k, v in arc_legend.items()}),
            CRU_DESCRIPTIONS=json.dumps({k: v["description"] for k, v in arc_legend.items()}),
        )
        dst.set_band_description(1, "Connectivity Response Unit class")

    generated = {"tif": str(output_path)}

    if create_qgis_sidecars:
        qgis_json_path = output_path.with_suffix(".qgis.legend.json")
        qgis_txt_path = output_path.with_suffix(".qgis.colormap.txt")
        with qgis_json_path.open("w", encoding="utf-8") as f:
            json.dump(qgis_legend, f, indent=2)

        # QGIS-compatible palette text: value,R,G,B,A,label
        with qgis_txt_path.open("w", encoding="utf-8") as f:
            for item in qgis_legend["items"]:
                r, g, b, a = item["color_rgba"]
                f.write(
                    f"{item['value']},{r},{g},{b},{a},{item['label']}\n"
                )

        generated["qgis_json"] = str(qgis_json_path)
        generated["qgis_colormap_txt"] = str(qgis_txt_path)

    if create_arcgis_sidecars:
        arc_json_path = output_path.with_suffix(".arcgis.legend.json")
        with arc_json_path.open("w", encoding="utf-8") as f:
            json.dump(arc_legend, f, indent=2)
        generated["arcgis_json"] = str(arc_json_path)

    return generated
