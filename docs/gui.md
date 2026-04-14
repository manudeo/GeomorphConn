# GUI Usage

## Launch GUI

```bash
geomorphconn gui --backend streamlit
```

The GUI supports:

- Outlet and Target IC modes
- Weight factor selection (`rainfall`, `ndvi`, `roughness`)
- Uploaded local raster workflows
- Optional vector target path or upload in target mode
- Output-layer selection and export

## GUI example reference

A saved GUI page export is available as a PDF:

- [assets/GeomorphConn_GUI.pdf](assets/GeomorphConn_GUI.pdf)

## Notes

- Current GUI compute mode is local-raster-driven.
- Direct GEE-driven GUI workflows are planned as future work.

For complete option details, see [options.md](options.md).
