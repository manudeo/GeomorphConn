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

A saved GUI HTML example is available in this repository:

- [../GeomorphConn GUI.htm](../GeomorphConn GUI.htm)

If opened from docs, you may also need the sibling assets folder:

- [../GeomorphConn GUI_files](../GeomorphConn%20GUI_files)

## Notes

- Current GUI compute mode is local-raster-driven.
- Direct GEE-driven GUI workflows are planned as future work.

For complete option details, see [options.md](options.md).
