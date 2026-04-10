# Installation

GeomorphConn requires **Python 3.10 or later**.

---

## Current method: install from source (GitHub)

This is the recommended method while the package is under active development
and not yet published to PyPI or conda-forge.

### 1. Clone the repository

```bash
git clone https://github.com/manudeo/GeomorphConn.git
cd GeomorphConn
```

### 2. Create and activate an environment

**Option A — Python venv (no conda required)**

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux / WSL2:
```bash
python -m venv .venv
source .venv/bin/activate
```

> If PowerShell blocks the activation script, run this first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

**Option B — conda / mamba environment**

```bash
conda create -n geomorphconn python=3.11
conda activate geomorphconn
```

Or with a custom environment name:
```bash
conda create -n myenv python=3.11
conda activate myenv
```

> Python 3.11 is recommended for best compatibility with Landlab and Numba.
> Any version from 3.10 to 3.13 is supported.

### 3. Install the package

**Core only (no GEE, no GUI):**
```bash
pip install -e .
```

**With GUI (Streamlit):**
```bash
pip install -e ".[gui]"
```

**With Google Earth Engine support:**
```bash
pip install -e ".[gee]"
```

**With optional speed-up (Numba JIT):**
```bash
pip install -e ".[fast]"
```

**Everything at once:**
```bash
pip install -e ".[all]"
```

The `-e` flag installs in *editable* mode — any changes you pull from the
repository are immediately reflected without reinstalling.

### 4. Verify

```bash
geomorphconn --version
geomorphconn welcome
```

---

## Optional extras explained

| Extra | What it adds | When you need it |
|---|---|---|
| `gui` | Streamlit, geopandas, shapely, pyogrio | Running the interactive GUI or IC-toward-target in the GUI |
| `gee` | earthengine-api, xee, xarray | Fetching DEM / NDVI / rainfall from Google Earth Engine |
| `fast` | numba | Faster IC computation on grids larger than ~500 × 500 |
| `target` | geopandas, shapely, pyogrio | IC-toward-target via Python API or CLI only (no GUI) |
| `dev` | pytest, ruff, black, pre-commit | Development and testing |
| `all` | All of the above except `dev` | Full feature set |

---

## Getting updates

```bash
git pull
# No reinstall needed if you used -e (editable install)
```

---

<!--
## Install from PyPI  (once published)

### Core
```bash
pip install geomorphconn
```

### With GUI
```bash
pip install "geomorphconn[gui]"
```

### With GEE support
```bash
pip install "geomorphconn[gee]"
```

### Everything
```bash
pip install "geomorphconn[all]"
```
-->

<!--
## Install via conda-forge  (once published)

```bash
conda install -c conda-forge geomorphconn
```

With optional extras (conda-forge recipe will include these as separate packages):
```bash
conda install -c conda-forge geomorphconn streamlit earthengine-api numba
```
-->

---

## ArcGIS Pro users

No Python installation is required. Download the toolbox from the repository:

```
arcgis_tools/ConnectivityTools.atbx
```

Open it in ArcGIS Pro via **Catalog → Toolboxes → Add Toolbox**. See
[arcgis_tools/README.md](arcgis_tools/README.md) for full instructions.

The ArcGIS tools are recommended for **large catchments** (regional/national
scale) where the Python package's memory requirements for DINF/MFD routing
become prohibitive. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#large-dem--memoryerror)
for a full explanation.

---

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 or 3.12 |
| RAM (D8, any size) | 4 GB | 8 GB |
| RAM (DINF/MFD, <5 M nodes) | 8 GB | 16 GB |
| RAM (DINF/MFD, 5–30 M nodes) | 32 GB | 64 GB+ |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Latest |

> For DEMs larger than ~5 million nodes, **D8 flow direction or the ArcGIS
> tools are strongly recommended** regardless of available RAM.
> See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for details.

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues including:

- `geomorphconn` command not found after install
- MemoryError on large DEMs
- Raster alignment problems
- Target vector not producing nodes
- Streamlit GUI connection drops on Windows
