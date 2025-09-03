## Overview
Analyze a CT step‑wedge time profile to detect plateau levels and transitions, estimate step lengths and field width, and compute a Percentage Depth Dose (PDD) curve with $$D_{20}/D_{10}$$.  
The script reads a CSV with projection indices and signal values, fits a 7‑segment piecewise model, and saves a four‑panel results figure plus a short console summary.  

## Requirements
- Python >= 3.12 (automatically handled by uv; otherwise install Python 3.12 manually).  
- Dependencies (also declared in pyproject.toml): matplotlib, numpy, pandas, scipy.  

## Quick start
- Put profiles.csv in the project root with columns: Raw New Profile X and Raw New Profile Y.  
- Run the script using either the **uv** workflow or the standard Python workflow below.  

## Using uv
- If uv is installed, this is the simplest path; uv will create a virtual environment, install dependencies, and use Python 3.12 automatically.  
- First run (installs everything as needed):  
  - uv sync  
  - uv run python stepwedge.py  
- If Python 3.12 isn’t present and uv doesn’t auto‑install, force it:  
  - uv python install 3.12  
  - uv run --python 3.12 python stepwedge.py  
- Optional: pin the local Python version for this project (creates .python-version):  
  - uv python pin 3.12  

## Without uv
- Create and activate a virtual environment:  
  - Linux/macOS:  
    - python3.12 -m venv .venv  
    - source .venv/bin/activate  
  - Windows (PowerShell):  
    - py -3.12 -m venv .venv  
    - .\.venv\Scripts\Activate.ps1  
- Install dependencies from pyproject.toml:  
  - pip install .  
- Run the script:  
  - python stepwedge.py  

## Input data
- Place profiles.csv at the project root.  
- Required columns (case‑sensitive):  
  - Raw New Profile X: projection index or time‑like axis (numeric).  
  - Raw New Profile Y: measured signal (numeric).  
- Rows with missing values in either column are dropped automatically.  

## Configuration
Edit the configuration block at the top of the script if needed:  
- CSV_PATH, X_COL, Y_COL: file and column names.  
- sample_rate_hz, couch_speed_mm_s: acquisition/machine settings.  
- nominal_step_lengths_mm: nominal physical step sizes (mm).  
- depths_water_mm: water‑equivalent depths for PDD (mm).  
- OUT_PNG: output figure filename.  

## What it produces
- stepwedge_results.png with four panels:  
  - A) Measured vs fitted time profile.  
  - B) Fitted schematic with p1–p12 vertical markers and labeled $$s_1 \ldots s_7$$.  
  - C) PDD points (normalized at 5 cm) and exponential fit; shows $$D_{20}/D_{10}$$.  
  - D) Nominal vs measured step lengths with ±1% tolerance bands.  
- Console summary including:  
  - Fit $$R^2$$ for the time‑profile model.  
  - Plateau levels $$s_1 \ldots s_7$$.  
  - Transition points $$p_1 \ldots p_{12}$$ (in projections).  
  - Field width from $$(p_2 - p_1)$$.  
  - Measured and nominal step lengths (mm).  
  - PDD $$D_{20}/D_{10}$$.  

## Tips and troubleshooting
- If the CSV path or column names differ, update CSV_PATH, X_COL, and Y_COL accordingly.  
- If the plot looks over‑ or under‑smoothed, adjust the Savitzky–Golay window logic in _safe_savgol.  
- If the fit struggles (odd changepoints or plateaus), ensure X is strictly increasing; the script sorts by X before fitting.  
- If uv isn’t available, use the “Without uv” section; both workflows are equivalent in outcome.  

## Project metadata
- Name: stepwedge  
- Version: 0.1.0  
- Python: >= 3.12  
- Dependencies: matplotlib>=3.10.6, numpy>=2.3.2, pandas>=2.3.2, scipy>=1.16.1  

## Example commands
- Fast path with **uv**:  
  - uv sync  
  - uv run python stepwedge.py  
- Standard Python:  
  - python3.12 -m venv .venv && source .venv/bin/activate  
  - pip install .  
  - python stepwedge.py

[1](https://docs.astral.sh/uv/guides/install-python/)
[2](https://docs.astral.sh/uv/getting-started/installation/)
[3](https://github.com/astral-sh/uv)
[4](https://realpython.com/python-uv/)
[5](https://pydevtools.com/handbook/how-to/how-to-install-python-with-uv/)
[6](https://igorstechnoclub.com/uv-python-package-manager-beginners-guide/)
[7](https://www.digitalocean.com/community/conceptual-articles/uv-python-package-manager)
[8](https://www.youtube.com/watch?v=AMdG7IjgSPM)