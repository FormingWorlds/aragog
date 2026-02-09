# First-run tutorial 

Welcome to this first-run tutorial. We’ll run a single configuration, then export and inspect the results.

## What does aragog do?

Aragog solves the **time evolution of a 1-D radial temperature profile** inside a spherical shell (mantle) using conduction and optional parameterized convection, melt physics, and internal heating. 

## Goals
Succesfully install aragog; get to a **first successful model run** and a **NetCDF output and plots**.

## Assumptions
 - You’re using Python 3.10+ (3.11+ recommended).

## 1. Install aragog (editable install)

Clone aragog to a suitable location:

```sh
git clone git@github.com:FormingWorlds/aragog.git aragog
cd aragog
```

Create a Conda environment:

```sh
conda create -n aragog python
conda activate aragog
```

No `conda` installed? Create a virtual environment (venv):

```sh
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

Finally, install aragog and all dependencies:

```sh 
pip install -e ".[docs]"
```

## 2. OPTIONAL: Choose a data directory

aragog’s data helper uses `FWL_DATA` to decide where lookup tables/logs go. If you have not set this up yet, set it to something you control:

```sh
mkdir /your/local/path/FWL_DATA
```

```sh
export FWL_DATA="/your/local/path/FWL_DATA"   # macOS/Linux
```

Then check what aragog sees:

```bash
aragog env
```

You should see:
- `FWL_DATA location: <...>`

> **Note** If you are coming from a different module of the PROTEUS Framework, FWL_DATA might be set to a datafolder in this other module. Check this, so that you know where your data goes.

## 3. Download official lookup tables

To download published lookup tables, run:

```sh
aragog download all
```

## 4. Run aragog from Python 

Create `first.py`:

```python
from aragog.solver import Solver
from aragog.output import Output
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")  

# Load config, initialize, solve
solver = Solver.from_file("aragog/cfg/abe_mixed.cfg")
solver.initialize()
solver.solve()

# Postprocess
out = Output(solver)

print("\n=== Aragog run summary ===")
print("Config file: aragog/cfg/abe_mixed.cfg")
print("Stop early:", getattr(solver, "stop_early", None))
print("Time span (years):", out.time_range)
print("Final surface temperature (K):", out.solution_top_temperature)
print("Global melt fraction (final):", float(out.melt_fraction_global))
print("Mantle mass (kg):", float(out.mantle_mass))
print("Core mass (kg):", float(out.core_mass))

# make an output directory if it doesn't exist
out_dir = Path("out")
out_dir.mkdir(parents=True, exist_ok=True)

# Save a snapshot (last time index) to NetCDF
out.write_at_time("out/first_output.nc", tidx=-1, compress=True)

# Quick plots (temperature, viscosity, fluxes, etc.)
plot_file = "out/first_output.png"
out.plot(num_lines=7)
plt.savefig(plot_file, dpi=200, bbox_inches="tight")
plt.close("all")

print("Saved plot:", plot_file)

```

Run it:

```sh
python first.py
```

If everything is working, you should see:

- solver logging indicating integration succeeded
- a NetCDF file `first_output.nc` in the `out` directory
- a matplotlib figure with multiple panels vs pressure in the `out` directory

## 5. NetCDF outputs

aragog writes mesh variables like:

- `radius_b` (km), `pres_b` (GPa), `temp_b` (K), `phi_b` (melt fraction)
- `Fcond_b`, `Fconv_b`, `Ftotal_b` (W/m^2)
- `log10visc_b`, `density_b`, `heatcap_b`
- heating sources on staggered nodes: `Hradio_s`, `Hvol_s`, `Htidal_s`, `Htotal_s`

to NetCDF files. It also includes scalar metadata (time, global melt fraction, etc.).

Inspect the header of your file:

```sh
ncdump -h out/first_output.nc
```

For more info on opening, inspecting and plotting NetCDF files, you can go through the how-to [Opening NetCDF output](../How-to/netcdf.md). 


## 6. Troubleshooting

### Zenodo download problems
If you download:

- ensure `zenodo_get` is installed 
- ensure you have write permissions to `FWL_DATA`.

### Solver is slow 
Update your configuration file, living in `aragog/cgf`, with:

- fewer nodes (`number_of_nodes = 40`)
- shorter run (`end_time = 1e3`)
- looser tolerances (`atol=1e-5`, `rtol=1e-5`)
