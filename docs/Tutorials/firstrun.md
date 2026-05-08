# First-run tutorial

Welcome to this first-run tutorial. We will load a configuration, integrate a single mantle cooling problem with `EntropySolver`, and inspect the resulting profiles and scalar diagnostics.

## What does Aragog do?

Aragog evolves the **specific entropy $S(r,t)$** at staggered nodes inside a spherical mantle shell. The solver couples conduction, mixing-length convection, gravitational separation of melt, chemical mixing, radiogenic heating, and tidal heating. Temperature, density, and melt fraction are derived from $(P, S)$ via a tabulated equation of state.

## Goals

Reach a **first successful integration** of a coupled-mantle setup and read its results from the `SolverOutput` dataclass.

## Assumptions

- You are using Python 3.10+ (3.12 recommended).
- Aragog is installed (see [Installation](../How-to/installation.md)).
- A directory of SPIDER-format pressure-entropy EOS tables is available on disk (see [Reference: data](../Reference/data.md) for the file format and the canonical PALEOS data sources).

## 1. Locate or stage the EOS tables

Aragog's entropy solver requires a directory containing the following files:

| File | Format | Description |
|------|--------|-------------|
| `temperature_solid.dat`, `temperature_melt.dat` | 2D P-S grid | $T(P, S)$ for solid and liquid phases |
| `density_solid.dat`, `density_melt.dat` | 2D P-S grid | $\rho(P, S)$ |
| `heat_capacity_solid.dat`, `heat_capacity_melt.dat` | 2D P-S grid | $c_p(P, S)$ |
| `adiabat_temp_grad_solid.dat`, `adiabat_temp_grad_melt.dat` | 2D P-S grid | $(\partial T/\partial P)_S$ |
| `solidus_P-S.dat`, `liquidus_P-S.dat` | 2-column $(P, S)$ | Phase-boundary entropy at each pressure |

In a PROTEUS coupled run these tables are produced from the configured PALEOS or Wolf-Bower P-T file by the PROTEUS wrapper. For standalone work, point `eos_dir` at any directory containing this set.

## 2. Pick a configuration file

Aragog ships a few example configurations under `src/aragog/cfg/`. For this tutorial, copy `abe_solid.toml` to a working directory and edit the file paths in `[phase_solid]`, `[phase_liquid]`, and `[phase_mixed]` so that they point at your local data, then drop the `[scalings]` section if present (it is no longer consumed).

## 3. Run Aragog from Python

Create `first.py`:

```python
from pathlib import Path
from aragog import aragog_file_logger
from aragog.solver import EntropySolver

# Set up combined console + file logging in the current directory.
aragog_file_logger(log_dir=str(Path.cwd()))

# Build the solver from a TOML config and an EOS-table directory.
solver = EntropySolver.from_file(
    filename="abe_solid.toml",
    eos_dir="path/to/eos/tables",
)

solver.initialize()

# Set the initial entropy at staggered nodes (J/kg/K). A scalar
# value produces a uniform isentropic profile.
solver.set_initial_entropy(2900.0)

solver.solve()

out = solver.get_state()

print("=== Aragog run summary ===")
print(f"Status:       {out.status}")
print(f"T_magma:      {out.T_magma:.0f} K")
print(f"T_core:       {out.T_core:.0f} K")
print(f"Phi_global:   {out.Phi_global:.4f}")
print(f"M_mantle:     {out.M_mantle:.3e} kg")
print(f"E_th:         {out.E_th:.3e} J")
print(f"F_heat_total: {out.F_heat_total:.3e} W/m^2")
print(f"Cp_eff:       {out.Cp_eff:.0f} J/kg/K")
```

Run it:

```sh
python first.py
```

A successful run leaves `aragog.log` in the working directory and prints the scalar diagnostics. `out.S_final`, `out.T_stag`, `out.phi_stag`, and the basic-node fluxes (`out.jcond_b`, `out.jconv_b`, `out.jgrav_b`, `out.jmix_b`) are NumPy arrays you can plot directly.

## 4. Plot the entropy and melt-fraction profiles

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

axes[0].plot(out.S_final, out.r_stag * 1e-3)
axes[0].set_xlabel("Specific entropy [J/kg/K]")
axes[0].set_ylabel("Radius [km]")

axes[1].plot(out.phi_stag, out.r_stag * 1e-3)
axes[1].set_xlabel("Melt fraction")

fig.tight_layout()
fig.savefig("first_profiles.pdf")
```

## 5. Use Aragog inside PROTEUS

For a coupled atmosphere-interior simulation the configuration is built programmatically by the PROTEUS wrapper at `src/proteus/interior_energetics/aragog.py`. The wrapper drives `EntropySolver.set_initial_entropy()` from the previous step's profile, supplies the four-column external mesh file from Zalmoxis when `eos_method = 2`, and reads `SolverOutput` back into the PROTEUS `Interior_t` state. See [Standalone vs PROTEUS-integrated usage](../How-to/usage-paths.md) for the path comparison.

For a coupled walkthrough (atmosphere + interior + outgassing), see the PROTEUS [usage guide](https://proteus-framework.org/PROTEUS/How-to/usage.html).

## 6. Troubleshooting

**`status = -1` from `solve()`.** The solver hit an integration failure: typically the integrator collapsed its step size at a phase boundary or the EOS table was queried outside its $(P, S)$ domain. PROTEUS handles these via a retry ladder that calls `set_initial_dSdr_cmb` and a tolerance-relaxation knob. In standalone use, inspect `aragog.log` for the warning trail and consider enabling the SUNDIALS CVODE path with `solver_method = "cvode"` if `scikits.odes` is installed.

**Slow integration.** Loosen tolerances (`atol = 1e-5`, `rtol = 1e-5`), reduce `number_of_nodes` to 40-80 for first runs, or shorten the time window. The phase-aware `max_step` cap activates automatically near solidus and liquidus crossings.

**Entropy out of table range.** Check that the EOS table covers the expected $(P, S)$ envelope. For PALEOS tables sampled at 150 points per decade, the entropy axis typically spans roughly `[-100, 5500]` J/kg/K for MgSiO₃; values outside that range usually indicate an incorrect initial entropy or an oversharp boundary flux.
