# Solid-state convection and stagnant-lid rheology

This tutorial walks through running Aragog with temperature-, pressure-, and stress-dependent solid-state mantle rheology (`[phase_solid.rheology]`). Starting from the bundled `abe_solid.toml` template, we compare a constant-viscosity solid mantle (`enabled = false`) against an Arrhenius diffusion-creep mantle (`enabled = true`), inspect the stagnant-lid boundary-layer diagnostics on `solver.state._lid_state`, and lower the Byerlee yield stress to observe plastic yielding in the cold upper boundary layer.

For the mathematical formulation and benchmark comparisons, see [Solid-state convection](../Explanations/solid_state_convection.md) and [Configuration: `[phase_solid.rheology]`](../How-to/configuration.md#phase_solidrheology).

## Prerequisites

- Aragog is installed (`pip install -e .`).
- The SPIDER-format pressure-entropy EOS tables are staged on disk (see [First-run tutorial](firstrun.md#1-locate-or-stage-the-eos-tables) for the `spider_eos_test_data.tar.gz` download commands and `ARAGOG_TEST_EOS_DIR`).

## 1. Enable `[phase_solid.rheology]` in a TOML configuration

Scaffold a working configuration from the bundled `abe_solid` template:

```bash
aragog new ssc_demo --from abe_solid
```

To turn on solid-state rheology in `ssc_demo.toml`, set `viscosity = 3.0e+17` in the existing `[phase_solid]` section and append a `[phase_solid.rheology]` table:

```toml
[phase_solid.rheology]
enabled = true
stress_closure_mode = "lid"
g_homologous = 25.0
t_sol_0 = 1400.0
arrhenius_t_ref = 1600.0
viscosity_max_log10 = 40.0
yield_stress_c = 50.0e6
yield_stress_mu = 0.6
yield_stress_max = 500.0e6
s_top_inf = 0.22
c_theta = 0.15
```

When `enabled = false` (the default), `[phase_solid].viscosity` is a uniform solid-phase dynamic viscosity ($\eta_\mathrm{solid} = 3.0e+17\,\mathrm{Pa\,s}$). When `enabled = true`, `[phase_solid].viscosity` is the reference viscosity $\eta_0$ at $(T_\mathrm{ref}, P = 0)$ on the reference solidus `t_sol_0`, and the solver evaluates the two-pass stagnant-lid closure (`stress_closure_mode = "lid"`, the default mode when enabled) at every right-hand-side call.

## 2. Compare rheology-off and rheology-on runs from Python

The script below loads `cfg/abe_solid.toml` (or `ssc_demo.toml` via `Parameters.from_file`), sets an initial entropy profile $S(r) = 2050.0 - 1720.0\,(0.2\,\xi + 0.8\,\xi^4)\,\mathrm{J/(kg\,K)}$ spanning from $2050.0\,\mathrm{J/(kg\,K)}$ at the core-mantle boundary to $330.0\,\mathrm{J/(kg\,K)}$ at the surface, integrates for $10.0\,\mathrm{yr}$, and prints every comparison quantity for `enabled = False` (default) and `enabled = True`:

```python
from dataclasses import replace
import importlib.resources
import os
from pathlib import Path
import numpy as np

from aragog.eos import EntropyEOS
from aragog.parser import Parameters
from aragog.rheology import SolidRheologyParams
from aragog.solver import EntropySolver

eos_dir = Path(os.environ["ARAGOG_TEST_EOS_DIR"])
cfg_path = Path(str(importlib.resources.files("aragog").joinpath("cfg/abe_solid.toml")))


def run_case(rheology: SolidRheologyParams) -> EntropySolver:
    params = Parameters.from_file(cfg_path)
    params.boundary_conditions = replace(
        params.boundary_conditions,
        core_bc="quasi_steady",
        inner_boundary_condition=1,
    )
    params.solver.end_time = 10.0
    params.energy.kappah_floor = 0.0
    params.phase_solid = replace(
        params.phase_solid,
        viscosity=3.0e+17,
        rheology=rheology,
    )
    solver = EntropySolver(params, EntropyEOS(eos_dir))
    solver.initialize()
    n_stag = params.mesh.number_of_nodes - 1
    xi = np.linspace(0.0, 1.0, n_stag)
    s_init = 2050.0 - 1720.0 * (0.2 * xi + 0.8 * xi**4)
    solver.set_initial_entropy(s_init)
    solver.solve()
    out = solver.get_state()
    solver.dSdt(params.solver.end_time, out.S_final)
    return solver


solver_off = run_case(SolidRheologyParams(enabled=False))
solver_on = run_case(SolidRheologyParams(enabled=True))

for label, solver in (("off", solver_off), ("on", solver_on)):
    out = solver.get_state()
    visc_b = np.asarray(solver.state.phase_basic.viscosity()).ravel()
    eta_diff_b = np.asarray(solver.state.phase_basic.eta_diff()).ravel()
    print(
        f"Rheology {label}: status={out.status}, "
        f"eta_surf={visc_b[-1]:.3e} Pa s, "
        f"eta_cmb={visc_b[0]:.3e} Pa s, "
        f"eta_diff_cmb={eta_diff_b[0]:.3e} Pa s, "
        f"max_jconv={np.max(out.jconv_b):.2f} W/m^2, "
        f"max_kappah={np.max(out.eddy_diff):.4f} m^2/s, "
        f"T_magma={out.T_magma:.2f} K, "
        f"T_core={out.T_core:.2f} K"
    )
```

With the CI test EOS tables, the two runs produce:

| Quantity | `enabled = False` (default) | `enabled = True` (`stress_closure_mode = "lid"`) |
|---|---|---|
| Solver status (`out.status`) | `0` | `0` |
| Surface viscosity $\eta_\mathrm{surf}$ (`phase_basic.viscosity()[-1]`) [Pa s] | `$3.000e+17$` | `$9.352e+27$` |
| Bulk CMB viscosity $\eta_\mathrm{cmb}$ (`phase_basic.viscosity()[0]`) [Pa s] | `$1.664e+17$` | `$1.916e+18$` |
| Solid CMB diffusion viscosity $\eta_\mathrm{diff,cmb}$ (`phase_basic.eta_diff()[0]`) [Pa s] | `$3.000e+17$` | `$3.601e+18$` |
| Peak convective heat flux $\max(F_\mathrm{conv})$ (`np.max(out.jconv_b)`) [W/m$^2$] | `$166020.23$` | `$249.54$` |
| Peak eddy diffusivity $\max(\kappa_h)$ (`np.max(out.eddy_diff)`) [m$^2$/s] | `$16.3075$` | `$0.0422$` |
| Near-surface magma temperature $T_\mathrm{magma}$ (`out.T_magma`) [K] | `$781.64$` | `$781.55$` |
| Core temperature $T_\mathrm{core}$ (`out.T_core`) [K] | `$3563.96$` | `$3563.96$` |

With `enabled = False`, the cold upper boundary layer keeps the uniform solid viscosity ($\eta_\mathrm{surf} = 3.000e+17\,\mathrm{Pa\,s}$), allowing vigorous near-surface overturn ($\max(F_\mathrm{conv}) = 166020.23\,\mathrm{W/m^2}$). With `enabled = True`, Arrhenius stiffening raises the surface viscosity to $\eta_\mathrm{surf} = 9.352e+27\,\mathrm{Pa\,s}$ and forms a conductive stagnant lid that suppresses convective transport in the cold upper shell ($\max(F_\mathrm{conv}) = 249.54\,\mathrm{W/m^2}$). At the partially molten CMB node, `phase_basic.eta_diff()[0]` reports the pure solid-phase diffusion viscosity ($3.000e+17\,\mathrm{Pa\,s}$ vs $3.601e+18\,\mathrm{Pa\,s}$), whereas `phase_basic.viscosity()[0]` blends that solid viscosity with the liquid phase across the melt fraction ($1.664e+17\,\mathrm{Pa\,s}$ vs $1.916e+18\,\mathrm{Pa\,s}$).

## 3. Read the stagnant-lid diagnostics

When `enabled = True` and `stress_closure_mode = "lid"`, `solver.state._lid_state` stores the boundary-layer diagnostics computed by `compute_stagnant_lid_state` at the most recent state evaluation:

```python
lid = solver_on.state._lid_state
print(f"has_lid:     {lid['has_lid']}")
print(f"d_lid:       {float(lid['d_lid']) * 1e-3:.2f} km")
print(f"r_lid_base:  {float(lid['r_lid_base']) * 1e-3:.2f} km")
print(f"T_i:         {float(lid['T_i']):.2f} K")
print(f"T_lid:       {float(lid['T_lid']):.2f} K")
print(f"theta:       {float(lid['theta']):.3f}")
print(f"dT_rh:       {float(lid['dT_rh']):.2f} K")
print(f"eta_i:       {float(lid['eta_i']):.3e} Pa s")
print(f"tau_d_lid:   {float(lid['tau_d_lid']) * 1e-3:.2f} kPa")
print(f"tau_y_lid:   {float(lid['tau_y_lid']) * 1e-6:.2f} MPa")
print(f"w_y:         {float(lid['w_y']):.3e}")
```

For the `enabled = True` run at $t = 10.0\,\mathrm{yr}$, these diagnostics evaluate to:

- `has_lid = True`, with conductive lid thickness $d_\mathrm{lid} = 31.53\,\mathrm{km}$ ($d_\mathrm{lid} = 31534.83\,\mathrm{m}$) and lid-base radius $r_\mathrm{lid,base} = 6339.47\,\mathrm{km}$.
- Convecting interior temperature $T_i = 1962.84\,\mathrm{K}$, lid temperature $T_\mathrm{lid} = 1441.97\,\mathrm{K}$, and lid-base isotherm target $T_\mathrm{lid,iso} = 1788.91\,\mathrm{K}$.
- Frank-Kamenetskii parameter $\theta = 14.942$, rheological temperature scale $\Delta T_\mathrm{rh} = 79.06\,\mathrm{K}$, and interior viscosity $\eta_i = 4.494e+18\,\mathrm{Pa\,s}$.
- Lid-scaled convective driving stress $\tau_{d,\mathrm{lid}} = 653.04\,\mathrm{kPa}$ ($653043.19\,\mathrm{Pa}$) against a Byerlee lid yield strength of $\tau_{y,\mathrm{lid}} = 500.00\,\mathrm{MPa}$, so the plastic yielding weight is $w_y = 2.116e-09$ (`0` yielding basic nodes) and the lid remains in the unyielded stagnant-lid regime.

## 4. Modify yield parameters to trigger plastic lid mobilisation

To observe plastic yielding in Pass 2 of the stress closure, lower the cohesion `yield_stress_c` below $\tau_{d,\mathrm{lid}} = 653.04\,\mathrm{kPa}$ and set `yield_stress_mu = 0.0`:

```python
solver_yield = run_case(
    SolidRheologyParams(
        enabled=True,
        stress_closure_mode="lid",
        yield_stress_c=150.0,
        yield_stress_mu=0.0,
        yield_stress_max=1.0e8,
    )
)
lid_yield = solver_yield.state._lid_state
eta_diff = np.asarray(solver_yield.state.phase_basic.eta_diff()).ravel()
eta_eff = np.asarray(solver_yield.state.phase_basic.viscosity()).ravel()
n_yielding = int(np.sum(eta_eff < eta_diff * (1.0 - 1.0e-12)))
print(f"w_y = {float(lid_yield['w_y']):.1f}, yielding nodes = {n_yielding}")
print(f"Surface eta_diff = {eta_diff[-1]:.3e} Pa s -> eta_eff = {eta_eff[-1]:.3e} Pa s")
```

With `yield_stress_c = 150.0` Pa and `yield_stress_mu = 0.0`:

- The lid yield strength is $\tau_{y,\mathrm{lid}} = 150.0\,\mathrm{Pa}$, below $\tau_{d,\mathrm{lid}} = 652.85\,\mathrm{kPa}$.
- The yielding blend weight reaches $w_y = 1.0$, and `10` basic nodes inside the upper boundary layer yield plastically (`eta_eff < eta_diff`).
- At the surface basic node, plastic yielding reduces the effective viscosity from $\eta_\mathrm{diff,surf} = 9.193e+27\,\mathrm{Pa\,s}$ to $\eta_\mathrm{eff,surf} = 1.033e+15\,\mathrm{Pa\,s}$, increasing the peak convective heat flux from $249.54\,\mathrm{W/m^2}$ to $249.69\,\mathrm{W/m^2}$.

![Radial profiles of viscosity and convective heat flux for rheology-off, default stagnant-lid, and plastic-yielding cases](../assets/tutorials/ssc_tutorial_profiles.svg#only-light)
![Radial profiles of viscosity and convective heat flux for rheology-off, default stagnant-lid, and plastic-yielding cases](../assets/tutorials/ssc_tutorial_profiles_dark.svg#only-dark)
