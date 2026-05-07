# Aragog vs SPIDER

Aragog is a deliberate descendant of SPIDER (Bower, Sanan, Wolf 2018, *Phys. Earth Planet. Inter.*, **274**, 49-62, doi:10.1016/j.pepi.2017.11.004) and shares its formulation: a 1-D spherically symmetric finite-volume mantle with specific entropy as the prognostic variable, modified-Newton BDF time integration, and SPIDER-style two-branch tanh smoothing across phase boundaries. The two codes diverge in implementation language, in the way the Jacobian is built, in the EOS tables they consume by default, and in how they connect to the outside world.

This page summarises the points of agreement and divergence and gives guidance on when to pick which code.

## Summary table

| Feature | SPIDER | Aragog |
|---------|--------|--------|
| Implementation language | C with PETSc (serial) | Python with optional JAX |
| Primary state variable | Specific entropy $S$ at staggered nodes | Specific entropy $S$ at staggered nodes |
| Extra ODE state at CMB | $dS/dr|_\mathrm{cmb}$ via the `bc.c:76-131` formula | $dS/dr|_\mathrm{cmb}$ via the same formula in `core_bc = energy_balance` |
| Time integrator | SUNDIALS CVODE/BDF via PETSc `TSSUNDIALS` (`main.c:86-95`) | SUNDIALS CVODE/BDF via `scikits.odes`; scipy `Radau`/`BDF` fallbacks |
| Linear solver inside Newton | Dense (`TSSundialsSetUseDense`) | CVODE dense (default) |
| Jacobian source | Finite difference (CVODE-internal) | JAX analytic (default in CHILI), CVODE finite difference (fallback) |
| Phase-boundary smoothing | `get_smoothing` two-branch tanh, `matprop_smooth_width` (`util.c:245-265`) | Identical two-branch tanh, `matprop_smooth_width = 0.01` (default) |
| Pressure-EOS profile | Adams-Williamson (`eos_adamswilliamson.c`) or external mesh file | Adams-Williamson (`eos_method = 1`) or external mesh file (`eos_method = 2`) |
| Thermodynamic EOS tables | P-S lookup tables, bilinear interpolation (`eos_lookup.c`), Wolf-Bower (2018) by default | P-S lookup tables, bilinear interpolation, PALEOS (Attia et al. 2026, in prep.) by default; SPIDER-bundled Wolf-Bower (2018) tables for parity runs |
| Default core BC | `bc.c:76-131` $dS/dr$ ODE | `core_bc = energy_balance` (bit-parity with SPIDER); `quasi_steady` for fast standalone exploration |
| Step-size control during phase change | CVODE adaptive control with user tolerances | CVODE adaptive control plus an Aragog-only SUNDIALS root function `phi_step_cap` that bounds $|\Delta\Phi_\mathrm{global}|$ per call |
| Atmosphere coupling | Built-in atmosphere module: grey-body, Zahnle steam, Jeans/Zahnle escape, volatile speciation (`atmosphere.c`) | Surface BC modes 1, 4, 5 only (grey-body, prescribed flux, prescribed temperature); volatile chemistry handled by JANUS / AGNI / Atmodeller through PROTEUS |
| Mass-coordinate option | `MASS_COORDINATES` flag (`mesh.c`) | `mass_coordinates = true` |
| Output format | JSON timesteps via cJSON (`monitor.c`) | NetCDF via xarray (PROTEUS-side writer in `proteus/interior_energetics/aragog.py`) |
| Restart from checkpoint | `IC_INTERIOR = 2`, `restart.json` | `EntropySolver.set_initial_entropy`, NetCDF reload via the wrapper |

## Numerical differences that matter

### Jacobian source

SPIDER's `main.c:86-95` registers `TSSUNDIALS` with a dense linear solver but does not register an analytic RHS Jacobian.
The Newton iteration inside CVODE therefore builds the Jacobian by finite difference at every refresh, with cost $O(N)$ RHS evaluations per build for an $N$-cell mesh.
Aragog's default CHILI path registers a JAX-traced analytic Jacobian via `jax.jacrev`; the cost of one Jacobian build collapses to a single backward pass over the same JAX RHS, irrespective of $N$.

Where this matters in practice is at the rheological transition.
The mushy-band fluxes $j_\mathrm{grav}$, $F_\mathrm{mix}$, and the smoothing weight $\mathrm{smth}(\phi)$ are highly non-linear in $S$ on a narrow band around the solidus and liquidus.
Finite-difference Jacobians can lose accuracy when the FD step size sits near the smoothing skirt, forcing CVODE to refresh the Jacobian more often or to shrink the time step.
The analytic Jacobian removes that source of step-size pressure but does not change the underlying physics.
See [CVODE and JAX](cvode_jax.md) for the implementation details.

### JAX-traced RHS

The JAX RHS in `aragog.jax` and the numpy RHS in `aragog.solver.entropy_state` evaluate the same algebraic flux assembly.
Floating-point evaluation order differs, so bit-for-bit identity is not the right comparison; in float-64 the two paths agree to within machine epsilon on the diagnostic flux components (see `tests/test_jax_no_phi_vol_source.py` and the per-component flux verification in `Heat transport`, Figure 2).
On the assembled $\partial S/\partial t$ the two paths agree to $\sim 10^{-12}$ relative tolerance for typical CHILI states.

### `phi_step_cap` (Aragog only)

`energy.phi_step_cap` registers a SUNDIALS root function $g(t, y) = \mathrm{cap} - |\Phi_\mathrm{global}(t,y) - \Phi_\mathrm{global}(t_\mathrm{start})|$ that returns CVODE control at the step where the volume-weighted mean melt fraction first changes by `cap`.
SPIDER has no equivalent; in SPIDER the step size at the rheological transition is left entirely to CVODE's tolerance-based adaptive control.
See [Phi step cap](../How-to/phi-step-cap.md) for tuning guidance.

## Where the two codes should agree

In `core_bc = energy_balance` mode, with matching tolerances, matching `matprop_smooth_width`, and the SPIDER-bundled P-S tables, Aragog reproduces SPIDER's CHILI Earth reference.
Specifically:

- The bit-parity helper-formula tests in `tests/test_entropy_pytest.py::TestEnergyBalanceCoreBC::test_energy_balance_rhs_bit_parity_prescribed_inputs` reproduce the SPIDER `bc.c:76-131` evaluation to machine precision on hand-computed inputs.
- The Adams-Williamson density profile in `eos_method = 1` matches the exponential form $\rho(z) = \rho_s \exp(\beta z)$ that SPIDER uses (`eos_adamswilliamson.c:104`) when the same $\beta$ is configured; the `adams_williamson_beta` config knob accepts $\beta$ directly so that the two codes share an unambiguous scale.
- The `get_smoothing` two-branch tanh in `util.c:245-265` is reproduced by Aragog's `phase_smoothing = "tanh"` with the same width parameter.
- Six-isotope radiogenic heating with the Ruedas (2017) cocktail (Geochem. Geophys. Geosyst., 18(9), 3530-3541, doi:10.1002/2017GC006997) produces the same present-day $H_\mathrm{radio}$ in both codes when the same isotope abundances and half-lives are configured.
- Per-component heat fluxes ($F_\mathrm{cond}$, $F_\mathrm{conv}$, $F_\mathrm{grav}$, $F_\mathrm{mix}$) reconstruct the total to floating-point round-off in Aragog (Figure 2 of [Heat transport](heat_transport.md#decomposition-on-a-fully-mushy-state)), which is the same numerical relation SPIDER enforces in `energy.c`.

CHILI integration-level parity between the two codes is the criterion that the production validation track has been driving toward; the relevant harness lives at `output/coupled_parity/` in the PROTEUS tree (referenced inline in `tests/test_entropy_pytest.py:19`).

## Where they intentionally diverge

### Default thermodynamic tables

SPIDER ships with the Wolf-Bower (2018) RTpress liquid EOS as the default mantle EOS.
Aragog's default CHILI path uses PALEOS (Attia, Lichtenberg, Werlen, Bonati, Bower et al. 2026, in prep.) two-phase P-S tables.
PALEOS extends the entropy axis to 0 K and uses a phase-aware blend of independent solid and liquid sub-tables joined through the lever rule on $\phi$; this gives a usable EOS down to fully solid late-stage mantle states where the original Wolf-Bower table extrapolates.
Aragog can still consume Wolf-Bower (2018) tables via `phase_solid` / `phase_liquid` if exact SPIDER reproduction is the goal.

### Atmosphere chemistry and escape

SPIDER's `atmosphere.c` carries its own model for grey-body radiation, Zahnle steam atmosphere, Jeans escape, Zahnle XUV escape, and volatile speciation between melt, solid, and gas phases.
Aragog deliberately does none of this; it implements only the surface boundary modes 1, 4, and 5 (grey-body with optional UTBL correction, prescribed flux, prescribed temperature).
In a PROTEUS run the atmosphere is owned by JANUS, AGNI, or Atmodeller; in standalone use Aragog is run with mode 4 (prescribed flux) or mode 5 (prescribed temperature) as the upper BC.
This is a separation-of-concerns choice, not a missing feature: it lets Aragog be paired with any atmosphere model that PROTEUS configures.

### JAX path

SPIDER's analytic Jacobian was never derived; the C+PETSc stack uses CVODE finite differences.
Aragog's JAX path enables an analytic Jacobian for production CHILI runs without giving up bit-level reproducibility against the numpy reference (see [CVODE and JAX](cvode_jax.md)).

### Output format

SPIDER writes one JSON file per output time via cJSON (`monitor.c`); Aragog writes a single NetCDF per call via the PROTEUS-side wrapper (`proteus.interior_energetics.aragog.write_netcdf`).
The downstream PROTEUS plotting and analysis tooling consumes NetCDF.

## When to use which

**Choose SPIDER for:**

- Established SPIDER workflows and pre-2026 setups whose configuration depends on PETSc options files.
- Direct comparison against published SPIDER results.
- Use cases that need SPIDER's built-in volatile-speciation atmosphere model and that are not run inside PROTEUS.
- Any environment where a C+PETSc build is already in place and a Python toolchain is not available.

**Choose Aragog for:**

- New development inside PROTEUS, where the wrapper, NetCDF output, and PALEOS table chain are the standard path.
- Runs that need the Zalmoxis-derived structure (per-node $g(r)$, $\rho(r)$, $P(r)$ from a self-consistent interior solve) coupled into the mantle solver in a Python-native way.
- Production CHILI runs where the JAX analytic Jacobian and the `phi_step_cap` SUNDIALS root function are required to keep step sizes physical at the rheological transition.
- Standalone use where a pip-installable Python stack is preferable to building SPIDER and PETSc.

For the configuration details that map between the two codes, see [Configuration](../How-to/configuration.md) and [Standalone vs PROTEUS-integrated](../How-to/usage-paths.md).
