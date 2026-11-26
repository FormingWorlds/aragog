# Aragog vs SPIDER

Aragog is a deliberate descendant of SPIDER (Bower et al. (2018)[^cite-bower2018]) and shares its formulation: a 1-D spherically symmetric finite-volume mantle with specific entropy as the prognostic variable, modified-Newton BDF time integration, and SPIDER-style two-branch tanh smoothing across phase boundaries. The two codes diverge in implementation language, in the way the Jacobian is built, in the EOS tables they consume by default, and in how they connect to the outside world.

This page summarises the points of agreement and divergence and gives guidance on when to pick which code.

## Summary table

| Feature | SPIDER | Aragog |
|---------|--------|--------|
| Implementation language | C with PETSc (serial) | Python with optional JAX |
| Primary state variable | Specific entropy $S$ at staggered nodes | Specific entropy $S$ at staggered nodes |
| Extra ODE state at CMB | $dS/dr\vert_\mathrm{cmb}$ via the `bc.c:76-131` formula | $dS/dr\vert_\mathrm{cmb}$ via the same formula in `core_bc = energy_balance`, with $1/(\partial T/\partial S)_P$ of the node in place of $c_p/T$ ([explanation](core_bc.md#the-entropy-slope-at-the-cmb-node)) |
| Time integrator | SUNDIALS CVODE/BDF via PETSc `TSSUNDIALS` (`main.c:86-95`) | SUNDIALS CVODE/BDF via `scikits.odes`; scipy `Radau`/`BDF` fallbacks |
| Linear solver inside Newton | Dense (`TSSundialsSetUseDense`) | CVODE dense (default) |
| Jacobian source | Finite difference (CVODE-internal) | JAX analytic (default), CVODE finite difference (fallback) |
| Phase-boundary smoothing | `get_smoothing` two-branch tanh, `matprop_smooth_width` (`util.c:245-265`) | Identical two-branch tanh, `matprop_smooth_width = 0.01` (default) |
| Pressure-EOS profile | Adams-Williamson (`eos_adamswilliamson.c`) or external mesh file | Adams-Williamson (`eos_method = 1`) or external mesh file (`eos_method = 2`) |
| Thermodynamic EOS tables | P-S lookup tables, bilinear interpolation (`eos_lookup.c`), Wolf & Bower (2018)[^cite-wolfbower2018] by default | P-S lookup tables, bilinear interpolation, PALEOS (Attia et al. (2026)[^cite-attia2026]) by default; Wolf & Bower (2018) tables for parity runs |
| Default core BC | `bc.c:76-131` $dS/dr$ ODE | `core_bc = energy_balance` (SPIDER's formula; equal to SPIDER while the CMB node is in a single phase outside the melt band and its smoothing bands); `quasi_steady` for fast standalone exploration |
| Step-size control during phase change | CVODE adaptive control with user tolerances | CVODE adaptive control plus an Aragog-only SUNDIALS root function `phi_step_cap` that bounds $\lvert\Delta\Phi_\mathrm{global}\rvert$ per call |
| Atmosphere coupling | Built-in atmosphere module: grey-body, Zahnle steam, Jeans/Zahnle escape, volatile speciation (`atmosphere.c`) | Surface BC modes 1, 4, 5 only (grey-body, prescribed flux, prescribed temperature); volatile chemistry handled by JANUS / AGNI / Atmodeller through PROTEUS |
| Mass-coordinate option | `MASS_COORDINATES` flag (`mesh.c`) | `mass_coordinates = true` |
| Output format | JSON timesteps via cJSON (`monitor.c`) | NetCDF via xarray (PROTEUS-side writer in `proteus/interior_energetics/aragog.py`) |
| Restart from checkpoint | `IC_INTERIOR = 2`, `restart.json` | `EntropySolver.set_initial_entropy`, NetCDF reload via the wrapper |
| Solid mantle rheology | Constant solid end-member viscosity | Arrhenius creep, Byerlee yielding, stagnant lid (`[phase_solid.rheology]`) |

## Numerical differences that matter

### Jacobian source

SPIDER's `main.c:86-95` registers `TSSUNDIALS` with a dense linear solver but does not register an analytic RHS Jacobian.
The Newton iteration inside CVODE therefore builds the Jacobian by finite difference at every refresh, with cost $O(N)$ RHS evaluations per build for an $N$-cell mesh.
Aragog's default path registers a JAX-traced analytic Jacobian via `jax.jacrev`; the cost of one Jacobian build collapses to a single backward pass over the same JAX RHS, irrespective of $N$.

Where this matters in practice is at the rheological transition.
The mushy-band fluxes $j_\mathrm{grav}$, $F_\mathrm{mix}$, and the smoothing weight $\mathrm{smth}(\phi)$ are highly non-linear in $S$ on a narrow band around the solidus and liquidus.
Finite-difference Jacobians can lose accuracy when the FD step size sits near the smoothing skirt, forcing CVODE to refresh the Jacobian more often or to shrink the time step.
The analytic Jacobian removes that source of step-size pressure but does not change the underlying physics.
See [CVODE and JAX](cvode_jax.md) for the implementation details.

### JAX-traced RHS

The JAX RHS in `aragog.jax` and the numpy RHS in `aragog.solver.entropy_state` evaluate the same algebraic flux assembly.
Floating-point evaluation order differs, so bit-for-bit identity is not the right comparison; in float-64 the two paths agree to within machine epsilon on the diagnostic flux components (see `tests/test_jax_no_phi_vol_source.py` and the per-component flux verification in `Heat transport`, Figure 2).
On the assembled $\partial S/\partial t$ the two paths agree to $\sim 10^{-12}$ relative tolerance for typical magma-ocean states.

### `phi_step_cap` (Aragog only)

`energy.phi_step_cap` registers a SUNDIALS root function $g(t, y) = \mathrm{cap} - |\Phi_\mathrm{global}(t,y) - \Phi_\mathrm{global}(t_\mathrm{start})|$ that returns CVODE control at the step where the volume-weighted mean melt fraction first changes by `cap`.
SPIDER has no equivalent; in SPIDER the step size at the rheological transition is left entirely to CVODE's tolerance-based adaptive control.
See [Phi step cap](../How-to/phi-step-cap.md) for tuning guidance.

## Where the two codes should agree

In `core_bc = energy_balance` mode, with matching tolerances, matching `matprop_smooth_width`, `convective_threshold = 0`, and the SPIDER-bundled P-S tables, Aragog reproduces SPIDER's Earth reference while the CMB node stays outside the melt band. Inside the band the closure factors differ ([the entropy slope at the CMB node](core_bc.md#the-entropy-slope-at-the-cmb-node)).
Specifically:

- The bit-parity helper-formula tests in `tests/test_entropy_pytest.py::TestEnergyBalanceCoreBC::test_energy_balance_rhs_bit_parity_prescribed_inputs` reproduce the SPIDER `bc.c:76-131` evaluation to machine precision on hand-computed inputs. The test passes the heat capacity directly, so it checks the formula, not the factor the solver passes in the mush.
- The Adams-Williamson density profile in `eos_method = 1` matches the exponential form $\rho(z) = \rho_s \exp(\beta z)$ that SPIDER uses (`eos_adamswilliamson.c:104`) when the same $\beta$ is configured; the `adams_williamson_beta` config knob accepts $\beta$ directly so that the two codes share an unambiguous scale.
- The `get_smoothing` two-branch tanh in `util.c:245-265` is reproduced by Aragog's `phase_smoothing = "tanh"` with the same width parameter.
- Six-isotope radiogenic heating with the Ruedas (2017)[^cite-ruedas2017] cocktail produces the same present-day $H_\mathrm{radio}$ in both codes when the same isotope abundances and half-lives are configured.
- Per-component heat fluxes ($F_\mathrm{cond}$, $F_\mathrm{conv}$, $F_\mathrm{grav}$, $F_\mathrm{mix}$) reconstruct the total to floating-point round-off in Aragog (Figure 2 of [Heat transport](heat_transport.md#decomposition-on-a-fully-mushy-state)), which is the same numerical relation SPIDER enforces in `energy.c`.

Integration-level parity between the two codes is the criterion that the production validation track drives toward; the relevant harness lives at `output/coupled_parity/` in the PROTEUS tree.

## Where they intentionally diverge

### Convective threshold

SPIDER and the Aragog mixing-length drive both switch convection on at $\partial S/\partial r = 0$.
Aragog adds 2 things that SPIDER does not have:

- it can drive an unstable node by at least a small gradient $D$ (`energy.convective_threshold`, default 0);
- its `kappah_floor` acts up to $\partial S/\partial r = D$.

Apart from the floor, the 2 codes therefore differ at nodes with $-D < \partial S/\partial r < 0$.
They also differ at the CMB node when node 1 is in that band, because the CMB node takes the mixing-length eddy diffusivity of node 1.
With `kappah_floor > 0` the CMB node also takes its own floor where that is larger ([Heat transport](heat_transport.md#eddy-diffusivity-at-the-cmb-node)).
The default `convective_threshold = 0` gives the SPIDER switch; see [Heat transport](heat_transport.md#convection-mixing-length-theory).

### CMB closure in the melt band

SPIDER turns the core cooling rate into the entropy rate of the CMB node with $c_p/T$ of the node. Aragog divides by the slope $(\partial T/\partial S)_P$ of its own $T(P, S)$, so that the core loses the heat that crosses the CMB. While the node is in a single phase outside the melt band and its smoothing bands, that slope is $T/c_p$ of the tables, so the 2 factors are the same; the miss that remains there comes from the difference between $T/c_p$ and the slope of the tabulated $T$ ([limits of the slope closure](core_bc.md#limits-of-the-slope-closure)). Inside the band, $c_p/T$ makes the node cool faster or slower than the core: along the calls of [the CMB closure ledger across the melt band](../Validation/core.md#16-the-cmb-closure-ledger-across-the-melt-band) it would miss by up to 20 % of the CMB heat at the starts in the mush and by 41 % at the start in the smoothing band below the solidus ($g = -0.02$). See [the entropy slope at the CMB node](core_bc.md#the-entropy-slope-at-the-cmb-node).

### Default thermodynamic tables

SPIDER ships with the Wolf & Bower (2018) RTpress liquid EOS as the default mantle EOS.
Aragog's default path uses PALEOS (Attia et al. (2026)) two-phase P-S tables.
PALEOS extends the entropy axis to 0 K and uses a phase-aware blend of independent solid and liquid sub-tables joined through the lever rule on $\phi$; this gives a usable EOS down to fully solid late-stage mantle states where the original Wolf-Bower table extrapolates.
Aragog can still consume Wolf & Bower (2018) tables via `phase_solid` / `phase_liquid` if exact SPIDER reproduction is the goal.

### Atmosphere chemistry and escape

SPIDER's `atmosphere.c` carries its own model for grey-body radiation, Zahnle steam atmosphere, Jeans escape, Zahnle XUV escape, and volatile speciation between melt, solid, and gas phases.
Aragog deliberately does none of this; it implements only the surface boundary modes 1, 4, and 5 (grey-body with optional UTBL correction, prescribed flux, prescribed temperature; see [Surface boundary conditions](surface_bc.md)).
In a PROTEUS run the atmospheric climate is owned by JANUS or AGNI, and the volatile speciation / outgassing is owned by Atmodeller or CALLIOPE; in standalone use Aragog is run with mode 4 (prescribed flux) or mode 5 (prescribed temperature) as the upper BC.
This is a separation-of-concerns choice, not a missing feature: it lets Aragog be paired with whichever climate and outgassing modules PROTEUS configures.

### JAX path

SPIDER's analytic Jacobian was never derived; the C+PETSc stack uses CVODE finite differences.
Aragog's JAX path enables an analytic Jacobian for production runs without giving up bit-level reproducibility against the numpy reference (see [CVODE and JAX](cvode_jax.md)).

### Output format

SPIDER writes one JSON file per output time via cJSON (`monitor.c`); Aragog writes a single NetCDF per call via the PROTEUS-side wrapper (`proteus.interior_energetics.aragog.write_netcdf`).
The downstream PROTEUS plotting and analysis tooling consumes NetCDF.

### Solid mantle rheology and stagnant lids

SPIDER models mantle viscosity using a constant solid end-member viscosity. Aragog provides an optional solid-state convection module (`[phase_solid.rheology]`) with Arrhenius temperature-dependent creep, Byerlee brittle yielding, stagnant-lid boundary layer stress closures, and viscosity-contrast mixing-length slope scaling. See [Solid-state convection](solid_state_convection.md) for the formulation and benchmark verification.

For the configuration details that map between the two codes, see [Configuration](../How-to/configuration.md) and [Standalone vs PROTEUS-integrated](../How-to/usage-paths.md).

[^cite-attia2026]: M. Attia, T. Lichtenberg, E. Jungová, M. Sastre, *[PALEOS: Multiphase equations of state and mass-radius relations for exoplanet interiors](https://doi.org/10.1051/0004-6361/202660790)*, Astronomy & Astrophysics, 713, A225, 2026. [arXiv:2605.03741](https://arxiv.org/abs/2605.03741).
[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49–62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
[^cite-ruedas2017]: T. Ruedas, *[Radioactive heat production of six geologically important nuclides](https://doi.org/10.1002/2017GC006997)*, Geochemistry, Geophysics, Geosystems, 18, 3530–3541, 2017. [SciX](https://scixplorer.org/abs/2017GGG....18.3530R/abstract).
[^cite-wolfbower2018]: A. S. Wolf, D. J. Bower, *[An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO3 melt](https://doi.org/10.1016/j.pepi.2018.02.004)*, Physics of the Earth and Planetary Interiors, 278, 59–74, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..278...59W/abstract).
