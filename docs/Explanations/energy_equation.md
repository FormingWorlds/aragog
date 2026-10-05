# Energy equation

This page presents the governing equation, its semi-discrete finite-volume form, and the time-integration controls.

## Entropy balance in integral form

Aragog evolves the specific entropy $S(r,t)$ in a spherical mantle shell between the core-mantle boundary at $r = r_\mathrm{cmb}$ and the surface at $r = r_\mathrm{top}$. For a control volume $V$ with boundary $\partial V$,

$$
\int_V \rho T \left.\frac{\partial S}{\partial t}\right|_{\xi}\, dV
= -\int_{\partial V} \mathbf{F}\cdot\mathbf{n}\, dS
+ \int_V \rho H\, dV,
$$

with $\mathbf{F}$ the radial heat flux $[\mathrm{W\,m^{-2}}]$ and $H$ the specific internal heating rate $[\mathrm{W\,kg^{-1}}]$. The capacitance multiplying $\partial S/\partial t$ is $\rho T$ (not $\rho c_p$): the thermodynamic identity $T\,dS = c_p\,dT - (\alpha T/\rho)\,dP$ is enforced by the EOS lookups rather than by an explicit $c_p$ on the time derivative. The time derivative is taken at constant mass coordinate $\xi$.

Each finite-volume cell coincides with a material volume of constant mass; species fluxes (melt vs solid) may exist internally but their net mass flux through cell faces is zero.

## Semi-discrete finite-volume form

Discretising the mantle into $N$ shells gives, for each staggered node $i$,

$$
(\rho T V)_i \left.\frac{\partial S}{\partial t}\right|_i
= -F_{i+1/2}\,A_{i+1/2} + F_{i-1/2}\,A_{i-1/2} + \rho_i H_i V_i,
$$

with $A_{i\pm 1/2} = 4\pi r_{i\pm 1/2}^2$ and $V_i = \tfrac{4}{3}\pi(r_{i+1/2}^3 - r_{i-1/2}^3)$. Entropy lives at the $N$ staggered nodes, fluxes at the $N+1$ basic nodes, and $\rho$, $T$ are read from the EOS at the staggered pressures.

## Staggered mesh

- **Basic nodes** ($N+1$ radial positions, cell faces): hold radial fluxes, pressure, and the entropy gradient $\partial S/\partial r$.
- **Staggered nodes** ($N$ radial positions, cell centres): hold the prognostic entropy $S$ as well as the diagnostic temperature, density, melt fraction, viscosity, and capacitance.

The entropy gradient at basic nodes is the SPIDER-parity centred difference of the staggered $S$ values in uniform $\xi$-space, then chain-ruled through $d\xi/dr$. Boundary values copy from the nearest interior node; in `energy_balance` mode the CMB boundary value is overridden by the entropy-gradient state-vector entry on every RHS evaluation.

## Mass coordinate transform

The radial coordinate is uniform in radius (`mass_coordinates = false`) or uniform in mass coordinate (`mass_coordinates = true`). The mass coordinate is

$$
\xi(r) = \left( 3\int_{r_\mathrm{cmb}}^{r} \frac{\rho^*(r')}{\rho^*_\mathrm{planet}}\,r'^2\,dr' + \xi_\mathrm{cmb}^3 \right)^{1/3},
$$

with $\rho^*(r)$ the configured pressure-density relation. In mass-coordinate mode the basic-node spatial radii are recovered by a Newton solve at construction so that $\xi$ is uniformly spaced. Gradients convert via

$$
\frac{\partial \psi}{\partial r}
= \frac{\rho^*(r)}{\rho^*_\mathrm{planet}}\left(\frac{r}{\xi}\right)^2 \frac{\partial \psi}{\partial \xi}.
$$

When `mass_coordinates = false`, $\xi \equiv r$ and the chain rule reduces to the identity.

## Time integration

The semi-discrete ODE system $\mathbf{y}'(t) = \mathbf{f}(t, \mathbf{y})$ is advanced using an implicit stiff integrator. The selector is `solver_method`:

| Value | Integrator | When to use |
|-------|------------|-------------|
| `"cvode"` (default) | SUNDIALS CVODE via `scikits.odes` | Production-grade stiff integration; same solver SPIDER uses |
| `"radau"` | scipy `Radau` | Standalone tests when `scikits.odes` is not installed |
| `"bdf"` | scipy `BDF` | Alternative scipy path |

CVODE provides modified-Newton with cached Jacobian factorisation and adaptive step control that handles phase-transition discontinuities cleanly. scipy's BDF and Radau can collapse the step size at the crystallisation front when their Newton iteration fails to converge. When CVODE is requested but `scikits.odes` is not importable, the solver emits a warning at solve time and silently falls back to scipy Radau.

The default also enables the JAX-derived analytic Jacobian (`use_jax_jacobian = true`), built by tracing the pure-functional flux computation in `aragog.jax.phase` with `jax.jacrev`. This replaces the $O(N)$ finite-difference RHS evaluations per Jacobian build with one JAX-traced backward pass. The analytic Jacobian is used only when CVODE is the active solver and a JAX factory has been registered (the PROTEUS wrapper does this automatically); otherwise the solver silently falls back to CVODE's finite-difference Jacobian.

### Right-hand side

The right-hand side function `EntropySolver.dSdt(time, state_vec)` computes:

1. Phase, density, $T$, $c_p$, $\alpha$, $k$, $\partial T/\partial P|_S$, and the smoothing weight $\mathrm{smth}(\phi)$ at every node from the EOS lookup at $(P, S)$.
2. The entropy gradient $\partial S/\partial r$ at basic nodes (centred difference in uniform $\xi$, plus the optional `energy_balance` override).
3. The four flux contributions $F_\mathrm{cond}$, $F_\mathrm{conv}$, $F_\mathrm{grav}$, $F_\mathrm{mix}$ (see [Heat transport](heat_transport.md)).
4. The internal heating $H = H_\mathrm{radio} + H_\mathrm{tidal}$ at staggered nodes.
5. The flux divergence $\partial S/\partial t = (\rho T V)^{-1}[-(F\,A)_{i+1/2} + (F\,A)_{i-1/2}] + H/T$, returned in $\mathrm{J\,kg^{-1}\,K^{-1}\,yr^{-1}}$.
6. For extended-state modes, the additional ODE for the boundary state (the SPIDER `bc.c:76-131` formula in `energy_balance`, or the surface-entropy ODE in `gradient`).

### Nondimensionalisation layer

Inside the solver, the state vector and time are nondimensionalised before being passed to the integrator:

$$
\hat{S} = S / S_\mathrm{ref}, \qquad \hat{t} = t / t_\mathrm{ref},
$$

with $S_\mathrm{ref} = 2993.025\ \mathrm{J\,kg^{-1}\,K^{-1}}$ and $t_\mathrm{ref} = 10^5/(3.155\times 10^7)\ \mathrm{yr}$. This keeps the BDF tolerance control numerically well-behaved across the wide entropy and time scales encountered during interior cooling. Physical units are restored when the solution is written back to `SolverOutput`.

### Phase-aware step-size control

Whenever any cell sits within `phase_boundary_entropy_margin` (default $200\ \mathrm{J\,kg^{-1}\,K^{-1}}$) of a phase boundary, or inside the mushy band, step-size tightening activates; a `rate` call on CVODE also activates it within the stiff-zone half-width $\delta$ defined below. The step cap is configured via `phase_boundary_cap`:

- `fixed`: `max_step` is reduced to one year.
- `rate` (default, also when the key is unset): the call is integrated as event-driven CVODE segments. At the start of each segment, `max_step` is set to $0.1 \min_i t_{c,i}$, clipped to $[1, 100]\ \mathrm{yr}$, where $t_{c,i} = d_i / r_{\mathrm{used},i}$ with rate $r_{\mathrm{used},i} = \max(|\dot{S}_i|, r_{\mathrm{floor}})$ and mass-weighted rate floor $r_{\mathrm{floor}}$. For cells outside the stiff zone, $d_i$ is the distance to the boundary ahead; for cells inside the stiff zone ($\delta = \max(3w \max_j(S_{\mathrm{liq},j}-S_{\mathrm{sol},j}), 10\ \mathrm{J\,kg^{-1}\,K^{-1}})$, with $w = \mathtt{matprop\_smooth\_width}$) or moving slower than the rate floor, $d_i$ is the distance to the nearer boundary on either side. A root function ends a segment when an entry that started the segment outside the stiff zone enters it, when all entries have left it (hysteresis $2\delta$), or when an entry has moved half its start distance to the nearer boundary, and at least $\delta/2$; the per-call step caps end the call. The segments of a `rate` call on CVODE arm when a cell is within the larger of `phase_boundary_entropy_margin` and $\delta$; the per-call step caps, `fixed`, the gradient core and the scipy integrators use `phase_boundary_entropy_margin` only. The table below compares `rate` and `fixed` against `fixed` at $10^{-10}$; `tools/verification/run_phase_boundary_cap_accuracy.py` defines the cases in full and computes the numbers in the table. At `rtol` $= 10^{-8}$, the default, `rate` is within $1.3 \times 10^{-4}$ K of that reference in every case; above $10^{-7}$ the rate cap logs one warning. Separately, a CVODE run logs one warning above `rtol` `1e-8`, because a loose `rtol` can lock CVODE at the switch $dS/dr = 0$ (the convective mask and the $\kappa_h$ floor). The counts below come from a probe variant of the energy_balance isentropic start of `tests/test_phase_boundary_cap.py` (24 nodes, 200 yr, fixed cap unless stated) that draws the initial entropy uniformly between 8000 and 8400 J/kg/K for each run and adds a relative noise of $10^{-15}$; the probe script is not in the repository, and the runs were on macOS arm64. A run counts as locked when it has not reached 200 yr after 600 s of wall time: every run was screened at 60 s, and each run that missed that was run again alone to 600 s, by which time a locked run had used about $1.7 \times 10^5$ RHS evaluations. With `rtol` = `atol` = `1e-6`, CVODE locked in 12 of 1133 runs, in 4 of 474 on the quasi_steady core and in 2 of 286 with the rate cap, where 2 more runs finished only after 233 s and 346 s. With both at `1e-5`, all 80 runs were run to 600 s and none locked (too few runs to show it safe; the slowest finished after 100.6 s and $6.5 \times 10^4$ RHS evaluations), but the median run needed about $1.6 \times 10^4$ RHS evaluations, against about $1.3 \times 10^3$ at `1e-8`. With both at `1e-8`, CVODE locked in none of 1026, 627 and 1224 runs for three setups (fixed and rate cap on the energy_balance core, fixed cap on quasi_steady). On CI the AVX-512 numpy path made the test's own case, at its single fixed initial entropy, stall at `1e-6`, judged by the CI test timeout rather than by the 600 s criterion. `rate` needs CVODE; the scipy integrators use 1 yr.

| Case (test EOS, 24 nodes unless noted) | Call | CVODE steps, `fixed` / `rate` at $10^{-8}$ | max $\lvert\Delta T\rvert$, `fixed` $10^{-8}$ | `rate` $10^{-8}$ | `rate` $10^{-10}$ |
|---|---|---|---|---|---|
| `quasi_steady`, mushy start, $F = 10^4$ W m$^{-2}$, $w = 0.01$ | 1000 yr | 1260 / 308 | $1.2 \times 10^{-7}$ K | $6.9 \times 10^{-6}$ K | 0 |
| `quasi_steady`, 230 J kg$^{-1}$ K$^{-1}$ above the liquidus, $F = 10^5$ W m$^{-2}$, $w = 0.01$ | $10^4$ yr | 2040 / 4238 | $1.9 \times 10^{-4}$ K | $1.3 \times 10^{-4}$ K | $2.7 \times 10^{-6}$ K |
| `quasi_steady`, mushy start, $F = 10^4$ W m$^{-2}$, $w = 0$ | 2000 yr | 2948 / 115 | $1.0 \times 10^{-7}$ K | $1.0 \times 10^{-5}$ K | 0 |
| `energy_balance`, mushy start, $F = 10^4$ W m$^{-2}$, $w = 0.01$, 16 nodes | 3000 yr | 3048 / 83 | $2.5 \times 10^{-8}$ K | $2.8 \times 10^{-5}$ K | $1.6 \times 10^{-7}$ K |

$\Delta T$ is the largest cell difference at the call end against `fixed` at $10^{-10}$; $F$ is the prescribed surface flux. In the second case `fixed` does not arm (the cells start beyond the 200 J kg$^{-1}$ K$^{-1}$ margin) and takes fewer steps.

The gradient core and the scipy integrators use 1 yr in both modes; `rate` then logs one line per solver, at INFO when it is the default and at WARNING when it is set. Once the mantle is fully solid (mean $\phi < 0.01$), the absolute tolerance is relaxed by a factor of ten so the integrator does not stall on a stiff residual.

### Retry ladder hooks

`EntropySolver.set_initial_dSdr_cmb()` and `EntropySolver.get_current_dSdr_cmb()` give the PROTEUS wrapper a way to checkpoint and restore the CMB boundary state across retry attempts when the integrator returns `status = -1`. The `_atol_sf` attribute carries a per-attempt tolerance scaling that the wrapper increments on retry.

## Boundary conditions

The five outer BC modes and three inner BC modes are summarised in [Model overview](model.md#boundary-conditions). At the surface, the grey-body mode ($q_\mathrm{top} = \varepsilon\sigma(T_\mathrm{top}^4 - T_\mathrm{eqm}^4)$) optionally adds an upper-thermal-boundary-layer correction via Cardano's formula when `param_utbl = true`. At the CMB, the four `core_bc` formulations differ in whether the bottom-cell heat flux is partitioned by heat capacity (`quasi_steady`), driven by an evolved boundary-gradient state (`energy_balance`), reconstructed from a gradient state field (`gradient`), or set by one-sided Fourier conduction across the bottom half-cell (`bower2018`).

## Termination

The solver returns when the integrator reaches `end_time` from the configuration or when the integrator status flag goes negative (typically `-1`, indicating a step-size collapse). In coupled PROTEUS runs, the wrapper's outer loop also terminates when the global melt fraction drops below $\Phi_\mathrm{g} < 0.05$ or when the system reaches a radiative quasi-steady state.
