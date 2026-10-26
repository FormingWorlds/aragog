# First-principles verification

## What this page checks

This page checks the numerical solvers and physical conservation laws in Aragog.
Aragog solves the coupled equations of energy conservation, hydrostatic pressure, and mantle convection to simulate magma-ocean cooling and solid-mantle thermal evolution.
Validation here means running the solver on cases where another source provides the known answer.
The test then checks that the code matches that answer inside a stated numerical tolerance.
Known answers come from analytical limits, physical conservation laws, published benchmark studies, or independent second code paths.
If these numerical methods are wrong, the mantle calculates incorrect temperatures, melt fractions, and boundary heat fluxes.
Those errors then propagate directly into atmospheric outgassing, planet contraction, and orbital evolution in PROTEUS.
Each case below states why the check exists, what it tests, how we know the right answer, and why it matters.
The box "Test details" at the end of each case gives the reference, the tolerance, the test id, and the test command.

## Test categories

The tests are organised by the layer they cover:

### EOS unit tests (`tests/test_entropy_pytest.py`)

Exercise `EntropyEOS` and `EntropyPhaseEvaluator`:

- Property lookups (temperature, density, heat capacity, thermal expansivity, isentropic temperature gradient) at on-grid and off-grid $(P, S)$ points.
- Solidus and liquidus entropy lookups against the reference `solidus_P-S.dat` and `liquidus_P-S.dat` files.
- Smooth-clipped melt fraction $\phi(P, S)$: monotone in $S$, clamped to $[0, 1]$, $C^\infty$ through the solidus and liquidus crossings.
- The two-stage SPIDER-parity blend (cubic-Hermite or tanh smoothing followed by `combine_matprop`) for density, heat capacity, and viscosity.
- The latent-blend $c_p$ method against the linear blend.

![EOS bilinear-interpolation Jacobian](../figures/vv/fig_03_eos_bilinear_jacobian.png)

**Figure 1.** JAX-traced bilinear interpolation of the EOS table reproduces the numpy reference to float-64 machine precision and supplies finite, non-zero derivatives via `jax.jacrev` at on- and off-grid $(P, S)$ points. The agreement is the prerequisite for the analytic Jacobian path used by CVODE.

### Phase evaluator tests

Test the SPIDER-parity behaviour of `EntropyPhaseEvaluator`:

- Phase identification at $(P, S)$ in pure solid, pure liquid, and mushy regimes.
- Viscosity tanh blend through the rheological transition $\phi_\mathrm{rheo}$.
- Gravitational-separation velocity in viscous and inviscid regimes.
- Pressure-dependent latent heat $L(P) = T_\mathrm{liq}(P) [S_\mathrm{liq}(P) - S_\mathrm{sol}(P)]$.

### Solver integration tests

Drive `EntropySolver` through short integrations and check the resulting profiles:

- Standalone grey-body cooling: a fully-molten initial state cools toward the equilibrium temperature without status flag failures.
- The `set_initial_entropy`, `set_initial_dSdr_cmb`, and `set_initial_core_temperature` paths produce the expected state-vector layouts for each of the five `core_bc` modes.
- The retry-ladder hooks (`get_current_dSdr_cmb`, `_atol_sf`) round-trip cleanly.

### Conservation tests (`tests/test_entropy_verification.py`)

The conservation tests are the strongest class of verification because they hold under any valid run, not just specific test scenarios:

- **Energy conservation.** $E_\mathrm{th}$ change over a closed integration matches the time-integrated surface flux to within solver tolerance.
- **Mass conservation.** Mantle mass remains constant through phase changes.
- **Grey-body cooling.** Surface flux follows $\varepsilon\sigma(T_\mathrm{top}^4 - T_\mathrm{eqm}^4)$ within tolerance once the radiative regime is established.

![Mass-coordinate Jacobian sanity](../figures/vv/fig_07_mass_coord_jacobian.png)

**Figure 2.** Mass-coordinate mesh Jacobian. The Newton solve that maps from mass coordinate to spatial radius (used when `mass_coordinates = true`) is monotone, smooth, and consistent with the analytic Adams-Williamson density profile. Departures here would propagate to mis-aligned cell volumes and a spurious mass-conservation residual in the integrator.

![Radioactive decay](../figures/vv/fig_06_radio_decay.png)

**Figure 3.** Radioactive heating $H_\mathrm{radio}(t)$ for the bundled isotope cocktail (Ruedas 2017): present-day specific powers ($^{40}\mathrm{K}$, $^{232}\mathrm{Th}$, $^{235}\mathrm{U}$, $^{238}\mathrm{U}$) decayed back to Solar System formation. Used as the source-term verification for the closed-mantle energy balance: the cumulative `step_dE_Q_radio_J` integral must match the time integral of $\rho \cdot H_\mathrm{radio}$ over each call to within solver tolerance.

### JAX path tests (`tests/test_jax_entropy.py`)

Verify that the JAX-backed `EntropyEOS_JAX` and `compute_fluxes` agree with the numpy path:

- EOS parity: `EntropyEOS_JAX` returns the same temperature, density, heat capacity, thermal expansivity, $\partial T/\partial P|_S$, and conductivity as `EntropyEOS` on the same $(P, S)$ inputs.
- Phase evaluator parity: the JAX two-stage blend matches `EntropyPhaseEvaluator` to within floating-point tolerance.
- JIT and `vmap` compatibility: the JAX flux assembly is differentiable in full with `jax.grad`/`jax.jacrev` and survives JIT compilation without recompilation hot-spots.
- Boundary-copy tests: `dSdr` boundary values copy from the nearest interior node (SPIDER `ic.c:450` parity) instead of being extrapolated.
- Solver parity: integrating the same problem with the JAX-RHS-driven CVODE path and the numpy RHS produces matching trajectories.

### Numpy ↔ JAX RHS parity at representative magma-ocean states

![Numpy ↔ JAX RHS parity](../figures/vv/fig_01_rhs_parity.png)

**Figure 4.** Numpy $\leftrightarrow$ JAX RHS parity on an 80-cell Earth mesh at three representative magma-ocean states. *Top row, panels (a-c)*: per-cell **absolute** residual $|\dot S_\text{numpy} - \dot S_\text{jax}|$ in J/kg/K/yr; the dotted line is $\epsilon_\mathrm{mach}\,\max|\dot S|$ (the floating-point round-off floor expected on a single FP operation). *Bottom row, panels (d-f)*: per-cell **relative** residual $|\dot S_\text{numpy} - \dot S_\text{jax}|/|\dot S_\text{numpy}|$; the dotted line is float-64 machine epsilon. The three states are (a/d) initial condition (full magma ocean, $S \approx 3900$ J/kg/K, dimensionally uniform), (b/e) mid-solidification ($S = 3300 \to 3700$ J/kg/K from CMB to surface), (c/f) near-solid ($S = 3000 \to 3300$ J/kg/K).

Per-panel residuals:

- **Mid-solidification (b, e) and near-solid (c, f).** Interior cells sit at the float-64 round-off floor. Maximum absolute residual $\sim 3\times 10^{-7}$ J/kg/K/yr, dominated by a single spike at the surface boundary cell where the BC assembly is re-derived independently between numpy and JAX. Median relative error $\le 3\times 10^{-5}$.
- **Uniform-entropy IC (a, d).** Absolute residual $\sim 1$ J/kg/K/yr. Both paths run EOS lookup, lever-rule, and table interpolation on a uniform input; the arithmetic differs at the interpolation level. The relative error inflates because $|\dot S|$ itself is small in equilibrium.

Numpy and JAX paths are independent re-derivations, not a bit-exact port.

![UTBL Cardano-formula consistency](../figures/vv/fig_05_utbl_cardano.png)

**Figure 5.** Surface-temperature correction from the upper-thermal-boundary-layer parameterisation (`solver/boundary.py::_utbl_tsurf`). The Cardano-formula solution to $b x^3 + x - T_\mathrm{int} = 0$ recovers $T_\mathrm{surf} < T_\mathrm{int}$ for the entire physical range and matches the cubic-equation residual to floating-point tolerance. Verifies the analytic surface boundary condition used when `param_utbl = true`.

### Mesh tests (`tests/test_jax_mesh_gravity_fallback.py`)

Verify that when the external mesh file (`eos_method = 2`) supplies an `eos_gravity` column, the JAX path interpolates the per-node gravity profile onto the basic grid rather than broadcasting a single scalar.

### Numpy <-> JAX RHS parity

**Why this check exists.** Aragog implements the right-hand side of the energy equation in both NumPy and JAX. The NumPy right-hand side is the reference implementation, and CVODE evaluates the compiled JAX right-hand side to integrate planet cooling. If the two paths differ in their rheology formulas or flux assembly, CVODE will integrate a different physical system than the reference equations. This test catches discrepancies in homologous temperature parameters on active-solid states (`abe_solid_kappah_floor_0`), where the JAX homologous perturbation probe produces relative differences in $[8.057 \times 10^{-3}, 1.0582 \times 10^{-2}]$ and fails the $10^{-6}$ tolerance by a factor of 8057 to 10582 ($> 8000\times$). It catches missing plastic yielding in JAX on yielded states (`abe_solid_yield_active`), where disabling plastic yielding produces relative differences in $[2.331 \times 10^{-4}, 7.270 \times 10^{-4}]$ and fails tolerance by a factor of 233 to 727 ($> 200\times$). The default state set (`abe_solid_default`) does not show either fault. The test also catches silent inactivity of solid rheology, caught by the activity assertion (`assert float(np.min(rel_on_off)) >= 100.0 * 1.0e-6`), which confirms physical right-hand side shifts of $[8.595 \times 10^{-3}, 1.0677 \times 10^{-2}]$ on `abe_solid_kappah_floor_0`.

**What it tests.** With solid-state rheology enabled (`[phase_solid.rheology] enabled = true`), the right-hand side evaluates the rate of specific entropy change $\dot{S} = \partial S/\partial t$ in both `EntropyState.update` (NumPy) and `aragog.jax.phase.compute_fluxes` (JAX). The test evaluates 100 cooling states on three state sets:

1. Default configuration `cfg/abe_solid.toml`: initial entropy $S_\text{init} = 2900\text{ J/(kg K)}$, eddy-diffusivity floor $\kappa_{h,\text{floor}} = 10.0\text{ m}^2/\text{s}$, and solid reference viscosity $\eta_\text{solid} = 1.0\times 10^{21}\text{ Pa s}$. Solid rheology alters basic-node viscosity with solid blend weight $1 - w(\phi) > 0$ at all 100 basic nodes, with effective viscosity $\eta_\text{eff} = \eta_\text{diff}$ across all states ($n_\text{states}(\eta_\text{eff} \ne \eta_\text{diff}) = 0$). Convective floor damping alters the right-hand side relative to rheology-off by $[8.101 \times 10^{-5}, 1.268 \times 10^{-3}]$.
2. Active-solid Arrhenius variant `abe_solid_kappah_floor_0`: $S_\text{init} = 2500\text{ J/(kg K)}$, bottom entropy $S_\text{bot} \in [1850, 2500]\text{ J/(kg K)}$, $\kappa_{h,\text{floor}} = 0.0\text{ m}^2/\text{s}$, $\eta_\text{solid} = 1.0\times 10^{18}\text{ Pa s}$, cohesion $c = 1.0\times 10^7\text{ Pa}$, friction coefficient $\mu = 0.1$, and maximum yield stress $\tau_{y,\text{max}} = 1.0\times 10^8\text{ Pa}$. Homologous Arrhenius strengthening ($\eta_\text{diff} \ge 10^{22}\text{ Pa s}$) keeps Pass 1 convective flux below conductive flux through the column ($f_\text{conv} < 0.5$), so `has_lid = False` and $n_\text{nodes}(\eta_\text{eff} \ne \eta_\text{diff}) = 0$ in all 100 states while Arrhenius strengthening changes the RHS by $[8.595 \times 10^{-3}, 1.0677 \times 10^{-2}]$ (and $0.99467$ along the integrated 100-yr trajectory on 60 basic nodes).
3. Yield-active state set `abe_solid_yield_active`: warm sub-solidus interior $S_\text{init} = 2150\text{ J/(kg K)}$, $S_\text{bot} \in [1950, 2150]\text{ J/(kg K)}$, $\Delta S \in [1620, 1820]\text{ J/(kg K)}$, $\kappa_{h,\text{floor}} = 0.0\text{ m}^2/\text{s}$, $\eta_\text{solid} = 3.0\times 10^{17}\text{ Pa s}$, $c = 150.0\text{ Pa}$, $\mu = 0.0$, and $\tau_{y,\text{max}} = 1.0\times 10^8\text{ Pa}$. Pass 1 interior convection exceeds conduction ($f_\text{conv} > 0.5$) beneath a cold conductive upper boundary layer (`has_lid = True` in all 100 states), generating driving stress $\tau_{d,\text{lid}} \gg \tau_{y,\text{lid}} = 150\text{ Pa}$ so plastic yielding reduces $\eta_\text{eff} < \eta_\text{diff}$ in 9 to 13 nodes in all 100 states. Yielding shifts the RHS relative to yield-off by $[2.331 \times 10^{-4}, 7.270 \times 10^{-4}]$ (median $3.918 \times 10^{-4}$).

When single-phase thermal expansivity $\alpha$ is harmonised between both paths (`identity_alpha = True`), NumPy and JAX right-hand sides agree to floating-point round-off: maximum relative difference $\le 4.435 \times 10^{-16}$ (median $1.373 \times 10^{-16}$, interior heat-flux relative difference $\le 5.076 \times 10^{-14}$) on `abe_solid_kappah_floor_0`, $\le 6.105 \times 10^{-16}$ (median $2.433 \times 10^{-16}$, interior heat-flux relative difference $\le 9.704 \times 10^{-14}$) on `abe_solid_yield_active`, and $\le 4.998 \times 10^{-15}$ (median $2.127 \times 10^{-15}$, interior heat-flux relative difference $\le 1.669 \times 10^{-14}$) on `abe_solid_default`. With unharmonised single-phase $\alpha$ where `EntropyPhaseEvaluator` interpolates the table directly and `EntropyEOS_JAX` evaluates the thermodynamic relation, differences remain within $10^{-6}$ in all three state sets: $\le 2.038 \times 10^{-14}$ (median $1.631 \times 10^{-16}$, interior heat-flux relative difference $\le 6.402 \times 10^{-7}$) on `abe_solid_kappah_floor_0`, $\le 1.944 \times 10^{-7}$ (median $5.950 \times 10^{-8}$, interior heat-flux relative difference $\le 4.628 \times 10^{-5}$) on `abe_solid_yield_active`, and $\le 4.998 \times 10^{-15}$ (median $2.127 \times 10^{-15}$, interior heat-flux relative difference $\le 1.669 \times 10^{-14}$) on `abe_solid_default`. Live evaluation of a single profile verifies parity to within $2.5 \times 10^{-13}$ relative error.

**How we know the right answer.** The NumPy and JAX implementations are independent code paths that derive from the same physical formulas. They must yield identical right-hand side rates within floating-point round-off.

**Why it matters.** The integrator relies on the JAX right-hand side for speed and differentiability. Discrepancies between NumPy and JAX would mean that model predictions depend on which code path is active.

![NumPy <-> JAX RHS parity with solid rheology](../assets/validation/ssc_rhs_parity.svg#only-light){ width="80%" }
![NumPy <-> JAX RHS parity with solid rheology](../assets/validation/ssc_rhs_parity_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The top row shows absolute differences between NumPy and JAX right-hand sides along mantle radius for the three state sets. The bottom row shows relative differences. A pass means that all points lie below the dashed tolerance line at $10^{-6}$.

??? info "Test details"
    - Reference: NumPy `EntropySolver.dSdt` two-pass implementation on `abe_solid_kappah_floor_0` (`s_init = 2500`, `kappah_floor = 0`, `viscosity_solid = 1e18`), `abe_solid_yield_active` (`s_init = 2150`, `kappah_floor = 0`, `viscosity_solid = 3e17`, `yield_stress_c = 150 Pa`), and inert default `abe_solid_default` (`s_init = 2900`, `kappah_floor = 10`, `viscosity_solid = 1e21`).
    - Tolerance: $\max_i |\dot{S}_{\text{NumPy}, i} - \dot{S}_{\text{JAX}, i}| / \max(|\dot{S}_{\text{NumPy}, i}|, 10^{-30}) \le 10^{-6}$ for all 100 states of each state set; active-solid on-vs-off and yield-active yield-on vs yield-off relative RHS differences $\ge 10^{-4}$ at every state.
    - Test id: `tests/test_rheology_solver_wiring_smoke.py::test_rhs_numpy_jax_parity_rheology_on`
    - Command: `pytest tests/test_rheology_solver_wiring_smoke.py -k test_rhs_numpy_jax_parity_rheology_on` (add `--validation-plots` to write the plot pair)

### Round-off perturbation spread with solid rheology

**Why this check exists.** Nonlinear rheology coupling can cause numerical instability where tiny floating-point round-off errors grow exponentially. This check verifies that microscopic initial perturbations do not cause divergent trajectory spreads during planetary thermal evolution.

**What it tests.** Five perturbed initial entropy profiles $S_0^{(k)}(r) = S_0(r)\left(1 + k \cdot 10^{-12}\right)$ ($k \in \{-2, -1, 0, +1, +2\}$) are integrated forward for 100 years on the active-solid variant (`s_init = 2500 J/(kg K)`, `kappah_floor = 0.0 m^2/s`, `viscosity_solid = 1.0e18 Pa s`) and the default `cfg/abe_solid.toml` (`s_init = 2900 J/(kg K)`, `kappah_floor = 10.0 m^2/s`, `viscosity_solid = 1.0e21 Pa s`). The spread $\Delta T(r) = \max_k T_k(r) - \min_k T_k(r)$ over all 100 basic nodes and all 99 staggered nodes measures trajectory sensitivity:

1. Active-solid variant (`abe_solid_kappah_floor_0`): solid rheology alters final temperatures by $\max_r |T_\text{on}(r, t_\text{end}) - T_\text{off}(r, t_\text{end})| = 7.9874\text{ K}$ ($T_\text{surf} = 482.081\text{ K}$ vs $481.620\text{ K}$, relative difference at the surface $(T_\text{on,surf} - T_\text{off,surf}) / T_\text{off,surf} = 9.568 \times 10^{-4}$, relative difference normalized by peak mantle temperature $\max_r |T_\text{on} - T_\text{off}| / \max_r T_\text{off} = 2.205 \times 10^{-3}$, RHS relative difference $7.656 \times 10^{-2}$, `visc_on != visc_off` at all 100 basic nodes, and $\kappa_{h,\text{on}} \ne \kappa_{h,\text{off}}$ at 60 basic nodes). Microscopic round-off perturbations yield maximum radial spread with rheology enabled of $\max_r \Delta T_\text{on}(r) = 1.751 \times 10^{-5}\text{ K}$ ($\Delta T_{\text{on,surf}} = 1.509 \times 10^{-6}\text{ K}$, $\Delta T_{\text{on,cmb}} = 1.237 \times 10^{-9}\text{ K}$, $\max_r \Delta T_{\text{on,stag}}(r) = 3.389 \times 10^{-5}\text{ K}$) against $\max_r \Delta T_\text{off}(r) = 3.439 \times 10^{-6}\text{ K}$ ($\Delta T_{\text{off,surf}} = 3.439 \times 10^{-6}\text{ K}$, $\Delta T_{\text{off,cmb}} = 1.255 \times 10^{-9}\text{ K}$, $\max_r \Delta T_{\text{off,stag}}(r) = 7.680 \times 10^{-6}\text{ K}$), with ratio $\max_r \Delta T_\text{on} / \max_r \Delta T_\text{off} = 5.090$ passing through the $0.05\text{ K}$ absolute floor.
2. Default configuration `cfg/abe_solid.toml`: the floor bounds solid convection and limits basic temperature deviations to $\max_r |T_\text{on}(r, t_\text{end}) - T_\text{off}(r, t_\text{end})| = 0.1939\text{ K}$, relative difference $5.280 \times 10^{-5}$, RHS relative difference $7.996 \times 10^{-3}$, with $\max_r \Delta T_\text{on}(r) = 5.557 \times 10^{-7}\text{ K}$ and $\max_r \Delta T_\text{off}(r) = 1.035 \times 10^{-6}\text{ K}$ ($\Delta T_\text{surf} = 5.557 \times 10^{-7}\text{ K}$).

**How we know the right answer.** Trajectories with rheology enabled must remain stable: their perturbation spread must not exceed twice the rheology-off spread of the same 5 perturbed starts unless bounded by the $0.05\text{ K}$ absolute floor through the mantle.

**Why it matters.** Unstable numerical behaviour causes planetary thermal evolution simulations to diverge or crash during multi-gigayear coupled runs.

![End-time temperature profiles and 5-start radial spread profiles Delta T(r)](../assets/validation/ssc_perturbed_spread.svg#only-light){ width="80%" }
![End-time temperature profiles and 5-start radial spread profiles Delta T(r)](../assets/validation/ssc_perturbed_spread_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The left panels show end-time radial temperature profiles for rheology on and off. The right panels show radial perturbation spreads $\Delta T(r)$. A pass means the solid curve (rheology on) remains below the dashed ceiling line and does not diverge from the baseline (rheology off).

??? info "Test details"
    - Reference: 5 perturbed trajectories with `rheology.enabled = false` on the active-solid variant and default `cfg/abe_solid.toml` (`tests/reference/ssc_wp2_validation_benchmarks.json`).
    - Tolerance: $\Delta T_\text{on}(r) \le \max\big(2\,\Delta T_\text{off}(r), \; 0.05\text{ K}\big)$ over the entire radial grid $r$, requiring that $\Delta T_\text{on}(r)$ does not exceed twice the rheology-off spread of the same 5 perturbed starts unless bounded by the $0.05\text{ K}$ absolute floor (measured ratio 5.090), with active-solid on-vs-off relative difference $\ge 10^{-4}$.
    - Test id: `tests/test_rheology_solver_wiring.py::test_perturbed_starts_spread`
    - Command: `pytest tests/test_rheology_solver_wiring.py -k test_perturbed_starts_spread` (add `--validation-plots` to write the plot pair)

### Two-pass solid rheology RHS cost

**Why this check exists.** Solid rheology calculations execute inside the innermost loop of the differential equation solver. Redundant lookups or repeated compilation would make simulations run slowly. This check counts kernel executions and measures execution time to confirm that two-pass solid rheology adds only the expected kernel calls and bounds JAX wall-time overhead to 25%.

**What it tests.** Enabling two-pass solid rheology (`rheology.enabled = true`) reuses the basic-node solidus temperature $T_\text{sol}(P)$ already evaluated during phase lookup, skips staggered-node rheology viscosity via `defer_rheology_viscosity = True` (NumPy) and `include_rheology = False` (JAX), and calls the validation-free `_core` kernels directly inside the RHS. The test measures call-count deltas and wall times:

| Per-RHS item | Expected delta (`on - off`) | Measured `'lid'` delta (NumPy / JAX) | Measured `'local'` delta (NumPy / JAX) | Difference (`'lid'` - expected) | Reason |
|:---|:---:|:---:|:---:|:---:|:---|
| EOS lookups (solidus temperature) | `0` | `0 / 0` (`5 - 5` / `6 - 6`) | `0 / 0` | `0` | `T_sol(P)` is cached on basic nodes during phase evaluation and reused without extra EOS lookups. |
| `compute_arrhenius_viscosity` | `+2` | `+2 / +2` | `+1 / +1` | `0` | In `'lid'` mode: `1` vector call on $N_b$ basic nodes in phase evaluation plus `1` scalar call at $(T_i, P_i)$ inside `_compute_stagnant_lid_state_core` (`+2`). In `'local'` mode: `1` vector call on $N_b$ basic nodes (`+1`). |
| `compute_yield_stress` | `+2` | `+2 / +2` | `+1 / +1` | `0` | In `'lid'` mode: `1` vector call on $N_b$ basic nodes in phase evaluation plus `1` scalar call at $P_\text{lid,base}$ inside `_compute_stagnant_lid_state_core` (`+2`). In `'local'` mode: `1` vector call on $N_b$ basic nodes (`+1`). |
| `compute_stagnant_lid_state` | `+1` | `+1 / +1` | `0 / 0` | `0` | In `'lid'` mode: `1` call per RHS between Pass 1 (unyielded probe) and Pass 2 (`+1`). In `'local'` mode: active when `stress_closure_mode == 'lid'` (`0`). |
| `compute_effective_viscosity` | `+1` | `+1 / +1` | `+1 / +1` | `0` | `1` vector call on $N_b$ basic nodes in Pass 2 (`+1` in both `'lid'` and `'local'` modes). |
| `viscous_mixing_length_factor` | `+2` | `+2 / +2` | `+2 / +2` | `0` | Relative to `rheology.enabled = false` (`0` calls), Pass 1 (`theta = None`) and Pass 2 (`theta = lid_state['theta']` in `'lid'` mode, `None` in `'local'` mode) each invoke `_viscous_mixing_length_factor_core` once, giving `+2` calls per RHS in both NumPy and JAX. |

Over 1000 RHS evaluations on the 100-basic-node `cfg/abe_solid.toml` mesh (`tests/reference/ssc_wp2_validation_benchmarks.json`), the JIT-compiled JAX RHS wall-time ratio is $1.115\times$ in `'lid'` mode ($530.07\,\mu\text{s}$ vs $475.45\,\mu\text{s}$ per call) and $1.021\times$ in `'local'` mode ($485.22\,\mu\text{s}$ vs $475.45\,\mu\text{s}$ per call), both inside the $1.25\times$ ceiling; on the eager NumPy path, the two-pass RHS wall-time ratio is $1.575\times$ in `'lid'` mode ($2131.63\,\mu\text{s}$ vs $1353.80\,\mu\text{s}$) and $1.188\times$ in `'local'` mode ($1608.12\,\mu\text{s}$ vs $1353.80\,\mu\text{s}$).

**How we know the right answer.** The expected call numbers come directly from code design: Pass 1 calculates unyielded convection and Pass 2 applies lid geometry and yielding. Wall times are measured directly against identical runs with rheology disabled.

**Why it matters.** Coupled planetary simulations run tens of thousands of solver steps. Restricting JAX overhead to below 25% ensures solid-state convection can run in production without slowing down whole-planet models.

![Per-RHS kernel call-count deltas and wall-time ratios](../assets/validation/ssc_rhs_cost.svg#only-light){ width="80%" }
![Per-RHS kernel call-count deltas and wall-time ratios](../assets/validation/ssc_rhs_cost_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The top panel shows call counts for the six kernel components. Points show planned deltas, measured NumPy deltas, and measured JAX deltas (matching across all components except for the planned Pass 2 delta on viscous mixing length factor, which is +1 in the plan and +2 in the measured implementation). The bottom panel shows wall-time ratios for JAX (solid line) and NumPy (dashed line). A pass means JAX ratios remain below the dotted ceiling line at $1.25$.

??? info "Test details"
    - Reference: `rheology.enabled = false` per-RHS call counts and 1000-call wall time on `cfg/abe_solid.toml`.
    - Tolerance: per-RHS call-count deltas $(0, +2, +2, +1, +1, +2)$ in `'lid'` mode and $(0, +1, +1, 0, +1, +2)$ in `'local'` mode on both NumPy and JAX; JIT-compiled JAX wall-time ratio $\le 1.25$.
    - Test id: `tests/test_rheology_solver_wiring.py::test_rhs_call_counts_and_cost`
    - Command: `pytest tests/test_rheology_solver_wiring.py -k test_rhs_call_counts_and_cost` (add `--validation-plots` to write the plot pair)

## Running the verification suite

The standalone suite uses `pytest`:

```sh
# Run the entire suite
pytest tests/

# Run a specific category
pytest tests/test_entropy_pytest.py
pytest tests/test_entropy_verification.py
pytest tests/test_jax_entropy.py

# Mark-filtered subsets
pytest -m unit tests/             # unit tests only
pytest -m smoke tests/            # smoke tests only
```

The JAX-path tests require `jax` and `equinox` to be installed; tests are skipped (not failed) when the optional dependencies are missing. Some tests also require the SPIDER-format EOS table directory; they look in `ARAGOG_TEST_EOS_DIR`, then in `$FWL_DATA/aragog/spider_eos`, then in a parity cache next to the repository (`tests/conftest.py`), and skip when none of them exists.

## Physical constraints checked

The suite asserts the following physical invariants on every solver-driven test run:

1. **Temperature positivity.** $T > 0$ everywhere at all integration times. EOS lookups outside the table domain raise rather than silently extrapolating.
2. **Melt fraction bounds.** $0 \le \phi \le 1$ via the smooth-clip; the assertion runs at every staggered node.
3. **Pressure positivity and monotonicity.** $P(r)$ is positive and decreases monotonically from CMB to surface.
4. **Entropy gradient continuity.** $\partial S/\partial r$ and the resulting fluxes are $C^0$ through the solidus and liquidus crossings (the smoothing functions enforce this).
5. **Energy conservation.** Within the integrator tolerance, $\Delta E_\mathrm{th}$ matches the time-integrated boundary fluxes plus internal heating sources.
6. **Mass conservation.** Total mantle mass is preserved under gravitational separation and chemical mixing (the segregation flux is divergence-free in mass).
