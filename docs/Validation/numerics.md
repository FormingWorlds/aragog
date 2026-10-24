# First-principles verification

Aragog ships a verification test suite that exercises the entropy solver against analytic limits, conservation laws, and SPIDER-parity references. The suite runs under `pytest` from the repository root and is the primary way to confirm a build is correct.

## Test categories

The tests are organised by the layer they cover:

### EOS unit tests (`tests/test_entropy_pytest.py`)

Exercise `EntropyEOS` and `EntropyPhaseEvaluator`:

- Property lookups (temperature, density, heat capacity, thermal expansivity, isentropic temperature gradient) at on-grid and off-grid $(P, S)$ points.
- Solidus and liquidus entropy lookups against the reference `solidus_P-S.dat` and `liquidus_P-S.dat` files.
- Smooth-clipped melt fraction $\phi(P, S)$: monotone in $S$, clamped to $[0, 1]$, $C^\infty$ across the solidus and liquidus crossings.
- The two-stage SPIDER-parity blend (cubic-Hermite or tanh smoothing followed by `combine_matprop`) for density, heat capacity, and viscosity.
- The latent-blend $c_p$ method against the linear blend.

![EOS bilinear-interpolation Jacobian](../figures/vv/fig_03_eos_bilinear_jacobian.png)

**Figure 1.** JAX-traced bilinear interpolation of the EOS table reproduces the numpy reference to float-64 machine precision and supplies finite, non-zero derivatives via `jax.jacrev` at on- and off-grid $(P, S)$ points. The agreement is the prerequisite for the analytic Jacobian path used by CVODE.

### Phase evaluator tests

Test the SPIDER-parity behaviour of `EntropyPhaseEvaluator`:

- Phase identification at $(P, S)$ in pure solid, pure liquid, and mushy regimes.
- Viscosity tanh blend across the rheological transition $\phi_\mathrm{rheo}$.
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
- JIT and `vmap` compatibility: the JAX flux assembly is differentiable end-to-end with `jax.grad`/`jax.jacrev` and survives JIT compilation without recompilation hot-spots.
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

With solid-state rheology enabled (`[phase_solid.rheology] enabled = true`), the right-hand side evaluates the two-pass stagnant-lid and effective-viscosity update in both `EntropyState.update` (NumPy) and `aragog.jax.phase.compute_fluxes` / `dSdt` (JAX).

On the default `cfg/abe_solid.toml` (`s_init = 2900 J/(kg K)`, `kappah_floor = 10.0 m^2/s`, `viscosity_solid = 1.0e21 Pa s`), the solid rheology alters basic-node viscosity (`visc_on != visc_off` with solid blend weight $1 - w(\phi) > 0$ at all 100 basic nodes and $\eta_\text{eff} \ne \eta_\text{diff}$ at up to 16 nodes across 78 of 100 states), yet `kappah_floor * f_floor(0) = 0.04805 m^2/s` exceeds the unfloored solid convective diffusivity at $\eta_\text{solid} = 10^{21}\text{ Pa s}$, clamping every solid convective cell to the floor so that $\kappa_{h,\text{on}} = \kappa_{h,\text{off}}$ at all 100 basic nodes and $\text{RHS}_\text{on} = \text{RHS}_\text{off}$ identically (`max |RHS_on - RHS_off| / |RHS_off| = 0.0`). On the active-solid Arrhenius variant (`abe_solid_kappah_floor_0`: `s_init = 2500 J/(kg K)`, $S_\text{bot} \in [1850, 2500]\text{ J/(kg K)}$, `kappah_floor = 0.0 m^2/s`, `viscosity_solid = 1.0e18 Pa s`, `yield_stress_c = 1.0e7 Pa`, `yield_stress_mu = 0.1`, `yield_stress_max = 1.0e8 Pa`), homologous Arrhenius strengthening ($\eta_\text{diff} \ge 10^{22}\text{ Pa s}$) keeps Pass 1 convective flux below conductive flux across the column ($f_\text{conv} < 0.5$), so `has_lid = False` and $n_\text{nodes}(\eta_\text{eff} \ne \eta_\text{diff}) = 0$ in all 100 states while Arrhenius strengthening changes the RHS by $[8.595 \times 10^{-3}, \; 2.267 \times 10^{-1}]$. To exercise both stagnant-lid detection and plastic yielding with a direct effect on the RHS, the benchmark adds a third yield-active state set (`abe_solid_yield_active`: warm sub-solidus interior $S_\text{init} = 2650\text{ J/(kg K)}$, $S_\text{bot} \in [2500, 2650]\text{ J/(kg K)}$, $\Delta S \in [350, 550]\text{ J/(kg K)}$, `kappah_floor = 0.0 m^2/s`, `viscosity_solid = 1.0e15 Pa s`, `yield_stress_c = 150.0 Pa`, `yield_stress_mu = 0.0`, `yield_stress_max = 1.0e8 Pa`), where Pass 1 interior convection exceeds conduction ($f_\text{conv} > 0.5$) beneath a cold conductive upper boundary layer (`has_lid = True` in all 100 states), generating driving stress $\tau_{d,\text{lid}} \gg \tau_{y,\text{lid}} = 150\text{ Pa}$ so plastic yielding reduces $\eta_\text{eff} < \eta_\text{diff}$ in `34` to `52` nodes across all `100/100` states.

Across the 100 states of `abe_solid_kappah_floor_0`, the solid rheology modifies viscosity at all 100 basic nodes and modifies $\kappa_h$ at 99 of 100 basic nodes (`93` to `94` basic nodes along the integrated 100-yr `EntropySolver` trajectory), shifting the right-hand side between `rheology.enabled = true` and `false` by $\max |\text{RHS}_\text{on} - \text{RHS}_\text{off}| / \max |\text{RHS}_\text{off}| \in [8.595 \times 10^{-3}, \; 2.267 \times 10^{-1}]$ ($1.844 \times 10^{-2}$ along the integrated trajectory), and the NumPy and JIT-compiled JAX right-hand sides agree to $\max_i |\dot{S}_{\text{NumPy}, i} - \dot{S}_{\text{JAX}, i}| / \max_i |\dot{S}_{\text{NumPy}, i}| \le 2.038 \times 10^{-14}$ ($6.498 \times 10^{-16}$ on the default configuration). In `abe_solid_kappah_floor_0`, the diagnostic interior heat-flux ratio $\max |\Delta J_\text{tot}[1:-1]| / \max |J_\text{tot}[1:-1]| = 6.402 \times 10^{-7}$ and the RHS ratio $2.038 \times 10^{-14}$ both occur at state index `4`, basic node `k = 64` (staggered cell `ks = 63`, $P = 15.32\text{ GPa}$, $S = 2348.11\text{ J/(kg K)}$), where all rheology quantities ($\eta_\text{diff} = \eta_\text{eff} = 2.071 \times 10^{24}\text{ Pa s}$) and conductive flux $J_\text{cond}[64] = 4.292 \times 10^{-3}\text{ W/m}^2$ agree to `0.0`, while single-phase thermal expansivity $\alpha[64]$ differs by $1.416 \times 10^{-4}$ between direct interpolation of `thermal_exp_solid.dat` in NumPy ($3.097833 \times 10^{-5}\text{ K}^{-1}$) and the thermodynamic identity $(\partial T/\partial P)_S \rho C_p / T$ in JAX ($3.098272 \times 10^{-5}\text{ K}^{-1}$); this shifts $J_\text{conv}[64]$ by $4.320 \times 10^{-9}\text{ W/m}^2$, which is $6.402 \times 10^{-7}$ of the maximum interior flux $\max |J_\text{tot}[1:-1]| = 6.749 \times 10^{-3}\text{ W/m}^2$ (excluding the surface radiative boundary node `k = 99` at $6.219 \times 10^4\text{ W/m}^2$), whereas the resulting flux-divergence difference at cell `ks = 63` ($1.147 \times 10^{-12}\text{ J/(kg K yr)}$) divided by the global RHS norm $\max |\dot{S}| = |\dot{S}[98]| = 56.274\text{ J/(kg K yr)}$ at the surface-cooled top cell is $2.038 \times 10^{-14}$. Across the 100 states of `abe_solid_yield_active`, plastic yielding changes the right-hand side relative to yield-off ($\tau_y \to \infty$, `yield_stress_c = 1.0e30 Pa, yield_stress_max = 1.0e30 Pa`) by $\max |\text{RHS}_\text{yield,on} - \text{RHS}_\text{yield,off}| / \max |\text{RHS}_\text{yield,off}| \in [1.233 \times 10^{-4}, \; 1.475 \times 10^{-3}]$ (median $4.027 \times 10^{-4}$, $\ge 10^{-4}$ in all 100 states), while the NumPy and JIT-compiled JAX right-hand sides agree to $\le 8.453 \times 10^{-7}$ (median $1.395 \times 10^{-7}$), within the $10^{-6}$ tolerance.

![NumPy vs JAX RHS relative difference across 100 active-solid cooling states](../assets/validation/ssc_a2_2_rhs_parity.svg#only-light){ width="80%" }
![NumPy vs JAX RHS relative difference across 100 active-solid cooling states](../assets/validation/ssc_a2_2_rhs_parity_dark.svg#only-dark){ width="80%" }

- Reference: NumPy `EntropySolver.dSdt` two-pass implementation on `abe_solid_kappah_floor_0` (`s_init = 2500`, `kappah_floor = 0`, `viscosity_solid = 1e18`), `abe_solid_yield_active` (`s_init = 2650`, `kappah_floor = 0`, `viscosity_solid = 1e15`, `yield_stress_c = 150 Pa`), and inert default `abe_solid_default` (`s_init = 2900`, `kappah_floor = 10`, `viscosity_solid = 1e21`).
- Tolerance: $\max_i |\dot{S}_{\text{NumPy}, i} - \dot{S}_{\text{JAX}, i}| / \max(|\dot{S}_{\text{NumPy}, i}|, 10^{-30}) \le 10^{-6}$ across all 100 states of each state set; active-solid on-vs-off and yield-active yield-on vs yield-off relative RHS differences $\ge 10^{-4}$ at every state.
- Discrimination: perturbing `g_homologous = 0.1` in JAX only on `abe_solid_kappah_floor_0` produces relative RHS differences in $[8.057 \times 10^{-3}, \; 1.846 \times 10^{-1}]$ ($> 8000\times$ the $10^{-6}$ tolerance), and disabling plastic yielding (`yield_stress_c = 1.0e30 Pa, yield_stress_max = 1.0e30 Pa`) in JAX only on `abe_solid_yield_active` produces relative RHS differences in $[1.233 \times 10^{-4}, \; 1.475 \times 10^{-3}]$ ($> 123\times$ the $10^{-6}$ tolerance at every state).
- Test id: `tests/test_rheology_solver_wiring.py::test_a2_2_rhs_numpy_jax_parity_rheology_on`
- Command: `pytest tests/test_rheology_solver_wiring.py -k test_a2_2_rhs_numpy_jax_parity_rheology_on` (add `--validation-plots` to write the plot pair)

### Round-off perturbation spread with solid rheology

To verify that the two-pass stagnant-lid and effective-viscosity coupling introduces no chaotic step-rejection or trajectory divergence under round-off perturbations (`A2-3`), `EntropySolver` integrates 5 trajectories over $100\text{ yr}$ from perturbed initial entropy profiles:

$$
S_0^{(k)}(r) = S_0(r)\left(1 + k \cdot 10^{-12}\right), \quad k \in \{-2, -1, 0, +1, +2\}
$$

with `rheology.enabled = false` and `rheology.enabled = true` on both the active-solid variant (`s_init = 2500 J/(kg K)`, `kappah_floor = 0.0 m^2/s`, `viscosity_solid = 1.0e18 Pa s`) and the default `cfg/abe_solid.toml` (`s_init = 2900 J/(kg K)`, `kappah_floor = 10.0 m^2/s`, `viscosity_solid = 1.0e21 Pa s`). At $t_\text{end} = 100\text{ yr}$, the full radial temperature profiles $T_k(r, t_\text{end})$ on all 100 basic nodes and 99 staggered nodes define the radial spread profiles $\Delta T_\text{on}(r) = \max_{j,k} |T_{\text{on}, j}(r, t_\text{end}) - T_{\text{on}, k}(r, t_\text{end})|$ and $\Delta T_\text{off}(r) = \max_{j,k} |T_{\text{off}, j}(r, t_\text{end}) - T_{\text{off}, k}(r, t_\text{end})|$.

On the active-solid variant, the rheology shifts the $k=0$ end-time basic-node temperature profile by $\max_r |T_\text{on}(r, t_\text{end}) - T_\text{off}(r, t_\text{end})| = 0.9558\text{ K}$ ($T_\text{surf} = 414.548\text{ K}$ vs $415.504\text{ K}$, relative difference $4.229 \times 10^{-4}$ normalized by $\max_r T_\text{off}$ and $2.300 \times 10^{-3}$ at the surface, with $\max |\text{RHS}_\text{on} - \text{RHS}_\text{off}| / \max |\text{RHS}_\text{off}| = 8.622 \times 10^{-3}$, `visc_on != visc_off` at all 100 basic nodes, and $\kappa_{h,\text{on}} \ne \kappa_{h,\text{off}}$ at 94 basic nodes), while all 5 perturbed trajectories take 81 accepted BDF steps and yield $\max_r \Delta T_\text{on}(r) = 1.702 \times 10^{-8}\text{ K}$ ($\Delta T_{\text{on,surf}} = 2.301 \times 10^{-9}\text{ K}$, $\Delta T_{\text{on,cmb}} = 1.702 \times 10^{-8}\text{ K}$, staggered-node $\max_r \Delta T_{\text{on,stag}}(r) = 1.700 \times 10^{-8}\text{ K}$) against $\max_r \Delta T_\text{off}(r) = 2.546 \times 10^{-6}\text{ K}$ ($\Delta T_{\text{off,surf}} = 2.546 \times 10^{-6}\text{ K}$, $\Delta T_{\text{off,cmb}} = 1.702 \times 10^{-8}\text{ K}$, staggered-node $\max_r \Delta T_{\text{off,stag}}(r) = 2.303 \times 10^{-6}\text{ K}$). On the default configuration (`kappah_floor = 10.0 m^2/s`, `viscosity_solid = 1.0e21 Pa s`), the rheology-on and rheology-off trajectories are bitwise identical at all 85 steps ($\max_r |T_\text{on} - T_\text{off}| = 0.0\text{ K}$, $\max |\text{RHS}_\text{on} - \text{RHS}_\text{off}| = 0.0$), with $\max_r \Delta T_\text{on}(r) = \max_r \Delta T_\text{off}(r) = 2.569 \times 10^{-8}\text{ K}$ ($\Delta T_\text{surf} = 1.537 \times 10^{-9}\text{ K}$).

![End-time temperature profiles and 5-start radial spread profiles Delta T(r)](../assets/validation/ssc_a2_3_perturbed_spread.svg#only-light){ width="80%" }
![End-time temperature profiles and 5-start radial spread profiles Delta T(r)](../assets/validation/ssc_a2_3_perturbed_spread_dark.svg#only-dark){ width="80%" }

- Reference: 5 perturbed trajectories with `rheology.enabled = false` on the active-solid variant and default `cfg/abe_solid.toml` (`tests/reference/ssc_wp2_validation_benchmarks.json`).
- Tolerance: $\Delta T_\text{on}(r) \le \max\big(2\,\Delta T_\text{off}(r), \; 0.05\text{ K}\big)$ across the entire radial grid $r$, with active-solid on-vs-off relative difference $\ge 10^{-4}$.
- Test id: `tests/test_rheology_solver_wiring.py::test_a2_3_perturbed_starts_spread`
- Command: `pytest tests/test_rheology_solver_wiring.py -k test_a2_3_perturbed_starts_spread` (add `--validation-plots` to write the plot pair)

### Two-pass solid rheology RHS cost

Enabling two-pass solid rheology (`rheology.enabled = true`) reuses the basic-node solidus temperature $T_\text{sol}(P)$ already evaluated during phase lookup, skips staggered-node rheology viscosity via `defer_rheology_viscosity = True` (NumPy) and `include_rheology = False` (JAX), and calls the validation-free `_core` kernels directly inside the RHS.

| Per-RHS item | Plan `A2-4` delta (`on - off`) | Measured `'lid'` delta (NumPy / JAX) | Measured `'local'` delta (NumPy / JAX) | Difference (`'lid'` - plan) | Reason |
|:---|:---:|:---:|:---:|:---:|:---|
| EOS lookups (solidus temperature) | `0` | `0 / 0` (`5 - 5` / `3 - 3`) | `0 / 0` | `0` | `T_sol(P)` is cached on basic nodes during phase evaluation and reused without extra EOS lookups. |
| `compute_arrhenius_viscosity` | `+2` | `+2 / +2` | `+1 / +1` | `0` | In `'lid'` mode: `1` vector call on $N_b$ basic nodes in phase evaluation plus `1` scalar call at $(T_i, P_i)$ inside `_compute_stagnant_lid_state_core` (`+2`). In `'local'` mode: `1` vector call on $N_b$ basic nodes (`+1`). |
| `compute_yield_stress` | `+2` | `+2 / +2` | `+1 / +1` | `0` | In `'lid'` mode: `1` vector call on $N_b$ basic nodes in phase evaluation plus `1` scalar call at $P_\text{lid,base}$ inside `_compute_stagnant_lid_state_core` (`+2`). In `'local'` mode: `1` vector call on $N_b$ basic nodes (`+1`). |
| `compute_stagnant_lid_state` | `+1` | `+1 / +1` | `0 / 0` | `0` | In `'lid'` mode: `1` call per RHS between Pass 1 (unyielded probe) and Pass 2 (`+1`). In `'local'` mode: gated on `stress_closure_mode == 'lid'` (`0`). |
| `compute_effective_viscosity` | `+1` | `+1 / +1` | `+1 / +1` | `0` | `1` vector call on $N_b$ basic nodes in Pass 2 (`+1` in both `'lid'` and `'local'` modes). |
| `viscous_mixing_length_factor` | `+1` | `+2 / +2` | `+2 / +2` | `+1` | Row `A2-4` quoted the Pass 2 increment (`+1` relative to Pass 1); relative to `rheology.enabled = false` (`0` calls), Pass 1 (`theta = None`) and Pass 2 (`theta = lid_state['theta']` in `'lid'` mode, `None` in `'local'` mode) each invoke `_viscous_mixing_length_factor_core` once, giving `+2` calls per RHS in both NumPy and JAX. |

Over 1000 RHS evaluations on the 100-basic-node `cfg/abe_solid.toml` mesh (`tests/reference/ssc_wp2_validation_benchmarks.json`), the JIT-compiled JAX RHS wall-time ratio is $1.123\times$ in `'lid'` mode ($600.37\,\mu\text{s}$ vs $534.54\,\mu\text{s}$ per call) and $1.110\times$ in `'local'` mode ($593.43\,\mu\text{s}$ vs $534.54\,\mu\text{s}$ per call), both within the $1.25\times$ ceiling; on the eager NumPy path, the two-pass RHS wall-time ratio is $1.523\times$ in `'lid'` mode ($2184.71\,\mu\text{s}$ vs $1434.44\,\mu\text{s}$) and $1.113\times$ in `'local'` mode ($1596.96\,\mu\text{s}$ vs $1434.44\,\mu\text{s}$).

![Per-RHS kernel call-count deltas and wall-time ratios](../assets/validation/ssc_a2_4_rhs_cost.svg#only-light){ width="80%" }
![Per-RHS kernel call-count deltas and wall-time ratios](../assets/validation/ssc_a2_4_rhs_cost_dark.svg#only-dark){ width="80%" }

- Reference: `rheology.enabled = false` per-RHS call counts and 1000-call wall time on `cfg/abe_solid.toml`.
- Tolerance: per-RHS call-count deltas $(0, +2, +2, +1, +1, +2)$ in `'lid'` mode and $(0, +1, +1, 0, +1, +2)$ in `'local'` mode on both NumPy and JAX; JIT-compiled JAX wall-time ratio $\le 1.25$.
- Test id: `tests/test_rheology_solver_wiring.py::test_a2_4_rhs_call_counts_and_cost`
- Command: `pytest tests/test_rheology_solver_wiring.py -k test_a2_4_rhs_call_counts_and_cost` (add `--validation-plots` to write the plot pair)

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
4. **Entropy gradient continuity.** $\partial S/\partial r$ and the resulting fluxes are $C^0$ across the solidus and liquidus crossings (the smoothing functions enforce this).
5. **Energy conservation.** Within the integrator tolerance, $\Delta E_\mathrm{th}$ matches the time-integrated boundary fluxes plus internal heating sources.
6. **Mass conservation.** Total mantle mass is preserved under gravitational separation and chemical mixing (the segregation flux is divergence-free in mass).
