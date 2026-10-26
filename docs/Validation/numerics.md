# First-principles verification

Aragog ships a verification test suite that exercises the entropy solver against analytic limits, conservation laws, and SPIDER-parity references. The suite runs under `pytest` from the repository root and is the primary way to confirm a build is correct.

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

- **Energy conservation.** $E_\mathrm{th}$ change over a closed integration matches the time-integrated surface flux within solver tolerance.
- **Mass conservation.** Mantle mass remains constant through phase changes.
- **Grey-body cooling.** Surface flux follows $\varepsilon\sigma(T_\mathrm{top}^4 - T_\mathrm{eqm}^4)$ inside tolerance once the radiative regime is established.

![Mass-coordinate Jacobian sanity](../figures/vv/fig_07_mass_coord_jacobian.png)

**Figure 2.** Mass-coordinate mesh Jacobian. The Newton solve that maps from mass coordinate to spatial radius (used when `mass_coordinates = true`) is monotone, smooth, and consistent with the analytic Adams-Williamson density profile. Departures here would propagate to mis-aligned cell volumes and a spurious mass-conservation residual in the integrator.

![Radioactive decay](../figures/vv/fig_06_radio_decay.png)

**Figure 3.** Radioactive heating $H_\mathrm{radio}(t)$ for the bundled isotope cocktail (Ruedas 2017): present-day specific powers ($^{40}\mathrm{K}$, $^{232}\mathrm{Th}$, $^{235}\mathrm{U}$, $^{238}\mathrm{U}$) decayed back to Solar System formation. Used as the source-term verification for the closed-mantle energy balance: the cumulative `step_dE_Q_radio_J` integral must match the time integral of $\rho \cdot H_\mathrm{radio}$ over each call to solver tolerance.

### JAX path tests (`tests/test_jax_entropy.py`)

Verify that the JAX-backed `EntropyEOS_JAX` and `compute_fluxes` agree with the numpy path:

- EOS parity: `EntropyEOS_JAX` returns the same temperature, density, heat capacity, thermal expansivity, $\partial T/\partial P|_S$, and conductivity as `EntropyEOS` on the same $(P, S)$ inputs.
- Phase evaluator parity: the JAX two-stage blend matches `EntropyPhaseEvaluator` to floating-point tolerance.
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

## Phase-boundary rate controller verification

### What this section checks

This section verifies the event-driven phase-boundary rate controller (`phase_boundary_cap = "rate"`).
The controller regulates time steps in the CVODE solver near phase boundaries (solidus and liquidus) and during solid mantle cooling.
We verify that rate segmentation maintains temperature accuracy to $1.3 \times 10^{-4}\text{ K}$ against a high-resolution reference (`fixed` step size with tolerances of $10^{-10}$).
We also verify round-off stability, non-stalling behavior during mantle solidification, and coupled PROTEUS evolution.
The cases below evaluate accuracy, stiff-zone boundary landing, post-solidification stepping, and coupled behavior.

### Accuracy of event-driven rate segmentation at phase boundaries

**Why this check exists.**
Near phase boundaries, latent heat release causes sharp changes in effective heat capacity.
If time steps are too large near a phase boundary, the numerical integrator incurs large local truncation errors or step rejections.

**What it tests.**
Four test cases (Cases A, B, C, D) cover mushy initial conditions, different core boundary conditions, and spatial resolutions (16 to 24 nodes) with smoothing width $w = 0.01$.
The test compares the temperature profile $T(r)$ from `phase_boundary_cap = "rate"` at relative tolerance $10^{-8}$ against a high-resolution reference (`phase_boundary_cap = "fixed"` with $\Delta t = 1\text{ yr}$ and tolerances of $10^{-10}$).

**How we know the right answer.**
A fixed step size of $\Delta t = 1\text{ yr}$ with tolerances of $10^{-10}$ resolves phase boundaries with high precision.
Numerical errors stay near floating-point precision rather than temporal discretization limits.

**Why it matters.**
Accurate temperatures near phase boundaries set the timing and depth of magma ocean crystallization in planetary evolution models.

??? info "Test details"
    - Reference: Fixed $1\text{ yr}$ step size reference at relative and absolute tolerances of $10^{-10}$
    - Tolerance: $\max_i |T_{\text{rate}, 10^{-8}, i} - T_{\text{fixed}, 10^{-10}, i}| \le 1.3 \times 10^{-4}\text{ K}$ (observed: Case A $6.93 \times 10^{-6}\text{ K}$, Case B $8.65 \times 10^{-5}\text{ K}$, Case C $6.93 \times 10^{-6}\text{ K}$, Case D $2.83 \times 10^{-5}\text{ K}$)
    - Test id: V1
    - Command: `ARAGOG_TEST_EOS_DIR=... python tools/verification/run_phase_boundary_cap_accuracy.py --check`

### Stiff-zone boundary landing under slow cooling

**Why this check exists.**
When a cell cools slowly near the boundary, its cooling rate $|\dot{S}|$ can approach zero while the distance to the boundary remains small.
Without a floor on the effective progress distance, the step cap could collapse to the $1\text{ yr}$ minimum step bound, causing many tiny steps.

**What it tests.**
A 12-node mantle starts two cells above the solidus plus the stiff-zone half-width $\delta$ under a low surface heat flux ($F = 20\text{ W/m}^2$) over $15\,000\text{ yr}$.
The test verifies that the stiff-zone landing rule ($d_{\text{eff}} = \max(d_{\text{near}}, 0.5\,\delta)$) allows the cell to advance cleanly into Regime 3 without step collapse.

**How we know the right answer.**
The solution matches a high-precision reference solve with tolerances of $10^{-10}$ and stays inside the perturbation envelope $\Delta_{\text{pert}}(T)$ under $10^{-12}$ round-off perturbations.

**Why it matters.**
Low surface fluxes occur in insulated mantles under thick primordial atmospheres, where efficient stepping is required to simulate long cooling timescales.

??? info "Test details"
    - Reference: Fixed $1\text{ yr}$ step size reference at tolerances of $10^{-10}$ and 5 round-off perturbations
    - Tolerance: $\|T_{\text{rate}}^{(0)} - T_{\text{fixed}, 10^{-10}}\|_\infty \le 1.3 \times 10^{-4}\text{ K}$ (observed: $3.62 \times 10^{-5}\text{ K}$, envelope $\Delta_{\text{pert}} = 3.40 \times 10^{-5}\text{ K}$)
    - Test id: V3-R3
    - Command: `python scripts/run_v3_r3.py`

### Sub-solidus upper-bound relaxation and tolerance tightening

**Why this check exists.**
Once the entire mantle cools below the solidus, no phase boundary lies ahead in the cooling direction.
Imposing a $100\text{ yr}$ step ceiling on sub-solidus cooling slows down computation.
Removing the $100\text{ yr}$ ceiling without tightening tolerances could degrade solution accuracy.

**What it tests.**
A 12-node mantle starts above the solidus and cools for $200\,000\text{ yr}$ under $F = 200\text{ W/m}^2$.
Once all cells cool below the solidus, the upper step bound relaxes to $\infty$, and segment tolerances tighten to $\mathtt{rtol} \le 10^{-9}$ and $\mathtt{atol} \le 10^{-8}/\text{scale}$.

**How we know the right answer.**
The solution matches the fixed high-precision reference to $1.3 \times 10^{-4}\text{ K}$ while reducing total integration steps from 3688 to 1031.

**Why it matters.**
Post-magma-ocean solid-state mantle cooling covers billions of years; step relaxation allows long coupled thermal evolution runs to complete quickly.

??? info "Test details"
    - Reference: Fixed $1\text{ yr}$ step size reference at tolerances of $10^{-10}$ and 5 round-off perturbations
    - Tolerance: $\|T_{\text{rate}}^{(0)} - T_{\text{fixed}, 10^{-10}}\|_\infty \le 1.3 \times 10^{-4}\text{ K}$ (observed: $7.54 \times 10^{-5}\text{ K}$, envelope $\Delta_{\text{pert}} = 1.07 \times 10^{-6}\text{ K}$)
    - Test id: V3-R4
    - Command: `python scripts/run_v3_r4.py`

### Full solidification without solver stalling

**Why this check exists.**
In deep mantle cells at high pressures, thermal boundary layer sharpening and phase-boundary crossing previously triggered integrator stalls in 100-node setups, failing to reach full solidification.

**What it tests.**
A 100-node mantle using PALEOS low-temperature equation of state tables is integrated past historical stall points ($83.684\text{ Myr}$ and $107.655\text{ Myr}$) under both stagnant-lid rheology (`rheo_on`) and rheology off (`rheo_off`) until full solidification ($\phi_{\max} = 0$, status 0).

**How we know the right answer.**
Each run passes its historical stall point ($83.684\text{ Myr}$ for `rheo_on`, $107.655\text{ Myr}$ for `rheo_off`), reaches full solidification ($\phi_{\max} = 0$), and completes with status 0, while keeping entropy inside the valid equation of state table domain.

**Why it matters.**
Long planetary evolution runs require robust integration from magma ocean through complete mantle solidification into solid-state convection.

**Known limit.**
Per Tim's ruling (2026-10-26, Ruling 19), integration to $1.0 \times 10^9\text{ yr}$ is not required; each run passes its old stall point and reaches full solidification ($\phi_{\max} = 0$) with status 0.
The first `rheo_on` run used finite-difference Jacobians because the standalone runner installed no JAX factory, and does not count for V4; only the run with the JAX factory installed (as in coupled PROTEUS runs) counts (Ruling 22b).

??? info "Test details"
    - Reference: Historical stall points at $83.684\text{ Myr}$ and $107.655\text{ Myr}$
    - Tolerance: Each run passes its old stall point (rheo_on $83.684\text{ Myr}$, rheo_off $107.655\text{ Myr}$), reaches $\phi_{\max} = 0$, status 0; $10^9\text{ yr}$ not required. The first rheo_on run used finite-difference Jacobians (no JAX factory in the standalone runner) and does not count; only the factory run counts. Evidence: unique progress files (Ruling 19, Ruling 22b).
    - Test id: V4
    - Command: `python scripts/run_v4_rheo_off_r18.py` and `python scripts/run_v4_rheo_on_r22.py`

### Coupled planetary evolution under super-solidus conditions

**Why this check exists.**
Coupling Aragog to the PROTEUS framework requires exchanging heat fluxes and boundary temperatures with the atmospheric model (Janus).
Unchanged super-solidus paths must stay stable and perturbation-robust under coupling.

**What it tests.**
Five coupled PROTEUS + Aragog iterations on `tests/integration/aragog_janus.toml` start at $T_{\text{surf}} = 4000\text{ K}$, comparing the branch tip against `origin/main` over 5 round-off perturbations.

**How we know the right answer.**
Unperturbed before and after differences stay inside the round-off perturbation envelope $\Delta_{\text{pert}}$ or $10^{-8}$ relative on all tracked planetary helpfile columns and staggered profiles.

**Why it matters.**
Ensures the phase-boundary controller introduces no regressions into existing coupled magma ocean simulations.

??? info "Test details"
    - Reference: `origin/main` coupled trajectory and 5 round-off perturbation runs
    - Tolerance: Before/after difference $\le \max(\Delta_{\text{pert}}, 10^{-8}\text{ rel})$ on all helpfile columns and profiles
    - Test id: V5
    - Command: `python scripts/run_v5.py`

### Coupled planetary evolution crossing the solidus into the solid regime

**Why this check exists.**
When a coupled PROTEUS planet cools below the critical melt fraction into the solid regime, the interior solver transitions into Regime 4.
The coupled run must cross the solidus and execute solid-state steps smoothly.

**What it tests.**
Coupled PROTEUS + Aragog + Calliope run on `tests/integration/zalmoxis_aragog_calliope.toml` starts in the mushy zone ($T_{\text{surf}} = 1750\text{ K}$), continuing until 3 coupling steps after the global melt fraction reaches zero.

**How we know the right answer.**
The branch's error against the strictest reference (rtol $10^{-10}$ with $1\text{ yr}$ steps in Regime 4) is not larger than main's on all 11 columns, confirmed on the hand-back tip (Tim's ruling 2026-10-26, Ruling 19).

**Why it matters.**
Shows stable coupling through the final stages of planetary solidification into the rocky mantle regime.

**Known limit.**
Step ceiling relaxation and tolerance tightening in Regime 4 alter step sizes and numerical truncation errors by design. Validation verifies that the branch's error against the strictest reference ($1\text{ yr}$ Regime 4 steps at $10^{-10}$) is not larger than main's on all 11 columns.

??? info "Test details"
    - Reference: Strictest reference (rtol $10^{-10}$ with $1\text{ yr}$ steps in Regime 4) and `origin/main` coupled trajectory
    - Tolerance: The branch's error against the strictest reference (rtol $10^{-10}$ with $1\text{ yr}$ steps in Regime 4) is not larger than main's on all 11 columns, confirmed on the hand-back tip (Tim's ruling 2026-10-26, Ruling 19)
    - Test id: V5b
    - Command: `python scripts/run_v5b.py`
