# Energy-conservation diagnostics

Aragog emits a **two-track** energy-conservation diagnostic on every solver call:

1. A **physical conservation budget** that compares the integrated mantle enthalpy against the cumulative time integral of the boundary fluxes and volumetric sources.
2. A **solver-correctness check** that verifies the entropy ODE itself was integrated to its stated tolerance, independent of any thermodynamic-frame considerations.

The first track answers _did the simulation conserve energy as a physical principle_, the second answers _did CVODE solve its ODE accurately_. Both are exposed as columns in the PROTEUS `runtime_helpfile.csv` and as fields on Aragog's [`SolverOutput`](../Reference/api/aragog.solver.md) dataclass.

## Per-call integrals (`SolverOutput`)

Every call to `EntropySolver.solve()` returns a [`SolverOutput`](../Reference/api/aragog.solver.md) with seven energy-related integrals computed over the actual CVODE sub-step trajectory.

| Field [J] | Definition | Sign convention |
|---|---|---|
| `step_dE_F_int_J` | $-\int F_\text{int}\,A_\text{int}\,dt$ | Negative when the mantle loses heat to the atmosphere |
| `step_dE_F_cmb_J` | $+\int F_\text{cmb}\,A_\text{cmb}\,dt$ | Positive when the core delivers heat to the mantle |
| `step_dE_Q_radio_J` | $+\int Q_\text{radio}\,dt$ (state-mass) | Non-negative; uses time-varying $\rho(P,S)\,V$ |
| `step_dE_Q_tidal_J` | $+\int Q_\text{tidal}\,dt$ (state-mass) | Non-negative; uses time-varying $\rho(P,S)\,V$ |
| `step_dE_Q_radio_cons_J` | $+\int Q_\text{radio}\,dt$ (frozen mass) | Non-negative; uses fixed $\rho_\text{struct}\,V$ |
| `step_dE_Q_tidal_cons_J` | $+\int Q_\text{tidal}\,dt$ (frozen mass) | Non-negative; uses fixed $\rho_\text{struct}\,V$ |
| `step_solver_residual_J` | $\int (\text{LHS} - \text{RHS})\,dt$ | $\sim 0$ at machine precision |

All integrals use the trapezoidal rule on the integrator's accepted sub-steps. Integrating over the actual CVODE trajectory rather than between end-of-step snapshots is required because a single CVODE phase-boundary spike on a coarse end-of-step interpolation can shift the integral by orders of magnitude.

The two surface-flux integrals (`step_dE_F_int_J` and `step_dE_F_cmb_J`) are area-weighted only and have no mass-weighting variant. The volumetric source integrals come in two flavours, distinguished by the mass weighting on $Q_\text{radio} = \rho\,h_\text{radio}\,V$ and $Q_\text{tidal} = \rho\,h_\text{tidal}\,V$:

- **State-mass**: $\rho_\text{state} = \rho(P_\text{stag},\,S(t))$, evolves with the solver state. Used for instantaneous power diagnostics.
- **Frozen mass**: $\rho_\text{struct}\,V$ from `mesh.staggered_effective_density × mesh.basic.volume`, fixed at IC. Used in the conservation-grade integrated enthalpy `E_state_cons` (see below).

## Cumulative conservation residuals (PROTEUS helpfile)

The PROTEUS coupling layer (`proteus.utils.coupler._populate_energy_residual`) accumulates the per-call integrals into running cumulative columns on `runtime_helpfile.csv`:

| Column [J or 1] | Definition |
|---|---|
| `E_state_cons_J` | $\sum_\text{cells} h(P,S(t))\,\rho_\text{struct}\,V$ (frozen-mass integrated enthalpy) |
| `dE_predicted_cons_J` | $\sum_\text{rows}(\text{step\_dE\_F\_int\_J} + \text{step\_dE\_F\_cmb\_J} + \text{step\_dE\_Q\_radio\_cons\_J} + \text{step\_dE\_Q\_tidal\_cons\_J})$ |
| `E_residual_cons_J` | $\Delta E_\text{state\_cons} - \text{dE\_predicted\_cons\_J}$, with $\Delta E_\text{state\_cons} = E_\text{state\_cons}(t) - E_\text{state\_cons}(0)$ |
| `E_residual_cons_frac` | `E_residual_cons_J` $/\,\max(\lvert\Delta E_\text{state\_cons}\rvert,\,1\,\text{J})$ |
| `solver_residual_J` | $\sum_\text{rows}\text{step\_solver\_residual\_J}$ (cumulative entropy-ODE LHS-RHS) |

`E_residual_cons_J` is the **physical conservation residual**: the difference between the actual change in mantle enthalpy and the integrated boundary-and-source budget. `solver_residual_J` is the **solver-correctness residual**: by construction the entropy ODE satisfies $\sum_\text{cells}\rho T\,(\partial S/\partial t)\,V = -F_\text{int}\,A_\text{int} + F_\text{cmb}\,A_\text{cmb} + Q_\text{radio} + Q_\text{tidal}$ at every accepted CVODE sub-step, so any drift in `solver_residual_J` flags real CVODE step rejection or atol/rtol issues.

## Closed-mantle energy balance

Aragog's closed-mantle energy balance is

$$
\frac{d E_\text{mantle}}{dt}
= -F_\text{int}\,A_\text{int}
+ F_\text{cmb}\,A_\text{cmb}
+ Q_\text{radio} + Q_\text{tidal}.
$$

A volumetric-work source $\Phi_\text{vol}$ does **not** appear in this budget: the volumetric work done when a melt of different density is transported across a pressure gradient is already implicit in the divergence of the $\Delta h$-weighted mass-flux contributions to `_heat_flux`. By definition $\Delta h = \Delta u + P\,\Delta v$, and on a hydrostatic column $\partial \Delta h/\partial r \supset \Delta v\,\partial P/\partial r = -\rho g\,\Delta v$, so $-\partial/\partial r(j\,\Delta h)$ already carries the volumetric-work term. A separate $\Phi_\text{vol}$ source would double-count ([Bower et al. 2018](https://scixplorer.org/abs/2018PEPI..274...49B/abstract)[^cite-bower2018] §3, SPIDER `energy.c`); a negative regression test in `tests/test_jax_no_phi_vol_source.py` enforces the absence.

## Why frozen-mass weighting

`E_state_cons_J` integrates the EOS-consistent specific enthalpy $h(P,S)$ table against a **frozen** mass-per-shell profile $\rho_\text{struct}\,V$ taken once from the equilibrium structure at IC, not against the time-varying state-dependent mass $\rho(P,S(t))\,V$.

The reason is dimensional consistency with the entropy ODE's frame. The entropy ODE evolves in fixed-volume Eulerian coordinates with state-dependent $\rho(P,S(t))$. Its per-cell discretisation gives

$$
\rho_i\,T_i\,(\partial S_i/\partial t)\,V_i
= F_i\,A_i - F_{i+1}\,A_{i+1}
+ H_i\,\rho_i\,V_i,
$$

and summing over cells yields the surface telescope $\sum \rho_i T_i (\partial S_i/\partial t)\,V_i = -F_\text{int}\,A_\text{int} + F_\text{cmb}\,A_\text{cmb} + Q_\text{radio} + Q_\text{tidal}$. The mass weighting in this identity is $\rho(t)\,V$ at every instant.

If the conservation diagnostic uses **state-dependent** mass too, the actual integrated enthalpy is

$$
E_\text{state}(t)
= \sum_\text{cells} h(P,S(t))\,\rho(P,S(t))\,V
= \sum_\text{cells} h\,\rho_\text{state}\,V,
$$

whose time derivative picks up an $h\,\partial\rho/\partial t$ cross term that **does not** appear in the divergence-of-flux + sources budget on the right-hand side:

$$
\frac{d E_\text{state}}{dt}
= \underbrace{\sum \rho\,(\partial h/\partial t)\,V}_{\text{matches entropy budget}}
+ \underbrace{\sum h\,(\partial \rho/\partial t)\,V}_{\text{frame artefact}}.
$$

The frame artefact grows with mantle cooling because $\rho(P,S(t))$ increases as the mantle solidifies. On a static-Zalmoxis dry $a_1$ benchmark, a state-mass residual reaches **51 % of total cooling** at 100 kyr — a non-physical signal that masks any real conservation drift.

`E_state_cons_J` uses the frozen mass-per-shell, eliminating $h\,\partial\rho/\partial t$ from the diagnostic side. The same trajectory closes to **5.3 % of total cooling** — bounded and physically meaningful. `E_residual_cons_J` and `E_residual_cons_frac` are therefore the canonical conservation columns.

## Why two diagnostics

The frozen-mass residual `E_residual_cons_frac` answers a slightly different question than the solver residual `solver_residual_J`. Both are useful:

- **`E_residual_cons_frac`** quantifies the gap between the physical enthalpy change and the integrated divergence-of-flux + sources budget. It still has a small frame correction at the ~1 % level because $\rho_\text{struct}\,V \ne \rho(t)\,V$ in general, so the frozen-mass enthalpy time-derivative is not exactly the entropy-budget LHS. Acceptable values are bounded around a few percent over multi-Myr trajectories; a sustained drift well above this signals a real conservation regression.

- **`solver_residual_J`** is the cumulative integral of the per-substep mismatch between $\sum \rho T\,(\partial S/\partial t)\,V$ and the entropy-budget RHS, evaluated at the same state-dependent $\rho$. By construction the entropy ODE satisfies this identity at every accepted CVODE step; any drift here is purely the gap between the trapezoidal accumulation Aragog uses for diagnostics and CVODE's own internal step accuracy. Closes to ~10⁻⁷ of total cooling on the verified benchmark — machine-precision noise floor. A non-trivial drift here flags real solver issues (step rejection, atol/rtol problems, hidden NaN propagation).

In production CHILI runs you should monitor both columns. `E_residual_cons_frac` is the headline "is the simulation conserving energy?" check; `solver_residual_J` is the gate that catches numerical pathologies before they show up as a misleading conservation drift.

## Verification

Reference smoke runs on a static-Zalmoxis dry $a_1$ ($1\,M_\oplus$, no atmosphere, no radio, no tidal) trajectory starting from a fully molten mantle at $T_\text{magma} \approx 4244\,\text{K}$:

| $t_\text{max}$ | Final $T_\text{magma}$ | Final $\Phi$ | `E_residual_cons_frac` | `solver_residual_J / dE_actual` |
|---|---|---|---|---|
| 10 kyr | 3393 K | 0.97 | 11.5 % | $1.6\times10^{-7}$ |
| 100 kyr | 2331 K | 0.73 | 5.3 % | $3.4\times10^{-7}$ |

The frozen-mass residual stays bounded around a few percent over multiple orders of magnitude in time; the solver residual sits at the floating-point noise floor.

## Worked example

For a single 100 kyr coupled step with surface flux $F_\text{int} \sim 10^3\,\text{W/m}^2$ on an Earth-sized planet:

- $A_\text{int} \approx 5 \times 10^{14}\,\text{m}^2$, so $\sim 10^3 \cdot 5 \times 10^{14} \cdot 3.16 \times 10^{12} = 1.6 \times 10^{30}\,\text{J}$.
- `step_dE_F_int_J` should land near $-1.6 \times 10^{30}\,\text{J}$ for that step.
- `step_dE_F_cmb_J` is order $10^{27}\text{-}10^{28}\,\text{J}$ depending on core thermal state.
- `step_dE_Q_radio_J` and `step_dE_Q_tidal_J` are typically order $10^{25}\text{-}10^{27}\,\text{J}$ for primordial Earth.

Cumulative `E_residual_cons_frac` of more than ~5 % over a multi-Myr run, or `solver_residual_J / |dE_actual|` of more than ~10⁻⁵, signals an integrator-tolerance problem or a missing energy contribution. Plot both columns over time using `tools/plot_energy_balance.py` (PROTEUS-side):

```bash
python tools/plot_energy_balance.py output/<run_dir> --label <run_label>
# writes output_files/energy_balance/<label>_balance.pdf and _powers.pdf
```

[^cite-bower2018]: D. J. Bower, et al., *Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets*, Physics of the Earth and Planetary Interiors, 2018.
