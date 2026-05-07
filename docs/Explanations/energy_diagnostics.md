# Per-call energy diagnostics

Every call to `EntropySolver.solve()` returns a [`SolverOutput`](../Reference/api/aragog.solver.md) with four energy integrals over the CVODE sub-step trajectory:

| Field | Meaning | Sign convention |
|------|--------|----------------|
| `step_dE_F_int_J` | $-\int F_\text{int} A_\text{int}\,dt$ | negative when the mantle loses heat to the atmosphere |
| `step_dE_F_cmb_J` | $+\int F_\text{cmb} A_\text{cmb}\,dt$ | positive when the core delivers heat to the mantle |
| `step_dE_Q_radio_J` | $+\int Q_\text{radio}\,dt$ | non-negative |
| `step_dE_Q_tidal_J` | $+\int Q_\text{tidal}\,dt$ | non-negative |

All four are integrated by Aragog over the actual CVODE sub-step trajectory, using the integrator's own intermediate $y(t)$, not by trapezoidal interpolation between end-of-step snapshots. The previous diagnostic that recomputed $\Delta E$ from end-of-step $F$ snapshots was sensitive to transient values at sub-step boundaries; the new integrals are the conservation primitive.

## Why this matters

The PROTEUS coupling wrapper cumulatively sums these four fields across calls to track the running energy balance. The closed-mantle balance is

$$
\Delta E_\text{mantle}
= \Delta E_{F_\text{int}} + \Delta E_{F_\text{cmb}}
+ \Delta E_{Q_\text{radio}} + \Delta E_{Q_\text{tidal}},
$$

where each $\Delta E_{X}$ is the corresponding `step_dE_*_J` field summed over the run. If the integrals are correctly computed, the cumulative sum should track the change in the mantle's integrated specific enthalpy `E_state` to within numerical-noise tolerance over a coupled run.

## Why no volumetric-work source

The four integrals above are exhaustive for the entropy-form solver: the volumetric work done when a melt of different density is transported across a pressure gradient is already implicit in the divergence of the $\Delta h$-weighted mass-flux contributions to `_heat_flux`. By definition $\Delta h = \Delta u + P\,\Delta v$, and on a hydrostatic column $\partial \Delta h/\partial r \supset \Delta v\,\partial P/\partial r = -\rho g\,\Delta v$, so $-\partial/\partial r(j\,\Delta h)$ already carries the same volumetric-work term. Exposing a separate $\Phi_\text{vol}$ source would double-count ([Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract) §3, SPIDER `energy.c`); a negative regression test in `tests/test_jax_no_phi_vol_source.py` ensures the source stays absent.

## Worked example

For a single 100 kyr coupled step with surface flux $F_\text{int} \sim 10^3\,\text{W/m}^2$ on an Earth-sized planet:

- $A_\text{int} \approx 5 \times 10^{14}\,\text{m}^2$, so the time integral is $\sim 10^3 \cdot 5 \times 10^{14} \cdot 3.16 \times 10^{12} = 1.6 \times 10^{30}\,\text{J}$.
- `step_dE_F_int_J` should land near $-1.6 \times 10^{30}\,\text{J}$ for that step.
- `step_dE_F_cmb_J` is order $10^{27}$-$10^{28}\,\text{J}$ depending on core thermal state.
- `step_dE_Q_radio_J` and `step_dE_Q_tidal_J` are typically order $10^{25}$-$10^{27}\,\text{J}$ for primordial Earth.

A residual `dE_actual / sum(step_dE_*) - 1` of more than a few percent over the run signals either an integrator tolerance problem or a missing energy contribution.
