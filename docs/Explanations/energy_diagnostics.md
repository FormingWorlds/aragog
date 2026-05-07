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

The PROTEUS coupling wrapper cumulatively sums these four fields across calls to track the running energy balance. The closed-mantle balance is:

$$
\Delta E = \text{step\_dE\_F\_int\_J} + \text{step\_dE\_F\_cmb\_J} + \text{step\_dE\_Q\_radio\_J} + \text{step\_dE\_Q\_tidal\_J}
$$

If the integrals are correctly computed, the cumulative sum should track the change in the mantle's integrated specific enthalpy `E_state` to within numerical-noise tolerance over a coupled CHILI run.

## Why no `step_dE_Q_dil_J`

Earlier versions of Aragog tracked a fifth integral, `step_dE_Q_dil_J`, for the explicit dilatation source $\Phi_\text{vol}$. That source has been deleted (aragog `dcd7f37`, May 2026) because the volumetric work is already implicit in the divergence of the $\Delta h$-weighted mass-flux contributions to `_heat_flux` (chain rule on $\Delta h = \Delta u + P\,\Delta v$ with hydrostatic $\partial P/\partial r = -\rho g$). Adding it explicitly was a 2x over-supply, locking the integrator at a heat-pump quasi-equilibrium. The deleted integral is preserved as a negative regression test in `tests/test_jax_no_phi_vol_source.py`.

## Worked example

For a single 100 kyr coupled step with surface flux $F_\text{int} \sim 10^3\,\text{W/m}^2$ on an Earth-sized planet:

- $A_\text{int} \approx 5 \times 10^{14}\,\text{m}^2$, so the time integral is $\sim 10^3 \cdot 5 \times 10^{14} \cdot 3.16 \times 10^{12} = 1.6 \times 10^{30}\,\text{J}$.
- `step_dE_F_int_J` should land near $-1.6 \times 10^{30}\,\text{J}$ for that step.
- `step_dE_F_cmb_J` is order $10^{27}$-$10^{28}\,\text{J}$ depending on core thermal state.
- `step_dE_Q_radio_J` and `step_dE_Q_tidal_J` are typically order $10^{25}$-$10^{27}\,\text{J}$ for primordial Earth.

A residual `dE_actual / sum(step_dE_*) - 1` of more than a few percent over the run signals either an integrator tolerance problem or a missing energy contribution.
