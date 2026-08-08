# Core boundary condition modes

The `boundary_conditions.core_bc` setting selects the formulation used at the core-mantle boundary (CMB) when `inner_boundary_condition = 1` (core cooling). Five modes are available; they differ in what is treated as the primary state variable, what is reconstructed, and how strongly the bottom mantle cell is coupled to the core.

## `quasi_steady`

State vector length: $N$ (entropy at staggered nodes only).

The CMB heat flux is computed from a quasi-steady balance using an $\alpha$-factor partition between the bottom mantle cell and the core, weighted by the heat-capacity ratio. The core temperature is reconstructed from the bottom-cell entropy and pressure via the EOS at every output time.

This mode is fast, stable, and conservative in the sense that it does not require an extra ODE state. It underestimates the true CMB heat loss relative to the SPIDER reference by 5-10 % over the first solidification cycle of a 1 M$_\oplus$ run, which is the right trade-off for quick standalone exploration where SPIDER parity is not the goal.

## `energy_balance` (default)

State vector length: $N + 1$ (entropy at staggered nodes plus the entropy gradient at the CMB basic node, $dS/dr |_\text{cmb}$).

The entropy gradient at the CMB is added as an extra state variable and is integrated via SPIDER's [`bc.c:76-131`](https://github.com/FormingWorlds/SPIDER) formula:

$$
\frac{d}{dt}\left(\frac{dS}{dr}\bigg|_\text{cmb}\right) = \frac{2}{\Delta r}\left( -F_\text{cmb} A_\text{cmb} \cdot \text{fac}_\text{cmb} - \frac{dS}{dt}\bigg|_0\right)
$$

with

$$
\text{fac}_\text{cmb} = \frac{c_p^\text{cmb}}{c_p^\text{core} \, T_\text{cmb} \, \text{tfac} \, M_\text{core}}
$$

This is the SPIDER-parity formulation. It produces bit-for-bit agreement with SPIDER on the Earth reference. Use this mode for any run that needs to reproduce SPIDER results, including the published verification suite. State vector size and Jacobian sparsity are slightly larger than `quasi_steady`, but the integrator overhead is small.

## `gradient`

State vector length: $N + 2$ (the entropy gradient is the primary field; entropy is reconstructed by cumulative integration from the surface).

Treats $dS/dr$ as the integrated quantity rather than $S$ itself, with two extra boundary states (one for each surface). Produces the same physics as `energy_balance` but with a different conditioning that is more stable in regimes where $S$ has a sharp kink at the rheological transition. Has not been validated against SPIDER as thoroughly as `energy_balance`; treat as experimental.

## `bower2018`

State vector length: $N + 1$ (with $T_\text{core}$ as the extra state).

Treats the core temperature as an ODE state variable, with the CMB heat flux computed from conduction across the bottom half-cell. The conduction-only flux underestimates the true CMB heat loss by orders of magnitude for any planet with active mantle convection; this mode is retained for parity testing only and is **not recommended for production**.

## `core_module`

State vector length: $N + 1$ (with $T_\text{cmb}$ as the extra state, driven by the staged core-evolution budget of [`aragog.core`](../Reference/api/aragog.core.md)).

The core carries its own physics instead of an isothermal reservoir: the CMB temperature evolves through an effective heat capacity

$$
Q_\text{cmb} = -\tilde{C}(T_\text{cmb})\, \frac{dT_\text{cmb}}{dt} + Q_\text{sources}, \qquad \tilde{C} = \tilde{C}_\text{secular} + \tilde{C}_\text{latent} + \tilde{C}_\text{grav},
$$

where the secular term integrates the mass-weighted adiabat over the closed-form Gaussian core profiles, and the latent and gravitational terms follow inner-core growth through the implicit-function sensitivity of the adiabat-liquidus crossing, activated smoothly at nucleation onset and wound down smoothly at freeze-out completion. The melting curve is the PALEOS iron prescription (Anzellini et al. 2013) with a light-element depression, or the Nimmo (2015) quadratic. All terms are cross-validated against the Leeds `thermal_history` implementation and pinned against Nimmo (2015) Table 2; the entropy side ([`CoreEntropyBudget`](../Reference/api/aragog.core.md)) adds the dynamo criterion and field-strength scaling.

Two properties matter for coupled stability. The reported core temperature is the integrated boundary state, not the lowermost mantle node's EOS read-off, so it does not inherit the node's phase-branch snaps at crystallisation onset. And the budget is smooth with correct derivatives (the boundary solve carries a custom JVP), so anything differentiating through the core state sees the true sensitivities; the CVODE analytic-Jacobian factory does not yet cover this mode, so the solver falls back to its finite-difference Jacobian, which the smooth budget keeps well-conditioned.

With every feature disabled the mode reduces to the isothermal-reservoir law, which is its regression anchor against `bower2018`-style behaviour.

## How to choose

| Need | Recommended `core_bc` |
|------|----------------------|
| Production PROTEUS runs and SPIDER-parity validation | `energy_balance` (default) |
| Quick standalone exploration where SPIDER parity is not required | `quasi_steady` |
| Very steep mushy-band gradient that destabilises `energy_balance` | `gradient` (experimental) |
| Reproducing pre-2026 results | `bower2018` (legacy) |
| Core evolution with inner-core growth, dynamo diagnostics, or a smooth core temperature through crystallisation onset | `core_module` |

The state-vector layout for each mode is documented in [`solver/entropy_solver.py`](https://github.com/FormingWorlds/aragog/blob/main/src/aragog/solver/entropy_solver.py) at the `_build_jac_sparsity` and `set_initial_entropy` methods; the test class `TestEnergyBalanceCoreBC` in `tests/test_entropy_pytest.py` exercises the `energy_balance` mode directly.
