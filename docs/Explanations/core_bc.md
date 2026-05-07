# Core boundary condition modes

The `boundary_conditions.core_bc` setting selects the formulation used at the core-mantle boundary (CMB) when `inner_boundary_condition = 1` (core cooling). Four modes are available; they differ in what is treated as the primary state variable, what is reconstructed, and how strongly the bottom mantle cell is coupled to the core.

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

This is the SPIDER-parity formulation. It produces bit-for-bit agreement with SPIDER on the CHILI Earth reference. Use this mode for any run that needs to reproduce SPIDER results, including the published verification suite. State vector size and Jacobian sparsity are slightly larger than `quasi_steady`, but the integrator overhead is small.

## `gradient`

State vector length: $N + 2$ (the entropy gradient is the primary field; entropy is reconstructed by cumulative integration from the surface).

Treats $dS/dr$ as the integrated quantity rather than $S$ itself, with two extra boundary states (one for each surface). Produces the same physics as `energy_balance` but with a different conditioning that is more stable in regimes where $S$ has a sharp kink at the rheological transition. Has not been validated against SPIDER as thoroughly as `energy_balance`; treat as experimental.

## `bower2018`

State vector length: $N + 1$ (with $T_\text{core}$ as the extra state).

Treats the core temperature as an ODE state variable, with the CMB heat flux computed from conduction across the bottom half-cell. The conduction-only flux underestimates the true CMB heat loss by orders of magnitude for any planet with active mantle convection; this mode is retained for parity testing only and is **not recommended for production**.

## How to choose

| Need | Recommended `core_bc` |
|------|----------------------|
| Production CHILI runs and SPIDER-parity validation | `energy_balance` (default) |
| Quick standalone exploration where SPIDER parity is not required | `quasi_steady` |
| Very steep mushy-band gradient that destabilises `energy_balance` | `gradient` (experimental) |
| Reproducing pre-2026 results | `bower2018` (legacy) |

The state-vector layout for each mode is documented in [`solver/entropy_solver.py`](https://github.com/FormingWorlds/aragog/blob/main/src/aragog/solver/entropy_solver.py) at the `_build_jac_sparsity` and `set_initial_entropy` methods; the test class `TestEnergyBalanceCoreBC` in `tests/test_entropy_pytest.py` exercises the `energy_balance` mode directly.
