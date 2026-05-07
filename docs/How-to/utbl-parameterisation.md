# Upper thermal boundary layer (UTBL) parameterisation

When the surface boundary uses the grey-body atmosphere mode (`outer_boundary_condition = 1`), the radiating surface temperature can differ from the temperature of the topmost mantle staggered node. Aragog's UTBL parameterisation accounts for the unresolved thermal boundary layer between the magma ocean interior and the radiating surface by solving a cubic relation derived in Bower et al. (2018, Eq. 18). The implementation is in `src/aragog/solver/boundary.py::_utbl_tsurf` and uses Cardano's formula for the analytic real root.

## When to enable

| Situation | `param_utbl` | Notes |
|-----------|--------------|-------|
| Standalone hot magma ocean with grey-body BC | `true` | Resolves the temperature drop across the unresolved UTBL. |
| Coupled CHILI runs (PROTEUS) | `false` | The atmosphere module supplies a flux directly via `outer_boundary_condition = 4`; the UTBL would be applied twice. |
| Prescribed-flux or prescribed-temperature BC modes (`4`, `5`) | `false` | The parameterisation is only consulted in the grey-body branch. |
| SPIDER-parity diagnostic runs | `true` | Required for bit-parity against SPIDER's UTBL implementation. |

The parser default is `param_utbl = false` (see `src/aragog/parser.py`). Set it explicitly to `true` only in the standalone grey-body case.

## How it works

The grey-body radiator emits at the surface radiating temperature $T_\mathrm{surf}$, not at the interior staggered-node temperature $T_\mathrm{int}$. Bower et al. (2018, Eq. 18) parameterise the temperature drop across the UTBL as

$$
\Delta T = b\, T_\mathrm{surf}^3
\qquad\Longrightarrow\qquad
T_\mathrm{int} = T_\mathrm{surf} + b\, T_\mathrm{surf}^3,
$$

where $b$ has units of $\mathrm{K^{-2}}$ and is supplied by `param_utbl_const`. Rearranged into standard cubic form,

$$
T_\mathrm{surf}^3 + \frac{1}{b}\,T_\mathrm{surf} - \frac{T_\mathrm{int}}{b} = 0,
$$

which has one real root by Cardano's formula:

$$
T_\mathrm{surf} = \sqrt[3]{-q/2 + \sqrt{q^2/4 + p^3/27}} + \sqrt[3]{-q/2 - \sqrt{q^2/4 + p^3/27}},
$$

with $p = 1/b$, $q = -T_\mathrm{int}/b$. The result is always $T_\mathrm{surf} < T_\mathrm{int}$, so the radiator emits less flux than the interior temperature alone would imply.

When `param_utbl = false`, the parser zeroes `param_utbl_const` and the surface flux uses $T_\mathrm{surf} = T_\mathrm{int}$ directly.

## Picking `param_utbl_const`

The default value `param_utbl_const = 1.0e-7` $\mathrm{K^{-2}}$ matches the SPIDER standalone reference for Earth-mass magma oceans. The parameter trades off as follows:

- Larger $b$: bigger temperature drop across the UTBL at fixed $T_\mathrm{int}$; the radiator is colder, so the planet cools more slowly.
- Smaller $b$: the UTBL drop shrinks; in the limit $b \to 0$ the parameterisation collapses to $T_\mathrm{surf} = T_\mathrm{int}$ and the result agrees with `param_utbl = false`.

For super-Earth or sub-Earth standalone runs, calibrate $b$ against a higher-resolution surface-layer simulation; the default is not validated outside the 1 $M_\oplus$ regime.

## Configuration example

```toml
[boundary_conditions]
outer_boundary_condition = 1      # grey-body atmosphere (UTBL applies here only)
outer_boundary_value     = 280.0  # W/m^2 reference flux
emissivity               = 1.0
equilibrium_temperature  = 273.0
param_utbl               = true
param_utbl_const         = 1.0e-7
```

## Common pitfalls

- **Enabling `param_utbl` with PROTEUS coupling**: in coupled runs the atmosphere module supplies $F_\mathrm{atm}$ and the boundary mode is `4` (prescribed flux). The UTBL parameterisation is bypassed silently in modes `4` and `5`; turning it on does no harm but also does nothing.
- **Negative `param_utbl_const`**: not validated by the parser, but produces non-physical solutions ($T_\mathrm{surf} > T_\mathrm{int}$) when the cubic discriminant flips sign.
- **Comparing UTBL on / off across runs**: the diagnostic `T_magma` returned by `SolverOutput` is the topmost basic-node interior temperature, not the UTBL-corrected radiator temperature. To extract $T_\mathrm{surf}$ for diagnostics, recompute it from `T_magma` and `param_utbl_const` using the same Cardano formula.

## References

- Bower, D.J., Sanan, P., & Wolf, A.S. (2018). *Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets*. **Physics of the Earth and Planetary Interiors**, 274, 49 to 62. Eq. 18 derives the UTBL parameterisation.

## Cross-references

- [Configuration: `[boundary_conditions]`](configuration.md#boundary_conditions): the schema rows for `param_utbl` and `param_utbl_const`.
- [Heat-transport explainer](../Explanations/heat_transport.md): grey-body BC alongside the prescribed-flux and prescribed-temperature modes.
