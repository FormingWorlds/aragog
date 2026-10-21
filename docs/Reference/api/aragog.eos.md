# `aragog.eos`

The `aragog.eos` package provides the pressure-entropy equation of state used by the solver. The production path is a single phase-aware evaluator backed by SPIDER-format $(P, S)$ tables; there is no abstract evaluator protocol or single/mixed/composite hierarchy.

| Name | Role |
|------|------|
| `EntropyEOS` | P-S table loader and bilinear interpolator. Provides `temperature(P, S)`, `density(P, S)`, `melt_fraction(P, S)`, `solidus_entropy(P)`, `liquidus_entropy(P)`, `latent_heat(P)`, `solidus_entropy_dP(P)`, `liquidus_entropy_dP(P)`. |
| `EntropyPhaseEvaluator` | Wraps `EntropyEOS` with the SPIDER-parity two-stage phase blend, viscosity tanh transition, gravitational-separation velocity, and the per-cell property cache. |
| `mobility_function` | Three-regime permeability over porosity $F(\phi)$ [m²] that the gravitational-separation velocity uses, from the porosity and the grain size. |
| `consistency.residuals` | How far the P-S tables of one phase are from $(\partial T/\partial S)_P = T/c_p$, from the tabulated $(\partial T/\partial P)_S$, from $(\partial T/\partial P)_S = -(1/\rho)(\partial \ln\rho/\partial S)_P$ and from $(\partial T/\partial P)_S = \alpha T/(\rho c_p)$, as relative numbers. Slopes come from central differences on the table grid; a step that meets a missing cell gives none. |
| `consistency.check` | Tests each cell of those tables against a Richardson bound: the round-off floor of the stored values, at 9 significant digits at best, plus twice the $h^4$ term measured at the cell. The floor also covers the rounding of the stored $P$ and $S$ coordinates, taken as the departure of each coordinate from the uniform or geometric fit of its axis ($\epsilon |x|$ for an irregular axis). A missing value (NaN) leaves its cells untested; a stored 0 or inf is a value. Where all values an identity reads are present but its residual is not finite, the cell fails, so a column that is 0 or inf everywhere makes every identity that reads it fail; an identity is skipped only when no cell holds all values it reads, and the check raises when every identity is skipped or a table is missing. On a grid linear in $P$ the $(\partial T/\partial P)_S$ identity finds a $10^{-6}$ temperature fault only where $P$ is large against the step. |

For the file format expected by `EntropyEOS`, see [Reference: data](../data.md).

::: aragog.eos

::: aragog.eos.consistency
