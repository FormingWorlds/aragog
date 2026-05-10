# `aragog.eos`

The `aragog.eos` package provides the pressure-entropy equation of state used by the solver. The production path is a single phase-aware evaluator backed by SPIDER-format $(P, S)$ tables; there is no abstract evaluator protocol or single/mixed/composite hierarchy.

| Name | Role |
|------|------|
| `EntropyEOS` | P-S table loader and bilinear interpolator. Provides `temperature(P, S)`, `density(P, S)`, `melt_fraction(P, S)`, `solidus_entropy(P)`, `liquidus_entropy(P)`, `latent_heat(P)`, `solidus_entropy_dP(P)`, `liquidus_entropy_dP(P)`. |
| `EntropyPhaseEvaluator` | Wraps `EntropyEOS` with the SPIDER-parity two-stage phase blend, viscosity tanh transition, gravitational-separation velocity, and the per-cell property cache. |

For the file format expected by `EntropyEOS`, see [Reference: data](../data.md).

::: aragog.eos
