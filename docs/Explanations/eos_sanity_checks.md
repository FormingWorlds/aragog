# EOS sanity checks at load time

When `EntropySolver.initialize()` runs it scans the loaded EOS tables once and warns if any quantity sits close to a runtime floor. The motivation is to surface a misconfigured or partially-loaded EOS *before* the integrator starts, so that a silently biased flux contribution is not blamed on the solver later.

This page explains what the checks are, why they exist, and how to interpret a warning.

## Two runtime floors

The hot-path numpy and JAX implementations clamp two quantities from below to keep the flux assembly numerically well-defined.

- **Heat capacity floor**: `c_p` is clamped at $100\;\mathrm{J\,kg^{-1}\,K^{-1}}$ before division. The conduction superadiabatic term reads $-k\,\frac{T}{c_p}\,\partial S/\partial r$, so a stray $c_p \to 0^+$ in a corrupt or partially-loaded table would otherwise blow up the conductive flux. A production MgSiO$_3$ P-S table stays $> 1000\;\mathrm{J\,kg^{-1}\,K^{-1}}$ everywhere.
- **Phase-boundary spread floor**: $\Delta S \equiv S_\mathrm{liq}(P) - S_\mathrm{sol}(P)$ is clamped at $1\;\mathrm{J\,kg^{-1}\,K^{-1}}$ before being used as a denominator in the two-phase fraction $\phi = (S - S_\mathrm{sol}) / \Delta S$. A vanishing $\Delta S$ would push $\phi$ unbounded; the floor makes the lever rule degenerate gracefully to a single-phase lookup.

These floors are *guards*, not physical assumptions. They should never be the binding constraint in a well-formed table.

## What `_check_eos_floors` does

`EntropySolver._check_eos_floors` runs once at the end of `initialize()`. It:

1. Walks `eos._tables['heat_capacity_solid']` and `eos._tables['heat_capacity_melt']` and reports `min(c_p)` across both.
2. Walks the solidus and liquidus arrays, restricting to the *intersection* of their P ranges, and computes $\Delta S$ at every solidus pressure.
3. Emits one `WARNING`-level log line per offending quantity if either statistic sits within `2x` of the runtime floor (`c_p < 200`, $\Delta S < 10$).

The check is skipped under the `const_properties` analytic path (which never loads tables) and when `entropy_eos` is `None` (test stubs).

## Why a separate scan instead of relying on the runtime clamp

The clamp itself is silent: it just replaces a too-small value and continues. If the EOS table has a few corrupt cells in the high-P tail of the liquid phase, the integrator will run, the clamp will fire on every RHS call that touches those cells, and the conductive flux in that region will be biased by an unknown amount. Nothing in the standard log output points back at the EOS.

The load-time scan turns that silent bias into one upfront warning. The user sees:

```
EntropySolver EOS check: min Cp = 87.3 J/kg/K in the loaded P-S
tables, within 2x of the 100 J/kg/K runtime floor. Conduction
superadiabatic term will be biased upward where Cp clips. Verify
the EOS heat-capacity tables.
```

before any time-stepping starts. That is enough to:

- decide whether to abort and regenerate the table,
- correlate later artefacts in `aragog.log` (a flat conductive flux profile, an unphysical CMB residual) with a known-good or known-suspect EOS,
- distinguish a true physics question from a table-quality issue.

## Tuning the warn thresholds

The warn thresholds are intentionally `2x` and `10x` the clamp floors, not equal to them. A table that touches the floor exactly is already a problem; one that sits within $2\times$ is on a trajectory toward the problem and is worth investigating before it lands. Outside the warn band the check stays silent so a clean table does not produce noise.

The thresholds live in `_check_eos_floors` as local constants (`cp_warn_threshold = 200.0`, `ds_warn_threshold = 10.0`). They are deliberately not exposed as config: the floors themselves are runtime-internal, and exposing the warn thresholds invites users to silence warnings instead of fixing tables.

## Related runtime guards

A separate, narrower runtime guard fires inside `EntropyState.update` and `aragog.jax.phase` if a $\Delta S$ value drops below the integration-time floor *during* a solve (e.g. a P excursion outside the loaded grid). That guard is also documented in the source comments at those sites; it is independent of the load-time scan and serves as the last-line defence against a runtime regression.

For the broader EOS architecture and the table file format see [Pressure EOS](pressure_eos.md) and [Phase transitions](phase_transitions.md).
