# The JAX Jacobian factory pattern

Aragog's production integrator path is SUNDIALS CVODE with an analytic Jacobian built by JAX. The design choice that makes this practical is the **factory pattern**: the solver does not own the JAX trace itself but accepts a callable that *constructs* the (RHS, Jacobian) pair from the bound mesh, EOS, phase parameters, and boundary conditions at solve time.

This page explains why the factory is the unit of separation and what its consequences are.

## What the factory returns

`build_jax_rhs_and_jacobian` (in `aragog.solver.cvode_jax`) takes the runtime state of an `EntropySolver`:

- the staggered + basic mesh arrays,
- the JAX-friendly EOS (`EntropyEOS_JAX`),
- the bound `PhaseParams`,
- the bound `BoundaryParams`,
- the radiogenic + tidal heating closures,

and returns a tuple `(rhs, jac)`:

- `rhs(t, y)` evaluates $dS/dt$ for a given state,
- `jac(t, y)` evaluates $\partial(dS/dt)/\partial y$ via `jax.jacrev`.

Both callables are JAX `pure_callback`-wrapped JITted functions. CVODE calls them like any other RHS / Jacobian; it never sees the JAX layer directly.

## Why a factory and not a method

The Jacobian depends on more than just `(t, y)`. It depends on the EOS tables, the mesh geometry, the radiogenic heating closure, the boundary-condition mode, and the phase parameters. All of these are *constants* over a single `solve()` call but vary across calls (PROTEUS rebuilds the mesh on every Zalmoxis re-solve).

Three competing designs were considered:

1. **Make `EntropySolver` own the JAX trace.** The solver would need to import JAX at module load time, which would break installs that don't have `[jax]`. JAX is an optional extra precisely because we want users on platforms where JAX cannot be installed (rare but real) to still run the scipy paths.

2. **Build the trace lazily inside `solve()`.** The trace then has to walk into `self.parameters`, which is a `Parameters` dataclass with numpy arrays and Python floats. JAX cannot trace through `Parameters` directly; it would have to be unpacked at every call. The unpacking is non-trivial (e.g. `mesh.basic_radii`, `eos._tables['temperature_solid']`) and would couple the solver class to the JAX representation.

3. **Factory pattern (the chosen design).** The solver exposes a `set_jax_cvode_factory` hook. PROTEUS (or any caller that has JAX installed) imports `build_jax_rhs_and_jacobian` and registers it once. Inside `solve()`, the solver calls the factory with the already-unpacked mesh / EOS / params and gets back the two callables. The solver class never imports JAX. The factory function lives in a module that is only imported by callers who explicitly opt in.

The third option keeps the solver class JAX-agnostic and lets the factory be the only place that knows the trace shape. Refactors of the JAX layer (e.g. splitting `compute_fluxes` into smaller pieces, adding a new flux contribution) only touch `cvode_jax.py` and `aragog.jax.phase`; the solver is unaffected.

## What "registered before solve()" means

`EntropySolver.set_jax_cvode_factory(build_jax_rhs_and_jacobian)` stores the factory on the instance. When `solve()` runs with `solver_method = 'cvode'` and `use_jax_jacobian = true`, it:

1. Calls the factory with the bound runtime objects,
2. Hands the resulting `(rhs, jac)` to CVODE,
3. Lets CVODE drive the integration.

If the factory is not registered, the solver silently falls back to CVODE's built-in finite-difference Jacobian. This is intentional: it lets the standalone unit-test suite run without JAX installed, and it gives users on JAX-less platforms a working (slower) path.

The fallback is deliberate; an exception would make the JAX extra a hard dependency. The cost is that a misconfigured environment (JAX installed, factory not registered) silently runs at FD-Jacobian speed instead of failing loudly. Mitigations:

- The CLI emits a one-line `INFO` at solve time naming the active Jacobian path.
- `aragog --versions` lists JAX, so a bug report on slow integration can be diagnosed remotely.
- The PROTEUS wrapper always registers the factory; standalone users who care about speed read the docs.

## Bit-parity with the numpy path

Every flux contribution in `aragog.jax.phase` is constructed to match the numpy `EntropyState.update` to floating-point precision. The verification suite asserts equality on a per-component basis (conduction, convection, gravitational separation, mixing) for representative `(P, S, T, mesh)` tuples; see [first-principles verification](verification.md).

Bit-parity is the contract that lets the JAX path be the production default without breaking a regression that was tuned against numpy. If the numpy path is being used as a "ground truth" and the JAX trace diverges, the parity tests catch the drift before it lands.

## When the factory itself is the problem

A run that hits unexpected divergence under JAX but converges under FD points to a problem in the trace, not in CVODE. Diagnostic loop:

1. Set `use_jax_jacobian = false`, rerun. If it converges, the bug is in the trace.
2. Add a print of `jac(t, y)` at the failing step to the factory; compare against a numerical Jacobian computed by FD. The trace and FD must match to the tolerance of `jax.jacrev` rounding.
3. If they do not match, suspect a non-differentiable branch (a tanh switch with too-narrow width, a `where` over a NaN, a JAX-unfriendly clip).

For broader CVODE+JAX rationale and the full integrator dispatch decision tree see [CVODE and JAX](cvode_jax.md). For the actual flux assembly under each path see [Heat transport](heat_transport.md).
