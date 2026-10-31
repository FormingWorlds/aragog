# CVODE with JAX-derived Jacobians

The default integrator in Aragog is SUNDIALS CVODE via [`scikits.odes`](https://scikits-odes.readthedocs.io/), the same modified-Newton, cached-Jacobian solver SPIDER uses. CVODE is paired by default with a JAX-derived analytic Jacobian, built by tracing the pure-functional flux computation in `aragog.jax.phase` with `jax.jacrev`.

This document explains *why* CVODE+JAX is the production default, *how* the factory pattern works, and *when* to fall back to the scipy paths.

## Why CVODE

The mantle ODE is stiff almost everywhere. The mushy band (around the rheological transition at melt fraction $\phi \approx 0.4$) sets the dominant timescale; conductive cells far from it sit at very different timescales. A modified-Newton solver with a cached Jacobian converges in 1-2 iterations on each step, while a finite-difference reconstruction would re-evaluate the RHS $O(N)$ times per step.

CVODE applies a frozen-Jacobian strategy: the Jacobian is reused across multiple steps until convergence degrades, at which point it is rebuilt. This is the same strategy SPIDER uses through PETSc TS. With a properly factored Jacobian a 1 $M_\oplus$ Earth-mantle run takes minutes rather than hours.

When `scikits.odes` is not installed at runtime, the solver falls back to scipy `Radau` and emits a warning at solve time. scipy `BDF` is also selectable via `solver.solver_method = "bdf"`; both scipy paths are slower than CVODE but work on platforms where the SUNDIALS bindings are unavailable.

## Why JAX

Without JAX, CVODE rebuilds its Jacobian by finite-differencing the RHS column by column. The cost is $O(N^2)$ RHS calls per Jacobian and the resulting matrix has finite-difference noise that hurts Newton convergence near the rheological transition.

The JAX path replaces that with one `jax.jacrev` backward pass over the traced flux computation. A single backward pass produces the full analytic Jacobian, exact to machine precision, in time comparable to a few RHS evaluations. Coupled PROTEUS runs that benchmark at 50 minutes with FD Jacobians complete in 5-10 minutes with JAX.

## How the factory plumbing works

PROTEUS registers a JAX factory on the solver instance before calling `solve()`:

```python
from aragog import EntropySolver
from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

solver = EntropySolver.from_file(config_path)
solver.initialize()
solver.set_initial_entropy(...)


def factory(scales, core_bc_mode):
    rhs, jac, _info = build_jax_rhs_and_jacobian(
        eos_jax, phase_params, mesh_arrays, boundary_params, heating, scales,
        core_bc_mode=core_bc_mode,
        radio_isotope_params=radio,
        core_module_budget=getattr(solver, '_core_module_budget', None),
        core_module_q_radio=float(getattr(solver, '_core_module_q_radio', 0.0)),
        core_module_ra_crit_cmb=getattr(solver, '_core_module_ra_crit_cmb', None),
    )
    return rhs, jac


solver.set_jax_cvode_factory(factory)
result = solver.solve()
```

`eos_jax`, `phase_params`, `mesh_arrays` and `boundary_params` are the JAX pytrees of the EOS tables, the phase parameters, the mesh and the boundary conditions (`EntropyEOS_JAX`, `PhaseParams`, `MeshArrays`, `BoundaryParams`), `heating` is the internal heating per cell that does not change with time [W/kg] (tidal), and `radio` is the 5-tuple `(heat_prod, abundance, concentration, t0_years, half_life_years)` of the isotopes, or `()`, so that the trace evaluates the radiogenic heating at the integrator time; the 3 `core_module` arguments are needed only for `core_bc = "core_module"`. The PROTEUS wrapper rebuilds the mesh and boundary pytrees from the live solver on every call. `solve()` calls `factory(scales, core_bc_mode)` and hands the pair `(rhs, jac)` to CVODE; `build_jax_rhs_and_jacobian` also returns a dict of diagnostics. CVODE calls `rhs(t, y, ydot)` and `jac(t, y, fy, J)` in the scikits.odes signatures, on nondimensional time and state. Both callables are JIT-compiled on first call; subsequent calls hit the compiled binary directly.

When `use_jax_jacobian = true` but no factory is registered (e.g. standalone Aragog tests), the solver silently falls back to CVODE's default finite-difference Jacobian. This is intentional: it means the standalone test suite does not depend on JAX being installed. A registered factory that raises `UnsupportedCoreBCError` (`core_bc` `bower2018` or `gradient`, which have no JAX right-hand side) gives the same fallback after a warning; any other factory error stops the solve.

## When to opt out

Use `use_jax_jacobian = false` when:

- Debugging an integrator divergence and you want a numerical Jacobian as a sanity reference.
- Reproducing pre-JAX runs bit-for-bit (the FD path is the historical default for older tags).
- Running on a platform where JAX cannot be installed.

Use `solver_method = "radau"` or `"bdf"` when:

- `scikits.odes` cannot be installed (uncommon, but the SUNDIALS C bindings sometimes fail to build on rolling-release Linux distributions).
- Bisecting a regression: scipy paths are simpler to instrument and have fewer hidden caches.
- Running short pure-numpy unit tests where CVODE setup overhead dominates.

## Verification

The JAX RHS must produce the same numerical result as the numpy `EntropyState.update` to floating-point precision. The verification suite asserts this for every public flux contribution; see [first-principles verification](../Validation/numerics.md) for the parity tests and [code architecture](code_architecture.md) for the file-level layout.
