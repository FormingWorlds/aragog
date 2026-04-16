# Option Z: JAX RHS + analytic Jacobian via CVODE

Branch: `tl/z-analytic-jacobian` (worktree at `/Users/timlichtenberg/git/aragog-Z/` and `/Users/timlichtenberg/git/PROTEUS-Z/`).

## What's done so far (2026-04-16 evening session)

Two new modules in `src/aragog/solver/`:

- **`cvode_jax.py`**: `build_jax_rhs_and_jacobian()` builds CVODE-compatible RHS + Jacobian functions backed by JAX, including the nondim transformation. Plus `verify_jax_vs_numpy_rhs()` for the gating parity check.
- **`jacobian_jax.py`**: earlier scaffolding for "JAX-Jacobian-on-numpy-RHS" approach (deprecated in favor of full JAX path).

The JAX physics already exists in `aragog/jax/`:
- `solver.py` has `dSdt(t, S, args)` — JAX-traceable RHS, signature matches what we need
- `phase.py` has `PhaseParams`, `MeshArrays`, EOS evaluation
- `eos.py` has `EntropyEOS_JAX` with bilinear interpolation in JAX

The PROTEUS wrapper in `proteus/interior_energetics/aragog_jax.py` already builds all JAX components from PROTEUS config — we can reuse this construction code.

## What remains for a working Z

### Z.1: Verify JAX dSdt matches numpy dSdt (CRITICAL pre-flight)

The JAX dSdt and numpy `_dSdt_single` are independent code paths. They must produce the same physical RHS to within ~1e-10 relative error, otherwise CVODE's Newton iteration will fail to converge with a Jacobian that doesn't match its RHS.

**Implementation**:
1. Pick a real (t, S) snapshot from a CHILI run (e.g. row 50 of chili_a_tightatol)
2. Initialize the numpy EntropySolver with the CHILI config
3. Initialize the JAX components via the existing `aragog_jax.py` builder
4. Call both `solver._dSdt_single(t, S)` and `jax_dsdt(t, S, args)`
5. Compute relative error per component
6. If any |rel_err| > 1e-8, identify which physics term diverges (likely Jgrav, Jmix, or MLT smoothing)
7. Fix the JAX path until parity is achieved

**Likely issues to find**:
- Phase boundary smoothing (cubic Hermite vs tanh) — production now uses tanh; JAX may still default to cubic Hermite
- MLT blend width — production uses 0.01*RE_CRIT (post 9742619 fix); JAX may have older 0.2*RE_CRIT
- Dilatation heating — production has it off; JAX path has its own setting
- Per-component flux diagnostics — JAX path may not populate them

### Z.2: Wire JAX RHS into CVODE solver

Modify `entropy_solver.py:_solve_cvode` to optionally use the JAX RHS:

```python
solver_method = getattr(self.parameters.energy, 'solver_method', 'cvode')
if solver_method == 'cvode_jax':
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
    # (need access to JAX components; either build here or pass in)
    rhs_fn, jacfn, info = build_jax_rhs_and_jacobian(
        eos_jax, phase_params, mesh_arrays, bc_jax, heating,
        state_scale, rhs_scale, t_ref,
    )
    cvode_options['jacfn'] = jacfn
else:
    # existing numpy RHS path
    rhs_fn = ...
```

The `build_jax_rhs_and_jacobian` signature in `cvode_jax.py` is already designed for this.

**Open question**: where do the JAX components (eos_jax, phase_params, mesh_arrays, bc_jax) get constructed? Options:

(a) Construct lazily inside `_solve_cvode` from `self.parameters` and `self.evaluator.mesh`. Adds construction cost on first call.
(b) Construct in `EntropySolver.__init__` or similar, store as attributes. Need PROTEUS to pass the config through.
(c) Have PROTEUS's `aragog_jax.py` builder build them and attach them to `interior_o`, then pass them in via setter.

Option (c) is cleanest because aragog_jax.py is already doing this construction.

### Z.3: Handle energy_balance N+1 state

Currently the JAX path only models the N-element entropy state. The energy_balance core_bc adds dSdr_cmb as state[N], with its own closure equation.

**Two options**:
- (A) Restrict Z to quasi_steady mode for now. Document limitation.
- (B) Extend JAX path to handle N+1 state. Need to:
  1. Define an analytic dSdr_cmb closure in JAX that mirrors the numpy implementation in `entropy_solver.py:_energy_balance_rhs_per_s`
  2. Extend the state vector handling
  3. Re-verify parity

For a first verification run, do (A). Plan (B) as Z.6 follow-up.

### Z.4: Test on CHILI

Once Z.1-Z.3 done, run a CHILI verification with `solver_method='cvode_jax'`:
- Compare endpoint to chili_tier3 / chili_a_tightatol (T_core ~4145-4148 K, Phi=0.05)
- Compare per-step wall time to numpy/CVODE baseline
- Count CVODE failures and retries

Expected outcome:
- Same endpoint (within bit-parity if JAX RHS matches numpy exactly)
- 2-5x faster Aragog solve per step (JIT-compiled JAX + analytic Jacobian)
- Net wall-time gain ~5-15% (AGNI dominates)
- 0 retries needed (analytic Jacobian eliminates FD-noise-induced Newton failures)

### Z.5: Performance benchmarking

Per-step wall time, RHS call count, Jacobian update count vs numpy/CVODE baseline.

### Z.6: Energy_balance support (follow-up)

Extend JAX path to handle N+1 state. Needed before Z can replace numpy/CVODE as the production default (since energy_balance is the production core_bc).

## Why this architecture is the right call (vs alternatives)

**Alternative 1: JAX Jacobian on numpy RHS** (the original sketch)
- Risk: JAX Jacobian and numpy RHS may differ in physics → CVODE Newton fails
- We'd need to verify two RHS implementations match anyway
- Benefit smaller: only saves the FD Jacobian cost; numpy RHS stays slow

**Alternative 2: Stick with diffrax for JAX path**
- Existing JAX path already uses diffrax, but it's much slower than CVODE for this problem (per the existing notes in entropy_solver.py)
- Doesn't get us CVODE's robustness (max_steps, retry-friendly structure, etc.)

**Alternative 3 (chosen): JAX RHS + JAX Jacobian + CVODE solver**
- Single source of truth for physics (JAX)
- Free analytic Jacobian via jax.jacrev
- Keep all CVODE robustness features (max_steps, BDF order, retry ladder integration, sanity check)
- JIT-compiled RHS faster than numpy
- Diffrax stays available for differentiable simulations (sensitivity analysis, optimization) but isn't the production solver

## Estimated remaining effort

- Z.1 verify parity: 4-8 hours
- Z.2 wire RHS+Jacobian: 2-4 hours
- Z.3 (option A) limit to quasi_steady: 1 hour
- Z.4 CHILI test: 1 hour active + several hours wall time
- Z.5 benchmarking: 2 hours
- Z.6 energy_balance support: 4-8 hours

**Total: 1.5-3 days realistic effort.**

## Files modified so far (this session)

- New: `aragog-Z/src/aragog/solver/cvode_jax.py` (CVODE+JAX integration)
- New: `aragog-Z/src/aragog/solver/jacobian_jax.py` (deprecated; superseded by cvode_jax.py)
- New: `aragog-Z/OPTION_Z_PLAN.md` (this file)

Nothing wired into entropy_solver.py yet. The next session should start with Z.1 (parity verification).
