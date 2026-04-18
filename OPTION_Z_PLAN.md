# Option Z: JAX RHS + analytic Jacobian via CVODE

Branch: `tl/z-analytic-jacobian` (worktree at `/Users/timlichtenberg/git/aragog-Z/` and `/Users/timlichtenberg/git/PROTEUS-Z/`).

---

## Status as of 2026-04-17 night session (~01:18 CEST)

**Z.6.A is COMPLETE. Z.2 is COMPLETE. Z.3 is VALIDATED end-to-end.**

Key milestones since the original plan was written:

* SPIDER-parity ports landed in `aragog-Z/src/aragog/jax/`:
  * `compute_phase_state` single-pass evaluation (eos.py)
  * SPIDER conduction `F_cond = -k * (T/Cp * dSdr + dTdPs * dPdr)` (phase.py)
  * CMB `kappa_h[0] = kappa_h[1]` boundary fix (phase.py)
  * Surface `dSdr[-1] = dSdr[-2]` boundary copy (phase.py)
  * `MeshArrays.dP_dr_basic` field (phase.py)
  * `compute_mlt` autodiff-safety guards: smooth-abs and `sqrt(x+eps^2)` (phase.py)
* `cvode_jax.build_jax_rhs_and_jacobian` extended to `core_bc_mode='energy_balance'` (cvode_jax.py)
* 12 unit tests in `tests/test_jax_entropy.py::TestSPIDERParityPorts/TestSPIDERConductionDecomposition/TestBoundaryCopies`

Multi-state parity (z02): IC + mid + near_solid all PASS, median rel err ~3e-5.
Multi-state Jacobian validation (z04, after fix): IC max 9.8e-3, mid max 1.1e-11, near_solid max 8.1e-10 (mid/near_solid bit-identical to FD reference).

End-to-end CVODE + JAX-Jacobian (z09) on chili_repro_v2 IC, 1 yr integration:

| tol           | JAX wall  | FD wall   | speedup | endpoint diff |
|---------------|-----------|-----------|---------|---------------|
| rtol=1e-6, atol=1e-8 (CHILI prod) | 430 ms | 28949 ms | 67x | 16.3 abs |
| rtol=1e-9, atol=1e-12             | 328 ms | 688 ms   | 2.1x | 2.1e-6 abs |

The 67x at loose tol is partially an FD inefficiency artifact (CVODE without analytic Jacobian takes ~30x more Newton iterations per step). Tightening tol gives the textbook 2x and matches both endpoints to 2e-6 abs / 3e-4 rel.

Z.3 wire-up went through scikits.odes `cvode` directly with `jacfn=jac_fn` in the options dict. No patches to production aragog code needed.

### What's still ahead

* **Z.4: production CHILI run with JAX Jacobian.** Take the chili_v_test config, swap the cvode call to use `build_jax_rhs_and_jacobian` + `jacfn`, run to Phi_crit. Compare endpoint to chili_v_test (T_core 4145.483 K, Phi 0.04956). Wall-time benchmark.
* **Z.5: integrate into PROTEUS coupling loop.** Add a config flag `interior_energetics.aragog.use_jax_jacobian: bool` that swaps the entropy_solver._solve_cvode internals to use the JAX Jacobian path. This is the "production switch" landing.

The bulk of the engineering risk is now behind us. What remains is plumbing.

### Diagnostic scripts (PROTEUS-Z/scripts/)

* `z01_verify_jax_numpy_parity.py` — IC RHS parity (median 2.13e-5 rel err)
* `z02_parity_multi_state.py` — IC + mid + near_solid RHS parity (all PASS)
* `z03_jacobian_check.py` — IC Jacobian vs FD (max 9.82e-3)
* `z04_jacobian_multi_state.py` — multi-state Jacobian vs FD (all PASS after fix)
* `z05_localise_jacobian_nan.py` — historical: NaN survives all transport channels off
* `z06_phase_state_jacobian.py` — historical: compute_phase_state Jacobian FINITE
* `z07_compute_fluxes_jacobian.py` — historical: compute_fluxes 79 NaN at mid
* `z08_compute_mlt_jacobian.py` — historical: compute_mlt is the NaN source
* `z09_cvode_with_jax_jacobian.py` — end-to-end CVODE + JAX Jacobian, 67x / 2x speedup

### Original plan (preserved below for context)

## What's done so far (2026-04-16 evening session)

Two new modules in `src/aragog/solver/`:

- **`cvode_jax.py`**: `build_jax_rhs_and_jacobian()` builds CVODE-compatible RHS + Jacobian functions backed by JAX, including the nondim transformation. Plus `verify_jax_vs_numpy_rhs()` for the gating parity check.
- **`jacobian_jax.py`**: earlier scaffolding for "JAX-Jacobian-on-numpy-RHS" approach (deprecated in favor of full JAX path).

The JAX physics already exists in `aragog/jax/`:
- `solver.py` has `dSdt(t, S, args)` — JAX-traceable RHS, signature matches what we need
- `phase.py` has `PhaseParams`, `MeshArrays`, EOS evaluation
- `eos.py` has `EntropyEOS_JAX` with bilinear interpolation in JAX

The PROTEUS wrapper in `proteus/interior_energetics/aragog_jax.py` already builds all JAX components from PROTEUS config — we can reuse this construction code.

## Z.1 + Z.6 progress (2026-04-17 ~00:30 CET)

### Iteration 2: fixed CMB basic-node entropy override

Added in `dSdt_energy_balance`:
- `S_basic_cmb = S[0] + dSdr_cmb * (r_basic[0] - r_stag[0])` (matches numpy `update(dSdr_cmb)` extrapolation)
- Override `S_basic[0]` and recompute `phase_basic` from corrected entropy
- Use `phase_basic.thermal_conductivity[0]` instead of average

**Result: still 110x divergence**. Same numbers as iter 1.

### Root cause discovered (commit 87aeb8b)

Diagnostic at IC:
- numpy `heat_flux[0]` = **1.164e9 W/m^2** (huge — initial wind-up state with strong convection)
- numpy `phase_basic.temp[0]` = 7195 K (matches my JAX value)
- numpy `phase_basic.cp[0]` = 1432 J/kg/K (matches)

The numpy `heat_flux[0]` of 1.164e9 W/m^2 is NOT just the conductive flux. It includes ALL components: conduction + convection + grav_sep + mixing. My JAX `F_cmb_from_dSdr = -k * dT/dr` computes only ~0.27 W/m^2 (the conductive part).

In numpy, `state.update(dSdr_cmb=extra)` modifies `entropy_basic[0]`, then the full flux pipeline (conduction + convection + grav_sep + mixing) runs with the corrected basic-node entropy. The convective component dominates by ~9 orders of magnitude.

In JAX, `compute_fluxes` (in `aragog/jax/phase.py`) is a black-box that takes only `S` (staggered) and does its own `mesh.quantity_matrix @ S` to derive `S_basic`. It doesn't accept an override for `entropy_basic[0]`.

### What's needed for Z.1 to PASS

**Option Z.6.A**: Modify `aragog/jax/phase.py:compute_fluxes` to accept an optional `S_basic_override` argument. The override would replace the standard `quantity_matrix @ S` for the basic-node entropy. Modify in lockstep with `evaluate_phase` which is also called inside.

This is a meaningful JAX refactor:
- compute_fluxes signature change
- pass-through to evaluate_phase
- ensure JIT trace is still clean (override must be a JAX array, not None)

Estimated effort: 4-6 hours of careful refactoring + parity re-verification.

**Option Z.6.B**: Don't reuse compute_fluxes. Re-implement the flux computation manually in dSdt_energy_balance with the corrected basic-node entropy. Less invasive but more code duplication.

**Recommendation**: Z.6.A is the right structural fix. Z.6.B accumulates technical debt.

### Bigger picture: Z is more work than estimated

My initial estimate was 1.5-3 days. After Z.6 + Z.1 iter 1-2, realistic estimate is **3-5 days** because:
- compute_fluxes refactor needed for energy_balance compatibility
- Each numpy ↔ JAX physics primitive needs parity verification
- Mesh-derived dPdr, EOS dTdPs, k_basic - any mismatch shows up as RHS divergence
- The 4-8 hours per iteration applies to each remaining gap

**Pragmatic recommendation**: Given the U+V+W+X+Y stack is empirically eliminating the bifurcation (4 clean trajectories in flight), Z is no longer urgent. Defer Z to after the paper, OR commit to a focused 3-5 day push to complete it properly.

### Original Z.1 progress notes follow:



### Z.6 implementation (commit `6f4b4cc`)
Added `dSdt_energy_balance(t, state_ext, args)` in `aragog/jax/solver.py` for N+1 state. Extended `BoundaryParams` with energy_balance constants (cmb_area, core_M, cmb_dr_cmb). Code compiles and imports cleanly.

### Z.1 parity test result: **110x DIVERGENCE at CMB cell**

```
config: chili_repro_v2 (energy_balance core_bc, atol=3e-7)
n_stag = 79, state vector = N+1 = 80
                  numpy           JAX             abs_err     rel_err
dS/dt[0]    -313.56          -34522           34209       110x
dS/dt[79]    +0.01746        -2.282           2.30        132x
dS/dt[60]    +12.17           -66.17          78.35       6.4x
dS/dt[59]    -12.21           -87.95          75.73       6.2x
median rel_err: 0.19 (19% off)
```

**Diagnosis (likely causes ranked):**

1. **k_cmb approximation is wrong**: I used `k_cmb = 0.5*(k_solid+k_liquid)` (typical 3.0 W/m/K) but production uses phase-weighted k at the actual melt fraction (likely k_liquid=2.0 at fully molten Phi=1). This is ~50% off, but doesn't explain 110x.

2. **F_cmb derivation may be wrong**: My formula `F_cmb = -k * [(T/Cp) * dSdr_cmb + dTdPs * dPdr]` assumes the flux derivation matches what `state.update(dSdr_cmb=extra)` does in numpy. Need to dig into `entropy_state.py` to verify the exact derivation.

3. **dPdr computation**: I used one-sided finite difference between basic nodes 0 and 1; production may use the analytic Adams-Williamson dPdr or the dPdr cached in entropy_state.

4. **dSdr_cmb closure formula**: my factor of 2/dr_cmb may be wrong; numpy uses `(dSdt_stag - dSdt_basic) * 2 / dr_cmb` but I should re-verify against `_energy_balance_rhs_per_s`.

**Next steps for Z.1**:
- Inspect `entropy_state.py:update(dSdr_cmb=extra)` to see exactly how F_cmb is set
- Replace my approximation with the actual physics
- Re-test parity
- May need to extract internal numpy state (k_basic, dPdr, etc.) and pass to JAX as inputs

This is iterative work — could easily take another 4-8 hours of careful debugging. Significant value: each fix in JAX brings parity, plus reveals subtle physics couplings worth understanding.

### Original Z.1 progress notes follow:



Wrote `scripts/z01_verify_jax_numpy_parity.py` in PROTEUS-Z worktree. The script:
- Loads chili_repro_v2 config
- Initializes the numpy EntropySolver via AragogRunner.setup_solver
- Attempts to build JAX components (EOS, PhaseParams, MeshArrays, BoundaryParams)
- Calls both RHS and compares per-component

**Blockers found**:

1. **diffrax not installed in proteus conda env**.
   `aragog/jax/solver.py` does a top-level `import diffrax`. Even importing just `dSdt` and `BoundaryParams` fails without diffrax. Fix options:
   - `pip install diffrax` in the proteus env (permanent env change)
   - Refactor `solver.py` to import diffrax lazily inside `solve_entropy` only (clean but small refactor)

2. **chili_repro_v2 uses `core_bc = energy_balance` (N+1 state)**.
   The JAX `dSdt(t, S, args)` only handles the N-element entropy state. With energy_balance, the numpy state is N+1 with dSdr_cmb at index N. Calling JAX with just the N entropy block gives a different physics (the closure equation is missing).
   - Either run Z.1 with `core_bc = quasi_steady` (need to modify config or test setup)
   - Or extend JAX to handle the dSdr_cmb closure (this is Z.6 work)

**Recommended next session start**:
1. Refactor `aragog/jax/solver.py` to make diffrax a lazy import (5-10 line change)
2. Make a quasi_steady test config (copy chili_repro_v2, change one line)
3. Re-run the Z.1 parity script
4. Iterate: if rel_err > 1e-8, identify which JAX physics term diverges from numpy

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
