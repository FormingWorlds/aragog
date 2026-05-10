# Why `phi_step_cap` is a SUNDIALS root function

The `phi_step_cap` knob limits the per-call change in the mass-weighted mean melt fraction $\Phi_\mathrm{global}$. The user-facing description and configuration recipe are in [How-to: phi_step_cap](../How-to/phi-step-cap.md). This page covers the *design choice*: why a SUNDIALS root function is the right tool for this and what alternatives were considered.

## The problem

CVODE's adaptive step selection is based on local truncation error in the state vector. The state vector for Aragog is the entropy profile (plus a single dS/dr value at the CMB in `energy_balance` mode). The step-size controller has no notion of melt fraction.

Across the rheological transition this matters. A single CVODE step that an LTE controller considers acceptable can take the mantle from $\Phi_\mathrm{global} = 0.45$ to $\Phi_\mathrm{global} = 0.10$ in one shot. Locally the truncation error in $S(r)$ is fine; globally the planet has just transitioned through the rheological front in a way that:

- the user cannot see in the helpfile (the transition is interior to the step),
- the next-step initial guess for the integrator is a mushy-zone solution that's significantly different from the previous step,
- and the PROTEUS outer loop (which compares interior cooling against atmospheric flux at end-of-step) sees a big jump it did not request.

The problem is not numerical; the truncation error is bounded. The problem is *resolution*: the integrator was free to skip past a physically meaningful state.

## Three candidate fixes

### 1. Tighten `atol` / `rtol`

Tightening tolerances drops the typical step size globally. It works but is wasteful: most of the integration is far from the rheological transition and gets none of the benefit. A 1 Myr CHILI run would slow by 5x or more.

### 2. Add a step-size cap (`max_step`)

CVODE supports `max_step` directly. But `max_step` is a constant in time; it has no notion of "the mantle is currently in the mushy zone". A cap tight enough to resolve the transition forces the same tightness on the smooth pre-transition phase.

### 3. SUNDIALS root function on $|\Delta\Phi_\mathrm{global}|$

CVODE's rootfn API lets the caller register a function $g(t, y)$ and asks CVODE to return at the time where $g$ first crosses zero. The return is *exact*: CVODE does its own Newton refinement to land at the zero-crossing to within a tolerance the caller specifies.

Aragog registers

$$
g(t, y) = \mathrm{cap} - |\Phi_\mathrm{global}(t, y) - \Phi_\mathrm{global}(t_\mathrm{start})|
$$

at every call into `solve()`. The integrator runs at its native step size until $g$ reaches zero, then returns control with `CV_ROOT_RETURN`. The next `solve()` call resumes from there. Cost: one rootfn evaluation per step, evaluated only when the gating decides the cap could plausibly fire. Benefit: the rest of the integration runs unimpeded.

The third option is what landed.

## Why mass-weighted, not volume-weighted

`Phi_global` in the helpfile is mass-weighted ($M_\mathrm{liquid} / M_\mathrm{mantle}$). The volume-weighted variant `Phi_global_vol` is also reported but not used by the cap. The choice is not arbitrary:

- Mass-weighted $\Phi_\mathrm{global}$ tracks how much *thermodynamic* state has changed (latent heat scales with mass, not volume).
- Volume-weighted $\Phi_\mathrm{global}$ is a geometry diagnostic (which fraction of the mantle volume is liquid).

A cap on $\Delta\Phi_\mathrm{global}$ (mass) tracks the energy budget; a cap on $\Delta\Phi_\mathrm{global,vol}$ would track the visual geometry. The integrator is bounded by energy resolution, not by visual geometry, so the mass-weighted choice is the right one.

## Why gating before registration

A pure-mantle (fully solid or fully molten) profile cannot fire the cap at all in the current step (the path through the mushy zone is far away). Registering the rootfn anyway means CVODE evaluates $g$ at every internal step, paying the cost (one $\Phi_\mathrm{global}$ evaluation per step) for nothing.

`EntropySolver.solve()` checks `near_liq` / `near_sol` / `in_mushy` before deciding to register the rootfn. If the profile is nowhere near the transition, the rootfn is not registered for this call; the solver runs at full speed. The gating uses the same phase-evaluator threshold logic as the flux assembly, so it is consistent with the rest of the runtime.

## Scipy-fallback equivalence

The two scipy paths (`Radau` / `BDF`) do not have CVODE's native rootfn API but they have `solve_ivp` events. Aragog registers the same $g(t, y)$ as a `scipy.integrate.solve_ivp` event with `terminal = True` and `direction = -1`. The behaviour is equivalent up to integrator class.

Both paths are tested in the verification suite; bit-parity is not enforced (the integrators are not bit-equivalent in general) but the cap-fire time agrees to within `atol`.

## Failure modes the cap does *not* fix

The cap caps the step, not the trajectory. A run that's already trapped in a hidden quasi-equilibrium will not be unstuck by adding a cap. Symptoms that are *not* the cap's job:

- `T_magma` byte-pinned at a constant for thousands of steps (consistent with a `prevent_warming = true` clamp; see [PROTEUS coupling](proteus_coupling.md)).
- Phi oscillating at the rheological transition with no monotone trend (consistent with a thermodynamic attractor; the cap will fire but neither prevent nor accelerate the oscillation).
- Numerical Jacobian noise near the rheological transition (use the JAX Jacobian; see [JAX Jacobian factory](jax_jacobian_factory.md)).

The cap is a step-resolution tool, not a state-correction tool. Match it to the symptom before reaching for it.

## Cross-references

- [How-to: phi_step_cap](../How-to/phi-step-cap.md): config recipe and tuning advice.
- [Heat transport](heat_transport.md): the mushy-zone smoothing that motivates the cap.
- [CVODE and JAX](cvode_jax.md): the integrator family the rootfn is registered with.
