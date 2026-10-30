# Why `phi_step_cap` is a SUNDIALS root function

The `phi_step_cap` parameter limits the per-call change in the mass-weighted mean melt fraction $\Phi_\mathrm{global}$. The user-facing description and configuration recipe are in [How-to: phi_step_cap](../How-to/phi-step-cap.md). This page covers the *design choice*: why a SUNDIALS root function is the right tool for this and what alternatives were considered.

## The problem

CVODE's adaptive step selection is based on local truncation error in the state vector. The state vector for Aragog is the entropy profile (plus the extra states of the `core_bc` mode: a single dS/dr value at the CMB in the default `energy_balance`). The step-size controller has no notion of melt fraction.

Through the rheological transition this matters. A single CVODE step that an LTE controller considers acceptable can take the mantle from $\Phi_\mathrm{global} = 0.45$ to $\Phi_\mathrm{global} = 0.10$ in one shot. Locally the truncation error in $S(r)$ is fine; globally the planet has just transitioned through the rheological front in a way that:

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

## How `phase_boundary_cap = 'rate'` segments steps through the solidus

How does the rate-based phase-boundary step controller (`phase_boundary_cap = 'rate'`) choose the maximum CVODE step size as a mantle cools through the liquidus and solidus, and why does it relax once every cell is below the solidus?

While `phi_step_cap` limits the global melt-fraction change per `solve()` call, `phase_boundary_cap = 'rate'` splits a single CVODE call into segments whenever any cell sits near or inside the melting interval. At the start of each segment, `_rate_phase_boundary_max_step` (`src/aragog/solver/entropy_solver.py`) sets `max_step` to one-tenth of the shortest estimated boundary-crossing time over all staggered cells. Once every cell has cooled to or below the solidus, the controller removes the $100\ \mathrm{yr}$ upper step ceiling and tightens the segment tolerances so long solid-state cooling calls can advance without stalling.

At each staggered cell $i$, let $S_i$ be the specific entropy in $\mathrm{J\,kg^{-1}\,K^{-1}}$, $\dot{S}_i = \mathrm{d}S_i/\mathrm{d}t$ be the entropy rate of change in $\mathrm{J\,kg^{-1}\,K^{-1}\,yr^{-1}}$, $m_i$ be the cell mass in $\mathrm{kg}$, and $S_{\mathrm{sol},i}$ and $S_{\mathrm{liq},i}$ be the solidus and liquidus entropies in $\mathrm{J\,kg^{-1}\,K^{-1}}$. The stiff-zone half-width $\delta$ around each phase boundary is

$$
\delta = \max\!\left(3w \max_j(S_{\mathrm{liq},j} - S_{\mathrm{sol},j}),\; 10\ \mathrm{J\,kg^{-1}\,K^{-1}}\right),
$$

where $w = \mathtt{matprop\_smooth\_width}$ is the dimensionless material-property smoothing width (`src/aragog/parser.py`).

To prevent a single stagnant cell from dividing by zero, the controller defines a mass-weighted rate floor $r_{\mathrm{floor}}$ in $\mathrm{J\,kg^{-1}\,K^{-1}\,yr^{-1}}$ equal to the mass-weighted mean of $|\dot{S}_i|$ over the mantle. The effective rate at cell $i$ is

$$
r_{\mathrm{used},i} = \max(|\dot{S}_i|,\; r_{\mathrm{floor}}).
$$

For cells outside the stiff zone ($d_i \ge \delta$, where $d_i = \min(|S_i - S_{\mathrm{liq},i}|, |S_i - S_{\mathrm{sol},i}|)$ is the entropy distance to the nearer phase boundary in $\mathrm{J\,kg^{-1}\,K^{-1}}$), the controller measures the distance $d_{\mathrm{ahead},i}$ to the next phase boundary in the direction of $\dot{S}_i$.

The progress root of `_PhaseBoundarySegmentRoot` (`src/aragog/solver/entropy_solver.py`) ends a segment once a moving cell has moved by $\mathrm{reach}_i = \max(0.5\,d_i,\; 0.5\,\delta)$; the stiff-zone entry and exit roots can end it earlier. When a cooling cell steps through a stiff-zone entry root at $d_i = \delta$ and two consecutive progress roots of $0.5\,\delta$ each, it lands directly on the phase boundary ($d_i \approx 0$). To match the progress-root stride $\mathrm{reach}_i \ge 0.5\,\delta$ and prevent $d_i \approx 0$ from pinning `max_step` to the $1\ \mathrm{yr}$ lower bound while the next root is $0.5\,\delta$ away, the controller uses, for a cell inside the stiff zone ($d_i < \delta$) or a cell above the solidus ($S_i > S_{\mathrm{sol},i}$) that moves slower than $r_{\mathrm{floor}}$, the effective stiff-zone distance

$$
d_{\mathrm{eff},i} = \max(d_i,\; 0.5\,\delta),
$$

and computes the cell crossing time $t_{c,i} = d_{\mathrm{eff},i} / r_{\mathrm{used},i}$ in years.

The segment step cap is then ten percent of the minimum crossing time over all cells, clipped between a lower bound $t_{\min} = 1\ \mathrm{yr}$ and an upper bound $t_{\max}$:

$$
\Delta t_{\max} = \mathrm{clip}\!\left(0.1 \min_i t_{c,i},\; t_{\min},\; t_{\max}\right).
$$

As a worked example of `_rate_phase_boundary_max_step` (`src/aragog/solver/entropy_solver.py`) with one cell, consider a cell with solidus entropy $S_{\mathrm{sol}} = 1000.0\ \mathrm{J\,kg^{-1}\,K^{-1}}$, stiff-zone half-width $\delta = 50.0\ \mathrm{J\,kg^{-1}\,K^{-1}}$, boundary distance $d_i = 2.4e-11\ \mathrm{J\,kg^{-1}\,K^{-1}}$ after landing on the solidus, and cooling rate $\dot{S}_i = -0.05\ \mathrm{J\,kg^{-1}\,K^{-1}\,yr^{-1}}$. The effective distance is $d_{\mathrm{eff},i} = \max(2.4e-11,\; 25.0) = 25.0\ \mathrm{J\,kg^{-1}\,K^{-1}}$, giving $\Delta t_{\max} = 0.1 \times 25.0 / 0.05 = 50.0\ \mathrm{yr}$ instead of collapsing to the $1.0\ \mathrm{yr}$ floor. Once that cell cools to $S_i = 980.0\ \mathrm{J\,kg^{-1}\,K^{-1}}$ (below the solidus inside the stiff zone) at $\dot{S}_i = -0.005\ \mathrm{J\,kg^{-1}\,K^{-1}\,yr^{-1}}$, the upper bound $t_{\max}$ relaxes to $\infty$ and $\Delta t_{\max} = 0.1 \times 25.0 / 0.005 = 500.0\ \mathrm{yr}$; once it cools below $S_{\mathrm{sol}} - \delta = 950.0\ \mathrm{J\,kg^{-1}\,K^{-1}}$ ($S_i = 940.0\ \mathrm{J\,kg^{-1}\,K^{-1}}$), no phase boundary lies ahead ($t_{c,i} = \infty$) and $\Delta t_{\max} = \infty$.

The controller enforces four bounds and regime transitions:

- In a partially molten or super-solidus mantle ($\max_i(S_i - S_{\mathrm{sol},i}) > 0$), the segment upper bound is $t_{\max} = 100.0\ \mathrm{yr}$ and the lower bound is $t_{\min} = 1.0\ \mathrm{yr}$, with CVODE using the caller's `rtol` and `atol`.
- In a fully sub-solidus mantle ($\max_i(S_i - S_{\mathrm{sol},i}) \le 0$), no phase boundary remains in the cooling direction, so the upper bound relaxes to $t_{\max} = \infty$ (`np.inf`). A cell within $\delta$ of a boundary that heats ($\dot{S}_i > 0$) still has the solidus ahead, so a segment that starts with such a cell keeps $t_{\max} = 100.0\ \mathrm{yr}$. In every segment that starts with $\max_i(S_i - S_{\mathrm{sol},i}) \le 0$, `EntropySolver._solve_cvode_segments` tightens the segment relative tolerance to $\min(\mathtt{rtol},\; 10^{-9})$ and the physical absolute tolerance to at most $10^{-8}$ in the unit of each state entry ($\mathrm{J\,kg^{-1}\,K^{-1}}$ for entropy, $\mathrm{K}$ for a temperature; `atol = np.minimum(caller_atol, 1e-8 / state_scale)`). A tighter caller tolerance is kept. A looser one, also from a retry that raises `rtol` or the absolute-tolerance scale, does not loosen these segments. With these tolerances the sub-solidus controller case V3-R4 on the CI test table in [Numerical verification](../Validation/numerics.md#phase-boundary-rate-controller-verification) stays inside $1.3 \times 10^{-4}\ \mathrm{K}$ without the $100\ \mathrm{yr}$ ceiling.
- When `phase_boundary_cap = 'rate'` is used with a loose caller tolerance (`rtol > 1e-7`), `EntropySolver` logs one `WARNING` because loose local-truncation-error control degrades segment-boundary accuracy.
- When `rate` is selected on non-CVODE integrators (`Radau`, `BDF`) or `core_bc = 'gradient'`, the solver falls back to the $1.0\ \mathrm{yr}$ fixed step cap and logs one message (`INFO` when `rate` is the default, `WARNING` when explicitly configured).

**Where this is checked.** The controller and its accuracy bounds are checked in `tests/test_phase_boundary_cap.py` and in [Numerical verification](../Validation/numerics.md).

### What the default trades

`rate` is the default because it keeps long runs from stalling near the solidus.
Without the distance floor, a cell that starts a segment almost on its solidus limits the step to $1\ \mathrm{yr}$ for as long as it stays there.
In the 100-node `rheo_on` validation case, one such interval near $96\ \mathrm{Myr}$ takes 459 767 steps with $1\ \mathrm{yr}$ steps and 4598 steps with the floor.
The price is accuracy close to the solidus.
In the sub-solidus check on the CI test table, the run takes 1031 steps with the rules and 3688 without them; its error against a run with $1\ \mathrm{yr}$ steps is $7.54 \times 10^{-5}\ \mathrm{K}$ with the rules and $2.70 \times 10^{-5}\ \mathrm{K}$ without them, inside the bound of $1.3 \times 10^{-4}\ \mathrm{K}$.
In the slow-landing check the run takes 224 steps with the rules and 272 without them, for an error of $1.16 \times 10^{-5}$ against $1.06 \times 10^{-5}\ \mathrm{K}$.
In the 96 Myr interval the end temperature at rtol $10^{-9}$ differs by up to $8.4 \times 10^{-4}\ \mathrm{K}$ from the run with $1\ \mathrm{yr}$ steps; at rtol $10^{-10}$ to $10^{-12}$ the difference is at most $4.3 \times 10^{-6}\ \mathrm{K}$.
In the coupled case that crosses the solidus, `T_magma` ends $1.0 \times 10^{-3}\ \mathrm{K}$ from a strict reference, against $7.5 \times 10^{-4}\ \mathrm{K}$ without the rules near and below the solidus.
There the run also takes 9.7 % more steps.
[Numerical verification](../Validation/numerics.md#full-solidification-without-solver-stalling) states both as known limits ([coupled case through the solidus](../Validation/numerics.md#coupled-planetary-evolution-crossing-the-solidus-into-the-solid-regime)).

## Cross-references

- [How-to: phi_step_cap](../How-to/phi-step-cap.md): config recipe and tuning advice.
- [Heat transport](heat_transport.md): the mushy-zone smoothing that motivates the cap.
- [CVODE and JAX](cvode_jax.md): the integrator family the rootfn is registered with.
- [Energy equation](energy_equation.md#phase-aware-step-size-control): phase-boundary step-size control and accuracy table.
