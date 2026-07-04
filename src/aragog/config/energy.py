"""Energy source configuration."""

from __future__ import annotations

import logging

import attrs
import numpy as np
import numpy.typing as npt

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


@attrs.define
class EnergyConfig:
    r"""Physics toggle flags, heating parameters, and solver knobs.

    Defaults are aligned with the PROTEUS production path
    (SUNDIALS CVODE + JAX analytic Jacobian + tanh phase smoothing
    with the bottom-up gravitational-separation gate enabled and a
    phase-dependent eddy-diffusivity floor of 10 m^2/s).

    Parameters
    ----------
    conduction : bool
    convection : bool
    gravitational_separation : bool
    mixing : bool
    radionuclides : bool
    tidal : bool
    eddy_diffusivity_chemical : float
        Ratio kappa_c / kappa_h for chemical eddy diffusivity.
    eddy_diffusivity_thermal : float
        Multiplier on the raw thermal eddy diffusivity $\kappa_h$
        before the floor is applied. The SPIDER convention also
        permits negative values, in which case the diffusivity is
        pinned to ``|eddy_diffusivity_thermal|``. Default 1.0
        passes the MLT-derived value through unchanged.
    kappah_floor : float
        Phase-dependent eddy-diffusivity floor [m^2/s]. Clamps the
        eddy diffusivity from below by ``floor * f(phi)`` where
        ``f`` falls from 1 (liquid) to 0 (solid) across the
        rheological transition. Default 10.0 matches PROTEUS
        production; set 0.0 for textbook MLT.
    bottom_up_grav_sep : bool
        SPIDER-analogue bottom-up gate for the gravitational
        mass-flux: only allow melt/solid separation across an
        interface when the staggered cell below is non-pure. Default
        True; off only for regression tests against the pre-fix CMB
        drain at first crystallisation.
    phase_smoothing : str
        Phase-boundary smoothing scheme for J_grav and J_mix:
        ``'tanh'`` (SPIDER parity, production default) or
        ``'cubic_hermite'`` (parameter-free fallback for residual
        EOS-mismatch cases).
    solver_method : str
        ODE backend: ``'cvode'`` (SUNDIALS via scikits.odes,
        production default), ``'radau'`` (scipy implicit Runge-Kutta),
        or ``'bdf'`` (scipy BDF). When scikits.odes is not installed
        the solver falls back to scipy Radau and emits a warning.
    use_jax_jacobian : bool
        When True and ``solver_method='cvode'``, the integrator uses
        a JAX-derived analytic Jacobian (``jax.jacrev``) instead of
        CVODE's finite-difference Jacobian. Requires JAX and the
        ``aragog.jax`` module. Default True matches PROTEUS
        production; silently falls back to FD if no JAX factory was
        registered before ``solve()``.
    phi_step_cap : float or None
        Per-call mass-weighted ``|ΔΦ_global|`` cap (SUNDIALS root
        function). When positive and at least one staggered cell sits in
        or near the mushy band at ``solve()`` entry, CVODE returns
        at the exact time where the change first reaches this value.
        ``None`` (the default) disables the cap; any non-positive value
        disables it too. 0.05 is typical for production evolution of
        1 M_E PALEOS-2phase mantles.
    temperature_step_cap : float or None
        Per-call per-cell ``|ΔT|`` cap [K] (SUNDIALS root function),
        firing on the maximum single-cell temperature change since
        ``solve()`` entry. Bounds the core-temperature drop on the solid
        adiabat below the solidus, which the melt-fraction cap cannot see.
        ``None`` (the default) disables the cap.
    entropy_step_cap : float or None
        Per-call per-cell ``|ΔS|`` cap [J/kg/K] in the native solver
        variable; same role as the temperature cap without an EOS lookup
        in the root function. ``None`` (the default) disables the cap.
    phase_boundary_entropy_margin : float
        Proximity band [J/kg/K] within which a staggered cell counts as
        "near" a phase boundary, tightening the integrator ``max_step`` to
        1 yr so CVODE resolves the stiff RHS across the solidus/liquidus.
        This is a solver-accuracy control, not a physics threshold. At the
        default it reproduces the previous fixed band, so the converged
        trajectory is unchanged; lowering it can under-resolve a real phase
        crossing and shift the converged state by more than the nominal
        tolerance, because CVODE's local error control can accept an
        over-large step across the near-discontinuous two-phase RHS. Default
        200.0, a fraction of a typical silicate fusion entropy (S_liquidus -
        S_solidus), wide enough to arm the tighter stepping before a cell
        reaches the boundary yet narrow enough to leave the deep-solid
        thermal history on unrestricted steps. A non-finite or non-positive
        value falls back to the default.
    tidal_array : ndarray
        Tidal heating per unit mass [W/kg] at each layer.
    """

    conduction: bool
    convection: bool
    gravitational_separation: bool
    mixing: bool
    radionuclides: bool
    tidal: bool
    eddy_diffusivity_chemical: float = 1.0
    eddy_diffusivity_thermal: float = 1.0
    kappah_floor: float = 10.0
    bottom_up_grav_sep: bool = True
    phase_smoothing: str = 'tanh'
    solver_method: str = 'cvode'
    use_jax_jacobian: bool = True
    phi_step_cap: float | None = None
    temperature_step_cap: float | None = None
    entropy_step_cap: float | None = None
    phase_boundary_entropy_margin: float = 200.0
    tidal_array: npt.NDArray = attrs.Factory(lambda: np.array([0.0], dtype=float))
