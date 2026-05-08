"""Energy source configuration."""

from __future__ import annotations

import logging

import attrs
import numpy as np
import numpy.typing as npt

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


@attrs.define
class EnergyConfig:
    """Physics toggle flags, heating parameters, and solver knobs.

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
    phi_step_cap : float
        Per-call mass-weighted ``|ΔΦ_global|`` cap (SUNDIALS root
        function). When > 0 and at least one staggered cell sits in
        or near the mushy band at ``solve()`` entry, CVODE returns
        at the exact time where the change first reaches this value.
        Default 0.0 (disabled); 0.05 is typical for production
        evolution of 1 M_E PALEOS-2phase mantles.
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
    kappah_floor: float = 10.0
    bottom_up_grav_sep: bool = True
    phase_smoothing: str = 'tanh'
    solver_method: str = 'cvode'
    use_jax_jacobian: bool = True
    phi_step_cap: float = 0.0
    tidal_array: npt.NDArray = attrs.Factory(lambda: np.array([0.0], dtype=float))
