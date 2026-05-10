"""Nondimensional scaling spec shared by the numpy and JAX RHS paths.

Single source of truth for the (state_scale, rhs_scale, t_ref) triplet
that scales physical-units state into the BDF integrator's O(1) work
space. Built once by the EntropySolver, consumed by both the
scipy/CVODE wrapper (entropy_solver.py) and the JAX CVODE factory
(cvode_jax.py).

Internal contract enforced by ``__post_init__``:

    rhs_scale = t_ref / state_scale          (per component)
    state_scale > 0, rhs_scale > 0, t_ref > 0
    state_scale.shape == rhs_scale.shape

By construction, every ``NonDimScales`` instance enforces the internal
contract (``rhs_scale = t_ref / state_scale``, all positive, shapes
matching) inside its ``__post_init__`` before any caller can use it. The
two callers that consume an instance --- the scipy/CVODE wrapper in
``entropy_solver.py`` and the JAX CVODE factory in ``cvode_jax.py`` ---
therefore do NOT need to re-check those invariants on entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class NonDimScales:
    """Per-component nondim scales for the entropy solver state vector.

    Parameters
    ----------
    state_scale : ndarray, shape (n,)
        Physical-units scale per state component:
        ``y_phys = y_nd * state_scale``. Different components can have
        different scales (entropy in J/kg/K, dSdr in J/kg/K/m, T_core
        in K), so the scale is per-element.
    t_ref : float
        Time scale: ``t_phys = t_nd * t_ref``. Same value for every
        state component.
    rhs_scale : ndarray, shape (n,) or None, default None
        Optional precomputed RHS scale ``dydt_nd = dydt_phys * rhs_scale``.
        If None, derived as ``t_ref / state_scale`` automatically. When
        provided, must satisfy the contract or ``__post_init__`` raises.
    """

    state_scale: npt.NDArray
    t_ref: float
    rhs_scale: npt.NDArray = field(default=None)

    def __post_init__(self):
        # Coerce to float64 ndarray for downstream JAX/scipy
        # interoperability. ``object.__setattr__`` is required because
        # the dataclass is frozen.
        ss = np.asarray(self.state_scale, dtype=float)
        object.__setattr__(self, 'state_scale', ss)

        if not (np.isfinite(self.t_ref) and float(self.t_ref) > 0.0):
            raise ValueError(f't_ref must be finite and strictly positive; got {self.t_ref!r}')

        if not np.all(np.isfinite(ss)) or not np.all(ss > 0.0):
            raise ValueError('state_scale must be finite and strictly positive')

        # Derive rhs_scale if not supplied; otherwise validate it
        # against the contract rhs_scale = t_ref / state_scale.
        if self.rhs_scale is None:
            object.__setattr__(self, 'rhs_scale', float(self.t_ref) / ss)
        else:
            rs = np.asarray(self.rhs_scale, dtype=float)
            object.__setattr__(self, 'rhs_scale', rs)
            if rs.shape != ss.shape:
                raise ValueError(
                    'state_scale and rhs_scale must have the same shape; '
                    f'got {ss.shape} vs {rs.shape}'
                )
            if not np.all(np.isfinite(rs)) or not np.all(rs > 0.0):
                raise ValueError('rhs_scale must be finite and strictly positive')
            residual = float(np.max(np.abs(rs * ss - float(self.t_ref)) / float(self.t_ref)))
            if residual > 1.0e-10:
                raise ValueError(
                    'Nondim contract violated: rhs_scale * state_scale != t_ref '
                    f'(max relative residual {residual:.3e}). The convention is '
                    'rhs_scale = t_ref / state_scale; pass rhs_scale=None to '
                    'have it derived automatically.'
                )

    @property
    def n(self) -> int:
        """Number of state-vector components."""
        return int(self.state_scale.size)
