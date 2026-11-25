"""Solver configuration."""

from __future__ import annotations

import logging

import attrs

logger: logging.Logger = logging.getLogger('fwl.' + __name__)

EOS_DOMAIN_MODES = ('off', 'warn', 'raise')


def check_eos_domain_mode(mode) -> None:
    """Refuse an ``eos_domain_check`` that is not one of ``EOS_DOMAIN_MODES``."""
    if not isinstance(mode, str) or mode not in EOS_DOMAIN_MODES:
        raise ValueError(f'eos_domain_check must be one of {EOS_DOMAIN_MODES}, got {mode!r}')


def check_eos_domain_tolerance(tol) -> float:
    """Return ``eos_domain_tolerance_cells`` as a float; refuse one not finite and > 0."""
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise TypeError(f'eos_domain_tolerance_cells must be a number, got {tol!r}')
    if not 0.0 < tol < float('inf'):
        raise ValueError(f'eos_domain_tolerance_cells must be finite and > 0, got {tol!r}')
    return float(tol)


@attrs.define
class SolverConfig:
    """ODE solver parameters.

    Parameters
    ----------
    start_time : float
        Start time [years].
    end_time : float
        End time [years].
    atol : float
        Absolute tolerance, floored at 1e-8. Default 1e-8.
    rtol : float
        Relative tolerance. Default 1e-8, the value at which
        ``phase_boundary_cap = 'rate'`` is verified. Above 1e-8 a CVODE run logs one
        warning: a loose rtol can lock CVODE at the convective switch (the convective
        mask and the kappa_h floor).
    tsurf_poststep_change : float
        Maximum surface temperature change per step [K].
    cvode_output_points : int
        Number of points on the CVODE dense output grid.
    max_steps : int
        Maximum number of internal CVODE steps per output interval (SUNDIALS mxstep).
    tcore_change_limit : float or None
        Optional per-solve core-temperature change limit [K]. The flag is
        also set, independent of this limit, when any sampled core
        temperature is non-finite.
    strict_accepted_range : bool
        With the rheology off: an accepted state of a call outside the entropy range of the EOS
        tables raises ``AcceptedStateRangeError`` instead of failing the call. Off by default.
    eos_domain_check : str
        Check of the EOS lookup points of each call against the valid window of the tables
        (``valid_mask_*.dat``): ``'off'``, ``'warn'`` (1 warning per call, the default) or
        ``'raise'`` (``EOSDomainError``). Off for tables without mask files.
    eos_domain_tolerance_cells : float
        Distance outside the window, in table entropy cells, at which a point counts. Default 5.
    """

    start_time: float
    end_time: float
    atol: float = 1e-8
    rtol: float = 1e-8
    tsurf_poststep_change: float = 30.0
    # Not the live config path: runtime validation uses
    # aragog.parser._SolverParameters, not this attrs schema.
    cvode_output_points: int = attrs.field(
        default=65,
        validator=attrs.validators.and_(
            attrs.validators.instance_of(int),
            attrs.validators.ge(2),
        ),
    )
    max_steps: int = attrs.field(
        default=100000,
        validator=attrs.validators.and_(
            attrs.validators.not_(attrs.validators.instance_of(bool)),
            attrs.validators.instance_of(int),
            attrs.validators.ge(1),
        ),
    )
    tcore_change_limit: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.and_(
                attrs.validators.not_(attrs.validators.instance_of(bool)),
                attrs.validators.instance_of((int, float)),
                attrs.validators.gt(0.0),
            )
        ),
    )
    strict_accepted_range: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    eos_domain_check: str = attrs.field(
        default='warn', validator=lambda _inst, _attr, v: check_eos_domain_mode(v)
    )
    eos_domain_tolerance_cells: float = attrs.field(
        default=5.0, validator=lambda _inst, _attr, v: check_eos_domain_tolerance(v)
    )
