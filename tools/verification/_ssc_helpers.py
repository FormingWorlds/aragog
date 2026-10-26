"""Shared non-test helpers for solid-state convection verification tools."""

from __future__ import annotations

import contextlib
import copy
import functools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from aragog.eos.entropy import EntropyEOS
from aragog.solver import entropy_solver as es_mod
from aragog.solver import entropy_state as estate_mod


@functools.cache
def _parse(eos_class: type, eos_dir: str):
    try:
        return eos_class(eos_dir), None, None
    except Exception as exc:
        return None, exc, exc.__traceback__


def _parsed(eos_class: type, eos_dir: str):
    """Return the cached parse of ``eos_dir``; a failed parse re-raises without a retry."""
    eos, exc, traceback = _parse(eos_class, eos_dir)
    if exc is not None:
        raise exc.with_traceback(traceback)
    return eos


def entropy_eos_copy(
    eos_dir: Path | str,
    strict_range: bool = False,
    identity_alpha: bool = False,
    eos_class: type = EntropyEOS,
) -> EntropyEOS:
    """Return an independent copy of ``EntropyEOS(eos_dir, strict_range)``.

    The tables are parsed once per process and each call returns a deep copy,
    which starts in the state of a fresh build; only the read-only solidus and
    liquidus interpolation functions are shared. ``identity_alpha`` drops the
    ``thermal_exp`` tables, so the expansivity comes from the identity
    ``rho cp |dT/dP_S| / T`` that the JAX EOS uses.
    """
    eos = copy.deepcopy(_parsed(eos_class, str(eos_dir)))
    eos.strict_range = strict_range
    if identity_alpha:
        for name in ('thermal_exp_solid', 'thermal_exp_melt'):
            eos._tables.pop(name, None)
        eos._has_alpha_tables = False
    return eos


def entropy_eos_jax(eos_dir: Path | str):
    """Return the ``EntropyEOS_JAX`` of ``eos_dir``, parsed once per process.

    The instance is a frozen equinox module of JAX arrays, so callers share it.
    """
    from aragog.jax.eos import EntropyEOS_JAX

    return _parsed(EntropyEOS_JAX, str(eos_dir))


@contextlib.contextmanager
def unbounded_cvode_step() -> Iterator[None]:
    """Override ``EntropySolver._solve_cvode`` with ``max_step = np.inf`` on const-property runs."""
    orig_cvode = es_mod.EntropySolver._solve_cvode

    def _unclamped_cvode(self_s: Any, *a: Any, **kw: Any) -> Any:
        if self_s.entropy_eos is None and getattr(
            self_s.parameters.phase_mixed, 'const_properties', False
        ):
            if len(a) > 5:
                a = (*a[:5], np.inf, *a[6:])
            else:
                kw['max_step'] = np.inf
        return orig_cvode(self_s, *a, **kw)

    es_mod.EntropySolver._solve_cvode = _unclamped_cvode
    try:
        yield
    finally:
        es_mod.EntropySolver._solve_cvode = orig_cvode


@contextlib.contextmanager
def half_cell_surface_flux() -> Iterator[None]:
    """Override ``EntropyState.update`` on NumPy const-property runs so BC 5 uses pure half-cell conduction."""
    orig_update = estate_mod.EntropyState.update

    def _half_cell_update(
        self_st: Any, entropy: Any, time_val: Any, *a: Any, **kw: Any
    ) -> None:
        orig_update(self_st, entropy, time_val, *a, **kw)
        bc_set = self_st._evaluator.boundary_conditions._settings
        if int(bc_set.outer_boundary_condition) == 5:
            k_surf = float(np.asarray(self_st.phase_basic.thermal_conductivity()).ravel()[-1])
            t_cell = float(np.asarray(self_st.phase_staggered.temperature()).ravel()[-1])
            r_b = np.asarray(self_st._evaluator.mesh.basic.radii).ravel()
            dr_half = 0.5 * float(r_b[-1] - r_b[-2])
            self_st._heat_flux[-1] = (
                k_surf * (t_cell - float(bc_set.outer_boundary_value)) / dr_half
            )

    estate_mod.EntropyState.update = _half_cell_update
    try:
        yield
    finally:
        estate_mod.EntropyState.update = orig_update
