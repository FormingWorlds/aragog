"""Shared non-test helpers for solid-state convection verification tools."""

from __future__ import annotations

import copy
import functools
from pathlib import Path

from aragog.eos.entropy import EntropyEOS


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
