"""Defensive contract tests for the JAX CVODE factory.

Locks the OQ3 option B nondim-contract assertion in
``aragog/solver/cvode_jax.py::build_jax_rhs_and_jacobian``: the factory
must raise on a violated ``rhs_scale * state_scale == t_ref`` identity,
on shape mismatches, or on non-finite / non-positive scales.

The factory build path is exercised with the lightest possible JAX
pytrees and a tiny heating array; no solver integration is performed.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')


def _build_factory(state_scale, rhs_scale, t_ref, heating, core_bc_mode):
    """Call ``build_jax_rhs_and_jacobian`` with stub pytrees.

    The factory only references the JAX pytrees lazily (inside the
    closures), so the contract checks at the top of the function fire
    before the dummy values are touched.
    """
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    return build_jax_rhs_and_jacobian(
        eos_jax=None,
        phase_params=None,
        mesh_arrays=None,
        boundary_params=None,
        heating_array=np.asarray(heating, dtype=float),
        state_scale=np.asarray(state_scale, dtype=float),
        rhs_scale=np.asarray(rhs_scale, dtype=float),
        t_ref=float(t_ref),
        core_bc_mode=core_bc_mode,
    )


@pytest.mark.unit
def test_contract_holds_quasi_steady():
    """A correctly-constructed (state, rhs, t_ref) triplet must succeed."""
    n = 5
    t_ref = 1.234
    state_scale = np.full(n, 3.0e3)
    rhs_scale = t_ref / state_scale
    heating = np.zeros(n)
    rhs_fn, jac_fn, info = _build_factory(
        state_scale, rhs_scale, t_ref, heating, 'quasi_steady'
    )
    assert callable(rhs_fn)
    assert callable(jac_fn)
    assert info['rhs_calls'] == 0


@pytest.mark.unit
def test_contract_holds_energy_balance_extended_state():
    """Energy_balance state vector is heating.size + 1; mixed scales OK."""
    n_stag = 4
    t_ref = 100.0
    state_scale = np.empty(n_stag + 1)
    state_scale[:n_stag] = 3.0e3       # entropy
    state_scale[n_stag] = 1.0e-6       # dSdr_cmb
    rhs_scale = t_ref / state_scale
    heating = np.zeros(n_stag)
    rhs_fn, jac_fn, info = _build_factory(
        state_scale, rhs_scale, t_ref, heating, 'energy_balance'
    )
    assert callable(rhs_fn)


@pytest.mark.unit
def test_contract_violation_raises():
    """``rhs_scale * state_scale != t_ref`` must raise at factory build."""
    n = 5
    t_ref = 1.234
    state_scale = np.full(n, 3.0e3)
    rhs_scale = np.full(n, 1.0)        # WRONG: not t_ref / state_scale
    heating = np.zeros(n)
    with pytest.raises(ValueError, match='Nondim contract violated'):
        _build_factory(state_scale, rhs_scale, t_ref, heating, 'quasi_steady')


@pytest.mark.unit
def test_shape_mismatch_state_vs_rhs_raises():
    n = 5
    state_scale = np.full(n, 3.0e3)
    rhs_scale = np.full(n + 1, 1.0e-3)
    heating = np.zeros(n)
    with pytest.raises(ValueError, match='same shape'):
        _build_factory(state_scale, rhs_scale, 1.0, heating, 'quasi_steady')


@pytest.mark.unit
def test_state_size_vs_heating_quasi_steady_raises():
    """quasi_steady requires state_scale.size == heating.size."""
    n = 5
    state_scale = np.full(n + 2, 3.0e3)   # too long
    rhs_scale = 1.0 / state_scale
    heating = np.zeros(n)
    with pytest.raises(ValueError, match='incompatible'):
        _build_factory(state_scale, rhs_scale, 1.0, heating, 'quasi_steady')


@pytest.mark.unit
def test_state_size_vs_heating_energy_balance_raises():
    """energy_balance requires state_scale.size == heating.size + 1."""
    n_stag = 5
    state_scale = np.full(n_stag, 3.0e3)   # missing the dSdr_cmb slot
    rhs_scale = 1.0 / state_scale
    heating = np.zeros(n_stag)
    with pytest.raises(ValueError, match='incompatible'):
        _build_factory(
            state_scale, rhs_scale, 1.0, heating, 'energy_balance'
        )


@pytest.mark.unit
def test_negative_state_scale_raises():
    n = 4
    state_scale = np.array([3.0e3, 3.0e3, -1.0, 3.0e3])
    rhs_scale = 1.0 / np.abs(state_scale)
    heating = np.zeros(n)
    with pytest.raises(ValueError, match='state_scale.*positive'):
        _build_factory(state_scale, rhs_scale, 1.0, heating, 'quasi_steady')


@pytest.mark.unit
def test_nonfinite_t_ref_raises():
    n = 4
    state_scale = np.full(n, 3.0e3)
    rhs_scale = 1.0 / state_scale
    heating = np.zeros(n)
    with pytest.raises(ValueError, match='t_ref.*positive'):
        _build_factory(state_scale, rhs_scale, np.inf, heating, 'quasi_steady')


@pytest.mark.unit
def test_zero_t_ref_raises():
    n = 4
    state_scale = np.full(n, 3.0e3)
    rhs_scale = 1.0 / state_scale
    heating = np.zeros(n)
    with pytest.raises(ValueError, match='t_ref.*positive'):
        _build_factory(state_scale, rhs_scale, 0.0, heating, 'quasi_steady')


@pytest.mark.unit
@pytest.mark.parametrize('mode', ['bower2018', 'gradient', 'unknown'])
def test_unsupported_core_bc_mode_raises_with_clear_message(mode, caplog):
    """A4: bower2018 / gradient / typo modes must raise with a message
    that names the supported alternatives, AND log a warning so the
    entropy_solver catch-all fallback explains the FD-Jacobian downgrade.

    Previously the error said only "must be 'quasi_steady' or
    'energy_balance', got ..." without explaining the user-facing
    workaround (set use_jax_jacobian=false).
    """
    import logging
    n = 4
    state_scale = np.full(n, 3.0e3)
    rhs_scale = 1.0 / state_scale
    heating = np.zeros(n)
    with caplog.at_level(logging.WARNING, logger='aragog.solver.cvode_jax'):
        with pytest.raises(ValueError, match='is not supported'):
            _build_factory(state_scale, rhs_scale, 1.0, heating, mode)
    # Warning text must mention both the offending mode and the
    # supported workaround so downstream operators don't have to
    # bisect their own logs.
    msgs = ' '.join(rec.message for rec in caplog.records)
    assert mode in msgs
    assert 'quasi_steady' in msgs and 'energy_balance' in msgs
