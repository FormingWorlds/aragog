"""Defensive contract tests for the JAX CVODE factory.

Locks the OQ3 option C contract for ``NonDimScales`` and the
per-call shape checks in ``build_jax_rhs_and_jacobian``:
- NonDimScales.__post_init__ enforces
  ``rhs_scale = t_ref / state_scale`` plus per-component positivity.
- The JAX factory checks shape compatibility against
  ``heating_array`` and ``core_bc_mode``.

The factory build path is exercised with the lightest possible JAX
pytrees and a tiny heating array; no solver integration is performed.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')


def _make_scales(state_scale, t_ref, rhs_scale=None):
    """Construct NonDimScales for tests; rhs_scale=None auto-derives."""
    from aragog.jax.nondim import NonDimScales

    return NonDimScales(
        state_scale=np.asarray(state_scale, dtype=float),
        t_ref=float(t_ref),
        rhs_scale=(np.asarray(rhs_scale, dtype=float) if rhs_scale is not None else None),
    )


def _build_factory(state_scale, rhs_scale, t_ref, heating, core_bc_mode):
    """Call ``build_jax_rhs_and_jacobian`` with stub pytrees.

    Constructs a NonDimScales (which itself enforces the internal
    contract) then forwards. Tests targeting NonDimScales contract
    failures construct it directly via ``_make_scales``.
    """
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    scales = _make_scales(state_scale, t_ref, rhs_scale)
    return build_jax_rhs_and_jacobian(
        eos_jax=None,
        phase_params=None,
        mesh_arrays=None,
        boundary_params=None,
        heating_array=np.asarray(heating, dtype=float),
        scales=scales,
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
    state_scale[:n_stag] = 3.0e3  # entropy
    state_scale[n_stag] = 1.0e-6  # dSdr_cmb
    rhs_scale = t_ref / state_scale
    heating = np.zeros(n_stag)
    rhs_fn, jac_fn, info = _build_factory(
        state_scale, rhs_scale, t_ref, heating, 'energy_balance'
    )
    assert callable(rhs_fn)


@pytest.mark.unit
def test_contract_violation_raises():
    """``rhs_scale * state_scale != t_ref`` must raise inside NonDimScales."""
    n = 5
    t_ref = 1.234
    state_scale = np.full(n, 3.0e3)
    rhs_scale = np.full(n, 1.0)  # WRONG: not t_ref / state_scale
    with pytest.raises(ValueError, match='Nondim contract violated'):
        _make_scales(state_scale, t_ref, rhs_scale=rhs_scale)


@pytest.mark.unit
def test_shape_mismatch_state_vs_rhs_raises():
    """Mismatched state_scale / rhs_scale shapes must raise in
    NonDimScales (contract is owned by the dataclass)."""
    n = 5
    state_scale = np.full(n, 3.0e3)
    rhs_scale = np.full(n + 1, 1.0e-3)
    with pytest.raises(ValueError, match='same shape'):
        _make_scales(state_scale, 1.0, rhs_scale=rhs_scale)


@pytest.mark.unit
def test_state_size_vs_heating_quasi_steady_raises():
    """quasi_steady requires state_scale.size == heating.size."""
    n = 5
    state_scale = np.full(n + 2, 3.0e3)  # too long
    rhs_scale = 1.0 / state_scale
    heating = np.zeros(n)
    with pytest.raises(ValueError, match='incompatible'):
        _build_factory(state_scale, rhs_scale, 1.0, heating, 'quasi_steady')


@pytest.mark.unit
def test_state_size_vs_heating_energy_balance_raises():
    """energy_balance requires state_scale.size == heating.size + 1."""
    n_stag = 5
    state_scale = np.full(n_stag, 3.0e3)  # missing the dSdr_cmb slot
    rhs_scale = 1.0 / state_scale
    heating = np.zeros(n_stag)
    with pytest.raises(ValueError, match='incompatible'):
        _build_factory(state_scale, rhs_scale, 1.0, heating, 'energy_balance')


@pytest.mark.unit
def test_negative_state_scale_raises():
    """Negative state_scale must raise in NonDimScales."""
    state_scale = np.array([3.0e3, 3.0e3, -1.0, 3.0e3])
    with pytest.raises(ValueError, match='state_scale.*positive'):
        _make_scales(state_scale, 1.0)  # rhs_scale auto-derived


@pytest.mark.unit
def test_nonfinite_t_ref_raises():
    """Inf t_ref must raise in NonDimScales."""
    state_scale = np.full(4, 3.0e3)
    with pytest.raises(ValueError, match='t_ref.*positive'):
        _make_scales(state_scale, np.inf)


@pytest.mark.unit
def test_zero_t_ref_raises():
    """Zero t_ref must raise in NonDimScales."""
    state_scale = np.full(4, 3.0e3)
    with pytest.raises(ValueError, match='t_ref.*positive'):
        _make_scales(state_scale, 0.0)


@pytest.mark.unit
def test_factory_rejects_legacy_positional_scales():
    """OQ3 option C: passing the old (state_scale, rhs_scale, t_ref)
    triplet to ``build_jax_rhs_and_jacobian`` must raise TypeError.

    Forces all callers onto the new NonDimScales contract; eliminates
    the silent-divergence risk that motivated OQ3 in the first place.
    """
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    n = 4
    with pytest.raises(TypeError, match='NonDimScales'):
        build_jax_rhs_and_jacobian(
            eos_jax=None,
            phase_params=None,
            mesh_arrays=None,
            boundary_params=None,
            heating_array=np.zeros(n),
            scales=np.full(n, 3.0e3),  # ← raw ndarray, not NonDimScales
            core_bc_mode='quasi_steady',
        )


@pytest.mark.unit
def test_nondim_scales_rhs_scale_auto_derived():
    """When rhs_scale=None, NonDimScales derives t_ref / state_scale."""
    state_scale = np.array([3.0e3, 1.0e-6, 5.0e2])
    t_ref = 12.5
    sc = _make_scales(state_scale, t_ref)
    np.testing.assert_allclose(
        np.asarray(sc.rhs_scale),
        t_ref / state_scale,
        rtol=1e-15,
        atol=0.0,
    )
    # And n property reports the right size
    assert sc.n == 3


@pytest.mark.unit
def test_nondim_scales_immutable():
    """NonDimScales is frozen — direct field mutation must raise."""
    sc = _make_scales(np.full(3, 3.0e3), 1.0)
    with pytest.raises(Exception):  # FrozenInstanceError on dataclasses
        sc.t_ref = 2.0  # type: ignore[misc]


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
