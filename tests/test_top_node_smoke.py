"""The limited top basic node in a solve: state, surface temperature, floor and the JAX mesh."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from aragog.mesh import top_node_entropy
from aragog.solver.entropy_solver import EntropySolver

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos
from .test_eos_floor import _steep_top
from .test_phi_step_cap_armed_smoke import _build_mushy_parameters

pytestmark = [pytest.mark.smoke, needs_eos]

_TOP3 = [2000.0, 236.0, 40.0]  # linear top node at S_min - 58, top cell at S_min + 40


@pytest.fixture(scope='module')
def shared_eos():
    return entropy_eos_copy(EOS_DIR)


def _solver(eos, mode, S0=None, method='cvode', end_time=10.0):
    """A call under no surface flux from ``S0`` (the cold-top profile by default)."""
    p = _build_mushy_parameters(solver_method=method, end_time=end_time)
    p.boundary_conditions.outer_boundary_condition = 4
    p.boundary_conditions.outer_boundary_value = 0.0
    p.boundary_conditions.top_node = mode
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(_steep_top(eos, _TOP3) if S0 is None else S0)
    return s


def _ratio(mesh):
    return float(np.ravel(mesh.basic.delta_mesh)[-1] / np.ravel(mesh.staggered.delta_mesh)[-1])


def test_the_linear_mode_keeps_the_quantity_transform(shared_eos):
    s = _solver(shared_eos, 'linear')
    S = np.asarray(s._S0[: s._n_stag], dtype=float)
    mesh = s.evaluator.mesh
    assert np.array_equal(
        mesh.entropy_at_basic_nodes(S), mesh.quantity_at_basic_nodes(S).ravel()
    )


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_cold_top_cell_solves_with_the_limited_node_inside_the_table(shared_eos, method):
    """The linear node below the table is refused by the floor; the limited node falls back to
    the top cell, inside the table, and the call advances with T_magma at that node."""
    S_min = float(shared_eos.S_min)
    lin = _solver(shared_eos, 'linear', method=method)
    lin.solve()
    assert lin._solution.status == 1 and lin.get_state().dt_actual == 0.0
    s = _solver(shared_eos, 'limited', method=method)
    s.solve()
    out = s.get_state()
    assert s._solution.status >= 0 and out.dt_actual > 0.0
    mesh, S_end = s.evaluator.mesh, np.asarray(out.S_final, dtype=float)
    node = top_node_entropy(S_end, _ratio(mesh), S_min, float(shared_eos.S_max))
    assert S_min <= node and s.state._entropy_basic[-1] == pytest.approx(node, abs=1e-9)
    P_top = float(np.ravel(mesh.basic_pressure)[-1])
    assert out.T_magma == pytest.approx(float(np.ravel(shared_eos.temperature(P_top, node))[0]))


def test_a_smooth_profile_gives_the_same_call_in_both_modes(shared_eos):
    """With equal cell steps at the top the limited node is the linear node, so the call and its
    surface temperature are bit-identical."""
    S0 = np.linspace(float(shared_eos.S_min) + 2600.0, float(shared_eos.S_min) + 2000.0, 11)
    a, b = _solver(shared_eos, 'linear', S0=S0), _solver(shared_eos, 'limited', S0=S0)
    a.solve()
    b.solve()
    assert np.array_equal(a._solution.y, b._solution.y)
    assert a.get_state().T_magma == b.get_state().T_magma


def test_the_jax_mesh_carries_the_limited_node(shared_eos):
    """MeshArrays built from the numpy mesh maps entropy to basic nodes as the numpy mesh does."""
    jnp = pytest.importorskip('jax.numpy')
    from aragog.jax.phase import MeshArrays, entropy_at_basic_nodes

    for mode in ('linear', 'limited'):
        s = _solver(shared_eos, mode)
        S = np.asarray(s._S0[: s._n_stag], dtype=float)
        mesh = s.evaluator.mesh
        got = np.asarray(
            entropy_at_basic_nodes(MeshArrays.from_numpy_mesh(mesh), jnp.asarray(S))
        )
        assert np.allclose(got, mesh.entropy_at_basic_nodes(S), rtol=0.0, atol=1e-9)


def test_the_limited_mode_is_logged(shared_eos, caplog):
    with caplog.at_level(logging.INFO):
        _solver(shared_eos, 'limited')
    assert any('top node: limited' in r.getMessage() for r in caplog.records)
