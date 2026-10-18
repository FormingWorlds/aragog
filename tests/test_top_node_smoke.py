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


def _solver(eos, mode, S0=None, method='cvode', end_time=10.0, bc=4):
    """A call from ``S0`` (the cold-top profile by default) under no surface flux (``bc`` 4) or
    the grey-body surface at 255 K (``bc`` 1)."""
    p = _build_mushy_parameters(solver_method=method, end_time=end_time)
    p.boundary_conditions.outer_boundary_condition = bc
    p.boundary_conditions.outer_boundary_value = 0.0
    p.boundary_conditions.emissivity = 1.0
    p.boundary_conditions.equilibrium_temperature = 255.0
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


@pytest.mark.parametrize('top3', [[3000.0, 1700.0, 450.0], [3000.0, 2000.0, 900.0]])
@pytest.mark.parametrize('core_bc', ['core_module', 'energy_balance', 'quasi_steady'])
def test_the_jax_rhs_uses_the_limited_node_as_the_numpy_rhs_does(core_bc, top3):
    """Under the grey-body surface the top node sets the radiated flux. On a top that falls back
    to the top cell, and on one where the next step sets the node, the JAX top-cell rate matches
    the numpy one in limited mode and differs from the linear mode, so the node reaches both."""
    jnp = pytest.importorskip('jax.numpy')
    from aragog.jax.solver import dSdt, dSdt_core_module, dSdt_energy_balance

    from .test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build
    from .test_jax_dsdt_core_module import _build_jax_pieces

    eos = entropy_eos_copy(EOS_DIR, identity_alpha=True)
    rate = {}
    for mode in ('linear', 'limited'):
        s = _build(core_bc, eos, CORE_MODULE_PARAMS if core_bc == 'core_module' else None)
        s.parameters.boundary_conditions.top_node = mode
        s.initialize()
        s.set_initial_entropy(2700.0)
        y, n = np.asarray(s._S0, dtype=float).copy(), s._n_stag
        y[n - 3 : n] = float(eos.S_min) + np.asarray(top3)
        if core_bc == 'core_module':
            f_jax = dSdt_core_module(0.0, jnp.asarray(y), _build_jax_pieces(s))
        else:
            s._core_module_budget, s._core_module_q_radio, s._core_module_ra_crit_cmb = (
                None,
                0,
                0,
            )
            fn = dSdt_energy_balance if core_bc == 'energy_balance' else dSdt
            f_jax = fn(0.0, jnp.asarray(y), _build_jax_pieces(s)[:6])
        f_np = np.asarray(s.dSdt(0.0, y)).ravel()
        assert float(np.ravel(f_jax)[n - 1]) == pytest.approx(f_np[n - 1], rel=1e-8)
        rate[mode] = f_np[n - 1]
    assert rate['limited'] != pytest.approx(rate['linear'], rel=1e-6)


def test_a_cold_top_cell_under_the_grey_body_surface_solves_in_limited_mode(shared_eos):
    """Under the grey-body surface the linear node is refused by the floor; the limited node
    advances, and the radiated flux uses the temperature at that node."""
    lin = _solver(shared_eos, 'linear', bc=1)
    lin.solve()
    assert lin._solution.status == 1 and lin.get_state().dt_actual == 0.0
    s = _solver(shared_eos, 'limited', bc=1)
    s.solve()
    out = s.get_state()
    assert s._solution.status >= 0 and out.dt_actual > 0.0
    T = float(np.ravel(s.state.top_temperature)[0])
    assert T == pytest.approx(out.T_magma)


def test_the_gradient_core_refuses_the_limited_node():
    from aragog.config.boundary import BoundaryConfig

    with pytest.raises(ValueError, match='top_node'):
        BoundaryConfig(
            outer_boundary_condition=4, outer_boundary_value=0.0, inner_boundary_condition=1,
            inner_boundary_value=0.0, emissivity=1.0, equilibrium_temperature=255.0,
            core_heat_capacity=880.0, core_bc='gradient', top_node='limited',
        )  # fmt: skip


def test_fewer_than_3_cells_refuse_the_limited_node(shared_eos):
    p = _build_mushy_parameters(solver_method='cvode', n_nodes=3, end_time=1.0)
    p.boundary_conditions.top_node = 'limited'
    with pytest.raises(ValueError, match='top_node'):
        EntropySolver(p, entropy_eos=shared_eos).initialize()


def test_the_mesh_takes_the_mode_from_the_parameters_and_the_bounds_from_the_eos(shared_eos):
    """A new EOS on the solver moves the bounds of the limited node with it."""
    s = _solver(shared_eos, 'limited')
    mesh = s.evaluator.mesh
    assert mesh.top_node == 'limited'
    assert mesh.top_bounds == (float(shared_eos.S_min), float(shared_eos.S_max))
    other = entropy_eos_copy(EOS_DIR)
    other.S_min, other.S_max = float(shared_eos.S_min) + 500.0, float(shared_eos.S_max) - 500.0
    s.entropy_eos = other
    assert s.evaluator.mesh.top_bounds == (float(other.S_min), float(other.S_max))
