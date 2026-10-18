"""The limited top basic node in a solve: state, surface temperature, floor and the JAX mesh."""

from __future__ import annotations

import dataclasses
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


def test_the_grey_body_flux_is_continuous_across_the_band(shared_eos):
    """Under the grey-body surface the top-cell rate follows the top node: moving the top cell in
    steps of 0.1 J/kg/K through the band changes it by far less than the switch to the top cell
    did (about 3.3e-3 on this profile)."""
    s = _solver(shared_eos, 'limited', bc=1)
    y0, n, S_min = np.asarray(s._S0, dtype=float).copy(), s._n_stag, float(shared_eos.S_min)
    rates, free = [], []
    for t in 333.0 + 0.1 * np.arange(-300, 701):
        y = y0.copy()
        y[n - 3 : n] = S_min + np.array([1700.0, 1000.0, t])
        rates.append(float(np.ravel(s.dSdt(0.0, y))[n - 1]))
        free.append(float(top_node_entropy(y[:n], 1.0)))
    free = np.asarray(free)
    assert (free < S_min).any() and (free > S_min + 50.0).any()
    assert np.max(np.abs(np.diff(rates))) < 1.0e-3


@pytest.mark.parametrize(
    ('margin', 'band'), [(30.0, 15.0), (None, 50.0), (0.0, 50.0), (-5.0, 50.0), (5e-324, 50.0)]
)
def test_the_band_is_half_the_floor_margin(shared_eos, margin, band):
    """The blend band is half the EOS floor margin; without the floor, or where half the margin
    is not positive, it is 50 J/kg/K. The JAX mesh carries the same band."""
    from aragog.jax.phase import MeshArrays

    p = _build_mushy_parameters(solver_method='cvode', end_time=1.0)
    p.boundary_conditions.top_node = 'limited'
    p.energy = dataclasses.replace(p.energy, eos_floor_margin=margin)
    s = EntropySolver(p, entropy_eos=shared_eos)
    s.initialize()
    assert s.evaluator.mesh.top_band == band
    assert MeshArrays.from_numpy_mesh(s.evaluator.mesh).top_band == band


def test_the_band_follows_a_margin_change_at_the_next_solve(shared_eos):
    s = _solver(shared_eos, 'limited', S0=np.full(11, float(shared_eos.S_min) + 2000.0))
    s.parameters.energy = dataclasses.replace(s.parameters.energy, eos_floor_margin=40.0)
    s.solve()
    assert s.evaluator.mesh.top_band == 20.0


def test_a_plain_entropy_eos_assignment_and_a_patch_keep_working(shared_eos):
    """The EOS is a plain attribute: any object can be assigned in linear mode, and a patch of it
    restores the solver's own EOS."""
    from unittest import mock

    s = _solver(shared_eos, 'linear')
    with mock.patch.object(s, 'entropy_eos', object()):
        pass
    assert s.entropy_eos is shared_eos
    s.entropy_eos = object()


def test_the_limited_mode_is_logged_once_per_solver(shared_eos, caplog):
    with caplog.at_level(logging.INFO):
        s = _solver(shared_eos, 'limited')
        s.initialize()
    lines = [r.getMessage() for r in caplog.records if 'top node: limited' in r.getMessage()]
    assert len(lines) == 1 and 'blends into the top cell' in lines[0]


def test_the_jax_rhs_matches_numpy_inside_the_band(shared_eos):
    """With a 30 J/kg/K floor margin (band 15) and the limited node 7 J/kg/K above S_min, the JAX
    top-cell rate under the grey-body surface matches the numpy one."""
    jnp = pytest.importorskip('jax.numpy')
    from aragog.jax.solver import dSdt

    from .test_entropy_solver_core_module_smoke import _build
    from .test_jax_dsdt_core_module import _build_jax_pieces

    eos = entropy_eos_copy(EOS_DIR, identity_alpha=True)
    s = _build('quasi_steady', eos)
    s.parameters.boundary_conditions.top_node = 'limited'
    s.parameters.energy = dataclasses.replace(s.parameters.energy, eos_floor_margin=30.0)
    s.initialize()
    s.set_initial_entropy(2700.0)
    y, n = np.asarray(s._S0, dtype=float).copy(), s._n_stag
    y[n - 3 : n] = float(eos.S_min) + np.array([1700.0, 1000.0, 338.0])
    free = float(top_node_entropy(y[:n], s.evaluator.mesh.top_ratio)) - float(eos.S_min)
    assert 0.0 < free < 15.0 and s.evaluator.mesh.top_band == 15.0
    s._core_module_budget, s._core_module_q_radio, s._core_module_ra_crit_cmb = None, 0, 0
    f_jax = float(np.ravel(dSdt(0.0, jnp.asarray(y), _build_jax_pieces(s)[:6]))[n - 1])
    assert f_jax == pytest.approx(float(np.ravel(s.dSdt(0.0, y))[n - 1]), rel=1e-8)


def test_a_grey_body_solve_inside_the_band_finishes_and_closes_its_energy(shared_eos):
    """A top cell at S_min + 340 over cells at S_min + [1700, 1000] puts the limited node inside
    the band, where the node is not monotone in the top cell, on a part of the table where T is
    not flat: CVODE finishes the call without a stall and the energy ledger closes."""
    S_min = float(shared_eos.S_min)
    S0 = np.full(11, S_min + 2600.0)
    S0[-3:] = S_min + np.array([1700.0, 1000.0, 340.0])
    s = _solver(shared_eos, 'limited', S0=S0, bc=1, end_time=50.0)
    s.solve()
    out = s.get_state()
    terms = (
        out.step_dE_F_int_J
        + out.step_dE_F_cmb_J
        + out.step_dE_Q_radio_J
        + out.step_dE_Q_tidal_J
    )
    assert s._solution.status >= 0 and out.dt_actual == pytest.approx(50.0)
    assert getattr(s._solution, 'cvode_nst', 0) < 5000
    assert abs(out.step_dE_state_heat_J - terms) <= 1e-3 * abs(out.step_dE_F_int_J)
