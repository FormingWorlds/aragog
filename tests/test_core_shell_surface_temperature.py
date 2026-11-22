"""The shell contrast of the core_module CMB law takes the surface temperature.

Thiriet et al. (2019) eq. 16 takes the internal Rayleigh number of their eq. 3, whose
temperature difference is the one across the shell, from the core to the surface. The solver
takes the surface temperature of outer boundary 5, else ``surface_temperature`` of the boundary
settings, else the top cell's temperature.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
pytest.importorskip('equinox')

jax.config.update('jax_enable_x64', True)

from aragog.jax.solver import _dSdt_core_module_parts  # noqa: E402
from tests.conftest import entropy_eos_copy, needs_eos  # noqa: E402
from tests.test_jax_dsdt_core_module import (  # noqa: E402
    _build_jax_pieces,
    _build_numpy_solver,
)

pytestmark = [pytest.mark.timeout(300)]

BAD_VALUES = [0.0, -1.0, math.nan, math.inf, True, '1600']


@pytest.fixture(scope='module')
def harness():
    """The driven core_module solver with a core 1000 K above the base (boundary-layer branch)
    and a copy of its state whose top cell is 2000 J/kg/K hotter."""
    solver = _build_numpy_solver(entropy_eos_copy(identity_alpha=True))
    args = _build_jax_pieces(solver)
    n = solver._n_stag
    y = np.asarray(solver._S0, dtype=float)
    y[n + 1] += 1000.0
    hot_top = y.copy()
    hot_top[n - 1] += 2000.0
    return solver, args, y, hot_top


def _flux_np(solver, state):
    solver.dSdt(0.0, state)
    return float(solver.state._heat_flux[0])


def _flux_jax(args, state, t_s):
    parts = _dSdt_core_module_parts(0.0, jnp.asarray(state), args + (t_s,))
    return float(parts.heat_flux[0])


@pytest.mark.smoke
@pytest.mark.physics_invariant
@needs_eos
def test_a_top_cell_change_moves_the_cmb_flux_only_without_a_surface_temperature(
    harness, monkeypatch
):
    """With ``surface_temperature`` set, a 2000 J/kg/K change of the top cell leaves the CMB
    flux of both paths unchanged; without it the top cell stands in for the surface and the same
    change moves the flux. A surface temperature equal to the top cell's gives the fallback flux,
    so the surface value takes the place of the top cell one to one."""
    solver, args, y, hot_top = harness
    settings = solver.evaluator.boundary_conditions._settings
    monkeypatch.setattr(settings, 'surface_temperature', None)
    base, moved = _flux_np(solver, y), _flux_np(solver, hot_top)
    assert base > 1e3  # the boundary-layer branch, where Ra_crit enters
    assert abs(moved / base - 1.0) > 1e-3
    assert _flux_jax(args, y, math.nan) == pytest.approx(base, rel=1e-10)
    assert _flux_jax(args, hot_top, math.nan) == pytest.approx(moved, rel=1e-10)

    monkeypatch.setattr(settings, 'surface_temperature', 1600.0)
    held = _flux_np(solver, y)
    assert _flux_np(solver, hot_top) == pytest.approx(held, rel=1e-14)
    assert _flux_jax(args, y, 1600.0) == pytest.approx(held, rel=1e-10)
    assert _flux_jax(args, hot_top, 1600.0) == pytest.approx(held, rel=1e-10)
    assert abs(held / base - 1.0) > 1e-6

    t_top = float(
        np.asarray(
            solver.entropy_eos.temperature(
                float(solver._P_stag_flat[-1]), y[solver._n_stag - 1]
            )
        ).flat[0]
    )
    monkeypatch.setattr(settings, 'surface_temperature', t_top)
    assert _flux_np(solver, y) == pytest.approx(base, rel=1e-12)
    assert _flux_jax(args, y, t_top) == pytest.approx(base, rel=1e-10)


@pytest.mark.smoke
@needs_eos
def test_the_surface_temperature_of_boundary_5_comes_first(harness, monkeypatch):
    """Boundary 5 gives its own surface temperature, then ``surface_temperature``, then NaN
    (the top cell)."""
    solver, _, _, _ = harness
    settings = solver.evaluator.boundary_conditions._settings
    monkeypatch.setattr(settings, 'surface_temperature', None)
    assert math.isnan(solver.cmb_shell_surface_temperature())
    monkeypatch.setattr(settings, 'surface_temperature', 1600.0)
    assert solver.cmb_shell_surface_temperature() == 1600.0
    monkeypatch.setattr(solver, '_outer_bc_kind', 5)
    monkeypatch.setattr(solver.state, '_surface_temperature_bc', (3000.0, 1500.0, 1.0))
    monkeypatch.setattr(solver, '_refresh_surface_temperature', lambda: None)
    assert solver.cmb_shell_surface_temperature() == 1500.0


@pytest.mark.smoke
@needs_eos
@pytest.mark.parametrize('value', BAD_VALUES)
def test_a_surface_temperature_set_between_calls_is_checked(harness, monkeypatch, value):
    """A value set on the settings after construction is checked when the solver reads it."""
    solver, _, _, _ = harness
    settings = solver.evaluator.boundary_conditions._settings
    monkeypatch.setattr(settings, 'surface_temperature', value)
    with pytest.raises(ValueError, match='surface_temperature'):
        solver.cmb_shell_surface_temperature()


@pytest.mark.unit
@pytest.mark.parametrize('value', BAD_VALUES)
def test_surface_temperature_refuses_every_other_value(value):
    """A surface_temperature that is not a finite positive number is refused by name."""
    from aragog.config.boundary import check_surface_temperature

    with pytest.raises(ValueError, match='surface_temperature'):
        check_surface_temperature(value)


@pytest.mark.unit
@pytest.mark.parametrize('value', [None, 1600.0, 300])
def test_surface_temperature_accepts_none_and_positive_numbers(value):
    """None and any finite positive number pass, in every outer boundary mode."""
    from aragog.config.boundary import BoundaryConfig, check_surface_temperature
    from aragog.parser import _BoundaryConditionsParameters

    check_surface_temperature(value)
    for cls in (BoundaryConfig, _BoundaryConditionsParameters):
        bc = cls(
            outer_boundary_condition=1,
            outer_boundary_value=0.0,
            inner_boundary_condition=1,
            inner_boundary_value=4000.0,
            emissivity=1.0,
            equilibrium_temperature=273.0,
            core_heat_capacity=880.0,
            surface_temperature=value,
        )
        assert bc.surface_temperature == value


@pytest.mark.unit
def test_the_surface_temperature_of_the_factory_is_keyword_only():
    """The factory takes the surface temperature by keyword only, so a caller that passes
    ``start_time`` by position still binds it to ``start_time``."""
    import inspect

    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    params = inspect.signature(build_jax_rhs_and_jacobian).parameters
    kind = params['core_module_surface_temperature'].kind
    assert kind is inspect.Parameter.KEYWORD_ONLY
