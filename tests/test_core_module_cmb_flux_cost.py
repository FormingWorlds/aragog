"""Cost of the core_module CMB flux: one phase update of the boundary-layer evaluator per call
(the bottom cell and the layer together) and eq. 16 only where the law needs it, with the flux
of one update per state, bit for bit, under 450, eq. 16 and the law by phase."""

from __future__ import annotations

import copy
import math
from unittest import mock

import numpy as np
import pytest

pytest.importorskip('jax')

import aragog.core.boundary_layer as boundary_layer  # noqa: E402
from aragog.core import RA_CRIT_PHASE, cmb_boundary_layer_flux  # noqa: E402
from aragog.core.boundary_layer import RA_CRIT_UPPER_LAYER  # noqa: E402
from aragog.eos.entropy_phase import EntropyPhaseEvaluator  # noqa: E402
from tests.conftest import needs_eos  # noqa: E402
from tests.test_cmb_law_by_phase import _band_point, _bottom_state, _phase_solver  # noqa: E402

LAWS = [RA_CRIT_UPPER_LAYER, None, RA_CRIT_PHASE]
# bottom cell solid, at 3 points across the rheological transition, and liquid
STATES = ['solid', 0.1, 0.5, 0.85, 'liquid']


def _state(solver, where):
    phi = {'solid': 0.0, 'liquid': 1.0}.get(where)
    return _bottom_state(solver, _band_point(solver, where) if phi is None else phi)


def _one_update_per_state_flux(solver, t_core, s_bottom, s_top):
    """The CMB flux with a one-point phase update for the bottom cell state and another for
    the layer state, the properties of the law otherwise as ``_core_module_cmb_flux``."""
    bl = copy.deepcopy(solver._bl_phase)
    bl.set_pressure(np.ravel(bl.pressure)[:1])
    bl._g = np.ravel(bl._g)[:1]
    eos, p = solver.entropy_eos, float(solver._P_basic_flat[0])

    def first(values):
        return float(np.asarray(values).flat[0])

    t_hot = max(float(t_core), 1.0)
    t_m = first(eos.temperature(p, s_bottom))
    t_surf = solver.cmb_shell_surface_temperature()
    if math.isnan(t_surf):
        t_surf = first(eos.temperature(float(solver._P_stag_flat[-1]), s_top))
    s_bl = eos.entropy_at_temperature(p, 0.5 * (t_hot + t_m))
    bl.set_entropy(np.array([s_bottom]))
    bl.update()
    eta_interior, phi = first(bl.viscosity()), first(bl.melt_fraction())
    law, kw = solver._core_module_ra_crit_cmb, {}
    if law == RA_CRIT_PHASE:
        pm = solver.parameters.phase_mixed
        band = (pm.rheological_transition_melt_fraction, pm.rheological_transition_width)
        kw = dict(melt_fraction=phi, liquid_band=tuple(float(x) for x in band))
    bl.set_entropy(np.array([s_bl]))
    bl.update()
    return float(
        cmb_boundary_layer_flux(
            t_hot,
            t_m,
            conductivity=first(bl.thermal_conductivity()),
            density=first(bl.density()),
            heat_capacity=first(eos.heat_capacity(p, s_bl)),
            expansivity=first(eos.material_expansivity(p, s_bl)),
            viscosity=first(bl.viscosity()),
            gravity=first(solver.state.phase_basic.gravitational_acceleration()),
            dr_half=solver._cmb_dr_half,
            ra_crit=None if kw else law,
            shell_thickness=float(solver._r_basic_flat[-1] - solver._r_basic_flat[0]),
            shell_contrast=t_hot - t_surf,
            interior_viscosity=eta_interior,
            **kw,
        )
    )


@pytest.mark.smoke
@needs_eos
@pytest.mark.parametrize('law', LAWS)
def test_one_layer_update_per_call_and_eq16_only_where_the_law_needs_it(law):
    """A CMB flux call updates the boundary-layer evaluator once, and evaluates eq. 16 once
    under eq. 16 and the law by phase and not at a fixed critical Rayleigh number."""
    solver = _phase_solver()
    solver._core_module_ra_crit_cmb = law
    n, y = solver._n_stag, _state(solver, 0.5)
    update, eq16 = EntropyPhaseEvaluator.update, boundary_layer.critical_rayleigh_eq16
    with (
        mock.patch.object(
            EntropyPhaseEvaluator, 'update', autospec=True, side_effect=update
        ) as up,
        mock.patch.object(boundary_layer, 'critical_rayleigh_eq16', side_effect=eq16) as e16,
    ):
        q = solver._core_module_cmb_flux(y[n + 1], y[0], y[n - 1])
    assert math.isfinite(q) and q > 0.0
    assert (up.call_count, e16.call_count) == (1, 0 if law == RA_CRIT_UPPER_LAYER else 1)


@pytest.mark.smoke
@pytest.mark.physics_invariant
@needs_eos
@pytest.mark.parametrize('where', STATES)
@pytest.mark.parametrize('law', LAWS)
def test_the_cmb_flux_equals_one_update_per_state_bit_for_bit(law, where):
    """The flux of the call equals, bit for bit, the flux with one phase update per state."""
    solver = _phase_solver()
    solver._core_module_ra_crit_cmb = law
    n, y = solver._n_stag, _state(solver, where)
    q = solver._core_module_cmb_flux(y[n + 1], y[0], y[n - 1])
    assert q == _one_update_per_state_flux(solver, y[n + 1], y[0], y[n - 1])
