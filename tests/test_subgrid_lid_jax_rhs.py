"""The JAX sub-grid lid against the NumPy lid of the solver."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import entropy_eos_copy, entropy_eos_jax
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, _build, needs_eos

jax = pytest.importorskip('jax')

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(600), needs_eos]


def _lid_cases():
    from aragog.parser import SolidRheologyParams

    return {
        'no_rheology': None,
        'lid_closure': SolidRheologyParams(enabled=True),
        'fixed_local': SolidRheologyParams(
            enabled=True,
            stress_closure_mode='local',
            lid_base_mode='fixed',
            lid_base_temperature=1350.0,
        ),
        'arrhenius_local': SolidRheologyParams(enabled=True, stress_closure_mode='local'),
    }


@pytest.mark.parametrize('case', list(_lid_cases()))
def test_lid_top_matches_the_numpy_lid(case):
    """Fed the NumPy top-cell state, the JAX lid gives the NumPy surface loss, fluid flux,
    fluid volume and lid rates to relative 1e-10, on molten and partly solid top cells with
    and without the solid rheology."""
    from aragog.jax import surface_lid as jl
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=10, end_time=1.0)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.normalize()
    if _lid_cases()[case] is not None:
        p.phase_solid.rheology = _lid_cases()[case]
    s = EntropySolver(p, entropy_eos=entropy_eos_copy(EOS_DIR))
    s.initialize()
    s.set_initial_entropy(3050.0)
    lp, eos_jax = s.jax_lid_params(), entropy_eos_jax(EOS_DIR)
    seen = {}
    evaluate = s._lid_evaluate

    def spy(entropy):
        seen['F_grid'] = float(s.state._heat_flux[-1])
        seen['ev'] = evaluate(entropy)
        return seen['ev']

    s._lid_evaluate = spy
    rng = np.random.default_rng(13)
    for _ in range(4):
        y = s._S0.copy()
        y[: s._n_stag] = np.linspace(
            rng.uniform(2900, 3100), rng.uniform(2300, 3100), s._n_stag
        )
        delta = 10.0 ** rng.uniform(0.0, 2.5)
        y[-2:] = (delta, delta * float(s._lid_hv.H(rng.uniform(600.0, 1300.0))))
        s._dSdt_single(0.0, y)
        ev, ph = seen['ev'], s.state.phase_staggered

        def top(v):
            return float(np.asarray(v).ravel()[-1])

        props = dict(
            T=top(ph.temperature()),
            rho=top(ph.density()),
            cp=top(ph.heat_capacity()),
            k=top(ph.thermal_conductivity()),
            alpha=top(ph.thermal_expansivity()),
            g=top(ph.gravitational_acceleration()),
            eta=top(ph.viscosity()),
            phi=top(ph.melt_fraction()),
        )
        ls = s.state.lid_state
        T_iso = float(np.asarray(ls['T_lid_iso'])) if ls is not None else np.nan
        got = jl.lid_top(
            lp,
            eos_jax,
            y[-2:],
            float(y[s._n_stag - 1]),
            props,
            float(s.state._heat_flux[-2]),
            seen['F_grid'],
            T_iso,
            top(s.state.heating),
        )
        r = ev.rates
        want = (r.F_top, r.F_fluid, ev.V_fluid, r.ddelta, r.dE)
        np.testing.assert_allclose(np.array(got, dtype=float), want, rtol=1e-10, atol=1e-30)
