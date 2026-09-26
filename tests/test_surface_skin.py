"""Tests for the conductive surface skin (outer BC 6) and the table-edge cutoff.

The skin temperature solves ``eps sigma (T_s^4 - T_eq^4) = G (T_top - T_s)``
with ``G = k_top / dr_half``; the references here come from
``scipy.optimize.brentq`` on that balance, not from the Newton solver.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.constants import Stefan_Boltzmann as SIGMA
from scipy.optimize import brentq

from aragog.surface_skin import skin_temperature, solid_weight, table_edge_factor

pytestmark = pytest.mark.unit


def _root(T_top, G, eps, T_eq):
    f = lambda T: eps * SIGMA * (T**4 - T_eq**4) - G * (T_top - T)  # noqa: E731
    lo, hi = min(T_top, T_eq), max(T_top, T_eq)
    return lo if lo == hi else brentq(f, lo, hi, xtol=1e-13, rtol=1e-15)


def test_skin_temperature_matches_bracketed_root_over_a_wide_grid():
    """Newton from the upper bound converges to the brentq root to 1e-12 over
    T_top 50 to 6000 K, G 1e-7 to 1e5 W/m^2/K, T_eq 30 to 2000 K."""
    worst = 0.0
    for T_top, G, T_eq, eps in itertools.product(
        np.geomspace(50.0, 6000.0, 9),
        np.geomspace(1e-7, 1e5, 9),
        (30.0, 273.0, 2000.0),
        (0.2, 1.0),
    ):
        ref = _root(T_top, G, eps, T_eq)
        worst = max(worst, abs(skin_temperature(T_top, G, eps, T_eq, SIGMA) - ref) / ref)
    assert worst < 1e-12


def test_skin_flux_of_a_solid_lid_is_conduction_limited():
    """T_top 400 K, k 4 W/m/K, dr_half 15 km, T_eq 273 K: the flux is about
    0.034 W/m^2, bounded by G (T_top - T_eq); the plain grey body at 400 K
    radiates about 1065 W/m^2."""
    G = 4.0 / 15e3
    T_s = skin_temperature(400.0, G, 1.0, 273.0, SIGMA)
    F = G * (400.0 - T_s)
    assert F == pytest.approx(G * (400.0 - _root(400.0, G, 1.0, 273.0)), rel=1e-10)
    assert 0.0 < F < G * (400.0 - 273.0)
    assert F == pytest.approx(0.0339, rel=1e-2)
    assert SIGMA * (400.0**4 - 273.0**4) > 1e4 * F


def test_solid_weight_is_exactly_one_and_zero_outside_the_ramp():
    phi = np.array([0.0, 0.3, 0.30001, 0.4, 0.49999, 0.50001, 1.0])
    s = solid_weight(phi, 0.4)
    assert s[0] == 1.0 and s[1] == 1.0 and s[5] == 0.0 and s[6] == 0.0
    assert s[3] == pytest.approx(0.5, abs=1e-15)
    assert 0.0 < s[4] < s[2] < 1.0
    # first and second derivatives vanish at both ends of the ramp
    h = 1e-6
    for x in (0.3, 0.5):
        d1 = (solid_weight(x + h, 0.4) - solid_weight(x - h, 0.4)) / (2 * h)
        assert abs(d1) < 1e-8


def test_table_edge_factor_is_half_at_the_offset_and_small_at_the_edge():
    assert table_edge_factor(-1506.0, -1606.0) == pytest.approx(0.5, abs=1e-12)
    assert table_edge_factor(-1606.0, -1606.0) < 1.5e-3
    assert table_edge_factor(0.0, -1606.0) == 1.0


# ---------------------------------------------------------------------------
# Solver right-hand side (numpy and JAX), with the SPIDER P-S test tables
# ---------------------------------------------------------------------------

from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, _build  # noqa: E402

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(), reason=f'P-S tables not found at {EOS_DIR}'
)


def _solver(outer_bc, S_top, table_edge_cutoff=False, T_eq=273.0):
    from aragog.eos.entropy import EntropyEOS
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=outer_bc, inner_bc=2, inner_bc_value=0.0)
    p.boundary_conditions.equilibrium_temperature = T_eq
    p.boundary_conditions.table_edge_cutoff = table_edge_cutoff
    s = EntropySolver(p, entropy_eos=EntropyEOS(EOS_DIR))
    s.initialize()
    y = np.linspace(3000.0, S_top, s._n_stag)
    s.set_initial_entropy(y)
    s.dSdt(0.0, y)
    return s, y


def _top(s):
    ps = s.state.phase_staggered
    r_b = np.asarray(s._r_basic_flat)
    return (
        float(np.asarray(ps.temperature()).flat[-1]),
        float(np.asarray(ps.thermal_conductivity()).flat[-1]) / (0.5 * (r_b[-1] - r_b[-2])),
        float(np.asarray(ps.melt_fraction()).flat[-1]),
    )


@needs_eos
def test_bc6_solid_top_cell_gives_the_skin_flux():
    """A solid top cell (S 200, melt fraction 0): the applied flux equals the
    brentq skin flux; outer BC 1 at the same state radiates orders more."""
    s6, _ = _solver(6, 200.0)
    T_top, G, phi = _top(s6)
    assert phi == 0.0
    expected = G * (T_top - _root(T_top, G, 1.0, 273.0))
    assert float(s6.state.heat_flux[-1]) == pytest.approx(expected, rel=1e-10)
    s1, _ = _solver(1, 200.0)
    assert float(s1.state.heat_flux[-1]) > 1e3 * expected


@needs_eos
def test_bc6_molten_top_cell_is_the_grey_body_bit_for_bit():
    s6, _ = _solver(6, 6000.0)
    s1, _ = _solver(1, 6000.0)
    assert _top(s6)[2] >= 0.5
    assert float(s6.state.heat_flux[-1]) == float(s1.state.heat_flux[-1])


@needs_eos
def test_cutoff_scales_bc4_flux_near_the_table_edge_and_reports_the_deficit():
    """Outer BC 4 with the cutoff: near the edge the applied flux is the
    prescribed flux times the edge factor; without the cutoff it is unchanged."""
    from aragog.eos.entropy import EntropyEOS

    S_edge = EntropyEOS(EOS_DIR).S_min_solid
    S_top = S_edge + 60.0
    on, y = _solver(4, S_top, table_edge_cutoff=True)
    off, _ = _solver(4, S_top, table_edge_cutoff=False)
    F_prescribed = float(off.state.heat_flux[-1])
    assert float(on.state.heat_flux[-1]) == pytest.approx(
        F_prescribed * table_edge_factor(float(y[-1]), S_edge), rel=1e-12
    )
    assert on._surface_flux_nominal == F_prescribed
    assert float(on.state.heat_flux[-1]) < 0.1 * F_prescribed


@needs_eos
def test_jax_surface_bc_matches_numpy_and_has_a_finite_exact_jacobian():
    """JAX ``_apply_surface_bc`` type 6 equals the numpy flux at a solid, a
    mushy and a molten top cell, and its derivative in T_top equals a central
    difference."""
    import jax
    import jax.numpy as jnp

    from aragog.jax.solver import BoundaryParams, _apply_surface_bc

    jax.config.update('jax_enable_x64', True)

    class _Mesh:
        pass

    for S_top in (200.0, 1400.0, 6000.0):
        s6, _ = _solver(6, S_top)
        T_top, G, phi = _top(s6)
        r_b = np.asarray(s6._r_basic_flat)
        mesh = _Mesh()
        mesh.radii_basic = jnp.asarray(r_b)
        T_basic_top = float(s6.state.top_temperature.item())
        k_top = G * 0.5 * (r_b[-1] - r_b[-2])
        bc = BoundaryParams(
            outer_bc_type=6,
            outer_bc_value=0.0,
            emissivity=1.0,
            T_eq=273.0,
            inner_bc_type=2,
            inner_bc_value=0.0,
            core_density=1.0e4,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
            S_table_edge=-1.0e30,
            phi_rheo=0.4,
        )

        def flux(T):
            ph = _Mesh()
            ph.temperature = jnp.array([T])
            ph.thermal_conductivity = jnp.array([k_top])
            ph.melt_fraction = jnp.array([phi])
            hf = _apply_surface_bc(
                jnp.zeros(3), bc, jnp.array([T_basic_top]), ph, jnp.asarray(0.0), mesh
            )
            return hf[-1]

        assert float(flux(T_top)) == pytest.approx(float(s6.state.heat_flux[-1]), rel=1e-12)
        d_ad = float(jax.grad(flux)(T_top))
        # h = 0.5 K: the grey-body part dominates F in mush, so a small h loses the
        # derivative to round-off; the skin part is close to linear in T_top.
        d_fd = (float(flux(T_top + 0.5)) - float(flux(T_top - 0.5))) / 1.0
        assert np.isfinite(d_ad) and d_ad == pytest.approx(d_fd, rel=1e-5, abs=1e-12)
