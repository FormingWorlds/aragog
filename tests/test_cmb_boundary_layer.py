"""CMB flux from the lower thermal boundary layer law of Deschamps and Sotin (2000).

Module under test: ``aragog.cmb_boundary_layer.cmb_flux`` and its use in the numpy
``EntropySolver`` (inner BC 3 and quasi_steady inner BC 1) and in the JAX
``_apply_cmb_bc``. Invariants: the flux is ``k dT_c / delta`` with ``Ra_delta =
0.28 Ra^0.21`` (their eqs. 32 and 33); it is odd in ``T_c - T_m``; with the law off
every flux is unchanged; the law sets only its face flux.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.cmb_boundary_layer import RA_C, RA_CRIT_EXPONENT, RA_CRIT_PREFACTOR, cmb_flux
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, _build

pytestmark = [pytest.mark.unit, pytest.mark.timeout(300)]

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(), reason=f'P-S tables not found at {EOS_DIR}'
)

# Thiriet et al. (2019) Mars1 at t = 0 (their Tables 2 and 3), interior at 1800 K.
MARS = dict(T_c=2250.0, T_m=1800.0, T_s=250.0, depth=1.7e6, rho=3500.0, g=3.7, alpha=2.5e-5)
MARS_ETA = 1e21 * np.exp(300e3 / 8.3144 * (1.0 / 1800.0 - 1.0 / 1600.0))


def _mars_flux(**kw):
    args = dict(MARS, kappa=1e-6, k=4.0, eta=MARS_ETA)
    args.update(kw)
    return cmb_flux(**args)


@pytest.mark.physics_invariant
def test_flux_is_the_published_law_at_the_mars_start():
    """Mars1 at t = 0: Ra 3.9e7, Ra_delta 11.0, delta 18.3 km, q 98.2 mW/m^2, the
    same order as the 3-D value of about 89 mW/m^2 (Thiriet et al. 2019, Fig. 2b).
    The reference below is eqs. 32 and 33 written out, not the module's algebra."""
    rho, g, alpha, kappa, eta = MARS['rho'], MARS['g'], MARS['alpha'], 1e-6, MARS_ETA
    Ra = rho * g * alpha * (MARS['T_c'] - MARS['T_s']) * MARS['depth'] ** 3 / (kappa * eta)
    delta = (0.28 * Ra**0.21 * kappa * eta / (rho * g * alpha * 450.0)) ** (1.0 / 3.0)
    q = _mars_flux()
    assert q == pytest.approx(4.0 * 450.0 / delta, rel=1e-12)
    assert 18.0e3 < delta < 18.6e3 and 0.097 < q < 0.099
    assert (RA_CRIT_PREFACTOR, RA_CRIT_EXPONENT) == (0.28, 0.21)
    # The value discriminates the exponents: 0.20 in place of 0.21 moves it by over 1 %.
    assert abs((0.28 * Ra**0.20) ** (-1 / 3) / (0.28 * Ra**0.21) ** (-1 / 3) - 1.0) > 0.01


@pytest.mark.physics_invariant
def test_flux_is_odd_in_the_temperature_jump_and_zero_without_it():
    """A core colder than the interior draws heat in with the same magnitude; the
    flux grows as dT_c^(4/3) (Ra_delta fixed, delta ~ dT_c^(-1/3))."""
    up, down = _mars_flux(T_m=1800.0), _mars_flux(T_m=2700.0)
    assert down == pytest.approx(-up, rel=1e-12)
    assert _mars_flux(T_m=2250.0) == 0.0
    ratio = _mars_flux(T_m=2025.0) / up  # dT_c 225 against 450 K
    assert ratio == pytest.approx(0.5 ** (4.0 / 3.0), rel=1e-12)


@pytest.mark.physics_invariant
def test_flux_continuity_across_ra_c_and_nu_1():
    """Continuity of q across Ra = Ra_c and across Nu = 1.

    Asserts |q(x+h) - q(x-h)| -> 0 monotonically at 3 decreasing values of h.
    """
    rho, g, alpha, kappa, eta, k, depth = (
        MARS['rho'],
        MARS['g'],
        MARS['alpha'],
        1e-6,
        MARS_ETA,
        4.0,
        MARS['depth'],
    )
    buoyancy = rho * g * alpha / (kappa * eta)
    T_s = MARS['T_s']

    # 1. Across Ra = Ra_c at fixed T_m
    delta_T_crit = RA_C / (buoyancy * depth**3)
    T_c_crit = T_s + delta_T_crit
    T_m = 1800.0
    diffs_ra = []
    h_steps_ra = [1e-2 * delta_T_crit, 1e-3 * delta_T_crit, 1e-4 * delta_T_crit]
    for h in h_steps_ra:
        qp = cmb_flux(T_c_crit + h, T_m, T_s, depth, rho, g, alpha, kappa, k, eta)
        qm = cmb_flux(T_c_crit - h, T_m, T_s, depth, rho, g, alpha, kappa, k, eta)
        diffs_ra.append(abs(qp - qm))
    assert diffs_ra[1] < diffs_ra[0]
    assert diffs_ra[2] < diffs_ra[1]
    assert diffs_ra[2] < 1e-5

    # 2. Across Nu = 1 at Ra >> Ra_c
    T_c_conv = 2250.0
    ra_conv = buoyancy * (T_c_conv - T_s) * depth**3
    ra_dc = RA_CRIT_PREFACTOR * ra_conv**RA_CRIT_EXPONENT
    dT_nu1 = ra_dc / (buoyancy * depth**3)
    T_m_crit = T_c_conv - dT_nu1
    diffs_nu = []
    h_steps_nu = [1e-2 * dT_nu1, 1e-3 * dT_nu1, 1e-4 * dT_nu1]
    for h in h_steps_nu:
        qp = cmb_flux(T_c_conv, T_m_crit - h, T_s, depth, rho, g, alpha, kappa, k, eta)
        qm = cmb_flux(T_c_conv, T_m_crit + h, T_s, depth, rho, g, alpha, kappa, k, eta)
        diffs_nu.append(abs(qp - qm))
    assert diffs_nu[1] < diffs_nu[0]
    assert diffs_nu[2] < diffs_nu[1]
    assert diffs_nu[2] < 1e-8


@pytest.mark.physics_invariant
def test_flux_continuity_at_ra_just_above_ra_c_with_large_ra_l():
    """When Ra is just above Ra_c with Ra_l large, max(1, .) keeps q continuous.

    At Ra ~ Ra_c with large T_c - T_m, without flooring Ra at Ra_c the convective
    branch would diverge as (T_c - T_s)^-0.07. Flooring at Ra_c maintains continuity.
    """
    rho, g, alpha, kappa, eta, k, depth = (
        MARS['rho'],
        MARS['g'],
        MARS['alpha'],
        1e-6,
        MARS_ETA,
        4.0,
        MARS['depth'],
    )
    buoyancy = rho * g * alpha / (kappa * eta)
    T_s = MARS['T_s']
    T_m = 2500.0  # Large Ra_l
    delta_T_crit = RA_C / (buoyancy * depth**3)
    T_c_crit = T_s + delta_T_crit
    eps = 1e-4 * delta_T_crit

    q_above = cmb_flux(T_c_crit + eps, T_m, T_s, depth, rho, g, alpha, kappa, k, eta)
    q_below = cmb_flux(T_c_crit - eps, T_m, T_s, depth, rho, g, alpha, kappa, k, eta)
    assert abs(q_above - q_below) < 1e-4
    assert np.isfinite(q_above) and np.isfinite(q_below)


@pytest.mark.physics_invariant
def test_flux_monotonic_in_dt_c_at_fixed_surface_temperature():
    """Flux q is strictly monotonic in dT_c >= 0 at fixed T_s."""
    rho, g, alpha, kappa, eta, k, depth = (
        MARS['rho'],
        MARS['g'],
        MARS['alpha'],
        1e-6,
        MARS_ETA,
        4.0,
        MARS['depth'],
    )
    T_s = MARS['T_s']
    T_m = 1800.0
    dT_vals = np.linspace(0.0, 1500.0, 151)
    q_vals = [
        cmb_flux(T_m + dt, T_m, T_s, depth, rho, g, alpha, kappa, k, eta) for dt in dT_vals
    ]
    diffs = np.diff(q_vals)
    assert np.all(diffs > 0.0)


@pytest.mark.physics_invariant
def test_flux_odd_symmetry_on_conductive_branch():
    """On the conductive branch (Nu = 1), q(-dT_c) = -q(dT_c)."""
    rho, g, alpha, kappa, eta, k, depth = (
        MARS['rho'],
        MARS['g'],
        MARS['alpha'],
        1e-6,
        MARS_ETA,
        4.0,
        MARS['depth'],
    )
    T_s = MARS['T_s']
    T_m = 2000.0
    dT_c = 1e-8
    q_pos = cmb_flux(T_m + dT_c, T_m, T_s, depth, rho, g, alpha, kappa, k, eta)
    q_neg = cmb_flux(T_m - dT_c, T_m, T_s, depth, rho, g, alpha, kappa, k, eta)
    assert q_pos == pytest.approx(-q_neg, rel=1e-12)
    assert q_pos == pytest.approx(k * dT_c / depth, rel=1e-12)


@pytest.mark.physics_invariant
def test_flux_finite_and_continuous_at_tc_equals_ts():
    """Flux q is finite and continuous at T_c = T_s."""
    rho, g, alpha, kappa, eta, k, depth = (
        MARS['rho'],
        MARS['g'],
        MARS['alpha'],
        1e-6,
        MARS_ETA,
        4.0,
        MARS['depth'],
    )
    T_s = MARS['T_s']
    T_m = 1800.0
    q_at = cmb_flux(T_s, T_m, T_s, depth, rho, g, alpha, kappa, k, eta)
    assert np.isfinite(q_at)

    diffs = []
    for h in [1e-2, 1e-3, 1e-4]:
        qp = cmb_flux(T_s + h, T_m, T_s, depth, rho, g, alpha, kappa, k, eta)
        qm = cmb_flux(T_s - h, T_m, T_s, depth, rho, g, alpha, kappa, k, eta)
        diffs.append(abs(qp - qm))
    assert diffs[1] < diffs[0]
    assert diffs[2] < diffs[1]
    assert diffs[2] < 1e-6


def test_loader_needs_quasi_steady_with_inner_bc_1_or_3():
    from aragog.parser import _BoundaryConditionsParameters

    def bc(core_bc, inner, law='deschamps_sotin_2000'):
        p = _BoundaryConditionsParameters(
            outer_boundary_condition=1,
            outer_boundary_value=0.0,
            inner_boundary_condition=inner,
            inner_boundary_value=2000.0,
            emissivity=1.0,
            equilibrium_temperature=255.0,
            core_heat_capacity=880.0,
            core_bc=core_bc,
            cmb_flux_law=law,
        )
        p.normalize()
        return p

    for ok in (('quasi_steady', 1), ('quasi_steady', 3)):
        assert bc(*ok).cmb_flux_law == 'deschamps_sotin_2000'
    for core_bc in ('energy_balance', 'gradient', 'bower2018'):
        with pytest.raises(ValueError, match='cmb_flux_law'):
            bc(core_bc, 1)
    with pytest.raises(ValueError, match='cmb_flux_law'):
        bc('quasi_steady', 2)
    with pytest.raises(ValueError, match='must be one of'):
        bc('quasi_steady', 1, law='thiriet')


# ---------------------------------------------------------------------------
# Solver right-hand side
# ---------------------------------------------------------------------------


def _solver(inner_bc, law, S_bottom=2600.0):
    """Tables mode, 20 nodes: the entropy is uniform (2400) over the middle half
    of the mantle and rises to ``S_bottom`` in the lowest cells."""
    from aragog.eos.entropy import EntropyEOS
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=inner_bc, n_nodes=20)
    p.boundary_conditions.inner_boundary_value = 4000.0 if inner_bc == 3 else 0.0
    p.boundary_conditions.cmb_flux_law = law
    s = EntropySolver(p, entropy_eos=EntropyEOS(EOS_DIR))
    s.initialize()
    y = np.full(s._n_stag, 2400.0)
    y[:3] = [S_bottom, 2550.0, 2450.0]
    s.set_initial_entropy(y)
    s.dSdt(0.0, y)
    return s, y


def _expected(s, T_c, face):
    """Law at the solver state from quantities read off the state: the interior
    entropy is 2400 by construction, so no weighting enters."""
    ps = s.state.phase_staggered
    m = s._cmb_law_interior
    visc = np.asarray(ps.viscosity()).ravel()[m]
    eta = np.exp(
        np.average(
            np.log(visc), weights=s._volume_flat[m] * np.asarray(ps.density()).ravel()[m]
        )
    )
    rho, k, cp = (
        float(np.asarray(f()).ravel()[0])
        for f in (ps.density, ps.thermal_conductivity, ps.heat_capacity)
    )
    return cmb_flux(
        T_c,
        s.entropy_eos.temperature_scalar(float(s._P_basic_flat[face]), 2400.0),
        float(s.state.top_temperature.item()),
        float(s._r_basic_flat[-1] - s._r_basic_flat[0]),
        rho,
        float(s._g_basic_flat[face]),
        float(np.asarray(ps.thermal_expansivity()).ravel()[0]),
        k / (rho * cp),
        k,
        eta,
    )


@needs_eos
@pytest.mark.physics_invariant
def test_bc3_face_flux_is_the_law_and_the_rest_is_unchanged():
    """Inner BC 3 at 4000 K: face 0 carries the law flux; every other face equals
    the run with the law off, and the off run conducts across the half cell."""
    on, _ = _solver(3, 'deschamps_sotin_2000')
    off, _ = _solver(3, 'none')
    F_on, F_off = (
        np.asarray(on.state.heat_flux).ravel(),
        np.asarray(off.state.heat_flux).ravel(),
    )
    assert F_on[0] == pytest.approx(_expected(on, 4000.0, 0), rel=1e-12)
    np.testing.assert_array_equal(F_on[1:], F_off[1:])
    T0 = float(np.asarray(off.state.phase_staggered.temperature()).flat[0])
    k0 = float(np.asarray(off.state.phase_staggered.thermal_conductivity()).flat[0])
    assert F_off[0] == k0 * (4000.0 - T0) / off._cmb_dr_half
    assert F_on[0] != pytest.approx(F_off[0], rel=1e-3)


@needs_eos
@pytest.mark.physics_invariant
def test_quasi_steady_law_sets_face_1_and_the_core_takes_its_share():
    """quasi_steady: the core and cell 0 are one unit, so the law acts on face 1
    with T_c the cell-0 temperature, and face 0 keeps alpha times face 1."""
    on, _ = _solver(1, 'deschamps_sotin_2000')
    off, _ = _solver(1, 'none')
    F_on, F_off = (
        np.asarray(on.state.heat_flux).ravel(),
        np.asarray(off.state.heat_flux).ravel(),
    )
    T0 = float(np.asarray(on.state.phase_staggered.temperature()).flat[0])
    assert F_on[1] == pytest.approx(_expected(on, T0, 1), rel=1e-12)
    assert F_on[0] / F_on[1] == pytest.approx(F_off[0] / F_off[1], rel=1e-12)
    np.testing.assert_array_equal(F_on[2:], F_off[2:])
    assert F_on[1] != pytest.approx(F_off[1], rel=1e-3)


@needs_eos
def test_jax_face_flux_equals_numpy_and_has_a_finite_exact_gradient():
    """JAX ``_apply_cmb_bc`` with the numpy state (tables through EntropyEOS_JAX):
    the face flux equals numpy for both modes, and d q / d T_c equals a central
    difference."""
    jax = pytest.importorskip('jax')
    from types import SimpleNamespace

    import jax.numpy as jnp

    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.solver import BoundaryParams, _apply_cmb_bc

    jax.config.update('jax_enable_x64', True)
    eos = EntropyEOS_JAX(EOS_DIR)
    for inner_bc, face in ((3, 0), (1, 1)):
        s, y = _solver(inner_bc, 'deschamps_sotin_2000')
        ps = s.state.phase_staggered
        arr = lambda f: jnp.asarray(np.asarray(f()).ravel())  # noqa: E731
        phase = SimpleNamespace(
            density=arr(ps.density),
            heat_capacity=arr(ps.heat_capacity),
            thermal_conductivity=arr(ps.thermal_conductivity),
            thermal_expansivity=arr(ps.thermal_expansivity),
            viscosity=arr(ps.viscosity),
            temperature=arr(ps.temperature),
        )
        mesh = SimpleNamespace(
            area=jnp.asarray(s._area_flat),
            volume=jnp.asarray(s._volume_flat),
            P_basic=jnp.asarray(s._P_basic_flat),
            radii_basic=jnp.asarray(s._r_basic_flat),
            gravity=jnp.asarray(s._g_basic_flat),
        )
        T_top = float(s.state.top_temperature.item())

        def face_flux(T_c, inner_bc=inner_bc, face=face):
            bc = BoundaryParams(
                outer_bc_type=1,
                outer_bc_value=0.0,
                emissivity=1.0,
                T_eq=255.0,
                inner_bc_type=inner_bc,
                inner_bc_value=T_c,
                core_density=10500.0,
                core_heat_capacity=880.0,
                tfac_core_avg=1.147,
                cmb_flux_law=True,
                cmb_law_interior=s._cmb_law_interior.astype(float),
            )
            ph = SimpleNamespace(
                **{**vars(phase), 'temperature': phase.temperature.at[0].set(T_c)}
            )
            # Faces 0 and 1 start at 0, so only the JAX law can fill them.
            hf = jnp.asarray(np.asarray(s.state.heat_flux).ravel()).at[:2].set(0.0)
            out = _apply_cmb_bc(
                hf,
                bc,
                mesh,
                ph.density,
                ph.heat_capacity,
                0.0,
                phase_stag_T=ph.temperature,
                phase_stag_k=ph.thermal_conductivity,
                law_inputs=(eos, ph, jnp.asarray(y), T_top),
            )
            return out[face]

        T_c = 4000.0 if inner_bc == 3 else float(phase.temperature[0])
        assert float(face_flux(T_c)) == pytest.approx(
            float(np.asarray(s.state.heat_flux).ravel()[face]), rel=1e-6
        )
        d_ad = float(jax.grad(face_flux)(T_c))
        d_fd = (float(face_flux(T_c + 1.0)) - float(face_flux(T_c - 1.0))) / 2.0
        assert np.isfinite(d_ad) and d_ad == pytest.approx(d_fd, rel=1e-5)


@needs_eos
@pytest.mark.physics_invariant
def test_energy_closes_with_the_law_in_a_short_run():
    """quasi_steady, 100 yr with the law: the discrete energy residual stays at the
    level of the run without it (the law replaces one face flux, it adds no source)."""
    res = {}
    for law in ('none', 'deschamps_sotin_2000'):
        s, y = _solver(1, law)
        s.parameters.solver.end_time = 100.0
        s.set_initial_entropy(y)
        s.solve()
        st = s.get_state()
        res[law] = (
            abs(float(st.energy_residual)),
            abs(float(np.asarray(st.heat_flux).ravel()[-1])) * float(s._area_flat[-1]),
        )
    for law, (resid, power) in res.items():
        assert resid < 1e-6 * power, (law, resid, power)


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_steady_basal_heating_carries_the_law_flux_to_the_surface(monkeypatch):
    """Wiring in an integrated run: constant properties and viscosity, no internal
    heating, fixed CMB (1273 K) and surface (273 K) temperatures, 1000 km shell,
    Ra about 4e6. At steady state the CMB power equals the surface power and the
    applied CMB flux is the law at the final state."""
    from tests.test_mesh_refinement_halfspace import _params

    p = _params(
        cell_km=0.0,
        outer_bc=5,
        T_top=273.0,
        T_cmb=1273.0,
        T0=1100.0,
        r_in=5.0e6,
        r_out=6.0e6,
        n=60,
        t_end_yr=2e11,
        convection=True,
        log10visc=20.5,
    )
    p.boundary_conditions.cmb_flux_law = 'deschamps_sotin_2000'
    from aragog.solver.entropy_solver import EntropySolver

    # Constant properties report melt fraction 1, which caps CVODE at 100 yr steps.
    orig = EntropySolver._solve_cvode
    monkeypatch.setattr(
        EntropySolver,
        '_solve_cvode',
        lambda self, *a, **k: orig(self, *a, **{**k, 'max_step': np.inf}),
    )

    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    s.set_initial_entropy(3000.0)
    s.solve()
    st = s.get_state()
    s.dSdt(s._solution.t[-1], np.asarray(st.S_final, float).ravel())
    F = np.asarray(s.state.heat_flux, float).ravel()
    A = s._area_flat
    assert F[0] > 0.0 and F[0] * A[0] == pytest.approx(F[-1] * A[-1], rel=1e-3)
    assert F[0] == pytest.approx(
        s._cmb_law_flux(np.asarray(st.S_final, float).ravel()), rel=1e-12
    )


@pytest.mark.physics_invariant
def test_fixed_surface_temperature_sets_the_rayleigh_number():
    """Outer BC 5 (surface held at 273 K) with a uniform 1200 K mantle: the top node
    still reads 1200 K, but the law uses the prescribed 273 K; an isothermal start at
    T_c gives zero flux, not a division by zero."""
    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_mesh_refinement_halfspace import _params

    def face0(T0):
        p = _params(
            0.0,
            outer_bc=5,
            T_top=273.0,
            T_cmb=1273.0,
            T0=T0,
            n=40,
            convection=True,
            log10visc=20.5,
        )
        p.boundary_conditions.cmb_flux_law = 'deschamps_sotin_2000'
        s = EntropySolver(p, entropy_eos=None)
        s.initialize()
        y = np.full(s._n_stag, 3000.0)
        s.set_initial_entropy(y)
        s.dSdt(0.0, y)
        return s, float(np.asarray(s.state.heat_flux).ravel()[0])

    s, q = face0(1200.0)
    ps = s.state.phase_staggered
    args = dict(
        T_c=1273.0,
        T_m=1200.0,
        depth=float(s._r_basic_flat[-1] - s._r_basic_flat[0]),
        rho=4000.0,
        g=float(s._g_basic_flat[0]),
        alpha=float(np.asarray(ps.thermal_expansivity()).ravel()[0]),
        kappa=1e-6,
        k=4.0,
        eta=float(np.asarray(ps.viscosity()).ravel()[0]),
    )
    assert float(s.state.top_temperature.item()) == pytest.approx(1200.0, rel=1e-12)
    assert q == pytest.approx(cmb_flux(T_s=273.0, **args), rel=1e-12)
    assert abs(cmb_flux(T_s=1199.0, **args) / q - 1.0) > 0.1
    assert abs(face0(1273.0)[1]) < 1e-15


@needs_eos
@pytest.mark.physics_invariant
def test_quasi_steady_law_with_bottom_cell_heating_matches_numpy_and_closed_form():
    """Inner BC 1 with the DS2000 law and bottom-cell internal heating.

    The numpy and JAX CMB fluxes agree (rtol <= 1e-10) and both match the closed-form
    alpha * (q_law - Q_first / A_1) evaluated from input state quantities.
    """
    jax = pytest.importorskip('jax')
    from types import SimpleNamespace

    import jax.numpy as jnp

    from aragog.eos.entropy import EntropyEOS
    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.solver import BoundaryParams, _apply_cmb_bc
    from aragog.solver.entropy_solver import EntropySolver

    jax.config.update('jax_enable_x64', True)

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=1, n_nodes=20)
    p.boundary_conditions.cmb_flux_law = 'deschamps_sotin_2000'
    h_0 = 5.0e-11
    p.energy.tidal = True
    p.energy.tidal_array = np.full(20 - 1, h_0)
    s = EntropySolver(p, entropy_eos=EntropyEOS(EOS_DIR))
    s.initialize()
    y = np.full(s._n_stag, 2400.0)
    y[:3] = [2600.0, 2550.0, 2450.0]
    s.set_initial_entropy(y)
    s.dSdt(0.0, y)

    ps = s.state.phase_staggered
    arr = lambda f: jnp.asarray(np.asarray(f()).ravel())  # noqa: E731
    phase = SimpleNamespace(
        density=arr(ps.density),
        heat_capacity=arr(ps.heat_capacity),
        thermal_conductivity=arr(ps.thermal_conductivity),
        thermal_expansivity=arr(ps.thermal_expansivity),
        viscosity=arr(ps.viscosity),
        temperature=arr(ps.temperature),
    )
    mesh = SimpleNamespace(
        area=jnp.asarray(s._area_flat),
        volume=jnp.asarray(s._volume_flat),
        P_basic=jnp.asarray(s._P_basic_flat),
        radii_basic=jnp.asarray(s._r_basic_flat),
        gravity=jnp.asarray(s._g_basic_flat),
    )
    t_top = float(s.state.top_temperature.item())
    bc = BoundaryParams(
        outer_bc_type=1,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=1,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
        cmb_flux_law=True,
        cmb_law_interior=s._cmb_law_interior.astype(float),
    )

    hf = jnp.asarray(np.asarray(s.state.heat_flux).ravel()).at[:2].set(0.0)
    heating_first = float(np.asarray(s.state.heating).flat[0])
    out = _apply_cmb_bc(
        hf,
        bc,
        mesh,
        phase.density,
        phase.heat_capacity,
        heating_first,
        phase_stag_T=phase.temperature,
        phase_stag_k=phase.thermal_conductivity,
        law_inputs=(EntropyEOS_JAX(EOS_DIR), phase, jnp.asarray(y), t_top),
    )

    f_numpy = float(np.asarray(s.state.heat_flux).ravel()[0])
    f_jax = float(out[0])

    r_cmb = float(s._r_basic_flat[0])
    r_above = float(s._r_basic_flat[1])
    vol_0 = float(s._volume_flat[0])
    rho_0 = float(np.asarray(s.state.phase_staggered.density()).flat[0])
    cp_0 = float(np.asarray(s.state.phase_staggered.heat_capacity()).flat[0])
    c_cell = vol_0 * rho_0 * cp_0
    c_core = (4.0 / 3.0) * np.pi * r_cmb**3 * 10500.0 * 880.0
    alpha = (r_above / r_cmb) ** 2 / (c_cell / (c_core * 1.147) + 1.0)
    q_law = float(_expected(s, float(phase.temperature[0]), 1))
    a_1 = float(s._area_flat[1])
    q_first = heating_first * rho_0 * vol_0
    f_analytical = alpha * (q_law - q_first / a_1)

    assert f_jax == pytest.approx(f_numpy, rel=1e-10)
    assert f_jax == pytest.approx(f_analytical, rel=1e-10)
    assert f_numpy == pytest.approx(f_analytical, rel=1e-10)

    # Canary: omitting Q_first from the JAX path must shift the flux and fail.
    out_canary = _apply_cmb_bc(
        hf,
        bc,
        mesh,
        phase.density,
        phase.heat_capacity,
        0.0,
        phase_stag_T=phase.temperature,
        phase_stag_k=phase.thermal_conductivity,
        law_inputs=(EntropyEOS_JAX(EOS_DIR), phase, jnp.asarray(y), t_top),
    )
    f_canary = float(out_canary[0])
    assert abs(f_canary - f_analytical) > 1.0e-3
