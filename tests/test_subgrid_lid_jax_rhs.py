"""The JAX sub-grid lid against the NumPy lid of the solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aragog import surface_lid as sl
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
        want = (r.F_top, r.F_fluid, ev.V_fluid, r.ddelta, r.dE, ev.H_f, ev.m_lid)
        got = np.array([*got[:5], *got[5]], dtype=float)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-30)


def _lid_solver(core_bc='quasi_steady', rheology=None, heating=None, lid=True, convection=True):
    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS
    from tests.test_subgrid_lid_solver import _heat

    inner = 2 if core_bc == 'quasi_steady' else 1
    p = _build(core_bc=core_bc, outer_bc=1, inner_bc=inner, n_nodes=10, end_time=1.0)
    _heat(p, heating)
    p.boundary_conditions.subgrid_lid = lid
    if core_bc == 'core_module':
        p.boundary_conditions.core_module_params = CORE_MODULE_PARAMS
    p.boundary_conditions.normalize()
    p.energy.convection = convection
    if rheology is not None:
        p.phase_solid.rheology = _lid_cases()[rheology]
    # core_module: the expansivity of the JAX RHS, as tests/test_jax_dsdt_core_module.py
    eos = entropy_eos_copy(EOS_DIR, identity_alpha=core_bc == 'core_module')
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(3050.0)
    return s


def _evaluate(rhs_fn, jac_fn, y, scales):
    """The RHS and the Jacobian of a callback pair at the physical state ``y``."""
    x = (y - scales.state_offset) / scales.state_scale
    f, J = np.empty(y.size), np.empty((y.size, y.size))
    assert rhs_fn(0.0, x, f) == 0 and jac_fn(0.0, x, f, J) == 0
    return f, J


def _factory_rhs(s, lid_params, bc_lid=None):
    """The JAX CVODE callbacks of ``s`` from the factory, as PROTEUS builds them."""
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    bc, en, n = s.parameters.boundary_conditions, s.parameters.energy, s._n_stag
    radio = s.parameters.radionuclides if en.radionuclides else []
    keys = ('heat_production', 'abundance', 'concentration', 't0_years', 'half_life_years')
    tidal = np.broadcast_to(np.asarray(en.tidal_array, float), n) if en.tidal else np.zeros(n)
    core = s._core_bc == 'core_module'
    return build_jax_rhs_and_jacobian(
        eos_jax=entropy_eos_jax(EOS_DIR),
        phase_params=PhaseParams.from_parameters(s.parameters),
        mesh_arrays=MeshArrays.from_numpy_mesh(
            s.evaluator.mesh, validate_rheology=bool(s._rheology_enabled)
        ),
        boundary_params=BoundaryParams(
            outer_bc_type=1,
            outer_bc_value=0.0,
            emissivity=float(bc.emissivity),
            T_eq=float(bc.equilibrium_temperature),
            inner_bc_type=2 if s._core_bc == 'quasi_steady' else 1,
            inner_bc_value=0.0,
            core_density=float(s._core_density),
            core_heat_capacity=float(s._core_cp),
            tfac_core_avg=float(s._core_tfac),
            cmb_area=float(s._cmb_area),
            core_M=float(s._core_M),
            cmb_dr_cmb=float(s._cmb_dr_cmb),
            lid=bc_lid,
        ),
        heating_array=np.array(tidal),
        radio_isotope_params=tuple(np.array([getattr(r, k) for r in radio]) for k in keys)
        if radio
        else None,
        scales=s._build_nondim_scales(),
        core_bc_mode=s._core_bc,
        core_module_budget=s._core_module_budget if core else None,
        core_module_q_radio=float(s._core_module_q_radio) if core else 0.0,
        core_module_ra_crit_cmb=s._core_module_ra_crit_cmb if core else None,
        lid_params=lid_params,
    )


def _physical(rhs_fn, sc):
    def f(y, t=0.0):
        out = np.empty(y.size)
        assert rhs_fn(t / sc.t_ref, (y - sc.state_offset) / sc.state_scale, out) == 0
        return out / sc.rhs_scale

    return f


def _lid_states(s, seed=13, n=4):
    """Random mantle profiles with lids of 1 m to 300 m, and one handed-over lid."""
    rng = np.random.default_rng(seed)
    for i in range(n + 1):
        y = s._S0.copy()
        y[: s._n_stag] = np.linspace(
            rng.uniform(2900, 3100), rng.uniform(2300, 3100), s._n_stag
        )
        d = 10.0 ** rng.uniform(0.0, 2.5) if i < n else 0.0
        y[-2:] = (d, d * float(s._lid_hv.H(rng.uniform(600.0, 1300.0))))
        yield y


PARITY = [
    ('quasi_steady', None, None),
    ('quasi_steady', 'lid_closure', None),
    ('quasi_steady', 'fixed_local', 'tidal'),
    ('quasi_steady', 'arrhenius_local', 'radio'),
    ('energy_balance', None, 'tidal'),
    ('energy_balance', 'lid_closure', 'radio'),
    ('core_module', None, None),
    ('core_module', 'lid_closure', 'tidal'),
]


@pytest.mark.parametrize(('core_bc', 'rheology', 'heating'), PARITY)
def test_jax_rhs_matches_the_numpy_rhs_with_the_lid(core_bc, rheology, heating):
    """The JAX factory right-hand side with the lid equals the NumPy one on active and
    handed-over lids: the top cell and the 2 lid rates to relative 1e-12, the other slots to
    1e-12 of the largest mantle rate. The call counter proves the JAX path ran."""
    _assert_rhs_parity(_lid_solver(core_bc, rheology, heating))


def test_jax_rhs_matches_the_numpy_rhs_with_the_lid_without_convection():
    """With ``energy.convection`` off neither right-hand side gives the lid base the
    boundary-layer flux: the lid rates agree to relative 1e-12. The top-cell rate is then the
    small difference of two fluxes of about 1.6e-3 W/m^2
    and agrees to 1e-8 (measured 1.1e-9)."""
    _assert_rhs_parity(_lid_solver(convection=False), rtol_top=1e-8)


def _assert_rhs_parity(s, rtol_top=1e-12):
    rhs_fn, _, info = _factory_rhs(s, s.jax_lid_params())
    assert info['lid'] and rhs_fn.lid
    f, n = _physical(rhs_fn, s._build_nondim_scales()), 0
    for y in _lid_states(s):
        want = np.asarray(s._dSdt_single(0.0, y), dtype=float)
        got = f(y)
        n += 1
        assert got.size == y.size
        lid = np.isin(np.arange(y.size), [s._n_stag - 1, y.size - 2, y.size - 1])
        np.testing.assert_allclose(got[-2:], want[-2:], rtol=1e-12, atol=1e-300)
        top = s._n_stag - 1
        np.testing.assert_allclose(got[top], want[top], rtol=rtol_top, atol=1e-300)
        rest, atol = ~lid, 1e-12 * np.abs(want[: s._n_stag]).max()
        np.testing.assert_allclose(got[rest], want[rest], rtol=1e-12, atol=atol)
    assert info['rhs_calls'] == n


@pytest.mark.physics_invariant
@pytest.mark.parametrize('cap', ['fixed', 'rate'])
def test_lid_hands_over_inside_a_call_on_the_jax_factory(cap):
    """Without the solid rheology a top cell 0.2 J/kg/K above its solidus cools through it
    inside a 1-yr CVODE call on the JAX factory, with no root replaced: the lid hands over and
    ends at exactly 0, and the step ledger closes to 1e-10 of the surface power."""
    from aragog.solver import entropy_solver as es

    if not es._CVODE_AVAILABLE:
        pytest.skip('scikits.odes is not installed')
    s = _lid_solver()
    s.parameters.energy.solver_method = 'cvode'
    s.parameters.energy.use_jax_jacobian = True
    s.parameters.energy.phase_boundary_cap = cap
    S = s._S0[: s._n_stag].copy()
    S[-1] = s._lid_S_sol + 0.2
    s.set_initial_entropy(S)
    info = {}

    def factory(scales, mode):
        rhs_fn, jac_fn, info['info'] = _factory_rhs(s, s.jax_lid_params())
        return rhs_fn, jac_fn

    s.set_jax_cvode_factory(factory)
    s.solve()
    assert s._solution.status == 0 and info['info']['rhs_calls'] > 0
    assert 0.0 < s.get_state().lid_handover_time < 1.0
    assert s._solution.y[-2, 0] > 0.0 and np.all(s._solution.y[-2:, -1] == 0.0)
    ei = s._solution.energy_integrals
    assert abs(ei['solver_residual']) <= 1e-10 * abs(ei['F_int'])


@pytest.mark.parametrize(
    ('core_bc', 'rheology', 'heating'),
    [PARITY[0], PARITY[1], ('energy_balance', 'arrhenius_local', 'tidal')],
)
def test_jax_jacobian_lid_rows_match_central_differences(core_bc, rheology, heating):
    """The JAX Jacobian rows of the 2 lid slots and the top cell equal central differences of
    the JAX right-hand side (measured: lid rows 3.5e-9, top row 1.5e-6 at its best step)."""
    s = _lid_solver(core_bc, rheology, heating)
    rhs_fn, jac_fn, _ = _factory_rhs(s, s.jax_lid_params())
    sc = s._build_nondim_scales()

    def fd(x, rel):
        J = np.empty((x.size, x.size))
        for j in range(x.size):
            h, a, b = rel * max(abs(x[j]), 1e-12), np.empty(x.size), np.empty(x.size)
            rhs_fn(0.0, x + h * (np.arange(x.size) == j), a)
            rhs_fn(0.0, x - h * (np.arange(x.size) == j), b)
            J[:, j] = (a - b) / (2.0 * h)
        return J

    def row_err(J, Jfd, r):
        return np.abs(J[r] - Jfd[r]).max() / np.abs(J[r]).max()

    states = list(_lid_states(s, n=3))
    J = np.zeros((s._S0.size,) * 2)
    assert jac_fn(0.0, (states[-1] - sc.state_offset) / sc.state_scale, J[0], J) == 0
    assert np.all(np.isfinite(J)), 'the Jacobian of a handed-over lid (0, 0) is not finite'
    for y in states[:3]:
        x = (y - sc.state_offset) / sc.state_scale
        J = np.zeros((x.size, x.size))
        assert jac_fn(0.0, x, np.zeros(x.size), J) == 0
        Jfd = {rel: fd(x, rel) for rel in (1e-4, 1e-5, 1e-6)}
        assert max(row_err(J, Jfd[1e-6], r) for r in (-2, -1)) < 1e-7
        assert min(row_err(J, Jfd[rel], s._n_stag - 1) for rel in Jfd) < 1e-5


def _cvode_solver(lid, factory_lid):
    """A lid or no-lid solver on CVODE whose JAX factory is built with or without the lid."""
    from aragog.solver import entropy_solver as es

    if not es._CVODE_AVAILABLE:
        pytest.skip('scikits.odes is not installed')
    s = _lid_solver(lid=lid)
    s.parameters.energy.solver_method = 'cvode'
    s.parameters.energy.use_jax_jacobian = True
    lp = _lid_solver().jax_lid_params() if factory_lid else None
    s.set_jax_cvode_factory(lambda scales, mode: _factory_rhs(s, lp)[:2])
    return s


def test_lid_solve_refuses_a_factory_without_lid_params():
    """A lid solve with a JAX factory built without the lid raises and names the fix."""
    s = _cvode_solver(lid=True, factory_lid=False)
    with pytest.raises(
        ValueError, match=r'^JAX CVODE factory: .*pass lid_params=solver\.jax_lid_params\(\)'
    ):
        s.solve()


def test_factory_without_lid_params_drops_a_lid_of_the_boundary_params():
    """``lid_params`` decides the lid of the factory: with None, a lid left in the boundary
    parameters gives the RHS and the Jacobian of the factory built without it."""
    lp = _lid_solver().jax_lid_params()
    s = _lid_solver(lid=False)
    r0, j0, _ = _factory_rhs(s, None)
    r1, j1, _ = _factory_rhs(s, None, bc_lid=lp)
    assert r1.lid is False
    sc = s._build_nondim_scales()
    (a, Ja), (b, Jb) = (_evaluate(r, j, s._S0, sc) for r, j in ((r0, j0), (r1, j1)))
    np.testing.assert_array_equal(b, a)
    np.testing.assert_array_equal(Jb, Ja)


def test_factory_with_the_lid_already_in_the_boundary_params_gives_the_same_rhs():
    """A ``lid_params`` that is the lid of the boundary parameters gives the RHS and the
    Jacobian of the factory built with it alone."""
    s = _lid_solver()
    lp = s.jax_lid_params()
    r0, j0, _ = _factory_rhs(s, lp)
    r1, j1, _ = _factory_rhs(s, lp, bc_lid=lp)
    assert r1.lid is True
    sc = s._build_nondim_scales()
    (a, Ja), (b, Jb) = (_evaluate(r, j, s._S0, sc) for r, j in ((r0, j0), (r1, j1)))
    np.testing.assert_array_equal(b, a)
    np.testing.assert_array_equal(Jb, Ja)


def test_solve_without_the_lid_refuses_a_factory_with_lid_params():
    """A solve without the lid and a JAX factory built with the lid raises; it never drops
    the 2 lid slots."""
    s = _cvode_solver(lid=False, factory_lid=True)
    with pytest.raises(
        ValueError,
        match=r'^JAX CVODE factory: .*runs without the sub-grid lid; pass lid_params=None',
    ):
        s.solve()


@pytest.mark.parametrize('lid', [True, False])
def test_solve_refuses_a_foreign_factory_that_disagrees_on_the_lid(lid):
    """A factory of another origin whose right-hand side does not state the solver's lid
    flag is refused before the solve."""
    from aragog.solver import entropy_solver as es

    if not es._CVODE_AVAILABLE:
        pytest.skip('scikits.odes is not installed')
    s = _lid_solver(lid=lid)
    s.parameters.energy.solver_method = 'cvode'
    s.parameters.energy.use_jax_jacobian = True

    def rhs(t, y, ydot):
        pytest.fail('the refused right-hand side must not run')

    rhs.lid = not lid
    s.set_jax_cvode_factory(lambda scales, mode: (rhs, None))
    with pytest.raises(ValueError, match='disagree on the sub-grid lid'):
        s.solve()


def test_short_lid_solve_on_the_jax_factory_matches_the_numpy_solve():
    """A 1-yr CVODE solve with a 30 m lid on the JAX factory ends where the NumPy CVODE solve
    with the finite-difference Jacobian ends, to 10 times the solver rtol of 1e-6."""
    from aragog.solver import entropy_solver as es

    if not es._CVODE_AVAILABLE:
        pytest.skip('scikits.odes is not installed')
    ends, calls = [], []
    for jax_on in (True, False):
        s = _lid_solver()
        s.parameters.energy.solver_method = 'cvode'
        s.parameters.energy.use_jax_jacobian = jax_on
        s.set_initial_lid(30.0, 30.0 * float(s._lid_hv.H(900.0)))
        s.set_initial_entropy(3050.0)
        if jax_on:
            info = {}

            def factory(scales, mode, s=s, info=info):
                rhs_fn, jac_fn, info['info'] = _factory_rhs(s, s.jax_lid_params())
                return rhs_fn, jac_fn

            s.set_jax_cvode_factory(factory)
        s.solve()
        assert s._solution.status == 0
        ends.append(np.asarray(s._solution.y, dtype=float)[:, -1])
        calls.append(info['info']['rhs_calls'] if jax_on else 0)
    assert calls[0] > 0
    assert ends[0][-2] > 0.0
    np.testing.assert_allclose(ends[0], ends[1], rtol=1e-5)


@pytest.mark.parametrize(
    ('core_bc', 'rheology', 'heating'),
    [
        ('quasi_steady', None, 'tidal'),
        ('quasi_steady', 'lid_closure', 'radio'),
        ('energy_balance', None, 'radio'),
        ('core_module', 'lid_closure', 'tidal'),
    ],
)
def test_jax_step_powers_match_the_numpy_ledger_with_the_lid(core_bc, rheology, heating):
    """The batched JAX step powers with the lid equal the NumPy step powers (measured 7.1e-12)
    and close the entropy ledger with the lid energy and the lid mass heated (residual 3e-16
    of the largest power), on active and handed-over lids with tidal or radiogenic heat."""
    from aragog.jax.solver import StepPowersAux

    s = _lid_solver(core_bc, rheology, heating)
    rhs_fn, _, _ = _factory_rhs(s, s.jax_lid_params())
    vol, rb = s._volume_flat, s._r_basic_flat
    aux = StepPowersAux(
        A_int=4.0 * np.pi * float(rb[-1]) ** 2,
        A_cmb=4.0 * np.pi * float(rb[0]) ** 2,
        volume=vol,
        mass_struct=np.asarray(s.evaluator.mesh.staggered_effective_density).ravel() * vol,
        P_stag=s._P_stag_flat,
    )
    Y = np.array(list(_lid_states(s))).T
    got = np.asarray(rhs_fn.step_powers(np.zeros(Y.shape[1]), Y, aux))
    want = np.array([s._step_powers(0.0, y) for y in Y.T])
    np.testing.assert_allclose(got[:, :6], want[:, :6], rtol=1e-10, atol=1e-300)
    scale = np.abs(want[:, :6]).max(axis=1)
    assert np.all(np.abs(got[:, 6]) <= 1e-12 * scale)


def test_a_handed_over_lid_skips_the_lid_and_equals_the_rhs_without_it(monkeypatch):
    """A zero lid does not evaluate ``lid_top``, a live lid does once. The RHS and the Jacobian
    of the mantle equal those without the lid, the lid rows and columns are zero and the
    derivatives stay finite on both sides of the hand-over."""
    from aragog.jax import surface_lid as jl

    s, free = _lid_solver(), _lid_solver(lid=False)
    (rhs, jac, _), (rhs_free, jac_free, _) = (
        _factory_rhs(s, s.jax_lid_params()),
        _factory_rhs(free, None),
    )
    sc, sc_free, n = s._build_nondim_scales(), free._build_nondim_scales(), s._S0.size - 2

    y = s._S0.copy()
    y[-2:] = 0.0
    f, J = _evaluate(rhs, jac, y, sc)
    f_free, J_free = _evaluate(rhs_free, jac_free, y[:n], sc_free)
    np.testing.assert_allclose(f[:n], f_free, rtol=1e-13)
    np.testing.assert_allclose(J[:n, :n], J_free, rtol=1e-13, atol=1e-13 * np.abs(J_free).max())
    assert np.all(f[n:] == 0.0) and np.all(J[n:] == 0.0) and np.all(J[:, n:] == 0.0)
    live = y.copy()
    live[n - 1] = float(s._lid_S_sol) + 1e-9
    live[-2:] = 0.5, 0.5 * float(s._lid_hv.H(1000.0))
    assert np.all(np.isfinite(_evaluate(rhs, jac, live, sc)[1]))

    calls, top = [0], jl.lid_top

    def counted(*a):
        calls[0] += 1
        return top(*a)

    monkeypatch.setattr(jl, 'lid_top', counted)
    out = np.empty(y.size)
    with jax.disable_jit():
        assert rhs(0.0, (y - sc.state_offset) / sc.state_scale, out) == 0 and calls[0] == 0
        assert rhs(0.0, (live - sc.state_offset) / sc.state_scale, out) == 0 and calls[0] == 1


CLOSURE_BOUND = 1.0e-9  # measured 5.9e-12 on Radau, 6.2e-11 on the JAX path


def _heat_of_the_residual(s, source, entropy):
    """Power [W] of a heating array as the entropy residual has it after an RHS of ``s``. It
    holds a state where the table density equals the phase density; in the mush the output
    totals differ from it."""
    rho = np.asarray(s.state.phase_staggered.density()).ravel()
    table = np.asarray(s.entropy_eos.density(s._P_stag_flat, entropy)).ravel()
    np.testing.assert_allclose(rho, table, rtol=1e-12)
    vol = s._volume_flat.copy()
    vol[-1] = s._lid_last.V_fluid
    return float(np.dot(source, rho * vol)) + source[-1] * s._lid_area * s._lid_last.m_lid


@pytest.mark.parametrize(('heating', 'column'), [('radio', 2), ('tidal', 3)])
def test_source_powers_with_a_live_lid_are_the_heating_of_the_residual(heating, column):
    """With a live lid the top cell is heated through its fluid volume plus the lid mass. The
    NumPy and JAX source powers report that heating, not the whole top cell at the fluid
    density, and the frozen-mass powers stay the structural mass."""
    from aragog.jax.solver import StepPowersAux

    s = _lid_solver('quasi_steady', None, heating)
    rhs_fn, _, _ = _factory_rhs(s, s.jax_lid_params())
    vol = s._volume_flat
    mass_struct = np.asarray(s.evaluator.mesh.staggered_effective_density).ravel() * vol
    aux = StepPowersAux(
        A_int=4.0 * np.pi * float(s._r_basic_flat[-1]) ** 2,
        A_cmb=4.0 * np.pi * float(s._r_basic_flat[0]) ** 2,
        volume=vol,
        mass_struct=mass_struct,
        P_stag=s._P_stag_flat,
    )
    y = s._S0.copy()
    y[-2:] = 300.0, 300.0 * float(s._lid_hv.H(1000.0))
    numpy_powers = s._step_powers(0.0, y)
    name = 'heating_radio' if heating == 'radio' else 'heating_tidal'
    source = np.asarray(getattr(s.state, name)).ravel()
    want = _heat_of_the_residual(s, source, y[: s._n_stag])
    whole_cell = float(
        np.dot(source, np.asarray(s.state.phase_staggered.density()).ravel() * vol)
    )
    assert abs(want / whole_cell - 1.0) > 1e-7
    np.testing.assert_allclose(numpy_powers[column], want, rtol=1e-13)
    jax_powers = np.asarray(rhs_fn.step_powers(np.zeros(1), y[:, None], aux))[0]
    np.testing.assert_allclose(jax_powers[column], want, rtol=1e-10)
    frozen = float(np.dot(source, mass_struct))
    assert numpy_powers[column + 2] == frozen
    np.testing.assert_allclose(jax_powers[column + 2], frozen, rtol=1e-12)


@pytest.mark.parametrize('heating', ['tidal', 'radio', 'radio_and_tidal'])
@pytest.mark.parametrize('method', ['radau', 'jax'])
def test_output_source_totals_with_a_live_lid_are_the_heating_of_the_residual(method, heating):
    """``Q_radio_total``, ``Q_tidal_total`` and ``F_heat_total`` of the output of a solve that
    ends with a live lid count the fluid part of the top cell plus the lid mass, on the NumPy
    and JAX paths."""
    if method == 'jax':
        s = _jax_cvode_lid_solver(end=1.0e-3, heating=heating)
    else:
        s = _lid_solver('quasi_steady', None, heating)
        s.parameters.solver.end_time = 1.0e-3
    s.set_initial_lid(300.0, 300.0 * float(s._lid_hv.H(1000.0)))
    s.set_initial_entropy(3050.0)
    s.solve()
    assert s._solution.status == 0
    out = s.get_state()
    assert out.lid_delta > 0.0
    entropy = s._solution.y[: s._n_stag, -1]
    radio = _heat_of_the_residual(s, np.asarray(s.state.heating_radio).ravel(), entropy)
    tidal = _heat_of_the_residual(s, np.asarray(s.state.heating_tidal).ravel(), entropy)
    assert (radio > 0.0) == ('radio' in heating) and (tidal > 0.0) == ('tidal' in heating)
    np.testing.assert_allclose(out.Q_radio_total, radio, rtol=1e-13)
    np.testing.assert_allclose(out.Q_tidal_total, tidal, rtol=1e-13)
    area = 4.0 * np.pi * float(s._r_basic_flat[-1]) ** 2
    np.testing.assert_allclose(out.F_heat_total * area, radio + tidal, rtol=1e-13)


@pytest.mark.parametrize('method', ['radau', 'jax'])
def test_energy_integrals_of_a_live_lid_call_close_with_large_tidal_heat(method):
    """The state heat of a call with a live lid equals the integrated surface, core and source
    powers. The source powers are read from the output and do not repeat the residual formula."""
    if method == 'jax':
        s = _jax_cvode_lid_solver(end=0.05, heating='tidal')
    else:
        s = _lid_solver(heating='tidal')
        s.parameters.solver.end_time = 0.05
    s.parameters.energy.tidal_array = s.state._tidal_array = np.array([3.0e-4])
    s.set_initial_lid(300.0, 300.0 * float(s._lid_hv.H(1000.0)))
    s.set_initial_entropy(3050.0)
    s.solve()
    assert s._solution.status == 0 and s._solution.y[-2, -1] > 0.0
    ei = s._solution.energy_integrals
    gap = ei['state_heat'] - (ei['F_int'] + ei['F_cmb'] + ei['Q_radio'] + ei['Q_tidal'])
    assert abs(gap) <= CLOSURE_BOUND * abs(ei['Q_tidal'])


def _jax_cvode_lid_solver(end=1.0, heating=None):
    """A lid solver on CVODE with the JAX factory built with the lid."""
    from aragog.solver import entropy_solver as es

    if not es._CVODE_AVAILABLE:
        pytest.skip('scikits.odes is not installed')
    s = _lid_solver(heating=heating)
    s.parameters.energy.solver_method = 'cvode'
    s.parameters.energy.use_jax_jacobian = True
    s.parameters.solver.end_time = end
    s.set_jax_cvode_factory(lambda scales, mode: _factory_rhs(s, s.jax_lid_params())[:2])
    return s


def test_a_guard_cut_removes_the_hand_over_time_on_the_jax_path():
    """A table exit on the JAX CVODE path whose kept states end with a live lid fails the call
    without the time of the hand-over it cut away."""
    s = _jax_cvode_lid_solver(end=10.0)
    S = np.full(s._n_stag, 3050.0)
    S[-1], S[0] = s._lid_S_sol + 1e-4, float(s.entropy_eos.S_max) + 2e-3
    s.set_initial_entropy(S)
    s.solve()
    assert s._solution.status == -1 and 'EOS table guard' in str(s._solution.message)
    out = s.get_state()
    assert out.lid_delta > 0.0
    assert np.isnan(out.lid_handover_time)


def test_lid_step_check_runs_on_the_jax_batch_path():
    """The lid invariant fails an accepted state past the growth stop on the JAX batch path,
    as on the NumPy loop."""
    from tests.test_subgrid_lid_solver import _phantom_energy

    s = _jax_cvode_lid_solver()
    d = 1.01 * sl.LID_CAP_FRAC * s._lid_h_top
    E, _ = _phantom_energy(s, d, 0.5 * (s._lid_T_sol + 300.0))
    s.set_initial_lid(d, E)
    s.set_initial_entropy(3050.0)
    with pytest.raises(RuntimeError, match='lid invariant violated at step') as err:
        s.solve()
    assert 'outside' in str(err.value)
    assert s._batch_powers_calls >= 1


def test_growth_stop_warns_once_on_the_jax_batch_path(caplog):
    """A lid held at the growth stop warns once on the JAX batch path, as on the NumPy loop."""
    import logging

    from tests.test_subgrid_lid_solver import _phantom_energy

    s = _jax_cvode_lid_solver(end=10.0)
    d = (sl.LID_CAP_FRAC - 0.5 * sl.LID_CAP_WIDTH) * s._lid_h_top
    E, _ = _phantom_energy(s, d, 257.0)
    s.set_initial_lid(d, E)
    s.set_initial_entropy(3050.0)
    with caplog.at_level(logging.WARNING, logger='fwl.aragog.solver.entropy_solver'):
        s.solve()
    assert s._batch_powers_calls > 0
    msgs = [r.getMessage() for r in caplog.records if 'growth stop holds' in r.getMessage()]
    assert len(msgs) == 1


def test_enthalpy_floor_warns_on_the_jax_batch_path(caplog):
    """A top cell just above the solidus, the lid-base entropy without the rheology, puts the
    Stefan enthalpy jump under its floor; the JAX batch path warns once, like the NumPy loop."""
    import logging

    s = _jax_cvode_lid_solver()
    s.set_initial_lid(30.0, 30.0 * float(s._lid_hv.H(900.0)))
    s.set_initial_entropy(s._lid_S_sol + 0.1)
    with caplog.at_level(logging.WARNING, logger='fwl.aragog.solver.entropy_solver'):
        s.solve()
    assert s._batch_powers_calls > 0
    msgs = [r.getMessage() for r in caplog.records if 'below the floor' in r.getMessage()]
    assert len(msgs) == 1


def test_entropy_outside_the_heat_table_fails_the_jax_batch_path():
    """An accepted state whose fluid entropy lies past the end of the heat table fails the lid
    invariant on the JAX batch path; the right-hand side only continues the table there."""
    from scipy.interpolate import PchipInterpolator

    s = _jax_cvode_lid_solver()
    pp = s._lid_W_of_S
    solid = pp.x <= s._lid_S_sol
    s._lid_W_of_S = PchipInterpolator(pp.x[solid], pp(pp.x[solid]))
    s.set_initial_entropy(3050.0)
    with pytest.raises(Exception) as err:
        s.solve()
    assert isinstance(err.value, RuntimeError), repr(err.value)
    assert 'outside the lid heat table' in str(err.value)
    assert s._batch_powers_calls >= 1


@pytest.mark.parametrize('end', [0, -1], ids=['bottom', 'top'])
@pytest.mark.parametrize(
    ('past', 'refused'), [(5e-4, False), (2e-3, True)], ids=['in_the_band', 'above_the_band']
)
def test_heat_table_check_of_the_jax_batch_path_has_the_guard_band(end, past, refused):
    """On the trace of the JAX batch path a fluid entropy past a table end passes within the
    tolerance of the EOS table guard, and fails beyond it."""
    s = _jax_cvode_lid_solver()
    s.solve()
    sol = s._solution
    t, y = (np.array(v, dtype=float) for v in (sol.get('energy_trace') or (sol.t, sol.y)))
    sign = -1.0 if end == 0 else 1.0
    y[s._n_stag - 1, t.size // 2] = float(s._lid_W_of_S.x[end]) + sign * past
    sol['energy_trace'] = (t, y)
    if refused:
        with pytest.raises(RuntimeError, match='outside the lid heat table'):
            s._compute_step_energy_integrals()
    else:
        calls = s._batch_powers_calls
        s._compute_step_energy_integrals()
        assert s._batch_powers_calls > calls
        ints = sol.energy_integrals
        assert all(math.isfinite(v) for k, v in ints.items() if k != 'core_tol')
        assert math.isnan(ints['core_tol'])  # no core tolerance scale outside core_module


def test_intermediate_entropy_outside_the_heat_table_fails_the_jax_batch_path():
    """The JAX batch path checks the fluid entropy of every accepted state against the heat
    table, not only the last one: one state in the middle of the trace past the table top
    fails the lid invariant, as on the NumPy loop."""
    s = _jax_cvode_lid_solver()
    s.solve()
    sol = s._solution
    t, y = (np.array(v, dtype=float) for v in (sol.get('energy_trace') or (sol.t, sol.y)))
    hi, k = float(s._lid_W_of_S.x[-1]), t.size // 2
    y[s._n_stag - 1, k] = hi + 100.0
    assert s._stag_entropy(y[:, k])[-1] > hi >= s._stag_entropy(y[:, -1])[-1]
    sol.energy_trace = (t, y)
    calls = s._batch_powers_calls
    with pytest.raises(RuntimeError, match='outside the lid heat table'):
        s._compute_step_energy_integrals()
    assert s._batch_powers_calls == calls + 1


@pytest.mark.parametrize('lid', ['before', 'after'])
def test_handover_call_checks_the_lid_states_on_the_jax_batch_path(lid):
    """In a call that ends with the lid handed over, the JAX batch path checks the fluid
    entropy of the states that still carry a lid against the heat table, and not of the
    states after the hand-over."""
    s = _jax_cvode_lid_solver()
    s.solve()
    sol = s._solution
    t, y = (np.array(v, dtype=float) for v in (sol.get('energy_trace') or (sol.t, sol.y)))
    hi, k = float(s._lid_W_of_S.x[-1]), t.size // 2
    y[s._n_stag - 1, k] = hi + 100.0
    y[-2:, k + 1 if lid == 'before' else k :] = 0.0
    sol.energy_trace = (t, y)
    calls = s._batch_powers_calls
    if lid == 'before':
        with pytest.raises(RuntimeError, match='outside the lid heat table'):
            s._compute_step_energy_integrals()
    else:
        try:
            s._compute_step_energy_integrals()
        except RuntimeError as err:
            pytest.fail(f'batch path checked a handed-over state: {err}')
    assert s._batch_powers_calls == calls + 1


def test_entropy_on_the_heat_table_ends_passes_the_jax_batch_path():
    """Accepted states whose fluid entropy lies exactly on an end of the heat table pass the
    range check of the JAX batch path."""
    s = _jax_cvode_lid_solver()
    s.solve()
    sol = s._solution
    t, y = (np.array(v, dtype=float) for v in (sol.get('energy_trace') or (sol.t, sol.y)))
    lo, hi = s._lid_W_of_S.x[[0, -1]]
    k = t.size // 2
    y[s._n_stag - 1, k - 1 : k + 1] = (lo, hi)
    assert [s._stag_entropy(y[:, i])[-1] for i in (k - 1, k)] == [lo, hi]
    sol.energy_trace = (t, y)
    calls = s._batch_powers_calls
    try:
        s._compute_step_energy_integrals()
    except RuntimeError as err:
        pytest.fail(f'batch path refused a table end: {err}')
    assert s._batch_powers_calls == calls + 1


def test_jax_base_flux_is_never_negative(monkeypatch):
    """With no convective flux, a face flux out of the top cell into the mesh gives the JAX lid
    the base flux 0: the same lid as a zero face flux."""
    from aragog.jax import surface_lid as jl

    s = _lid_solver()
    y = s._S0.copy()
    y[-2:] = (30.0, 30.0 * float(s._lid_hv.H(900.0)))
    s._dSdt_single(0.0, y)
    ph = s.state.phase_staggered

    def top(v):
        return float(np.asarray(v).ravel()[-1])

    props = {
        k: top(getattr(ph, f)())
        for k, f in dict(
            T='temperature',
            rho='density',
            cp='heat_capacity',
            k='thermal_conductivity',
            alpha='thermal_expansivity',
            g='gravitational_acceleration',
            eta='viscosity',
            phi='melt_fraction',
        ).items()
    }
    monkeypatch.setattr(jl, 'convective_base_flux', lambda *a: -1.0)
    lp, eos_jax = s.jax_lid_params(), entropy_eos_jax(EOS_DIR)

    def lid(F_in):
        out = jl.lid_top(
            lp,
            eos_jax,
            y[-2:],
            float(y[s._n_stag - 1]),
            props,
            F_in,
            float(s.state._heat_flux[-1]),
            np.nan,
            0.0,
        )
        return np.array([*out[:5], *out[5]], dtype=float)

    np.testing.assert_array_equal(lid(-100.0), lid(0.0))
    assert not np.array_equal(lid(50.0), lid(0.0))


@pytest.mark.parametrize('case', ['growth', 'melt_back', 'start_below'])
def test_intermediate_lid_past_a_stop_fails_the_jax_batch_path(case):
    """The JAX batch path checks every accepted state, not only the last: one state in the
    middle of the trace past the growth stop with its slack, or thinner than the melt-back
    stop, fails the lid invariant at its step. A lid that starts the call below ``delta_lo``
    passes, and the lower bound is then its start thickness."""
    s = _jax_cvode_lid_solver()
    s.solve()
    sol = s._solution
    t, y = (np.array(v, dtype=float) for v in (sol.get('energy_trace') or (sol.t, sol.y)))
    k = t.size // 2
    assert y[-2, k] > 0.0
    if case == 'growth':
        y[-2, k] = sl.LID_CAP_FRAC * s._lid_h_top * (1.0 + 2e-3)
    elif case == 'melt_back':
        y[-2, k] = 0.5 * min(s._lid_delta_lo, y[-2, 0])
    else:
        y[-2] *= 0.8 * s._lid_delta_lo / y[-2, 0]
        sol.energy_trace = (t, y)
        calls = s._batch_powers_calls
        s._compute_step_energy_integrals()
        assert s._batch_powers_calls == calls + 1
        y[-2, k] = y[-2, 0] * (1.0 - 2e-6)
    sol.energy_trace = (t, y)
    calls = s._batch_powers_calls
    with pytest.raises(RuntimeError, match=f'at step {k},.*thickness'):
        s._compute_step_energy_integrals()
    assert s._batch_powers_calls == calls + 1


def test_jax_path_records_the_largest_lid_excess_of_the_call():
    """On the JAX batch path lid_cap_excess is the largest delta / cap - 1 over the accepted
    states of the call, here at the start of a lid set above the growth stop."""
    from tests.test_subgrid_lid_solver import _phantom_energy

    s = _jax_cvode_lid_solver(end=100.0)
    cap = sl.LID_CAP_FRAC * s._lid_h_top
    E, _ = _phantom_energy(s, cap * (1.0 + 5e-4), 2140.0)
    s.set_initial_lid(cap * (1.0 + 5e-4), E)
    s.set_initial_entropy(3050.0)
    s.solve()
    sol = s._solution
    _, y = sol.get('energy_trace') or (sol.t, sol.y)
    d = np.asarray(y, dtype=float)[-2]
    assert s._batch_powers_calls >= 1
    assert sol.lid_cap_excess == np.max(d[d > 0.0]) / cap - 1.0
    assert sol.lid_cap_excess == pytest.approx(5e-4, rel=1e-9)


def test_lid_params_follows_the_positional_arguments_of_main():
    """A caller that passes ``start_time`` by position, as the signature of main allows, still
    binds it to ``start_time``: the lid argument comes last and has a keyword only."""
    import inspect

    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    params = inspect.signature(build_jax_rhs_and_jacobian).parameters
    names = list(params)
    assert names.index('start_time') == names.index('core_module_ra_crit_cmb') + 1
    assert params['lid_params'].kind is inspect.Parameter.KEYWORD_ONLY
