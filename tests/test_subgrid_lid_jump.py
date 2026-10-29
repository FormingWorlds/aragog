"""The Stefan jump of the sub-grid lid on both paths: the heat per volume the fluid releases as
it joins the lid, ``int_{S_b}^{S_f} rho T dS`` at the top-cell pressure."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.integrate import quad

from aragog import surface_lid as sl
from tests.conftest import entropy_eos_copy, entropy_eos_jax
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, _build, needs_eos

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60), needs_eos]

# Jump tolerance: 1e-6 relative plus 0.1 J/kg per base density.
REL, ABS_PER_MASS = 1.0e-6, 0.1


@pytest.fixture(scope='module')
def eos():
    return entropy_eos_copy(EOS_DIR)


def _solver(eos, rheology, S_top):
    """A 10-node lid column and a state ``y`` with the top-cell entropy ``S_top``; with the
    solid rheology the interior is solid at 0.3 of the way down from the solidus to the table
    floor, so the lid base lies below the solidus."""
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=10, end_time=1.0)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.normalize()
    if rheology:
        p.phase_solid.rheology = SolidRheologyParams(enabled=True)
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(3150.0)
    y = s._S0.copy()
    if rheology:
        y[: s._n_stag] = s._lid_S_sol - 0.3 * (s._lid_S_sol - float(eos.S_min))
    y[s._n_stag - 1] = S_top(s) if callable(S_top) else S_top
    y[-2:] = (30.0, 30.0 * float(s._lid_hv.H(900.0)))
    return s, y


def _reference(s, S_b, S_f):
    """``int_{S_b}^{S_f} rho T dS`` [J/m^3] at the top-cell pressure, scipy quad split at the
    entropy knots of the EOS tables, the solidus and the liquidus (smooth between them)."""
    eos, P = s.entropy_eos, s._lid_P

    def rhoT(S):
        return float(np.asarray(eos.density(P, S) * eos.temperature(P, S)).item())

    lo, hi = sorted((S_b, S_f))
    knots = np.concatenate(
        [t['S'] for t in eos._tables.values()] + [[s._lid_S_sol, s._lid_S_liq]]
    )
    cuts = np.unique(np.r_[lo, knots[(knots > lo) & (knots < hi)], hi])
    total = sum(
        quad(rhoT, a, b, epsabs=0.0, epsrel=1e-9, limit=200)[0] for a, b in zip(cuts, cuts[1:])
    )
    return total if S_f >= S_b else -total


def _numpy_jump(s, y, monkeypatch, same=False):
    """``(rho dh, S_b, S_f)`` the NumPy RHS passes to the Stefan rate; ``same`` puts the lid
    base at the fluid entropy."""
    seen = {}
    rate, base_state = sl.stefan_rate, s._lid_base_state

    def spy(q_b, F_b, rho, dh, *args):
        seen.update(rho=rho, dh=dh)
        return rate(q_b, F_b, rho, dh, *args)

    def base(S_f, T_f, phi_f):
        out = base_state(S_f, T_f, phi_f)
        return {**out, 'S_b': S_f} if same else out

    monkeypatch.setattr(sl, 'stefan_rate', spy)
    monkeypatch.setattr(s, '_lid_base_state', base)
    s._dSdt_single(0.0, y)
    return seen['rho'] * seen['dh'], s._lid_last.S_b, float(y[s._n_stag - 1])


def _jax_jump(s, y, monkeypatch, same=False):
    """``(rho dh, S_b, S_f)`` the JAX lid passes to the Stefan rate at the NumPy top-cell state."""
    pytest.importorskip('jax')
    from aragog.jax import surface_lid as jl

    s._dSdt_single(0.0, y)
    ph = s.state.phase_staggered

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
    seen = {}
    rate, lid_base = jl.stefan_rate, jl.lid_base

    def spy(q_b, F_b, rho, dh, *args):
        seen.update(rho=float(rho), dh=float(dh))
        return rate(q_b, F_b, rho, dh, *args)

    def base(lp, eos, S_f, *args):
        out = lid_base(lp, eos, S_f, *args)
        out = {**out, 'S_b': S_f} if same else out
        seen['S_b'] = float(out['S_b'])
        return out

    monkeypatch.setattr(jl, 'stefan_rate', spy)
    monkeypatch.setattr(jl, 'lid_base', base)
    S_f = float(y[s._n_stag - 1])
    jl.lid_top(
        s.jax_lid_params(),
        entropy_eos_jax(EOS_DIR),
        y[-2:],
        S_f,
        props,
        float(s.state._heat_flux[-2]),
        float(s.state._heat_flux[-1]),
        T_iso,
        top(s.state.heating),
    )
    return seen['rho'] * seen['dh'], seen['S_b'], S_f


PATHS = {'numpy': _numpy_jump, 'jax': _jax_jump}


@pytest.mark.physics_invariant
@pytest.mark.parametrize('path', list(PATHS))
@pytest.mark.parametrize('frac', [0.05, 0.3, 0.8])
def test_jump_is_zero_at_the_fluid_entropy(eos, monkeypatch, path, frac):
    """Fluid at the lid-base entropy joins the lid without releasing heat: the jump is 0."""
    s, y = _solver(eos, True, lambda s: s._lid_S_sol - frac * (s._lid_S_sol - float(eos.S_min)))
    jump, S_b, S_f = PATHS[path](s, y, monkeypatch, same=True)
    assert S_b == S_f
    assert jump == 0.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('path', list(PATHS))
@pytest.mark.parametrize('frac', [0.05, 0.3, 0.8])
def test_solid_regime_jump_is_the_integral_of_rho_T(eos, monkeypatch, path, frac):
    """With the solid rheology the lid base lies below the solidus, where the density changes
    with entropy: the jump is ``int_{S_b}^{S_f} rho T dS``, not ``rho_b`` times a specific
    heat content."""
    s, y = _solver(eos, True, lambda s: s._lid_S_sol - frac * (s._lid_S_sol - float(eos.S_min)))
    jump, S_b, S_f = PATHS[path](s, y, monkeypatch)
    assert S_b < s._lid_S_sol
    rho = [float(np.asarray(eos.density(s._lid_P, S)).item()) for S in (S_b, s._lid_S_sol)]
    assert abs(rho[0] / rho[1] - 1.0) > 1e-3
    want = _reference(s, S_b, S_f)
    assert jump == pytest.approx(want, rel=REL, abs=ABS_PER_MASS * float(rho[0]))


@pytest.mark.physics_invariant
@pytest.mark.parametrize('path', list(PATHS))
@pytest.mark.parametrize('frac', [0.05, 0.5, 1.0, 1.3])
def test_melt_regime_jump_is_the_integral_of_rho_T(eos, monkeypatch, path, frac):
    """Without the solid rheology the lid base is the solidus: the jump of a fluid in the melt
    band or above the liquidus is ``int_{S_sol}^{S_f} rho T dS``."""
    s, y = _solver(eos, False, lambda s: s._lid_S_sol + frac * (s._lid_S_liq - s._lid_S_sol))
    jump, S_b, S_f = PATHS[path](s, y, monkeypatch)
    assert S_b == s._lid_S_sol
    rho_b = float(np.asarray(eos.density(s._lid_P, S_b)).item())
    assert jump == pytest.approx(_reference(s, S_b, S_f), rel=REL, abs=ABS_PER_MASS * rho_b)


@pytest.mark.physics_invariant
def test_heat_table_is_zero_at_the_solidus_and_rises(eos):
    """W is 0 at the solidus entropy and rises with entropy (rho T > 0) over its knots."""
    s, _ = _solver(eos, False, 3000.0)
    knots = s._lid_W_of_S.x
    assert float(s._lid_W(s._lid_S_sol)) == 0.0
    assert np.all(np.diff(s._lid_W(knots)) > 0.0)
    assert s._lid_S_sol in knots and s._lid_S_liq in knots


def test_heat_table_continues_linearly_on_both_paths(eos):
    """Outside its range the heat table continues linearly with its end slope, the same on
    the NumPy and JAX paths, as the clamped EOS keeps rho T constant there."""
    pytest.importorskip('jax')
    from aragog.jax import surface_lid as jl

    s, _ = _solver(eos, False, 3000.0)
    lo, hi = float(eos.S_min), float(eos.S_max)
    pp = s._lid_W_of_S
    for edge, dS in ((lo, -50.0), (hi, 50.0)):
        linear = float(pp(edge)) + float(pp(edge, 1)) * dS
        got = float(s._lid_W(edge + dS))
        assert got == pytest.approx(linear, rel=1e-12)
        want = float(jl.enthalpy_H(s.jax_lid_params().hs, edge + dS))
        assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize(('dS', 'warns'), [(0.1, True), (5.0, False)])
def test_step_check_warns_on_a_jump_below_the_floor(eos, monkeypatch, caplog, dS, warns):
    """The step check warns when the jump per mass ``(W_f - W_b) / rho_b`` falls below
    ``LID_DH_MIN``: a lid base 0.1 J/kg/K below the fluid in the solid regime gives about
    150 J/kg, 5 J/kg/K gives several times the floor."""
    s, y = _solver(eos, True, lambda s: s._lid_S_sol - 0.3 * (s._lid_S_sol - float(eos.S_min)))
    base_state = s._lid_base_state
    monkeypatch.setattr(s, '_lid_base_state', lambda *a: {**base_state(*a), 'S_b': a[0] - dS})
    s._dSdt_single(0.0, y)
    ev = s._lid_last
    dh = (float(s._lid_W(ev.S_f)) - float(s._lid_W(ev.S_b))) / ev.rho
    assert (dh < sl.LID_DH_MIN) is warns
    with caplog.at_level(logging.WARNING, logger='fwl.aragog.solver.entropy_solver'):
        s._check_lid_step(0.0, 1, 0.0)
    assert any('below the floor' in r.getMessage() for r in caplog.records) is warns
