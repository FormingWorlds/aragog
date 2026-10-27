"""Validation of the phase-boundary blend at the default matprop_smooth_width on the SPIDER P-S
test tables: the evaluator properties are continuous across the solidus and the liquidus, equal
the mean of the two-phase and the single-phase branch on each boundary, and do not change far
from both boundaries; the JAX right-hand side matches the NumPy one inside the blend."""

from __future__ import annotations

import numpy as np
import pytest

from aragog.solver.entropy_solver import EntropySolver
from tests.conftest import EOS_DIR, entropy_eos_copy, needs_eos
from tests.test_entropy_solver_core_module_smoke import _build
from tests.test_phi_step_cap_armed_smoke import _build_mushy_parameters

pytestmark = [pytest.mark.smoke, needs_eos, pytest.mark.timeout(300)]

_PROPS = ('heat_capacity', 'density', 'thermal_expansivity', 'dTdPs', 'log10_viscosity')
_NODE = 6  # basic node of the plot, P = 65.7 GPa on the 12-node mesh


def _solver(width=None):
    """12-node table solver at the default width, or at ``width`` set past the validator (0 is
    the hard switch the default replaces)."""
    p = _build_mushy_parameters(solver_method='bdf')
    if width is not None:
        p.phase_mixed.matprop_smooth_width = width
    s = EntropySolver(p, entropy_eos=entropy_eos_copy(EOS_DIR))
    s.initialize()
    return s


def _props(s, entropy):
    """The blended properties of the basic-node evaluator at ``entropy`` [J/kg/K]."""
    ev = s.state.phase_basic
    ev.set_entropy(np.asarray(entropy, dtype=float))
    ev.update()
    out = {p: np.ravel(getattr(ev, p)()).copy() for p in _PROPS[:4]}
    out['log10_viscosity'] = np.log10(np.ravel(ev.viscosity()))
    return out


def _boundaries(s):
    P = s._P_basic_flat
    eos = s.entropy_eos
    return {
        'solidus': np.ravel(eos.solidus_entropy(P)),
        'liquidus': np.ravel(eos.liquidus_entropy(P)),
    }


@pytest.mark.physics_invariant
@pytest.mark.parametrize('boundary', ['solidus', 'liquidus'])
def test_the_default_width_keeps_every_property_continuous_across_the_boundary(boundary):
    """Between 1e-9 J/kg/K below and above the solidus or the liquidus, at all 12 node
    pressures (0 to 146 GPa), Cp, rho, alpha, dT/dP_S and log10 viscosity change by less than
    1e-6 relative; the hard switch at width 0 makes Cp jump by more than 7x at the solidus."""
    s = _solver()
    S_b = _boundaries(s)[boundary]
    below, above = _props(s, S_b - 1e-9), _props(s, S_b + 1e-9)
    for name in _PROPS:
        np.testing.assert_allclose(above[name], below[name], rtol=1e-6, err_msg=name)
    if boundary == 'solidus':  # discrimination: the width-0 switch jumps at every node
        hard = _solver(0.0)
        jump = (
            _props(hard, S_b + 1e-9)['heat_capacity']
            / _props(hard, S_b - 1e-9)['heat_capacity']
        )
        assert np.all(jump > 7.0)


def _branch_means(s_hard, s, boundary):
    """At S exactly on the boundary: the property of the default width, the two-phase branch
    (width 0 there gives weight 1) and the single-phase branch from the tables directly."""
    S_b = _boundaries(s)[boundary]
    blended, mixed = _props(s, S_b), _props(s_hard, S_b)
    eos, P = s.entropy_eos, s._P_basic_flat
    side = 'solid' if boundary == 'solidus' else 'melt'

    def table(name):  # clipped to the table as the evaluator does (the surface node is at P 0)
        tbl = eos._tables[f'{name}_{side}']
        P_c, S_c = (
            np.clip(P, tbl['P'][0], tbl['P'][-1]),
            np.clip(S_b, tbl['S'][0], tbl['S'][-1]),
        )
        return tbl['interp'](np.column_stack([P_c, S_c]))

    single = {
        p: table(t)
        for p, t in (
            ('heat_capacity', 'heat_capacity'),
            ('density', 'density'),
            ('dTdPs', 'dTdPs'),
        )
    }
    phase = s.parameters.phase_solid if side == 'solid' else s.parameters.phase_liquid
    single['log10_viscosity'] = np.full_like(P, np.log10(float(phase.viscosity)))
    return blended, mixed, single


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_on_each_boundary_every_property_is_the_mean_of_the_two_branches(validation_plot):
    """SPIDER get_smoothing (util.c) gives the two-phase branch the weight
    0.5 (1 + tanh(gphi / w)) below gphi 0.5 and 0.5 (1 - tanh((gphi - 1) / w)) above, so at
    gphi = 0 and gphi = 1 the weight is 0.5 for any width w > 0: Cp, rho, dT/dP_S and log10
    viscosity equal the mean of the two-phase value and the single-phase table value to 1e-12
    relative, at all 12 node pressures. alpha is left out: the clamp of negative values after
    the blend is not linear."""
    s, s_hard = _solver(), _solver(0.0)
    dS = np.linspace(-300.0, 300.0, 31)
    curves = {}
    for boundary in ('solidus', 'liquidus'):
        blended, mixed, single = _branch_means(s_hard, s, boundary)
        for name in ('heat_capacity', 'density', 'dTdPs', 'log10_viscosity'):
            mean = 0.5 * (mixed[name] + single[name])
            np.testing.assert_allclose(blended[name], mean, rtol=1e-12, err_msg=name)
        # discrimination: weight 1 (the width-0 switch) or 0 misses the mean of Cp by over 1 %
        gap = np.abs(mixed['heat_capacity'] - single['heat_capacity'])
        assert np.all(gap > 0.02 * np.abs(blended['heat_capacity']))

        S_b = _boundaries(s)[boundary][_NODE]
        for width, solver in (('0.01', s), ('0', s_hard)):
            rows = [_props(solver, np.full(s._P_basic_flat.shape, S_b + d)) for d in dS]
            for p in (
                'heat_capacity',
                'log10_viscosity',
            ):  # relative to the value on the boundary
                x = np.array([r[p][_NODE] for r in rows])
                ref = blended[p][_NODE]
                curves[boundary, width, p] = (
                    10 ** (x - ref) if p == 'log10_viscosity' else x / ref
                )

    def figure():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.6, 4.8))
        names = {'heat_capacity': '$c_p$', 'log10_viscosity': r'$\eta$'}
        for ax, boundary in zip(axes, ('solidus', 'liquidus')):
            for i, (p, name) in enumerate(names.items()):
                for width, style in (('0', ':'), ('0.01', '-')):
                    label = name if boundary == 'solidus' and width == '0.01' else None
                    ax.plot(dS, curves[boundary, width, p], style, color=f'C{i}', label=label)
            ax.set(ylabel=f'$X / X_\\mathrm{{{boundary[:3]}}}$')
        axes[1].set(xlabel=r'$S - S_\mathrm{b}$ [J/kg/K]')
        axes[0].legend(fontsize='small', ncol=2)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'phase_blend_at_solidus_and_liquidus')


def _node_gphi(s):
    """Normalised melt coordinate of every basic and staggered node after the last update."""
    eos, out = s.entropy_eos, []
    for ev in (s.state.phase_basic, s.state.phase_staggered):
        P, S = np.ravel(ev.pressure), np.ravel(ev.entropy)
        S_sol, S_liq = np.ravel(eos.solidus_entropy(P)), np.ravel(eos.liquidus_entropy(P))
        out.append((S - S_sol) / (S_liq - S_sol))
    return np.concatenate(out)


@pytest.mark.physics_invariant
def test_far_from_both_boundaries_the_rhs_does_not_change_with_the_width():
    """With every basic and staggered node at least 0.2 in gphi from the solidus and the
    liquidus (tanh(20) rounds to 1, so the blend weight is exactly 0 or 1), the right-hand side
    at the default width equals the one at width 0 bit for bit; with the cells at gphi 0.02 it
    does not. The surface node sits at about 0.1 + 1.9 gphi of the cells, so 0.35 is the
    mushy state that keeps it inside the band."""
    s, s_hard = _solver(), _solver(0.0)
    eos, P = s.entropy_eos, s._P_stag_flat
    S_sol, S_liq = np.ravel(eos.solidus_entropy(P)), np.ravel(eos.liquidus_entropy(P))
    for g in (-0.25, 0.35, 1.3, 0.02):
        y = S_sol + g * (S_liq - S_sol)
        f, f_hard = np.ravel(s.dSdt(0.0, y)), np.ravel(s_hard.dSdt(0.0, y))
        gphi = _node_gphi(s)
        far = np.min(np.minimum(np.abs(gphi), np.abs(gphi - 1.0))) >= 0.2
        assert far == (g != 0.02), (g, gphi)
        assert np.all(np.isfinite(f))
        assert np.array_equal(f, f_hard) == far, g


@pytest.mark.physics_invariant
def test_the_jax_rhs_matches_numpy_inside_the_blend_at_the_default_width():
    """Cells on both tanh skirts and on the boundaries: the JAX right-hand side, built from the
    config width, matches the NumPy one to 1e-10 relative; JAX at width 0 misses it by more
    than 1e-6, so the width reaches the JAX path."""
    jnp = pytest.importorskip('jax.numpy')
    eqx = pytest.importorskip('equinox')
    from aragog.jax.solver import dSdt
    from tests.test_jax_dsdt_core_module import _build_jax_pieces

    s = _build('quasi_steady', entropy_eos_copy(EOS_DIR, identity_alpha=True))
    assert s.parameters.phase_mixed.matprop_smooth_width == pytest.approx(0.01, rel=1e-15)
    eos, P = s.entropy_eos, s._P_stag_flat
    S_sol, S_liq = np.ravel(eos.solidus_entropy(P)), np.ravel(eos.liquidus_entropy(P))
    g = np.array([-0.02, -0.005, 0.0, 0.004, 0.012, 0.5, 0.99, 1.0, 1.015])
    s.set_initial_entropy(S_sol + g * (S_liq - S_sol))
    y = np.asarray(s._S0, dtype=float)
    f_np = np.ravel(s.dSdt(0.0, y))
    s._core_module_budget, s._core_module_q_radio, s._core_module_ra_crit_cmb = None, 0, 0
    args = tuple(_build_jax_pieces(s))[:6]
    f_jax = np.ravel(dSdt(0.0, jnp.asarray(y), args))
    n = s._n_stag
    np.testing.assert_allclose(
        f_jax[:n], f_np[:n], rtol=1e-10, atol=1e-10 * np.abs(f_np[:n]).max()
    )
    sharp = eqx.tree_at(lambda q: q.matprop_smooth_width, args[1], 0.0)
    f_sharp = np.ravel(dSdt(0.0, jnp.asarray(y), (args[0], sharp, *args[2:])))
    assert np.max(np.abs(f_sharp[:n] / f_np[:n] - 1.0)) > 1e-6
