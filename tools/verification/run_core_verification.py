"""Regenerate the figures and numbers of the core module verification page.

Writes ``docs/figures/vv/fig_08..fig_19_*.png`` (with a PDF beside each) and
``docs/figures/vv/core_verification_values.json``, the source of every number
quoted on ``docs/Explanations/core_verification.md``. The solver-level items
need the SPIDER-format EOS tables (``ARAGOG_TEST_EOS_DIR``); the Leeds
``thermal_history`` comparisons read the reference tables in
``tools/verification/data/``.

Usage::

    pip install -e '.[jax,verification]'
    python tools/verification/run_core_verification.py [--only 1,2,...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import cumulative_trapezoid, quad, solve_ivp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from thermal_history_reference import INPUTS as NIMMO  # noqa: E402

from aragog.core import (  # noqa: E402
    CoreEnergyBudget,
    CoreEntropyBudget,
    GaussianCoreProfiles,
    IronMeltingCurve,
    QuadraticMeltingCurve,
    cmb_boundary_layer_flux,
)
from aragog.core import melting as m  # noqa: E402
from aragog.core.entropy import _CHR09_F_GEOMETRY  # noqa: E402

jax.config.update('jax_enable_x64', True)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'figures' / 'vv'
VALUES_FILE = OUT / 'core_verification_values.json'
WIDTH = 6.4  # inches, one width for every figure on the page
G = 6.674_30e-11
YEAR = 365.25 * 86400.0
GYR = 1e9 * YEAR
MYR_LEEDS = 1e6 * 365 * 86400.0  # thermal_history counts years of 365 days

# Earth-like core of the test suite (tests/test_core_profiles.py, tests/test_core_budget.py).
EARTH = dict(
    rho_cen=12500.0, length_scale=7200e3, r_cmb=3480e3, p_cmb=136e9, alpha=1.35e-5, c_p=840.0
)

VALUES: dict[str, dict] = {}


def record(item: int, key: str, value: float) -> float:
    """Store one page number under its item; returns the value unchanged."""
    VALUES.setdefault(str(item), {})[key] = float(value)
    return value


def style():
    """PROTEUS light theme, or the matplotlib default when proteus-mpl is absent."""
    try:
        import proteus_mpl

        proteus_mpl.use('light')
        plt.rcParams['figure.constrained_layout.use'] = True
        return proteus_mpl.DOMAINS['interior'], proteus_mpl.COLORS
    except ImportError:  # the figures stay reproducible without the theme
        plt.rcParams['figure.constrained_layout.use'] = True
        return 'C3', {}


def colour(name: str, fallback: str) -> str:
    return COLORS.get(name, fallback)


def save(fig, name: str) -> None:
    fig.align_ylabels()
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)


# ---------------------------------------------------------------- 1. structure
def item1_structure() -> None:
    """Gaussian profiles against independent quadrature and hydrostatic balance."""
    prof = GaussianCoreProfiles(**EARTH)
    lab = GaussianCoreProfiles(**EARTH, pressure_mode='labrosse')
    r = np.linspace(1e3, prof.r_cmb, 200)
    rho = np.asarray(prof.density(r))
    g = np.asarray(prof.gravity(r))
    p_quad, p_lab = np.asarray(prof.pressure(r)), np.asarray(lab.pressure(r))

    def m_quad(x):
        return quad(
            lambda s: float(prof.density(s)) * 4 * np.pi * s**2, 0.0, x, epsabs=0, epsrel=1e-13
        )[0]

    radii = (0.3e6, 1.5e6, prof.r_cmb)
    mass_err = max(abs(m_quad(x) / float(prof.enclosed_mass(x)) - 1) for x in radii)
    g_err = max(abs(G * m_quad(x) / x**2 / float(prof.gravity(x)) - 1) for x in radii)
    h = 50.0
    rr = r[(r > 2 * h) & (r < prof.r_cmb - 2 * h)]
    dpdr = (np.asarray(prof.pressure(rr + h)) - np.asarray(prof.pressure(rr - h))) / (2 * h)
    hydro = np.abs(dpdr / -(np.asarray(prof.density(rr)) * np.asarray(prof.gravity(rr))) - 1)
    record(1, 'mass_rel_err', mass_err)
    record(1, 'gravity_rel_err', g_err)
    record(1, 'hydrostatic_max_rel_err', hydro.max())
    record(1, 'pressure_lab_vs_quad_max_rel', np.max(np.abs(p_lab / p_quad - 1)))
    record(1, 'p_cen_GPa', float(prof.pressure(0.0)) / 1e9)
    record(1, 'rho_cmb', rho[-1])
    record(1, 'm_core', float(prof.enclosed_mass(prof.r_cmb)))
    slope = 4 * np.pi * G * EARTH['rho_cen'] / 3
    record(1, 'gravity_centre_slope_rel_err', abs(float(prof.gravity(1.0)) / slope - 1))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 5.2), sharex=True, height_ratios=(2, 1))
    x = r / 1e3
    ax.plot(x, rho / 1e3, color=CORE, label=r'$\rho$ (g cm$^{-3}$)')
    ax.plot(x, g, color=colour('ocean', 'C0'), label=r'$g$ (m s$^{-2}$)')
    ax.plot(x, p_quad / 1e10, color=colour('ink', 'k'), label=r'$P$ (10 GPa)')
    ax.legend(frameon=False, fontsize='small', loc='center left')
    ax.set_ylabel('profile value')
    ax.set_xlim(0, x[-1])
    ax2.semilogy(
        rr / 1e3, np.maximum(hydro, 1e-16), color=CORE, label=r'$|dP/dr + \rho g|/\rho g$'
    )
    ax2.semilogy(
        x,
        np.maximum(np.abs(p_lab / p_quad - 1), 1e-16),
        color=colour('fog', 'C7'),
        label='closed form vs quadrature $P$',
    )
    ax2.set_xlabel('radius (km)')
    ax2.set_ylabel('relative residual')
    ax2.legend(frameon=False, fontsize='small')
    save(fig, 'fig_08_core_structure')


# ------------------------------------------------------------ 2. energy identities
def _model_budget(curve, **kw):
    return CoreEnergyBudget(
        GaussianCoreProfiles(**EARTH), curve, ds_fusion=170.0, icn_width=10.0, **kw
    )


def item2_energy() -> None:
    """Capacity split across onset and freeze-out; content difference = capacity integral."""
    budget = _model_budget(
        IronMeltingCurve(light_element_fraction=0.1, depression=1.2), alpha_c=0.6, c_light=0.05
    )
    t_on, t_fr = float(budget.t_onset), float(budget.t_freeze)
    t = np.linspace(t_fr - 150.0, t_on + 150.0, 40001)
    t = np.unique(np.concatenate([t, [t_on, t_fr, t_fr - 1e-6, t_fr + 1e-6]]))
    sec = float(budget.secular_capacity())
    lat = np.asarray(jax.jit(jax.vmap(budget.latent_capacity))(t))
    grav = np.asarray(jax.jit(jax.vmap(budget.gravitational_capacity))(t))
    c_eff = np.asarray(jax.jit(jax.vmap(budget.effective_capacity))(t))
    integral = cumulative_trapezoid(c_eff, t, initial=0.0)
    # The content difference at 121 checkpoints against the dense trapezoid of C_eff.
    tc = np.linspace(t[0], t[-1], 121)[1:]
    content = np.array([float(budget.heat_content(x)) for x in np.concatenate([[t[0]], tc])])
    resid = np.abs((content[1:] - content[0]) - np.interp(tc, t, integral)) / np.abs(
        integral[-1]
    )
    record(2, 't_onset', t_on)
    record(2, 't_freeze', t_fr)
    record(2, 'content_vs_integral_max_rel', resid.max())
    # The freeze-out jump without the gravitational term, as in test_freeze_out_capacity_jump.
    for name, curve in (
        ('iron', IronMeltingCurve(light_element_fraction=0.1, depression=1.2)),
        ('quadratic', QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)),
    ):
        b = _model_budget(curve)
        tf = float(b.t_freeze)
        above, below = (
            float(b.effective_capacity(tf + 1e-4)),
            float(b.effective_capacity(tf - 1e-4)),
        )
        record(2, f'{name}_freeze_out_ratio', above / below)
        record(2, f'{name}_freeze_out_drop', (below - above) / above)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 5.2), sharex=True, height_ratios=(2, 1))
    ax.plot(t, np.full_like(t, sec) / 1e27, color=colour('solar', 'C1'), label='secular')
    ax.plot(t, lat / 1e27, color=CORE, label='latent')
    ax.plot(t, grav / 1e27, color=colour('ocean', 'C0'), label='gravitational')
    ax.plot(t, c_eff / 1e27, color=colour('ink', 'k'), lw=1.6, label=r'$\tilde C$ total')
    for tt, lab in ((t_on, 'onset'), (t_fr, 'freeze-out')):
        ax.axvline(tt, color=colour('fog', '0.7'), lw=0.8)
        ax.annotate(
            lab,
            (tt, 0.0),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize='small',
            color=colour('fog', 'C7'),
        )
    ax.set_ylabel(r'capacity ($10^{27}$ J K$^{-1}$)')
    ax.legend(frameon=False, fontsize='small', ncols=2)
    ax2.semilogy(tc, np.maximum(resid, 1e-17), 'o', ms=2.5, color=CORE)
    ax2.set_xlabel(r'$T_\mathrm{cmb}$ (K)')
    ax2.set_ylabel('rel. difference')
    save(fig, 'fig_09_core_energy_identities')


# ------------------------------------------------------- 3. boundary-layer flux
BL_PROPS = dict(
    conductivity=4.0,
    density=4000.0,
    heat_capacity=1000.0,
    expansivity=3e-5,
    gravity=10.0,
    dr_half=1e4,
)


def item3_flux() -> None:
    """q(dT) for a liquid and a solid base, both signs, two Ra_crit values."""
    dT = np.geomspace(1e-3, 1e3, 241)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    styles = {}
    for eta, name, col in (
        (0.1, 'liquid base', CORE),
        (1e21, 'solid base', colour('fog', 'C7')),
    ):
        for ra, ls in ((450.0, '-'), (1800.0, '--')):
            q = np.asarray(
                cmb_boundary_layer_flux(
                    4000.0 + dT, 4000.0, viscosity=eta, ra_crit=ra, **BL_PROPS
                )
            )
            ax.loglog(dT, q, ls, color=col)
            styles[(name, ra)] = q
        ax.annotate(
            name,
            (dT[-1], styles[(name, 450.0)][-1]),
            xytext=(4, 0),
            textcoords='offset points',
            va='center',
            color=col,
            fontsize='small',
        )
    # A colder core conducts across the half cell over either base: one line for both.
    qn = -np.asarray(cmb_boundary_layer_flux(4000.0 - dT, 4000.0, viscosity=0.1, **BL_PROPS))
    ax.loglog(dT, qn, ':', lw=2.4, color=colour('ocean', 'C0'))
    ax.annotate(
        'core colder (either base)',
        (dT[60], qn[60]),
        xytext=(0, 10),
        textcoords='offset points',
        rotation=18,
        color=colour('ocean', 'C0'),
        fontsize='small',
    )
    record(
        3,
        'cold_core_flux_equals_solid_base',
        float(np.max(np.abs(qn / styles[('solid base', 450.0)] - 1))),
    )
    q_liq = styles[('liquid base', 450.0)]
    hi = dT > 10
    slope = np.polyfit(np.log(dT[hi]), np.log(q_liq[hi]), 1)[0]
    record(3, 'convective_slope', slope)
    record(
        3,
        'ra_crit_ratio_1800_450',
        float(np.median(styles[('liquid base', 1800.0)][hi] / q_liq[hi])),
    )
    record(
        3,
        'q_liquid_100K',
        float(cmb_boundary_layer_flux(4100.0, 4000.0, viscosity=0.1, **BL_PROPS)),
    )
    record(
        3,
        'q_solid_100K',
        float(cmb_boundary_layer_flux(4100.0, 4000.0, viscosity=1e21, **BL_PROPS)),
    )
    record(3, 'conduction_slope', BL_PROPS['conductivity'] / BL_PROPS['dr_half'])
    ax.set_xlabel(r'$|T_\mathrm{core} - T_m|$ (K)')
    ax.set_ylabel(r'$|q_\mathrm{cmb}|$ (W m$^{-2}$)')
    ax.set_xlim(dT[0], dT[-1] * 40)
    ax.text(
        0.98,
        0.04,
        r'solid: $Ra_\mathrm{crit}=450$; dashed: 1800',
        transform=ax.transAxes,
        ha='right',
        fontsize='small',
    )
    save(fig, 'fig_10_cmb_boundary_layer_flux')


# ---------------------------------------------------------- 4. nucleation
def item4_nucleation() -> None:
    """r_icb ~ (T_on - T)^(1/2) and the implicit-function sensitivity."""
    budget = _model_budget(
        IronMeltingCurve(light_element_fraction=0.1, depression=1.2), alpha_c=0.6, c_light=0.05
    )
    t_on = float(budget.t_onset)
    under = np.geomspace(1e-3, 30.0, 61)
    r_icb = np.asarray(jax.vmap(budget.r_icb)(t_on - under))
    small = under < 1.0
    slope = np.polyfit(np.log(under[small]), np.log(r_icb[small]), 1)[0]
    drdt = np.asarray(jax.vmap(jax.grad(budget.r_icb))(t_on - under))
    h = 1e-3 * under  # a step relative to the undercooling keeps the cusp out of the difference
    r_of = jax.vmap(budget.r_icb)
    fd = (np.asarray(r_of(t_on - under + h)) - np.asarray(r_of(t_on - under - h))) / (2 * h)
    rel = np.abs(drdt / fd - 1)
    record(4, 't_onset', t_on)
    record(4, 'sqrt_slope', slope)
    record(4, 'jvp_vs_fd_max_rel', rel.max())

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 5.2), sharex=True, height_ratios=(2, 1))
    ax.loglog(under, r_icb / 1e3, color=CORE, label=r'$r_\mathrm{icb}$')
    ref = r_icb[0] * (under / under[0]) ** 0.5
    ax.loglog(
        under,
        ref / 1e3,
        '--',
        color=colour('fog', 'C7'),
        label=r'$\propto (T_\mathrm{on}-T)^{1/2}$',
    )
    ax.set_ylabel(r'$r_\mathrm{icb}$ (km)')
    ax.legend(frameon=False, fontsize='small')
    ax2.loglog(under, np.maximum(rel, 1e-16), color=CORE)
    ax2.set_xlabel(r'$T_\mathrm{on} - T_\mathrm{cmb}$ (K)')
    ax2.set_ylabel('JVP vs central diff.')
    save(fig, 'fig_11_inner_core_nucleation')


# ------------------------------------------------------------- 7. dynamo scaling
def _core_budget(inp, pressure_mode, **kw):
    """CoreEnergyBudget of the quadratic-melting-curve core that ``inp`` describes."""
    keys = ('rho_cen', 'length_scale', 'r_cmb', 'p_cmb', 'alpha', 'c_p')
    prof = GaussianCoreProfiles(**{k: inp[k] for k in keys}, pressure_mode=pressure_mode)
    curve = QuadraticMeltingCurve(t_m0=inp['t_m0'], t_m1=inp['t_m1'], t_m2=inp['t_m2'])
    return CoreEnergyBudget(
        prof,
        curve,
        ds_fusion=170.0,
        icn_width=10.0,
        latent_heat=inp['latent_heat'],
        alpha_c=inp['alpha_c'],
        c_light=inp['c_light'],
        **kw,
    )


def item7_dynamo() -> None:
    """Entropy margin and Christensen et al. (2009) field strength against CMB heat flow."""
    ent, t_c = _nimmo_budget()[1], 4180.0
    printed = 1.35e-5 * 10.7 * 3.48e6 / 840.0
    for geometry, factor in _CHR09_F_GEOMETRY.items():
        record(7, f'F_{geometry}_printed_inputs', factor * printed)
    record(7, 'F_const_flux_profile', float(ent.chr09_efficiency_factor()))
    qk = record(7, 'Q_k_TW', float(ent.adiabatic_heat_flow(t_c)) / 1e12)
    threshold = record(
        7,
        'dynamo_threshold_TW',
        brentq(lambda q: float(ent.entropy_margin(t_c, q)), 1e12, 40e12) / 1e12,
    )
    q = np.linspace(1e12, 30e12, 300)
    margin = np.asarray(jax.jit(jax.vmap(lambda x: ent.entropy_margin(t_c, x)))(q))
    b_rms = np.asarray(jax.jit(jax.vmap(lambda x: ent.b_rms_core(t_c, x)))(q))
    record(7, 'b_rms_17TW_mT', float(ent.b_rms_core(t_c, 17e12)) * 1e3)
    earth_b = record(
        7, 'earth_internal_field_mT', 7 * 0.26
    )  # B/B_dip about 7 times 0.26 mT, CHR09 p. 168

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 5.2), sharex=True)
    ax.plot(q / 1e12, margin / 1e6, color=CORE)
    ax.axhline(0.0, color=colour('fog', '0.6'), lw=0.8)
    ax.axvline(threshold, color=colour('fog', '0.6'), lw=0.8, ls='--')
    ax.annotate(
        f'dynamo threshold {threshold:.2f} TW',
        (threshold, 0.0),
        xytext=(6, 8),
        textcoords='offset points',
        fontsize='small',
    )
    ax.set_ylabel(r'$\Delta E$ (MW K$^{-1}$)')
    ax2.plot(q / 1e12, b_rms * 1e3, color=CORE, label=r'$B_\mathrm{rms}$, Eq. 2, $c=0.63$')
    ax2.axhline(earth_b, color=colour('ocean', 'C0'), ls='--', lw=1)
    ax2.annotate(
        r'Earth: $7 \times 0.26$ mT',
        (q[0] / 1e12, earth_b),
        xytext=(2, 4),
        textcoords='offset points',
        fontsize='small',
        color=colour('ocean', 'C0'),
    )
    ax2.axvline(qk, color=colour('fog', '0.6'), lw=0.8, ls=':')
    ax2.annotate(
        r'$Q_k$', (qk, 0.0), xytext=(3, 3), textcoords='offset points', fontsize='small'
    )
    ax2.set_xlabel(r'CMB heat flow $Q_\mathrm{cmb}$ (TW)')
    ax2.set_ylabel(r'core field (mT)')
    save(fig, 'fig_14_dynamo_scaling')


# ----------------------------------------------------------- 8. melting curve
def _simon_branches(p):
    """The two Anzellini et al. (2013) Simon branches, switching at the triple point [K]."""
    gpa = p / 1e9
    low = m._T0 * ((gpa - m._P0 / 1e9) / m._DP_LOW + 1.0) ** m._EXP_LOW
    high = (
        m._TT * ((np.maximum(gpa, m._PT / 1e9) - m._PT / 1e9) / m._DP_HIGH + 1.0) ** m._EXP_HIGH
    )
    return np.where(p < m._PT, low, high)


def item8_melting() -> None:
    """PALEOS iron curve against the two Anzellini et al. (2013) Simon branches."""
    p = np.linspace(5.2e9, 360e9, 2000)
    gpa = p / 1e9
    pure = np.asarray(IronMeltingCurve.t_melt_pure(p))
    piecewise = _simon_branches(p)
    record(
        8,
        'branch_jump_K',
        m._T0 * ((m._PT - m._P0) / 1e9 / m._DP_LOW + 1.0) ** m._EXP_LOW - m._TT,
    )
    record(8, 'blend_max_dev_K', float(np.max(np.abs(pure - piecewise))))
    record(
        8,
        'outside_blend_max_rel',
        float(np.max(np.abs(pure / piecewise - 1)[np.abs(gpa - 98.5) > 3.0])),
    )
    record(8, 't_melt_136GPa', float(IronMeltingCurve.t_melt_pure(136e9)))
    record(8, 't_melt_330GPa', float(IronMeltingCurve.t_melt_pure(330e9)))
    alloy = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    record(
        8, 'depression_factor', float(alloy.t_melt(136e9) / IronMeltingCurve.t_melt_pure(136e9))
    )

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 5.4), height_ratios=(2, 1))
    ax.plot(gpa, pure, color=colour('ink', 'k'), label='pure Fe (PALEOS)')
    ax.plot(
        gpa, np.asarray(alloy.t_melt(p)), color=CORE, label=r'alloy, $x=0.1$, depression 1.2'
    )
    ax.axvline(98.5, color=colour('fog', '0.6'), lw=0.8)
    ax.annotate(
        r'$\gamma$-$\epsilon$-liquid triple point',
        (98.5, 6000),
        xytext=(4, 0),
        textcoords='offset points',
        fontsize='small',
        color=colour('fog', '0.5'),
    )
    ax.set_ylabel(r'$T_m$ (K)')
    ax.set_xlabel('pressure (GPa)')
    ax.legend(frameon=False, fontsize='small', loc='lower right')
    pz = np.linspace(94e9, 103e9, 901)
    ax2.plot(
        pz / 1e9, np.asarray(IronMeltingCurve.t_melt_pure(pz)) - _simon_branches(pz), color=CORE
    )
    ax2.axhline(0.0, color=colour('fog', '0.6'), lw=0.8)
    ax2.set_xlabel('pressure (GPa), around the branch switch')
    ax2.set_ylabel(r'blend $-$ Eq. 2/3 (K)')
    save(fig, 'fig_15_iron_melting_curve')


# ----------------------------------------------------- 9. CVODE across the onset
def _solver_helpers():
    """The solver set-up of tests/test_entropy_solver_core_module_smoke.py."""
    import sys

    sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
    from test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build

    from tests.conftest import entropy_eos_copy

    return CORE_MODULE_PARAMS, _build, entropy_eos_copy


def item9_cvode_onset() -> None:
    """Core heat across the inner-core onset: solver ledger, heat content and CMB heat."""
    params0, build, eos_copy = _solver_helpers()
    params = {
        k: v for k, v in params0.items() if k not in ('light_element_fraction', 'depression')
    }
    params.update(melting_curve='quadratic', t_m0=4015.5, t_m1=2.95e-12, t_m2=8.37e-25)
    eos = eos_copy()
    solver = build('core_module', eos, params, end_time=4.0, solver_method='cvode')
    budget, n = solver._core_module_budget, solver._n_stag
    t_on = float(budget.t_onset)
    solver.set_initial_core_temperature(t_on + 2.0)
    solver.set_initial_entropy(np.linspace(7000.0, 6700.0, n))
    solver.solve()
    out = solver.get_state()
    sol = solver._solution
    t_yr, t_core = np.asarray(sol.t), np.asarray(sol.y[n + 1])
    content = np.array([float(budget.heat_content(x)) for x in t_core]) - float(
        budget.heat_content(t_core[0])
    )
    flux = []
    for i in range(t_yr.size):
        solver.dSdt(t_yr[i], sol.y[:, i])
        flux.append(
            solver._core_module_cmb_flux(t_core[i], float(sol.y[0, i])) * solver._cmb_area
        )
    cmb_heat = cumulative_trapezoid(flux, t_yr, initial=0.0) * YEAR
    record(9, 't_onset', t_on)
    record(9, 't_core_start', t_core[0])
    record(9, 't_core_end', t_core[-1])
    record(9, 'r_icb_end_km', float(budget.r_icb(t_core[-1])) / 1e3)
    record(9, 'core_vs_content_rel', abs(out.step_dE_core_J / content[-1] - 1))
    record(9, 'core_vs_cmb_rel', abs(out.step_dE_core_J / -out.step_dE_F_cmb_J - 1))
    record(9, 'step_dE_core_J', out.step_dE_core_J)
    record(9, 'n_outputs', t_yr.size)

    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1, figsize=(WIDTH, 6.4), sharex=True, height_ratios=(1.2, 1.2, 1)
    )
    ax.plot(t_yr, t_core, color=CORE)
    ax.axhline(t_on, color=colour('fog', '0.6'), lw=0.8, ls='--')
    ax.annotate(
        'inner-core onset',
        (t_yr[-1], t_on),
        xytext=(-4, 4),
        textcoords='offset points',
        ha='right',
        fontsize='small',
        color=colour('fog', '0.5'),
    )
    ax.set_ylabel(r'$T_\mathrm{core}$ (K)')
    ax2.plot(t_yr, -content / 1e27, color=CORE, label='heat-content difference')
    ax2.plot(
        t_yr,
        cmb_heat / 1e27,
        '--',
        color=colour('ink', 'k'),
        label=r'$\int Q_\mathrm{cmb}\,dt$',
    )
    ax2.set_ylabel(r'heat lost ($10^{27}$ J)')
    ax2.legend(frameon=False, fontsize='small')
    rel = np.abs(-content[1:] / cmb_heat[1:] - 1)
    record(9, 'reconstructed_cmb_max_rel', rel.max())
    ax3.semilogy(t_yr[1:], rel, color=CORE)
    ax3.set_xlabel('time (yr)')
    ax3.set_ylabel('rel. difference')
    save(fig, 'fig_16_cvode_onset_ledger')


# ------------------------------------------------- 12. NumPy vs JAX and Jacobian
def item12_jax_parity() -> None:
    """NumPy and JAX right-hand sides on three core states; the analytic core column against
    central differences over a range of steps."""
    core_params, build, eos_copy = _solver_helpers()
    from test_jax_dsdt_core_module import _build_jax_pieces

    from aragog.jax.solver import dSdt_core_module

    steps = np.geomspace(1e-4, 30.0, 40)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 5.6))
    cols = {
        'nucleating': CORE,
        'above onset': colour('ink', 'k'),
        'stratified': colour('ocean', 'C0'),
    }
    for state, col in cols.items():
        params = dict(core_params)
        if state == 'stratified':
            params |= {'stratification': True, 'k_core': 130.0}
        solver = build('core_module', eos_copy(), params, s_init='driven')
        budget, n = solver._core_module_budget, solver._n_stag
        y = np.asarray(solver._S0, dtype=float)
        if state == 'nucleating':
            scan = np.linspace(3200.0, 6000.0, 281)
            latent = np.asarray(jax.vmap(budget.latent_capacity)(scan))
            y[n + 1] = np.median(scan[latent > 0.01 * float(budget.secular_capacity())])
        elif state == 'above onset':
            y[n + 1] = float(budget.t_onset) + 100.0
        else:
            y[n + 1] += 50.0
        args = _build_jax_pieces(solver)
        f_np = np.asarray(solver.dSdt(0.0, y)).ravel()
        f_jax = np.asarray(dSdt_core_module(0.0, jnp.asarray(y), args)).ravel()
        rel = np.abs(f_jax - f_np) / np.maximum(np.abs(f_np), 1e-300)
        key = state.replace(' ', '_')
        record(12, f'rhs_max_rel_{key}', rel.max())
        ax.semilogy(
            np.arange(rel.size), np.maximum(rel, 1e-17), 'o-', ms=3, color=col, label=state
        )
        jac = np.asarray(jax.jacrev(lambda v: dSdt_core_module(0.0, v, args))(jnp.asarray(y)))[
            :, n + 1
        ]

        def rhs(v):
            return np.asarray(dSdt_core_module(0.0, jnp.asarray(v), args))

        errs = []
        for h in steps:
            up, down = y.copy(), y.copy()
            up[n + 1] += h
            down[n + 1] -= h
            fd = (rhs(up) - rhs(down)) / (2.0 * h)
            errs.append(abs(fd[n + 1] / jac[n + 1] - 1))
        errs = np.array(errs)
        record(12, f'jac_tcore_min_rel_{key}', errs.min())
        record(12, f'jac_tcore_best_step_{key}', steps[np.argmin(errs)])
        ax2.loglog(steps, np.maximum(errs, 1e-17), color=col, label=state)
    ax.set_xlabel(r'state index (the last two: $dS/dr$ at the CMB, $T_\mathrm{core}$)')
    ax.set_ylabel('|JAX / NumPy - 1|')
    ax2.legend(
        *ax.get_legend_handles_labels(), frameon=False, fontsize='small', loc='lower left'
    )
    ax2.set_xlabel(r'central-difference step in $T_\mathrm{core}$ (K)')
    ax2.set_ylabel(r'$\partial \dot T_\mathrm{core} / \partial T_\mathrm{core}$ rel. error')
    save(fig, 'fig_19_numpy_jax_parity')


# ------------------------------------------- 6 and 10. Leeds thermal_history
TH_TABLE = ROOT / 'tools' / 'verification' / 'data' / 'thermal_history_evolution.csv'
LAYER_TABLE = TH_TABLE.with_name('thermal_history_stable_layer.csv')


def _thermal_history():
    """The Leeds table, its inputs, and aragog's budgets on the same inputs."""
    with TH_TABLE.open() as fh:
        header = json.loads(fh.readline()[2:])
    data = np.loadtxt(TH_TABLE, delimiter=',', comments='#')
    inp = header['inputs']
    budget = _core_budget(inp, 'quadrature')
    cols = {name: data[:, i] for i, name in enumerate(header['columns'])}
    return header, cols, budget, CoreEntropyBudget(budget, k_core=inp['k_core'])


def item6_leeds_terms() -> None:
    """Every budget term against the Leeds model on the states of its own thermal history."""
    header, th, budget, ent = _thermal_history()
    t_cmb, ratio = th['T_cmb'], th['T_cen'] / th['T_cmb']  # Leeds terms are per dT_cen
    grown = th['r_icb'] > 0
    ours = {
        'secular': np.full_like(t_cmb, float(budget.secular_capacity())),
        'latent': np.asarray(jax.vmap(budget.latent_capacity)(t_cmb)),
        'gravitational': np.asarray(jax.vmap(budget.gravitational_capacity)(t_cmb)),
        'secular entropy': np.asarray(jax.vmap(ent.secular_entropy_capacity)(t_cmb)),
        'latent entropy': np.asarray(jax.vmap(ent.latent_entropy_capacity)(t_cmb)),
        'gravitational entropy': np.asarray(
            jax.vmap(ent.gravitational_entropy_capacity)(t_cmb)
        ),
    }
    theirs = {
        'secular': -th['Qs_per_dTcen'] * ratio,
        'latent': -th['Ql_per_dTcen'] * ratio,
        'gravitational': -th['Qg_per_dTcen'] * ratio,
        'secular entropy': -th['Es_per_dTcen'] * ratio,
        'latent entropy': -th['El_per_dTcen'] * ratio,
        'gravitational entropy': -th['Eg_per_dTcen'] * ratio,
    }
    enrich = (
        th['conc_l'] / th['conc_l'][0]
    )  # Leeds enriches the outer core as the inner core grows
    record(6, 'conduction_sink_rel', abs(float(ent.conduction_sink()) / th['Ek'][0] - 1))
    record(6, 'enrichment_end', enrich[-1])
    time = th['time_myr']
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(WIDTH, 8.4))
    styles = {
        'secular': ('-', colour('solar', 'C1')),
        'latent': ('-', CORE),
        'gravitational': ('-', colour('ocean', 'C0')),
    }
    for name in ours:
        mask = grown if 'secular' not in name else np.ones_like(grown)
        rel = np.abs(ours[name][mask] / theirs[name][mask] - 1)
        key = name.replace(' ', '_')
        record(6, f'{key}_max_rel', rel.max())
        if name.endswith('entropy'):
            continue
        ls, col = styles[name]
        ax.semilogy(time[mask], np.maximum(rel, 1e-16), ls, color=col, label=name)
        if name == 'gravitational':
            corrected = np.abs(ours[name][mask] * enrich[mask] / theirs[name][mask] - 1)
            record(6, 'gravitational_enrichment_corrected_max_rel', corrected.max())
            ax.semilogy(
                time[mask],
                corrected,
                ':',
                color=col,
                label='gravitational, Leeds enrichment applied',
            )
    ax.set_xlabel('time (Myr)')
    ax.set_ylabel('|aragog / Leeds - 1|')
    ax.set_xlim(time[0], time[-1])
    ax.legend(frameon=False, fontsize='small')
    _stable_layer_panel(ax2, ax3)
    save(fig, 'fig_13_leeds_budget_terms')


def _stable_layer_panel(ax, ax_t) -> None:
    """Layer thickness and central temperature: thermal_history against aragog's own
    core-only history under the same fixed CMB flows, its layer forming at time zero."""
    with LAYER_TABLE.open() as fh:
        header = json.loads(fh.readline()[2:])
    data = np.loadtxt(LAYER_TABLE, delimiter=',', comments='#')
    inp = header['inputs']
    budget = _core_budget(inp, 'quadrature', stratification=True, k_core=inp['k_core'])
    r_cmb = inp['r_cmb']
    for q, col in zip(header['q_cmb'], (CORE, colour('ocean', 'C0'))):
        rows = data[data[:, 0] == q]
        t, r_s, t_cen_leeds = rows[:, 1], rows[:, 5], rows[:, 3]
        rate = jax.jit(
            lambda time, temp, q=q: budget.dtcmb_dt(temp, q, t_layer=time * MYR_LEEDS)
        )
        sol = solve_ivp(
            lambda time, y, rate=rate: [float(rate(time, y[0])) * MYR_LEEDS],
            (t[0], t[-1]),
            [inp['t_cmb_start']],
            t_eval=t,
            method='BDF',
            rtol=1e-9,
            atol=1e-6,
        )
        t_cmb = sol.y[0]
        capped = np.asarray(
            jax.vmap(lambda x, a, q=q: budget.convecting_radius(x, q, a))(t_cmb, t * MYR_LEEDS)
        )
        quasi = np.asarray(jax.vmap(lambda x, q=q: budget.convecting_radius(x, q))(t_cmb))
        t_cen = np.asarray(jax.vmap(budget.profiles.t_cen)(t_cmb))
        tw = f'{q / 1e12:.0f} TW'
        ax.plot(t, (r_cmb - r_s) / 1e3, '--', color=col, label=f'Leeds, {tw}')
        ax.plot(t, (r_cmb - quasi) / 1e3, ':', color=col, label=f'quasi-static, {tw}')
        ax.plot(t, (r_cmb - capped) / 1e3, color=col, label=f'aragog, {tw}')
        ax_t.plot(t, t_cen_leeds, '--', color=col, label=f'Leeds, {tw}')
        ax_t.plot(t, t_cen, color=col, label=f'aragog, {tw}')
        tag = f'{q / 1e12:.0f}TW'
        record(6, f'tcen_max_abs_diff_K_{tag}', np.max(np.abs(t_cen - t_cen_leeds)))
        for when in (10.0, 50.0, 200.0, 500.0):
            k = int(np.argmin(np.abs(t - when)))
            leeds, cap = (r_cmb - r_s[k]) / 1e3, (r_cmb - capped[k]) / 1e3
            record(6, f'layer_leeds_km_{tag}_{when:.0f}myr', leeds)
            record(6, f'layer_aragog_km_{tag}_{when:.0f}myr', (r_cmb - quasi[k]) / 1e3)
            record(6, f'layer_capped_km_{tag}_{when:.0f}myr', cap)
            record(6, f'layer_capped_over_leeds_{tag}_{when:.0f}myr', cap / leeds)
            record(6, f'tcen_aragog_K_{tag}_{when:.0f}myr', t_cen[k])
            record(6, f'tcen_leeds_K_{tag}_{when:.0f}myr', t_cen_leeds[k])
    record(
        6, 'q_k_TW', float(budget.conducted_adiabatic_flow(r_cmb, inp['t_cmb_start'])) / 1e12
    )
    ax.set_xlabel('time (Myr)')
    ax.set_ylabel('layer thickness (km)')
    ax.legend(frameon=False, fontsize='x-small', ncols=2, loc='lower right')
    ax_t.set_xlabel('time (Myr)')
    ax_t.set_ylabel(r'$T_\mathrm{cen}$ (K)')
    ax_t.legend(frameon=False, fontsize='x-small', ncols=2)


def item10_leeds_history() -> None:
    """A core-only thermal history under a fixed CMB heat flow against thermal_history."""
    from scipy.integrate import solve_ivp

    header, th, budget, _ = _thermal_history()
    inp = header['inputs']
    rate = jax.jit(lambda t: budget.dtcmb_dt(t, inp['q_cmb']))
    sol = solve_ivp(
        lambda _, y: [float(rate(y[0])) * MYR_LEEDS],
        (th['time_myr'][0], th['time_myr'][-1]),
        [inp['t_cmb_start']],
        t_eval=th['time_myr'],
        rtol=1e-10,
        atol=1e-8,
        method='LSODA',
    )
    t_cmb = sol.y[0]
    r_icb = np.asarray(jax.vmap(budget.r_icb)(t_cmb))
    onset = th['time_myr'][np.argmax(r_icb > 0)]
    onset_th = th['time_myr'][np.argmax(th['r_icb'] > 0)]
    record(10, 'q_cmb_TW', inp['q_cmb'] / 1e12)
    record(10, 't_end_myr', th['time_myr'][-1])
    record(10, 't_cmb_max_abs_diff', np.max(np.abs(t_cmb - th['T_cmb'])))
    record(10, 'onset_myr_aragog', onset)
    record(10, 'onset_myr_leeds', onset_th)
    record(10, 'r_icb_end_km_aragog', r_icb[-1] / 1e3)
    record(10, 'r_icb_end_km_leeds', th['r_icb'][-1] / 1e3)
    before = np.arange(len(t_cmb)) < np.argmax(th['r_icb'] > 0)
    record(10, 't_cmb_max_abs_diff_before_onset', np.max(np.abs(t_cmb - th['T_cmb'])[before]))
    record(10, 't_cmb_abs_diff_at_onset', abs(t_cmb - th['T_cmb'])[np.argmax(~before)])
    record(10, 'r_icb_max_abs_diff_km', np.max(np.abs(r_icb - th['r_icb'])) / 1e3)
    record(10, 'inner_core_age_myr', th['time_myr'][-1] - onset)

    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1, figsize=(WIDTH, 6.4), sharex=True, height_ratios=(1.2, 1.2, 1)
    )
    t = th['time_myr']
    ax.plot(t, t_cmb, color=CORE, label='aragog')
    ax.plot(t, th['T_cmb'], '--', color=colour('ink', 'k'), label='thermal_history')
    ax.set_ylabel(r'$T_\mathrm{cmb}$ (K)')
    ax.legend(frameon=False, fontsize='small')
    ax2.plot(t, r_icb / 1e3, color=CORE)
    ax2.plot(t, th['r_icb'] / 1e3, '--', color=colour('ink', 'k'))
    ax2.set_ylabel(r'$r_\mathrm{icb}$ (km)')
    ax3.semilogy(t, np.maximum(np.abs(t_cmb - th['T_cmb']), 1e-12), color=CORE)
    ax3.set_ylabel(r'$|\Delta T_\mathrm{cmb}|$ (K)')
    ax3.set_xlabel('time (Myr)')
    save(fig, 'fig_17_leeds_thermal_history')


# ------------------------------------------------------------ 11. coupled PROTEUS
def _coupled(mode: str) -> dict:
    data = np.loadtxt(
        ROOT / 'tools' / 'verification' / 'data' / f'coupled_{mode}.csv', delimiter=','
    )
    return dict(zip(('t', 't_core', 't_node', 'f_cmb', 'phi', 'residual'), data.T))


def item11_coupled() -> None:
    """Flux sign, core insulation and solidification in a coupled PROTEUS run, core_module
    against energy_balance on the same configuration."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 5.6), sharex=True)
    for mode, col in (('core_module', CORE), ('energy_balance', colour('ink', 'k'))):
        run = _coupled(mode)
        live = run['t'] > 0
        contrast = run['t_core'] - run['t_node']
        wrong = live & (run['f_cmb'] * contrast < 0.0)
        record(11, f'rows_{mode}', live.sum())
        record(11, f'wrong_sign_rows_{mode}', wrong.sum())
        record(11, f'solidified_kyr_{mode}', run['t'][np.argmax(run['phi'] < 0.05)] / 1e3)
        record(11, f'contrast_min_K_{mode}', contrast[live].min())
        record(11, f'contrast_max_K_{mode}', contrast[live].max())
        record(11, f't_core_end_K_{mode}', run['t_core'][-1])
        record(11, f't_node_end_K_{mode}', run['t_node'][-1])
        t = run['t'][live]
        ax.semilogx(t, run['t_core'][live], color=col, label=f'core, {mode}')
        if mode == 'core_module':  # the mantle side cools alike in both runs
            mantle = dict(color=colour('fog', '0.5'), label='mantle side of the CMB')
            ax.semilogx(t, run['t_node'][live], '--', **mantle)
        ax2.semilogx(t, run['f_cmb'][live], color=col, label=mode)
        ax2.semilogx(t[wrong[live]], run['f_cmb'][live][wrong[live]], 'x', ms=3, color=col)
    run = _coupled('core_module')
    record(11, 'core_residual_frac_end', run['residual'][-1])
    record(
        11, 'core_residual_frac_max_after_1kyr', np.abs(run['residual'][run['t'] > 1e3]).max()
    )
    ax.set_ylabel('temperature (K)')
    ax.legend(frameon=False, fontsize='x-small', loc='lower left')
    ax2.set_yscale('symlog', linthresh=1.0)
    ax2.set_xlabel('time (yr)')
    ax2.set_ylabel(r'$F_\mathrm{cmb}$ (W m$^{-2}$)')
    ax2.legend(frameon=False, fontsize='x-small', loc='lower left')
    save(fig, 'fig_18_coupled_proteus')


# ---------------------------------------------------------------- 5. Nimmo (2015)
# Nimmo (2015, ch. 8.02) Table 4, p. 46: the 'This work (K = 0)' columns at 15.2 and 12 TW.
NIMMO_T4 = {
    15.2e12: dict(
        Qs=6.1,
        QL=5.7,
        Qg=3.4,
        Qk=15.0,
        Es=183.0,
        EL=327.0,
        Eg=809.0,
        Ek=450.0,
        cooling=104.0,
        growth=1050.0,
        age=0.59,
    ),
    12.0e12: dict(
        Qs=4.8,
        QL=4.5,
        Qg=2.7,
        Qk=15.0,
        Es=144.0,
        EL=258.0,
        Eg=639.0,
        Ek=450.0,
        cooling=82.0,
        growth=829.0,
        age=0.75,
    ),
}


def _nimmo_budget():
    """The Nimmo (2015, ch. 8.02) Table 2 core, as tests/test_core_entropy.py builds it."""
    budget = _core_budget(NIMMO, 'labrosse')
    return budget, CoreEntropyBudget(budget, k_core=NIMMO['k_core'])


def _nimmo_terms(budget, ent, t_c, q):
    """aragog's Table 4 quantities at CMB temperature t_c [K] and heat flow q [W]."""
    cooling = -float(budget.dtcmb_dt(t_c, q))  # K/s
    r_icb = float(budget.r_icb(t_c))
    drdt = float(jax.grad(budget.r_icb)(t_c))
    age = quad(lambda x: float(budget.effective_capacity(x)), t_c, float(budget.t_onset))[0] / q
    return dict(
        Qs=float(budget.secular_capacity()) * cooling / 1e12,
        QL=float(budget.latent_capacity(t_c)) * cooling / 1e12,
        Qg=float(budget.gravitational_capacity(t_c)) * cooling / 1e12,
        Qk=float(ent.adiabatic_heat_flow(t_c)) / 1e12,
        Es=float(ent.secular_entropy_capacity(t_c)) * cooling / 1e6,
        EL=float(ent.latent_entropy_capacity(t_c)) * cooling / 1e6,
        Eg=float(ent.gravitational_entropy_capacity(t_c)) * cooling / 1e6,
        Ek=float(ent.conduction_sink()) / 1e6,
        cooling=cooling * GYR,
        growth=-drdt * cooling * GYR / 1e3,
        age=age / GYR,
    ), r_icb


def item5_nimmo() -> None:
    """The present-day Earth budget of Nimmo (2015, ch. 8.02, Table 4) on its Table 2 core."""
    budget, ent = _nimmo_budget()
    t_1220 = brentq(
        lambda x: float(budget.r_icb(x)) - 1220e3, 3500.0, float(budget.t_onset) - 1e-6
    )
    record(5, 't_cmb_r1220_K', t_1220)
    record(5, 'delta_t_onset_K', float(budget.t_onset) - t_1220)
    for q in NIMMO_T4:  # Table 4's age matches Delta T_c / (dT_c/dt) at the present rate
        rate = -float(budget.dtcmb_dt(t_1220, q)) * GYR
        record(5, f'age_linear_{q / 1e12:g}TW', (float(budget.t_onset) - t_1220) / rate)
    record(5, 'Cr_m_per_K', -float(jax.grad(budget.r_icb)(t_1220)))
    names = list(NIMMO_T4[15.2e12])
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    x = np.arange(len(names))
    cases = (
        (15.2e12, 4180.0, r'15.2 TW, $T_c$ = 4180 K', CORE),
        (15.2e12, t_1220, r'15.2 TW, $r_\mathrm{icb}$ = 1220 km', colour('ocean', 'C0')),
        (12.0e12, t_1220, r'12 TW, $r_\mathrm{icb}$ = 1220 km', colour('fog', 'C7')),
    )
    for k, (q, t_c, label, col) in enumerate(cases):
        ours, r_icb = _nimmo_terms(budget, ent, t_c, q)
        ratio = [ours[n] / NIMMO_T4[q][n] for n in names]
        tag = f'{q / 1e12:g}TW_{"tc" if t_c == 4180.0 else "r1220"}'
        record(5, f'r_icb_km_{tag}', r_icb / 1e3)
        for n in names:
            record(5, f'{n}_{tag}', ours[n])
            record(5, f'{n}_ratio_{tag}', ours[n] / NIMMO_T4[q][n])
        ax.plot(x + 0.12 * (k - 1), ratio, 'o', color=col, label=label)
        if t_c != 4180.0:  # the age as Delta T_c over the present cooling rate
            linear = (float(budget.t_onset) - t_c) / ours['cooling'] / NIMMO_T4[q]['age']
            ax.plot(x[-1] + 0.12 * (k - 1), linear, 'o', mfc='none', color=col)
    ax.axhline(1.0, color=colour('fog', '0.6'), lw=0.8)
    ticks = ['$Q_s$', '$Q_L$', '$Q_g$', '$Q_k$', '$E_s$', '$E_L$', '$E_g$', '$E_k$']
    ax.set_xticks(x, ticks + [r'$\dot T_c$', r'$\dot r_\mathrm{icb}$', 'age'])
    ax.set_ylabel('aragog / Nimmo (2015)')
    ax.legend(frameon=False, fontsize='small')
    save(fig, 'fig_12_nimmo_budget')


ITEMS = {
    1: item1_structure,
    2: item2_energy,
    3: item3_flux,
    4: item4_nucleation,
    5: item5_nimmo,
    6: item6_leeds_terms,
    7: item7_dynamo,
    8: item8_melting,
    9: item9_cvode_onset,
    10: item10_leeds_history,
    11: item11_coupled,
    12: item12_jax_parity,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--only', default='', help='comma list of item numbers')
    args = parser.parse_args(argv)
    wanted = [int(i) for i in args.only.split(',') if i] or list(ITEMS)
    OUT.mkdir(parents=True, exist_ok=True)
    if VALUES_FILE.exists():
        VALUES.update(json.loads(VALUES_FILE.read_text()))
    start = time.perf_counter()
    for i in wanted:
        t0, VALUES[str(i)] = time.perf_counter(), {}
        ITEMS[i]()
        VALUES_FILE.write_text(json.dumps(VALUES, indent=1, sort_keys=True) + '\n')
        print(f'item {i}: {time.perf_counter() - t0:.1f} s', flush=True)
    print(
        f'total {time.perf_counter() - start:.1f} s; values in {VALUES_FILE.relative_to(ROOT)}'
    )


CORE, COLORS = style()

if __name__ == '__main__':
    main()
