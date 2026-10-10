"""Regenerate the figures and numbers of the core module verification page.

Writes ``docs/figures/vv/fig_08..fig_20_*.png`` (with a PDF beside each) and
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
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import cumulative_trapezoid, quad, solve_ivp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from aragog.core import (  # noqa: E402
    CoreEnergyBudget,
    CoreEntropyBudget,
    GaussianCoreProfiles,
    IronMeltingCurve,
    QuadraticMeltingCurve,
    build_core_module_budget,
    cmb_boundary_layer_flux,
)
from aragog.core import melting as m  # noqa: E402
from aragog.core.module import CORE_MODULE_KEYS  # noqa: E402

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
    record(1, 'tcen_ratio_exact', float(prof.t_cen(1.0)))
    small = GaussianCoreProfiles(**EARTH, adiabat_mode='small_radius')
    record(1, 'tcen_ratio_small_radius', float(small.t_cen(1.0)))
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
    small = under < 0.1  # the square root is the limit at the onset
    slope = np.polyfit(np.log(under[small]), np.log(r_icb[small]), 1)[0]
    kelvin = under < 1.0
    record(4, 'sqrt_slope_1K', np.polyfit(np.log(under[kelvin]), np.log(r_icb[kelvin]), 1)[0])
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
    """CoreEnergyBudget of the quadratic-melting-curve core that ``inp`` describes, on the
    small-radius adiabat that Nimmo (2015) and thermal_history use."""
    keys = ('rho_cen', 'length_scale', 'r_cmb', 'p_cmb', 'alpha', 'c_p')
    prof = GaussianCoreProfiles(
        **{k: inp[k] for k in keys}, pressure_mode=pressure_mode, adiabat_mode='small_radius'
    )
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
    record(7, 'F_const_flux_printed_inputs', 0.88 * printed)  # CHR09 factors, set here
    record(7, 'F_zero_outer_printed_inputs', 0.45 * printed)  # independently of aragog
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
    # The figure and the other values come from the last, rtol 1e-10, solve; atol has a 1e-8 floor.
    for rtol, key in ((1e-8, 'core_vs_cmb_rel_rtol1e-8'), (1e-10, 'core_vs_cmb_rel')):
        solver = build('core_module', eos_copy(), params, end_time=4.0, solver_method='cvode')
        solver.parameters.solver.rtol = rtol
        budget, n = solver._core_module_budget, solver._n_stag
        t_on = float(budget.t_onset)
        solver.set_initial_core_temperature(t_on + 2.0)
        solver.set_initial_entropy(np.linspace(7000.0, 6700.0, n))
        solver.solve()
        out = solver.get_state()
        record(9, key, abs(out.step_dE_core_J / -out.step_dE_F_cmb_J - 1))
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
        solver = build('core_module', eos_copy(identity_alpha=True), params, s_init='driven')
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
    # Over a mantle above its liquidus the single-phase expansivity enters: the tables against
    # the identity rho cp |dT/dP_S| / T that the JAX EOS uses.
    for identity, key in ((True, 'identity'), (False, 'tables')):
        eos = eos_copy(identity_alpha=identity)
        solver = build('core_module', eos, core_params)
        n = solver._n_stag
        y = np.asarray(solver._S0, dtype=float)
        y[:n] = np.asarray(eos.liquidus_entropy(solver._P_stag_flat)) + 300.0
        y[n + 1] = (
            float(np.asarray(eos.temperature(solver._P_basic_flat[0], y[0])).flat[0]) + 50
        )
        f_np = np.asarray(solver.dSdt(0.0, y)).ravel()
        f_jax = np.asarray(dSdt_core_module(0.0, jnp.asarray(y), _build_jax_pieces(solver)))
        rel = np.abs(f_jax - f_np) / np.maximum(np.abs(f_np), 1e-300)
        record(12, f'rhs_liquid_max_rel_{key}', rel.max())
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
with LAYER_TABLE.open() as _fh:
    LAYER_HEADER = json.loads(_fh.readline()[2:])


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
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
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
    save(fig, 'fig_13_leeds_budget_terms')


def _theta_base(shell, t_shell, t_c, fraction=0.1):
    """Layer base [m] by temperature, as thermal_history lowers r_s to T_s = T_a: the deepest
    radius, reached from the top, where the excess over the adiabat is still ``fraction`` of
    its top value, interpolated between cells; the CMB without a warm top."""
    theta = t_shell - np.asarray(shell.adiabatic_profile(t_c))
    r, target = np.asarray(shell.r_cells), fraction * theta[-1]
    if theta[-1] <= 0.0:
        return float(shell.profiles.r_cmb)
    k = len(theta) - 1
    while k > 0 and theta[k - 1] >= target:
        k -= 1
    if k == 0:
        return float(shell.r_base)
    w = (target - theta[k - 1]) / (theta[k] - theta[k - 1])
    return float(r[k - 1] + w * (r[k] - r[k - 1]))


# Relative tolerances of the layer comparison, of its unconverged run and of its tighter check
RTOL_LAYER, RTOL_LOOSE, RTOL_TIGHT = 1e-11, 1e-8, 1e-12
DEPTH_LATE_MYR = 300.0  # the depths are compared closely from here on
DEPTH_SETTLED_MYR = 200.0  # the reference depth hardly depends on the flow from here on
_RATES = {}


def _rates(budget):
    """Jitted rate and Jacobian of the convecting core and shell, one pair per budget."""
    if budget not in _RATES:
        rate = jax.jit(
            lambda y, q: jnp.concatenate(
                [jnp.atleast_1d(v) for v in budget.core_rates(y[0], y[1:], q)]
            )
        )
        _RATES[budget] = rate, jax.jit(jax.jacfwd(rate))
    return _RATES[budget]


SHELL_RESTARTS = []  # BDF restarts of each shell run, for item 13


def _shell_run(budget, segments, times, rtol=None, diagnostics=True):
    """aragog's convecting core and shell from the adiabat at the table's start temperature
    (no layer) under the CMB flows ``segments`` [(end Myr, W)], one SciPy BDF solve per flow
    with atol 100 rtol, sampled at ``times`` [Leeds Myr], which hold every segment end but the
    last.

    ``identity`` is the largest departure of the core and shell heat rates from the CMB flow,
    over the samples, relative to the conducted adiabatic flow at the start; ``top_mixed``
    marks the samples where the mixing acts on the face below the top cell."""
    shell, p, rtol = budget.shell, budget.profiles, rtol or RTOL_LAYER
    rate, jac = _rates(budget)
    t0 = LAYER_HEADER['inputs']['t_cmb_start']
    y, start, states, flows = (
        np.concatenate([[t0], np.asarray(shell.adiabatic_profile(t0))]),
        0.0,
        [],
        [],
    )
    sampled, restarts = -np.inf, 0
    for end, q in segments:  # no step crosses a change of flow
        for _ in range(4):  # a step that stalls at a kink of the mixing restarts from there
            sol = solve_ivp(
                lambda _, y, q=q: np.asarray(rate(y, q)) * MYR_LEEDS,
                (start, end),
                y,
                method='BDF',
                jac=lambda _, y, q=q: np.asarray(jac(y, q)) * MYR_LEEDS,
                rtol=rtol,
                atol=100 * rtol,
                dense_output=True,
            )
            t_new = times[(times > sampled) & (times <= sol.t[-1])]
            if t_new.size:
                states += list(sol.sol(t_new).T)
                flows += [q] * t_new.size
                sampled = t_new[-1]
            y, start = sol.y[:, -1], sol.t[-1]
            if sol.success:
                break
            restarts += 1
        assert sol.success, (sol.message, end)
    SHELL_RESTARTS.append(restarts)
    ys = np.array(states)
    t_c, t_shell = ys[:, 0], ys[:, 1:]
    out = {
        't_cen': np.asarray(jax.vmap(p.t_cen)(t_c)),
        'r_icb': np.asarray(jax.vmap(budget.r_icb)(t_c)),
    }
    if not diagnostics:
        return out

    def heat_rate(y, q):
        d_core, d_shell = budget.core_rates(y[0], y[1:], q)
        upper = shell.layer_base(y[1:], y[0])
        return (
            budget.effective_capacity(y[0], gravitational_upper=upper) * d_core
            + p.c_p * (shell.mass @ d_shell)
            + q
        )

    def top_mixed(y):
        return shell._anomaly_gradient(y[1:], y[0])[0][-1] < 0.0

    q_k = float(budget.conducted_adiabatic_flow(p.r_cmb, t0))
    return out | {
        't_top': t_shell[:, -1],
        'depth': p.r_cmb - np.array([_theta_base(shell, s, c) for s, c in zip(t_shell, t_c)]),
        'top_mixed': np.asarray(jax.vmap(top_mixed)(jnp.asarray(ys))),
        'stored': p.c_p
        * (t_shell - np.asarray(jax.vmap(shell.adiabatic_profile)(t_c)))
        @ np.asarray(shell.mass),
        'q_ad': np.asarray(
            jax.vmap(lambda t: budget.conducted_adiabatic_flow(p.r_cmb, t))(t_c)
        ),
        'identity': np.max(
            np.abs(jax.vmap(heat_rate)(jnp.asarray(ys), jnp.asarray(flows, dtype=float)))
        )
        / q_k,
    }


def _layer_cases():
    """thermal_history's layer cases: name -> (flow segments, columns)."""
    data = np.loadtxt(LAYER_TABLE, delimiter=',', comments='#')
    names = LAYER_HEADER['columns']
    out = {}
    for case, (name, segments) in enumerate(LAYER_HEADER['cases'].items()):
        rows = data[data[:, 0] == case]
        out[name] = ([tuple(s) for s in segments], dict(zip(names, rows.T)))
    return out


def _core_only_run(budget, segments, times):
    """Centre temperature [K] of aragog's core without a layer under the CMB flows
    ``segments`` [(end Myr, W)], sampled at ``times`` [Leeds Myr], from the table's start."""
    rate = jax.jit(lambda t, q: budget.dtcmb_dt(t, q) * MYR_LEEDS)
    jac = jax.jit(jax.grad(rate))
    y, start, states = [LAYER_HEADER['inputs']['t_cmb_start']], 0.0, []
    for end, q in segments:
        sol = solve_ivp(
            lambda _, y, q=q: [float(rate(y[0], q))],
            (start, end),
            y,
            t_eval=times[(times >= start) & (times <= end)][int(start > 0) :],
            method='BDF',
            jac=lambda _, y, q=q: [[float(jac(y[0], q))]],
            rtol=RTOL_LAYER,
            atol=100 * RTOL_LAYER,
        )
        assert sol.success and (sol.t[-1] == end or end >= times[-1]), (sol.message, end)
        states += list(sol.y[0])
        y, start = sol.y[:, -1], end
    return np.asarray(jax.vmap(budget.profiles.t_cen)(jnp.asarray(states)))


def _first(t, mask):
    """First sample [Myr] where ``mask`` holds; an event that never happens is an error."""
    if not mask.any():
        raise ValueError('the event does not occur in the run')
    return t[np.argmax(mask)]


def _compare(tag, ours, th):
    """Record one layer case against thermal_history: T_cmb, T_cen, theta-0.1 depth ratio,
    inner-core radius at the end and its onset."""
    t = th['time_myr']
    record(13, f'tcmb_max_abs_diff_K_{tag}', np.max(np.abs(ours['t_top'] - th['T_cmb'])))
    record(13, f'tcen_max_abs_diff_K_{tag}', np.max(np.abs(ours['t_cen'] - th['T_cen'])))
    both = (ours['depth'] > 10e3) & (th['depth_theta'] > 10e3) & (t >= 1.0)
    ratio = ours['depth'][both] / th['depth_theta'][both]
    record(13, f'layer_ratio_min_{tag}', ratio.min())
    record(13, f'layer_ratio_max_{tag}', ratio.max())
    record(13, f'identity_{tag}', ours['identity'])
    record(13, f'ricb_end_rel_diff_{tag}', abs(ours['r_icb'][-1] / th['r_icb'][-1] - 1))
    record(13, f'onset_myr_aragog_{tag}', _first(t, ours['r_icb'] > 0))
    record(13, f'onset_myr_leeds_{tag}', _first(t, th['r_icb'] > 0))
    k = np.max(np.abs(ours['t_top'] - th['T_cmb']))
    return (
        np.max(np.abs(ratio[t[both] >= DEPTH_LATE_MYR] - 1)),
        t[both][[np.argmin(ratio), np.argmax(ratio)]].max(),
        t[np.argmax(np.abs(ours['t_top'] - th['T_cmb']))],
        (ours['t_top'] - th['T_cmb'])[np.argmax(np.abs(ours['t_top'] - th['T_cmb']))] / k,
    )


def item13_stable_layer() -> None:
    """The resolved layer against leeds_thermal under fixed CMB flows below the conducted
    adiabatic flow, under a flow that erodes the layer and then lets it re-form, and the
    change under the mixing constants (model uncertainty)."""
    inp, cases = LAYER_HEADER['inputs'], _layer_cases()
    SHELL_RESTARTS.clear()
    budget = _core_budget(inp, 'quadrature', stratification=True, k_core=inp['k_core'])
    r_cmb, shell = inp['r_cmb'], budget.shell
    record(
        13, 'q_k_TW', float(budget.conducted_adiabatic_flow(r_cmb, inp['t_cmb_start'])) / 1e12
    )
    record(13, 'top_cell_depth_m', r_cmb - float(shell.r_cells[-1]))
    record(
        13,
        'top_cell_offset_K',
        float(shell.adiabatic_profile(inp['t_cmb_start'])[-1]) - inp['t_cmb_start'],
    )
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 8.0), sharex='row')
    (a_dep, a_cen), (a_icb, a_warm), (a_ero, a_ero_dep) = axes
    record(13, 't_cmb_start_K', inp['t_cmb_start'])
    record(13, 'shell_base_fraction', float(shell.r_base) / r_cmb)
    record(13, 'shell_cells', shell.n_cells)
    record(13, 'rtol', RTOL_LAYER)
    record(13, 'rtol_loose', RTOL_LOOSE)
    record(13, 'rtol_tight', RTOL_TIGHT)
    record(13, 'depth_late_from_myr', DEPTH_LATE_MYR)
    runs, late, loose_cen, loose_onset, noise = {}, [], [], [], []
    for name, col in zip(('8 TW', '12 TW', '-2 TW', '0 TW'), (CORE, colour('ocean', 'C0')) * 2):
        segments, th = cases[name]
        t, tag = th['time_myr'], name.replace(' ', '')
        runs[name] = ours = _shell_run(budget, segments, t)
        late.append(_compare(tag, ours, th))
        loose = _shell_run(budget, segments, t, rtol=RTOL_LOOSE, diagnostics=False)
        loose_cen.append(np.max(np.abs(loose['t_cen'] - th['T_cen'])))
        loose_onset.append(abs(_first(t, loose['r_icb'] > 0) - _first(t, th['r_icb'] > 0)))
        tight = _shell_run(budget, segments, t, rtol=RTOL_TIGHT, diagnostics=False)
        noise.append(np.max(np.abs(tight['t_cen'] - ours['t_cen'])))
        panels = (
            (
                (a_dep, ours['depth'] / 1e3, th['depth_theta'] / 1e3),
                (a_cen, ours['t_cen'], th['T_cen']),
                (a_icb, ours['r_icb'] / 1e3, th['r_icb'] / 1e3),
            )
            if name in ('8 TW', '12 TW')
            else ((a_warm, ours['t_top'], th['T_cmb']),)
        )
        for ax, a, b in panels:
            ax.plot(t, b, '--', color=col)
            ax.plot(t, a, color=col, label=name)
    record(13, 'depth_late_max_rel', max(x[0] for x in late))
    record(13, 'depth_ratio_extremes_last_myr', max(x[1] for x in late))
    record(13, 'tcmb_max_diff_myr_first', min(x[2] for x in late))
    record(13, 'tcmb_max_diff_myr_last', max(x[2] for x in late))
    record(13, 'tcmb_max_diff_sign_max', max(x[3] for x in late))
    th = [cases[n][1] for n in ('8 TW', '12 TW', '-2 TW', '0 TW')]
    t, onset = th[0]['time_myr'], min(_first(c['time_myr'], c['r_icb'] > 0) for c in th)
    before, settled = t < onset, (t >= DEPTH_SETTLED_MYR) & (t < onset)
    record(13, 'leeds_depth_settled_from_myr', DEPTH_SETTLED_MYR)
    record(13, 'leeds_onset_first_myr', onset)
    record(13, 'fixed_end_myr', t[-1])
    record(
        13, 'leeds_tcen_flow_spread_K', np.ptp([c['T_cen'][before] for c in th], axis=0).max()
    )
    depths = np.array([c['depth_theta'][settled] for c in th])
    record(13, 'leeds_depth_flow_spread_rel', (np.ptp(depths, axis=0) / depths.mean(0)).max())
    record(13, 'tcen_max_abs_diff_K_rtol1e-8', max(loose_cen))
    record(13, 'onset_max_abs_diff_myr_rtol1e-8', max(loose_onset))
    record(13, 'tcen_max_rtol_noise_K', max(noise))

    # Erosion: aragog's top mixes at once and the mixed region grows down; the heat stored in
    # the shell drains through the CMB at the flow above Q_ad until the whole shell mixes.
    segments, th = cases['erosion']
    t, start, end, q_erode = th['time_myr'], segments[0][0], segments[1][0], segments[1][1]
    ero = _shell_run(budget, segments, t)
    after = t > start
    removed = _first(t, after & (ero['depth'] <= 0.0))
    removed_leeds = _first(t, after & (th['depth_theta'] <= 0.0))
    diff = ero['t_top'] - th['T_cmb']
    both, window, gone = (
        (t >= start) & (t <= removed_leeds),
        (t >= start) & (t <= removed),
        (t > removed) & (t <= end),
    )
    record(13, 'erosion_start_myr', start)
    record(13, 'erosion_base_flow_TW', segments[0][1] / 1e12)
    record(13, 'erosion_flow_TW', q_erode / 1e12)
    record(13, 'erosion_fall_myr', end)
    record(13, 'erosion_last_segment_end_myr', segments[-1][0])
    record(13, 'erosion_cmb_mixed_myr_aragog', _first(t, after & ero['top_mixed']))
    record(13, 'erosion_removed_myr_aragog', removed)
    record(13, 'erosion_mixing_myr', removed - start)
    record(13, 'erosion_removed_myr_leeds', removed_leeds)
    record(13, 'erosion_rs_removed_myr_leeds', _first(t, after & (th['depth_r_s'] <= 0.0)))
    record(13, 'erosion_tcmb_max_diff_K_both_layers', diff[both][np.argmax(np.abs(diff[both]))])
    record(13, 'erosion_tcmb_max_diff_myr', t[both][np.argmax(np.abs(diff[both]))])
    record(13, 'erosion_tcmb_offset_K_after', np.mean(diff[gone]))
    record(13, 'erosion_tcmb_offset_spread_K_after', np.ptp(diff[gone]))
    record(13, 'erosion_tcen_max_abs_diff_K', np.max(np.abs(ero['t_cen'] - th['T_cen'])))
    plain = _core_only_run(_core_budget(inp, 'quadrature', k_core=inp['k_core']), segments, t)
    record(13, 'erosion_tcen_self_max_K', np.max(np.abs(ero['t_cen'] - plain)[gone]))
    record(13, 'erosion_tcen_self_at_start_K', (ero['t_cen'] - plain)[t == start][0])
    record(13, 'erosion_identity', ero['identity'])
    record(13, 'erosion_stored_J', float(ero['stored'][t == start][0]))
    drain = q_erode - np.mean(ero['q_ad'][window])
    record(13, 'erosion_drain_TW', drain / 1e12)
    record(13, 'erosion_drain_myr', float(ero['stored'][t == start][0]) / drain / MYR_LEEDS)
    record(13, 'erosion_end_myr', t[-1])
    record(13, 'erosion_depth_km_end_aragog', ero['depth'][-1] / 1e3)
    record(13, 'erosion_depth_km_end_leeds', th['depth_theta'][-1] / 1e3)
    depth = np.where(ero['top_mixed'], np.nan, ero['depth'])  # no stable top, no depth
    for ax, a, b in (
        (a_ero, ero['t_top'], th['T_cmb']),
        (a_ero_dep, depth / 1e3, th['depth_theta'] / 1e3),
    ):
        ax.plot(t, b, '--', color=CORE)
        ax.plot(t, a, color=CORE, label='erosion')
        ax.axvspan(start, removed, color=colour('fog', '0.9'), alpha=0.25, lw=0)

    # Model uncertainty: the mixing constants changed one at a time
    base_k, base_g, shifts = shell.k_mix, shell.g_mix, []
    for k_mix, g_mix in (
        (10 * base_k, base_g),
        (0.1 * base_k, base_g),
        (base_k, 10 * base_g),
        (base_k, 0.1 * base_g),
    ):
        b = _core_budget(
            inp,
            'quadrature',
            stratification=True,
            k_core=inp['k_core'],
            layer={'k_mix': k_mix, 'g_mix': g_mix},
        )
        for name in ('8 TW', '12 TW'):
            run, ref = _shell_run(b, cases[name][0], cases[name][1]['time_myr']), runs[name]
            deep = ref['depth'] > 10e3
            shifts.append(
                (
                    np.max(np.abs(run['t_cen'] - ref['t_cen'])),
                    np.max(np.abs(run['depth'][deep] / ref['depth'][deep] - 1)),
                    0.0,
                )
            )
        run = _shell_run(b, segments, t)
        shifts.append((0.0, 0.0, abs(_first(t, after & (run['depth'] <= 0.0)) - removed)))
    record(13, 'mixing_tcen_max_K', max(s[0] for s in shifts))
    # smallest T_cen change of the four settings in the 8 and 12 TW runs: zero if one is unused
    record(
        13, 'mixing_tcen_min_K', min(max(s[0] for s in shifts[i : i + 2]) for i in (0, 3, 6, 9))
    )
    record(13, 'mixing_depth_max_rel', max(s[1] for s in shifts))
    record(13, 'mixing_removal_max_myr', max(s[2] for s in shifts))

    labels = (
        'layer depth (km)',
        r'$T_\mathrm{cen}$ (K)',
        r'$r_\mathrm{icb}$ (km)',
        r'$T_\mathrm{cmb}$ (K)',
        r'$T_\mathrm{cmb}$ (K)',
        'layer depth (km)',
    )
    for ax, label, letter in zip(axes.ravel(), labels, 'abcdef'):
        ax.set_ylabel(label)
        ax.text(0.02, 0.9, f'({letter})', transform=ax.transAxes)
    for ax in axes[1:].ravel():
        ax.set_xlabel('time (Myr)')
    for ax in (a_dep, a_warm, a_ero):
        ax.legend(frameon=False, fontsize='x-small')
    save(fig, 'fig_20_leeds_stable_layer')
    record(13, 'bdf_restarts', sum(SHELL_RESTARTS))
    # runs of this item separate from the script, with their commits and platforms in the file
    spread = json.loads((ROOT / 'tools/verification/data/stable_layer_spread.json').read_text())
    base, bit, tight = (spread['runs'][k] for k in ('base', 'last_bit', 'rtol1e-12'))
    tcen, ricb = 'tcen_max_abs_diff_K_', 'ricb_end_rel_diff_'
    t2, linux = f'{tcen}-2TW', spread['runs']['linux'][f'{tcen}8TW']
    record(13, 'spread_tcen_-2TW_last_bit_K', bit[t2])
    record(13, 'spread_tcen_-2TW_rtol1e-12_K', tight[t2])
    record(13, 'spread_tcen_-2TW_changed_runs_K', abs(bit[t2] - tight[t2]))
    shift = {k: abs(bit[k] - base[k]) for k in base}
    others = ('8TW', '12TW', '0TW')
    record(13, 'spread_tcen_-2TW_last_bit_shift_K', shift[t2])
    record(13, 'spread_tcen_others_max_K', max(shift[f'{tcen}{t}'] for t in others))
    record(13, 'spread_ricb_others_max', max(shift[f'{ricb}{t}'] for t in others))
    record(13, 'spread_ricb_-2TW', shift[f'{ricb}-2TW'])
    record(13, 'spread_depth_late', shift['depth_late_max_rel'])
    record(13, 'spread_mixing_tcen_K', shift['mixing_tcen_max_K'])
    record(13, 'spread_mixing_depth_rtol1e-12', tight['mixing_depth_max_rel'])
    record(13, 'linux_tcen_8TW_K', linux)
    record(13, 'linux_tcen_8TW_shift_K', linux - base[f'{tcen}8TW'])


def item10_leeds_history() -> None:
    """A core-only thermal history under a fixed CMB heat flow against thermal_history."""
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
    t_cmb, k = sol.y[0], np.argmax(th['r_icb'] > 0)
    r_icb, diff = np.asarray(jax.vmap(budget.r_icb)(t_cmb)), t_cmb - th['T_cmb']
    onset = th['time_myr'][np.argmax(r_icb > 0)]
    record(10, 'q_cmb_TW', inp['q_cmb'] / 1e12)
    record(10, 't_end_myr', th['time_myr'][-1])
    record(10, 't_cmb_max_abs_diff', np.max(np.abs(diff)))
    record(10, 't_cmb_max_abs_diff_myr', th['time_myr'][np.argmax(np.abs(diff))])
    record(10, 'onset_myr_aragog', onset)
    record(10, 'onset_myr_leeds', th['time_myr'][k])
    record(10, 'r_icb_end_km_aragog', r_icb[-1] / 1e3)
    record(10, 'r_icb_end_km_leeds', th['r_icb'][-1] / 1e3)
    record(10, 't_cmb_max_abs_diff_before_onset', np.max(np.abs(diff[:k])))
    record(10, 't_cmb_abs_diff_at_onset', abs(diff[k]))
    record(10, 't_cmb_diff_max_after_onset_K', diff[k:].max())
    flip = np.flatnonzero(np.sign(diff[k:]) != np.sign(diff[k]))[0]
    record(10, 'sign_change_myr', th['time_myr'][k + flip])
    record(10, 'r_icb_max_abs_diff_km', np.max(np.abs(r_icb - th['r_icb'])) / 1e3)
    record(10, 'r_icb_end_rel_diff', abs(r_icb[-1] / th['r_icb'][-1] - 1))
    t_cen = np.asarray(jax.vmap(budget.profiles.t_cen)(t_cmb))
    record(10, 't_cen_max_abs_diff', np.max(np.abs(t_cen - th['T_cen'])))
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
    ax3.semilogy(t, np.maximum(np.abs(diff), 1e-12), color=CORE)
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
        zero = live & (run['f_cmb'] == 0.0)
        record(11, f'zero_flux_rows_{mode}', zero.sum())
        record(11, f'zero_flux_contrast_max_K_{mode}', np.abs(contrast[zero]).max(initial=0.0))
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
    ax.set_ylabel('temperature (K)')
    ax.legend(frameon=False, fontsize='x-small', loc='lower left')
    ax2.set_yscale('symlog', linthresh=1.0)
    ax2.set_xlabel('time (yr)')
    ax2.set_ylabel(r'$F_\mathrm{cmb}$ (W m$^{-2}$)')
    ax2.legend(frameon=False, fontsize='x-small', loc='lower left')
    save(fig, 'fig_18_coupled_proteus')
    for case in ('1me', '1me_rtol1e-10', '3me', '5me'):
        path = ROOT / 'tools' / 'verification' / 'data' / f'coupled_ledger_{case}.csv'
        t, core, cmb, impact, frac = np.loadtxt(path, delimiter=',').T
        step = core + cmb - impact
        record(11, f'closure_end_{case}', abs(frac[-1]))
        record(11, f'closure_max_after_1kyr_{case}', np.abs(frac[t > 1e3]).max())
        first = np.flatnonzero(cmb)[0]  # first call of nonzero length, 1 to 2 yr
        record(11, f'closure_first_call_{case}', abs(step[first] / cmb[first]))
        share = abs(step[abs(t - 122.0) < 0.5].item()) / np.abs(step).sum()
        record(11, f'share_22_122yr_{case}', share)
    data = ROOT / 'tools' / 'verification' / 'data'
    for case in ('3me', '5me', 'wb_1me'):
        t, t_core, t_node, f_cmb = np.loadtxt(
            data / f'coupled_flux_{case}.csv', delimiter=','
        ).T
        live, contrast = t > 0, t_core - t_node
        record(11, f'rows_{case}', live.sum())
        record(11, f'wrong_sign_rows_{case}', (live & (f_cmb * contrast < 0.0)).sum())
        zero = live & (f_cmb == 0.0)
        record(11, f'zero_flux_rows_{case}', zero.sum())
        record(11, f'zero_flux_contrast_max_K_{case}', np.abs(contrast[zero]).max(initial=0.0))
        record(11, f'contrast_min_K_{case}', contrast[live].min())
        record(11, f'contrast_max_K_{case}', contrast[live].max())
    config = tomllib.loads((data / 'coupled_config.toml').read_text())
    params = config['interior_energetics']['aragog']['core_module']
    params = {k: v for k, v in params.items() if k in CORE_MODULE_KEYS}
    for mass, r_cmb, p_cmb, rho_cen, length, _, t_end in np.loadtxt(
        data / 'coupled_core_structure.csv', delimiter=','
    ):
        profile = dict(rho_cen=rho_cen, length_scale=length)
        budget = build_core_module_budget(params | profile, r_cmb=r_cmb, p_cmb_fallback=p_cmb)
        tag = f'{mass:g}me'
        record(11, f'p_cmb_GPa_{tag}', p_cmb / 1e9)
        record(11, f'p_cen_GPa_{tag}', float(budget.profiles.pressure(0.0)) / 1e9)
        record(11, f't_onset_K_{tag}', budget.t_onset)
        record(11, f't_freeze_K_{tag}', budget.t_freeze)
        record(11, f't_cmb_end_K_{tag}', t_end)
    # E1: the 1 Earth-mass run at 40 to 320 mantle levels, and at rtol 1e-10 on 80
    rows = np.loadtxt(data / 'coupled_mesh_convergence.csv', delimiter=',')
    names = 't_bf T_core_bf T_core_end F10 F100 F1000 F_bf F_2bf F_4bf F_10kyr F_100kyr F_500kyr'.split()
    run = {(int(r[0]), r[1]): dict(zip(names, r[2:])) for r in rows}
    sets = {  # the largest change of each set is recorded too
        'bounded': ('t_bf', 'F10', 'F100', 'F1000', 'F_bf'),
        'late': ('F_2bf', 'F_4bf', 'F_10kyr', 'F_100kyr', 'F_500kyr'),
    }
    for a, b in ((40, 80), (80, 160), (160, 320), (80, 'rtol')):
        x, y = run[(a, 1e-8)], run[(b, 1e-8)] if b != 'rtol' else run[(80, 1e-10)]
        d = {}
        for q in names:
            diff = abs(x[q] - y[q]) if q.startswith('T_core') else abs(x[q] / y[q] - 1)
            d[q] = record(11, f'e1_{q}_d_{a}_{b}', diff)
        for name, qs in sets.items():
            record(11, f'e1_{name}_max_d_{a}_{b}', max(d[q] for q in qs))
    for q in names:
        record(11, f'e1_{q}_320', run[(320, 1e-8)][q])
    # the call-mean flux at 100 kyr of the two tolerances, placed at the middle of its call
    mid = {}
    for case in ('1me', '1me_rtol1e-10'):
        t, _, cmb, _, _ = np.loadtxt(data / f'coupled_ledger_{case}.csv', delimiter=',').T
        live = np.diff(t) > 0
        dt, end, heat = np.diff(t)[live], t[1:][live], cmb[1:][live]
        record(11, f'e1_call_kyr_100kyr_{case}', dt[np.searchsorted(end, 1e5)] / 1e3)
        mid[case] = np.interp(1e5, end - dt / 2, heat / dt)
    record(11, 'e1_F_100kyr_d_80_rtol_midcall', abs(mid['1me'] / mid['1me_rtol1e-10'] - 1))
    for n in (40, 80, 160, 320):  # the late flux and the cooling after basal freezing
        record(11, f'e1_F_500kyr_{n}', run[(n, 1e-8)]['F_500kyr'])
        record(
            11,
            f'e1_cooling_after_bf_K_{n}',
            run[(n, 1e-8)]['T_core_bf'] - run[(n, 1e-8)]['T_core_end'],
        )


# ---------------------------------------------------------------- 5. Nimmo (2015)
NIMMO = dict(  # Nimmo (2015, ch. 8.02) Table 2 core
    rho_cen=12500.0,
    length_scale=7272e3,
    r_cmb=3480e3,
    p_cmb=139e9,
    alpha=1.25e-5,
    c_p=840.0,
    t_m0=2677.0,
    t_m1=2.95e-12,
    t_m2=8.37e-25,
    latent_heat=750e3,
    alpha_c=1.0,
    c_light=560.0 / 12150.0,
    k_core=130.0,
)
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


# Nimmo (2015, ch. 8.02) Table 5, p. 46, 'This work': energies [1e28 J] released since
# inner-core onset and the inner-core age [Gyr] under a constant 10 TW.
NIMMO_T5 = dict(
    Ws=11.6, Wg=4.4, WL=6.9, Wtot=23.0, age_10TW=0.73, delta_t=60.0, tm_centre=5800.0
)


def item5_nimmo() -> None:
    """The present-day Earth budget of Nimmo (2015, ch. 8.02, Tables 4 and 5) on its Table 2
    core at the chapter's CMB temperature."""
    budget, ent = _nimmo_budget()
    t_c, t_on = 4180.0, float(budget.t_onset)
    r_icb = record(5, 'r_icb_km', float(budget.r_icb(t_c)) / 1e3)
    record(5, 'p_icb_GPa', float(budget.profiles.pressure(r_icb * 1e3)) / 1e9)
    record(5, 'delta_t_onset_K', t_on - t_c)
    record(5, 'tm_centre_K', float(budget.melting_curve.t_melt(budget.profiles.pressure(0.0))))
    record(5, 'Cr_m_per_K', -float(jax.grad(budget.r_icb)(t_c)))
    released = {
        'Ws': float(budget.secular_capacity()) * (t_on - t_c),
        'WL': quad(lambda x: float(budget.latent_capacity(x)), t_c, t_on)[0],
        'Wg': quad(lambda x: float(budget.gravitational_capacity(x)), t_c, t_on)[0],
    }
    released['Wtot'] = sum(released.values())
    for n, w in released.items():
        record(5, f'{n}_1e28J', w / 1e28)
        record(5, f'{n}_ratio', w / 1e28 / NIMMO_T5[n])
    record(5, 'age_10TW_Gyr', released['Wtot'] / 10e12 / GYR)
    # The chapter's own secular capacity (Table 4, Q_s over the cooling rate) times its drop.
    capacity_t4 = NIMMO_T4[15.2e12]['Qs'] * 1e12 * GYR / NIMMO_T4[15.2e12]['cooling']
    record(5, 'Ws_from_table4_1e28J', capacity_t4 * NIMMO_T5['delta_t'] / 1e28)
    record(5, 'secular_capacity_vs_table4', released['Ws'] / (t_on - t_c) / capacity_t4)
    names = list(NIMMO_T4[15.2e12])
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    x = np.arange(len(names))
    for k, (q, col) in enumerate(((15.2e12, CORE), (12.0e12, colour('ocean', 'C0')))):
        ours, _ = _nimmo_terms(budget, ent, t_c, q)
        tag = f'{q / 1e12:g}TW'
        for n in names:
            record(5, f'{n}_{tag}', ours[n])
            record(5, f'{n}_ratio_{tag}', ours[n] / NIMMO_T4[q][n])
        linear = record(5, f'age_linear_{tag}', (t_on - t_c) / ours['cooling'])
        ratio = [ours[n] / NIMMO_T4[q][n] for n in names]
        ax.plot(x + 0.12 * (2 * k - 1), ratio, 'o', color=col, label=f'{q / 1e12:g} TW')
        ax.plot(
            x[-1] + 0.12 * (2 * k - 1), linear / NIMMO_T4[q]['age'], 'o', mfc='none', color=col
        )
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
    13: item13_stable_layer,
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
