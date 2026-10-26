"""Calibrated viscous mixing length of the solid-state MLT closure.

Module under test: ``aragog.rheology.viscous_mixing_length_factor`` and
``aragog.rheology.compute_mlt_top_slope``. With rheology enabled, the viscous
branch uses ``l_v = min(s_bot (r - r_in), s_top(theta) (r_out - r))`` in place
of the Abe length ``l``: the viscous eddy diffusivity scales by ``(l_v / l)^4``
and the inviscid branch keeps ``l``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aragog.rheology import (
    SolidRheologyParams,
    compute_mlt_top_slope,
    viscous_mixing_length_factor,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

TOP = -3  # Node near the surface where the top slope sets l_v.
BOTTOM = 3  # Node near the CMB where the bottom slope sets l_v.
RE_CRIT = 9.0 / 8.0


def _mlt_kappa_h(
    log10visc: float,
    radii: np.ndarray,
    mixing_length: np.ndarray,
    q_factor: np.ndarray | float = 1.0,
    xp=np,
):
    """Evaluate the local MLT viscous-inviscid eddy diffusivity blend with factor q."""
    rho = 4000.0
    cp = 1200.0
    alpha = 3.0e-5
    g = 9.81
    temp = 2000.0
    drive = 1.0e-6
    nu = (10.0**log10visc) / rho

    prefactor = alpha * g * temp * drive / cp
    v_visc_abe = prefactor * (mixing_length**3) / (18.0 * nu)
    v_invisc = xp.sqrt(prefactor * (mixing_length**2) / 16.0)

    v_visc = q_factor * v_visc_abe
    reynolds = q_factor * (v_visc * mixing_length / nu)
    blend_width = 0.01 * RE_CRIT
    w_invisc = 0.5 * (1.0 + xp.tanh((reynolds - RE_CRIT) / blend_width))
    v_conv = (1.0 - w_invisc) * v_visc + w_invisc * v_invisc
    return v_conv * mixing_length


@pytest.mark.physics_invariant
def test_factor_is_the_fourth_power_of_the_length_ratio():
    """Top slope 0.5 gives 0.5^4 near the surface, 1 near the CMB, and 1 at boundaries."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    q = viscous_mixing_length_factor(r, r[0], r[-1], ml, 0.5, 1.0)
    assert q[-2] == pytest.approx(0.0625, rel=1e-12)
    assert q[1] == pytest.approx(1.0, rel=1e-12)
    assert q[0] == pytest.approx(1.0, rel=0.0) and q[-1] == pytest.approx(1.0, rel=0.0)
    np.testing.assert_array_equal(
        viscous_mixing_length_factor(r, r[0], r[-1], ml, 1.0, 1.0), 1.0
    )


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_theta_dependent_top_slope_hand_values_and_limits():
    """Verify s_top(theta) = s_top_inf + (1 - s_top_inf) exp(-c_theta max(theta, 0))."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    s_inf = 0.22
    c_th = 0.15

    # 1. Isoviscous limit (theta -> 0): s_top(0) = 1.0 exactly, recovering Abe length.
    assert compute_mlt_top_slope(0.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        1.0, rel=1e-15
    )
    q_zero = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=0.0, c_theta=c_th
    )
    np.testing.assert_allclose(q_zero, 1.0, rtol=1e-15)

    # 2. Negative theta (inverted temperature difference): max(theta, 0) = 0 clamps s_top at 1.0.
    assert compute_mlt_top_slope(-5.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        1.0, rel=1e-15
    )
    q_neg = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=-5.0, c_theta=c_th
    )
    np.testing.assert_allclose(q_neg, 1.0, rtol=1e-15)

    # 3. Hand-computed intermediate value at theta = 15:
    # s_top(15) = 0.22 + 0.78 * exp(-0.15 * 15) = 0.22 + 0.78 * exp(-2.25) = 0.3022113951582542.
    expected_s15 = 0.22 + 0.78 * math.exp(-2.25)
    expected_q15 = expected_s15**4
    assert compute_mlt_top_slope(15.0) == pytest.approx(expected_s15, rel=1e-14)
    assert compute_mlt_top_slope(15.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        expected_s15, rel=1e-14
    )
    q_15 = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=15.0, c_theta=c_th
    )
    assert q_15[TOP] == pytest.approx(expected_q15, rel=1e-14)
    assert q_15[BOTTOM] == pytest.approx(1.0, rel=1e-14)

    # 4. Stagnant-lid asymptotic limit (theta >> 8, e.g. theta = 250): s_top -> s_top_inf = 0.22.
    assert compute_mlt_top_slope(250.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        s_inf, rel=1e-13
    )
    q_inf = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=250.0, c_theta=c_th
    )
    assert q_inf[TOP] == pytest.approx(s_inf**4, rel=1e-13)
    assert q_inf[BOTTOM] == pytest.approx(1.0, rel=1e-14)


@pytest.mark.physics_invariant
def test_viscous_mixing_length_factor_theta_dispatch_and_limits(monkeypatch):
    """Verify viscous_mixing_length_factor evaluates compute_mlt_top_slope when theta is given."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    s_inf = 0.22
    c_th = 0.15

    # In the isoviscous limit (theta = 0), s_top(0) = 1.0, recovering the Abe mixing length factor.
    q_zero = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=0.0, c_theta=c_th
    )
    assert q_zero[TOP] == pytest.approx(1.0, rel=1e-14)

    # When theta is None, s_top defaults to s_top_inf, yielding s_top_inf^4 at the surface.
    q_none = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=None
    )
    assert q_none[TOP] == pytest.approx(s_inf**4, rel=1e-14)

    # At finite theta = 15.0, factor matches analytical compute_mlt_top_slope^4.
    expected_s15 = compute_mlt_top_slope(15.0, s_top_inf=s_inf, c_theta=c_th)
    q_15 = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=15.0, c_theta=c_th
    )
    assert q_15[TOP] == pytest.approx(expected_s15**4, rel=1e-14)

    # Monkeypatching compute_mlt_top_slope proves viscous_mixing_length_factor delegates to it.
    monkeypatch.setattr('aragog.rheology.compute_mlt_top_slope', lambda *a, **kw: 0.5)
    q_patched = viscous_mixing_length_factor(r, r[0], r[-1], ml, theta=15.0)
    assert q_patched[TOP] == pytest.approx(0.5**4, rel=1e-14)


@pytest.mark.physics_invariant
def test_viscous_mixing_length_factor_bottom_slope_calibration():
    """Verify viscous_mixing_length_factor with non-unit bottom_slope scales the CMB boundary layer."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    # At the CMB boundary node (node 0), mixing_length is zero; factor is 1.0.
    # At node 1 (near-CMB), radii - r_in governs l_v; with bottom_slope=0.5, l_v / l = 0.5.
    q_05 = viscous_mixing_length_factor(r, r[0], r[-1], ml, bottom_slope=0.5)
    assert q_05[0] == 1.0
    assert q_05[1] == pytest.approx(0.5**4, rel=1e-12)
    assert q_05[BOTTOM] == pytest.approx(0.5**4, rel=1e-12)


@pytest.mark.physics_invariant
def test_branch_switch_stays_continuous_in_kappa():
    """Sweep viscosity over 20 decades at the near-surface node for constant and theta slopes.

    Scaling the switch Reynolds number by q keeps the transition at kappa_v = kappa_i,
    so ln(kappa_h) never steps by more than the viscous slope (ln 10 * 0.02 = 0.046).
    """
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    lvs = np.arange(-2.0, 18.0, 0.02)

    cases = [
        (viscous_mixing_length_factor(r, r[0], r[-1], ml, s_top_inf=0.5), 0.5**4),
        (viscous_mixing_length_factor(r, r[0], r[-1], ml, s_top_inf=0.22), 0.22**4),
        (
            viscous_mixing_length_factor(
                r, r[0], r[-1], ml, s_top_inf=0.22, theta=15.0, c_theta=0.15
            ),
            (0.22 + 0.78 * math.exp(-2.25)) ** 4,
        ),
    ]
    for q_vec, q_expected in cases:
        k = np.array([_mlt_kappa_h(lv, r, ml, q_vec)[TOP] for lv in lvs])
        step = np.abs(np.diff(np.log(k)))
        assert step.max() < 0.05
        assert np.all(k > 0.0)
        k_ref = [_mlt_kappa_h(lv, r, ml, 1.0)[TOP] for lv in (lvs[0], lvs[-1])]
        assert k[0] / k_ref[0] == pytest.approx(1.0, rel=1e-10)
        assert k[-1] / k_ref[1] == pytest.approx(q_expected, rel=1e-10)


@pytest.mark.physics_invariant
def test_numpy_and_jax_theta_top_slope_parity():
    """Verify NumPy and JAX evaluate identical s_top(theta) and mixing-length factors."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    r_j = jnp.asarray(r)
    ml_j = jnp.asarray(ml)

    for th in (-3.0, 0.0, 8.0, 15.0, 40.0):
        s_np = compute_mlt_top_slope(th, s_top_inf=0.22, c_theta=0.15, xp=np)
        s_jx = float(
            compute_mlt_top_slope(jnp.asarray(th), s_top_inf=0.22, c_theta=0.15, xp=jnp)
        )
        assert s_jx == pytest.approx(s_np, rel=1e-14)

        q_np = viscous_mixing_length_factor(
            r, r[0], r[-1], ml, s_top_inf=0.22, bottom_slope=1.0, xp=np, theta=th, c_theta=0.15
        )
        q_jx = np.asarray(
            viscous_mixing_length_factor(
                r_j,
                r_j[0],
                r_j[-1],
                ml_j,
                s_top_inf=0.22,
                bottom_slope=1.0,
                xp=jnp,
                theta=jnp.asarray(th),
                c_theta=0.15,
            )
        )
        np.testing.assert_allclose(q_jx, q_np, rtol=1e-14, atol=1e-30)


def test_solid_rheology_params_top_slope_defaults_and_bounds():
    """Verify SolidRheologyParams defaults and bounds validation for s_top_inf and c_theta."""
    params = SolidRheologyParams()
    assert not hasattr(params, 'mlt_top_slope')
    assert params.mlt_bottom_slope == pytest.approx(1.0, rel=1e-15)
    assert params.s_top_inf == pytest.approx(0.22, rel=1e-15)
    assert params.c_theta == pytest.approx(0.15, rel=1e-15)

    try:
        custom = SolidRheologyParams(s_top_inf=0.18, c_theta=0.0)
    except ValueError:
        pytest.fail('c_theta = 0.0 must be accepted')
    assert custom.s_top_inf == pytest.approx(0.18, rel=1e-15)
    assert custom.c_theta == pytest.approx(0.0, abs=1e-15)

    with pytest.raises(ValueError, match='s_top_inf must be in'):
        SolidRheologyParams(s_top_inf=0.0)
    with pytest.raises(ValueError, match='s_top_inf must be in'):
        SolidRheologyParams(s_top_inf=-0.1)
    with pytest.raises(ValueError, match='s_top_inf must be in'):
        SolidRheologyParams(s_top_inf=1.1)
    with pytest.raises(ValueError, match='mlt_bottom_slope must be in'):
        SolidRheologyParams(mlt_bottom_slope=0.0)
    with pytest.raises(ValueError, match='mlt_bottom_slope must be in'):
        SolidRheologyParams(mlt_bottom_slope=1.1)
    with pytest.raises(ValueError, match='c_theta must be non-negative'):
        SolidRheologyParams(c_theta=-0.01)


def _korenaga_ti_fixed_point(
    t_stag: np.ndarray,
    z_over_d: np.ndarray,
    t_surf: float,
    dth: float,
    *,
    max_iter: int = 100,
    tol_k: float = 1.0e-9,
) -> float:
    """Evaluate the Korenaga (2009, eq. 20) iterative sub-lid interior temperature T_i."""
    order = np.argsort(z_over_d)
    z_s, t_s_arr = z_over_d[order], t_stag[order]
    ti = float(t_s_arr.max())
    for _ in range(max_iter):
        dl = (ti - t_surf) / dth
        sub_z = z_s[z_s > dl]
        if sub_z.size == 0:
            raise RuntimeError(
                f'_korenaga_ti_fixed_point failed: conductive lid depth dl={dl} exceeds domain'
            )
        zz = np.concatenate([[dl], sub_z])
        tt = np.interp(zz, z_s, t_s_arr)
        ti_new = float(np.trapezoid(tt, zz) / (zz[-1] - dl))
        if abs(ti_new - ti) < tol_k:
            return ti_new
        ti = ti_new
    raise RuntimeError(
        f'_korenaga_ti_fixed_point did not converge within {tol_k} K after {max_iter} iterations'
    )


def test_korenaga_ti_fixed_point_raises_on_non_convergence() -> None:
    """Verify _korenaga_ti_fixed_point raises RuntimeError when iteration fails to converge."""
    z_grid = np.linspace(0.01, 0.99, 50)
    t_prof = 273.0 + 1500.0 * z_grid * (1.0 - 0.5 * z_grid)
    with pytest.raises(RuntimeError, match='did not converge'):
        _korenaga_ti_fixed_point(t_prof, z_grid, 273.0, 1500.0, max_iter=2, tol_k=1.0e-15)
    with pytest.raises(RuntimeError, match='exceeds domain'):
        _korenaga_ti_fixed_point(t_prof, z_grid, 273.0, 100.0)


def _ols(x_cols: list[np.ndarray], y_vec: np.ndarray) -> dict[str, float]:
    """Fit ``y = c_0 + sum_j c_j x_j`` by ordinary least squares with standard errors."""
    n = len(y_vec)
    x_mat = np.column_stack([np.ones(n), *x_cols])
    coeffs, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    resid = y_vec - x_mat @ coeffs
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    dof = max(n - x_mat.shape[1], 1)
    cov = (ss_res / dof) * np.linalg.inv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(cov))
    if len(x_cols) == 2:
        ln_a, alpha, beta = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        return {
            'a': float(np.exp(ln_a)),
            'ln_a': ln_a,
            'alpha': alpha,
            'beta': beta,
            'std_err_ln_a': float(se[0]),
            'std_err_alpha': float(se[1]),
            'std_err_beta': float(se[2]),
            'r_squared': r2,
        }
    return {
        'slope': float(coeffs[1]),
        'std_err_slope': float(se[1]),
        'intercept_log10': float(coeffs[0]),
        'r_squared': r2,
    }


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a6_2_stagnant_lid_scaling_exponents(validation_plot, tmp_path) -> None:
    """Compare 12-case internal-heating stagnant-lid scaling exponents against theory.

    Across the 12-case planar internal-heating sweep (design Ra_i in [1e6, 1e9] at
    theta = 15 and design theta in [10, 25] at Ra_i = 1e7) at (s_top_inf, c_theta) =
    (0.25, 0.25), the least-squares fit of Nu = a * theta**(-alpha) * Ra_i**beta on
    design parameters gives beta = 0.2820 +/- 0.0134 and alpha = 1.0371 +/- 0.0973
    (a = 0.6411, R^2 = 0.9849; 1-D sweep slopes beta_sweep = 0.2785 +/- 0.0172 and
    alpha_sweep = 1.0109 +/- 0.0592), and the fit on realized steady-state parameters
    (Ra_i_realized_kor, theta_realized_kor) gives beta = 0.1129 +/- 0.0286 and
    alpha = -0.0787 +/- 0.4406 (a = 0.6932, R^2 = 0.7744). Both fits fall below the
    Solomatov & Moresi (2000) internal-heating acceptance window beta in [0.294, 0.350]
    and alpha in [1.294, 1.350]:
    1. Two low-vigor cases sit at the discrete N = 100 staggered-grid pure-conduction
       floor Nu = 2.2472 (analytic Korenaga planar floor 1 + sqrt(6)/2 = 2.2247, peak Nu = 2.0):
       case_01 (design Ra_i = 1e6, realized Ra_i = 244.3, theta = 15, Nu = 2.2500) and
       case_12 (design Ra_i = 1e7, realized Ra_i = 227.9, theta = 25, Nu = 2.2509).
    2. Even on the 7-case convecting subset (Nu_est >= 2.8, a post-hoc diagnostic chosen
       after the result), where design-parameter beta = 0.3114 +/- 0.0026 enters
       [0.294, 0.350], the Frank-Kamenetskii exponent is alpha = 1.1710 +/- 0.0288
       (1-D sweep 1.1563 +/- 0.0119 on 3 points, 1 degree of freedom), below
       [1.294, 1.350] because s_top(theta) = 0.25 + 0.75 * exp(-0.25 * theta) converges
       to the constant asymptote s_top_inf = 0.25 for theta >= 15.

    This test pins the 12-case and 7-case design and realized-state fit parameters,
    uncertainties, discrete conduction floor, per-case Nu values, and live EntropySolver
    steady-state RHS evaluations and residual bounds (|dS/dt| <= 1e-9 J/(kg K yr)) across
    five snapshots (case_01, case_03, case_06, case_07, case_12).
    """
    import json
    from pathlib import Path

    from aragog.parser import Parameters
    from aragog.solver.entropy_solver import EntropySolver

    ref_path = Path(__file__).parent / 'reference' / 'ssc_wp6_a6_2_benchmarks.json'
    data = json.loads(ref_path.read_text(encoding='utf-8'))
    win = data['metadata']['acceptance_window']
    rel_pin = float(win['round_off_spread_rel_tol'])
    assert rel_pin == pytest.approx(1.0e-7, rel=1.0e-12)
    assert float(win['beta_min']) == pytest.approx(0.294, rel=1.0e-12)
    assert float(win['beta_max']) == pytest.approx(0.350, rel=1.0e-12)
    assert float(win['alpha_min']) == pytest.approx(1.294, rel=1.0e-12)
    assert float(win['alpha_max']) == pytest.approx(1.350, rel=1.0e-12)

    cal = data['metadata']['rheology_calibration']
    assert bool(cal['enabled']) is True
    assert float(cal['s_top_inf']) == pytest.approx(0.25, rel=1.0e-12)
    assert float(cal['c_theta']) == pytest.approx(0.25, rel=1.0e-12)

    fp = data['metadata']['fixed_parameters']
    steady_rhs_bound = float(fp['steady_residual_bound_J_kg_K_yr'])
    assert steady_rhs_bound == pytest.approx(1.0e-9, rel=1.0e-12)

    # Discrete N = 100 staggered-grid pure-conduction floor T(z)/dTH = z*(1 - z/2)
    z_stag_99 = ((np.arange(99, dtype=float) + 0.5) / 99.0)[::-1]
    t_cond_norm = z_stag_99 * (1.0 - 0.5 * z_stag_99)
    ti_cond_norm = _korenaga_ti_fixed_point(t_cond_norm, z_stag_99, 0.0, 1.0, tol_k=1.0e-12)
    nu_cond_floor_planar_n100 = 1.0 / ti_cond_norm
    assert float(fp['conduction_floor_Nu_korenaga_analytic_planar']) == pytest.approx(
        1.0 + 0.5 * np.sqrt(6.0), rel=1.0e-12
    )
    assert nu_cond_floor_planar_n100 == pytest.approx(2.227638239622486, rel=1.0e-11)
    assert nu_cond_floor_planar_n100 == pytest.approx(
        float(fp['conduction_floor_Nu_korenaga_planar_N100']), rel=1.0e-11
    )
    assert float(fp['conduction_floor_Nu_korenaga_spherical_N100']) == pytest.approx(
        2.24715117664871, rel=1.0e-11
    )

    # 1. Verify all 12 steady-state cases and the 2 conduction-floor cases
    cases = data['cases_results']
    assert len(cases) == 12
    by_id = {c['case_id']: c for c in cases}
    assert float(by_id['case_01']['Ra_i']) == pytest.approx(1.0e6, rel=1.0e-12)
    assert float(by_id['case_01']['Ra_i_realized_kor']) == pytest.approx(
        244.29905754722245, rel=rel_pin
    )
    assert float(by_id['case_01']['theta']) == pytest.approx(15.0, rel=1.0e-12)
    assert float(by_id['case_01']['Nu_korenaga']) == pytest.approx(
        2.249968281583638, rel=rel_pin
    )
    assert float(by_id['case_12']['Ra_i']) == pytest.approx(1.0e7, rel=1.0e-12)
    assert float(by_id['case_12']['Ra_i_realized_kor']) == pytest.approx(
        227.89157411179806, rel=rel_pin
    )
    assert float(by_id['case_12']['theta']) == pytest.approx(25.0, rel=1.0e-12)
    assert float(by_id['case_12']['Nu_korenaga']) == pytest.approx(
        2.2508789377583125, rel=rel_pin
    )

    ra_all = np.array([float(c['Ra_i']) for c in cases])
    theta_all = np.array([float(c['theta']) for c in cases])
    ra_real_all = np.array([float(c['Ra_i_realized_kor']) for c in cases])
    theta_real_all = np.array([float(c['theta_realized_kor']) for c in cases])
    nu_all = np.array([float(c['Nu_korenaga']) for c in cases])
    nu_est_all = np.array([float(c['Nu_est']) for c in cases])
    sweeps_all = [c['sweep'] for c in cases]

    # 2. Recompute and verify the primary 12-case design-parameter and realized-state OLS fits
    fit12_live = _ols([-np.log(theta_all), np.log(ra_all)], np.log(nu_all))
    p12 = data['primary_fit_all_12_nu_korenaga']['multilinear']
    assert fit12_live['beta'] == pytest.approx(0.28199233476514773, rel=rel_pin)
    assert fit12_live['std_err_beta'] == pytest.approx(0.013368393573574713, rel=rel_pin)
    assert fit12_live['alpha'] == pytest.approx(1.0371353132596806, rel=rel_pin)
    assert fit12_live['std_err_alpha'] == pytest.approx(0.0972550223705178, rel=rel_pin)
    assert fit12_live['a'] == pytest.approx(0.6411234486031506, rel=rel_pin)
    assert fit12_live['ln_a'] == pytest.approx(-0.4445332530796477, rel=rel_pin)
    assert fit12_live['std_err_ln_a'] == pytest.approx(0.35754872383129743, rel=rel_pin)
    assert fit12_live['r_squared'] == pytest.approx(0.9848557284223107, rel=rel_pin)
    for k in (
        'a',
        'ln_a',
        'alpha',
        'beta',
        'std_err_ln_a',
        'std_err_alpha',
        'std_err_beta',
        'r_squared',
    ):
        assert fit12_live[k] == pytest.approx(float(p12[k]), rel=rel_pin)

    fit12_real_live = _ols([-np.log(theta_real_all), np.log(ra_real_all)], np.log(nu_all))
    p12_real = data['realized_state_fits']['all_12_nu_korenaga_on_realized_kor']['multilinear']
    assert fit12_real_live['beta'] == pytest.approx(0.11288990833184448, rel=rel_pin)
    assert fit12_real_live['std_err_beta'] == pytest.approx(0.0286205953168069, rel=rel_pin)
    assert fit12_real_live['alpha'] == pytest.approx(-0.07868422903802338, rel=rel_pin)
    assert fit12_real_live['std_err_alpha'] == pytest.approx(0.44064050899447305, rel=rel_pin)
    assert fit12_real_live['a'] == pytest.approx(0.6932068186285084, rel=rel_pin)
    assert fit12_real_live['r_squared'] == pytest.approx(0.774431133990773, rel=rel_pin)
    for k in (
        'a',
        'ln_a',
        'alpha',
        'beta',
        'std_err_ln_a',
        'std_err_alpha',
        'std_err_beta',
        'r_squared',
    ):
        assert fit12_real_live[k] == pytest.approx(float(p12_real[k]), rel=rel_pin)

    mask_ra_12 = np.array([s == 'Ra_i' for s in sweeps_all])
    mask_th_12 = np.array([s == 'theta' for s in sweeps_all])
    b_sw12 = _ols([np.log10(ra_all[mask_ra_12])], np.log10(nu_all[mask_ra_12]))
    a_sw12 = _ols([-np.log10(theta_all[mask_th_12])], np.log10(nu_all[mask_th_12]))
    assert b_sw12['slope'] == pytest.approx(0.27846707421182404, rel=rel_pin)
    assert b_sw12['std_err_slope'] == pytest.approx(0.01723673778237447, rel=rel_pin)
    assert a_sw12['slope'] == pytest.approx(1.0108926163649687, rel=rel_pin)
    assert a_sw12['std_err_slope'] == pytest.approx(0.059156542457968614, rel=rel_pin)

    # 3. Recompute and verify the post-hoc 7-case convecting diagnostic subset (Nu_est >= 2.8)
    mask_conv7 = nu_est_all >= 2.8
    assert int(np.sum(mask_conv7)) == 7
    fit7_live = _ols(
        [-np.log(theta_all[mask_conv7]), np.log(ra_all[mask_conv7])],
        np.log(nu_all[mask_conv7]),
    )
    p7 = data['diagnostic_fits']['convecting_7_cases_nu_korenaga']['multilinear']
    assert fit7_live['beta'] == pytest.approx(0.31141624364043297, rel=rel_pin)
    assert fit7_live['std_err_beta'] == pytest.approx(0.0026400367811558407, rel=rel_pin)
    assert fit7_live['alpha'] == pytest.approx(1.170990332281924, rel=rel_pin)
    assert fit7_live['std_err_alpha'] == pytest.approx(0.028832895797668207, rel=rel_pin)
    assert fit7_live['a'] == pytest.approx(0.5414999050752781, rel=rel_pin)
    assert fit7_live['r_squared'] == pytest.approx(0.9997126951723827, rel=rel_pin)
    for k in ('a', 'alpha', 'beta', 'std_err_alpha', 'std_err_beta', 'r_squared'):
        assert fit7_live[k] == pytest.approx(float(p7[k]), rel=rel_pin)

    b_sw7 = _ols(
        [np.log10(ra_all[mask_ra_12 & mask_conv7])],
        np.log10(nu_all[mask_ra_12 & mask_conv7]),
    )
    a_sw7 = _ols(
        [-np.log10(theta_all[mask_th_12 & mask_conv7])],
        np.log10(nu_all[mask_th_12 & mask_conv7]),
    )
    assert b_sw7['slope'] == pytest.approx(0.3131675274330532, rel=rel_pin)
    assert b_sw7['std_err_slope'] == pytest.approx(0.003756369949778376, rel=rel_pin)
    assert a_sw7['slope'] == pytest.approx(1.1563304546538864, rel=rel_pin)
    assert a_sw7['std_err_slope'] == pytest.approx(0.011894069031624573, rel=rel_pin)

    # 4. Live EntropySolver steady-state RHS and residual bound across all 5 snapshots
    spec_by_id = {c['case_id']: c for c in data['cases_spec']}
    snaps = data['steady_state_snapshots']

    def _build_case_solver(cid: str, s_top_inf_val: float = 0.25) -> EntropySolver:
        cs = spec_by_id[cid]
        f_top_bc = float(snaps[cid]['F_top_W_m2'])
        log10_visc = float(np.log10(float(cs['eta_i_Pa_s'])))
        toml_text = f"""[boundary_conditions]
outer_boundary_condition = 4
outer_boundary_value = {f_top_bc}
inner_boundary_condition = 2
inner_boundary_value = 0.0
emissivity = 1.0
equilibrium_temperature = {fp['T_surf_K']}
core_heat_capacity = 880.0
core_bc = "quasi_steady"

[energy]
conduction = true
convection = true
gravitational_separation = false
mixing = false
radionuclides = true
tidal = false
solver_method = "cvode"
use_jax_jacobian = false
kappah_floor = 0.0

[initial_condition]
initial_condition = 1
surface_temperature = {fp['T_i_nominal_K']}
basal_temperature = {fp['T_i_nominal_K']}

[mesh]
outer_radius = {fp['R_out_m']}
inner_radius = {fp['R_in_m']}
number_of_nodes = {fp['number_of_nodes']}
mixing_length_profile = "nearest_boundary"
core_density = {fp['rho_kg_m3']}
eos_method = 1
surface_density = {fp['rho_kg_m3']}
gravitational_acceleration = {fp['g_m_s2']}
adiabatic_bulk_modulus = 260000000000.0
adams_williamson_beta = 0.0
surface_pressure = 0.0
mass_coordinates = false

[phase_liquid]
density = {fp['rho_kg_m3']}
viscosity = 100.0
heat_capacity = {fp['cp_J_kg_K']}
melt_fraction = 1.0
thermal_conductivity = {fp['k_W_m_K']}
thermal_expansivity = {fp['alpha_1_K']}

[phase_solid]
density = {fp['rho_kg_m3']}
viscosity = {cs['eta_i_Pa_s']}
heat_capacity = {fp['cp_J_kg_K']}
melt_fraction = 0.0
thermal_conductivity = {fp['k_W_m_K']}
thermal_expansivity = {fp['alpha_1_K']}

[phase_solid.rheology]
enabled = true
s_top_inf = {s_top_inf_val}
c_theta = 0.25
mlt_bottom_slope = 1.0
activation_energy = {cs['E_activation_J_per_mol']}
activation_volume = 0.0
arrhenius_t_ref = {fp['T_i_nominal_K']}
yield_stress_c = 1.0e30
yield_stress_mu = 0.0
yield_stress_max = 1.0e30
viscosity_max_log10 = 40.0

[phase_mixed]
latent_heat_of_fusion = 400000.0
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = "solidus.dat"
liquidus = "liquidus.dat"
phase = "mixed"
phase_transition_width = 0.01
grain_size = 0.001
matprop_smooth_width = 0.01
const_properties = true
const_rho = {fp['rho_kg_m3']}
const_Cp = {fp['cp_J_kg_K']}
const_alpha = {fp['alpha_1_K']}
const_cond = {fp['k_W_m_K']}
const_log10visc = {log10_visc}
const_T_ref = {fp['T_i_nominal_K']}
const_S_ref = 3000.0

[solver]
start_time = 0.0
end_time = 4.0e10
atol = 1.0e-6
rtol = 1.0e-6
tsurf_poststep_change = 30.0
cvode_output_points = 65

[radionuclide_heating]
name = "const"
t0_years = 0.0
abundance = 1.0
concentration = 1000000.0
heat_production = {cs['H_W_per_kg']}
half_life_years = 1.0e30
"""
        cfg_file = tmp_path / f'{cid}_{s_top_inf_val:.3f}.toml'
        cfg_file.write_text(toml_text, encoding='utf-8')
        p_case = Parameters.from_file(cfg_file)
        sol = EntropySolver(p_case, entropy_eos=None)
        sol.initialize()
        return sol

    for cid in ('case_01', 'case_03', 'case_06', 'case_07', 'case_12'):
        snap = snaps[cid]
        sol = _build_case_solver(cid, s_top_inf_val=0.25)
        s_vec = np.asarray(snap['S_final'], dtype=float)
        rhs_live = np.asarray(sol._dSdt_single(float(snap['t_end_yr']), s_vec), dtype=float)
        np.testing.assert_allclose(
            rhs_live, np.asarray(snap['rhs_final'], dtype=float), rtol=rel_pin, atol=1.0e-18
        )
        max_abs_rhs = float(np.max(np.abs(rhs_live)))
        assert max_abs_rhs <= steady_rhs_bound
        assert max_abs_rhs == pytest.approx(
            float(snap['max_abs_rhs_J_kg_K_yr']), rel=1.0e-5, abs=1.0e-18
        )
        t_stag_live = float(fp['T_i_nominal_K']) * np.exp(
            (s_vec - 3000.0) / float(fp['cp_J_kg_K'])
        )
        np.testing.assert_allclose(
            t_stag_live, np.asarray(snap['T_stag_final'], dtype=float), rtol=rel_pin, atol=0.0
        )
        r_b = np.asarray(sol.evaluator.mesh.basic.radii, dtype=float).ravel()
        r_s = np.asarray(sol.evaluator.mesh.staggered.radii, dtype=float).ravel()
        d_m = float(fp['D_m'])
        dr_surf_half = 0.5 * float(r_b[-1] - r_b[-2])
        f_top_halfcell = (
            float(fp['k_W_m_K'])
            * (float(t_stag_live[-1]) - float(snap['T_surf_K']))
            / dr_surf_half
        )
        assert f_top_halfcell == pytest.approx(float(snap['F_top_W_m2']), rel=rel_pin)
        ti_kor_live = _korenaga_ti_fixed_point(
            t_stag_live, (r_b[-1] - r_s) / d_m, float(snap['T_surf_K']), float(snap['dTH_K'])
        )
        nu_kor_live = float(snap['dTH_K']) / (ti_kor_live - float(snap['T_surf_K']))
        assert ti_kor_live == pytest.approx(float(snap['T_i_korenaga_K']), rel=rel_pin)
        assert nu_kor_live == pytest.approx(float(snap['Nu_korenaga']), rel=rel_pin)
        assert float(np.max(sol.state._jconv)) == pytest.approx(
            float(snap['jconv_max_W_m2']), rel=rel_pin
        )

    assert float(snaps['case_01']['jconv_max_W_m2']) == pytest.approx(
        0.0062746085758870985, rel=rel_pin
    )
    assert float(snaps['case_03']['jconv_max_W_m2']) == pytest.approx(
        0.822944920830173, rel=rel_pin
    )
    assert float(snaps['case_06']['jconv_max_W_m2']) == pytest.approx(
        7.06894591400015, rel=rel_pin
    )
    assert float(snaps['case_07']['jconv_max_W_m2']) == pytest.approx(
        1.9547419248964315, rel=rel_pin
    )
    assert float(snaps['case_12']['jconv_max_W_m2']) == pytest.approx(
        0.012250557884946865, rel=rel_pin
    )

    # 5. Canaries: asserting pre-run acceptance bounds or perturbing inputs fails
    with pytest.raises(AssertionError):
        assert float(win['beta_min']) <= fit12_live['beta'] <= float(win['beta_max'])
    with pytest.raises(AssertionError):
        assert float(win['alpha_min']) <= fit12_live['alpha'] <= float(win['alpha_max'])
    with pytest.raises(AssertionError):
        assert float(win['beta_min']) <= fit12_real_live['beta'] <= float(win['beta_max'])
    with pytest.raises(AssertionError):
        assert float(win['alpha_min']) <= fit7_live['alpha'] <= float(win['alpha_max'])

    nu_perturbed = nu_all.copy()
    nu_perturbed[0] *= 1.01
    fit_perturbed = _ols([-np.log(theta_all), np.log(ra_all)], np.log(nu_perturbed))
    with pytest.raises(AssertionError):
        assert fit_perturbed['beta'] == pytest.approx(fit12_live['beta'], rel=rel_pin)

    sol_c03_pert = _build_case_solver('case_03', s_top_inf_val=0.26)
    s_c03 = np.asarray(snaps['case_03']['S_final'], dtype=float)
    rhs_c03_pert = np.asarray(
        sol_c03_pert._dSdt_single(float(snaps['case_03']['t_end_yr']), s_c03), dtype=float
    )
    with pytest.raises(AssertionError):
        assert float(np.max(sol_c03_pert.state._jconv)) == pytest.approx(
            float(snaps['case_03']['jconv_max_W_m2']), rel=rel_pin
        )
    with pytest.raises(AssertionError):
        assert float(np.max(np.abs(rhs_c03_pert))) <= steady_rhs_bound
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            rhs_c03_pert,
            np.asarray(snaps['case_03']['rhs_final'], dtype=float),
            rtol=rel_pin,
            atol=1.0e-18,
        )

    def figure():
        import matplotlib.pyplot as plt

        ra_sw = [c for c in cases if c['sweep'] == 'Ra_i']
        th_sw = [c for c in cases if c['sweep'] == 'theta']

        x_ra = np.array([np.log10(float(c['Ra_i'])) for c in ra_sw])
        y_ra = np.array(
            [
                np.log10(float(c['Nu_korenaga']) * (float(c['theta']) ** (4.0 / 3.0)))
                for c in ra_sw
            ]
        )
        x_th = np.array([np.log10(float(c['theta'])) for c in th_sw])
        y_th = np.array(
            [
                np.log10(float(c['Nu_korenaga']) * (float(c['Ra_i']) ** (-1.0 / 3.0)))
                for c in th_sw
            ]
        )

        x_ra_grid = np.array([5.85, 9.15])
        y_ra_fit12 = (
            np.log10(fit12_live['a'])
            + (4.0 / 3.0 - fit12_live['alpha']) * np.log10(15.0)
            + fit12_live['beta'] * x_ra_grid
        )
        y_ra_fit7 = (
            np.log10(fit7_live['a'])
            + (4.0 / 3.0 - fit7_live['alpha']) * np.log10(15.0)
            + fit7_live['beta'] * x_ra_grid
        )
        y_ra_th = np.log10(0.53) + (1.0 / 3.0) * x_ra_grid

        x_th_grid = np.array([0.97, 1.43])
        y_th_fit12 = (
            np.log10(fit12_live['a'])
            + (fit12_live['beta'] - 1.0 / 3.0) * 7.0
            - fit12_live['alpha'] * x_th_grid
        )
        y_th_fit7 = (
            np.log10(fit7_live['a'])
            + (fit7_live['beta'] - 1.0 / 3.0) * 7.0
            - fit7_live['alpha'] * x_th_grid
        )
        y_th_th = np.log10(0.53) - (4.0 / 3.0) * x_th_grid

        plt.rcParams.update(
            {
                'axes.labelsize': 9,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 6.5,
                'legend.title_fontsize': 6.5,
                'axes.grid': False,
            }
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
        cyc = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ax1.plot(x_ra, y_ra, 'o', color=cyc[0], markersize=4.5, label='1D cases')
        ax1.plot(x_ra_grid, y_ra_fit12, '-', color=cyc[1], label='12-case fit')
        ax1.plot(x_ra_grid, y_ra_fit7, '-.', color=cyc[4], label='7-case subset*')
        ax1.plot(x_ra_grid, y_ra_th, '--', color=cyc[2], label='SM00 theory')
        ax1.set(
            xlabel='(a) log10(Ra_i)',
            ylabel='log10(Nu theta^(4/3))',
            xlim=(5.8, 9.2),
            ylim=(1.8, 3.0),
            xticks=[6, 7, 8, 9],
            yticks=[2.0, 2.5, 3.0],
        )
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2)

        ax2.plot(x_th, y_th, 'o', color=cyc[0], markersize=4.5, label='1D cases')
        ax2.plot(x_th_grid, y_th_fit12, '-', color=cyc[1], label='12-case fit')
        ax2.plot(x_th_grid, y_th_fit7, '-.', color=cyc[4], label='7-case subset*')
        ax2.plot(x_th_grid, y_th_th, '--', color=cyc[2], label='SM00 theory')
        ax2.set(
            xlabel='(b) log10(theta)',
            ylabel='log10(Nu Ra_i^(-1/3))',
            xlim=(0.96, 1.44),
            ylim=(-2.25, -1.45),
            xticks=[1.0, 1.2, 1.4],
            yticks=[-2.2, -1.8, -1.5],
        )
        ax2.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.24),
            ncol=2,
            title='* diagnostic chosen after the result',
        )

        fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.38, wspace=0.34)
        return fig

    validation_plot(figure, 'ssc_a6_2_stagnant_lid_scalings')
