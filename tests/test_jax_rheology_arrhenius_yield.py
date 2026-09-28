"""Unit tests for Arrhenius viscosity, yield stress, and CVODE JAX Jacobian.

Verifies:
1. Arrhenius viscosity formula and reference state consistency.
2. Plastic yield stress formulation and smooth regularization.
3. Effective viscosity harmonic average and asymptotic limits.
4. Smooth C^infty differentiability without gradient kinks.
5. CVODE JAX analytic Jacobian autodiff through the full rheology pipeline
   and agreement with finite differences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aragog.jax.phase import (
    PhaseParams,
    compute_arrhenius_viscosity,
    compute_effective_viscosity,
    compute_yield_stress,
)
from aragog.solver.cvode_jax import (
    build_jax_rhs_and_jacobian,
)

jax.config.update('jax_enable_x64', True)

pytestmark = pytest.mark.unit


def test_cvode_jax_no_reexports():
    """Verify cvode_jax does not re-export rheology functions."""
    import aragog.solver.cvode_jax as cvode_mod

    assert not hasattr(cvode_mod, 'compute_arrhenius_viscosity')
    assert not hasattr(cvode_mod, 'compute_yield_stress')
    assert not hasattr(cvode_mod, 'compute_effective_viscosity')
    assert not hasattr(cvode_mod, 'eta_diff')
    assert not hasattr(cvode_mod, 'eta_eff')


def test_arrhenius_reference_state():
    """At T=1600 K and P=0 Pa, eta_diff must equal viscosity_solid exactly."""
    visc_solid = 1.0e21
    T = jnp.asarray(1600.0)
    P = jnp.asarray(0.0)
    E_a = 3.0e5
    V_a = 1.5e-5

    eta = compute_arrhenius_viscosity(T, P, visc_solid, E_a, V_a)
    np.testing.assert_allclose(float(eta), visc_solid, rtol=1e-12)


def test_arrhenius_derivatives():
    """Viscosity must decrease with T and increase with P."""
    visc_solid = 1.0e21
    E_a = 3.0e5
    V_a = 1.5e-5

    def eta_fn(T_val, P_val):
        return compute_arrhenius_viscosity(T_val, P_val, visc_solid, E_a, V_a)

    g_T, g_P = jax.grad(eta_fn, argnums=(0, 1))(1500.0, 1.0e9)
    assert float(g_T) < 0.0  # warmer -> lower viscosity
    assert float(g_P) > 0.0  # higher pressure -> higher viscosity
    assert np.isfinite(float(g_T))
    assert np.isfinite(float(g_P))


def test_yield_stress_values_and_smoothness():
    """tau_y = yield_stress_c + yield_stress_mu * P, smooth and positive."""
    c = 1.0e8
    mu = 0.1

    # In normal positive regime, matches linear formula to float precision
    P_pos = jnp.asarray(1.0e9)
    tau_pos = compute_yield_stress(P_pos, c, mu)
    np.testing.assert_allclose(float(tau_pos), c + mu * 1.0e9, rtol=1e-12)

    # At zero cohesion and zero pressure, softplus floor keeps it strictly positive
    tau_zero = compute_yield_stress(jnp.asarray(0.0), 0.0, 0.0)
    assert float(tau_zero) > 0.0
    assert np.isfinite(float(tau_zero))

    # Gradient with respect to P is smooth
    dtau_dP = jax.grad(compute_yield_stress, argnums=0)(P_pos, c, mu)
    np.testing.assert_allclose(float(dtau_dP), mu, rtol=1e-8)


def test_effective_viscosity_limits():
    """Effective viscosity satisfies diffusion and plastic yielding limits."""
    eta_diff = jnp.asarray(1.0e22)
    tau_y = jnp.asarray(1.0e8)

    # Zero strain rate -> pure diffusion creep
    eta_zero_strain = compute_effective_viscosity(eta_diff, tau_y, strain_rate=0.0)
    np.testing.assert_allclose(float(eta_zero_strain), float(eta_diff), rtol=1e-12)

    # High strain rate / low yield stress -> yielding limit eta_plast = tau_y / (2 * strain_rate)
    strain_rate_high = 1.0e-12
    eta_plast = float(tau_y) / (2.0 * strain_rate_high)  # 5e19
    eta_eff = compute_effective_viscosity(eta_diff, tau_y, strain_rate=strain_rate_high)
    # Harmonic mean of 1e22 and 5e19 is within 1% of 5e19
    np.testing.assert_allclose(float(eta_eff), eta_plast, rtol=0.01)


def test_effective_viscosity_smooth_gradients():
    """Gradients through eta_eff must be smooth across the yielding transition."""
    visc_solid = 1.0e21
    E_a = 3.0e5
    V_a = 1.5e-5
    c = 1.0e8
    mu = 0.1
    strain_rate = 1.0e-15

    def eta_total(T_val, P_val):
        eta_d = compute_arrhenius_viscosity(T_val, P_val, visc_solid, E_a, V_a)
        tau = compute_yield_stress(P_val, c, mu)
        return compute_effective_viscosity(eta_d, tau, strain_rate)

    grad_fn = jax.jit(jax.grad(eta_total, argnums=(0, 1)))

    # Scan across temperatures from cold lid (yielding) to warm interior (diffusion)
    for T in [500.0, 1000.0, 1400.0, 1600.0, 1800.0, 2500.0]:
        val = eta_total(T, 1.0e9)
        g_T, g_P = grad_fn(T, 1.0e9)
        assert np.isfinite(float(val))
        assert np.isfinite(float(g_T))
        assert np.isfinite(float(g_P))
        assert float(g_T) <= 0.0  # monotonic temperature thinning


def test_cvode_jax_analytic_jacobian_finite_differences():
    """CVODE JAX analytic Jacobian must differentiate through the rheology
    pipeline and match finite differences.
    """
    from aragog.jax.nondim import NonDimScales
    from aragog.jax.phase import MeshArrays

    N = 6
    R_INNER = 3.480e6
    R_OUTER = 6.371e6
    r_stag = np.linspace(R_INNER, R_OUTER, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_INNER
    r_basic[-1] = R_OUTER
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    area = 4.0 * np.pi * r_basic**2
    volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.maximum(np.minimum(r_basic - R_INNER, R_OUTER - r_basic), 1.0)
    d_dr = np.zeros((N + 1, N))
    for i in range(1, N):
        d_dr[i, i - 1] = -1.0 / dr[i - 1]
        d_dr[i, i] = 1.0 / dr[i - 1]
    d_dr[0, :] = d_dr[1, :]
    d_dr[-1, :] = d_dr[-2, :]
    q_mat = np.zeros((N + 1, N))
    q_mat[0, 0] = 1.0
    q_mat[-1, -1] = 1.0
    for i in range(1, N):
        q_mat[i, i - 1] = 0.5
        q_mat[i, i] = 0.5
    P_stag = np.linspace(135e9, 1e5, N)
    P_basic = q_mat @ P_stag
    mesh = MeshArrays(
        d_dr_matrix=jnp.asarray(d_dr),
        quantity_matrix=jnp.asarray(q_mat),
        area=jnp.asarray(area),
        volume=jnp.asarray(volume),
        radii_basic=jnp.asarray(r_basic),
        radii_stag=jnp.asarray(r_stag),
        mixing_length=jnp.asarray(ml),
        mixing_length_sq=jnp.asarray(ml**2),
        mixing_length_cu=jnp.asarray(ml**3),
        P_stag=jnp.asarray(P_stag),
        P_basic=jnp.asarray(P_basic),
        gravity=jnp.full(N + 1, 10.0),
    )

    from aragog.jax.solver import BoundaryParams

    bc = BoundaryParams(
        outer_bc_type=4,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=255.0,
        inner_bc_type=2,
        inner_bc_value=0.0,
        core_density=10500.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
    )

    # PhaseParams with active Arrhenius and Yield Stress parameters
    params = PhaseParams(
        viscosity_solid=1.0e21,
        activation_energy=3.0e5,
        activation_volume=1.5e-5,
        yield_stress_c=1.0e8,
        yield_stress_mu=0.1,
        convection=True,
        conduction=True,
    )

    from tests.test_cvode_jax_factory_invocation import EOS_DIR

    if not EOS_DIR.exists():
        pytest.skip(f'SPIDER EOS tables not found at {EOS_DIR}')

    from aragog.jax.eos import EntropyEOS_JAX

    eos_jax = EntropyEOS_JAX(EOS_DIR)

    state_scale = np.full(N, 3.0e3)
    scales = NonDimScales(state_scale=state_scale, t_ref=1.0)
    heating = np.zeros(N)

    rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh,
        boundary_params=bc,
        heating_array=heating,
        scales=scales,
        core_bc_mode='quasi_steady',
    )

    y_nd = np.full(N, 3050.0 / 3.0e3)
    ydot_nd = np.zeros(N)
    rc = rhs_fn(0.0, y_nd, ydot_nd)
    assert rc == 0
    assert np.all(np.isfinite(ydot_nd))

    J_analytic = np.zeros((N, N))
    rc_jac = jac_fn(0.0, y_nd, None, J_analytic)
    assert rc_jac == 0
    assert np.all(np.isfinite(J_analytic))
    # Analytic Jacobian must be non-trivial
    assert np.any(J_analytic != 0.0)

    # Finite difference check of Jacobian
    eps_fd = 1.0e-7
    J_fd = np.zeros((N, N))
    for j in range(N):
        y_plus = y_nd.copy()
        y_plus[j] += eps_fd
        ydot_plus = np.zeros(N)
        rhs_fn(0.0, y_plus, ydot_plus)

        y_minus = y_nd.copy()
        y_minus[j] -= eps_fd
        ydot_minus = np.zeros(N)
        rhs_fn(0.0, y_minus, ydot_minus)

        J_fd[:, j] = (ydot_plus - ydot_minus) / (2.0 * eps_fd)

    # Verify analytic Jacobian matches finite differences where non-negligible
    mask = np.abs(J_analytic) > 1.0e-5
    if np.any(mask):
        np.testing.assert_allclose(J_analytic[mask], J_fd[mask], rtol=5.0e-3, atol=1.0e-4)
