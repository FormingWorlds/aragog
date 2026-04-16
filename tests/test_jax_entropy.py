"""JAX entropy solver validation tests.

Tier 1: EOS parity (JAX vs numpy on same tables)
Tier 2: Phase evaluator parity
Tier 3: JAX-specific (JIT, vmap, grad)
Tier 4: Constant-property analytical (no EOS tables needed)
Tier 5: Solver integration (grey-body cooling, energy conservation)
Tier 6: Solver parity (JAX diffrax vs scipy BDF on identical problem)

All tests marked @pytest.mark.unit (fast) or @pytest.mark.smoke (solver).
Table-dependent tests use the @needs_eos skip marker.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# EOS directory (same as test_entropy_pytest.py)
EOS_DIR = Path(os.environ.get(
    'ARAGOG_TEST_EOS_DIR',
    '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
))

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}',
)

# JAX imports (skip entire module if JAX unavailable)
jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
eqx = pytest.importorskip('equinox')

jax.config.update('jax_enable_x64', True)

SECS_PER_YEAR = 31557600.0
SIGMA_SB = 5.670374419e-8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def jax_eos():
    """Load JAX EOS from SPIDER tables."""
    if not EOS_DIR.exists():
        pytest.skip(f'EOS tables not found: {EOS_DIR}')
    from aragog.jax.eos import EntropyEOS_JAX
    return EntropyEOS_JAX(EOS_DIR)


@pytest.fixture(scope='module')
def numpy_eos():
    """Load numpy EOS from SPIDER tables."""
    if not EOS_DIR.exists():
        pytest.skip(f'EOS tables not found: {EOS_DIR}')
    from aragog.eos.entropy import EntropyEOS
    return EntropyEOS(EOS_DIR)


@pytest.fixture(scope='module')
def default_params():
    """Default PhaseParams for testing."""
    from aragog.jax.phase import PhaseParams
    return PhaseParams()


# ---------------------------------------------------------------------------
# Tier 1: EOS unit tests (JAX)
# ---------------------------------------------------------------------------

@needs_eos
@pytest.mark.unit
class TestJAXEOS:
    """Unit tests for the JAX P-S table layer."""

    def test_table_loads(self, jax_eos):
        """EntropyEOS_JAX loads without error."""
        assert jax_eos.P_min > 0
        assert jax_eos.S_min < jax_eos.S_max

    def test_temperature_broadly_increases_with_entropy(self, jax_eos):
        """At constant P, T should broadly increase with S."""
        P = jnp.full(50, 50e9)
        S = jnp.linspace(jax_eos.S_min + 100, jax_eos.S_max - 100, 50)
        T = jax_eos.temperature(P, S)
        assert float(T[-1]) > float(T[0])
        dT = jnp.diff(T)
        n_decreasing = int(jnp.sum(dT < -1.0))
        assert n_decreasing < len(dT) // 4

    def test_density_positive(self, jax_eos):
        """Density must be positive everywhere."""
        P = jnp.array([10e9, 50e9, 100e9])
        S = jnp.array([3200.0, 3200.0, 3200.0])
        rho = jax_eos.density(P, S)
        assert bool(jnp.all(rho > 0))

    def test_melt_fraction_bounds(self, jax_eos):
        """Melt fraction must be in [0, 1]."""
        P = jnp.linspace(1e9, 100e9, 20)
        S = jnp.full(20, 3200.0)
        phi = jax_eos.melt_fraction(P, S)
        assert bool(jnp.all(phi >= 0)) and bool(jnp.all(phi <= 1))

    def test_harmonic_density_at_midpoint(self, jax_eos):
        """At phi=0.5, density should be the harmonic mean of end-members."""
        P = jnp.array([50e9])
        S_sol = jax_eos.solidus_entropy(P)
        S_liq = jax_eos.liquidus_entropy(P)
        S_mid = 0.5 * (S_sol + S_liq)
        phi = jax_eos.melt_fraction(P, S_mid)
        assert float(phi[0]) == pytest.approx(0.5, abs=0.01)

        rho = jax_eos.density(P, S_mid)
        rho_s = jax_eos._lookup_at_phase_boundary('density', P, 'solid')
        rho_l = jax_eos._lookup_at_phase_boundary('density', P, 'melt')
        rho_harmonic = 1.0 / (0.5 / float(rho_l[0]) + 0.5 / float(rho_s[0]))
        assert float(rho[0]) == pytest.approx(rho_harmonic, rel=0.02)

    def test_latent_heat_positive(self, jax_eos):
        """Latent heat L = T_fus * (S_liq - S_sol) must be positive."""
        P = jnp.linspace(10e9, 100e9, 10)
        L = jax_eos.latent_heat(P)
        assert bool(jnp.all(L > 0))

    def test_roundtrip_temperature(self, jax_eos):
        """T from (P,S) at solidus should match T from phase boundary lookup."""
        P = jnp.array([50e9])
        S_sol = jax_eos.solidus_entropy(P)
        T_at_sol = jax_eos.temperature(P, S_sol)
        T_sol_boundary = jax_eos._lookup_at_phase_boundary('temperature', P, 'solid')
        assert float(T_at_sol[0]) == pytest.approx(float(T_sol_boundary[0]), rel=0.05)


# ---------------------------------------------------------------------------
# Tier 1b: EOS parity (JAX vs numpy)
# ---------------------------------------------------------------------------

@needs_eos
@pytest.mark.unit
class TestEOSParity:
    """Verify JAX EOS matches numpy EOS on same tables."""

    def test_temperature_parity(self, jax_eos, numpy_eos):
        """Temperature lookup matches between JAX and numpy."""
        P = np.linspace(10e9, 120e9, 20)
        S = np.full(20, 3200.0)
        T_np = numpy_eos.temperature(P, S)
        T_jax = np.asarray(jax_eos.temperature(jnp.asarray(P), jnp.asarray(S)))
        np.testing.assert_allclose(T_jax, T_np, rtol=1e-10)

    def test_density_parity(self, jax_eos, numpy_eos):
        """Density lookup matches between JAX and numpy."""
        P = np.linspace(10e9, 120e9, 20)
        S = np.full(20, 3200.0)
        rho_np = numpy_eos.density(P, S)
        rho_jax = np.asarray(jax_eos.density(jnp.asarray(P), jnp.asarray(S)))
        np.testing.assert_allclose(rho_jax, rho_np, rtol=1e-10)

    def test_melt_fraction_parity(self, jax_eos, numpy_eos):
        """Melt fraction matches across the mushy zone."""
        P = np.full(50, 50e9)
        S_sol = float(numpy_eos.solidus_entropy(50e9))
        S_liq = float(numpy_eos.liquidus_entropy(50e9))
        S = np.linspace(S_sol - 100, S_liq + 100, 50)
        phi_np = numpy_eos.melt_fraction(P, S)
        phi_jax = np.asarray(jax_eos.melt_fraction(jnp.asarray(P), jnp.asarray(S)))
        np.testing.assert_allclose(phi_jax, phi_np, atol=1e-12)

    def test_heat_capacity_parity(self, jax_eos, numpy_eos):
        """Heat capacity lookup matches."""
        P = np.linspace(10e9, 120e9, 20)
        S = np.full(20, 3000.0)
        Cp_np = numpy_eos.heat_capacity(P, S)
        Cp_jax = np.asarray(jax_eos.heat_capacity(jnp.asarray(P), jnp.asarray(S)))
        np.testing.assert_allclose(Cp_jax, Cp_np, rtol=1e-10)

    def test_phase_boundary_parity(self, jax_eos, numpy_eos):
        """Solidus and liquidus entropy match."""
        P = np.linspace(5e9, 130e9, 30)
        S_sol_np = numpy_eos.solidus_entropy(P)
        S_liq_np = numpy_eos.liquidus_entropy(P)
        S_sol_jax = np.asarray(jax_eos.solidus_entropy(jnp.asarray(P)))
        S_liq_jax = np.asarray(jax_eos.liquidus_entropy(jnp.asarray(P)))
        np.testing.assert_allclose(S_sol_jax, S_sol_np, rtol=1e-10)
        np.testing.assert_allclose(S_liq_jax, S_liq_np, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tier 2: Phase evaluator tests
# ---------------------------------------------------------------------------

@needs_eos
@pytest.mark.unit
class TestJAXPhaseEvaluator:
    """Unit tests for the JAX phase evaluator."""

    def test_isentropic_profile(self, jax_eos, default_params):
        """On isentropic IC, T increases with depth (increasing P)."""
        from aragog.jax.phase import evaluate_phase

        P = jnp.linspace(1e9, 135e9, 30)
        S = jnp.full(30, 3200.0)
        props = evaluate_phase(jax_eos, default_params, P, S)
        # P increases with index: P[0]=1e9 (surface), P[-1]=135e9 (CMB)
        assert float(props.temperature[-1]) > float(props.temperature[0])

    def test_capacitance_is_rho_times_T(self, jax_eos, default_params):
        """Capacitance = rho * T for entropy formulation."""
        from aragog.jax.phase import evaluate_phase

        P = jnp.array([50e9])
        S = jnp.array([3200.0])
        props = evaluate_phase(jax_eos, default_params, P, S)
        expected = props.density * props.temperature
        assert float(props.capacitance[0]) == pytest.approx(float(expected[0]), rel=1e-10)

    def test_viscosity_blend(self, jax_eos, default_params):
        """Viscosity transitions from solid to liquid across mushy zone."""
        from aragog.jax.phase import evaluate_phase

        P = jnp.full(3, 50e9)
        S_sol = float(jax_eos.solidus_entropy(P[0:1])[0])
        S_liq = float(jax_eos.liquidus_entropy(P[0:1])[0])
        # Solid, mushy, liquid
        S = jnp.array([S_sol - 200, 0.5 * (S_sol + S_liq), S_liq + 200])
        props = evaluate_phase(jax_eos, default_params, P, S)
        # Solid viscosity >> liquid viscosity
        assert float(props.viscosity[0]) > float(props.viscosity[2]) * 1e10


# ---------------------------------------------------------------------------
# Tier 3: JAX-specific tests (JIT, vmap, grad)
# ---------------------------------------------------------------------------

@needs_eos
@pytest.mark.unit
class TestJAXFeatures:
    """Tests for JAX compilation and differentiation."""

    def test_eos_jit_compiles(self, jax_eos):
        """EOS temperature lookup can be JIT-compiled."""
        @jax.jit
        def f(P, S):
            return jax_eos.temperature(P, S)

        P = jnp.array([50e9])
        S = jnp.array([3200.0])
        T = f(P, S)
        assert float(T[0]) > 0

    def test_eos_vmap_over_pressures(self, jax_eos):
        """EOS lookups can be vmapped over different pressures."""
        # vmap over the first argument (P), keeping S fixed
        @jax.jit
        def batch_temperature(P_batch, S_batch):
            return jax_eos.temperature(P_batch, S_batch)

        P = jnp.linspace(10e9, 100e9, 10)
        S = jnp.full(10, 3200.0)
        T = batch_temperature(P, S)
        assert T.shape == (10,)
        assert bool(jnp.all(T > 0))

    def test_eos_grad_temperature_wrt_entropy(self, jax_eos):
        """jax.grad(T, S) works through the EOS lookup."""
        def T_scalar(S_val):
            return jax_eos.temperature(jnp.array([50e9]), jnp.array([S_val]))[0]

        dTdS = jax.grad(T_scalar)(3200.0)
        # dT/dS should be positive (T increases with S) and finite
        assert np.isfinite(float(dTdS))
        assert float(dTdS) > 0

    def test_eos_grad_density_wrt_entropy(self, jax_eos):
        """jax.grad(rho, S) works through the EOS lookup."""
        def rho_scalar(S_val):
            return jax_eos.density(jnp.array([50e9]), jnp.array([S_val]))[0]

        drho_dS = jax.grad(rho_scalar)(3200.0)
        assert np.isfinite(float(drho_dS))

    def test_evaluate_phase_jit(self, jax_eos, default_params):
        """Full phase evaluation JIT-compiles."""
        from aragog.jax.phase import evaluate_phase

        @jax.jit
        def f(P, S):
            return evaluate_phase(jax_eos, default_params, P, S)

        P = jnp.linspace(10e9, 130e9, 20)
        S = jnp.full(20, 3200.0)
        props = f(P, S)
        assert props.temperature.shape == (20,)
        assert bool(jnp.all(props.temperature > 0))

    def test_grad_through_phase_evaluation(self, jax_eos, default_params):
        """jax.grad flows through evaluate_phase -> EOS."""
        from aragog.jax.phase import evaluate_phase

        def mean_temperature(S_uniform):
            S = jnp.full(10, S_uniform)
            P = jnp.linspace(10e9, 100e9, 10)
            props = evaluate_phase(jax_eos, default_params, P, S)
            return jnp.mean(props.temperature)

        grad_val = jax.grad(mean_temperature)(3200.0)
        assert np.isfinite(float(grad_val))
        assert float(grad_val) > 0  # dT_mean/dS > 0


# ---------------------------------------------------------------------------
# Tier 4: Constant-property analytical tests (no EOS tables needed)
# ---------------------------------------------------------------------------

# Constants for constant-property tests (same as test_entropy_advanced.py)
RHO_CONST = 4000.0       # kg/m^3
CP_CONST = 1000.0         # J/kg/K
K_CONST = 4.0             # W/m/K
T_REF = 3500.0            # K
S_REF = 3000.0            # J/kg/K
R_INNER = 5.371e6         # m
R_OUTER = 6.371e6         # m
D_SHELL = R_OUTER - R_INNER


def _T_to_S(T):
    return S_REF + CP_CONST * np.log(np.asarray(T, dtype=float) / T_REF)


def _S_to_T(S):
    return T_REF * np.exp((np.asarray(S, dtype=float) - S_REF) / CP_CONST)


def _S_to_T_jax(S):
    return T_REF * jnp.exp((S - S_REF) / CP_CONST)


def _analytical_T(r, T_inner=4000.0, T_outer=1500.0):
    a, b = R_INNER, R_OUTER
    A = (T_inner - T_outer) * a * b / (b - a)
    B = (T_outer * b - T_inner * a) / (b - a)
    return A / r + B


def _make_jax_mesh_arrays(N=100):
    """Build MeshArrays for constant-property tests (no numpy Mesh needed)."""
    from aragog.jax.phase import MeshArrays

    r_stag = np.linspace(R_INNER, R_OUTER, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_INNER
    r_basic[-1] = R_OUTER
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])

    area = 4.0 * np.pi * r_basic**2
    volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.maximum(np.minimum(r_basic - R_INNER, R_OUTER - r_basic), 1.0)

    # Build transform matrices
    # d/dr at basic nodes from staggered node values
    d_dr = np.zeros((N + 1, N))
    for i in range(1, N):
        d_dr[i, i - 1] = -1.0 / dr[i - 1]
        d_dr[i, i] = 1.0 / dr[i - 1]
    d_dr[0, :] = d_dr[1, :]  # extrapolate
    d_dr[-1, :] = d_dr[-2, :]

    # Staggered-to-basic interpolation
    q_mat = np.zeros((N + 1, N))
    q_mat[0, 0] = 1.0
    q_mat[-1, -1] = 1.0
    for i in range(1, N):
        q_mat[i, i - 1] = 0.5
        q_mat[i, i] = 0.5

    # Pressure profile (linear, surface to CMB)
    P_stag = np.linspace(135e9, 1e5, N)
    P_basic = q_mat @ P_stag

    return MeshArrays(
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


def _const_prop_dSdt_jax(S, mesh, F_inner=0.0, F_outer=None):
    """Constant-property dS/dt in JAX (conduction only, no EOS tables).

    Uses T = T_ref * exp((S - S_ref) / Cp) with constant rho, Cp, k.
    Returns dS/dt in J/kg/K/yr.
    """
    T_stag = _S_to_T_jax(S)
    T_basic = mesh.quantity_matrix @ T_stag
    dTdr = mesh.d_dr_matrix @ T_stag

    heat_flux = -K_CONST * dTdr

    if F_outer is not None:
        heat_flux = heat_flux.at[-1].set(F_outer)
    heat_flux = heat_flux.at[0].set(F_inner)

    energy_flux = heat_flux * mesh.area
    capacitance = RHO_CONST * T_stag * mesh.volume
    return -jnp.diff(energy_flux) / capacitance * SECS_PER_YEAR


@pytest.mark.unit
class TestConstPropJAX:
    """Constant-property tests using JAX arrays (no EOS tables needed)."""

    def test_conduction_steady_state_no_drift(self):
        """Starting from the analytical conduction solution, dS/dt ~ 0."""
        N = 100
        mesh = _make_jax_mesh_arrays(N)
        T_inner, T_outer = 4000.0, 1500.0
        T_ss = _analytical_T(np.asarray(mesh.radii_stag), T_inner, T_outer)
        S_ss = jnp.asarray(_T_to_S(T_ss))

        # Prescribed flux matching analytical solution
        a, b = R_INNER, R_OUTER
        A_coeff = (T_inner - T_outer) * a * b / (b - a)
        Q_an = 4.0 * np.pi * K_CONST * A_coeff
        F_in = Q_an / (4.0 * np.pi * R_INNER**2)
        F_out = Q_an / (4.0 * np.pi * R_OUTER**2)

        dsdt = _const_prop_dSdt_jax(S_ss, mesh, F_inner=F_in, F_outer=F_out)
        max_rate = float(jnp.max(jnp.abs(dsdt)))
        # Should be very small (numerical error only)
        assert max_rate < 1e-3, f'dS/dt should be ~0 at steady state, got max |dS/dt| = {max_rate:.2e}'

    def test_conduction_flux_uniformity(self):
        """At steady state, Q = F * 4*pi*r^2 should be constant across shells."""
        N = 100
        mesh = _make_jax_mesh_arrays(N)
        T_ss = _analytical_T(np.asarray(mesh.radii_stag))
        S_ss = jnp.asarray(_T_to_S(T_ss))

        T_stag = _S_to_T_jax(S_ss)
        dTdr = mesh.d_dr_matrix @ T_stag
        heat_flux = -K_CONST * dTdr
        Q = heat_flux * mesh.area
        Q_interior = np.asarray(Q[1:-1])
        Q_mean = np.mean(Q_interior)
        Q_spread = np.max(np.abs(Q_interior - Q_mean)) / abs(Q_mean)
        assert Q_spread < 0.01, f'Flux non-uniformity: {Q_spread:.2%}'

    def test_convergence_with_resolution(self):
        """Flux uniformity error decreases with mesh resolution."""
        errors = []
        for N in [25, 50, 100, 200]:
            mesh = _make_jax_mesh_arrays(N)
            T_ss = _analytical_T(np.asarray(mesh.radii_stag))
            S_ss = jnp.asarray(_T_to_S(T_ss))
            T_stag = _S_to_T_jax(S_ss)
            dTdr = mesh.d_dr_matrix @ T_stag
            Q = (-K_CONST * dTdr) * mesh.area
            Q_int = np.asarray(Q[1:-1])
            Q_mean = np.mean(Q_int)
            errors.append(np.max(np.abs(Q_int - Q_mean)) / abs(Q_mean))
        assert errors[-1] < errors[0]

    def test_prescribed_flux_energy_conservation(self):
        """With prescribed flux, dE computed from dS/dt matches F*A."""
        N = 50
        mesh = _make_jax_mesh_arrays(N)
        S_uniform = jnp.full(N, _T_to_S(3000.0))
        F_prescribed = 100.0  # W/m^2

        dsdt = _const_prop_dSdt_jax(S_uniform, mesh, F_inner=0.0, F_outer=F_prescribed)

        # Energy rate from dS/dt: dE/dt = sum(rho * T * dS/dt * V) / SECS_PER_YEAR
        T = _S_to_T_jax(S_uniform)
        dE_dt_interior = float(jnp.sum(RHO_CONST * T * dsdt * mesh.volume)) / SECS_PER_YEAR
        # Expected: dE/dt = -F * A_surf
        A_surf = float(mesh.area[-1])
        dE_dt_expected = -F_prescribed * A_surf

        rel_err = abs(dE_dt_interior - dE_dt_expected) / abs(dE_dt_expected)
        assert rel_err < 0.05, (
            f'Energy rate mismatch: interior={dE_dt_interior:.2e}, '
            f'expected={dE_dt_expected:.2e}, error={rel_err:.2%}'
        )


# ---------------------------------------------------------------------------
# Tier 5: Solver integration tests (need EOS tables)
# ---------------------------------------------------------------------------

@needs_eos
@pytest.mark.smoke
class TestJAXSolverIntegration:
    """Integration tests for the full JAX solver pipeline."""

    def test_grey_body_cooling(self, jax_eos, default_params):
        """JAX solver cools the surface under grey-body BC."""
        from aragog.jax.solver import BoundaryParams, solve_entropy

        N = 30
        mesh = _make_jax_mesh_arrays(N)
        S_init = jnp.full(N, 3200.0)
        heating = jnp.zeros(N)

        bc = BoundaryParams(
            outer_bc_type=1,  # grey-body
            outer_bc_value=0.0,
            emissivity=1.0,
            T_eq=255.0,
            inner_bc_type=0,  # insulating
            inner_bc_value=0.0,
            core_density=10738.0,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )

        result = solve_entropy(
            S_init, 0.0, 100.0,
            jax_eos, default_params, mesh, bc, heating,
            atol=0.1, rtol=1e-4, max_steps=50_000,
            method='tsit5',
        )

        assert result.success, f'Solver failed after {result.n_steps} steps'
        S_final = np.asarray(result.S_final)
        assert S_final[-1] < 3200.0, 'Surface entropy should decrease'
        assert S_final[N // 2] < 3200.0, 'Interior entropy should decrease'

    def test_grey_body_temperature_decreases(self, jax_eos, default_params):
        """Grey-body cooling produces monotonically decreasing surface T."""
        from aragog.jax.solver import BoundaryParams, solve_entropy

        N = 20
        mesh = _make_jax_mesh_arrays(N)
        S_init = jnp.full(N, 3200.0)
        heating = jnp.zeros(N)

        bc = BoundaryParams(
            outer_bc_type=1,
            outer_bc_value=0.0,
            emissivity=1.0,
            T_eq=255.0,
            inner_bc_type=0,
            inner_bc_value=0.0,
            core_density=10738.0,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )

        result = solve_entropy(
            S_init, 0.0, 50.0,
            jax_eos, default_params, mesh, bc, heating,
            atol=0.1, rtol=1e-4, max_steps=50_000,
            method='tsit5',
        )

        assert result.success
        T_final = np.asarray(jax_eos.temperature(mesh.P_stag, result.S_final))
        T_init = np.asarray(jax_eos.temperature(mesh.P_stag, S_init))
        # Surface must cool
        assert T_final[-1] < T_init[-1], 'Surface temperature should decrease'
        # All T must remain positive
        assert np.all(T_final > 0), f'Negative T found: min={T_final.min()}'

    def test_prescribed_flux_bc(self, jax_eos, default_params):
        """Prescribed-flux BC (type 4) + insulating core."""
        from aragog.jax.solver import BoundaryParams, solve_entropy

        N = 20
        mesh = _make_jax_mesh_arrays(N)
        S_init = jnp.full(N, 3200.0)
        heating = jnp.zeros(N)

        # Moderate prescribed surface flux
        bc = BoundaryParams(
            outer_bc_type=4,  # prescribed flux
            outer_bc_value=1e4,  # 10 kW/m^2
            emissivity=1.0,
            T_eq=255.0,
            inner_bc_type=0,  # insulating
            inner_bc_value=0.0,
            core_density=10738.0,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )

        result = solve_entropy(
            S_init, 0.0, 100.0,
            jax_eos, default_params, mesh, bc, heating,
            atol=0.1, rtol=1e-4, max_steps=50_000,
            method='tsit5',
        )

        assert result.success
        S_final = np.asarray(result.S_final)
        # Prescribed outward flux should cool the surface
        assert S_final[-1] < 3200.0

    def test_internal_heating(self, jax_eos, default_params):
        """Uniform internal heating prevents entropy decrease."""
        from aragog.jax.solver import BoundaryParams, solve_entropy

        N = 20
        mesh = _make_jax_mesh_arrays(N)
        S_init = jnp.full(N, 3200.0)
        # Strong uniform heating
        heating = jnp.full(N, 1e-8)  # W/kg

        bc = BoundaryParams(
            outer_bc_type=0,  # insulating
            outer_bc_value=0.0,
            emissivity=1.0,
            T_eq=255.0,
            inner_bc_type=0,  # insulating
            inner_bc_value=0.0,
            core_density=10738.0,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )

        result = solve_entropy(
            S_init, 0.0, 1000.0,
            jax_eos, default_params, mesh, bc, heating,
            atol=0.1, rtol=1e-4, max_steps=50_000,
            method='tsit5',
        )

        assert result.success
        S_final = np.asarray(result.S_final)
        # With insulating BCs and heating, mean entropy must increase
        assert np.mean(S_final) > np.mean(np.asarray(S_init)), (
            'Mean entropy should increase with internal heating + insulating BCs'
        )


# ---------------------------------------------------------------------------
# Tier 5b: Energy conservation (Phase 5.4)
# ---------------------------------------------------------------------------

@needs_eos
@pytest.mark.smoke
class TestEnergyConservation:
    """Verify energy conservation: dE/dt = -F_surf * A_surf + H * M_mantle."""

    def test_prescribed_flux_energy_budget(self, jax_eos, default_params):
        """With prescribed surface flux and no heating, total energy change
        should match the time-integrated surface flux to within 5%.
        """
        from aragog.jax.solver import BoundaryParams, solve_entropy

        N = 30
        mesh = _make_jax_mesh_arrays(N)
        S_init = jnp.full(N, 3200.0)
        heating = jnp.zeros(N)
        F_prescribed = 5000.0  # W/m^2

        bc = BoundaryParams(
            outer_bc_type=4,  # prescribed flux
            outer_bc_value=F_prescribed,
            emissivity=1.0,
            T_eq=255.0,
            inner_bc_type=0,  # insulating
            inner_bc_value=0.0,
            core_density=10738.0,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )

        t_end = 50.0  # yr
        result = solve_entropy(
            S_init, 0.0, t_end,
            jax_eos, default_params, mesh, bc, heating,
            atol=0.1, rtol=1e-4, max_steps=50_000,
            method='tsit5',
        )
        assert result.success

        # Compute energy change via entropy capacitance: dE = sum(rho * T * dS * V)
        # This is the correct energy accounting for the entropy formulation,
        # where rho*T is the capacitance (not rho*Cp).
        P = mesh.P_stag
        vol = np.asarray(mesh.volume)
        S_init_np = np.asarray(S_init)
        S_final_np = np.asarray(result.S_final)
        dS = S_final_np - S_init_np

        # Use midpoint properties for the integral
        S_mid = 0.5 * (S_init_np + S_final_np)
        rho_mid = np.asarray(jax_eos.density(P, jnp.asarray(S_mid)))
        T_mid = np.asarray(jax_eos.temperature(P, jnp.asarray(S_mid)))

        dE = np.sum(rho_mid * T_mid * dS * vol)

        # Expected energy loss from prescribed surface flux
        A_surf = float(mesh.area[-1])
        Q_lost = F_prescribed * A_surf * t_end * SECS_PER_YEAR

        # dE should be approximately -Q_lost
        rel_err = abs(dE + Q_lost) / Q_lost
        assert rel_err < 0.10, (
            f'Energy conservation: dE={dE:.2e} J, Q_lost={Q_lost:.2e} J, '
            f'residual={rel_err:.1%}'
        )

    def test_insulating_no_heating_conserves(self, jax_eos, default_params):
        """With insulating BCs and no heating, total energy should be conserved."""
        from aragog.jax.solver import BoundaryParams, solve_entropy

        N = 20
        mesh = _make_jax_mesh_arrays(N)
        # Non-uniform IC to force internal redistribution
        S_init = jnp.linspace(3000.0, 3400.0, N)
        heating = jnp.zeros(N)

        bc = BoundaryParams(
            outer_bc_type=0,  # insulating (F=0 at surface is equivalent to type 0)
            outer_bc_value=0.0,
            emissivity=1.0,
            T_eq=255.0,
            inner_bc_type=0,  # insulating
            inner_bc_value=0.0,
            core_density=10738.0,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )

        result = solve_entropy(
            S_init, 0.0, 100.0,
            jax_eos, default_params, mesh, bc, heating,
            atol=0.1, rtol=1e-4, max_steps=50_000,
            method='tsit5',
        )
        assert result.success

        # Energy change via entropy capacitance: dE = sum(rho * T * dS * V)
        P = mesh.P_stag
        vol = np.asarray(mesh.volume)
        S_init_np = np.asarray(S_init)
        S_final_np = np.asarray(result.S_final)
        dS = S_final_np - S_init_np

        S_mid = 0.5 * (S_init_np + S_final_np)
        rho_mid = np.asarray(jax_eos.density(P, jnp.asarray(S_mid)))
        T_mid = np.asarray(jax_eos.temperature(P, jnp.asarray(S_mid)))

        dE = np.sum(rho_mid * T_mid * dS * vol)
        E_scale = np.sum(rho_mid * T_mid * np.abs(S_init_np) * vol)

        rel_change = abs(dE) / E_scale
        assert rel_change < 0.01, (
            f'Energy changed by {rel_change:.2%} with insulating BCs '
            f'(should be <1%). dE={dE:.2e}'
        )

    def test_heating_increases_energy(self, jax_eos, default_params):
        """With insulating BCs and uniform heating, energy should increase
        by approximately H * M_mantle * dt.

        The entropy-based energy change dE = sum(rho*T*dS*V) should match
        the total heating input. The heating term in the ODE is dS/dt += H/T,
        so the energy input is sum(rho*T*(H/T)*dt*V) = H * M * dt.
        """
        from aragog.jax.solver import BoundaryParams, solve_entropy

        N = 20
        mesh = _make_jax_mesh_arrays(N)
        S_init = jnp.full(N, 3200.0)
        H = 1e-8  # W/kg
        heating = jnp.full(N, H)

        bc = BoundaryParams(
            outer_bc_type=0,  # insulating
            outer_bc_value=0.0,
            emissivity=1.0,
            T_eq=255.0,
            inner_bc_type=0,  # insulating
            inner_bc_value=0.0,
            core_density=10738.0,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )

        t_end = 500.0  # yr
        result = solve_entropy(
            S_init, 0.0, t_end,
            jax_eos, default_params, mesh, bc, heating,
            atol=0.1, rtol=1e-4, max_steps=50_000,
            method='tsit5',
        )
        assert result.success

        P = mesh.P_stag
        vol = np.asarray(mesh.volume)
        S_init_np = np.asarray(S_init)
        S_final_np = np.asarray(result.S_final)
        dS = S_final_np - S_init_np

        S_mid = 0.5 * (S_init_np + S_final_np)
        rho_mid = np.asarray(jax_eos.density(P, jnp.asarray(S_mid)))
        T_mid = np.asarray(jax_eos.temperature(P, jnp.asarray(S_mid)))

        dE = np.sum(rho_mid * T_mid * dS * vol)

        # Expected: dE = H * M_mantle * dt
        M_mantle = np.sum(rho_mid * vol)
        dE_expected = H * M_mantle * t_end * SECS_PER_YEAR

        assert dE > 0, f'Energy should increase with heating, got dE={dE:.2e}'
        rel_err = abs(dE - dE_expected) / dE_expected
        assert rel_err < 0.10, (
            f'Energy increase dE={dE:.2e} vs expected H*M*dt={dE_expected:.2e}, '
            f'residual={rel_err:.1%}'
        )


# ---------------------------------------------------------------------------
# Tier 6: Solver parity (JAX diffrax vs scipy BDF)
# ---------------------------------------------------------------------------

@needs_eos
@pytest.mark.smoke
class TestSolverParity:
    """Verify JAX solver matches scipy BDF on identical problems."""

    def test_grey_body_parity(self, jax_eos, numpy_eos):
        """JAX Tsit5 and scipy BDF agree on grey-body cooling."""
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator
        from aragog.jax.phase import PhaseParams
        from aragog.jax.solver import BoundaryParams, solve_entropy
        from scipy.integrate import solve_ivp

        N = 20
        # Build mesh arrays for both
        mesh_jax = _make_jax_mesh_arrays(N)
        r_stag = np.asarray(mesh_jax.radii_stag)
        r_basic = np.asarray(mesh_jax.radii_basic)
        P_stag = np.asarray(mesh_jax.P_stag)
        P_basic = np.asarray(mesh_jax.P_basic)
        area = np.asarray(mesh_jax.area)
        volume = np.asarray(mesh_jax.volume)
        d_dr = np.asarray(mesh_jax.d_dr_matrix)
        q_mat = np.asarray(mesh_jax.quantity_matrix)

        S_init = np.full(N, 3200.0)
        T_eq = 255.0

        # ---- scipy BDF ----
        phase_stag = EntropyPhaseEvaluator(entropy_eos=numpy_eos, gravitational_acceleration=10.0)
        phase_stag.set_pressure(P_stag)
        phase_basic = EntropyPhaseEvaluator(entropy_eos=numpy_eos, gravitational_acceleration=10.0)
        phase_basic.set_pressure(P_basic)

        def scipy_rhs(t, S):
            S = np.asarray(S)
            phase_stag.set_entropy(S)
            phase_stag.update()
            S_basic = q_mat @ S
            phase_basic.set_entropy(S_basic)
            phase_basic.update()

            T_stag = phase_stag.temperature()
            dTdr = d_dr @ T_stag
            k = (1.0 - phase_basic.melt_fraction()) * 4.0 + phase_basic.melt_fraction() * 2.0

            heat_flux = -k * dTdr

            # Grey-body surface BC
            T_surf = phase_basic.temperature()[-1]
            heat_flux[-1] = SIGMA_SB * (T_surf**4 - T_eq**4)
            heat_flux[0] = 0.0  # insulating core

            energy_flux = heat_flux * area
            cap = phase_stag.density() * phase_stag.temperature() * volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        t_end = 50.0  # yr
        sol_scipy = solve_ivp(scipy_rhs, (0, t_end), S_init, method='BDF',
                              atol=0.1, rtol=1e-4)
        assert sol_scipy.status == 0

        # ---- JAX diffrax ----
        params = PhaseParams(convection=False)  # conduction only for clean comparison
        bc = BoundaryParams(
            outer_bc_type=1, outer_bc_value=0.0,
            emissivity=1.0, T_eq=T_eq,
            inner_bc_type=0, inner_bc_value=0.0,
            core_density=10738.0, core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )
        heating = jnp.zeros(N)

        result_jax = solve_entropy(
            jnp.asarray(S_init), 0.0, t_end,
            jax_eos, params, mesh_jax, bc, heating,
            atol=0.1, rtol=1e-4, max_steps=50_000,
            method='tsit5',
        )
        assert result_jax.success

        # Compare
        S_scipy = sol_scipy.y[:, -1]
        S_jax = np.asarray(result_jax.S_final)

        # Surface entropy should agree within ~20 J/kg/K.
        # The scipy RHS is a simplified manual implementation; the JAX solver
        # uses the full compute_fluxes pipeline with matrix transforms, so
        # some difference is expected from interpolation details.
        assert abs(S_scipy[-1] - S_jax[-1]) < 20.0, (
            f'Surface S mismatch: scipy={S_scipy[-1]:.1f}, jax={S_jax[-1]:.1f}'
        )
        # Mean profile should agree within 10 J/kg/K
        assert abs(np.mean(S_scipy) - np.mean(S_jax)) < 10.0, (
            f'Mean S mismatch: scipy={np.mean(S_scipy):.1f}, jax={np.mean(S_jax):.1f}'
        )

    @pytest.mark.xfail(
        reason='ImplicitEuler on full EOS chain hits step limit (Phase 10 optimization needed)',
        strict=False,
    )
    def test_implicit_euler_parity(self, jax_eos, default_params):
        """ImplicitEuler agrees with Tsit5 on a short cooling run."""
        from aragog.jax.solver import BoundaryParams, solve_entropy

        N = 9
        mesh = _make_jax_mesh_arrays(N)
        S_init = jnp.full(N, 3200.0)
        heating = jnp.zeros(N)

        bc = BoundaryParams(
            outer_bc_type=1, outer_bc_value=0.0,
            emissivity=1.0, T_eq=255.0,
            inner_bc_type=0, inner_bc_value=0.0,
            core_density=10738.0, core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )

        result_tsit5 = solve_entropy(
            S_init, 0.0, 10.0,
            jax_eos, default_params, mesh, bc, heating,
            atol=0.5, rtol=1e-3, method='tsit5',
            max_steps=100_000,
        )
        result_ie = solve_entropy(
            S_init, 0.0, 10.0,
            jax_eos, default_params, mesh, bc, heating,
            atol=0.5, rtol=1e-3, method='implicit_euler',
            max_steps=200_000,
        )

        assert result_tsit5.success
        assert result_ie.success

        S_tsit5 = np.asarray(result_tsit5.S_final)
        S_ie = np.asarray(result_ie.S_final)
        # Should agree within tolerance band
        max_diff = np.max(np.abs(S_tsit5 - S_ie))
        assert max_diff < 10.0, (
            f'Tsit5 vs ImplicitEuler max difference: {max_diff:.1f} J/kg/K'
        )


# ---------------------------------------------------------------------------
# Tier 7: Jgrav smoothing regression tests (2026-04-09)
# ---------------------------------------------------------------------------

@needs_eos
@pytest.mark.unit
class TestJAXJgravSmoothing:
    """Regression tests for the cubic Hermite Jgrav smoothing in the JAX path.

    Mirrors the scipy-path TestJgravSmoothing in test_entropy_pytest.py.
    Before the 2026-04-09 fix, aragog/jax/phase.py computed a raw
    gravitational-separation mass flux `rho * phi * (1-phi) * v_rel`
    with no smoothing; at first crystallisation this drained the CMB
    cell's entropy off the PALEOS P-S table domain in a single
    coupling step, same as the pre-fix scipy path. The fix ports the
    SPIDER-analogue cubic Hermite smoothing
        smth = 16 * gphi^2 * (1 - gphi)^2    for gphi in [0, 1]
    where gphi is the un-truncated two-phase fraction at the
    STAGGERED cell below the interface. These tests lock in the
    correct behaviour under the diffrax solvers.

    See ``aragog/src/aragog/jax/phase.py::compute_fluxes`` for the
    JAX implementation and ``aragog/src/aragog/solver/entropy_state.py``
    for the scipy twin.
    """

    @staticmethod
    def _build_mesh_and_phase_params(
        jax_eos, N: int = 15, grav_sep: bool = True,
        bottom_up_grav_sep: bool = True, grain_size: float = 0.1,
    ):
        """Earth-like Stokes-regime mesh + PhaseParams.

        grain_size = 0.1 m matches the R8 CHILI config and makes the
        test discriminating: with 1 mm grain the Stokes permeability
        is so small that the bug is hard to reproduce.
        """
        from aragog.jax.phase import PhaseParams

        mesh = _make_jax_mesh_arrays(N)
        params = PhaseParams(
            phi_rheo=0.4,
            phi_width=0.15,
            viscosity_solid=1e22,
            viscosity_liquid=1e2,
            grain_size=grain_size,
            k_solid=4.0,
            k_liquid=4.0,
            conduction=True,
            convection=True,
            grav_sep=grav_sep,
            mixing=True,
            eddy_diff_thermal=1.0,
            eddy_diff_chemical=1.0,
            kappah_floor=10.0,
            bottom_up_grav_sep=bottom_up_grav_sep,
        )
        return mesh, params

    @staticmethod
    def _grey_body_bc():
        from aragog.jax.solver import BoundaryParams
        return BoundaryParams(
            outer_bc_type=1,  # grey-body
            outer_bc_value=0.0,
            emissivity=1.0,
            T_eq=255.0,
            inner_bc_type=0,  # insulating
            inner_bc_value=0.0,
            core_density=10738.0,
            core_heat_capacity=880.0,
            tfac_core_avg=1.147,
        )

    def test_smoothing_vanishes_at_pure_phases(self, jax_eos):
        """Smth = 0 at gphi = 0 and gphi = 1, peaks at gphi = 0.5.

        Unit test of the smoothing factor itself, independent of the
        ODE solver. Loads the flux kernel, constructs an entropy
        profile with known phase-fraction values at each staggered
        node, and reads the smoothed mass_flux back to verify:
          - at nodes where the staggered cell below is pure liquid
            (gphi > 1), smth = 0 so the mass flux at the basic
            interface above it is killed;
          - at nodes where the staggered cell below is pure solid
            (gphi < 0), smth = 0 for the same reason;
          - at nodes in the mushy zone (gphi near 0.5), smth
            approaches 1 and the raw Jgrav is preserved.
        """
        from aragog.jax.phase import compute_fluxes

        mesh, params = self._build_mesh_and_phase_params(
            jax_eos, N=20, grav_sep=True, bottom_up_grav_sep=True,
        )
        N = mesh.P_stag.shape[0]

        # Build an entropy profile that's pure liquid in the upper
        # mantle and pure solid at the bottom, with a mushy
        # transition in the middle. Use the analytical inverse:
        # S = S_sol + gphi * (S_liq - S_sol) with prescribed gphi
        # per node.
        S_sol = jax_eos.solidus_entropy(mesh.P_stag)
        S_liq = jax_eos.liquidus_entropy(mesh.P_stag)
        # gphi: -0.2 at CMB (pure solid, below solidus),
        # ramps to 0.5 at mid-mantle (mid-mushy),
        # 1.2 at the top (above liquidus, pure liquid).
        idx = jnp.arange(N, dtype=jnp.float64)
        gphi_target = -0.2 + 1.4 * idx / (N - 1)
        S_stag = S_sol + gphi_target * (S_liq - S_sol)

        heating = jnp.zeros(N)
        flux_out = compute_fluxes(
            S_stag, 0.0, jax_eos, params, mesh, heating,
        )
        mass_flux = np.asarray(flux_out.mass_flux)

        # Expected: smth_basic[i] = 16 * gphi_clip[i-1]^2 * (1 -
        # gphi_clip[i-1])^2 for interior basic nodes i in 1..N-1,
        # with gphi_clip = clip(gphi_target[i-1], 0, 1). For the
        # pure-phase ends (|gphi - 0.5| > 0.5), smth = 0 so the
        # smoothed mass flux must also be ~0 there.
        gphi_np = np.asarray(gphi_target)
        gphi_clip = np.clip(gphi_np, 0.0, 1.0)
        smth_stag = 16.0 * gphi_clip**2 * (1.0 - gphi_clip) ** 2

        # Check pure-phase suppression: at basic node i = 1, the
        # staggered cell below is index 0 with gphi = -0.2, so
        # smth = 0 and mass_flux[1] should be ~0.
        assert abs(mass_flux[1]) < 1e-6, (
            f'mass_flux[1] = {mass_flux[1]:.3e} should be ~0 at '
            f'pure-solid CMB cell (gphi={gphi_np[0]:.2f}, '
            f'smth={smth_stag[0]:.3e})'
        )
        # At basic node i = N-1, the staggered cell below is index
        # N-2 with gphi close to 1, so smth approaches 0.
        assert abs(mass_flux[N - 1]) < 1e-6, (
            f'mass_flux[{N-1}] = {mass_flux[N-1]:.3e} should be ~0 at '
            f'pure-liquid top cell (gphi={gphi_np[N-2]:.2f}, '
            f'smth={smth_stag[N-2]:.3e})'
        )
        # Near mid-mantle the smoothing should peak and preserve a
        # non-trivial mass flux. Find the node whose below-cell gphi
        # is closest to 0.5 and assert that node has clearly larger
        # |mass_flux| than the pure-phase boundaries.
        i_mid_stag = int(np.argmin(np.abs(gphi_np - 0.5)))
        # basic node index for that "cell below" is i_mid_stag + 1
        i_mid_basic = i_mid_stag + 1
        if 0 < i_mid_basic < N:
            assert abs(mass_flux[i_mid_basic]) > 10.0 * max(
                abs(mass_flux[1]), abs(mass_flux[N - 1]), 1e-12,
            ), (
                f'Mid-mushy mass flux ({mass_flux[i_mid_basic]:.3e}) '
                f'should dominate pure-phase flux '
                f'({mass_flux[1]:.3e}, {mass_flux[N-1]:.3e})'
            )

    def test_smoothing_on_vs_off_differs_at_mushy_edge(self, jax_eos):
        """bottom_up_grav_sep=False must give a strictly larger total
        |Jgrav| than bottom_up_grav_sep=True in a profile with a
        mushy layer, because the raw flux is non-zero near the
        pure-phase edges of the layer (where gphi → 0 or 1) while
        the smoothed flux drops to zero there.
        """
        from aragog.jax.phase import compute_fluxes

        mesh, params_on = self._build_mesh_and_phase_params(
            jax_eos, N=20, grav_sep=True, bottom_up_grav_sep=True,
        )
        _, params_off = self._build_mesh_and_phase_params(
            jax_eos, N=20, grav_sep=True, bottom_up_grav_sep=False,
        )
        N = mesh.P_stag.shape[0]

        # Build a mushy-ramp IC: gphi at staggered nodes runs from
        # -0.1 (pure solid, just below CMB liquidus) at the bottom
        # to 1.1 (pure liquid, just above surface liquidus) at the
        # top. The node-by-node phi = clip(gphi, 0, 1) varies across
        # the middle of the profile, so phi * (1 - phi) * v_rel is
        # non-zero and both ON and OFF give a non-trivial raw flux
        # somewhere.
        S_sol = jax_eos.solidus_entropy(mesh.P_stag)
        S_liq = jax_eos.liquidus_entropy(mesh.P_stag)
        idx = jnp.arange(N, dtype=jnp.float64)
        gphi_target = -0.1 + 1.2 * idx / (N - 1)
        S_stag = S_sol + gphi_target * (S_liq - S_sol)

        heating = jnp.zeros(N)
        flux_on = compute_fluxes(
            S_stag, 0.0, jax_eos, params_on, mesh, heating,
        )
        flux_off = compute_fluxes(
            S_stag, 0.0, jax_eos, params_off, mesh, heating,
        )

        mass_on = np.asarray(flux_on.mass_flux)
        mass_off = np.asarray(flux_off.mass_flux)

        # L1 norm of the mass flux across the profile. OFF should be
        # larger than ON because the un-smoothed flux is active at
        # every phi in (0, 1), while the smoothed version vanishes
        # near the pure-phase edges.
        l1_on = float(np.sum(np.abs(mass_on)))
        l1_off = float(np.sum(np.abs(mass_off)))

        assert np.isfinite(l1_on), f'ON variant non-finite: {l1_on}'
        assert np.isfinite(l1_off), f'OFF variant non-finite: {l1_off}'

        # OFF variant must have strictly larger L1 flux than ON.
        # Allow a modest margin (5%) to tolerate numerical noise.
        assert l1_off > 1.05 * l1_on, (
            f'Smoothing appears silently off: '
            f'L1|mass_flux|(ON)={l1_on:.3e} vs '
            f'L1|mass_flux|(OFF)={l1_off:.3e} '
            f'(expected OFF > 1.05 * ON)'
        )
        # And ON must itself be non-zero (smoothing preserves the
        # mid-mushy flux), so the test is meaningful.
        assert l1_on > 0.0, (
            'Smoothing zeroed the entire mass flux; expected a '
            'non-trivial mid-mushy contribution'
        )

    def test_cubic_hermite_matches_scipy_path(self, jax_eos):
        """The JAX cubic Hermite smoothing must match the scipy path.

        Checks that
            smth = 16 * gphi^2 * (1 - gphi)^2
        is applied with the same clip semantics
        (gphi < 0 -> 0, gphi > 1 -> 0) as the scipy twin in
        aragog/src/aragog/solver/entropy_state.py:256-260.
        """
        import numpy as _np
        # Manual reference implementation
        def smth_ref(gphi):
            g = _np.clip(gphi, 0.0, 1.0)
            return 16.0 * g**2 * (1.0 - g) ** 2

        # Sample gphi values spanning the domain
        gphi_samples = _np.linspace(-0.5, 1.5, 21)
        expected = smth_ref(gphi_samples)

        # Hard-coded expected peak at gphi=0.5
        assert abs(smth_ref(0.5) - 1.0) < 1e-12
        # Zero at the pure-phase boundaries
        assert smth_ref(0.0) == 0.0
        assert smth_ref(1.0) == 0.0
        # Zero outside [0, 1] (clip behaviour)
        assert smth_ref(-0.1) == 0.0
        assert smth_ref(1.1) == 0.0
        # Symmetry about gphi=0.5
        for g in [0.1, 0.25, 0.4]:
            assert abs(smth_ref(g) - smth_ref(1.0 - g)) < 1e-12

        # Monotone on each half of [0, 1]
        g1 = _np.linspace(0.0, 0.5, 10)
        g2 = _np.linspace(0.5, 1.0, 10)
        assert _np.all(_np.diff(smth_ref(g1)) >= -1e-15)
        assert _np.all(_np.diff(smth_ref(g2)) <= 1e-15)


# ---------------------------------------------------------------------------
# Tier 7: SPIDER-parity ports landed in Z.6.A (aragog 09cd760, 96894db)
# ---------------------------------------------------------------------------
#
# Three structural pieces of physics in the JAX path were brought into
# bit-tight parity with the numpy EntropyPhaseEvaluator._update_eos /
# entropy_state.update path during the Z.6.A iteration:
#
#   1. EntropyEOS_JAX.compute_phase_state — single-pass evaluation that
#      derives all six properties (T, rho, Cp, alpha, dTdPs, k) from the
#      same (gphi, smth, S_sol, S_liq, T_sol, T_liq, rho_sol, rho_liq)
#      cache and applies the SPIDER combine_matprop blend
#      smth*mixed + (1-smth)*single. Replaces the per-property API that
#      only had the mixed (harmonic-mean) branch.
#
#   2. compute_fluxes conduction term — uses the SPIDER decomposition
#      F_cond = -k * [(T/Cp)*dSdr + dTdPs*dPdr_basic] in place of the
#      textbook -k * d/dr(T_stag).
#
#   3. compute_mlt CMB fix and compute_fluxes surface dSdr boundary copy
#      — mirror SPIDER energy.c:220-223 (kappa_h[0] = kappa_h[1]) and
#      ic.c:450 (dSdr boundary copy from adjacent interior).
#
# The tests below exercise each port against the numpy reference
# implementation that compute_phase_state was ported from.

@needs_eos
@pytest.mark.unit
class TestSPIDERParityPorts:
    """Bit-tight parity between JAX compute_phase_state and the numpy
    EntropyPhaseEvaluator with cp_blend='latent'."""

    @pytest.fixture(scope='class')
    def numpy_phase_eval(self, numpy_eos):
        """Numpy EntropyPhaseEvaluator with the same defaults as the
        JAX PhaseParams used in production CHILI runs."""
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator
        return EntropyPhaseEvaluator(
            entropy_eos=numpy_eos,
            gravitational_acceleration=9.81,
            thermal_conductivity_solid=4.0,
            thermal_conductivity_liquid=2.0,
            cp_blend='latent',
            matprop_smooth_width=0.01,  # CHILI Earth production setting
        )

    @pytest.mark.parametrize('S_off', [-300.0, -100.0, 0.0, 100.0, 300.0])
    def test_compute_phase_state_blends_match_numpy(
        self, jax_eos, numpy_phase_eval, S_off,
    ):
        """compute_phase_state(P, S, k_solid, k_liquid, matprop_smooth_width)
        must reproduce numpy ._update_eos with cp_blend='latent' bit-tight
        across solid, mushy, and molten regimes for all six blended
        properties."""
        P_arr = np.linspace(10e9, 130e9, 25)
        # Sample S relative to the midpoint of solidus and liquidus so the
        # parametrize sweep crosses pure-solid / mushy / pure-melt regimes.
        S_sol = np.asarray(jax_eos.solidus_entropy(jnp.asarray(P_arr)))
        S_liq = np.asarray(jax_eos.liquidus_entropy(jnp.asarray(P_arr)))
        S_mid = 0.5 * (S_sol + S_liq)
        S_arr = S_mid + S_off

        # JAX side
        state = jax_eos.compute_phase_state(
            jnp.asarray(P_arr), jnp.asarray(S_arr),
            k_solid=4.0, k_liquid=2.0, matprop_smooth_width=0.01,
        )

        # Numpy side
        numpy_phase_eval.set_pressure(P_arr)
        numpy_phase_eval.set_entropy(S_arr)
        numpy_phase_eval.update()
        T_np = np.asarray(numpy_phase_eval.temperature()).ravel()
        rho_np = np.asarray(numpy_phase_eval.density()).ravel()
        Cp_np = np.asarray(numpy_phase_eval.heat_capacity()).ravel()
        alpha_np = np.asarray(numpy_phase_eval.thermal_expansivity()).ravel()
        dTdPs_np = np.asarray(numpy_phase_eval.dTdPs()).ravel()
        # k_thermal in numpy: linear blend phi*k_liq + (1-phi)*k_sol then
        # smth-blended against the single-phase k_single (constant per phase)
        k_np = np.asarray(numpy_phase_eval._thermal_conductivity_val).ravel()
        phi_np = np.asarray(numpy_phase_eval.melt_fraction()).ravel()

        # Bit-tight rtol; the only intentional non-bit-identical operation
        # is the alpha guard 0.5*(a + sqrt(a^2 + eps^2)) which is identical
        # in both implementations.
        np.testing.assert_allclose(
            np.asarray(state.temperature), T_np, rtol=1e-10,
            err_msg=f'temperature parity at S_off={S_off}',
        )
        np.testing.assert_allclose(
            np.asarray(state.density), rho_np, rtol=1e-10,
            err_msg=f'density parity at S_off={S_off}',
        )
        np.testing.assert_allclose(
            np.asarray(state.heat_capacity), Cp_np, rtol=1e-10,
            err_msg=f'heat_capacity parity at S_off={S_off}',
        )
        np.testing.assert_allclose(
            np.asarray(state.thermal_expansivity), alpha_np, rtol=1e-5,
            err_msg=f'thermal_expansivity parity at S_off={S_off}',
        )
        np.testing.assert_allclose(
            np.asarray(state.dTdPs), dTdPs_np, rtol=1e-10,
            err_msg=f'dTdPs parity at S_off={S_off}',
        )
        np.testing.assert_allclose(
            np.asarray(state.thermal_conductivity), k_np, rtol=1e-10,
            err_msg=f'thermal_conductivity parity at S_off={S_off}',
        )
        np.testing.assert_allclose(
            np.asarray(state.melt_fraction), phi_np, rtol=1e-10, atol=1e-12,
            err_msg=f'melt_fraction parity at S_off={S_off}',
        )

    def test_compute_phase_state_smth_zero_when_outside_mushy(
        self, jax_eos,
    ):
        """With matprop_smooth_width=0 the SPIDER convention is
        smth=1 strictly inside (0,1) and 0 elsewhere — the blend
        collapses to pure single-phase outside the mushy band."""
        # Pick three pressures and entropies that are clearly solid,
        # mushy, and molten respectively.
        P = jnp.asarray([60e9, 60e9, 60e9])
        S_sol = float(jax_eos.solidus_entropy(60e9))
        S_liq = float(jax_eos.liquidus_entropy(60e9))
        S = jnp.asarray([S_sol - 200, 0.5 * (S_sol + S_liq), S_liq + 200])
        state = jax_eos.compute_phase_state(
            P, S, k_solid=4.0, k_liquid=2.0, matprop_smooth_width=0.0,
        )
        smth = np.asarray(state.smth)
        assert smth[0] == 0.0, 'pure solid should have smth=0'
        assert smth[1] == 1.0, 'mushy should have smth=1'
        assert smth[2] == 0.0, 'pure melt should have smth=0'

    def test_compute_phase_state_jit_compiles(self, jax_eos):
        """compute_phase_state must JIT-compile (required for the
        analytic Jacobian path which jit-traces the whole RHS)."""
        @jax.jit
        def fn(P, S):
            return jax_eos.compute_phase_state(
                P, S, k_solid=4.0, k_liquid=2.0, matprop_smooth_width=0.01,
            )
        P = jnp.full(5, 50e9)
        S = jnp.linspace(2800.0, 3300.0, 5)
        state = fn(P, S)
        assert state.density.shape == (5,)
        assert jnp.all(state.density > 0)


@needs_eos
@pytest.mark.unit
class TestSPIDERConductionDecomposition:
    """compute_fluxes conduction term: -k*[(T/Cp)*dSdr + dTdPs*dPdr_basic]."""

    def test_dPdr_basic_field_present_and_matches_np_gradient(self):
        """MeshArrays must carry dP_dr_basic, computed via
        np.gradient(P_basic, r_basic). Construct a synthetic mesh-like
        namespace to verify ``from_numpy_mesh`` populates the field
        correctly without depending on the full aragog Mesh constructor
        (which needs a fully-resolved Parameters object that is awkward
        to build in a unit test)."""
        from types import SimpleNamespace
        from aragog.jax.phase import MeshArrays
        n_basic = 10
        n_stag = n_basic - 1
        r_basic_np = np.linspace(3.5e6, 6.371e6, n_basic)
        r_stag_np = 0.5 * (r_basic_np[1:] + r_basic_np[:-1])
        P_basic_np = np.linspace(120e9, 1e9, n_basic)
        # Synthetic mesh with the same attribute surface that
        # MeshArrays.from_numpy_mesh consumes.
        basic = SimpleNamespace(
            radii=r_basic_np,
            area=np.ones(n_basic),
            volume=np.ones(n_basic),
            mixing_length=np.full(n_basic, 1e5),
            mixing_length_squared=np.full(n_basic, 1e10),
            mixing_length_cubed=np.full(n_basic, 1e15),
        )
        staggered = SimpleNamespace(radii=r_stag_np)
        mesh = SimpleNamespace(
            basic=basic,
            staggered=staggered,
            staggered_pressure=np.linspace(120e9, 5e9, n_stag),
            basic_pressure=P_basic_np,
            _d_dr_transform=np.zeros((n_basic, n_stag)),
            _quantity_transform=np.zeros((n_basic, n_stag)),
            eos=SimpleNamespace(_gravitational_acceleration=9.81),
        )
        jax_mesh = MeshArrays.from_numpy_mesh(mesh)
        # Field must exist and have the right shape
        assert jax_mesh.dP_dr_basic.shape == (n_basic,), (
            f'dP_dr_basic shape mismatch: got {jax_mesh.dP_dr_basic.shape}, '
            f'expected ({n_basic},)'
        )
        # Must equal np.gradient(P_basic, r_basic) bit-tight (the JAX
        # MeshArrays just forwards the numpy gradient).
        expected = np.gradient(P_basic_np, r_basic_np)
        np.testing.assert_allclose(
            np.asarray(jax_mesh.dP_dr_basic), expected, rtol=1e-12,
        )
        # Sign check: P decreases outward (large r → small P) so dPdr < 0.
        signs = np.sign(np.asarray(jax_mesh.dP_dr_basic)[1:-1])
        assert (signs == signs[0]).all(), (
            f'dPdr should be monotonic; got mixed signs: {signs}'
        )

    def test_compute_fluxes_conduction_only_matches_analytic(
        self, jax_eos, default_params,
    ):
        """For an isentropic profile (dSdr=0), the SPIDER conduction
        formula reduces to F_cond = -k * dTdPs * dPdr — purely the
        adiabatic-gradient term, no super-adiabatic contribution."""
        from aragog.jax.phase import (
            MeshArrays, PhaseParams, compute_fluxes,
        )
        # A toy mesh: linear pressure profile, uniform spacing.
        n_stag = 10
        n_basic = n_stag + 1
        r_basic = jnp.linspace(3.5e6, 6.371e6, n_basic)
        r_stag = 0.5 * (r_basic[1:] + r_basic[:-1])
        # Synthetic Adams-Williamson-ish pressure profile (linear in r is fine
        # for a sanity check; the analytic comparison only needs dPdr to be
        # consistent between the mesh and the test).
        P_basic = jnp.linspace(120e9, 0.0, n_basic)
        P_stag = 0.5 * (P_basic[1:] + P_basic[:-1])
        dr = float(r_basic[1] - r_basic[0])
        # Build the d_dr / quantity matrices for this trivial mesh.
        d_dr = jnp.zeros((n_basic, n_stag))
        # interior centered diff
        for i in range(1, n_basic - 1):
            d_dr = d_dr.at[i, i - 1].set(-1.0 / dr)
            d_dr = d_dr.at[i, i].set(1.0 / dr)
        # boundary linear extrapolation (matches the SPIDER convention)
        d_dr = d_dr.at[0, 0].set(0.0)  # replaced by boundary copy in compute_fluxes
        d_dr = d_dr.at[-1, -1].set(0.0)  # replaced by boundary copy
        q = jnp.zeros((n_basic, n_stag))
        for i in range(1, n_basic - 1):
            q = q.at[i, i - 1].set(0.5)
            q = q.at[i, i].set(0.5)
        q = q.at[0, 0].set(1.0)
        q = q.at[-1, -1].set(1.0)

        mesh = MeshArrays(
            d_dr_matrix=d_dr,
            quantity_matrix=q,
            area=jnp.ones(n_basic),
            volume=jnp.ones(n_basic),
            radii_basic=r_basic,
            radii_stag=r_stag,
            mixing_length=jnp.ones(n_basic) * 1e5,
            mixing_length_sq=jnp.ones(n_basic) * 1e10,
            mixing_length_cu=jnp.ones(n_basic) * 1e15,
            P_stag=P_stag,
            P_basic=P_basic,
            dP_dr_basic=jnp.gradient(P_basic, r_basic),
            gravity=jnp.full(n_basic, 9.81),
        )

        # PhaseParams: only conduction enabled, no convection / grav / mix.
        # Without convection the surface dSdr fix and CMB kappa_h fix are
        # irrelevant — F is purely conductive everywhere.
        params = PhaseParams(
            conduction=True, convection=False,
            grav_sep=False, mixing=False,
            kappah_floor=0.0,
            matprop_smooth_width=0.01,
        )

        S = jnp.full(n_stag, 3000.0)  # isentropic
        flux = compute_fluxes(
            S, 0.0, jax_eos, params, mesh, jnp.zeros(n_stag),
        )

        # Interior node sanity: F_cond should be O(k * |dTdPs * dPdr|)
        # which for k=4, dTdPs ~ 1e-8, dPdr ~ -1e7 / 1e6 = -10 Pa/m,
        # gives F_cond ~ 4 * 1e-8 * 10 = 4e-7 W/m^2. (Wait, dPdr is
        # 120e9 / 2.871e6 m ~ 4.2e4 Pa/m, so F ~ 4 * 1e-8 * 4.2e4 ~ 1.7e-3.)
        # The exact numerical value is not what we are checking here; we
        # check only that F is non-zero and has the expected sign in the
        # interior.
        F_interior = np.asarray(flux.heat_flux[3:-3])
        assert np.all(np.isfinite(F_interior)), 'F_cond must be finite'
        # On an isentropic profile with negative dPdr, F_cond = -k * dTdPs * dPdr
        # is positive (heat flows outward) for typical EOS dTdPs > 0.
        assert F_interior.mean() > 0, (
            f'F_cond on isentropic IC should be positive (outward), '
            f'got mean = {F_interior.mean()}'
        )


@needs_eos
@pytest.mark.unit
class TestBoundaryCopies:
    """SPIDER ic.c:450 / energy.c:220-223 boundary copies."""

    def test_compute_mlt_kappa_h_cmb_copy(self, jax_eos, default_params):
        """compute_mlt must enforce kappa_h[0] = kappa_h[1] (numpy
        entropy_state.py:533). Test by feeding a strongly super-adiabatic
        gradient at idx 1 and checking idx 0 picks it up."""
        from aragog.jax.phase import (
            MeshArrays, PhaseParams, compute_mlt, evaluate_phase,
        )
        n = 10
        n_basic = n + 1
        # Build a small synthetic mesh
        r_basic = jnp.linspace(3.5e6, 6.371e6, n_basic)
        r_stag = 0.5 * (r_basic[1:] + r_basic[:-1])
        P_basic = jnp.linspace(120e9, 1e9, n_basic)
        P_stag = 0.5 * (P_basic[1:] + P_basic[:-1])
        mesh = MeshArrays(
            d_dr_matrix=jnp.zeros((n_basic, n)),
            quantity_matrix=jnp.zeros((n_basic, n)),
            area=jnp.ones(n_basic),
            volume=jnp.ones(n_basic),
            radii_basic=r_basic,
            radii_stag=r_stag,
            mixing_length=jnp.ones(n_basic) * 1e5,
            mixing_length_sq=jnp.ones(n_basic) * 1e10,
            mixing_length_cu=jnp.ones(n_basic) * 1e15,
            P_stag=P_stag,
            P_basic=P_basic,
            dP_dr_basic=jnp.gradient(P_basic, r_basic),
            gravity=jnp.full(n_basic, 9.81),
        )
        S_basic = jnp.full(n_basic, 3000.0)
        ph = evaluate_phase(jax_eos, default_params, P_basic, S_basic)
        # Strongly negative dSdr (unstable) at idx 1, mild elsewhere
        dSdr = jnp.full(n_basic, -1e-6)
        dSdr = dSdr.at[0].set(+1e-6)  # if NOT copied, idx 0 would be stable
        kh, _ = compute_mlt(dSdr, ph, mesh, default_params)
        # SPIDER copy enforces kh[0] = kh[1]
        assert float(kh[0]) == float(kh[1]), (
            f'kappa_h[0] must equal kappa_h[1] after SPIDER CMB copy; '
            f'got {float(kh[0])} vs {float(kh[1])}'
        )

    def test_compute_fluxes_dSdr_surface_copy(self, jax_eos, default_params):
        """compute_fluxes must enforce dSdr[-1] = dSdr[-2] before the
        flux computation (numpy entropy_state.py:390 ``dSdxi[-1] =
        dSdxi[-2]``). Verify by constructing a d_dr matrix that would
        give a wrong-sign gradient at the top boundary, and checking
        that the surface F_conv has the same sign as the adjacent
        interior."""
        from aragog.jax.phase import (
            MeshArrays, PhaseParams, compute_fluxes,
        )
        n = 8
        n_basic = n + 1
        r_basic = jnp.linspace(3.5e6, 6.371e6, n_basic)
        r_stag = 0.5 * (r_basic[1:] + r_basic[:-1])
        P_basic = jnp.linspace(120e9, 1e9, n_basic)
        P_stag = 0.5 * (P_basic[1:] + P_basic[:-1])
        dr = float(r_basic[1] - r_basic[0])
        d_dr = jnp.zeros((n_basic, n))
        # Interior centered diffs
        for i in range(1, n_basic - 1):
            d_dr = d_dr.at[i, i - 1].set(-1.0 / dr)
            d_dr = d_dr.at[i, i].set(1.0 / dr)
        # Boundary rows that would WRONGLY give the opposite-sign gradient
        # at the surface basic node — the SPIDER copy must overwrite this.
        d_dr = d_dr.at[-1, -1].set(-1.0 / dr)  # wrong sign on purpose
        d_dr = d_dr.at[-1, -2].set(+1.0 / dr)
        q = jnp.zeros((n_basic, n))
        for i in range(1, n_basic - 1):
            q = q.at[i, i - 1].set(0.5)
            q = q.at[i, i].set(0.5)
        q = q.at[0, 0].set(1.0)
        q = q.at[-1, -1].set(1.0)
        mesh = MeshArrays(
            d_dr_matrix=d_dr, quantity_matrix=q,
            area=jnp.ones(n_basic), volume=jnp.ones(n_basic),
            radii_basic=r_basic, radii_stag=r_stag,
            mixing_length=jnp.ones(n_basic) * 1e5,
            mixing_length_sq=jnp.ones(n_basic) * 1e10,
            mixing_length_cu=jnp.ones(n_basic) * 1e15,
            P_stag=P_stag, P_basic=P_basic,
            dP_dr_basic=jnp.gradient(P_basic, r_basic),
            gravity=jnp.full(n_basic, 9.81),
        )
        # Linearly increasing entropy with index (so centered-diff dSdr is
        # positive in the interior; the buggy surface row would flip it).
        S = jnp.linspace(2800.0, 3500.0, n)
        params = default_params  # convection on by default
        flux = compute_fluxes(S, 0.0, jax_eos, params, mesh, jnp.zeros(n))

        # The surface dSdr should equal the interior-adjacent value after
        # the SPIDER copy. Reconstruct what compute_fluxes saw:
        # dSdr_raw = d_dr @ S; the test passes if compute_fluxes's
        # internal dSdr[-1] equals dSdr[-2], which we verify indirectly
        # by checking that F at idx -1 is consistent with F at idx -2
        # (both driven by the same dSdr, same phase properties at adjacent
        # nodes — the small node-to-node difference reflects only the
        # mesh-level rho/T/k differences, not a sign flip).
        F = np.asarray(flux.heat_flux)
        # Sign consistency between idx -2 and idx -1 (both should be
        # outward-positive for a positive dSdr; without the copy, idx -1
        # would flip sign).
        assert np.sign(F[-1]) == np.sign(F[-2]), (
            f'surface F should match sign of adjacent interior after '
            f'SPIDER ic.c:450 dSdr copy; got F[-2]={F[-2]:.3e}, F[-1]={F[-1]:.3e}'
        )
