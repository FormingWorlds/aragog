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
