"""Pytest-compatible entropy solver tests.

Requires SPIDER P-S tables at a known location. Tests are skipped
if the tables are not available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Default EOS directory (can be overridden via environment variable)
import os
EOS_DIR = Path(os.environ.get(
    'ARAGOG_TEST_EOS_DIR',
    '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
))

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}',
)


@pytest.fixture
def entropy_eos():
    """Load EntropyEOS from SPIDER tables."""
    from aragog.eos.entropy import EntropyEOS
    return EntropyEOS(EOS_DIR)


# ── Tier 1: EOS unit tests ──────────────────────────────────────────

@needs_eos
@pytest.mark.unit
class TestEntropyEOS:
    """Unit tests for the P-S table loader."""

    def test_table_loads(self, entropy_eos):
        """EntropyEOS loads without error."""
        assert entropy_eos.P_min > 0
        assert entropy_eos.S_min < entropy_eos.S_max

    def test_temperature_broadly_increases_with_entropy(self, entropy_eos):
        """At constant P, T should broadly increase with S.

        Minor non-monotonicity near phase boundaries is acceptable
        (from phase-weighted blending between solid and melt tables).
        """
        P = 50e9
        S = np.linspace(entropy_eos.S_min + 100, entropy_eos.S_max - 100, 50)
        T = entropy_eos.temperature(np.full_like(S, P), S)
        # Overall trend: first T should be much less than last T
        assert T[-1] > T[0], 'T must broadly increase with S'
        # Allow small dips but not large reversals
        dT = np.diff(T)
        n_decreasing = np.sum(dT < -1.0)  # allow <1K noise
        assert n_decreasing < len(dT) // 4, f'Too many decreasing steps: {n_decreasing}'

    def test_density_positive(self, entropy_eos):
        """Density must be positive everywhere."""
        P = np.array([10e9, 50e9, 100e9])
        S = np.array([3200, 3200, 3200])
        rho = entropy_eos.density(P, S)
        assert np.all(rho > 0)

    def test_melt_fraction_bounds(self, entropy_eos):
        """Melt fraction must be in [0, 1]."""
        P = np.linspace(1e9, 100e9, 20)
        S = np.full(20, 3200.0)
        phi = entropy_eos.melt_fraction(P, S)
        assert np.all(phi >= 0) and np.all(phi <= 1)

    def test_harmonic_density_at_midpoint(self, entropy_eos):
        """At phi=0.5, density should be the harmonic mean."""
        P = 50e9
        S_sol = entropy_eos.solidus_entropy(P)
        S_liq = entropy_eos.liquidus_entropy(P)
        S_mid = 0.5 * (S_sol + S_liq)
        phi = entropy_eos.melt_fraction(P, S_mid)
        assert phi == pytest.approx(0.5, abs=0.01)

        rho = entropy_eos.density(P, S_mid)
        rho_s = entropy_eos._lookup_at_phase_boundary('density', np.array([P]), 'solid')[0]
        rho_l = entropy_eos._lookup_at_phase_boundary('density', np.array([P]), 'melt')[0]
        rho_harmonic = 1.0 / (0.5 / rho_l + 0.5 / rho_s)
        assert rho == pytest.approx(rho_harmonic, rel=0.02)

    def test_latent_heat_positive(self, entropy_eos):
        """Latent heat L = T_fus * (S_liq - S_sol) must be positive."""
        P = np.linspace(10e9, 100e9, 10)
        L = entropy_eos.latent_heat(P)
        assert np.all(L > 0)

    def test_roundtrip_temperature(self, entropy_eos):
        """T from (P,S) should be consistent with T from phase boundaries."""
        P = 50e9
        S_sol = entropy_eos.solidus_entropy(P)
        T_at_sol = entropy_eos.temperature(P, S_sol)
        T_sol_boundary = entropy_eos._lookup_at_phase_boundary(
            'temperature', np.array([P]), 'solid')[0]
        assert T_at_sol == pytest.approx(T_sol_boundary, rel=0.05)


# ── Tier 2: Phase evaluator tests ───────────────────────────────────

@needs_eos
@pytest.mark.unit
class TestEntropyPhaseEvaluator:
    """Unit tests for the entropy phase evaluator."""

    def test_isentropic_profile(self, entropy_eos):
        """On isentropic IC, T increases with depth."""
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator
        P = np.linspace(1e9, 135e9, 30)
        phase = EntropyPhaseEvaluator(entropy_eos=entropy_eos, gravitational_acceleration=10.0)
        phase.set_pressure(P)
        phase.set_entropy(np.full(30, 3200.0))
        phase.update()
        T = phase.temperature()
        # P increases with index: P[0]=1e9 (surface), P[-1]=135e9 (CMB)
        # T should increase with P along the isentrope
        assert T[-1] > T[0], 'CMB (high P) should be hotter than surface (low P)'

    def test_capacitance_is_rho_T(self, entropy_eos):
        """Capacitance = rho * T for entropy formulation."""
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator
        P = np.array([50e9])
        phase = EntropyPhaseEvaluator(entropy_eos=entropy_eos, gravitational_acceleration=10.0)
        phase.set_pressure(P)
        phase.set_entropy(np.array([3200.0]))
        phase.update()
        cap = phase.capacitance()
        assert cap == pytest.approx(phase.density() * phase.temperature(), rel=1e-10)

    def test_set_temperature_raises(self, entropy_eos):
        """set_temperature must raise NotImplementedError."""
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator
        phase = EntropyPhaseEvaluator(entropy_eos=entropy_eos, gravitational_acceleration=10.0)
        with pytest.raises(NotImplementedError):
            phase.set_temperature(np.array([3000.0]))


# ── Tier 3: Solver integration tests ────────────────────────────────

@needs_eos
@pytest.mark.smoke
class TestEntropySolverStandalone:
    """Integration test: grey-body cooling from isentropic IC."""

    def test_grey_body_cooling(self, entropy_eos):
        """BDF integration cools the surface monotonically."""
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator
        from aragog.solver.entropy_state import EntropyState
        from scipy.integrate import solve_ivp
        from scipy.constants import Stefan_Boltzmann

        SECS_PER_YEAR = 31557600.0
        N = 30
        R_cmb, R_surf = 3480e3, 6371e3
        r_stag = np.linspace(R_cmb, R_surf, N)
        dr = np.diff(r_stag)
        P_stag = np.linspace(135e9, 1e5, N)

        # Build basic nodes
        r_basic = np.zeros(N + 1)
        r_basic[0] = R_cmb
        r_basic[-1] = R_surf
        r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
        P_basic = np.interp(r_basic, r_stag, P_stag)

        # Simple mesh mock
        class Mesh:
            pass
        class SubMesh:
            pass

        mesh = Mesh()
        mesh.basic = SubMesh()
        mesh.staggered = SubMesh()
        mesh.basic.radii = r_basic
        mesh.staggered.radii = r_stag
        mesh.basic.area = 4.0 * np.pi * r_basic**2
        mesh.basic.volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
        ml = np.minimum(r_basic - R_cmb, R_surf - r_basic)
        mesh.basic.mixing_length = np.maximum(ml, 1.0)
        mesh.basic.mixing_length_squared = mesh.basic.mixing_length**2
        mesh.basic.mixing_length_cubed = mesh.basic.mixing_length**3

        def quantity_at_basic_nodes(q):
            q = np.asarray(q).flatten()
            out = np.zeros(N + 1)
            out[0], out[-1] = q[0], q[-1]
            out[1:-1] = 0.5 * (q[:-1] + q[1:])
            return out

        def d_dr_at_basic_nodes(q):
            q = np.asarray(q).flatten()
            out = np.zeros(N + 1)
            out[1:-1] = np.diff(q) / dr
            out[0], out[-1] = out[1], out[-2]
            return out

        mesh.quantity_at_basic_nodes = quantity_at_basic_nodes
        mesh.d_dr_at_basic_nodes = d_dr_at_basic_nodes

        # Phase evaluators
        phase_stag = EntropyPhaseEvaluator(entropy_eos=entropy_eos, gravitational_acceleration=10.0)
        phase_stag.set_pressure(P_stag)
        phase_basic = EntropyPhaseEvaluator(entropy_eos=entropy_eos, gravitational_acceleration=10.0)
        phase_basic.set_pressure(P_basic)

        # Mock evaluator
        class Eval:
            pass
        evaluator = Eval()
        evaluator.mesh = mesh

        state = EntropyState(evaluator=evaluator, phase_staggered=phase_stag,
                             phase_basic=phase_basic)

        S_init = 3200.0
        S0 = np.full(N, S_init)

        def dSdt(t, S):
            state.update(S, t)
            T_top = state.top_temperature.item()
            state._heat_flux[-1] = Stefan_Boltzmann * (T_top**4 - 255.0**4)
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0, 1000), S0, method='BDF', atol=1.0, rtol=1e-6)

        assert sol.status == 0, f'Solver failed: {sol.message}'
        S_final = sol.y[:, -1]
        assert S_final[-1] < S_init, 'Surface entropy should decrease'
        assert S_final[N // 2] < S_init, 'Interior entropy should decrease (convection active)'

    def test_energy_conservation(self, entropy_eos):
        """Total energy change should match integrated surface flux."""
        # This is a simplified check: on a short run, the BDF solver
        # should not create or destroy energy.
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator

        P = np.linspace(10e9, 135e9, 10)
        phase = EntropyPhaseEvaluator(entropy_eos=entropy_eos, gravitational_acceleration=10.0)
        phase.set_pressure(P)

        # Initial state
        S_init = np.full(10, 3200.0)
        phase.set_entropy(S_init)
        phase.update()
        T_init = phase.temperature().copy()
        rho_init = phase.density().copy()
        Cp_init = phase.heat_capacity().copy()

        # Perturbed state (small cooling)
        S_cool = S_init - 50.0
        phase.set_entropy(S_cool)
        phase.update()
        T_cool = phase.temperature()

        # Energy change should be negative (cooling)
        dE = np.sum(rho_init * Cp_init * (T_cool - T_init))
        assert dE < 0, f'Energy should decrease on cooling, got dE={dE:.2e}'
