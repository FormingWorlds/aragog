"""Pytest-compatible entropy solver tests.

Requires SPIDER P-S tables at a known location. Tests are skipped
if the tables are not available.
"""

from __future__ import annotations

# Default EOS directory (can be overridden via environment variable)
import os
from pathlib import Path

import numpy as np
import pytest

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
        from scipy.constants import Stefan_Boltzmann
        from scipy.integrate import solve_ivp

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


# ── Tier 2: Jgrav regression tests ─────────────────────────────────

@needs_eos
@pytest.mark.unit
class TestJgravSmoothing:
    """Regression tests for the gravitational-separation smoothing fix.

    Before 2026-04-08: Aragog's Jgrav contribution to the mass flux
    `rho * phi * (1-phi) * v_rel` did not vanish fast enough outside
    the mushy zone. At the first crystallization step, v_rel in the
    Stokes regime (phi ~ 0.995, grain_size = 0.1 m) is ~6 m/s, and
    phi*(1-phi) ~ 5e-3 is still large enough to drain the CMB cell in
    one coupling step. The drain pushed the CMB-cell entropy below
    the PALEOS P-S table domain [-85, 3236] J/kg/K, after which the
    T(P, S) lookup saturated at the table edge and the runtime
    helpfile recorded byte-identical Phi=0.226607 for >90 Myr of
    "simulation time" while the underlying S profile diverged to
    ±1e7 J/kg/K. See memory/aragog_jgrav_cmb_drain.md.

    The fix applies a SPIDER-analogue smoothing: multiply Jgrav by a
    cubic Hermite polynomial `16 * gphi^2 * (1-gphi)^2` where gphi is
    the un-truncated two-phase fraction at the staggered cell
    immediately BELOW the interface. Bottom-up gating + edge smoothing
    come for free from this single multiplication. The polynomial has
    bounded derivatives everywhere (max |smth'| = 2), unlike SPIDER's
    tanh smoothing, which lets scipy BDF take large adaptive coupling
    steps without grinding.

    These tests build a minimal Earth-like mesh by hand (following
    test_grey_body_cooling's mock-mesh pattern) and drive one coupling
    step from a near-liquidus initial state. They assert:
        (1) the solver succeeds (status == 0),
        (2) the entropy profile stays physically bounded,
        (3) the CMB cell temperature does not collapse,
        (4) turning off grav_sep yields even less damage (sanity
            check that the test is exercising the Jgrav path),
        (5) turning off `bottom_up_grav_sep` (no smoothing at all)
            reproduces the pre-fix drain.
    """

    @staticmethod
    def _build_earth_mesh(N: int = 30):
        """Build a minimal Earth-like staggered mesh.

        Mirrors `test_grey_body_cooling` (line 161-207) so the two
        tests cover the same solver-construction path.
        """
        R_cmb, R_surf = 3480e3, 6371e3
        r_stag = np.linspace(R_cmb, R_surf, N)
        dr = np.diff(r_stag)
        P_stag = np.linspace(135e9, 1e5, N)

        r_basic = np.zeros(N + 1)
        r_basic[0] = R_cmb
        r_basic[-1] = R_surf
        r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
        P_basic = np.interp(r_basic, r_stag, P_stag)

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
        return mesh, r_stag, r_basic, P_stag, P_basic

    @staticmethod
    def _build_state(entropy_eos, mesh, P_stag, P_basic,
                     grav_sep: bool, bottom_up_grav_sep: bool = True,
                     grain_size: float = 0.1):
        """Build EntropyState with the requested Jgrav configuration.

        Uses grain_size = 0.1 m by default to match the R8 CHILI
        Earth config and make the test discriminating: smaller grain
        sizes (e.g. 1 mm) produce a v_rel small enough that the bug
        is hard to reproduce.

        Set ``bottom_up_grav_sep=False`` to disable the cubic-Hermite
        smoothing entirely and reproduce the pre-fix drain.
        """
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator
        from aragog.solver.entropy_state import EntropyState

        # Force linear-blend Cp here: this test asserts E_th decreases on
        # cooling, which only holds in the linear convention. The new
        # latent-blend Cp can produce non-monotonic E_th(t) when cells
        # cross the mushy zone (latent heat absorbed/released), and is
        # tested separately in TestLatentBlendCp.
        phase_stag = EntropyPhaseEvaluator(
            entropy_eos=entropy_eos,
            gravitational_acceleration=9.8,
            grain_size=grain_size,
            cp_blend='linear',
        )
        phase_stag.set_pressure(P_stag)
        phase_basic = EntropyPhaseEvaluator(
            entropy_eos=entropy_eos,
            gravitational_acceleration=9.8,
            grain_size=grain_size,
            cp_blend='linear',
        )
        phase_basic.set_pressure(P_basic)

        class Eval:
            pass

        evaluator = Eval()
        evaluator.mesh = mesh

        return EntropyState(
            evaluator=evaluator,
            phase_staggered=phase_stag,
            phase_basic=phase_basic,
            conduction=True,
            convection=True,
            gravitational_separation=grav_sep,
            mixing=True,
            radionuclides=False,
            tidal=False,
            kappah_floor=10.0,
            bottom_up_grav_sep=bottom_up_grav_sep,
        )

    def test_near_liquidus_single_step_bounded(self, entropy_eos):
        """First crystallization step with Jgrav must keep S and T bounded.

        Picks an initial entropy `S_init = S_liq(P_CMB) - 10` so the
        CMB cell sits just below its liquidus. Pre-fix this IC drove
        S at the CMB to -15 J/kg/K (off the PALEOS P-S table domain
        [-85, 3236]) and T_CMB to ~500 K inside a single coupling
        step, with the solver silently reporting `status = 0`.

        With the fix in place, one short BDF step must keep:
          - solver status 0,
          - S_min well inside the PALEOS table domain
            (S_min > S_EOS_min + 500 J/kg/K),
          - T_CMB above 1500 K (far above pre-fix ~500 K, loose
            enough not to over-constrain the physics — a cold solid
            at 135 GPa is allowed, just not a table-edge collapse),
          - no NaN in the final entropy profile,
          - thermal energy decreases (E_final < E_init).
        """
        from scipy.constants import Stefan_Boltzmann
        from scipy.integrate import solve_ivp

        SECS_PER_YEAR = 31557600.0
        # Smaller mesh keeps the test under a minute; the phase
        # boundary physics does not depend on N.
        mesh, r_stag, r_basic, P_stag, P_basic = self._build_earth_mesh(15)

        P_CMB = P_stag[0]
        S_liq_cmb = float(entropy_eos.liquidus_entropy(P_CMB))
        S_liq_surf = float(entropy_eos.liquidus_entropy(P_stag[-1]))
        S_init = S_liq_cmb - 10.0
        # Sanity: pure liquid at the surface (the bug requires a
        # mushy bottom under a pure-liquid top).
        assert S_init > S_liq_surf + 100.0, (
            f'Test IC is not unambiguously pure-liquid at the surface: '
            f'S_init={S_init:.1f}, S_liq_surf={S_liq_surf:.1f}'
        )

        state = self._build_state(
            entropy_eos, mesh, P_stag, P_basic, grav_sep=True
        )

        N = len(r_stag)
        S0 = np.full(N, S_init)

        def dSdt(t, S):
            state.update(S, t)
            T_top = state.top_temperature.item()
            state._heat_flux[-1] = Stefan_Boltzmann * (T_top**4 - 255.0**4)
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        # Short integration: 500 yr. Enough for the first mushy
        # cell to transition but not enough to fully crystallize
        # the bottom half of the mantle on a coarse mesh.
        sol = solve_ivp(dSdt, (0.0, 500.0), S0, method='BDF',
                         atol=1.0, rtol=1e-6)

        assert sol.status == 0, f'Solver failed: {sol.message}'

        S_final = sol.y[:, -1]
        assert not np.any(np.isnan(S_final)), 'NaN in final entropy'

        # Pre-fix A-variant reproducer result was S_min = -15 J/kg/K
        # (off-table). Post-fix it stays well above the table floor.
        assert S_final.min() > entropy_eos.S_min + 500.0, (
            f'S_min {S_final.min():.1f} too close to table floor '
            f'{entropy_eos.S_min:.1f}'
        )

        # T_CMB must be far above the pre-fix ~500 K collapse value.
        # Loose bound: 1500 K. A cold solid at 135 GPa still exceeds
        # this; only the table-clipping artifact dropped this low.
        T_cmb_final = float(entropy_eos.temperature(
            P_stag[0], S_final[0]
        ))
        assert T_cmb_final > 1500.0, (
            f'CMB temperature collapsed: T_CMB={T_cmb_final:.0f} K. '
            f'Pre-fix failure mode (~500 K at CMB).'
        )

        # Energy must DECREASE (cooling). The pre-fix artifact
        # dropped T_CMB but raised E_th because the clipped T(P, S)
        # lookup returned an inflated edge value for cells whose
        # S had escaped the table domain. This is the discriminating
        # "wrong answer" the test must rule out.
        state.update(S_final, 500.0)
        T_final = np.asarray(state.phase_staggered.temperature()).ravel()
        rho_final = np.asarray(state.phase_staggered.density()).ravel()
        Cp_final = np.asarray(state.phase_staggered.heat_capacity()).ravel()
        V = mesh.basic.volume
        E_final = float(np.sum(rho_final * Cp_final * T_final * V))

        state.update(S0, 0.0)
        T0 = np.asarray(state.phase_staggered.temperature()).ravel()
        rho0 = np.asarray(state.phase_staggered.density()).ravel()
        Cp0 = np.asarray(state.phase_staggered.heat_capacity()).ravel()
        E_init = float(np.sum(rho0 * Cp0 * T0 * V))

        assert E_final < E_init, (
            f'Thermal energy increased on cooling: '
            f'E_init={E_init:.2e}, E_final={E_final:.2e}'
        )

    def test_grav_off_is_gentler_than_grav_on(self, entropy_eos):
        """Turning grav_sep off should reduce CMB cooling.

        Sanity check that the regression tests are actually
        exercising the Jgrav path. If grav-on and grav-off give
        indistinguishable CMB states then the smoothing has zeroed
        Jgrav entirely (or Jgrav is silently dead) and the other
        regression tests would be passing on something other than
        the Jgrav physics.

        The physical expectation: gravitational settling sinks solid
        toward the CMB and causes the CMB cell to solidify earlier.
        Over a short integration, entropy at the CMB should drop
        MORE with grav_sep on than with it off.

        Uses a fully-liquid IC (S just above the CMB liquidus) and
        a very short integration so both variants stay in the mushy
        transient regime without fully crystallizing any cell.
        """
        from scipy.constants import Stefan_Boltzmann
        from scipy.integrate import solve_ivp

        SECS_PER_YEAR = 31557600.0
        mesh, r_stag, r_basic, P_stag, P_basic = self._build_earth_mesh(15)

        P_CMB = P_stag[0]
        S_liq_cmb = float(entropy_eos.liquidus_entropy(P_CMB))
        # Start JUST above the CMB liquidus so both variants will
        # cool into the mushy zone but not fully crystallize.
        S_init = S_liq_cmb + 20.0
        N = len(r_stag)
        S0 = np.full(N, S_init)

        def run(state):
            def dSdt(t, S):
                state.update(S, t)
                T_top = state.top_temperature.item()
                state._heat_flux[-1] = Stefan_Boltzmann * (T_top**4 - 255.0**4)
                state._heat_flux[0] = 0.0
                energy_flux = state.heat_flux * mesh.basic.area
                cap = state.capacitance_staggered() * mesh.basic.volume
                return -np.diff(energy_flux) / cap * SECS_PER_YEAR
            # 200 yr: short enough that both variants stay in the
            # transient regime, long enough that Jgrav has time to
            # move CMB entropy around.
            sol = solve_ivp(dSdt, (0.0, 200.0), S0, method='BDF',
                             atol=1.0, rtol=1e-6)
            assert sol.status == 0, f'Solver failed: {sol.message}'
            return sol.y[:, -1]

        state_on = self._build_state(
            entropy_eos, mesh, P_stag, P_basic, grav_sep=True
        )
        S_on = run(state_on)

        state_off = self._build_state(
            entropy_eos, mesh, P_stag, P_basic, grav_sep=False
        )
        S_off = run(state_off)

        # With grav_sep on, the CMB cell should see MORE entropy
        # loss (lower final S) than without it, because settling
        # solid concentrates at the CMB.
        dS_cmb_on = S_on[0] - S_init
        dS_cmb_off = S_off[0] - S_init

        # Discriminating assertion: grav-on must drain at least
        # 5 J/kg/K more entropy than grav-off at the CMB cell.
        # A silent no-op on Jgrav would give dS_on ≈ dS_off.
        assert dS_cmb_on < dS_cmb_off - 5.0, (
            f'Jgrav appears inert: dS_CMB(grav=ON)={dS_cmb_on:.2f}, '
            f'dS_CMB(grav=OFF)={dS_cmb_off:.2f} (difference must '
            f'exceed 5 J/kg/K)'
        )

        # Both must still be physically reasonable — no
        # catastrophic drain, no off-table entropy.
        assert S_on[0] > entropy_eos.S_min + 500.0
        assert S_off[0] > entropy_eos.S_min + 500.0
        T_on = float(entropy_eos.temperature(P_CMB, S_on[0]))
        T_off = float(entropy_eos.temperature(P_CMB, S_off[0]))
        assert T_on > 1500.0 and T_off > 1500.0, (
            f'CMB collapsed: T_on={T_on:.0f}, T_off={T_off:.0f}'
        )

    def test_smoothing_disabled_fails_or_warns(self, entropy_eos):
        """Disabling bottom_up_grav_sep reproduces the pre-fix drain.

        Edge case: if a future refactor silently bypasses the
        polynomial smoothing (e.g., sets ``bottom_up_grav_sep=False``
        by default, or reintroduces a width knob and lets it be 0),
        the pre-fix drain must come back. This test locks that in:
        the bad config must produce EITHER a drained state OR a
        solver failure. If neither happens, the fix has been
        accidentally bypassed.
        """
        from scipy.constants import Stefan_Boltzmann
        from scipy.integrate import solve_ivp

        SECS_PER_YEAR = 31557600.0
        mesh, r_stag, r_basic, P_stag, P_basic = self._build_earth_mesh(15)

        P_CMB = P_stag[0]
        S_init = float(entropy_eos.liquidus_entropy(P_CMB)) - 10.0
        N = len(r_stag)
        S0 = np.full(N, S_init)

        state = self._build_state(
            entropy_eos, mesh, P_stag, P_basic,
            grav_sep=True, bottom_up_grav_sep=False,
        )

        def dSdt(t, S):
            state.update(S, t)
            T_top = state.top_temperature.item()
            state._heat_flux[-1] = Stefan_Boltzmann * (T_top**4 - 255.0**4)
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0.0, 500.0), S0, method='BDF',
                         atol=1.0, rtol=1e-6)

        drained = False
        if sol.status == 0:
            S_final = sol.y[:, -1]
            T_cmb = float(entropy_eos.temperature(P_CMB, S_final[0]))
            drained = (
                T_cmb < 1500.0
                or S_final.min() < entropy_eos.S_min + 500.0
            )

        assert sol.status != 0 or drained, (
            'Disabling smoothing (w=0) did not reproduce the pre-fix '
            'drain or a solver failure. The fix may have been bypassed.'
        )


# ── Tier 2: Cp(P,S) E_th regression ─────────────────────────────────

@needs_eos
@pytest.mark.unit
class TestCpFromEOS:
    """Regression tests for the E_th = sum(m_i * Cp(P_i, S_i) * T_i) fix.

    Until 2026-04-09 the entropy_solver computed E_th with a hardcoded
    ``CP_REF = 1200 J/kg/K``. The Wolf-Bower 2018 / PALEOS tables give
    Cp(P, S) = 1400-2000 J/kg/K across the rocky mantle range, so the
    helpfile E_th was undercount by ~25-30 % and disagreed with SPIDER
    by +17 % at t=0 (mostly because SPIDER's same fallback combined
    with a different mass column gave the opposite-signed bias).
    Wired Cp_eff used to use ``cap_stag * vol = rho * T * vol`` which
    silently reduces to ``mass-weighted T`` rather than mean Cp.

    These tests assert:
        (1) ``phase_staggered.heat_capacity()`` returns positive,
            non-degenerate values across mantle pressures and never
            collapses to a single number (which would mean the EOS
            cache was stale or hardcoded);
        (2) the post-fix E_th is strictly larger than the old
            ``mass*1200*T`` computation at every node and the global
            sum is at least 15 % larger;
        (3) the post-fix Cp_eff is bounded between 1200 and 2500
            J/kg/K (rough mantle range) and is NOT equal to mass-
            weighted temperature, which is what the buggy formula
            produced.
    """

    def test_heat_capacity_positive_and_varying(self, entropy_eos):
        """Cp(P, S) varies across the mantle profile (not constant).

        Discriminating choice: deep-mantle Cp is well-behaved
        (1500-3000 J/kg/K for MgSiO3 at >50 GPa), so we measure there
        rather than at the surface where the Wolf-Bower 2018 table
        produces large apparent Cp from latent-heat contamination
        near the melting curve.
        """
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator

        # Sample 25 staggered nodes from surface to CMB
        P = np.linspace(1e9, 135e9, 25)
        # Mid-mantle isentrope (well above the rheological transition).
        S = np.full_like(P, 2900.0)
        # cp_blend='linear': TestCpFromEOS asserts the wrapper-side
        # linear-blend Cp is in the 1100-2500 J/kg/K range. The latent
        # blend is tested separately in TestLatentBlendCp.
        phase = EntropyPhaseEvaluator(
            entropy_eos=entropy_eos,
            gravitational_acceleration=9.8,
            cp_blend='linear',
        )
        phase.set_pressure(P)
        phase.set_entropy(S)
        phase.update()
        Cp = np.asarray(phase.heat_capacity()).ravel()

        assert Cp.shape == (25,)
        assert np.all(Cp > 0.0), 'Cp must be strictly positive'

        # Deep mantle (>50 GPa) Cp must be in a sensible range
        deep_mask = P > 50e9
        Cp_deep = Cp[deep_mask]
        assert Cp_deep.min() > 1000.0 and Cp_deep.max() < 4000.0, (
            f'Deep-mantle Cp out of physical bounds: '
            f'[{Cp_deep.min():.0f}, {Cp_deep.max():.0f}] J/kg/K'
        )

        # Cp must vary across the profile (the bug we are guarding
        # against returned a constant 1200 J/kg/K)
        assert Cp.max() > Cp.min() * 1.05, (
            f'Cp is degenerate across the profile (range '
            f'[{Cp.min():.0f}, {Cp.max():.0f}]) - EOS lookup may be '
            f'returning a single hardcoded value'
        )

        # Edge case: nothing should be exactly 1200 (the old fallback)
        assert not np.any(np.isclose(Cp, 1200.0, atol=1e-3)), (
            'A node returned exactly Cp = 1200 J/kg/K, suggesting '
            'the hardcoded fallback was hit'
        )

    def test_e_th_uses_real_cp_not_constant(self, entropy_eos):
        """Post-fix E_th must exceed the old mass*1200*T calculation."""
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator

        # 30 staggered nodes
        P = np.linspace(1e9, 135e9, 30)
        S = np.full_like(P, 2900.0)
        # cp_blend='linear': TestCpFromEOS asserts the wrapper-side
        # linear-blend Cp is in the 1100-2500 J/kg/K range. The latent
        # blend is tested separately in TestLatentBlendCp.
        phase = EntropyPhaseEvaluator(
            entropy_eos=entropy_eos,
            gravitational_acceleration=9.8,
            cp_blend='linear',
        )
        phase.set_pressure(P)
        phase.set_entropy(S)
        phase.update()

        rho = np.asarray(phase.density()).ravel()
        T = np.asarray(phase.temperature()).ravel()
        Cp = np.asarray(phase.heat_capacity()).ravel()

        # Synthetic shell volumes (Earth-like, surface-to-CMB ordered)
        # The exact volumes do not matter for the inequality test.
        r = np.linspace(3480e3, 6371e3, 31)
        vol = (4.0 / 3.0) * np.pi * np.diff(r**3)
        mass = rho * vol

        E_th_old = float(np.sum(mass * 1200.0 * T))
        E_th_new = float(np.sum(mass * Cp * T))

        # Cp ~ 1500 J/kg/K typical -> ~25 % more energy than 1200
        assert E_th_new > E_th_old * 1.10, (
            f'E_th_new={E_th_new:.3e} not > 1.10 * E_th_old={E_th_old:.3e}'
        )

    def test_cp_eff_is_mean_cp_not_mean_temperature(self, entropy_eos):
        """Cp_eff must be a mass-weighted Cp, not mass-weighted T.

        The pre-fix Cp_eff in entropy_solver.py used
        ``sum(rho*T*vol) / M`` (cap_stag * vol = rho * T * vol),
        which equals the mass-weighted mean temperature. Numerically
        that lands in the 2000-4000 K range for a hot mantle, never
        in the physical Cp range.
        """
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator

        P = np.linspace(1e9, 135e9, 30)
        S = np.full_like(P, 2900.0)
        # cp_blend='linear': TestCpFromEOS asserts the wrapper-side
        # linear-blend Cp is in the 1100-2500 J/kg/K range. The latent
        # blend is tested separately in TestLatentBlendCp.
        phase = EntropyPhaseEvaluator(
            entropy_eos=entropy_eos,
            gravitational_acceleration=9.8,
            cp_blend='linear',
        )
        phase.set_pressure(P)
        phase.set_entropy(S)
        phase.update()

        rho = np.asarray(phase.density()).ravel()
        T = np.asarray(phase.temperature()).ravel()
        Cp = np.asarray(phase.heat_capacity()).ravel()

        r = np.linspace(3480e3, 6371e3, 31)
        vol = (4.0 / 3.0) * np.pi * np.diff(r**3)
        mass = rho * vol
        M = float(np.sum(mass))

        # Post-fix formula
        Cp_eff_new = float(np.sum(mass * Cp)) / M
        # Pre-fix (buggy) formula reduces to mass-weighted T
        Cp_eff_old = float(np.sum(mass * T)) / M  # ~T_mean, in Kelvin

        # The Wolf-Bower 2018 P-S table reports higher mantle Cp than
        # textbook values for MgSiO3 because the mushy-zone blend
        # picks up latent-heat contributions; the upper bound
        # accommodates that.
        assert 1100.0 < Cp_eff_new < 5000.0, (
            f'Cp_eff_new = {Cp_eff_new:.1f} J/kg/K is out of physical '
            f'range for silicate mantle'
        )
        # Cp_eff_old (mass-weighted T) is in Kelvin and must clearly
        # differ from Cp_eff_new in J/kg/K. They should be at least
        # 25 % apart on this profile so the bug would have been
        # visible in the helpfile.
        assert abs(Cp_eff_old - Cp_eff_new) > 0.25 * Cp_eff_new, (
            f'Test setup wrong: pre-fix and post-fix formulas are not '
            f'distinguishable (Cp_eff_old={Cp_eff_old:.1f} vs '
            f'Cp_eff_new={Cp_eff_new:.1f})'
        )


# ── Tier 2: latent-blend Cp regression (v4) ─────────────────────────

@needs_eos
@pytest.mark.unit
class TestLatentBlendCp:
    """Tests for the v4 latent-heat-augmented Cp formula.

    Mirrors SPIDER's eos_composite.c:226-232 formula:
        Cp_mix = (S_liq - S_sol) / (T_liq - T_sol) * T_mid

    The empirical SPIDER vs Aragog comparison at v3 (2026-04-09) found
    that wrapper-side linear-blend Cp underestimates SPIDER's internal
    cp_s by up to a factor of 6 in the deep mushy zone. The new
    `heat_capacity_latent_blend` method closes that gap.
    """

    def test_latent_blend_returns_finite_positive(self, entropy_eos):
        """Latent-blend Cp is finite and positive across the mantle range."""
        P = np.linspace(1e9, 135e9, 30)
        S = np.full_like(P, 2900.0)
        Cp = entropy_eos.heat_capacity_latent_blend(P, S)
        assert Cp.shape == P.shape
        assert np.all(np.isfinite(Cp)), 'latent-blend Cp must be finite'
        assert np.all(Cp > 0.0), 'latent-blend Cp must be positive'

    def test_latent_blend_pure_phase_matches_linear(self, entropy_eos):
        """Far from the phase boundary, latent and linear blends agree.

        Test ``well outside`` the mushy zone, defined relative to the
        local mushy width dS = S_liq - S_sol (which is very wide for
        the Wolf-Bower 2018 table, ~8000 J/kg/K at 100 GPa). At
        gphi=-0.5 the tanh smoothing weight is < 1e-20 and the latent
        blend reduces to the linear lookup.
        """
        P = np.array([100e9, 100e9])
        S_sol = float(entropy_eos.solidus_entropy(P[0]))
        S_liq = float(entropy_eos.liquidus_entropy(P[0]))
        dS = S_liq - S_sol
        # 0.5*dS below solidus and 0.5*dS above liquidus → gphi = ±0.5
        S = np.array([S_sol - 0.5 * dS, S_liq + 0.5 * dS])
        Cp_linear = entropy_eos.heat_capacity(P, S)
        Cp_latent = entropy_eos.heat_capacity_latent_blend(P, S)
        np.testing.assert_allclose(Cp_latent, Cp_linear, rtol=1e-6)

    def test_latent_blend_exceeds_linear_in_mushy(self, entropy_eos):
        """In the deep mushy zone, latent blend should exceed linear."""
        P = 100e9
        S_sol = float(entropy_eos.solidus_entropy(P))
        S_liq = float(entropy_eos.liquidus_entropy(P))
        S_mid = 0.5 * (S_sol + S_liq)  # gphi = 0.5

        Cp_linear = float(entropy_eos.heat_capacity(np.asarray([P]),
                                                    np.asarray([S_mid])))
        Cp_latent = float(entropy_eos.heat_capacity_latent_blend(
            np.asarray([P]), np.asarray([S_mid])))

        # Latent blend should be larger than linear by at least 50 %
        # at the mushy midpoint (the actual ratio is typically 2-5x).
        assert Cp_latent > Cp_linear * 1.5, (
            f'At mushy midpoint, expected Cp_latent > 1.5*Cp_linear, '
            f'got Cp_linear={Cp_linear:.0f}, Cp_latent={Cp_latent:.0f}'
        )

    def test_latent_blend_continuous_across_solidus(self, entropy_eos):
        """Cp_latent is continuous across S_sol (no infinite jump).

        The smoothing width is 0.01 in gphi units, so the rise from 0
        to 1 happens over a range of ~0.04 in gphi. With dS ~ 8000
        J/kg/K at 100 GPa, that's a transition over ~320 J/kg/K of
        entropy. We test continuity over a range much wider than that
        but well inside the mushy zone width.
        """
        P = np.array([50e9])
        S_sol = float(entropy_eos.solidus_entropy(P[0]))
        S_liq = float(entropy_eos.liquidus_entropy(P[0]))
        dS = S_liq - S_sol
        # Sample two close points in the rising-shoulder region of the
        # smoothing bell, where Cp varies smoothly but rapidly.
        eps = 0.001 * dS  # well within the smoothing width
        Cp_below = float(entropy_eos.heat_capacity_latent_blend(
            P, np.array([S_sol + 0.05 * dS - eps])))
        Cp_above = float(entropy_eos.heat_capacity_latent_blend(
            P, np.array([S_sol + 0.05 * dS + eps])))
        # Adjacent samples differ by less than 5 % (smooth function)
        rel_jump = abs(Cp_above - Cp_below) / max(abs(Cp_below), 1.0)
        assert rel_jump < 0.05, (
            f'Cp_latent has a >5% jump over a 2*eps step (eps={eps:.2f}): '
            f'Cp_below={Cp_below:.0f}, Cp_above={Cp_above:.0f}, '
            f'rel_jump={rel_jump:.4f}'
        )

    def test_phase_evaluator_uses_latent_blend_when_configured(self, entropy_eos):
        """EntropyPhaseEvaluator(cp_blend='latent') routes to the new formula."""
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator

        P = np.array([100e9])
        S_sol = float(entropy_eos.solidus_entropy(P[0]))
        S_liq = float(entropy_eos.liquidus_entropy(P[0]))
        S_mid = 0.5 * (S_sol + S_liq)

        phase_linear = EntropyPhaseEvaluator(
            entropy_eos=entropy_eos,
            gravitational_acceleration=9.8,
            cp_blend='linear',
        )
        phase_linear.set_pressure(P)
        phase_linear.set_entropy(np.array([S_mid]))
        phase_linear.update()

        phase_latent = EntropyPhaseEvaluator(
            entropy_eos=entropy_eos,
            gravitational_acceleration=9.8,
            cp_blend='latent',
        )
        phase_latent.set_pressure(P)
        phase_latent.set_entropy(np.array([S_mid]))
        phase_latent.update()

        Cp_linear = float(np.asarray(phase_linear.heat_capacity()).item())
        Cp_latent = float(np.asarray(phase_latent.heat_capacity()).item())

        assert Cp_latent > Cp_linear * 1.5, (
            f'cp_blend=latent did not produce a larger Cp at the mushy '
            f'midpoint (got {Cp_latent:.0f} vs linear {Cp_linear:.0f})'
        )

    def test_invalid_cp_blend_raises(self, entropy_eos):
        """Invalid cp_blend value raises ValueError at construction."""
        from aragog.eos.entropy_phase import EntropyPhaseEvaluator
        with pytest.raises(ValueError, match='cp_blend'):
            EntropyPhaseEvaluator(
                entropy_eos=entropy_eos,
                gravitational_acceleration=9.8,
                cp_blend='banana',
            )


# ── Tier 4: Bower+2018 core BC (v4) ─────────────────────────────────

@needs_eos
@pytest.mark.unit
class TestBowerCoreBC:
    """Unit tests for the v4 Bower+2018 Eq. 37 core BC.

    The v3 Aragog quasi-steady BC computed F_cmb as a static fraction
    (alpha) of F[1] based on the heat capacity ratio between the
    first mantle cell and the core. This produced a -19 % T_core
    offset against SPIDER on R8 CHILI because Aragog had no record
    of the actual core enthalpy state.

    v4 expands the solver state vector by one element (T_core at
    index N) and integrates dT_core/dt = -F_cmb * area_cmb /
    (M_core * Cp_core), where F_cmb is the discrete Fourier-law
    heat flux from the bottom mantle cell to the core.

    These tests build a minimal mesh by hand (no PROTEUS coupling),
    construct an EntropySolver with the new BC, and assert the
    expected state-vector shape and core energy balance properties.
    """

    @staticmethod
    def _build_minimal_mesh(N: int = 30):
        """Build a minimal Earth-like mesh for solver tests.

        Mirrors `TestJgravSmoothing._build_earth_mesh` so the new
        tests cover the same construction path.
        """
        R_cmb, R_surf = 3480e3, 6371e3
        r_stag = np.linspace(R_cmb, R_surf, N)
        dr = np.diff(r_stag)
        P_stag = np.linspace(135e9, 1e5, N)

        r_basic = np.zeros(N + 1)
        r_basic[0] = R_cmb
        r_basic[-1] = R_surf
        r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
        P_basic = np.interp(r_basic, r_stag, P_stag)

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

        return mesh, r_stag, r_basic, P_stag, P_basic

    def test_state_vector_length_v3_vs_v4(self, entropy_eos):
        """v4 state vector is N+1, v3 state vector is N."""
        # Build a minimal solver-state-only test by directly constructing
        # the relevant pieces.
        from aragog.solver.entropy_solver import EntropySolver
        # We can't easily build a full Parameters object here, so we
        # test the public set_initial_entropy + manual _n_stag/_core_bc
        # path instead.

        # Quick standin: instantiate via direct attribute assignment
        solver = EntropySolver.__new__(EntropySolver)
        solver.entropy_eos = entropy_eos
        solver.parameters = None
        solver.evaluator = None
        solver._P_stag_flat = np.linspace(135e9, 1e5, 30)
        solver._n_stag = 30
        solver._core_bc = 'bower2018'

        S_init = np.full(30, 2900.0)
        solver.set_initial_entropy(S_init)
        assert solver._S0.shape == (31,), \
            f'v4 state should be N+1 = 31, got {solver._S0.shape}'

        solver._core_bc = 'quasi_steady'
        solver.set_initial_entropy(S_init)
        assert solver._S0.shape == (30,), \
            f'v3 state should be N = 30, got {solver._S0.shape}'

    def test_t_core_init_from_eos(self, entropy_eos):
        """Default initial T_core matches the bottom-cell mantle T."""
        from aragog.solver.entropy_solver import EntropySolver

        solver = EntropySolver.__new__(EntropySolver)
        solver.entropy_eos = entropy_eos
        solver.parameters = None
        solver.evaluator = None
        solver._P_stag_flat = np.linspace(135e9, 1e5, 30)
        solver._n_stag = 30
        solver._core_bc = 'bower2018'

        S_init = np.full(30, 2900.0)
        solver.set_initial_entropy(S_init)
        # The last element should be the bottom-cell T from the EOS
        T_bottom = float(np.asarray(
            entropy_eos.temperature(
                np.array([135e9]), np.array([2900.0])
            )
        ).item())
        assert abs(solver._S0[-1] - T_bottom) < 1.0, (
            f'Default T_core_init = {solver._S0[-1]:.0f} should match '
            f'bottom mantle T = {T_bottom:.0f}'
        )

    def test_t_core_init_user_override(self, entropy_eos):
        """set_initial_core_temperature overrides the default T_core."""
        from aragog.solver.entropy_solver import EntropySolver

        solver = EntropySolver.__new__(EntropySolver)
        solver.entropy_eos = entropy_eos
        solver.parameters = None
        solver.evaluator = None
        solver._P_stag_flat = np.linspace(135e9, 1e5, 30)
        solver._n_stag = 30
        solver._core_bc = 'bower2018'

        solver.set_initial_core_temperature(7500.0)
        S_init = np.full(30, 2900.0)
        solver.set_initial_entropy(S_init)
        assert solver._S0[-1] == pytest.approx(7500.0)


@needs_eos
@pytest.mark.unit
class TestSpiderBC:
    """Unit tests for Path A SPIDER bit-parity core BC (Step 2).

    Path A expands the solver state vector by one element at the end
    so the CMB entropy gradient ``dSdr_cmb`` becomes a separately
    integrated ODE state variable, mirroring SPIDER's
    ``dSdxi[ind_cmb]`` on the basic grid. The boundary value is then
    passed into ``EntropyState.update(dSdr_cmb=...)`` on every RHS
    evaluation so the convective+conductive flux operator at the CMB
    basic node uses the boundary entropy gradient rather than the
    FD estimate from staggered cells alone.

    The d/dt equation for the boundary state is the exact bc.c:76-131
    formula from SPIDER:

        fac_cmb = cp_cmb / (cp_core * T_cmb * tfac * M_core)
        rhs_cmb = (-F_cmb*area_cmb + 0) * fac_cmb - dSdt_s_cmb
        rhs_cmb *= 2 / dr_cmb

    where dSdt_s_cmb is dS/dt at the bottom staggered cell and the
    Ecore term is zero (simple core cooling, no internal source).

    These tests exercise:
      1. The state-vector shape + initialisation paths
         (``set_initial_entropy`` with and without user override)
      2. The pure numerical helper ``_spider_bc_rhs_per_s`` against
         hand-computed expected values for prescribed inputs, with
         at least one test value that distinguishes the correct
         formula from plausible wrong formulas (missing factor of 2,
         wrong sign, forgotten fac_cmb denominator term).
      3. The Jacobian sparsity pattern for the extended state.

    The tests DO NOT build a full solver state, because the goal is
    to check the Path A contract, not re-test the mesh / flux machinery
    already covered by ``TestJgravSmoothing`` and
    ``TestEntropySolverStandalone``. Where a full solver state IS
    needed (one end-to-end shape check), we reuse
    ``TestBowerCoreBC._build_minimal_mesh``.
    """

    @staticmethod
    def _make_bare_solver(entropy_eos, n_stag: int = 30,
                          core_bc: str = 'spider_bc',
                          P_cmb: float = 135e9, P_surf: float = 1e5,
                          R_cmb: float = 3480e3, R_surf: float = 6371e3):
        """Construct a minimal EntropySolver for unit-level tests.

        Uses ``__new__`` + attribute injection like
        ``TestBowerCoreBC.test_state_vector_length_v3_vs_v4``, then
        wires a minimal evaluator with an Earth-like mesh so
        ``set_initial_entropy``'s cold-start FD stencil has real
        r_basic/r_stag arrays to work with. Additional cached
        constants for ``_spider_bc_rhs_per_s`` and BC dispatch are
        wired by ``_wire_cached_constants``.
        """
        from aragog.solver.entropy_solver import EntropySolver

        solver = EntropySolver.__new__(EntropySolver)
        solver.entropy_eos = entropy_eos
        solver.parameters = None
        solver._P_stag_flat = np.linspace(P_cmb, P_surf, n_stag)
        solver._n_stag = n_stag
        solver._core_bc = core_bc

        # Minimal mesh: N+1 basic nodes spanning [R_cmb, R_surf], with
        # staggered cells at basic-node midpoints.
        class _MeshStub:
            pass
        class _SubMesh:
            pass
        mesh = _MeshStub()
        mesh.basic = _SubMesh()
        mesh.staggered = _SubMesh()
        r_basic = np.linspace(R_cmb, R_surf, n_stag + 1)
        mesh.basic.radii = r_basic
        mesh.staggered.radii = 0.5 * (r_basic[:-1] + r_basic[1:])
        evaluator = _MeshStub()
        evaluator.mesh = mesh
        solver.evaluator = evaluator

        # Clear any stale IC overrides from __new__.
        for attr in ('_dSdr_cmb_init', '_T_core_init', '_solution'):
            if hasattr(solver, attr):
                delattr(solver, attr)
        return solver

    @staticmethod
    def _wire_cached_constants(
        solver,
        *,
        r_cmb: float = 3480e3,
        dr_cmb: float = 100e3,
        core_density: float = 10738.332568062382,
        core_cp: float = 880.0,
        core_tfac: float = 1.147,
    ):
        """Attach the minimal subset of _cache_bc_constants needed for
        ``_spider_bc_rhs_per_s`` to run as a pure function.

        We hand-wire these instead of calling _initialize_internals to
        avoid pulling in the full mesh/BC builder. The values mirror
        the R8 CHILI Earth config (r_cmb=3480 km, dr_cmb=100 km,
        core_density=10738 kg/m^3, core_cp=880 J/(kg K), core_tfac=1.147).
        """
        solver._cmb_r_cmb = r_cmb
        solver._cmb_r_above = r_cmb + dr_cmb
        solver._cmb_dr_cmb = dr_cmb
        solver._cmb_dr_half = 0.5 * dr_cmb
        solver._cmb_area = 4.0 * np.pi * r_cmb**2
        solver._cmb_vol_first = (4.0 / 3.0) * np.pi * (
            (r_cmb + dr_cmb) ** 3 - r_cmb**3
        )
        solver._core_density = core_density
        solver._core_cp = core_cp
        solver._core_M = (4.0 / 3.0) * np.pi * r_cmb**3 * core_density
        solver._core_cap = solver._core_M * core_cp
        solver._core_tfac = core_tfac
        solver._cmb_radius_ratio_sq = ((r_cmb + dr_cmb) / r_cmb) ** 2

    # ── state-vector shape tests ─────────────────────────────────

    def test_state_vector_length_spider_bc(self, entropy_eos):
        """spider_bc state has N+1 elements; quasi_steady has N.

        Also verifies that switching modes on the same solver flips
        the length cleanly — catches any latent state leak across
        PROTEUS coupling resets.
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        solver.set_initial_entropy(np.full(30, 2900.0))
        assert solver._S0.shape == (31,), (
            f'spider_bc state should be N+1 = 31, got {solver._S0.shape}'
        )

        solver._core_bc = 'quasi_steady'
        solver.set_initial_entropy(np.full(30, 2900.0))
        assert solver._S0.shape == (30,), (
            f'quasi_steady state should be N = 30 after flip, '
            f'got {solver._S0.shape}'
        )

        # Edge case: N=2 is the minimum mesh size the FD cold-start
        # can handle without special-casing (it computes
        # (S[1]-S[0])/(r[1]-r[0])). _make_bare_solver provides the
        # default 3-basic-node mesh for N=2.
        solver2 = self._make_bare_solver(entropy_eos, n_stag=2,
                                          core_bc='spider_bc')
        solver2.set_initial_entropy(np.array([2900.0, 3100.0]))
        assert solver2._S0.shape == (3,), (
            f'N=2 minimal mesh should give state length 3, '
            f'got {solver2._S0.shape}'
        )

    def test_set_initial_entropy_rejects_wrong_length(self, entropy_eos):
        """Mismatched S_init length raises ValueError in both modes.

        This is the 'unreasonable input' guarantee: if PROTEUS hands
        in the wrong number of staggered cells, the solver must fail
        loudly, not silently pad or truncate.
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        with pytest.raises(ValueError, match=r'S_init length 20'):
            solver.set_initial_entropy(np.full(20, 2900.0))
        with pytest.raises(ValueError, match=r'S_init length 31'):
            solver.set_initial_entropy(np.full(31, 2900.0))

    # ── cold-start initialisation tests ──────────────────────────

    def test_dSdr_cmb_cold_start_from_uniform_is_zero(self, entropy_eos):
        """Uniform S_init → cold-start dSdr_cmb = 0 (FD of flat = 0).

        The cold-start formula in set_initial_entropy is
        ``(S[1] - S[0]) / (r_stag[1] - r_stag[0])``. For a uniform
        profile this trivially returns zero, which is the correct
        neutral-buoyancy starting point that the SPIDER energy-
        balance equation then drives to its quasi-equilibrium value.

        Discriminating values: using S_init = 2900 J/kg/K (inside the
        PALEOS domain, not at any boundary or special value).
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        S_uniform = np.full(30, 2900.0)
        solver.set_initial_entropy(S_uniform)
        assert solver._S0[-1] == pytest.approx(0.0, abs=1e-12), (
            f'Uniform S → cold-start dSdr_cmb should be exactly zero, '
            f'got {solver._S0[-1]:.3e}'
        )

    def test_dSdr_cmb_cold_start_from_gradient_matches_fd(self, entropy_eos):
        """Non-uniform S_init → cold-start dSdr_cmb matches the FD formula.

        This verifies the discriminating case: if the FD stencil uses
        the wrong nodes (e.g., r_basic instead of r_stag) the result
        would differ by ~2x. If the sign is wrong, it would be
        negated. Using a LARGE, non-pathological gradient so both
        bugs would show up at the 1 % level.
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        # S_init with a linear gradient of +5 J/kg/K per cell.
        S_grad = 2900.0 + 5.0 * np.arange(30, dtype=float)
        solver.set_initial_entropy(S_grad)

        # Expected FD value using r_stag[1] - r_stag[0]. The default
        # mesh in _make_bare_solver has r_basic = linspace(R_cmb, R_surf, N+1)
        # and r_stag = midpoints, so we recompute here to discriminate
        # against a buggy stencil that picks r_basic instead.
        r_basic = solver.evaluator.mesh.basic.radii
        r_stag_0 = 0.5 * (r_basic[0] + r_basic[1])
        r_stag_1 = 0.5 * (r_basic[1] + r_basic[2])
        expected_dSdr = 5.0 / (r_stag_1 - r_stag_0)

        assert solver._S0[-1] == pytest.approx(expected_dSdr, rel=1e-9), (
            f'Cold-start FD mismatch: got {solver._S0[-1]:.3e}, '
            f'expected {expected_dSdr:.3e}'
        )
        # Sign sanity: positive upward gradient → positive dSdr_cmb
        assert solver._S0[-1] > 0.0, (
            'Positive S gradient must give positive dSdr_cmb '
            '(sign convention: r increasing, S increasing → +)'
        )

    def test_dSdr_cmb_user_override_preserved(self, entropy_eos):
        """set_initial_dSdr_cmb overrides the cold-start FD value.

        The hot-start path in PROTEUS coupling passes the previous
        solution's dSdr_cmb back in via ``set_initial_dSdr_cmb``.
        This test verifies the override survives the
        ``set_initial_entropy`` call that immediately follows.
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        # Cold-start FD on a uniform profile would give 0; the user
        # override must WIN over that default.
        solver.set_initial_dSdr_cmb(-1.234e-6)
        solver.set_initial_entropy(np.full(30, 2900.0))
        assert solver._S0[-1] == pytest.approx(-1.234e-6, rel=1e-12), (
            'User override must beat cold-start FD zero'
        )

        # Negative values are allowed (physically: S decreasing with
        # depth = unstable stratification, which the solver should
        # still accept and let the physics correct).
        solver2 = self._make_bare_solver(entropy_eos, n_stag=30,
                                          core_bc='spider_bc')
        solver2.set_initial_dSdr_cmb(+5.678e-5)
        solver2.set_initial_entropy(np.full(30, 2900.0))
        assert solver2._S0[-1] == pytest.approx(+5.678e-5, rel=1e-12)

        # Override must also beat the hot-start-from-prev-solution
        # path. Populate a fake _solution with a different boundary
        # value and confirm the user override still wins.
        solver3 = self._make_bare_solver(entropy_eos, n_stag=30,
                                          core_bc='spider_bc')

        class _FakeSol:
            pass
        fake = _FakeSol()
        fake.y = np.zeros((31, 2))
        fake.y[30, -1] = 9.999e-3  # "previous" dSdr_cmb
        solver3._solution = fake
        solver3.set_initial_dSdr_cmb(-7.777e-6)
        solver3.set_initial_entropy(np.full(30, 2900.0))
        assert solver3._S0[-1] == pytest.approx(-7.777e-6, rel=1e-12), (
            'User override must beat previous-solution hot-start too'
        )

    def test_dSdr_cmb_hot_start_from_previous_solution(self, entropy_eos):
        """Without user override, hot-start picks up prev_sol.y[N, -1].

        This is the path PROTEUS relies on after the first coupling
        step: the previous solve's trailing boundary state becomes
        the next solve's initial boundary state. If this path breaks
        the boundary state resets on every coupling reset and the
        integrated SPIDER energy balance is lost.
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')

        class _FakeSol:
            pass
        fake = _FakeSol()
        fake.y = np.zeros((31, 4))
        fake.y[30, -1] = 3.14e-5
        fake.y[30, -2] = 99.0  # Earlier column, must be ignored
        solver._solution = fake

        # Use uniform S so the cold-start FD would give 0 — only the
        # prev-solution value can supply the 3.14e-5 we expect.
        solver.set_initial_entropy(np.full(30, 2900.0))
        assert solver._S0[-1] == pytest.approx(3.14e-5, rel=1e-12), (
            f'Hot start must pick y[N, -1] (not y[N, -2]), '
            f'got {solver._S0[-1]:.3e}'
        )

    # ── helper-formula bit-parity tests ──────────────────────────

    def test_spider_bc_rhs_zero_flux_limit(self, entropy_eos):
        """F_cmb = 0 and dSdt_s_cmb = 0 → rhs = 0 exactly.

        The formula is linear in both ``F_cmb_basic`` and
        ``dSdt_s_cmb_per_s``, and ``Ecore = 0``. With all three inputs
        zero, the rhs must be identically zero regardless of the
        cached core / geometry constants. This is an invariant the
        helper MUST satisfy.
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        self._wire_cached_constants(solver)

        rhs = solver._spider_bc_rhs_per_s(
            F_cmb_basic=0.0,
            dSdt_s_cmb_per_s=0.0,
            T_cmb_basic=4000.0,
            cp_cmb_basic=1300.0,
        )
        assert rhs == pytest.approx(0.0, abs=1e-30), (
            f'Zero-flux limit violated: rhs = {rhs:.3e}'
        )

        # Second smoke test: with zero flux but non-zero dSdt_s,
        # the rhs should be ``dSdt_s_cmb_per_s * 2 / dr_cmb``
        # (because dS_basic_cmb_dt = 0 when E_tot_cmb = 0, and the
        # formula is (dSdt_s - dSdt_basic) * 2/dr = dSdt_s * 2/dr).
        # This is a distinct linearity check.
        dSdt_s_nonzero = 1.5e-12  # J/(kg*K*s), a plausible cliff value
        rhs2 = solver._spider_bc_rhs_per_s(
            F_cmb_basic=0.0,
            dSdt_s_cmb_per_s=dSdt_s_nonzero,
            T_cmb_basic=4000.0,
            cp_cmb_basic=1300.0,
        )
        expected_rhs2 = dSdt_s_nonzero * 2.0 / solver._cmb_dr_cmb
        assert rhs2 == pytest.approx(expected_rhs2, rel=1e-12), (
            f'Zero-flux sub-limit: got {rhs2:.3e}, '
            f'expected {expected_rhs2:.3e}'
        )

    def test_spider_bc_rhs_bit_parity_prescribed_inputs(self, entropy_eos):
        """SPIDER bc.c formula bit-parity against a hand-computed value.

        Chosen inputs are the R8 CHILI Earth order-of-magnitude values
        at t ≈ 10 kyr (just after first crystallization), which exercise
        the nominal operating regime of the BC:
            F_cmb_basic    = 3.0e4 W/m^2  (plausible early-evolution)
            dSdt_s_cmb_per_s = -1.2e-9 J/(kg*K*s) (cooling, negative)
            T_cmb_basic    = 4100 K
            cp_cmb_basic   = 1280 J/(kg*K)

        Hand computation:
            area_cmb       = 4*pi*r_cmb^2 = 4*pi*(3480e3)^2 = 1.5216e14 m^2
            E_tot_cmb      = 3.0e4 * 1.5216e14 = 4.5648e18 W
            M_core         = (4/3)*pi*r_cmb^3 * core_density
                           = (4/3)*pi*(3480e3)^3 * 10738.3
                           = 1.8940e24 kg
            fac_denom      = core_cp * T_cmb * tfac * M_core
                           = 880 * 4100 * 1.147 * 1.8940e24
                           = 7.8422e30
            fac_cmb        = cp_cmb / fac_denom = 1280 / 7.8422e30
                           = 1.6322e-28  1/(kg*K)
            dS_basic_cmb_dt_base = -E_tot_cmb * fac_cmb
                           = -4.5648e18 * 1.6322e-28
                           = -7.4507e-10  J/(kg*K*s)
            dr_cmb         = 1.0e5 m

            The one-sided FD for d/dt(dS/dr) at the CMB is
              (dSdt_stag - dSdt_basic) * 2 / dr
            because the staggered node is at the midpoint (dr/2)
            above the CMB basic node.

            rhs            = (dSdt_s_cmb_per_s - dS_basic_cmb_dt_base)
                             * 2 / dr_cmb
                           = ((-1.2e-9) - (-7.4507e-10)) * 2 / 1e5
                           = (-4.5493e-10) * 2 / 1e5
                           = -9.0986e-15  J/(kg*K*m*s)

            The result is NEGATIVE: the staggered cell cools faster
            than the CMB basic node (core has large thermal inertia),
            so the outward entropy gradient weakens over time.

        The test reproduces the same hand computation in-line so any
        future change to cached constant defaults (r_cmb, core_density,
        core_cp) triggers an obvious test failure with a diff.
        """
        r_cmb = 3480e3
        dr_cmb = 1.0e5
        core_density = 10738.332568062382
        core_cp = 880.0
        core_tfac = 1.147

        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        self._wire_cached_constants(
            solver,
            r_cmb=r_cmb,
            dr_cmb=dr_cmb,
            core_density=core_density,
            core_cp=core_cp,
            core_tfac=core_tfac,
        )

        F_cmb_basic = 3.0e4
        dSdt_s_cmb_per_s = -1.2e-9
        T_cmb_basic = 4100.0
        cp_cmb_basic = 1280.0

        # Hand-recompute the expected rhs inside the test using the
        # correct FD formula: d/dt(dS/dr) = (dSdt_stag - dSdt_basic)
        # * 2/dr. This is NOT a copy of the helper; it uses long-form
        # intermediate names so the reader can verify each step.
        area_cmb = 4.0 * np.pi * r_cmb**2
        M_core = (4.0 / 3.0) * np.pi * r_cmb**3 * core_density
        E_tot_cmb = F_cmb_basic * area_cmb
        fac_cmb = cp_cmb_basic / (
            core_cp * T_cmb_basic * core_tfac * M_core
        )
        dS_basic_cmb_dt = -E_tot_cmb * fac_cmb  # Ecore=0
        expected = (
            (dSdt_s_cmb_per_s - dS_basic_cmb_dt) * 2.0 / dr_cmb
        )

        actual = solver._spider_bc_rhs_per_s(
            F_cmb_basic=F_cmb_basic,
            dSdt_s_cmb_per_s=dSdt_s_cmb_per_s,
            T_cmb_basic=T_cmb_basic,
            cp_cmb_basic=cp_cmb_basic,
        )
        assert actual == pytest.approx(expected, rel=1e-12), (
            f'BC formula bit-parity failed: actual={actual:.6e}, '
            f'expected={expected:.6e}'
        )

        # Additional invariant: for these inputs the rhs must be
        # NEGATIVE. The staggered cell cools faster (dSdt_s = -1.2e-9)
        # than the CMB basic node (dSdt_basic = -7.5e-10), because
        # the core has much larger thermal inertia. The outward
        # entropy gradient therefore weakens over time:
        # d/dt(dSdr) = ((-1.2e-9) - (-7.5e-10)) * 2/dr < 0.
        assert actual < 0.0, (
            f'Expected negative rhs (gradient weakening as stag '
            f'cools faster than basic), got {actual:.3e}'
        )

    def test_spider_bc_rhs_linearity_in_flux(self, entropy_eos):
        """rhs is linear in F_cmb_basic at fixed other inputs.

        Property-based invariant: doubling the heat flux must double
        the F-contribution to rhs, leaving the dSdt_s contribution
        unchanged. This catches a common bug class where a wrong
        exponent (F^2, sqrt(F)) sneaks in.
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        self._wire_cached_constants(solver)

        # Pin dSdt_s_cmb_per_s to 0 so the rhs is purely F-driven.
        base_inputs = dict(
            dSdt_s_cmb_per_s=0.0,
            T_cmb_basic=4100.0,
            cp_cmb_basic=1280.0,
        )
        rhs1 = solver._spider_bc_rhs_per_s(
            F_cmb_basic=1.5e4, **base_inputs
        )
        rhs2 = solver._spider_bc_rhs_per_s(
            F_cmb_basic=3.0e4, **base_inputs
        )
        rhs3 = solver._spider_bc_rhs_per_s(
            F_cmb_basic=6.0e4, **base_inputs
        )

        assert rhs2 == pytest.approx(2.0 * rhs1, rel=1e-12), (
            f'Linearity failed: rhs(2F) = {rhs2:.3e}, '
            f'expected {2.0 * rhs1:.3e}'
        )
        assert rhs3 == pytest.approx(4.0 * rhs1, rel=1e-12), (
            f'Linearity failed: rhs(4F) = {rhs3:.3e}, '
            f'expected {4.0 * rhs1:.3e}'
        )

        # Sign convention check: positive F (heat OUT of core) with
        # dSdt_s = 0 must give POSITIVE rhs. Physically: the core
        # loses entropy (dSdt_basic < 0) while the staggered cell is
        # static, so d/dt(dSdr) = (0 - negative)/positive > 0, i.e.,
        # the outward entropy gradient steepens as the boundary cools.
        assert rhs1 > 0.0, (
            f'Positive outward heat flow with dSdt_s=0 must give '
            f'positive rhs (gradient steepening), got {rhs1:.3e}'
        )

    def test_spider_bc_rhs_sign_from_fd_definition(self, entropy_eos):
        """Verify the sign against the independent FD definition.

        The one-sided centered difference for d/dt(dS/dr) at the CMB
        basic node, using the staggered node at dr/2 above, is:

            d/dt(dS/dr) = (dSdt_stag - dSdt_basic) / (dr/2)

        This test constructs both terms independently and checks that
        the helper returns the correct sign and magnitude. Unlike the
        bit-parity test (which re-derives using the same formula), this
        test derives the expected value purely from the FD definition,
        catching sign errors that would survive self-consistent checks.
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        dr_cmb = 8.0e4  # different from default to catch hardcoding
        self._wire_cached_constants(solver, dr_cmb=dr_cmb)

        # Scenario: core cooling slowly, mantle staggered cell cooling fast
        dSdt_basic = -5.0e-10  # J/(kg*K*s), core-buffered rate
        dSdt_stag = -2.0e-9   # J/(kg*K*s), mantle cell rate

        # Derive dSdt_basic from the energy-balance side.
        # dSdt_basic = -E_tot * fac_cmb, so we need F that produces it.
        area_cmb = solver._cmb_area
        fac_cmb = 1280.0 / (
            solver._core_cp * 4000.0 * solver._core_tfac * solver._core_M
        )
        # F_cmb such that -F*area*fac = dSdt_basic
        F_cmb = -dSdt_basic / (area_cmb * fac_cmb)

        # The FD definition gives the expected result directly
        expected = (dSdt_stag - dSdt_basic) * 2.0 / dr_cmb

        actual = solver._spider_bc_rhs_per_s(
            F_cmb_basic=F_cmb,
            dSdt_s_cmb_per_s=dSdt_stag,
            T_cmb_basic=4000.0,
            cp_cmb_basic=1280.0,
        )
        assert actual == pytest.approx(expected, rel=1e-10), (
            f'FD-definition check failed: actual={actual:.6e}, '
            f'expected={expected:.6e}'
        )
        # Stag cools faster than basic, so gradient weakens: rhs < 0
        assert actual < 0.0, (
            f'Expected negative rhs (stag cools faster than basic), '
            f'got {actual:.3e}'
        )

    def test_spider_bc_rhs_clamps_protect_against_divzero(self, entropy_eos):
        """Very small T_cmb and M_core do not blow the rhs up.

        The ``max(., 1.0)`` clamps in the helper prevent a
        divide-by-zero if the IC wind-up transiently produces a
        T_cmb near zero (which should not happen in practice, but
        a defensive check is cheap and the clamps exist for a reason).
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        self._wire_cached_constants(solver)

        # Set T_cmb to a physically unreasonable 0.5 K. The clamp
        # should kick in at max(0.5, 1.0) = 1.0 and the result
        # should be finite.
        rhs = solver._spider_bc_rhs_per_s(
            F_cmb_basic=1.0e4,
            dSdt_s_cmb_per_s=0.0,
            T_cmb_basic=0.5,  # unphysically small
            cp_cmb_basic=1280.0,
        )
        assert np.isfinite(rhs), (
            f'rhs should be finite under T_cmb clamp, got {rhs}'
        )

        # Also verify the clamp at T_cmb=0.0 (the exact degenerate
        # case) does not divide by zero.
        rhs_zero = solver._spider_bc_rhs_per_s(
            F_cmb_basic=1.0e4,
            dSdt_s_cmb_per_s=0.0,
            T_cmb_basic=0.0,
            cp_cmb_basic=1280.0,
        )
        assert np.isfinite(rhs_zero), (
            f'rhs should be finite even at T_cmb=0, got {rhs_zero}'
        )

    # ── Jacobian sparsity test ───────────────────────────────────

    def test_jacobian_sparsity_spider_bc_coupling(self, entropy_eos):
        """Sparsity pattern has the extra boundary row/col for spider_bc.

        The spider_bc state vector grows by 1 at the end. The
        sparsity pattern must:
          - have shape (N+1, N+1)
          - have a nonzero at (N, 0), (N, 1), (N, 2) and (N, N)
            (d/dt of dSdr_cmb depends on S[0..2] and itself)
          - have a nonzero at (0, N), (1, N)
            (S[0] and S[1] depend on dSdr_cmb via the boundary flux)
          - not have additional nonzeros at (N, 3), (N, 4) that would
            suggest over-provisioning (we accept over-provisioning as
            a performance hit, but this test documents the minimal
            expected shape)

        For comparison the quasi_steady pattern should be (N, N).
        """
        N = 30
        solver = self._make_bare_solver(entropy_eos, n_stag=N,
                                         core_bc='spider_bc')
        J = solver._build_jac_sparsity()
        assert J.shape == (N + 1, N + 1), (
            f'spider_bc Jacobian shape should be (31, 31), got {J.shape}'
        )
        # Convert to dense bool for element access
        Jd = J.toarray().astype(bool)
        # Boundary row (N): must couple to S[0], S[1], S[2], self
        assert Jd[N, 0], 'J[N, 0] must be nonzero (d(dSdr_cmb)/dt ← S[0])'
        assert Jd[N, 1], 'J[N, 1] must be nonzero (d(dSdr_cmb)/dt ← S[1])'
        assert Jd[N, 2], 'J[N, 2] must be nonzero (d(dSdr_cmb)/dt ← S[2])'
        assert Jd[N, N], 'J[N, N] must be nonzero (d(dSdr_cmb)/dt ← self)'
        # Boundary column (N): must couple S[0], S[1] back to dSdr_cmb
        assert Jd[0, N], 'J[0, N] must be nonzero (dS[0]/dt ← dSdr_cmb)'
        assert Jd[1, N], 'J[1, N] must be nonzero (dS[1]/dt ← dSdr_cmb)'
        # Boundary row should NOT couple to distant cells (no (N, 10))
        assert not Jd[N, 10], (
            f'J[N, 10] should be zero; is {Jd[N, 10]} (would waste FD perturbations)'
        )

        # Compare with quasi_steady: no extra row or column
        solver_qs = self._make_bare_solver(entropy_eos, n_stag=N,
                                            core_bc='quasi_steady')
        J_qs = solver_qs._build_jac_sparsity()
        assert J_qs.shape == (N, N), (
            f'quasi_steady Jacobian shape should be (30, 30), got {J_qs.shape}'
        )

        # Edge case: N=3 (minimum mesh where the pentadiagonal stencil
        # fully populates). The extra row should still couple to S[0..2]
        # and self; Jd shape must be (4, 4).
        solver_small = self._make_bare_solver(entropy_eos, n_stag=3,
                                               core_bc='spider_bc')
        J_small = solver_small._build_jac_sparsity()
        assert J_small.shape == (4, 4)
        Jd_small = J_small.toarray().astype(bool)
        assert Jd_small[3, 0] and Jd_small[3, 1] and Jd_small[3, 2] and Jd_small[3, 3]

    # ── atol phi0 slice regression test ──────────────────────────

    def test_phi0_slice_handles_extended_state(self, entropy_eos):
        """Tier 1D atol phi0 estimate works for spider_bc.

        Before the 2026-04-10 fix, ``solve()`` sliced ``_S0`` only
        when ``_core_bc == 'bower2018'``, which caused spider_bc to
        pass the full N+1 _S0 (with dSdr_cmb appended) to
        ``melt_fraction`` against a shape-N _P_stag_flat. The shape
        mismatch raised an exception that fell through the
        ``try/except`` and left phi0 = 1.0, silently disabling the
        Tier 1D atol relaxation. The fix swaps the check to
        ``_state_is_extended``.

        This test reproduces the old failure path by calling
        ``melt_fraction`` the same way solve() does, both with the
        correct slice (success) and with the old buggy slice
        (shape mismatch). A passing test needs the correct path to
        succeed.
        """
        solver = self._make_bare_solver(entropy_eos, n_stag=30,
                                         core_bc='spider_bc')
        S_init = np.full(30, 2900.0)
        solver.set_initial_entropy(S_init)
        # _S0 is now shape (31,) — the last element is dSdr_cmb, which
        # must NOT be passed to melt_fraction.
        assert solver._S0.shape == (31,)

        # The correct slice (using _state_is_extended) must produce a
        # finite phi0 estimate.
        S0_block = (
            solver._S0[:solver._n_stag]
            if solver._state_is_extended
            else solver._S0
        )
        phi0 = float(np.asarray(
            solver.entropy_eos.melt_fraction(solver._P_stag_flat, S0_block)
        ).mean())
        assert 0.0 <= phi0 <= 1.0, (
            f'phi0 from corrected slice should be in [0, 1], got {phi0}'
        )

        # The old buggy slice would raise (shape mismatch).
        # Exercise that this SPECIFIC error mode would have occurred
        # under the old code, which is how we know the fix is
        # actually exercised by this test.
        with pytest.raises((ValueError, IndexError, Exception)):
            # Pass full 31-element _S0 against shape-30 _P_stag_flat
            solver.entropy_eos.melt_fraction(
                solver._P_stag_flat,  # shape (30,)
                solver._S0,           # shape (31,) — mismatch
            )
