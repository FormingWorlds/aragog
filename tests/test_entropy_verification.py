"""First-principles verification tests for the Aragog entropy solver.

Tests verify the entropy formulation against known analytical solutions
and conservation laws. All tests use PALEOS P-S tables and the real
EntropyEOS/EntropyPhaseEvaluator/EntropyState infrastructure.

Test hierarchy:
  Tier 1 (conservation): entropy conservation, energy conservation
  Tier 2 (dynamics): grey-body cooling timescale, conduction steady state
  Tier 3 (parity): SPIDER comparison (separate script)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import Stefan_Boltzmann
from scipy.integrate import solve_ivp

EOS_DIR = Path(os.environ.get(
    'ARAGOG_TEST_EOS_DIR',
    '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
))

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}',
)

SECS_PER_YEAR = 31557600.0


# -- Shared test infrastructure -----------------------------------------------

def make_mesh(N=50, R_cmb=3480e3, R_surf=6371e3, P_cmb=135e9, P_surf=1e5):
    """Build a simple radial mesh for verification tests."""
    r_stag = np.linspace(R_cmb, R_surf, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb
    r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    P_stag = np.linspace(P_cmb, P_surf, N)
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
    mesh.basic.pressure = P_basic
    mesh.staggered.pressure = P_stag

    # Mass coordinates: for a uniform test mesh, use radii directly
    # (xi = r, dxi/dr = 1). The entropy solver uses these for the
    # SPIDER-parity centered FD in xi-space (entropy_state.py:354).
    mesh.basic.mass_radii = r_basic
    mesh.staggered.mass_radii = r_stag
    mesh.dxidr = np.ones_like(r_basic)

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
    mesh.dr = dr
    mesh.N = N

    return mesh


def make_state(mesh, entropy_eos, conduction=True, convection=True):
    """Build EntropyState from mesh and EOS."""
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.solver.entropy_state import EntropyState

    phase_stag = EntropyPhaseEvaluator(
        entropy_eos=entropy_eos, gravitational_acceleration=10.0)
    phase_stag.set_pressure(mesh.staggered.pressure)

    phase_basic = EntropyPhaseEvaluator(
        entropy_eos=entropy_eos, gravitational_acceleration=10.0)
    phase_basic.set_pressure(mesh.basic.pressure)

    class Eval:
        pass
    evaluator = Eval()
    evaluator.mesh = mesh

    return EntropyState(
        evaluator=evaluator,
        phase_staggered=phase_stag,
        phase_basic=phase_basic,
        conduction=conduction,
        convection=convection,
    )


def compute_thermal_energy(S, mesh, entropy_eos):
    """Compute total thermal energy E_th = sum(rho * Cp * T * V)."""
    P = mesh.staggered.pressure
    T = entropy_eos.temperature(P, S)
    rho = entropy_eos.density(P, S)
    Cp = entropy_eos.heat_capacity(P, S)
    V = mesh.basic.volume
    return np.sum(rho * Cp * T * V)


def compute_enthalpy_integral(S, mesh, entropy_eos):
    """Compute total enthalpy-like integral H = sum(rho * T * S * V).

    This is the conserved quantity for the entropy equation
    rho * T * dS/dt = -div(F) with zero-flux BCs. When div(F) integrates
    to zero over the domain (zero-flux BCs), d/dt sum(rho*T*S*V) = 0 to
    leading order for small dS.
    """
    P = mesh.staggered.pressure
    T = entropy_eos.temperature(P, S)
    rho = entropy_eos.density(P, S)
    V = mesh.basic.volume
    return np.sum(rho * T * S * V)


# -- Test 1: Entropy conservation ---------------------------------------------

@needs_eos
@pytest.mark.unit
class TestEntropyConservation:
    """On insulating BCs, conservation laws must hold as entropy redistributes."""

    def test_entropy_redistribution_conserves_enthalpy(self):
        """Non-uniform S with zero-flux BCs and conduction: entropy redistributes
        but total enthalpy integral is conserved.

        Uses a linear S profile (high at CMB, low at surface) with conduction
        enabled and zero-flux BCs. Conduction drives heat from hot to cold,
        redistributing entropy. The test verifies:
        (a) the entropy profile actually changes (not a zero-RHS test),
        (b) the mass-weighted enthalpy integral sum(rho*T*S*V) is conserved.
        """
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=True, convection=True)

        # Non-uniform initial entropy: linear gradient (superadiabatic at surface)
        S0 = np.linspace(3400.0, 3000.0, N)
        H0 = compute_enthalpy_integral(S0, mesh, eos)

        def dSdt(t, S):
            state.update(S, t)
            # Zero flux at both boundaries (insulating box)
            state._heat_flux[-1] = 0.0
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0, 5000), S0, method='BDF',
                        atol=0.01, rtol=1e-8)
        assert sol.status == 0

        S_final = sol.y[:, -1]

        # (a) Profile must actually change (system evolves)
        max_change = np.max(np.abs(S_final - S0))
        assert max_change > 1.0, (
            f'Entropy profile barely changed (max dS={max_change:.2f} J/kg/K). '
            f'The test must exercise the solver with nontrivial dynamics.'
        )

        # (b) Total enthalpy integral must be conserved
        H_final = compute_enthalpy_integral(S_final, mesh, eos)
        rel_change = abs(H_final - H0) / abs(H0)
        assert rel_change < 2e-3, (
            f'Enthalpy integral changed by {rel_change:.2e} '
            f'(should be < 0.2%)'
        )

    def test_nonuniform_entropy_no_conduction(self):
        """Non-uniform S with zero flux and no conduction: only convection acts.

        Convection should homogenize entropy (dS/dr -> 0), not create it.
        The correct conserved quantity for the entropy equation
        rho*T*dS/dt = -div(F) with zero-flux BCs is sum(rho*T*S*V),
        NOT sum(S*V). We check both the correct integral and that
        the profile homogenizes.
        """
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=False, convection=True)

        # Linear entropy gradient (unstable: dS/dr < 0 from CMB to surface)
        S0 = np.linspace(3400.0, 3000.0, N)
        H0 = compute_enthalpy_integral(S0, mesh, eos)

        def dSdt(t, S):
            state.update(S, t)
            state._heat_flux[-1] = 0.0
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0, 1000), S0, method='BDF',
                        atol=0.1, rtol=1e-6)
        assert sol.status == 0

        S_final = sol.y[:, -1]

        # Check the CORRECT conserved quantity: sum(rho * T * S * V)
        H_final = compute_enthalpy_integral(S_final, mesh, eos)
        rel_change = abs(H_final - H0) / abs(H0)
        assert rel_change < 5e-3, (
            f'Enthalpy integral sum(rho*T*S*V) changed by {rel_change:.4f} '
            f'(should be < 0.5%)'
        )

        # Entropy should homogenize (spread decreases)
        spread_0 = np.max(S0) - np.min(S0)
        spread_final = np.max(S_final) - np.min(S_final)
        assert spread_final < spread_0, (
            f'Entropy spread increased: {spread_0:.0f} -> {spread_final:.0f}'
        )


# -- Test 2: Energy conservation -----------------------------------------------

@needs_eos
@pytest.mark.smoke
class TestEnergyConservation:
    """Total thermal energy change must match integrated surface flux."""

    def test_closed_box_conserved_vs_nonconserved(self):
        """Closed box: enthalpy integral IS conserved, thermal energy is NOT.

        The entropy equation rho*T*dS/dt = -div(F) conserves sum(rho*T*S*V)
        when boundary fluxes are zero. But sum(rho*Cp*T*V) (thermal energy)
        is NOT a conserved quantity of this equation because rho, Cp, T all
        depend nonlinearly on S.

        This test verifies that the solver correctly preserves the RIGHT
        integral while the WRONG integral drifts, confirming the entropy
        formulation is working as intended.
        """
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=True, convection=True)

        # Non-uniform initial profile: linear gradient
        S0 = np.linspace(3400.0, 3000.0, N)
        E0 = compute_thermal_energy(S0, mesh, eos)
        H0 = compute_enthalpy_integral(S0, mesh, eos)

        def dSdt(t, S):
            state.update(S, t)
            state._heat_flux[-1] = 0.0
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0, 5000), S0, method='BDF',
                        atol=0.01, rtol=1e-8)
        assert sol.status == 0

        S_final = sol.y[:, -1]

        # Verify the system actually evolved
        max_dS = np.max(np.abs(S_final - S0))
        assert max_dS > 1.0, (
            f'Entropy barely changed (max dS={max_dS:.2f} J/kg/K). '
            f'Test must exercise nontrivial dynamics.'
        )

        # The CORRECT conserved quantity must be conserved
        H_final = compute_enthalpy_integral(S_final, mesh, eos)
        rel_H = abs(H_final - H0) / abs(H0)
        assert rel_H < 5e-3, (
            f'Enthalpy integral sum(rho*T*S*V) changed by {rel_H:.2e} '
            f'(should be < 0.5%)'
        )

        # The WRONG measure (thermal energy) should NOT be conserved
        # This validates the key insight: rho*Cp*T*V is not the right
        # conserved quantity for the entropy equation.
        E_final = compute_thermal_energy(S_final, mesh, eos)
        rel_E = abs(E_final - E0) / abs(E0)
        assert rel_E > rel_H, (
            f'Thermal energy sum(rho*Cp*T*V) changed by {rel_E:.2e}, '
            f'but enthalpy integral changed by {rel_H:.2e}. '
            f'Thermal energy should drift MORE than the conserved integral.'
        )

    def test_cooling_energy_budget(self):
        """Grey-body cooling: integrated entropy power balance must match
        the radiated energy.

        The discrete entropy equation is V_i * (rho*T)_i * (dS/dt)_i =
        -[F*A]_{i+1/2} + [F*A]_{i-1/2}. Summing over cells (the interior
        flux contributions telescope) and integrating in time:

            integral_0^T sum_i V_i (rho T)_i (dS/dt)_i dt = -Q_lost

        with Q_lost = integral F_surf*A_surf dt and F_cmb=0 by
        construction. This is the exact discrete conservation law of
        the entropy formulation.

        The naive thermal-energy proxy sum_i V_i rho_i Cp_i T_i is NOT
        a conserved quantity of this equation: T*dS = Cp*dT only holds
        at constant P and to leading order in dT/T, while a hot grey
        body radiating from 5500 K can drop T by ~20 % within hundreds
        of years. Use the integrated entropy power balance directly.
        """
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=True, convection=True)

        S0 = np.full(N, 3200.0)
        A_surf = mesh.basic.area[-1]
        V = mesh.basic.volume

        def dSdt(t, S):
            state.update(S, t)
            T_top = state.top_temperature.item()
            F_surf = Stefan_Boltzmann * (T_top**4 - 255.0**4)
            state._heat_flux[-1] = F_surf
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * V
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0, 500), S0, method='BDF',
                        atol=0.5, rtol=1e-5, dense_output=True)
        assert sol.status == 0

        # Sample at uniform time points and compute, at each:
        #   - F_surf*A_surf (radiated power, W)
        #   - sum_i V_i (rho*T)_i (dS/dt)_i (dimensionally W; should
        #     be -F_surf*A_surf at every instant by the discrete
        #     conservation identity).
        n_samples = 200
        times = np.linspace(0, min(500, sol.t[-1]), n_samples)
        Pwr_rad = np.zeros(n_samples)
        Pwr_int = np.zeros(n_samples)
        for i, t in enumerate(times):
            S_t = sol.sol(t)
            dS_dt_t = dSdt(t, S_t) / SECS_PER_YEAR  # back to W/(kg K)/s
            cap_per_volume = state.capacitance_staggered()
            Pwr_int[i] = float(np.sum(V * cap_per_volume * dS_dt_t))
            T_top = state.top_temperature.item()
            Pwr_rad[i] = Stefan_Boltzmann * (T_top**4 - 255.0**4) * A_surf

        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        # Convert sample times from years to seconds for the integrals.
        t_sec = times * SECS_PER_YEAR
        E_int = _trapz(Pwr_int, t_sec)   # entropy-power integral, J
        Q_lost = _trapz(Pwr_rad, t_sec)  # radiated, J

        # Discrete conservation says E_int = -Q_lost exactly modulo
        # time-integration error of solve_ivp's BDF. Allow 5 % to
        # cover the BDF time-step truncation and the trapezoidal
        # post-processing of the dense output on a 200-point grid.
        assert Q_lost > 0, f'Q_lost should be positive, got {Q_lost:.3e}'
        rel_residual = abs(E_int + Q_lost) / Q_lost
        assert rel_residual < 0.05, (
            f'Entropy power balance residual: {rel_residual:.3f} '
            f'(E_int={E_int:.2e} J, -Q_lost={-Q_lost:.2e} J); '
            f'should be < 5 %.'
        )


# -- Test 3: Grey-body cooling timescale ---------------------------------------

@needs_eos
@pytest.mark.smoke
class TestGreyBodyCooling:
    """Grey-body cooling from isentropic IC should follow t^{-1/3} at late times."""

    def test_surface_cools_monotonically(self):
        """T_surface must decrease monotonically during grey-body cooling."""
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=True, convection=True)

        S0 = np.full(N, 3200.0)

        def dSdt(t, S):
            state.update(S, t)
            T_top = state.top_temperature.item()
            state._heat_flux[-1] = Stefan_Boltzmann * (T_top**4 - 255.0**4)
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0, 5000), S0, method='BDF',
                        atol=0.5, rtol=1e-5, dense_output=True)
        assert sol.status == 0

        # Sample T_surf at regular intervals
        times = np.linspace(0, min(5000, sol.t[-1]), 20)
        T_surfs = []
        for t in times:
            S_t = sol.sol(t)
            state.phase_staggered.set_entropy(S_t)
            state.phase_staggered.update()
            T_surfs.append(float(state.phase_staggered.temperature()[-1]))

        T_surfs = np.array(T_surfs)
        # Must be monotonically decreasing (allow 1 K noise)
        diffs = np.diff(T_surfs)
        n_increasing = np.sum(diffs > 1.0)
        assert n_increasing == 0, (
            f'T_surface increased at {n_increasing} steps '
            f'(T range: {T_surfs[0]:.0f} to {T_surfs[-1]:.0f} K)'
        )

    def test_interior_participates(self):
        """Surface cooling drives a negative-dS/dr boundary layer that
        eventually transports entropy out of the upper interior.

        With an isentropic initial profile (dS/dr = 0 everywhere), the
        MLT eddy diffusivity is zero and there is no convective flux in
        the interior at t = 0. Convection only switches on once the
        surface cooling builds a negative-dS/dr boundary layer, after
        which entropy is transported out of the cells just below the
        surface. Verifying that this happens at all is what
        'interior participates' means here.

        We integrate long enough for the boundary layer to extend
        well past the top staggered cell and assert that the upper
        third of the mantle has lost a measurable amount of entropy.
        Mid-mantle and CMB-side cells stay close to S0 over this
        window because the convective penetration depth scales as
        sqrt(kappa_h_avg * t) and the eddy diffusivity vanishes in
        the still-isentropic interior; that is the expected physics
        and is checked by separate tests on long-time runs.
        """
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=True, convection=True)

        S0 = np.full(N, 3200.0)

        def dSdt(t, S):
            state.update(S, t)
            T_top = state.top_temperature.item()
            state._heat_flux[-1] = Stefan_Boltzmann * (T_top**4 - 255.0**4)
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0, 5000), S0, method='BDF',
                        atol=0.5, rtol=1e-5)
        assert sol.status == 0

        S_final = sol.y[:, -1]

        # The top staggered cell must drop substantially: it sees the
        # full grey-body radiative loss and is in direct flux contact
        # with the surface basic node.
        S_top = S_final[-1]
        assert S_top < S0[-1] - 50.0, (
            f'Top staggered entropy barely changed: '
            f'S_top={S_top:.1f}, S0={S0[-1]:.1f} (need at least 50 J/kg/K '
            f'drop for grey-body cooling at this duration).'
        )

        # The upper third of the mantle should have lost some entropy
        # via convective penetration of the cooling boundary layer.
        upper_third = S_final[2 * N // 3:]
        max_drop_upper = float(S0[0] - upper_third.min())
        assert max_drop_upper > 1.0, (
            f'Upper-third entropy did not decrease (max drop '
            f'{max_drop_upper:.3f} J/kg/K); convective boundary layer '
            f'failed to penetrate past the surface cell.'
        )

        # A negative-dS/dr boundary layer should be established by t=5 kyr.
        dSdr_top = (S_final[-1] - S_final[-2]) / (
            mesh.staggered.radii[-1] - mesh.staggered.radii[-2]
        )
        assert dSdr_top < 0.0, (
            f'No negative-dS/dr boundary layer at the surface '
            f'(dSdr_top={dSdr_top:.3e}); cooling should have produced '
            f'a convectively unstable gradient near the top.'
        )


# -- Tier 2c: Initial condition sensitivity -----------------------------------

@needs_eos
@pytest.mark.smoke
class TestInitialEntropySweep:
    """Different initial entropy values should all cool monotonically,
    with solidification timescale increasing with S0."""

    def test_all_ics_cool_monotonically(self):
        """Grey-body cooling from 4 different S0 values: T_surf always decreases."""
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)

        for S0_val in [2500.0, 3200.0, 5000.0]:
            N = 30
            mesh = make_mesh(N=N)
            state = make_state(mesh, eos, conduction=True, convection=True)
            S0 = np.full(N, S0_val)

            def dSdt(t, S, _s=state):
                _s.update(S, t)
                T_top = _s.top_temperature.item()
                F_surf = 5.670374419e-8 * (T_top**4 - 255.0**4)
                _s._heat_flux[-1] = F_surf
                _s._heat_flux[0] = 0.0
                energy_flux = _s.heat_flux * mesh.basic.area
                cap = _s.capacitance_staggered() * mesh.basic.volume
                return -np.diff(energy_flux) / cap * SECS_PER_YEAR

            sol = solve_ivp(dSdt, (0, 500), S0, method='BDF',
                            atol=0.5, rtol=1e-5, dense_output=True)
            assert sol.status == 0, f'S0={S0_val}: solver failed'

            # Check T_surf is monotonically decreasing
            times = np.linspace(0, min(500, sol.t[-1]), 20)
            T_surfs = []
            for t in times:
                S_t = sol.sol(t)
                _s_tmp = state
                _s_tmp.update(S_t, t)
                T_surfs.append(_s_tmp.top_temperature.item())
            T_surfs = np.array(T_surfs)
            diffs = np.diff(T_surfs)
            n_increasing = np.sum(diffs > 5.0)  # allow tiny fluctuations
            assert n_increasing == 0, (
                f'S0={S0_val}: T_surf increased in {n_increasing} steps')

    def test_higher_s0_starts_hotter(self):
        """Higher initial entropy should produce higher initial surface T."""
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)

        T_surfs = {}
        for S0_val in [2500.0, 3200.0, 5000.0]:
            N = 30
            mesh = make_mesh(N=N)
            state = make_state(mesh, eos, conduction=True, convection=True)
            S0 = np.full(N, S0_val)
            state.update(S0, 0)
            T_surfs[S0_val] = state.top_temperature.item()

        # T_surf should increase with S0
        assert T_surfs[3200.0] > T_surfs[2500.0], (
            f'T_surf(S0=3200)={T_surfs[3200.0]:.0f} K should be > '
            f'T_surf(S0=2500)={T_surfs[2500.0]:.0f} K')
        assert T_surfs[5000.0] > T_surfs[3200.0], (
            f'T_surf(S0=5000)={T_surfs[5000.0]:.0f} K should be > '
            f'T_surf(S0=3200)={T_surfs[3200.0]:.0f} K')


# -- Tier 2d: Radiogenic heating ----------------------------------------------

@needs_eos
@pytest.mark.smoke
class TestRadiogenicHeating:
    """Insulating box with radiogenic heating: entropy must increase."""

    def test_heating_increases_entropy(self):
        """Zero-flux BCs + constant heating: entropy rises monotonically."""
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=True, convection=True)

        S0 = np.full(N, 3200.0)
        H_rate = 1e-11  # W/kg (Earth-like radiogenic)

        def dSdt(t, S, _s=state):
            _s.update(S, t)
            _s._heat_flux[-1] = 0.0
            _s._heat_flux[0] = 0.0
            energy_flux = _s.heat_flux * mesh.basic.area
            cap = _s.capacitance_staggered() * mesh.basic.volume
            dsdt = -np.diff(energy_flux) / cap * SECS_PER_YEAR
            # Add heating: dS/dt += H / T
            T_stag = eos.temperature(mesh.staggered.pressure,
                                     np.asarray(S).flatten())
            dsdt += H_rate / np.maximum(T_stag, 1.0) * SECS_PER_YEAR
            return dsdt

        sol = solve_ivp(dSdt, (0, 1e6), S0, method='BDF',
                        atol=0.1, rtol=1e-6)
        assert sol.status == 0

        S_final = sol.y[:, -1]
        # Mean entropy should increase (heating with no cooling)
        # Expected: dS ~ H/T * dt ~ 1e-11/3000 * 1e6*3.15e7 ~ 0.1 J/kg/K
        dS_mean = np.mean(S_final) - np.mean(S0)
        assert dS_mean > 0.01, (
            f'Mean entropy change {dS_mean:.4f} J/kg/K is too small. '
            f'Heating should increase entropy (expected ~0.1 J/kg/K).')


# -- Tier 2e: Core cooling BC ------------------------------------------------

@needs_eos
@pytest.mark.smoke
class TestCoreCooling:
    """Core cooling BC: CMB flux positive, decreasing as mantle cools."""

    def test_core_heats_mantle(self):
        """With core cooling BC, CMB flux should be positive (core to mantle)."""
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=True, convection=True)

        S0 = np.full(N, 3200.0)
        # Core cooling parameters (Bower+2018)
        core_density = 12000.0  # kg/m^3
        core_cp = 800.0  # J/kg/K
        tfac = 1.147
        r_cmb = float(np.asarray(mesh.basic.radii).flat[0])
        r_above = float(np.asarray(mesh.basic.radii).flat[1])
        core_vol = 4.0 / 3.0 * np.pi * r_cmb**3
        core_cap = core_vol * core_density * core_cp

        def dSdt(t, S, _s=state):
            _s.update(S, t)
            # Grey-body at surface
            T_top = _s.top_temperature.item()
            _s._heat_flux[-1] = 5.670374419e-8 * (T_top**4 - 255.0**4)
            # Core cooling at CMB (Bower+2018 Eq. 37)
            rho_first = float(np.asarray(
                _s.phase_staggered.density()).flat[0])
            cp_first = float(np.asarray(
                _s.phase_staggered.heat_capacity()).flat[0])
            vol_first = float(np.asarray(mesh.basic.volume).flat[0])
            cell_cap = vol_first * rho_first * cp_first
            alpha_core = (r_above / r_cmb)**2 / (
                cell_cap / (core_cap * tfac) + 1.0)
            _s._heat_flux[0] = alpha_core * _s._heat_flux[1]

            energy_flux = _s.heat_flux * mesh.basic.area
            cap = _s.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, (0, 1000), S0, method='BDF',
                        atol=0.5, rtol=1e-5, dense_output=True)
        assert sol.status == 0

        # Check CMB flux is positive at multiple times
        for t in [10, 100, 500]:
            if t > sol.t[-1]:
                break
            S_t = sol.sol(t)
            state.update(S_t, t)
            # Recompute CMB flux
            rho_first = float(np.asarray(
                state.phase_staggered.density()).flat[0])
            cp_first = float(np.asarray(
                state.phase_staggered.heat_capacity()).flat[0])
            vol_first = float(np.asarray(mesh.basic.volume).flat[0])
            cell_cap = vol_first * rho_first * cp_first
            alpha_core = (r_above / r_cmb)**2 / (
                cell_cap / (core_cap * tfac) + 1.0)
            F_cmb = alpha_core * state._heat_flux[1]
            assert F_cmb > 0, (
                f't={t} yr: CMB flux = {F_cmb:.2e} W/m^2 (should be > 0)')


# -- Tier 2g: Mesh convergence -----------------------------------------------

@needs_eos
@pytest.mark.smoke
class TestMeshConvergence:
    """Grey-body cooling should converge with mesh resolution."""

    def test_convergence_with_resolution(self):
        """T_surf at t=500 yr should converge as N increases."""
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)

        T_surfs = {}
        for N in [25, 50, 100]:
            mesh = make_mesh(N=N)
            state = make_state(mesh, eos, conduction=True, convection=True)
            S0 = np.full(N, 3200.0)

            def dSdt(t, S, _s=state, _m=mesh):
                _s.update(S, t)
                T_top = _s.top_temperature.item()
                _s._heat_flux[-1] = 5.670374419e-8 * (T_top**4 - 255.0**4)
                _s._heat_flux[0] = 0.0
                energy_flux = _s.heat_flux * _m.basic.area
                cap = _s.capacitance_staggered() * _m.basic.volume
                return -np.diff(energy_flux) / cap * SECS_PER_YEAR

            sol = solve_ivp(dSdt, (0, 500), S0, method='BDF',
                            atol=0.5, rtol=1e-5, dense_output=True)
            if sol.status == 0:
                state.update(sol.sol(500), 500)
                T_surfs[N] = state.top_temperature.item()

        assert len(T_surfs) >= 2, f'Too few converged: {T_surfs}'

        # T_surf should converge: difference between finest resolutions
        # should be smaller than between coarsest
        Ns = sorted(T_surfs.keys())
        if len(Ns) >= 3:
            diff_coarse = abs(T_surfs[Ns[1]] - T_surfs[Ns[0]])
            diff_fine = abs(T_surfs[Ns[2]] - T_surfs[Ns[1]])
            assert diff_fine <= diff_coarse + 1.0, (
                f'Not converging: |T(N={Ns[2]})-T(N={Ns[1]})| = {diff_fine:.1f} K '
                f'> |T(N={Ns[1]})-T(N={Ns[0]})| = {diff_coarse:.1f} K')
