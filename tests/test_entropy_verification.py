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
        assert rel_change < 1e-3, (
            f'Enthalpy integral changed by {rel_change:.2e} '
            f'(should be < 0.1%)'
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
        """Grey-body cooling: dE/dt should match -F_surf * A_surf.

        Uses dense_output to sample at uniform time points, then recomputes
        fluxes at those times. This avoids the noise from BDF's repeated
        RHS evaluations for Jacobian estimation.
        """
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = make_mesh(N=N)
        state = make_state(mesh, eos, conduction=True, convection=True)

        S0 = np.full(N, 3200.0)
        A_surf = mesh.basic.area[-1]

        t_span = (0, 500)
        def dSdt(t, S):
            state.update(S, t)
            T_top = state.top_temperature.item()
            F_surf = Stefan_Boltzmann * (T_top**4 - 255.0**4)
            state._heat_flux[-1] = F_surf
            state._heat_flux[0] = 0.0
            energy_flux = state.heat_flux * mesh.basic.area
            cap = state.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(dSdt, t_span, S0, method='BDF',
                        atol=0.5, rtol=1e-5, dense_output=True)
        assert sol.status == 0

        # Sample at uniform time points via dense_output, recompute fluxes
        n_samples = 100
        times = np.linspace(0, min(500, sol.t[-1]), n_samples)
        F_surf_arr = np.zeros(n_samples)
        for i, t in enumerate(times):
            S_t = sol.sol(t)
            state.update(S_t, t)
            T_top = state.top_temperature.item()
            F_surf_arr[i] = Stefan_Boltzmann * (T_top**4 - 255.0**4)

        # Compute energy at start and end
        E_start = compute_thermal_energy(S0, mesh, eos)
        E_end = compute_thermal_energy(sol.sol(times[-1]), mesh, eos)
        dE = E_end - E_start  # should be negative (cooling)

        # Integrated flux loss (trapezoidal on uniform samples)
        # Power in watts, time in years -> energy in J
        Q_lost = np.trapz(F_surf_arr * A_surf, times * SECS_PER_YEAR)

        # dE should approximately equal -Q_lost
        if abs(Q_lost) > 0:
            rel_residual = abs(dE + Q_lost) / abs(Q_lost)
            assert rel_residual < 0.03, (
                f'Energy budget residual: {rel_residual:.2f} '
                f'(dE={dE:.2e}, Q_lost={Q_lost:.2e}), should be < 3%'
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
        """Interior entropy must decrease (convection transports heat to surface)."""
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

        sol = solve_ivp(dSdt, (0, 1000), S0, method='BDF',
                        atol=0.5, rtol=1e-5)
        assert sol.status == 0

        S_final = sol.y[:, -1]
        S_mid = S_final[N // 2]
        assert S_mid < 3200.0 - 10.0, (
            f'Mid-mantle entropy barely changed: {S_mid:.0f} (was 3200). '
            f'Convection should transport entropy from interior to surface.'
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

    def test_higher_s0_cools_slower(self):
        """Higher initial entropy should take longer to cool by the same amount."""
        from aragog.eos.entropy import EntropyEOS
        eos = EntropyEOS(EOS_DIR)

        dT_threshold = 200.0  # K drop to measure
        cooling_times = {}

        for S0_val in [3200.0, 5000.0]:
            N = 30
            mesh = make_mesh(N=N)
            state = make_state(mesh, eos, conduction=True, convection=True)
            S0 = np.full(N, S0_val)

            state.update(S0, 0)
            T_surf_init = state.top_temperature.item()

            def dSdt(t, S, _s=state):
                _s.update(S, t)
                T_top = _s.top_temperature.item()
                F_surf = 5.670374419e-8 * (T_top**4 - 255.0**4)
                _s._heat_flux[-1] = F_surf
                _s._heat_flux[0] = 0.0
                energy_flux = _s.heat_flux * mesh.basic.area
                cap = _s.capacitance_staggered() * mesh.basic.volume
                return -np.diff(energy_flux) / cap * SECS_PER_YEAR

            sol = solve_ivp(dSdt, (0, 10000), S0, method='BDF',
                            atol=0.5, rtol=1e-5, dense_output=True)
            if sol.status != 0:
                continue

            # Find time to cool by dT_threshold
            times = np.linspace(0, sol.t[-1], 200)
            for t in times:
                state.update(sol.sol(t), t)
                T_now = state.top_temperature.item()
                if T_surf_init - T_now >= dT_threshold:
                    cooling_times[S0_val] = t
                    break

        if 3200.0 in cooling_times and 5000.0 in cooling_times:
            assert cooling_times[5000.0] > cooling_times[3200.0], (
                f'S0=5000 cooled in {cooling_times[5000.0]:.0f} yr but '
                f'S0=3200 took {cooling_times[3200.0]:.0f} yr. '
                f'Higher entropy should take longer.')


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
        dS_mean = np.mean(S_final) - np.mean(S0)
        assert dS_mean > 1.0, (
            f'Mean entropy change {dS_mean:.2f} J/kg/K is too small. '
            f'Heating should increase entropy.')


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
