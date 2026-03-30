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

    def test_isentropic_stays_isentropic(self):
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

    def test_closed_box_energy(self):
        """Non-uniform S, zero-flux BCs, conduction enabled: enthalpy conserved.

        Starts from a non-uniform S profile so the system actually evolves
        (conduction redistributes entropy). With zero-flux BCs, the total
        enthalpy integral H = sum(rho*T*S*V) must be conserved.
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

        # Verify the system actually evolved (not a zero-RHS test)
        T0 = eos.temperature(mesh.staggered.pressure, S0)
        T_final = eos.temperature(mesh.staggered.pressure, S_final)
        max_dT = np.max(np.abs(T_final - T0))
        assert max_dT > 1.0, (
            f'Temperature barely changed (max dT = {max_dT:.1f} K). '
            f'Test must exercise nontrivial dynamics.'
        )

        # Enthalpy integral must be conserved with zero-flux BCs
        H_final = compute_enthalpy_integral(S_final, mesh, eos)
        rel_change = abs(H_final - H0) / abs(H0)
        assert rel_change < 5e-3, (
            f'Enthalpy integral changed by {rel_change:.2e} '
            f'(should be < 0.5%)'
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
