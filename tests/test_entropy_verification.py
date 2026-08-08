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

EOS_DIR = Path(
    os.environ.get(
        'ARAGOG_TEST_EOS_DIR',
        '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
    )
)

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}',
)


def _cvode_available():
    try:
        from aragog.solver.entropy_solver import _CVODE_AVAILABLE

        return bool(_CVODE_AVAILABLE)
    except Exception:
        return False


needs_cvode = pytest.mark.skipif(
    not _cvode_available(),
    reason='scikits-odes-sundials (CVODE) not installed',
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

    phase_stag = EntropyPhaseEvaluator(entropy_eos=entropy_eos, gravitational_acceleration=10.0)
    phase_stag.set_pressure(mesh.staggered.pressure)

    phase_basic = EntropyPhaseEvaluator(
        entropy_eos=entropy_eos, gravitational_acceleration=10.0
    )
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

        sol = solve_ivp(dSdt, (0, 5000), S0, method='BDF', atol=0.01, rtol=1e-8)
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
            f'Enthalpy integral changed by {rel_change:.2e} (should be < 0.2%)'
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

        sol = solve_ivp(dSdt, (0, 1000), S0, method='BDF', atol=0.1, rtol=1e-6)
        assert sol.status == 0

        S_final = sol.y[:, -1]

        # Check the CORRECT conserved quantity: sum(rho * T * S * V)
        H_final = compute_enthalpy_integral(S_final, mesh, eos)
        rel_change = abs(H_final - H0) / abs(H0)
        assert rel_change < 5e-3, (
            f'Enthalpy integral sum(rho*T*S*V) changed by {rel_change:.4f} (should be < 0.5%)'
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

        sol = solve_ivp(dSdt, (0, 5000), S0, method='BDF', atol=0.01, rtol=1e-8)
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
            f'Enthalpy integral sum(rho*T*S*V) changed by {rel_H:.2e} (should be < 0.5%)'
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

        sol = solve_ivp(
            dSdt, (0, 500), S0, method='BDF', atol=0.5, rtol=1e-5, dense_output=True
        )
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
        E_int = _trapz(Pwr_int, t_sec)  # entropy-power integral, J
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


# -- Test 2b: per-call solver-residual telescoping -----------------------------


@needs_eos
@pytest.mark.smoke
class TestSolverResidualTelescoping:
    """The per-call entropy-equation self-consistency residual must telescope.

    ``EntropySolver._compute_step_energy_integrals`` reconstructs, at every
    accepted integrator substep, the discrete entropy-equation identity

        Σ_i capacitance_i (dS/dt)_i V_i == -F_int A_int + F_cmb A_cmb
                                            + Q_radio + Q_tidal

    where ``capacitance = ρ_phase T``. The interior flux contributions
    telescope, so the LHS-RHS residual is machine-zero at every state -- but
    only when the LHS mass weighting uses the SAME phase density the RHS
    capacitance used. Re-deriving the cell mass from the hard-masked table
    density (``EntropyEOS.density``) instead of the tanh-blended phase density
    (``EntropyPhaseEvaluator.density``) leaves a residual equal to the local
    flux divergence times the fractional density mismatch, concentrated in the
    phase-transition band: a sub-percent density difference against the
    ~1e16 W flux divergence inflates the integrated residual to ~1e23 J.

    Verifies the conservation invariant (exact discrete energy telescoping)
    with a discrimination guard against the inconsistent-density weighting.

    See docs/How-to/test_infrastructure.md, test_categorization.md,
    test_building.md.
    """

    @staticmethod
    def _build_two_phase_solver(end_time=50.0):
        """A small PALEOS-table solver whose stratified IC straddles the
        solidus, so part of the mantle sits in the mushy band where the two
        density routines diverge."""
        from aragog.eos.entropy import EntropyEOS
        from aragog.parser import (
            Parameters,
        )
        from aragog.parser import (
            _BoundaryConditionsParameters as BC,
        )
        from aragog.parser import (
            _EnergyParameters as EN,
        )
        from aragog.parser import (
            _InitialConditionParameters as IC,
        )
        from aragog.parser import (
            _MeshParameters as MESHP,
        )
        from aragog.parser import (
            _PhaseMixedParameters as PM,
        )
        from aragog.parser import (
            _PhaseParameters as PH,
        )
        from aragog.parser import (
            _SolverParameters as SV,
        )
        from aragog.solver.entropy_solver import EntropySolver

        eos = EntropyEOS(EOS_DIR)
        rhos = 4078.95095544
        bc = BC(
            outer_boundary_condition=4,
            outer_boundary_value=1e2,
            inner_boundary_condition=2,
            inner_boundary_value=0.0,
            emissivity=1.0,
            equilibrium_temperature=255.0,
            core_heat_capacity=880.0,
            core_bc='quasi_steady',
        )
        en = EN(
            conduction=True,
            convection=True,
            gravitational_separation=False,
            mixing=False,
            radionuclides=False,
            tidal=False,
            solver_method='bdf',
            use_jax_jacobian=False,
            eddy_diffusivity_thermal=1.0,
            kappah_floor=10.0,
        )
        ic = IC(initial_condition=1, surface_temperature=2000.0, basal_temperature=4000.0)
        mesh = MESHP(
            outer_radius=6.371e6,
            inner_radius=3.480e6,
            number_of_nodes=80,
            mixing_length_profile='constant',
            core_density=10500.0,
            eos_method=1,
            surface_density=rhos,
            adams_williamson_beta=1.1115348931e-07,
            adiabatic_bulk_modulus=260e9,
            gravitational_acceleration=9.81,
            surface_pressure=0.0,
        )
        pm = PM(
            latent_heat_of_fusion=4e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus=str(EOS_DIR / 'solidus_P-S.dat'),
            liquidus=str(EOS_DIR / 'liquidus_P-S.dat'),
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1e-3,
            matprop_smooth_width=1e-2,
            const_properties=False,
        )
        pl = PH(
            density=rhos,
            heat_capacity=1000.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            melt_fraction=1.0,
            viscosity=1e2,
        )
        ps = PH(
            density=rhos,
            heat_capacity=1000.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            melt_fraction=0.0,
            viscosity=1e21,
        )
        sv = SV(
            start_time=0.0, end_time=end_time, atol=1e-8, rtol=1e-8, tsurf_poststep_change=1e9
        )
        params = Parameters(
            boundary_conditions=bc,
            energy=en,
            initial_condition=ic,
            mesh=mesh,
            phase_solid=ps,
            phase_liquid=pl,
            phase_mixed=pm,
            radionuclides=[],
            solver=sv,
        )
        s = EntropySolver(params, entropy_eos=eos)
        s.initialize()
        return s, eos

    def _stratified_ic(self, solver):
        """Entropy profile that puts the deep half low (mushy / just-frozen)
        and the shallow half high (molten), so the field crosses the solidus
        somewhere in the mantle and a mushy band is guaranteed."""
        mesh = solver.evaluator.mesh
        rs = np.asarray(mesh.staggered.radii).ravel()
        Ps = np.asarray(mesh.staggered_pressure).ravel()
        S_sol = np.asarray(solver.entropy_eos.solidus_entropy(Ps)).ravel()
        S_liq = np.asarray(solver.entropy_eos.liquidus_entropy(Ps)).ravel()
        x = (rs - rs.min()) / (rs.max() - rs.min())  # 0 at CMB, 1 at surface
        # deep half mid-mushy (phi ~ 0.5), shallow half just above the
        # liquidus (molten), so the front sits at mid-radius.
        S_mushy = S_sol + 0.5 * (S_liq - S_sol)
        S_molten = S_liq * 1.02
        return np.where(x < 0.5, S_mushy, S_molten)

    def test_residual_telescopes_to_machine_zero_across_mushy_zone(self):
        """With a mushy zone present, the reported solver residual must be a
        machine-zero fraction of the boundary-flux integral, AND the
        inconsistent ``eos.density`` weighting must produce a residual orders
        of magnitude larger (the discrimination guard against a revert)."""
        from aragog.solver.entropy_solver import SECS_PER_YEAR

        solver, eos = self._build_two_phase_solver()
        S0 = self._stratified_ic(solver)
        solver.set_initial_entropy(S0)
        solver.solve()

        d = solver._compute_step_energy_integrals()
        flux_scale = max(abs(d['F_int']), abs(d['F_cmb']), 1.0)

        # The run must actually have a mushy zone and evolve, else the test
        # is vacuous (the density routines agree everywhere outside it).
        sol = solver._solution
        n_stag = solver._n_stag
        P_stag = solver._P_stag_flat
        S_first = sol.y[:n_stag, 0] if sol.y.ndim == 2 else sol.y[:n_stag]
        phi0 = np.asarray(eos.melt_fraction(P_stag, S_first)).ravel()
        n_mushy = int(np.sum((phi0 > 0.0) & (phi0 < 1.0)))
        assert n_mushy >= 3, f'IC must straddle the solidus; got {n_mushy} mushy cells'

        # Conservation invariant: the consistent-density residual telescopes.
        resid_frac = abs(d['solver_residual']) / flux_scale
        assert resid_frac < 1e-9, (
            f'solver_residual is {resid_frac:.2e} of the flux integral; '
            f'the discrete entropy equation must telescope to machine zero'
        )

        # Discrimination guard: recompute the residual with the OLD
        # ``eos.density`` (hard-masked) weighting. It must be many orders of
        # magnitude larger, so a revert to the inconsistent weighting fails.
        # Heating is off in this IC, so the RHS reduces to the boundary
        # fluxes and the guard isolates the LHS density-weighting error.
        vol = solver._volume_flat
        r_basic = solver._r_basic_flat
        A_int = 4.0 * np.pi * float(r_basic[-1]) ** 2
        A_cmb = 4.0 * np.pi * float(r_basic[0]) ** 2
        dt_s = np.diff(np.asarray(sol.t, dtype=float)) * SECS_PER_YEAR
        n_steps = int(sol.t.size)
        p_naive = np.zeros(n_steps)
        for i in range(n_steps):
            t_i = float(sol.t[i])
            y_col = sol.y[:, i] if sol.y.ndim == 2 else sol.y
            S_i = y_col[:n_stag]
            dSdt_full = np.asarray(solver.dSdt(t_i, y_col)).ravel()
            dSdt_stag = dSdt_full[:n_stag] / SECS_PER_YEAR
            F_int_i = float(solver.state._heat_flux[-1])
            F_cmb_i = float(solver.state._heat_flux[0])
            rho_eos = np.asarray(eos.density(P_stag, S_i)).ravel()
            T_phase = np.asarray(solver.state.phase_staggered.temperature()).ravel()
            lhs_naive = float(np.sum(rho_eos * T_phase * dSdt_stag * vol))
            p_naive[i] = lhs_naive - (-F_int_i * A_int + F_cmb_i * A_cmb)
        naive_resid = float(np.sum(0.5 * (p_naive[:-1] + p_naive[1:]) * dt_s))
        naive_frac = abs(naive_resid) / flux_scale
        assert naive_frac > 1e3 * resid_frac, (
            f'discrimination guard: eos.density weighting residual '
            f'({naive_frac:.2e}) must dwarf the consistent one ({resid_frac:.2e})'
        )

    def test_compression_work_matches_enthalpy_change(self):
        """The structure-re-solve compression term equals the adiabatic
        enthalpy change Sum mass (h(P_new, S) - h(P_old, S)) and vanishes for
        an unchanged structure.

        Verifies the conservation primitive that closes the dynamic-
        contraction term in E_residual_cons_frac, with a static-structure
        guard (zero) and a shape-change guard (zero), and a discrimination
        guard that a real pressure rise moves a non-trivial fraction of the
        mantle enthalpy."""
        solver, eos = self._build_two_phase_solver()
        S0 = self._stratified_ic(solver)
        P_old = np.asarray(solver._P_stag_flat).ravel().copy()
        mesh = solver.evaluator.mesh
        mass = np.asarray(mesh.staggered_effective_density).ravel() * solver._volume_flat

        # Static structure: P unchanged -> exactly zero compression.
        comp_static = solver._compute_compression_work(P_old, S0, mass)
        assert comp_static == 0.0, 'static structure must carry no compression work'

        # Contraction: a 5 percent pressure rise gains compression enthalpy.
        solver._P_stag_flat = P_old * 1.05
        comp = solver._compute_compression_work(P_old, S0, mass)
        h_old = np.asarray(eos.specific_enthalpy(P_old, S0)).ravel()
        h_new = np.asarray(eos.specific_enthalpy(P_old * 1.05, S0)).ravel()
        expect = float(np.sum(mass * (h_new - h_old)))
        np.testing.assert_allclose(comp, expect, rtol=1e-12)
        assert comp > 0.0, 'compression (rising P) must add energy'

        # Discrimination: the term is a non-trivial fraction of the enthalpy.
        e_state = float(np.sum(mass * h_old))
        assert abs(comp) > 1e-4 * abs(e_state), (
            'a 5 percent pressure rise must move a resolvable fraction of '
            'the mantle enthalpy, else the test is vacuous'
        )

        # Shape change (mesh resolution differs) -> undefined per-cell delta,
        # must return zero rather than crash.
        solver._P_stag_flat = (P_old * 1.05)[:-1]
        assert solver._compute_compression_work(P_old, S0, mass) == 0.0

    def test_entropy_remap_carries_field_by_mass_parcel(self):
        """The conservative remap interpolates the carried entropy onto the new
        mesh at equal cumulative mantle mass fraction, is a no-op when the grid
        is unchanged, and falls back to the input when the node count changes.

        The deformation moves r_core and r_surf by different factors and
        reshapes the interior mass distribution, so each parcel's mass fraction
        shifts. The remap must match interpolation on mass fraction ``f``, not
        on the raw length coordinate ``xi``.

        Discrimination guard: under this non-uniform deformation the mass-
        fraction remap differs from the raw-``xi`` interpolation by more than
        rounding, confirming the remap moves entropy by mass parcel rather than
        by length coordinate."""
        from aragog.solver.entropy_solver import _mantle_mass_fraction

        solver, _ = self._build_two_phase_solver()
        S0 = self._stratified_ic(solver)
        solver.set_initial_entropy(S0)
        solver.solve()
        xi = solver.staggered_mass_coordinates
        n = len(xi)
        assert n == solver._n_stag, 'mass-coordinate length must match the staggered mesh'
        assert np.all(np.diff(xi) > 0) or np.all(np.diff(xi) < 0), (
            'mass coordinates must be monotonic for np.interp'
        )

        S_field = np.linspace(3000.0, 4000.0, n)
        asc = xi[0] < xi[-1]
        # Non-uniform re-solve: endpoints move by different factors and the
        # interior density profile reshapes, so mass fractions shift per parcel
        # (a uniform scale would leave every f unchanged and remap nothing).
        f_old = _mantle_mass_fraction(xi)
        lo, hi = min(xi[0], xi[-1]), max(xi[0], xi[-1])
        lo_new, hi_new = lo * 0.994, hi * 0.967
        f_reshaped = f_old**1.08
        xi_new = np.cbrt(lo_new**3 + (hi_new**3 - lo_new**3) * f_reshaped)
        orig = type(solver).staggered_mass_coordinates
        try:
            type(solver).staggered_mass_coordinates = property(lambda self: xi_new)
            solver._xi_pre_resolve = xi.copy()
            S_remap = solver._remap_entropy_to_current_mesh(S_field.copy())
            f_new = _mantle_mass_fraction(xi_new)
            expect = (
                np.interp(f_new, f_old, S_field)
                if f_old[0] < f_old[-1]
                else np.interp(f_new[::-1], f_old[::-1], S_field[::-1])[::-1]
            )
            np.testing.assert_allclose(S_remap, expect, rtol=1e-12)
            # Discrimination: the mass-fraction remap differs from a raw-xi
            # interpolation of the same deformation by more than rounding.
            raw_xi = (
                np.interp(xi_new, xi, S_field)
                if asc
                else np.interp(xi_new[::-1], xi[::-1], S_field[::-1])[::-1]
            )
            assert np.max(np.abs(S_remap - raw_xi)) > 1e-6 * np.ptp(S_field), (
                'the remap must interpolate on mass fraction, not raw mass-radius'
            )
            # The remap genuinely moves the field (parcels shifted).
            assert np.max(np.abs(S_remap - S_field)) > 1e-6 * np.ptp(S_field), (
                'a non-uniform re-solve must move the entropy field by parcel'
            )
            assert solver._xi_pre_resolve is None, 'remap must consume the cached grid'

            # No-op when the mass grid is unchanged.
            solver._xi_pre_resolve = xi_new.copy()
            np.testing.assert_array_equal(
                solver._remap_entropy_to_current_mesh(S_field.copy()), S_field
            )
            # Node-count mismatch -> fall back to the input unchanged.
            solver._xi_pre_resolve = xi_new[:-1].copy()
            np.testing.assert_array_equal(
                solver._remap_entropy_to_current_mesh(S_field.copy()), S_field
            )
        finally:
            type(solver).staggered_mass_coordinates = orig


# -- Test 2c: CVODE energy-diagnostic output grid ------------------------------


@needs_eos
@needs_cvode
@pytest.mark.smoke
class TestCvodeEnergyOutputGrid:
    """The per-call energy integrals must resolve the within-call flux decay.

    In step-wise mode (the production default) the trajectory handed to the
    integrals is CVODE's own accepted internal steps, which resolve any flux
    transient the integrator resolved, independent of the output request.
    With ``_substep_integrals = False`` the solve returns only the times in
    the requested tspan, where a two-point grid leaves the trapezoidal
    integral of a steeply-decaying boundary flux tens of percent wrong and
    the front-loaded ``_cvode_output_points`` grid recovers it.

    Verifies that the step-wise trajectory recovers the flux integral even
    on a two-point output request, that the fixed dense grid recovers it
    while the fixed two-point grid is an order of magnitude worse, and that
    neither mode perturbs the integration window. See
    docs/How-to/test_infrastructure.md.
    """

    @staticmethod
    def _build_greybody_solver(method, n_out=None):
        """Grey-body surface cooling: the surface flux ~ sigma T_top^4 decays
        steeply within the call, the regime that breaks a 2-point integral."""
        from aragog.eos.entropy import EntropyEOS
        from aragog.parser import (
            Parameters,
        )
        from aragog.parser import (
            _BoundaryConditionsParameters as BC,
        )
        from aragog.parser import (
            _EnergyParameters as EN,
        )
        from aragog.parser import (
            _InitialConditionParameters as IC,
        )
        from aragog.parser import (
            _MeshParameters as MESHP,
        )
        from aragog.parser import (
            _PhaseMixedParameters as PM,
        )
        from aragog.parser import (
            _PhaseParameters as PH,
        )
        from aragog.parser import (
            _SolverParameters as SV,
        )
        from aragog.solver.entropy_solver import EntropySolver

        eos = EntropyEOS(EOS_DIR)
        rhos = 4078.95095544
        bc = BC(
            outer_boundary_condition=1,  # grey-body radiative surface
            outer_boundary_value=0.0,
            inner_boundary_condition=2,
            inner_boundary_value=0.0,
            emissivity=1.0,
            equilibrium_temperature=255.0,
            core_heat_capacity=880.0,
            core_bc='quasi_steady',
        )
        en = EN(
            conduction=True,
            convection=True,
            gravitational_separation=False,
            mixing=False,
            radionuclides=False,
            tidal=False,
            solver_method=method,
            use_jax_jacobian=False,
            eddy_diffusivity_thermal=1.0,
            kappah_floor=10.0,
        )
        ic = IC(initial_condition=1, surface_temperature=3000.0, basal_temperature=4000.0)
        mesh = MESHP(
            outer_radius=6.371e6,
            inner_radius=3.480e6,
            number_of_nodes=60,
            mixing_length_profile='constant',
            core_density=10500.0,
            eos_method=1,
            surface_density=rhos,
            adams_williamson_beta=1.1115348931e-07,
            adiabatic_bulk_modulus=260e9,
            gravitational_acceleration=9.81,
            surface_pressure=0.0,
        )
        pm = PM(
            latent_heat_of_fusion=4e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus=str(EOS_DIR / 'solidus_P-S.dat'),
            liquidus=str(EOS_DIR / 'liquidus_P-S.dat'),
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1e-3,
            matprop_smooth_width=1e-2,
            const_properties=False,
        )
        pl = PH(
            density=rhos,
            heat_capacity=1000.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            melt_fraction=1.0,
            viscosity=1e2,
        )
        ps = PH(
            density=rhos,
            heat_capacity=1000.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            melt_fraction=0.0,
            viscosity=1e21,
        )
        sv = SV(start_time=0.0, end_time=5e3, atol=1e-8, rtol=1e-8, tsurf_poststep_change=1e9)
        params = Parameters(
            boundary_conditions=bc,
            energy=en,
            initial_condition=ic,
            mesh=mesh,
            phase_solid=ps,
            phase_liquid=pl,
            phase_mixed=pm,
            radionuclides=[],
            solver=sv,
        )
        s = EntropySolver(params, entropy_eos=eos)
        s.initialize()
        if n_out is not None:
            s._cvode_output_points = n_out
        Ps = np.asarray(s.evaluator.mesh.staggered_pressure).ravel()
        S_liq = np.asarray(s.entropy_eos.liquidus_entropy(Ps)).ravel()
        s.set_initial_entropy(S_liq * 1.01)
        return s

    def test_dense_output_recovers_flux_integral(self):
        """Internal-step trajectories recover F_int regardless of the output
        request; the fixed-grid path keeps its dense-versus-2-point contrast
        and ``dt_actual`` is unchanged in both modes."""
        ref = self._build_greybody_solver('bdf')
        ref.solve()
        f_ref = ref._compute_step_energy_integrals()['F_int']
        assert ref._solution.t.size > 200, 'scipy reference grid must be dense'
        assert abs(f_ref) > 1e28, 'cooling must drive a non-trivial surface flux'

        # Step-wise mode (the production default): the integrals ride the
        # integrator's own accepted steps, so even a 2-point output request
        # recovers the integral. This is what discriminates the internal-step
        # trajectory from the output grid: on the fixed grid below, the same
        # 2-point request is an order of magnitude worse.
        s_sub = self._build_greybody_solver('cvode', n_out=2)
        s_sub.solve()
        err_sub = abs(s_sub._compute_step_energy_integrals()['F_int'] / f_ref - 1.0)
        assert s_sub._solution.t.size > 65, (
            'stepwise mode must return the internal steps, not the output grid'
        )
        assert err_sub < 2e-2, (
            f'stepwise F_int off by {err_sub:.2e} from the reference; the '
            f'internal-step trajectory must resolve the flux decay by construction'
        )

        # Fixed-grid path: the dense front-loaded grid recovers the integral,
        # the 2-point grid must not.
        s2 = self._build_greybody_solver('cvode', n_out=2)
        s2._substep_integrals = False
        s2.solve()
        d2 = s2._compute_step_energy_integrals()
        err_2pt = abs(d2['F_int'] / f_ref - 1.0)

        sN = self._build_greybody_solver('cvode', n_out=65)
        sN._substep_integrals = False
        sN.solve()
        dN = sN._compute_step_energy_integrals()
        err_dense = abs(dN['F_int'] / f_ref - 1.0)

        assert err_dense < 2e-2, (
            f'dense-grid F_int off by {err_dense:.2e} from the reference; '
            f'the front-loaded output grid must resolve the flux decay'
        )
        assert err_2pt > 10.0 * err_dense, (
            f'discrimination guard: 2-point F_int error ({err_2pt:.2e}) must '
            f'dwarf the dense-grid error ({err_dense:.2e})'
        )
        # Neither trajectory choice may perturb the integration window.
        dt_ref = float(ref._solution.t[-1] - ref._solution.t[0])
        dt_dense = float(sN._solution.t[-1] - sN._solution.t[0])
        dt_sub = float(s_sub._solution.t[-1] - s_sub._solution.t[0])
        np.testing.assert_allclose(dt_dense, dt_ref, rtol=1e-10)
        np.testing.assert_allclose(dt_sub, dt_ref, rtol=1e-10)


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

        sol = solve_ivp(
            dSdt, (0, 5000), S0, method='BDF', atol=0.5, rtol=1e-5, dense_output=True
        )
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

        sol = solve_ivp(dSdt, (0, 5000), S0, method='BDF', atol=0.5, rtol=1e-5)
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
        upper_third = S_final[2 * N // 3 :]
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

            sol = solve_ivp(
                dSdt, (0, 500), S0, method='BDF', atol=0.5, rtol=1e-5, dense_output=True
            )
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
            assert n_increasing == 0, f'S0={S0_val}: T_surf increased in {n_increasing} steps'

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
            f'T_surf(S0=2500)={T_surfs[2500.0]:.0f} K'
        )
        assert T_surfs[5000.0] > T_surfs[3200.0], (
            f'T_surf(S0=5000)={T_surfs[5000.0]:.0f} K should be > '
            f'T_surf(S0=3200)={T_surfs[3200.0]:.0f} K'
        )


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
            T_stag = eos.temperature(mesh.staggered.pressure, np.asarray(S).flatten())
            dsdt += H_rate / np.maximum(T_stag, 1.0) * SECS_PER_YEAR
            return dsdt

        sol = solve_ivp(dSdt, (0, 1e6), S0, method='BDF', atol=0.1, rtol=1e-6)
        assert sol.status == 0

        S_final = sol.y[:, -1]
        # Mean entropy should increase (heating with no cooling)
        # Expected: dS ~ H/T * dt ~ 1e-11/3000 * 1e6*3.15e7 ~ 0.1 J/kg/K
        dS_mean = np.mean(S_final) - np.mean(S0)
        assert dS_mean > 0.01, (
            f'Mean entropy change {dS_mean:.4f} J/kg/K is too small. '
            f'Heating should increase entropy (expected ~0.1 J/kg/K).'
        )


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
            rho_first = float(np.asarray(_s.phase_staggered.density()).flat[0])
            cp_first = float(np.asarray(_s.phase_staggered.heat_capacity()).flat[0])
            vol_first = float(np.asarray(mesh.basic.volume).flat[0])
            cell_cap = vol_first * rho_first * cp_first
            alpha_core = (r_above / r_cmb) ** 2 / (cell_cap / (core_cap * tfac) + 1.0)
            _s._heat_flux[0] = alpha_core * _s._heat_flux[1]

            energy_flux = _s.heat_flux * mesh.basic.area
            cap = _s.capacitance_staggered() * mesh.basic.volume
            return -np.diff(energy_flux) / cap * SECS_PER_YEAR

        sol = solve_ivp(
            dSdt, (0, 1000), S0, method='BDF', atol=0.5, rtol=1e-5, dense_output=True
        )
        assert sol.status == 0

        # Check CMB flux is positive at multiple times
        for t in [10, 100, 500]:
            if t > sol.t[-1]:
                break
            S_t = sol.sol(t)
            state.update(S_t, t)
            # Recompute CMB flux
            rho_first = float(np.asarray(state.phase_staggered.density()).flat[0])
            cp_first = float(np.asarray(state.phase_staggered.heat_capacity()).flat[0])
            vol_first = float(np.asarray(mesh.basic.volume).flat[0])
            cell_cap = vol_first * rho_first * cp_first
            alpha_core = (r_above / r_cmb) ** 2 / (cell_cap / (core_cap * tfac) + 1.0)
            F_cmb = alpha_core * state._heat_flux[1]
            assert F_cmb > 0, f't={t} yr: CMB flux = {F_cmb:.2e} W/m^2 (should be > 0)'


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

            sol = solve_ivp(
                dSdt, (0, 500), S0, method='BDF', atol=0.5, rtol=1e-5, dense_output=True
            )
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
                f'> |T(N={Ns[1]})-T(N={Ns[0]})| = {diff_coarse:.1f} K'
            )


# -- Tier 1: No explicit Φ_vol source -----------------------------------------
#
# The historical explicit form of segregation heating
#
#     Phi_vol = rho * g * (1/rho_m - 1/rho_s) * (j_cm + j_gm)         [W/m^3]
#
# is NOT added as an explicit volumetric source on top of the divergence
# of the Δh-weighted mass-flux contributions to ``state._heat_flux``.
# By definition Δh = Δu + P·Δv, and on a hydrostatic column
# ∂Δh/∂r ⊃ Δv·∂P/∂r = -ρg·Δv, so -∂/∂r(j·Δh) ⊃ +ρg·Δv·j is the same
# quantity. Adding it explicitly would double-count (Bower 2018 §3,
# SPIDER energy.c). The tests below pin that invariant by exercising
# the regime where the historical explicit source was largest.


def _make_test_mesh_with_staggered_interp(
    N=40,
    R_cmb=3480e3,
    R_surf=6371e3,
    P_cmb=135e9,
    P_surf=1e5,
):
    """Like make_mesh, but also exposes ``quantity_at_staggered_nodes``.

    Some heating code paths interpolate basic-node arrays to the
    staggered grid via ``mesh.quantity_at_staggered_nodes``. The
    default test mesh in this module does not define that method, so
    this helper extends it.
    """
    mesh = make_mesh(N=N, R_cmb=R_cmb, R_surf=R_surf, P_cmb=P_cmb, P_surf=P_surf)

    def quantity_at_staggered_nodes(basic_quantity):
        b = np.asarray(basic_quantity).ravel()
        return 0.5 * (b[:-1] + b[1:])

    mesh.quantity_at_staggered_nodes = quantity_at_staggered_nodes
    return mesh


def _make_phi_vol_test_state(
    mesh,
    eos,
    *,
    mixing=True,
    grav_sep=True,
):
    """EntropyState with grav_sep/mixing flags exposed.

    Used by the negative regression tests below to exercise the
    mushy-zone regime where the historical explicit Φ_vol source was
    largest.
    """
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator
    from aragog.solver.entropy_state import EntropyState

    phase_stag = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    phase_stag.set_pressure(mesh.staggered.pressure)

    phase_basic = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    phase_basic.set_pressure(mesh.basic.pressure)

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
        mixing=mixing,
    )


@needs_eos
@pytest.mark.unit
class TestNoExplicitPhiVolSource:
    """Negative regressions guarding against re-introduction of an
    explicit volumetric segregation source term

        Phi_vol = rho · g · (1/rho_m - 1/rho_s) · (j_cm + j_gm)

    on top of ``state.heating``. The volumetric work this would
    capture is already implicit in the divergence of the Δh-weighted
    mass-flux contributions to ``state._heat_flux`` (chain rule on
    Δh = Δu + P·Δv with hydrostatic ∂P/∂r = -ρg). Adding it
    explicitly double-counts, as the 2026-05-03 7-cell matrix
    demonstrated when F_dil/(-F_int) locked at -1.000 to four digits
    across three independent step caps. SPIDER's energy.c has no
    Φ_vol source either; the Bower 2018 derivation closes via the
    divergence alone.

    Each test below sets up the mushy-zone regime where the historical
    explicit source was largest, then asserts ``state.heating`` stays
    at zero (no other sources are wired in this fixture). A regression
    that reintroduces even a fractional copy of Φ_vol would flip these.
    """

    def test_no_phi_vol_in_mushy_zone_with_jgrav_only(self):
        """Mushy zone with mixing OFF, grav_sep ON: ``state.heating``
        must be exactly zero despite a non-zero ``state._mass_flux``
        from the gravitational separation flux.

        Pre-deletion: heating == g·Δv·j_grav, scale ~1e-9 W/kg.
        Post-deletion: heating == 0 because no source is added.

        Discriminating value: the test asserts max|heating| < 1e-15,
        which is ~6 orders of magnitude tighter than the historical
        H_dil scale, so any reintroduction of even a fractional copy
        would flip this test.
        """
        from aragog.eos.entropy import EntropyEOS

        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = _make_test_mesh_with_staggered_interp(N=N)
        state = _make_phi_vol_test_state(
            mesh,
            eos,
            mixing=False,
            grav_sep=True,
        )

        S_sol = np.asarray(eos.solidus_entropy(mesh.basic.pressure)).ravel()
        S_liq = np.asarray(eos.liquidus_entropy(mesh.basic.pressure)).ravel()
        S0_basic = 0.5 * (S_sol + S_liq)
        S0_basic = S0_basic + 60.0 * np.linspace(-1, 1, mesh.basic.radii.size)
        S0 = mesh.quantity_at_staggered_nodes(S0_basic)
        state.update(S0, 0.0)

        # Confirm the regime: j_grav is non-zero somewhere, so a
        # reintroduced explicit Φ_vol would produce non-zero H here.
        j_grav_max = float(np.max(np.abs(np.asarray(state._mass_flux).ravel())))
        assert j_grav_max > 1e-8, (
            f'Test setup did not engage j_grav (max|j|={j_grav_max:.2e}); '
            f'this fixture is supposed to exercise the regime where the '
            f'historical explicit Φ_vol would have been largest.'
        )
        # And yet heating must be exactly zero.
        H_max = float(np.max(np.abs(np.asarray(state.heating).ravel())))
        assert H_max < 1e-15, (
            f'Mushy regime with grav_sep ON produced non-zero '
            f'state.heating (max|H|={H_max:.2e}); the explicit Φ_vol '
            f'source must stay deleted.'
        )

    def test_no_phi_vol_in_mushy_zone_with_jmix_plus_jgrav(self):
        """Mushy zone with both mixing and grav_sep ON: ``state.heating``
        must be exactly zero despite non-zero j_grav AND j_mix.

        This is the regime where the historical explicit
        Φ_vol = g·Δv·(j_mix + j_grav) was largest in production
        runs (1 M⊕ rheological-transition layer). The 7-cell matrix
        attractor at F_dil/(-F_int) = -1.000 came from exactly this
        source. Post-deletion: zero.
        """
        from aragog.eos.entropy import EntropyEOS

        eos = EntropyEOS(EOS_DIR)
        N = 30
        mesh = _make_test_mesh_with_staggered_interp(N=N)
        state = _make_phi_vol_test_state(
            mesh,
            eos,
            mixing=True,
            grav_sep=True,
        )

        S_sol = np.asarray(eos.solidus_entropy(mesh.basic.pressure)).ravel()
        S_liq = np.asarray(eos.liquidus_entropy(mesh.basic.pressure)).ravel()
        S0_basic = 0.5 * (S_sol + S_liq)
        S0_basic = S0_basic + 80.0 * np.linspace(-1, 1, mesh.basic.radii.size)
        S0 = mesh.quantity_at_staggered_nodes(S0_basic)
        state.update(S0, 0.0)

        # Both fluxes engaged?
        L_basic = np.asarray(state.phase_basic.latent_heat()).ravel()
        jmix_heat = np.asarray(state._jmix_heat).ravel()
        j_mix = np.where(np.abs(L_basic) > 1.0, jmix_heat / np.maximum(L_basic, 1.0), 0.0)
        j_grav = np.asarray(state._mass_flux).ravel()
        assert np.max(np.abs(j_mix)) > 1e-8, (
            f'j_mix did not engage (max|j_mix|={np.max(np.abs(j_mix)):.2e})'
        )
        assert np.max(np.abs(j_grav)) > 1e-8, (
            f'j_grav did not engage (max|j_grav|={np.max(np.abs(j_grav)):.2e})'
        )
        # state.heating must STILL be zero — no Φ_vol added.
        H_max = float(np.max(np.abs(np.asarray(state.heating).ravel())))
        assert H_max < 1e-15, (
            f'Mushy regime with both fluxes engaged produced non-zero '
            f'state.heating (max|H|={H_max:.2e}); the explicit Φ_vol '
            f'source must stay deleted.'
        )


# -- Tier 1: Mass-coordinate transform identity ------------------------------
#
# Aragog supports a uniform-mass-coordinate basic grid (``mass_coordinates =
# True``), an extension over the SPIDER reference. Verify:
#   (a) when ``rho*(r) = const``, the transform reduces to identity
#       (xi = r, dxi/dr = 1);
#   (b) Newton round-trip ``r(xi(r)) = r`` to machine precision;
#   (c) for Adams-Williamson EOS (rho* increases with depth), the basic
#       radii on the uniform-xi grid are concentrated toward the CMB
#       (where mass is densest).


@pytest.mark.unit
class TestMassCoordinates:
    """First-principles tests of the mass-coordinate transform in
    ``aragog.mesh``. These don't depend on the PALEOS EOS tables — only
    on the Adams-Williamson pressure profile and the Newton solve in
    ``Mesh.__init__``.
    """

    @staticmethod
    def _build_parameters(*, mass_coordinates, rho_top=4090.0, K_S=260e9, N=40):
        """Construct a minimal Parameters object suitable for building
        a Mesh. Bypasses ConfigParser — keeps the test self-contained.
        """
        from aragog.parser import (
            Parameters,
            _BoundaryConditionsParameters,
            _EnergyParameters,
            _InitialConditionParameters,
            _MeshParameters,
            _PhaseMixedParameters,
            _PhaseParameters,
            _SolverParameters,
        )

        solver = _SolverParameters(
            start_time=0.0,
            end_time=1.0,
            atol=1e-6,
            rtol=1e-6,
            tsurf_poststep_change=30.0,
        )
        bc = _BoundaryConditionsParameters(
            outer_boundary_condition=1,
            outer_boundary_value=1500.0,
            inner_boundary_condition=2,
            inner_boundary_value=0.0,
            emissivity=1.0,
            equilibrium_temperature=273.0,
            core_heat_capacity=880.0,
        )
        mesh_p = _MeshParameters(
            outer_radius=6.371e6,
            inner_radius=3.480e6,
            number_of_nodes=N,
            mixing_length_profile='constant',
            core_density=10738.0,
            eos_method=1,
            surface_density=rho_top,
            gravitational_acceleration=9.81,
            adiabatic_bulk_modulus=K_S,
            mass_coordinates=mass_coordinates,
        )
        energy = _EnergyParameters(
            conduction=True,
            convection=True,
            gravitational_separation=False,
            mixing=False,
            radionuclides=False,
            tidal=False,
        )
        ic = _InitialConditionParameters(
            initial_condition=1,
            surface_temperature=4000.0,
            basal_temperature=4000.0,
        )
        phase_liq = _PhaseParameters(
            density=4000.0,
            viscosity=1e2,
            heat_capacity=1000.0,
            melt_fraction=1.0,
            thermal_conductivity=4.0,
            thermal_expansivity=1e-5,
        )
        phase_sol = _PhaseParameters(
            density=4200.0,
            viscosity=1e21,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=1e-5,
        )
        phase_mix = _PhaseMixedParameters(
            latent_heat_of_fusion=4e6,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='',
            liquidus='',
            phase='mixed',
            phase_transition_width=0.1,
            grain_size=1e-3,
        )
        params = Parameters(
            solver=solver,
            boundary_conditions=bc,
            mesh=mesh_p,
            energy=energy,
            initial_condition=ic,
            phase_liquid=phase_liq,
            phase_solid=phase_sol,
            phase_mixed=phase_mix,
            radionuclides=[],
        )
        return params

    def test_xi_approaches_r_as_density_becomes_uniform(self):
        """As K_S grows large, AW density ρ*(P)=ρ_top·exp(P/K_S) flattens
        toward ρ_top, and the mass-coord mesh approaches the uniform-r
        mesh.

        Discriminating: a buggy transform that did not respect the
        density-weighting would give the SAME mesh at all K values, or
        would diverge with large K. We verify that the L_inf node
        offset DECREASES as K increases (proper trend) and that at
        K = 1e15 Pa (effectively incompressible for a 3000 km mantle)
        the offset is < 0.1% of the mantle thickness.
        """
        from aragog.mesh import Mesh

        offsets = []
        K_values = [260e9, 1e12, 1e15]
        for K in K_values:
            params_uni = self._build_parameters(mass_coordinates=False, K_S=K, N=20)
            params_xi = self._build_parameters(mass_coordinates=True, K_S=K, N=20)
            mesh_uni = Mesh(params_uni)
            mesh_xi = Mesh(params_xi)
            r_uni = np.asarray(mesh_uni.basic.radii).ravel()
            r_xi = np.asarray(mesh_xi.basic.radii).ravel()
            mantle_thick = r_uni[-1] - r_uni[0]
            offsets.append(float(np.max(np.abs(r_xi - r_uni)) / mantle_thick))

        # Monotone trend toward identity (transform recovers no-op limit)
        assert offsets[-1] < offsets[0], (
            f'Mass-coord mesh did not approach uniform-r mesh as K→∞: '
            f'offsets {offsets} should be monotonically decreasing.'
        )
        # At K = 1e15 Pa, AW density variation across the mantle is
        # < exp(P_cmb / K) ≈ exp(1e11 / 1e15) ≈ 1.0001, i.e. < 0.01%.
        # The mass-coord and uniform-r meshes must coincide to that
        # accuracy (with some Newton-xtol slop).
        assert offsets[-1] < 1e-3, (
            f'At K = 1e15 Pa the mass-coord mesh should match uniform-r '
            f'to < 0.1% of mantle thickness; got {offsets[-1]:.2e}.'
        )

    def test_xi_grid_concentrates_at_cmb(self):
        """Adams-Williamson with realistic K_S (260 GPa): ρ*(P) increases
        with depth (Adams-Williamson is exponential), so the uniform-mass
        grid should place MORE basic nodes near the CMB than a uniform-r
        grid would.

        Discriminating test: compares the mean radial spacing in the
        bottom half of the mantle to that in the top half. For a uniform-
        r grid, both are equal. For a uniform-xi grid with monotone
        ρ*(r), the bottom half must have *smaller* mean spacing.
        """
        from aragog.mesh import Mesh

        params_xi = self._build_parameters(mass_coordinates=True, K_S=260e9, N=40)
        mesh_xi = Mesh(params_xi)
        r = np.asarray(mesh_xi.basic.radii).ravel()
        dr = np.diff(r)
        n_half = len(dr) // 2

        mean_dr_bottom = float(np.mean(dr[:n_half]))
        mean_dr_top = float(np.mean(dr[n_half:]))
        # Adams-Williamson rho* grows with depth, so cells should be
        # tighter near the CMB. Require a clearly resolvable contrast,
        # not just a sign — to discriminate against a near-uniform mesh
        # produced by a buggy transform.
        assert mean_dr_bottom < 0.95 * mean_dr_top, (
            f'Mass-coord mesh did not concentrate cells at the CMB: '
            f'mean dr (bottom half) = {mean_dr_bottom:.3e} m vs '
            f'mean dr (top half) = {mean_dr_top:.3e} m. '
            f'Ratio = {mean_dr_bottom / mean_dr_top:.4f}; expected < 0.95.'
        )

    def test_xi_round_trip_via_analytical_mass_integral(self):
        """The Newton solve in ``Mesh.__init__`` uses the analytical AW
        mass integral ``eos.get_mass_within_radii`` to form ``xi(r)``.
        Re-evaluating ``xi(r)`` at the resulting basic radii via the
        same formula must reproduce the input uniform-xi grid to
        Newton xtol (~1 m on a 3 Mm mantle, < 1e-6 relative).

        The discrete-shell helper
        ``get_basic_mass_coordinates_from_spatial_coordinates`` uses a
        shell-by-shell Riemann sum that has its own discretisation
        error and is NOT the inverse of the Newton solve; comparing
        against it would produce a meaningless ~0.5% gap (verified
        before the round-trip was rewritten).
        """
        from aragog.mesh import Mesh

        params_xi = self._build_parameters(mass_coordinates=True, K_S=260e9, N=30)
        mesh_xi = Mesh(params_xi)

        xi_basic = np.asarray(mesh_xi.basic.mass_radii).ravel()
        r_basic = np.asarray(mesh_xi.basic.radii).ravel()
        r_core = float(r_basic[0])
        # rho_avg = mantle-average density used inside Mesh.__init__.
        # Recompute via the same EOS path, sans 4pi.
        r_surf = float(r_basic[-1])
        M_4pi = (
            mesh_xi.eos.get_mass_within_radii(np.array([[r_surf]]))
            - mesh_xi.eos.get_mass_within_radii(np.array([[r_core]]))
        ).item()
        M_no4pi_total = M_4pi / (4.0 * np.pi)
        mantle_volume_no4pi = (r_surf**3 - r_core**3) / 3.0
        rho_avg = M_no4pi_total / mantle_volume_no4pi

        # xi(r_i) per the analytical formula used by Mesh.__init__.
        xi_recomputed = np.empty_like(r_basic)
        for j, r in enumerate(r_basic):
            M_shell_4pi = mesh_xi.eos.get_mass_within_radii(np.array([[float(r)]])).item()
            M_shell = M_shell_4pi / (4.0 * np.pi)
            xi_recomputed[j] = (r_core**3 + 3.0 * M_shell / rho_avg) ** (1.0 / 3.0)

        # 2 m abs (twice Newton xtol). The CMB and surface are exact
        # by construction; interior nodes carry the brentq tolerance.
        np.testing.assert_allclose(
            xi_recomputed,
            xi_basic,
            atol=2.0,
            err_msg='Round-trip xi(r_i) does not recover the uniform-xi '
            'grid via the same analytical formula the Newton '
            'solve targeted.',
        )

    def test_xi_grid_is_uniform_in_mass(self):
        """The basic xi grid is uniform by construction; verify the
        spacing is constant to floating-point precision (not just
        approximately uniform after the Newton solve)."""
        from aragog.mesh import Mesh

        params = self._build_parameters(mass_coordinates=True, K_S=260e9, N=25)
        mesh = Mesh(params)
        xi = np.asarray(mesh.basic.mass_radii).ravel()
        dxi = np.diff(xi)
        # Uniform: max - min is a few ulps of the spacing.
        assert (np.max(dxi) - np.min(dxi)) / np.mean(dxi) < 1e-12, (
            f'Basic xi grid is not uniform: dxi range '
            f'[{np.min(dxi):.6e}, {np.max(dxi):.6e}] '
            f'vs mean {np.mean(dxi):.6e}.'
        )

    def test_unphysical_negative_node_count_rejected(self):
        """Sanity check: requesting an unphysical mesh (one node) must
        not silently succeed — the basic-node array would be a
        single point with no spacing.
        """
        from aragog.mesh import Mesh

        params = self._build_parameters(mass_coordinates=True, K_S=260e9, N=1)
        # The Mesh constructor either raises or produces a degenerate
        # mesh. Either way, dxi/dr cannot be a useful array of length 0.
        try:
            mesh = Mesh(params)
        except (ValueError, IndexError, ZeroDivisionError):
            return  # acceptable — implementation refused the unphysical
        # If it returned, the mesh must have a degenerate basic.radii
        # array (length 1). A non-trivial mesh with N=1 would indicate
        # silent fallback.
        n_basic = np.asarray(mesh.basic.radii).ravel().size
        assert n_basic <= 2, (
            f'N=1 produced a {n_basic}-point basic mesh; expected '
            f'failure or degenerate mesh, not silent expansion.'
        )
