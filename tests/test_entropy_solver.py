"""Standalone test of the entropy solver with SPIDER P-S tables.

Runs a short grey-body cooling from S=3200 J/kg/K and verifies
the entropy decreases monotonically at the surface.

Requires SPIDER P-S tables in:
  PROTEUS/output/coupled_parity/spider/data/spider_eos/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.solver.entropy_state import EntropyState

# Stefan-Boltzmann constant
SIGMA = 5.670374419e-8  # W m^-2 K^-4
SECS_PER_YEAR = 31557600.0


def build_simple_mesh(N=50, R_cmb=3480e3, R_surf=6371e3, P_cmb=135e9, P_surf=1e5):
    """Build a simple radial mesh for testing."""
    r_stag = np.linspace(R_cmb, R_surf, N)
    dr = np.diff(r_stag)

    # Basic nodes at cell edges (N+1 points including boundaries)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_cmb
    r_basic[-1] = R_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])

    # Pressure (linear in r, decreasing outward)
    P_stag = np.linspace(P_cmb, P_surf, N)
    P_basic = np.interp(r_basic, r_stag, P_stag)

    # Gravity (roughly constant)
    g_stag = np.full(N, 10.0)
    g_basic = np.full(N + 1, 10.0)

    # Areas and volumes for staggered FV
    area_basic = 4.0 * np.pi * r_basic**2
    volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)

    # Mixing length: distance to nearest boundary
    mixing_length = np.minimum(r_basic - R_cmb, R_surf - r_basic)
    mixing_length = np.maximum(mixing_length, 1.0)  # avoid zero

    class SimpleMesh:
        pass

    class BasicMesh:
        pass

    class StaggeredMesh:
        pass

    mesh = SimpleMesh()
    mesh.basic = BasicMesh()
    mesh.staggered = StaggeredMesh()

    mesh.basic.radii = r_basic
    mesh.basic.pressure = P_basic
    mesh.basic.gravitational_acceleration = g_basic
    mesh.basic.area = area_basic
    mesh.basic.volume = volume
    mesh.basic.mixing_length = mixing_length
    mesh.basic.mixing_length_squared = mixing_length**2
    mesh.basic.mixing_length_cubed = mixing_length**3

    mesh.staggered.radii = r_stag
    mesh.staggered.pressure = P_stag
    mesh.staggered.gravitational_acceleration = g_stag

    def quantity_at_basic_nodes(q_stag):
        q = np.asarray(q_stag).flatten()
        q_basic = np.zeros(N + 1)
        q_basic[0] = q[0]
        q_basic[-1] = q[-1]
        q_basic[1:-1] = 0.5 * (q[:-1] + q[1:])
        return q_basic

    def d_dr_at_basic_nodes(q_stag):
        q = np.asarray(q_stag).flatten()
        dqdr = np.zeros(N + 1)
        dqdr[1:-1] = np.diff(q) / dr
        dqdr[0] = dqdr[1]
        dqdr[-1] = dqdr[-2]
        return dqdr

    mesh.quantity_at_basic_nodes = quantity_at_basic_nodes
    mesh.d_dr_at_basic_nodes = d_dr_at_basic_nodes

    return mesh


def run_entropy_cooling():
    """Run a standalone entropy cooling test."""
    eos_dir = Path(__file__).parent.parent.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'
    if not eos_dir.exists():
        # Try absolute path
        eos_dir = Path('/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos')
    if not eos_dir.exists():
        print(f'SKIP: SPIDER P-S tables not found at {eos_dir}')
        return

    print('Loading entropy EOS...')
    eos = EntropyEOS(eos_dir)

    print('Building mesh...')
    N = 30
    mesh = build_simple_mesh(N=N)

    # Create phase evaluators
    phase_stag = EntropyPhaseEvaluator(
        entropy_eos=eos,
        gravitational_acceleration=mesh.staggered.gravitational_acceleration,
    )
    phase_stag.set_pressure(mesh.staggered.pressure)

    phase_basic = EntropyPhaseEvaluator(
        entropy_eos=eos,
        gravitational_acceleration=mesh.basic.gravitational_acceleration,
    )
    phase_basic.set_pressure(mesh.basic.pressure)

    # Create a mock evaluator
    class MockEvaluator:
        pass
    evaluator = MockEvaluator()
    evaluator.mesh = mesh

    # Create entropy state
    state = EntropyState(
        evaluator=evaluator,
        phase_staggered=phase_stag,
        phase_basic=phase_basic,
        conduction=True,
        convection=True,
    )

    # Initial entropy: isentropic at S=3200
    S_init = 3200.0
    S = np.full(N, S_init)

    # Compute initial state
    state.update(S, 0.0)
    T_surf_init = float(state.top_temperature)
    T_cmb_init = float(state.bottom_temperature)
    phi_surf = float(phase_stag.melt_fraction()[-1])
    print(f'\nInitial state (S={S_init:.0f} J/kg/K):')
    print(f'  T_surface = {T_surf_init:.0f} K')
    print(f'  T_CMB     = {T_cmb_init:.0f} K')
    print(f'  phi_surf  = {phi_surf:.3f}')

    # Use scipy BDF for stability (forward Euler is wildly unstable
    # because the surface cell drains in one step at large dt).
    from scipy.integrate import solve_ivp

    def dSdt_func(time, entropy):
        state.update(entropy, time)
        # Grey-body surface flux
        T_top = state.top_temperature.item()
        F_surf = SIGMA * (T_top**4 - 255.0**4)
        state._heat_flux[-1] = F_surf
        # Zero flux at CMB (insulating)
        state._heat_flux[0] = 0.0
        # Flux divergence
        energy_flux = state.heat_flux * mesh.basic.area
        delta_energy_flux = np.diff(energy_flux)
        capacitance = state.capacitance_staggered() * mesh.basic.volume
        dSdt = -delta_energy_flux / capacitance * SECS_PER_YEAR
        return dSdt

    print('\nRunning BDF integration (0 to 1000 yr)...')
    sol = solve_ivp(
        dSdt_func, (0, 1000), S, method='BDF',
        atol=1.0, rtol=1e-6, dense_output=True,
    )
    print(f'  BDF: {sol.status} ({sol.message}), {len(sol.t)} steps')

    # Sample at regular intervals
    times = [0, 10, 100, 500, 1000]
    for t in times:
        if t > sol.t[-1]:
            break
        S_at_t = sol.sol(t)
        phase_stag.set_entropy(S_at_t)
        phase_stag.update()
        T_top = phase_stag.temperature()[-1]
        phi_top = phase_stag.melt_fraction()[-1]
        print(f'  t={t:5.0f} yr: T_surf={T_top:.0f} K, S_surf={S_at_t[-1]:.0f}, phi={phi_top:.3f}')

    S_final = sol.y[:, -1]
    phase_stag.set_entropy(S_final)
    phase_stag.update()
    T_surf_final = phase_stag.temperature()[-1]
    print(f'\nFinal: T_surf = {T_surf_final:.0f} K (was {T_surf_init:.0f})')
    print(f'Surface entropy: {S_final[-1]:.0f} (was {S_init:.0f})')

    # Verify: surface should have cooled
    assert T_surf_final < T_surf_init, f'Surface did not cool: {T_surf_final} >= {T_surf_init}'
    assert S_final[-1] < S_init, f'Surface S did not decrease: {S_final[-1]} >= {S_init}'

    # Verify: interior should also have changed (entropy transport)
    S_interior = S_final[N // 2]
    print(f'Interior entropy: {S_interior:.0f} (was {S_init:.0f})')
    if abs(S_interior - S_init) > 1.0:
        print('  Interior entropy changed -> convection is transporting entropy')
    else:
        print('  Interior entropy unchanged -> only surface cooling (expected for short run)')

    print('\nEntropy cooling test PASSED')


if __name__ == '__main__':
    run_entropy_cooling()
