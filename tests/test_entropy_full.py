"""Full integration test: EntropySolver with Aragog config pipeline.

Creates an EntropySolver using the standard Aragog Parameters + Evaluator
pipeline (real mesh, real BCs), but with entropy as state variable.

Runs a 10,000 yr grey-body cooling from S=3200 and compares against
SPIDER expectations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver
from aragog.parser import Parameters


def run_full_test():
    # Find EOS tables
    eos_dir = Path('/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos')
    if not eos_dir.exists():
        print(f'SKIP: SPIDER P-S tables not found at {eos_dir}')
        return

    # Load config from the standard test file
    config_file = Path(__file__).parent.parent / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml'
    if not config_file.exists():
        print(f'SKIP: Config not found at {config_file}')
        return

    print(f'Config: {config_file}')
    print(f'EOS: {eos_dir}')

    # Parse parameters
    parameters = Parameters.from_file(config_file)

    # Override solver settings for a reasonable test
    parameters.solver.start_time = 0
    parameters.solver.end_time = 10000  # 10 kyr
    parameters.solver.atol = 1.0  # entropy in J/kg/K
    parameters.solver.rtol = 1e-6

    # Override boundary conditions
    parameters.boundary_conditions.outer_boundary_condition = 1  # grey-body
    parameters.boundary_conditions.equilibrium_temperature = 255.0
    parameters.boundary_conditions.emissivity = 1.0
    parameters.boundary_conditions.inner_boundary_condition = 2  # prescribed flux
    parameters.boundary_conditions.inner_boundary_value = 0.0  # insulating CMB

    # Load entropy EOS
    entropy_eos = EntropyEOS(eos_dir)

    # Patch: set phase to "solid" to avoid loading melting curve files
    # The entropy solver doesn't use T-based phase evaluators anyway.
    parameters.phase_mixed.solidus = ''
    parameters.phase_mixed.liquidus = ''

    # Create and initialize solver
    solver = EntropySolver(parameters, entropy_eos)
    solver.initialize()

    r_stag = np.asarray(solver.evaluator.mesh.staggered.radii).flatten()
    print(f'\nMesh: {len(r_stag)} staggered nodes')
    print(f'  R_inner = {r_stag[0]/1e3:.0f} km')
    print(f'  R_outer = {r_stag[-1]/1e3:.0f} km')

    # Set isentropic IC
    S_init = 3200.0
    solver.set_initial_entropy(S_init)

    # Check initial state
    solver.state.update(solver._S0, 0.0)
    T_surf_init = solver.state.top_temperature.item()
    T_cmb_init = solver.state.bottom_temperature.item()
    phi_surf = solver.state.phase_staggered.melt_fraction()[-1]
    print(f'\nInitial state (S={S_init:.0f}):')
    print(f'  T_surface = {T_surf_init:.0f} K')
    print(f'  T_CMB     = {T_cmb_init:.0f} K')
    print(f'  phi_surf  = {phi_surf:.3f}')

    # Solve
    print(f'\nSolving (0 to {parameters.solver.end_time:.0f} yr)...')
    solver.solve()

    if solver.stop_early:
        print('  WARNING: solver stopped early')

    sol = solver.solution
    print(f'  Status: {sol.status} ({sol.message})')
    print(f'  Steps: {len(sol.t)}')
    print(f'  Time range: {sol.t[0]:.1f} to {sol.t[-1]:.1f} yr')

    # Sample final state
    S_final = sol.y[:, -1]
    solver.state.phase_staggered.set_entropy(S_final)
    solver.state.phase_staggered.update()
    T_surf_final = solver.state.phase_staggered.temperature()[-1]
    phi_surf_final = solver.state.phase_staggered.melt_fraction()[-1]
    S_surf_final = S_final[-1]
    S_mid_final = S_final[len(S_final) // 2]

    print(f'\nFinal state (t={sol.t[-1]:.0f} yr):')
    print(f'  T_surface = {T_surf_final:.0f} K (was {T_surf_init:.0f})')
    print(f'  S_surface = {S_surf_final:.0f} (was {S_init:.0f})')
    print(f'  S_mid     = {S_mid_final:.0f} (was {S_init:.0f})')
    print(f'  phi_surf  = {phi_surf_final:.3f} (was {phi_surf:.3f})')

    # Sample evolution at intermediate times
    print(f'\nEvolution:')
    for t_sample in [0, 100, 1000, 5000, 10000]:
        if t_sample > sol.t[-1]:
            break
        S_at_t = sol.sol(t_sample)
        solver.state.phase_staggered.set_entropy(S_at_t)
        solver.state.phase_staggered.update()
        T = solver.state.phase_staggered.temperature()[-1]
        phi = solver.state.phase_staggered.melt_fraction()[-1]
        print(f'  t={t_sample:6.0f} yr: T_surf={T:.0f} K, S_surf={S_at_t[-1]:.0f}, phi={phi:.3f}')

    # Assertions
    assert sol.status == 0, f'Solver failed: {sol.message}'
    assert T_surf_final < T_surf_init, f'Surface did not cool: {T_surf_final} >= {T_surf_init}'
    assert S_surf_final < S_init, f'Surface S did not decrease: {S_surf_final} >= {S_init}'
    assert S_mid_final < S_init, f'Interior S did not decrease: {S_mid_final} >= {S_init}'

    print('\nFull entropy solver test PASSED')


if __name__ == '__main__':
    run_full_test()
