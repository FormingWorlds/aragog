"""Integration tests for bundled configuration dispatch with full solver execution."""

from __future__ import annotations

import importlib.resources

import pytest

from tests.conftest import entropy_eos_copy, needs_eos

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@needs_eos
@pytest.mark.parametrize('cfg_name', ['abe_solid.toml', 'abe_solid.cfg', 'abe_mixed.cfg'])
def test_rheology_off_bundled_configs_bitwise_parity(tmp_path, cfg_name):
    """Verify bundled configs default to rheology.enabled=False and match explicit off bitwise."""
    import dataclasses
    import math
    from pathlib import Path

    import numpy as np

    from aragog.config import Config
    from aragog.parser import Parameters
    from aragog.rheology import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver, SolverOutput

    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath(f'cfg/{cfg_name}')
    ) as cfg_path:
        p_def = Parameters.from_file(cfg_path)
        c_def = Config.from_file(str(cfg_path))
        raw_text = Path(cfg_path).read_text(encoding='utf-8')

    assert p_def.phase_solid.rheology == SolidRheologyParams(enabled=False)
    assert p_def.phase_solid.enabled is False
    assert c_def.phase_solid.rheology == SolidRheologyParams(enabled=False)
    assert c_def.phase_solid.enabled is False

    is_toml = cfg_name.endswith('.toml')
    bool_false = 'false' if is_toml else 'False'
    mode_local = '"local"' if is_toml else 'local'
    lid_fixed = '"fixed"' if is_toml else 'fixed'
    explicit_off_block = (
        f'\n[phase_solid.rheology]\n'
        f'enabled = {bool_false}\n'
        f'g_homologous = 18.0\n'
        f't_sol_0 = 1500.0\n'
        f'yield_stress_c = 20.0e6\n'
        f'stress_closure_mode = {mode_local}\n'
        f'lid_base_mode = {lid_fixed}\n'
        f's_top_inf = 0.18\n'
    )
    explicit_path = tmp_path / cfg_name
    explicit_path.write_text(raw_text + explicit_off_block, encoding='utf-8')
    c_exp = Config.from_file(str(explicit_path))
    assert c_exp.phase_solid.rheology.enabled is False
    assert c_exp.phase_solid.g_homologous == 18.0
    assert c_exp.phase_solid.stress_closure_mode == 'local'

    eos = entropy_eos_copy()
    c_def.solver.end_time = 1.0
    c_exp.solver.end_time = 1.0
    s_init = 2900.0

    solver_def = EntropySolver(c_def, entropy_eos=eos)
    solver_def.initialize()
    solver_def.set_initial_entropy(s_init)
    solver_def.solve()
    state_def = solver_def.get_state()

    solver_exp = EntropySolver(c_exp, entropy_eos=eos)
    solver_exp.initialize()
    solver_exp.set_initial_entropy(s_init)
    solver_exp.solve()
    state_exp = solver_exp.get_state()

    array_count = 0
    scalar_count = 0
    for f in dataclasses.fields(SolverOutput):
        val_a = getattr(state_def, f.name)
        val_b = getattr(state_exp, f.name)
        if isinstance(val_a, np.ndarray):
            array_count += 1
            assert np.array_equal(val_a, val_b), (
                f'Bitwise mismatch on array {f.name} in {cfg_name}'
            )
        elif isinstance(val_a, float) and math.isnan(val_a):
            scalar_count += 1
            assert isinstance(val_b, float) and math.isnan(val_b)
        else:
            scalar_count += 1
            assert val_a == val_b, (
                f'Scalar mismatch on {f.name} in {cfg_name}: {val_a} != {val_b}'
            )
    assert array_count + scalar_count == len(dataclasses.fields(SolverOutput))
    assert array_count > 0
    assert scalar_count > 0

    c_canary = Config.from_file(str(explicit_path))
    c_canary.solver.end_time = 1.0
    solver_canary = EntropySolver(c_canary, entropy_eos=eos)
    solver_canary.initialize()
    solver_canary.set_initial_entropy(s_init * (1.0 + 1.0e-12))
    solver_canary.solve()
    state_canary = solver_canary.get_state()
    assert not np.array_equal(state_def.S_final, state_canary.S_final)
