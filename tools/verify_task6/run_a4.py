#!/usr/bin/env python3
"""Execute Step A4 bitwise-off checks and write A4_bitwise.csv."""
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path('/Users/timlichtenberg/work/ssc-verify-task6')
DATA_A = BASE_DIR / 'data' / 'A'
DATA_A.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_A / 'A4_bitwise.csv'
EOS_DIR = str(BASE_DIR / 'test-data' / 'spider_eos')

RUNNER_SOLID = '''
import sys, os
from pathlib import Path
import numpy as np

wt = Path(sys.argv[1])
sys.path.insert(0, str(wt / 'src'))
import aragog
assert Path(aragog.__file__).resolve().is_relative_to((wt / 'src').resolve()), aragog.__file__

from aragog.solver.entropy_solver import EntropySolver

eos_dir = sys.argv[2]
perturb = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
tag = sys.argv[4]

cfg = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'
s = EntropySolver.from_file(cfg, eos_dir=eos_dir)
s.parameters.boundary_conditions.outer_boundary_condition = 1
if hasattr(s.parameters.energy, 'phase_boundary_cap'):
    s.parameters.energy.phase_boundary_cap = 'fixed'
s.parameters.energy.use_jax_jacobian = False
s.initialize()

S_init = 3050.0 * (1.0 + perturb)
s.set_initial_entropy(S_init)

t = 0.0
for call in range(3):
    s.parameters.solver.start_time, s.parameters.solver.end_time = t, t + 100.0
    if call > 0:
        s.reset()
        s.set_initial_entropy(S)
    s.solve()
    st = s.get_state()
    S = np.asarray(st.S_final, float)
    t += float(st.dt_actual)

flux = np.asarray(s.state.heat_flux, float)
np.savez('/tmp/a4_' + tag + '.npz', S=S, flux=flux)
'''

RUNNER_SSC = '''
import sys, os
from pathlib import Path
import numpy as np

wt = Path(sys.argv[1])
sys.path.insert(0, str(wt / 'src'))
import aragog
assert Path(aragog.__file__).resolve().is_relative_to((wt / 'src').resolve()), aragog.__file__

from aragog.solver.entropy_solver import EntropySolver

eos_dir = sys.argv[2]
perturb = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
tag = sys.argv[4]

cfg = Path('/tmp/ssc_earth_4p5gyr_slope022.toml')
s = EntropySolver.from_file(cfg, eos_dir=eos_dir)
assert s.parameters.phase_solid.rheology.mlt_top_slope == 0.22
s.parameters.energy.use_jax_jacobian = False
s.initialize()

S_init = 3050.0 * (1.0 + perturb)
s.set_initial_entropy(S_init)

t = 0.0
for call in range(3):
    s.parameters.solver.start_time, s.parameters.solver.end_time = t, t + 100.0
    if call > 0:
        s.reset()
        s.set_initial_entropy(S)
    s.solve()
    st = s.get_state()
    S = np.asarray(st.S_final, float)
    t += float(st.dt_actual)

flux = np.asarray(s.state.heat_flux, float)
dE_int = float(st.step_dE_F_int_J)
dE_cmb = float(st.step_dE_F_cmb_J)
np.savez('/tmp/a4_' + tag + '.npz', S=S, flux=flux, dE_int=dE_int, dE_cmb=dE_cmb)
'''

def main():
    rows = ['branch,option,config,equal,canary_detected']

    # 1. Main baseline for abe_solid
    print("=== Running main baseline for abe_solid ===", flush=True)
    subprocess.run([sys.executable, '-c', RUNNER_SOLID, str(BASE_DIR / 'wt-aragog-main'), EOS_DIR, '0.0', 'main_solid'],
                   cwd=str(BASE_DIR / 'wt-aragog-main'), check=True)
    d_main_solid = np.load('/tmp/a4_main_solid.npz')

    # Test 1: skin branch
    print("=== Running Test 1: tl/aragog-skin-temperature ===", flush=True)
    subprocess.run([sys.executable, '-c', RUNNER_SOLID, str(BASE_DIR / 'wt-aragog-tl_aragog-skin-temperature'), EOS_DIR, '0.0', 'skin'],
                   cwd=str(BASE_DIR / 'wt-aragog-tl_aragog-skin-temperature'), check=True)
    subprocess.run([sys.executable, '-c', RUNNER_SOLID, str(BASE_DIR / 'wt-aragog-tl_aragog-skin-temperature'), EOS_DIR, '1e-9', 'skin_canary'],
                   cwd=str(BASE_DIR / 'wt-aragog-tl_aragog-skin-temperature'), check=True)
    d_skin = np.load('/tmp/a4_skin.npz')
    d_skin_canary = np.load('/tmp/a4_skin_canary.npz')

    eq_skin = np.array_equal(d_main_solid['S'], d_skin['S']) and np.array_equal(d_main_solid['flux'], d_skin['flux'])
    canary_skin = not np.array_equal(d_skin['S'], d_skin_canary['S']) and not np.array_equal(d_skin['flux'], d_skin_canary['flux'])
    rows.append(f"tl/aragog-skin-temperature,outer_boundary_condition=1,abe_solid.toml,{eq_skin},{canary_skin}")
    print(f"Test 1: equal={eq_skin}, canary={canary_skin}", flush=True)

    # Test 2: adaptive phase cap
    print("=== Running Test 2: tl/adaptive-phase-cap ===", flush=True)
    subprocess.run([sys.executable, '-c', RUNNER_SOLID, str(BASE_DIR / 'wt-aragog-tl_adaptive-phase-cap'), EOS_DIR, '0.0', 'cap'],
                   cwd=str(BASE_DIR / 'wt-aragog-tl_adaptive-phase-cap'), check=True)
    subprocess.run([sys.executable, '-c', RUNNER_SOLID, str(BASE_DIR / 'wt-aragog-tl_adaptive-phase-cap'), EOS_DIR, '1e-9', 'cap_canary'],
                   cwd=str(BASE_DIR / 'wt-aragog-tl_adaptive-phase-cap'), check=True)
    d_cap = np.load('/tmp/a4_cap.npz')
    d_cap_canary = np.load('/tmp/a4_cap_canary.npz')

    eq_cap = np.array_equal(d_main_solid['S'], d_cap['S']) and np.array_equal(d_main_solid['flux'], d_cap['flux'])
    canary_cap = not np.array_equal(d_cap['S'], d_cap_canary['S']) and not np.array_equal(d_cap['flux'], d_cap_canary['flux'])
    rows.append(f"tl/adaptive-phase-cap,phase_boundary_cap='fixed',abe_solid.toml,{eq_cap},{canary_cap}")
    print(f"Test 2: equal={eq_cap}, canary={canary_cap}", flush=True)

    # Test 3: ssc-cmb-bl-law vs ssc-mlt-calibration
    print("=== Running Test 3: tl/ssc-cmb-bl-law vs tl/ssc-mlt-calibration ===", flush=True)
    cfg_path = BASE_DIR / 'wt-aragog-tl_ssc-cmb-bl-law' / 'tools' / 'verification' / 'configs' / 'ssc_earth_4p5gyr.toml'
    text = cfg_path.read_text()
    text_mod = text.replace('[phase_solid]\n', '[phase_solid]\nmlt_top_slope = 0.22\n')
    tmp_cfg = Path('/tmp/ssc_earth_4p5gyr_slope022.toml')
    tmp_cfg.write_text(text_mod)

    subprocess.run([sys.executable, '-c', RUNNER_SSC, str(BASE_DIR / 'wt-aragog-tl_ssc-mlt-calibration'), EOS_DIR, '0.0', 'ssc_cal'],
                   cwd=str(BASE_DIR / 'wt-aragog-tl_ssc-mlt-calibration'), check=True)
    subprocess.run([sys.executable, '-c', RUNNER_SSC, str(BASE_DIR / 'wt-aragog-tl_ssc-cmb-bl-law'), EOS_DIR, '0.0', 'ssc_law'],
                   cwd=str(BASE_DIR / 'wt-aragog-tl_ssc-cmb-bl-law'), check=True)
    subprocess.run([sys.executable, '-c', RUNNER_SSC, str(BASE_DIR / 'wt-aragog-tl_ssc-cmb-bl-law'), EOS_DIR, '1e-9', 'ssc_law_canary'],
                   cwd=str(BASE_DIR / 'wt-aragog-tl_ssc-cmb-bl-law'), check=True)

    d_cal = np.load('/tmp/a4_ssc_cal.npz')
    d_law = np.load('/tmp/a4_ssc_law.npz')
    d_canary = np.load('/tmp/a4_ssc_law_canary.npz')

    eq_ssc = (np.array_equal(d_cal['S'], d_law['S']) and
              np.array_equal(d_cal['flux'], d_law['flux']) and
              np.array_equal(d_cal['dE_int'], d_law['dE_int']) and
              np.array_equal(d_cal['dE_cmb'], d_law['dE_cmb']))
    canary_ssc = not np.array_equal(d_law['S'], d_canary['S']) and not np.array_equal(d_law['flux'], d_canary['flux'])
    rows.append(f"tl/ssc-cmb-bl-law,cmb_flux_law=none,ssc_earth_4p5gyr.toml,{eq_ssc},{canary_ssc}")
    print(f"Test 3: equal={eq_ssc}, canary={canary_ssc}", flush=True)

    OUT_CSV.write_text('\n'.join(rows) + '\n')
    print(f"Wrote {OUT_CSV}", flush=True)

if __name__ == '__main__':
    main()
