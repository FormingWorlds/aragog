#!/usr/bin/env python3
"""Execute Step A3: Bitwise mesh comparison between main and mesh-refinement."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path('/Users/timlichtenberg/work/ssc-verify-task6')
DATA_A = BASE_DIR / 'data' / 'A'
DATA_A3 = DATA_A / 'A3_mesh'
DIR_MAIN = DATA_A3 / 'main'
DIR_REF = DATA_A3 / 'tl_aragog-mesh-refinement'

DIR_MAIN.mkdir(parents=True, exist_ok=True)
DIR_REF.mkdir(parents=True, exist_ok=True)

# Subprocess script to harvest mesh arrays in a given worktree
DUMP_SCRIPT = '''
import sys
from pathlib import Path
import numpy as np
import aragog
from aragog.parser import Parameters
from aragog.mesh import Mesh
from aragog.jax.phase import MeshArrays

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)

cfg_dir = Path(aragog.__file__).parent / 'cfg'
p_solid = Parameters.from_file(cfg_dir / 'abe_solid.toml')
p_mixed = Parameters.from_file(cfg_dir / 'abe_mixed.cfg')

from tests.test_entropy_solver_eos_method2_smoke import _build_parameters_eos_method_2, _write_synthetic_external_mesh
mesh_dat = Path('/tmp/test_mesh_ext.dat')
_write_synthetic_external_mesh(mesh_dat, n=30)
p_eos2 = _build_parameters_eos_method_2(mesh_dat, n_nodes=40)

configs = [
    ('abe_solid', p_solid),
    ('abe_mixed', p_mixed),
    ('eos_method2', p_eos2),
]

for cfg_name, p_base in configs:
    for mode_name, mass_coord in [('radius', False), ('mass', True)]:
        import copy
        p = copy.deepcopy(p_base)
        p.mesh.mass_coordinates = mass_coord
        if hasattr(p.mesh, 'surface_cell_thickness'):
            p.mesh.surface_cell_thickness = 0.0
            p.mesh.cmb_cell_thickness = 0.0
        
        m = Mesh(p)
        ma = MeshArrays.from_numpy_mesh(m)

        arrays = {}
        for k, v in vars(m.basic).items():
            if isinstance(v, np.ndarray):
                arrays[f'basic.{k}'] = np.asarray(v)
        for k, v in vars(m.staggered).items():
            if isinstance(v, np.ndarray):
                arrays[f'staggered.{k}'] = np.asarray(v)
        for k in ['_d_dr_transform', '_quantity_transform', '_dxidr', 'basic_pressure', 'staggered_pressure', 'staggered_effective_density']:
            if hasattr(m, k):
                v = getattr(m, k)
                if isinstance(v, np.ndarray):
                    arrays[f'mesh.{k}'] = v
        for field in getattr(ma, '_fields', vars(ma)):
            v = getattr(ma, field)
            arrays[f'jax.{field}'] = np.asarray(v)

        np.savez_compressed(out_dir / f'{cfg_name}_{mode_name}.npz', **arrays)
'''

def main():
    print("=== Dumping mesh arrays from wt-aragog-main ===", flush=True)
    subprocess.run(
        [sys.executable, '-c', DUMP_SCRIPT, str(DIR_MAIN)],
        cwd=str(BASE_DIR / 'wt-aragog-main'),
        check=True
    )

    print("=== Dumping mesh arrays from wt-aragog-tl_aragog-mesh-refinement ===", flush=True)
    subprocess.run(
        [sys.executable, '-c', DUMP_SCRIPT, str(DIR_REF)],
        cwd=str(BASE_DIR / 'wt-aragog-tl_aragog-mesh-refinement'),
        check=True
    )

    # Compare arrays
    total_arrays = 0
    total_equal = 0
    mismatches = []

    files_main = sorted(DIR_MAIN.glob('*.npz'))
    for fpath in files_main:
        name = fpath.name
        f_main = np.load(fpath)
        f_ref = np.load(DIR_REF / name)
        assert set(f_main.files) == set(f_ref.files), f"Files mismatch in {name}"

        for k in f_main.files:
            total_arrays += 1
            arr1 = f_main[k]
            arr2 = f_ref[k]
            if np.array_equal(arr1, arr2):
                total_equal += 1
            else:
                mismatches.append(f"{name}:{k}")

    print(f"Mesh comparison: {total_equal} / {total_arrays} arrays bitwise identical", flush=True)
    if mismatches:
        print(f"Mismatches: {mismatches}", flush=True)

    # Canary: change one array by 1 ulp and verify compare reports failure
    canary_detected = False
    test_npz = files_main[0]
    f_main = np.load(test_npz)
    arr_orig = f_main['basic.radii']
    arr_perturbed = arr_orig.copy()
    arr_perturbed[0] = np.nextafter(arr_perturbed[0], np.inf)

    if not np.array_equal(arr_orig, arr_perturbed):
        canary_detected = True

    compare_res = {
        'status': 'DONE',
        'n_arrays': total_arrays,
        'n_equal': total_equal,
        'canary_detected': canary_detected,
        'mismatches': mismatches,
    }

    out_json = DATA_A / 'A3_compare.json'
    out_json.write_text(json.dumps(compare_res, indent=2) + '\n')
    print(f"Wrote {out_json}: {compare_res}", flush=True)

if __name__ == '__main__':
    main()
