"""Rebuild the mesh of every stored study run; write radii and staggered pressure to a JSON file."""
import json, platform, sys
import numpy as np
sys.path.insert(0, '.')
from tests.test_mesh_refinement import _mesh
from tests.test_mesh_refinement_study import _study

out = {'platform': platform.platform(), 'machine': platform.machine(), 'numpy': np.__version__}
import pathlib; out['cpu'] = pathlib.Path('cpu.txt').read_text().strip() if pathlib.Path('cpu.txt').exists() else ''
import scipy; out['scipy'] = scipy.__version__
for name, v in _study().items():
    if name == 'header':
        continue
    m = _mesh(v['surface_cell_m'], v['cmb_cell_m'], True, v['nodes'])
    rb, rs, P = (np.ravel(x).tolist() for x in (m.basic.radii, m.staggered.radii, m.staggered_pressure))
    out[name] = {'r_basic': rb, 'r_stag': rs, 'P_stag': P}
    dP = np.abs(np.array(P) - v['P_cell_Pa'])
    print(f"{name:12s} max|P - stored| {dP.max():.6g} Pa  h_top rel {abs((rb[-1]-rb[-2])/v['h_top_m']-1):.3e}  h_cmb rel {abs((rb[1]-rb[0])/v['h_cmb_m']-1):.3e}")
open(sys.argv[1], 'w').write(json.dumps(out))
