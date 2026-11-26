"""Build ``tests/validation/data/cmb_eddy_flux_runs.npz`` from the coupled PROTEUS runs of the
CMB eddy-flux Validation page.

Each run couples this aragog (``energy_balance`` core, JAX right-hand side) to PROTEUS. The runner
of the runs prints one line per call with the CMB ledger of the call (``self.cmb_ledger``) and, for
the case 38 resume, the CMB power at S1 = S0 +- 1e-9 and +- 1e-12 J/kg/K at every output point of
the first calls. The file holds, per run, the helpfile columns Time, F_cmb, T_cmb and
step_dE_F_cmb_J of the rows after the resume time, |r|/B of each ledger call, the status, and the
sweep values. The provenance (run descriptions, aragog and PROTEUS commits, checksums) goes to
``cmb_eddy_flux_runs_provenance.json``.

Usage: ``python tests/reference/build_cmb_eddy_flux_runs.py <runs.json>``, where ``runs.json``
maps each key of ``RUNS`` to ``{"helpfile": ..., "log": ..., "status": ..., "run": ...,
"aragog": ..., "proteus": ...}`` (``run`` describes the case) and the key ``v9_parts_log`` to the
log of the V9 resume that prints the in-call CMB flux of each call. The provenance file holds the
descriptions, the commits and the sha256 of each helpfile and log, not the local paths.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'validation' / 'data' / 'cmb_eddy_flux_runs.npz'
# key: resume time [yr] (rows after it are kept; -1 keeps every row)
RUNS = {
    'v4_main': 6769704.31,
    'v4_tip': 6769704.31,
    'v6_main': 37854.62,
    'v6_tip': 37854.62,
    'v7_main': -1.0,
    'v7_tip': -1.0,
    'v7_main_graph': -1.0,
    'v9_main': 30169.30,
    'v9_tip': 30169.30,
}
LEDGER = re.compile(
    r"ledger t1=([-0-9.e+]+) terms=\{'r': np\.float64\(([-0-9.e+]+)\), "
    r"'bound': np\.float64\(([-0-9.e+]+)\)"
)
COLUMNS = ('Time', 'F_cmb', 'T_cmb', 'step_dE_F_cmb_J')


def _helpfile(path: str, t0: float) -> dict[str, np.ndarray]:
    """Columns of ``COLUMNS`` of the rows after ``t0``."""
    lines = Path(path).read_text().splitlines()
    head = lines[0].split('\t')
    idx = [head.index(c) for c in COLUMNS]
    rows = [[float(x) for x in (ln.split('\t')[i] for i in idx)] for ln in lines[1:]]
    arr = np.array([r for r in rows if r[0] > t0])
    return {c: arr[:, k] for k, c in enumerate(COLUMNS)}


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    out: dict[str, np.ndarray] = {}
    prov: dict[str, dict] = {}
    for key, t0 in RUNS.items():
        s = spec[key]
        for col, values in _helpfile(s['helpfile'], t0).items():
            out[f'{key}/{col}'] = values
        log = Path(s['log']).read_text()
        led = np.array(
            [
                [float(m.group(1)), float(m.group(2)) / float(m.group(3))]
                for m in LEDGER.finditer(log)
            ]
        ).reshape(-1, 2)
        out[f'{key}/ledger_t1'], out[f'{key}/ledger_rB'] = led[:, 0], led[:, 1]
        out[f'{key}/status'] = np.array(int(Path(s['status']).read_text().split()[0]))
        sweep = [
            json.loads(ln[len('VRUN sweep ') :])
            for ln in log.splitlines()
            if ln.startswith('VRUN sweep ')
        ]
        if sweep:
            pts = [p for r in sweep for p in r['points']]
            out[f'{key}/sweep_t'] = np.array([p['t'] for p in pts])
            for step in ('1e-09', '1e-12'):
                out[f'{key}/sweep_P_{step}'] = np.array([p['P'][step] for p in pts])
        prov[key] = {k: s[k] for k in ('run', 'aragog', 'proteus')}
        for k in ('helpfile', 'log'):
            prov[key][f'sha256_{k}'] = hashlib.sha256(Path(s[k]).read_bytes()).hexdigest()
    parts = [
        json.loads(ln[len('V9PARTS ') :])
        for ln in Path(spec['v9_parts_log']).read_text().splitlines()
        if ln.startswith('V9PARTS {')
    ]
    for name in ('t1', 'F_max', 'F_end', 'F_avg'):
        out[f'v9_tip/call_{name}'] = np.array([p[name] for p in parts])
    out['v9_tip/call_rB'] = np.array([p['r'] / p['bound'] for p in parts])
    out['v9_tip/call_Etrap_B'] = np.array([p['E_trap'] / p['bound'] for p in parts])
    prov['v9_parts_log_sha256'] = hashlib.sha256(
        Path(spec['v9_parts_log']).read_bytes()
    ).hexdigest()
    np.savez_compressed(OUT, **out)
    OUT.with_name('cmb_eddy_flux_runs_provenance.json').write_text(
        json.dumps(prov, indent=1) + '\n'
    )
    print(OUT, len(out), 'arrays')


if __name__ == '__main__':
    main(sys.argv[1])
