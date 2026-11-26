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
"aragog": ..., "proteus": ...}`` (``run`` describes the case; a run with the key ``rtol`` has that
CVODE rtol checked in its log and stored), the key ``v9x_copy`` to the helpfile, status, versions
and configured ``rtol`` of the copy-rule run of issue #113, the key ``rtol_calls`` to the single
calls rerun from a stored start (``log``, ``helpfile``, ``rtol``, ``call_end`` [yr], ``run``), the
key ``v9_parts_log`` to the
log of the V9 resume that prints the in-call CMB flux of each call, and the key ``v7_probe_log``
to the log of the V7 rule run with the JAX count of CMB evaluations that differ from the copy. The provenance file holds the
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
    'v7_main_repeat': -1.0,
    'v7_tip_callback': -1.0,
    'v9_main': 30169.30,
    'v9_tip': 30169.30,
    'v9x_tip': 30169.30,
    'v9x_tip_r9': 30169.30,
    'v9x_tip_r10': 30169.30,
}
# runs that print the in-call CMB flux of each call in their own log
PARTS_OWN = ('v9x_tip', 'v9x_tip_r9', 'v9x_tip_r10')
NUM = r'([-+0-9.eE]+|nan|inf|-inf)'
LEDGER = re.compile(
    rf"ledger t1={NUM} terms=\{{'r': np\.float64\({NUM}\), 'bound': np\.float64\({NUM}\)"
)
COLUMNS = ('Time', 'F_cmb', 'T_cmb', 'T_cmb_node', 'step_dE_F_cmb_J')


def _helpfile(path: str, t0: float) -> dict[str, np.ndarray]:
    """Columns of ``COLUMNS`` of the rows after ``t0``."""
    lines = Path(path).read_text().splitlines()
    head = lines[0].split('\t')
    idx = [head.index(c) for c in COLUMNS]
    rows = [[float(x) for x in (ln.split('\t')[i] for i in idx)] for ln in lines[1:]]
    arr = np.array([r for r in rows if r[0] > t0])
    return {c: arr[:, k] for k, c in enumerate(COLUMNS)}


def _parts(path: str) -> list[dict]:
    """The in-call records ``V9PARTS {...}`` of a run log."""
    return [
        json.loads(ln[len('V9PARTS ') :])
        for ln in Path(path).read_text().splitlines()
        if ln.startswith('V9PARTS {')
    ]


def _check_rtol(name: str, log: str, rtol: str) -> None:
    """Refuse a run whose CVODE options hold another rtol than ``rtol``."""
    rtols = set(re.findall(r"'rtol': ([0-9.e+-]+)", log))
    if rtols != {repr(float(rtol))}:
        raise ValueError(f'{name}: CVODE rtol {rtols}, not {rtol}')


def _store_parts(out: dict, key: str, parts: list[dict]) -> None:
    """Per-call CMB flux and ledger terms of ``parts`` under ``key``."""
    for name in ('t1', 'F_max', 'F_end', 'F_avg', 'g_cmb_end'):
        out[f'{key}/call_{name}'] = np.array([p[name] for p in parts])
    out[f'{key}/call_rB'] = np.array([p['r'] / p['bound'] for p in parts])
    out[f'{key}/call_Etrap_B'] = np.array([p['E_trap'] / p['bound'] for p in parts])


def _table(path: str) -> dict[str, np.ndarray]:
    """Every numeric column of a helpfile."""
    lines = Path(path).read_text().splitlines()
    head = lines[0].split('\t')
    rows = np.array([[float(x) for x in ln.split('\t')] for ln in lines[1:]])
    return {c: rows[:, k] for k, c in enumerate(head)}


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
        n_lines = log.count('ledger t1=')
        if n_lines == 0 or led.shape[0] != n_lines:
            raise ValueError(f'{key}: {n_lines} ledger lines, {led.shape[0]} parsed')
        out[f'{key}/ledger_t1'], out[f'{key}/ledger_rB'] = led[:, 0], led[:, 1]
        out[f'{key}/status'] = np.array(int(Path(s['status']).read_text().split()[0]))
        if 'rtol' in s:
            _check_rtol(key, log, s['rtol'])
            out[f'{key}/rtol'] = np.array(float(s['rtol']))
        sweep = [
            json.loads(ln[len('VRUN sweep ') :])
            for ln in log.splitlines()
            if ln.startswith('VRUN sweep ')
        ]
        if sweep:
            pts = [p for r in sweep for p in r['points']]
            out[f'{key}/sweep_t'] = np.array([p['t'] for p in pts])
            out[f'{key}/sweep_calls'] = np.array(len(sweep))
            for step in ('1e-09', '1e-12'):
                out[f'{key}/sweep_P_{step}'] = np.array([p['P'][step] for p in pts])
        prov[key] = {k: s[k] for k in ('run', 'aragog', 'proteus')}
        for k in ('helpfile', 'log'):
            prov[key][f'sha256_{k}'] = hashlib.sha256(Path(s[k]).read_bytes()).hexdigest()
    _store_parts(out, 'v9_tip', _parts(spec['v9_parts_log']))
    for key in PARTS_OWN:
        parts = _parts(spec[key]['log'])
        if len(parts) != out[f'{key}/ledger_rB'].size:
            raise ValueError(
                f'{key}: {len(parts)} call records, {out[f"{key}/ledger_rB"].size} calls'
            )
        _store_parts(out, key, parts)
        # every CVODE call of the run, so also each segment of a rate-cap call
        steps = re.findall(r'CVODE stats: nst=(\d+)', Path(spec[key]['log']).read_text())
        out[f'{key}/cvode_steps'] = np.array(sum(int(n) for n in steps))
    # the copy rule in the run that the #113 table comes from (an older build, no ledger lines)
    s = spec['v9x_copy']
    cols = _helpfile(s['helpfile'], 30169.30)
    keep = cols['Time'] <= 60000.5
    for col in COLUMNS:
        out[f'v9x_copy/{col}'] = cols[col][keep]
    out['v9x_copy/status'] = np.array(int(Path(s['status']).read_text().split()[0]))
    prov['v9x_copy'] = {k: s[k] for k in ('run', 'aragog', 'proteus', 'rtol')}
    prov['v9x_copy']['sha256_helpfile'] = hashlib.sha256(
        Path(s['helpfile']).read_bytes()
    ).hexdigest()
    # single calls rerun from a stored start at several rtol: |r|/B and the CMB heat of the call
    for name, c in spec['rtol_calls'].items():
        log = Path(c['log']).read_text()
        _check_rtol(name, log, c['rtol'])
        calls = [
            m for m in LEDGER.finditer(log) if abs(float(m.group(1)) - c['call_end']) < 1.0e-3
        ]
        if len(calls) != 1:
            raise ValueError(
                f'{name}: {len(calls)} ledger lines at the call end {c["call_end"]}'
            )
        t1 = float(calls[0].group(1))
        rb = float(calls[0].group(2)) / float(calls[0].group(3))
        rows = _helpfile(c['helpfile'], -1.0)
        k = int(np.argmin(np.abs(rows['Time'] - t1)))
        if abs(rows['Time'][k] - t1) > 1.0e-6:
            raise ValueError(f'{name}: no helpfile row at the call end {t1}')
        out[f'rtol_calls/{name}/rtol'] = np.array(float(c['rtol']))
        out[f'rtol_calls/{name}/t1'] = np.array(t1)
        out[f'rtol_calls/{name}/rB'] = np.array(rb)
        out[f'rtol_calls/{name}/q'] = np.array(rows['step_dE_F_cmb_J'][k])
        prov[f'rtol_calls/{name}'] = {
            'run': c['run'],
            'sha256_log': hashlib.sha256(log.encode()).hexdigest(),
        }
    probe = Path(spec['v7_probe_log']).read_text()
    counters = re.findall(r'V7PROBE evals (\d+) differing \{(.*?)\}', probe)
    if len(counters) != 1:
        raise ValueError(f'v7 probe log: {len(counters)} counter lines, not 1')
    out['v7_tip/jax_evals'] = np.array(int(counters[0][0]))
    out['v7_tip/jax_differing'] = np.array(
        sum(int(n) for n in re.findall(r'\): (\d+)', counters[0][1]))
    )
    kinds_lines = re.findall(r'VRUN kinds (.*)', Path(spec['v7_tip']['log']).read_text())
    if len(kinds_lines) != 1:
        raise ValueError(f'v7 tip log: {len(kinds_lines)} kind lines, not 1')
    kinds = re.findall(r"\(\('(\w)', '(same|differs)'\), (\d+)\)", kinds_lines[0])
    out['v7_tip/numpy_evals'] = np.array(sum(int(n) for _, _, n in kinds))
    out['v7_tip/numpy_differing'] = np.array(sum(int(n) for _, d, n in kinds if d == 'differs'))
    # whole helpfiles, exact values: pairs of runs and the number of columns that differ
    tables = {
        k: _table(spec[k]['helpfile'])
        for k in ('v7_main', 'v7_tip', 'v7_main_graph', 'v7_main_repeat', 'v7_tip_callback')
    }
    out['v7_columns_total'] = np.array(len(tables['v7_main']))
    for name, (x, y) in {
        'v7_tip_vs_main': ('v7_main', 'v7_tip'),
        'v7_graph_vs_main': ('v7_main', 'v7_main_graph'),
        'v7_tip_vs_graph': ('v7_main_graph', 'v7_tip'),
        'v7_repeat_vs_main': ('v7_main', 'v7_main_repeat'),
        'v7_callback_vs_main': ('v7_main', 'v7_tip_callback'),
    }.items():
        a, b = tables[x], tables[y]
        diff = [c for c in a if not np.array_equal(a[c], b[c], equal_nan=True)]
        out[f'{name}/columns_differ'] = np.array(len(diff))
        out[f'{name}/only_runtime'] = np.array(diff == ['runtime'])
    prov['numpy'] = np.__version__
    prov['v7_probe_log_sha256'] = hashlib.sha256(probe.encode()).hexdigest()
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
