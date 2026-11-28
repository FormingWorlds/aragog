"""Build ``tests/validation/data/cmb_ledger_repeat_runs.json`` from coupled PROTEUS runs whose
calls the CMB ledger check refuses at rtol 1e-8.

The runs are the PROTEUS layer-trapping tutorial with ``trap_mode = "none"`` (1 Earth mass,
energy_balance core) at the PROTEUS default rtol of 1e-8, each from a start with
``delta_T_super`` changed at round-off level. The run script wraps the ledger check before
``proteus start`` and prints one line per checked call:

.. code-block:: python

    import aragog.solver.entropy_solver as es

    check = es.EntropySolver._check_cmb_ledger

    def hook(self, t_pts, y_arr, p_cmb, q, d_top=None):
        try:
            return check(self, t_pts, y_arr, p_cmb, q, d_top=d_top)
        finally:
            print('VRUN ledger t1=%r terms=%r' % (float(t_pts[-1]), self.cmb_ledger))

    es.EntropySolver._check_cmb_ledger = hook

The log also holds the aragog warning of each repeat (read in the file handler format, once per
repeat) and the ``internal steps`` line of each call, which comes just before its ledger line. A
refused attempt is a ledger line with ``|r| > 10 B`` followed by a line of the same call end
time: that next line is its repeat at rtol 1e-10.

Usage: ``python tests/reference/build_cmb_ledger_repeat_runs.py <run folder> <note> <name>...``
with ``<note>`` a sentence on the commits of the runs. A name ``base:<name>`` is a run of the
same start without the repeat, stored with its end only. Each run needs
``<run folder>/logs/arm_<name>.log``, ``<run folder>/cfg/tlt_none_<name>.toml`` and
``<run folder>/output/tlt_<name>/status``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

OUT = (
    Path(__file__).resolve().parents[1] / 'validation' / 'data' / 'cmb_ledger_repeat_runs.json'
)
LEDGER = re.compile(r'VRUN ledger t1=([\d.eE+-]+) terms=(\{.*?\}|None)')
REPEAT = re.compile(
    r'fwl\.aragog\.solver\.entropy_solver - WARNING.*CMB ledger miss (\S+) above 10x the bound '
    r'(\S+) at rtol (\S+) in the call.*; the repeat at rtol (\S+) (passed|failed)'
)
STEPS = re.compile(r'EntropySolver: (\d+) internal steps')
HEAD = re.compile(r'aragog ([0-9a-f]{40}) proteus ([0-9a-f]{40}) zalmoxis ([0-9a-f]{40})')


def _term(text, key):
    """Value of ``key`` in a printed ``cmb_ledger`` dict."""
    return float(re.search(r"'%s': (?:np\.float64\()?([^,)]+)" % key, text)[1])


def _calls(text):
    """``(t_end, r, bound, steps)`` per ledger line; r and bound None for an unchecked call."""
    calls, steps = [], None
    for line in text.splitlines():
        if m := STEPS.search(line):
            steps = int(m[1])
        elif m := LEDGER.search(line):
            checked = m[2] != 'None' and "'excluded': True" not in m[2]
            r, b = (_term(m[2], 'r'), _term(m[2], 'bound')) if checked else (None, None)
            calls.append((float(m[1]), r, b, steps or 0))
            steps = None
    return calls


def _run(folder, name):
    """Refused attempts with their repeats, a summary and the provenance of one run."""
    log = folder / 'logs' / f'arm_{name}.log'
    text = log.read_text()
    head = HEAD.search(text)
    calls = _calls(text)
    repeats = [(float(m[1]), m[5]) for m in REPEAT.finditer(text)]
    if any(m[3] != '1e-08' or m[4] != '1e-10' for m in REPEAT.finditer(text)):
        raise ValueError(f'{log}: a repeat line with another rtol pair')
    rows, refused = [], set()
    for i, (t, r, b, _) in enumerate(calls[:-1]):
        if r is not None and abs(r) > 10.0 * b and calls[i + 1][0] == t:
            refused.add(i)
            _, r2, b2, _ = calls[i + 1]
            rows.append(
                {'run': name, 't': t, 'r': r, 'bound': b, 'r_repeat': r2, 'bound_repeat': b2}
            )
    lone = [
        i
        for i, (t, r, b, _) in enumerate(calls)
        if r is not None and abs(r) > 10.0 * b and i not in refused and i - 1 not in refused
    ]
    if len(repeats) != len(rows) or any(
        abs(r - row['r']) > 1.0e-5 * abs(row['r']) for (r, _), row in zip(repeats, rows)
    ):
        raise ValueError(
            f'{log}: the repeat lines do not match the {len(rows)} refused attempts'
        )
    passed = [s for _, s in repeats].count('passed')
    kept = [c for i, c in enumerate(calls) if i not in refused and c[1] is not None]
    status = (folder / 'output' / f'tlt_{name}' / 'status').read_text().split('\n', 1)
    summary = {
        't_end': max(c[0] for c in calls),
        'status': status[1].strip() if len(status) > 1 else status[0].strip(),
        'calls': len(calls) - len(rows),  # a refused attempt and its repeat are 1 call
        'repeats_passed': passed,
        'repeats_failed': len(repeats) - passed,
        'refusals_without_repeat': len(lone),
        'kept_1_to_10_B': sum(1 < abs(r) / b <= 10 for _, r, b, _ in kept),
        'kept_above_10_B': sum(abs(r) / b > 10 for _, r, b, _ in kept),
        'steps': sum(c[3] for c in calls),
        'steps_refused': sum(calls[i][3] for i in refused),
        'steps_repeat': sum(calls[i + 1][3] for i in refused),
    }
    cfg = tomllib.loads((folder / 'cfg' / f'tlt_none_{name}.toml').read_text())
    prov = {
        'aragog': head[1],
        'proteus': head[2],
        'zalmoxis': head[3],
        'delta_T_super': cfg['planet']['delta_T_super'],
        'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
    }
    return rows, summary, prov


def main(argv):
    """Write the JSON file from the runs named in ``argv``."""
    folder, note, names = Path(argv[0]), argv[1], argv[2:]
    rows, runs, base, prov = [], {}, {}, {}
    for name in names:
        is_base = name.startswith('base:')
        name = name.removeprefix('base:')
        r, s, p = _run(folder, name)
        prov[name] = p
        if is_base:
            base[name] = {k: s[k] for k in ('t_end', 'status', 'refusals_without_repeat')}
        else:
            rows += r
            runs[name] = s
    data = {
        'provenance': {
            'config': 'PROTEUS input/tutorials/tutorial_layer_trapping.toml of '
            'https://github.com/FormingWorlds/PROTEUS/pull/1090 with trap_mode = "none", '
            'aragog rtol 1e-8, each run with its own delta_T_super',
            'runs': prov,
            'builder': 'tests/reference/build_cmb_ledger_repeat_runs.py',
            'note': note,
        },
        'runs': runs,
        'base_runs': base,
        'rows': rows,
    }
    OUT.write_text(json.dumps(data, indent=1) + '\n')
    print(OUT, 'rows', len(rows), 'runs', len(runs), 'base runs', len(base))


if __name__ == '__main__':
    main(sys.argv[1:])
