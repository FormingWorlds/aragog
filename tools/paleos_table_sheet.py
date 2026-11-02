#!/usr/bin/env python3
"""Print a sheet that reads a P-S table set next to published magma ocean values.

One row per check: the measured value, the band, the source, and a status. ``pass`` and ``outside``
compare with a band of the same quantity, as a reading and not a verdict: the bands come from other
melting models. ``report`` gives a value that no source gives as the same quantity, or a property of
the table with no published value. ``n/a`` means the tables do not reach the check. Grid and edges
come from the loaded tables only.

Usage: ``python tools/paleos_table_sheet.py <table folder> [...] [--paleos-commit FOLDER=COMMIT]``
Run it with ``src`` of a checkout on PYTHONPATH to read with that code; otherwise it imports the
installed aragog. No table writer emits a commit line yet, so give the commit with the flag.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

from aragog.eos.entropy import EntropyEOS

GPA = 1.0e9
REFERENCES = {
    'MK19a': 'Miyazaki & Korenaga 2019, JGR Solid Earth, part 1',
    'MK19b': 'Miyazaki & Korenaga 2019, JGR Solid Earth, part 2',
    'MK22A': 'Miyazaki & Korenaga 2022, Astrobiology',
}
UPPER_MANTLE_GPA = (1.0, 5.0, 10.0, 20.0)
DELTA_S_LOW_P = (150.0, 600.0)  # J/kg/K, a factor 2 around the published 300, chosen here
# model cooling below the cotectic [K], not the same quantity as the table gap
GAP_COTECTIC_K = {50.0: 900.0, 100.0: 700.0}
PHI0_BAND_GPA = (3.0, 8.0)  # around the published 5 GPa, chosen here
TABLE_PROPERTY = 'none: a property of the table'
NEAR_SURFACE_GPA = 5.0  # the density rows with a same-quantity band reach this pressure
MIX_STEPS = (0.25, 0.01)  # steps of the mixed gradient, as fractions of the pressure


class Row(NamedTuple):
    """One line of the sheet: check, measured value, band, source, status."""

    check: str
    value: str
    band: str
    source: str
    status: str


def _reaches(eos: EntropyEOS, p: float) -> bool:
    """Whether every table holds pressure ``p`` and each phase table holds its boundary entropy."""
    if not all(table['P'][0] <= p <= table['P'][-1] for table in eos._tables.values()):
        return False
    s_sol, s_liq = float(eos.solidus_entropy(p)), float(eos.liquidus_entropy(p))
    solid, melt = eos._tables['temperature_solid']['S'], eos._tables['temperature_melt']['S']
    return bool(solid[0] <= s_sol <= solid[-1] and melt[0] <= s_liq <= melt[-1])


def _boundary(eos: EntropyEOS, prop: str, p: float, phase: str) -> float:
    return float(eos._lookup_at_phase_boundary(prop, np.array([p]), phase)[0])


def _alpha(eos: EntropyEOS, p: float, phase: str) -> float:
    """Expansivity [1/K] of a pure phase at its phase boundary, rho cp (dT/dP)_S / T."""
    rho, cp, grad, t = (
        _boundary(eos, name, p, phase)
        for name in ('density', 'heat_capacity', 'dTdPs', 'temperature')
    )
    return rho * cp * grad / t if t > 0.0 else float('nan')


def _delta_s(eos: EntropyEOS, p: float) -> float:
    return float(eos.liquidus_entropy(p) - eos.solidus_entropy(p))


def _mix_entropy(eos: EntropyEOS, p: float) -> float:
    return 0.5 * float(eos.solidus_entropy(p) + eos.liquidus_entropy(p))


def _mix_stays_mixed(eos: EntropyEOS, p: float, step: float) -> bool:
    """Whether the entropy halfway at ``p`` stays between solidus and liquidus over the step."""
    s = _mix_entropy(eos, p)
    ends = (p * (1.0 - step), p * (1.0 + step))
    return all(float(eos.solidus_entropy(x)) < s < float(eos.liquidus_entropy(x)) for x in ends)


def _mix_gradient(eos: EntropyEOS, p: float, step: float) -> float:
    """dT/dP [K/Pa] at fixed entropy halfway between solidus and liquidus, central difference."""
    s = _mix_entropy(eos, p)
    dp = step * p
    t_hi, t_lo = (
        float(eos.temperature(np.array([x]), np.array([s]))[0]) for x in (p + dp, p - dp)
    )
    return (t_hi - t_lo) / (2.0 * dp)


def _phi0_pressure(eos: EntropyEOS) -> float | None:
    """Pressure [Pa] where the isentrope with melt fraction 0.4 at the lowest pressure reaches 0.

    The isentrope keeps its entropy, so this is where the solidus entropy rises to it. None when
    the tables hold no such pressure.
    """
    p0 = float(eos.P_min)
    s = float(eos.solidus_entropy(p0)) + 0.4 * _delta_s(eos, p0)

    def above(p):
        return float(eos.solidus_entropy(p)) - s

    if not above(p0) < 0.0 <= above(float(eos.P_max)):
        return None
    return float(brentq(above, p0, float(eos.P_max)))


def _padded_span(table: dict, column: int) -> float:
    """Entropy span [J/kg/K] above the table minimum whose temperature copies the first row.

    NaN when the first row is not finite.
    """
    t = np.asarray(table['values'])[column]
    if not np.isfinite(t[0]):
        return float('nan')
    changed = np.flatnonzero(t != t[0])
    return (
        float(table['S'][changed[0] - 1] - table['S'][0])
        if changed.size
        else float(np.ptp(table['S']))
    )


def _melt_copied_above_liquidus(eos: EntropyEOS, p: float) -> float:
    """Entropy span [J/kg/K] above the liquidus where the melt temperature keeps its value there.

    Read on the table column nearest to ``p``, with the liquidus at the pressure of that column.
    """
    table = eos._tables['temperature_melt']
    i = int(np.argmin(np.abs(table['P'] - p)))
    column, s_grid = np.asarray(table['values'])[i], np.asarray(table['S'])
    s_liq = float(eos.liquidus_entropy(float(table['P'][i])))
    first_above = int(np.searchsorted(s_grid, s_liq))
    if first_above == s_grid.size:
        return 0.0
    at_liquidus = column[max(first_above - 1, 0)]
    changed = np.flatnonzero((s_grid >= s_grid[first_above]) & (column != at_liquidus))
    last_kept = changed[0] - 1 if changed.size else s_grid.size - 1
    return max(float(s_grid[last_kept] - s_liq), 0.0)


def sheet(eos: EntropyEOS) -> list[Row]:
    """Measure every check on one loaded table set.

    Parameters
    ----------
    eos : EntropyEOS
        The loaded tables.

    Returns
    -------
    list of Row
        The rows in sheet order; a row's status is ``pass``, ``outside``, ``report`` or ``n/a``.
    """
    rows: list[Row] = []

    def add(check, value, band, source, ok):
        rows.append(Row(check, value, band, source, 'pass' if ok else 'outside'))

    def report(check, value, band, source):
        rows.append(Row(check, value, band, source, 'report'))

    def absent(check, band, source):
        rows.append(Row(check, '-', band, source, 'n/a'))

    for p_gpa in UPPER_MANTLE_GPA:
        p = p_gpa * GPA
        check = f'rho_melt - rho_solid at {p_gpa:g} GPa [kg/m3]'
        near_surface = p_gpa <= NEAR_SURFACE_GPA
        band = (
            '< 0 near the surface (2900 - 3300 prescribed)'
            if near_surface
            else 'none: their model has melt heavier here, with a crossover near 75 GPa'
        )
        source = 'MK22A p23; MK19b p9'
        if not _reaches(eos, p):
            absent(check, band, source)
            continue
        d = _boundary(eos, 'density', p, 'melt') - _boundary(eos, 'density', p, 'solid')
        if near_surface:
            add(check, f'{d:.1f}', band, source, d < 0.0)
        else:
            report(check, f'{d:.1f}', band, source)

    for p_gpa in UPPER_MANTLE_GPA:
        p = p_gpa * GPA
        check = f'alpha_melt and alpha_solid, ordering at {p_gpa:g} GPa [1/K]'
        band = 'melt above solid above 0; magnitudes not compared (4.6e-5, 3.0e-5 prescribed)'
        source = 'MK22A p23'
        if not _reaches(eos, p):
            absent(check, band, source)
            continue
        a_m, a_s = _alpha(eos, p, 'melt'), _alpha(eos, p, 'solid')
        add(
            check,
            f'{a_m:.3g}, {a_s:.3g}',
            band,
            source,
            bool(np.isfinite(a_m) and a_m > a_s > 0.0),
        )

    for p_gpa in UPPER_MANTLE_GPA:
        p = p_gpa * GPA
        check = f'(dT/dP)_S mixed vs pure phases at {p_gpa:g} GPa [K/GPa]'
        band, source = 'mixed above both, read at the wide step', 'MK19b p5'
        wide = MIX_STEPS[0]
        ends = (p * (1.0 - wide), p * (1.0 + wide))
        if not (all(_reaches(eos, x) for x in ends) and _mix_stays_mixed(eos, p, wide)):
            absent(check, band, source)
            continue
        mix = [GPA * _mix_gradient(eos, p, step) for step in MIX_STEPS]
        melt = GPA * _boundary(eos, 'dTdPs', p, 'melt')
        solid = GPA * _boundary(eos, 'dTdPs', p, 'solid')
        value = f'{mix[0]:.2f} at step {wide:g} P, {mix[1]:.2f} at {MIX_STEPS[1]:g} P'
        add(
            check,
            f'{value} (melt {melt:.2f}, solid {solid:.2f})',
            band,
            source,
            mix[0] > max(melt, solid),
        )

    p_low = float(eos.P_min)
    d_low = _delta_s(eos, p_low)
    add(
        f'Delta S at the lowest pressure {p_low / GPA:.3g} GPa [J/kg/K]',
        f'{d_low:.1f}',
        f'300 prescribed; band chosen here {DELTA_S_LOW_P[0]:g}-{DELTA_S_LOW_P[1]:g}',
        'MK22A p23',
        DELTA_S_LOW_P[0] <= d_low <= DELTA_S_LOW_P[1],
    )
    check = 'Delta S at 100 GPa / lowest pressure'
    band, source = '< 1 in their model; other studies find little change', 'MK19a p3 Eq 3, p15'
    if _reaches(eos, 100.0 * GPA) and d_low > 0.0:
        ratio = _delta_s(eos, 100.0 * GPA) / d_low
        add(check, f'{ratio:.3f}', band, source, 0.0 < ratio < 1.0)
    else:
        absent(check, band, source)

    for p_gpa, cotectic in GAP_COTECTIC_K.items():
        p = p_gpa * GPA
        check = f'T_liquidus - T_solidus at {p_gpa:g} GPa [K]'
        band = (
            f'none: their model cools {cotectic:g} K below the cotectic (MgO-FeO-SiO2), '
            'chondritic data show under 500 K'
        )
        source = 'MK19a p12-13'
        if not _reaches(eos, p):
            absent(check, band, source)
            continue
        gap = _boundary(eos, 'temperature', p, 'melt') - _boundary(
            eos, 'temperature', p, 'solid'
        )
        report(check, f'{gap:.0f}', band, source)

    check = 'pressure where the isentrope with phi 0.4 at the surface reaches phi 0 [GPa]'
    band = (
        f'about 5 (Katz 2003 model); band chosen here {PHI0_BAND_GPA[0]:g}-{PHI0_BAND_GPA[1]:g}'
    )
    source = 'MK22A p5'
    p_phi0 = _phi0_pressure(eos)
    if p_phi0 is None:
        absent(check, band, source)
    else:
        ok = PHI0_BAND_GPA[0] * GPA <= p_phi0 <= PHI0_BAND_GPA[1] * GPA
        add(check, f'{p_phi0 / GPA:.2f}', band, source, ok)

    table = eos._tables['temperature_solid']
    spans = np.array([_padded_span(table, i) for i in range(len(table['P']))])
    for name, span in (('lowest', spans[0]), ('highest', spans[-1])):
        check = f'solid temperature rows copied above S_min, {name} pressure [J/kg/K]'
        report(check, f'{span:.1f}', TABLE_PROPERTY, 'the table')
    padded = np.flatnonzero(spans > 0.0)
    where = (
        f', {table["P"][padded[0]] / GPA:.3g}-{table["P"][padded[-1]] / GPA:.3g} GPa, '
        f'largest span {np.nanmax(spans):.1f} J/kg/K'
        if padded.size
        else ''
    )
    report(
        'pressure columns with solid temperature rows copied above S_min',
        f'{padded.size} of {spans.size}{where}',
        TABLE_PROPERTY,
        'the table',
    )
    for p_gpa in (1.0, 20.0):
        p = p_gpa * GPA
        check = f'melt temperature rows copied above the liquidus at {p_gpa:g} GPa [J/kg/K]'
        if not _reaches(eos, p):
            absent(check, TABLE_PROPERTY, 'the table')
            continue
        report(check, f'{_melt_copied_above_liquidus(eos, p):.1f}', TABLE_PROPERTY, 'the table')
    return rows


def _paleos_commit(folder: Path, given: str | None) -> str:
    """The PALEOS commit of a table set: given on the command line, else from a table header."""
    if given:
        return f'{given} (given)'
    for path in sorted(folder.glob('[!.]*.dat')):
        with path.open(errors='replace') as handle:
            for line in handle:
                if not line.startswith('#'):
                    break
                if line.startswith('# PALEOS commit:'):
                    return f'{line.split(":", 1)[1].strip()} (table header)'
    return 'not recorded in the tables; give it with --paleos-commit'


def describe(folder: Path, commit: str | None = None) -> str:
    """Sheet of one table folder as Markdown, headed by its range, PALEOS commit and table md5."""
    eos = EntropyEOS(folder)
    rows = sheet(eos)
    counts = Counter(row.status for row in rows)
    lines = [
        f'## {folder}',
        f'S {eos.S_min:.3f} to {eos.S_max:.3f} J/kg/K, P {eos.P_min:.3g} to {eos.P_max:.3g} Pa',
        f'PALEOS commit: {_paleos_commit(folder, commit)}',
    ]
    lines += [
        f'{path.name} md5 {hashlib.md5(path.read_bytes()).hexdigest()}'
        for path in sorted(folder.glob('[!.]*.dat'))
    ]
    lines += [
        f'{counts["pass"]} pass, {counts["outside"]} outside, {counts["report"]} report, '
        f'{counts["n/a"]} n/a. Pass and outside are readings against other melting models, not '
        'verdicts; report has no band of the same quantity.',
        '',
        '| check | value | band | source | status |',
        '|---|---|---|---|---|',
    ]
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    lines += ['', 'Sources: ' + '; '.join(f'{k}: {v}' for k, v in REFERENCES.items())]
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the sheet of each table folder; an outside row does not change the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folders', nargs='+', type=Path, help='P-S table folders')
    parser.add_argument(
        '--paleos-commit',
        action='append',
        default=[],
        metavar='FOLDER=COMMIT',
        help='PALEOS commit of a table set whose headers do not record it',
    )
    args = parser.parse_args(argv)
    commits = {}
    for item in args.paleos_commit:
        folder, sep, commit = item.rpartition('=')
        if not (sep and folder and commit):
            parser.error(f'--paleos-commit takes FOLDER=COMMIT, got {item!r}')
        commits[Path(folder)] = commit
    unused = set(commits) - set(args.folders)
    if unused:
        parser.error(f'--paleos-commit names a folder not given: {sorted(map(str, unused))}')
    for folder in args.folders:
        print(describe(folder, commits.get(folder)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
