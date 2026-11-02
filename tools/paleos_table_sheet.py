#!/usr/bin/env python3
"""Print a sheet that reads a P-S table set next to published magma ocean values.

One row per check: the measured value, the band, the source, and a status. ``pass`` and ``outside``
compare with a band that holds a published value of the same quantity, as a reading and not a
verdict: the bands come from other melting models. ``report`` gives a value with no reading: a
published number of the same name stands next to it for another composition, or no source gives the
quantity, or the row is a property of the table, or the 2 steps of a difference disagree. ``n/a``
means a table or melting curve does not hold the check, a difference step leaves the mixed region,
or a measured value is not finite. Grid and edges come from the loaded tables only.

Usage: ``python tools/paleos_table_sheet.py <table folder> [...] [--paleos-commit FOLDER=COMMIT]``
Run it with ``src`` of a checkout on PYTHONPATH to read with that code; otherwise it imports the
installed aragog. The tool reads the commit from a ``# PALEOS commit:`` header line if a table has
one; P-S tables have none, so give the commit with the flag.
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
# model cooling below the cotectic [K], not the same quantity as the table gap
GAP_COTECTIC_K = {50.0: 900.0, 100.0: 700.0}
TABLE_PROPERTY = 'none: a property of the table'
ALPHA_MELT, ALPHA_SOLID = 4.6e-5, 3.0e-5  # [1/K], prescribed by MK22A p23
NEAR_SURFACE_GPA = 5.0  # depth of the melt layer whose isentrope uses these values, MK22A p5
MIX_STEPS = (0.25, 0.01)  # steps of the mixed gradient, as fractions of the pressure
MIX_STEP_AGREE = 1.5  # a reading needs the 2 steps within this factor: else the step decides it
READ_TABLES = ('temperature', 'density', 'heat_capacity', 'dTdPs')


class Row(NamedTuple):
    """One line of the sheet: check, measured value, band, source, status."""

    check: str
    value: str
    band: str
    source: str
    status: str


def _reaches(eos: EntropyEOS, p: float) -> bool:
    """Whether every table the sheet reads and each melting curve holds ``p``, and each phase table
    its boundary entropy.

    A lookup clamps to the range of its own table without a word, so each range counts.
    """
    solid, melt = (
        [eos._tables[f'{q}_{phase}'] for q in READ_TABLES] for phase in ('solid', 'melt')
    )
    if not all(
        c['P'][0] <= p <= c['P'][-1] for c in (*solid, *melt, eos._solidus, eos._liquidus)
    ):
        return False
    s_sol, s_liq = float(eos.solidus_entropy(p)), float(eos.liquidus_entropy(p))
    return all(t['S'][0] <= s_sol <= t['S'][-1] for t in solid) and all(
        t['S'][0] <= s_liq <= t['S'][-1] for t in melt
    )


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


def _phi0_pressure(eos: EntropyEOS) -> tuple[float | None, float]:
    """First pressure [Pa] where the isentrope with melt fraction 0.4 at the lowest pressure reaches
    0, and the top of the searched range.

    The isentrope keeps its entropy, so this is the first node interval where the solidus entropy
    rises to it. The pressure is None when the solidus stays below it up to the top of the range.
    """
    p0 = float(eos.P_min)
    s = float(eos.solidus_entropy(p0)) + 0.4 * _delta_s(eos, p0)
    top = min(float(eos.P_max), float(eos._solidus['P'][-1]))
    grid = np.asarray(eos._solidus['P'], dtype=float)
    nodes = np.r_[p0, grid[(grid > p0) & (grid < top)], top]

    def above(p):
        return float(eos.solidus_entropy(p)) - s

    crossed = np.flatnonzero([above(float(p)) >= 0.0 for p in nodes])
    if not crossed.size:
        return None, top
    i = int(crossed[0])
    return float(brentq(above, nodes[i - 1], nodes[i])), top


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


def _nearest_column(table: dict, p: float) -> int:
    return int(np.argmin(np.abs(table['P'] - p)))


def _melt_copied_above_liquidus(eos: EntropyEOS, p: float) -> float:
    """Entropy span [J/kg/K] above the liquidus where the melt temperature keeps its value there.

    Read on the column at ``_nearest_column``, with the liquidus at the pressure of that column.
    """
    table = eos._tables['temperature_melt']
    i = _nearest_column(table, p)
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

    def row(check, value, band, source, status, measured):
        finite = np.all(np.isfinite(measured))
        rows.append(Row(check, value, band, source, status if finite else 'n/a'))

    def add(check, value, band, source, ok, *measured):
        row(check, value, band, source, 'pass' if ok else 'outside', measured)

    def report(check, value, band, source, *measured):
        row(check, value, band, source, 'report', measured)

    def absent(check, band, source):
        rows.append(Row(check, '-', band, source, 'n/a'))

    for p_gpa in UPPER_MANTLE_GPA:
        p = p_gpa * GPA
        check = f'rho_melt - rho_solid at {p_gpa:g} GPa [kg/m3]'
        near_surface = p_gpa <= NEAR_SURFACE_GPA
        if near_surface:
            band = '< 0 from the prescribed 2900 and 3300, cut at the 5 GPa melt layer depth'
            source = 'MK22A p23, p5'
        else:
            band = 'none: the MK19b model has melt heavier here, with a crossover near 75 GPa'
            source = 'MK19b p9'
        if not _reaches(eos, p):
            absent(check, band, source)
            continue
        d = _boundary(eos, 'density', p, 'melt') - _boundary(eos, 'density', p, 'solid')
        if near_surface:
            add(check, f'{d:.1f}', band, source, d < 0.0, d)
        else:
            report(check, f'{d:.1f}', band, source, d)

    for p_gpa in UPPER_MANTLE_GPA:
        p = p_gpa * GPA
        check = f'alpha_melt and alpha_solid, ordering at {p_gpa:g} GPa [1/K]'
        near_surface = p_gpa <= NEAR_SURFACE_GPA
        limit = '5 GPa limit: a reading of p5 and p23, not a statement of the paper'
        band = (
            f'melt above solid above 0, ordering only, magnitudes not compared; {limit}'
            if near_surface
            else f'none; {limit}'
        )
        source = 'MK22A p5, p23'
        if not _reaches(eos, p):
            absent(check, band, source)
            continue
        a_m, a_s = _alpha(eos, p, 'melt'), _alpha(eos, p, 'solid')
        value = (
            f'{a_m:.3g}, {a_s:.3g} ({a_m / ALPHA_MELT:.1f} and {a_s / ALPHA_SOLID:.1f} times '
            'the prescribed 4.6e-5, 3.0e-5)'
        )
        if near_surface:
            add(check, value, band, source, a_m > a_s > 0.0, a_m, a_s)
        else:
            report(check, value, band, source, a_m, a_s)

    for p_gpa in UPPER_MANTLE_GPA:
        p = p_gpa * GPA
        check = f'(dT/dP)_S mixed vs pure phases at {p_gpa:g} GPa [K/GPa]'
        band = f'mixed above both, at the wide step where the steps agree within {MIX_STEP_AGREE:g}'
        source = 'MK19b p5'
        wide = MIX_STEPS[0]
        ends = (p * (1.0 - wide), p * (1.0 + wide))
        if not (all(_reaches(eos, x) for x in ends) and _mix_stays_mixed(eos, p, wide)):
            absent(check, band, source)
            continue
        mix = [GPA * _mix_gradient(eos, p, step) for step in MIX_STEPS]
        melt = GPA * _boundary(eos, 'dTdPs', p, 'melt')
        solid = GPA * _boundary(eos, 'dTdPs', p, 'solid')
        value = f'{mix[0]:.2f} at step {wide:g} P, {mix[1]:.2f} at {MIX_STEPS[1]:g} P'
        value = f'{value} (melt {melt:.2f}, solid {solid:.2f})'
        measured = (*mix, melt, solid)
        small, large = sorted(map(abs, mix))
        if mix[0] * mix[1] > 0.0 and large <= MIX_STEP_AGREE * small:
            add(check, value, band, source, mix[0] > max(melt, solid), *measured)
        else:
            report(check, value, band, source, *measured)

    p_low = float(eos.P_min)
    low_held = _reaches(eos, p_low)
    d_low = _delta_s(eos, p_low)
    check = f'Delta S at the lowest pressure {p_low / GPA:.3g} GPa [J/kg/K]'
    band, source = 'none: 300 prescribed for a pyrolite', 'MK22A p23'
    if low_held:
        report(check, f'{d_low:.1f}', band, source)
    else:
        absent(check, band, source)
    check = 'Delta S at 100 GPa / lowest pressure'
    band = 'none: < 1 in their MgO model, a similar trend for MgSiO3; others find little change'
    source = 'MK19a p3 Eq 3, p15'
    if low_held and _reaches(eos, 100.0 * GPA) and d_low > 0.0:
        report(check, f'{_delta_s(eos, 100.0 * GPA) / d_low:.3f}', band, source)
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
        report(check, f'{gap:.0f}', band, source, gap)

    check = 'pressure where the isentrope with phi 0.4 at the surface reaches phi 0 [GPa]'
    band, source = 'none: about 5 for a pyrolite (Katz 2003 model)', 'MK22A p5'
    if low_held and d_low > 0.0:
        p_phi0, top = _phi0_pressure(eos)
        shown = f'> {top / GPA:.3g}' if p_phi0 is None else f'{p_phi0 / GPA:.2f}'
        report(check, shown, band, source)
    else:
        absent(check, band, source)

    table = eos._tables['temperature_solid']
    spans = np.array([_padded_span(table, i) for i in range(len(table['P']))])
    for name, span in (('lowest', spans[0]), ('highest', spans[-1])):
        check = f'solid temperature rows copied above S_min, {name} pressure [J/kg/K]'
        report(check, f'{span:.1f}', TABLE_PROPERTY, 'the table', span)
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
        melt_table = eos._tables['temperature_melt']
        at = float(melt_table['P'][_nearest_column(melt_table, p)])
        value = f'{_melt_copied_above_liquidus(eos, p):.1f} in the column at {at / GPA:.4g} GPa'
        report(check, value, TABLE_PROPERTY, 'the table')
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
        f'{counts["n/a"]} n/a. Pass and outside are readings against published values of other '
        'melting models, not verdicts; report gives a value with no reading.',
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
        key = Path(folder).resolve()
        if key in commits:
            parser.error(f'--paleos-commit names {folder} twice')
        commits[key] = commit
    unused = set(commits) - {folder.resolve() for folder in args.folders}
    if unused:
        parser.error(f'--paleos-commit names a folder not given: {sorted(map(str, unused))}')
    for folder in args.folders:
        try:
            print(describe(folder, commits.get(folder.resolve())))
        except FileNotFoundError as missing:
            parser.error(f'{folder}: {missing}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
