#!/usr/bin/env python3
"""Print a sheet that compares a P-S table set with published magma ocean values.

One row per check: the measured value, the published band, the source, and pass or outside. An
outside row is information, not a failure: the bands come from other melting models. Grid and
edges come from the loaded tables only; a check at a pressure the tables do not reach reads n/a.

Usage: ``python tools/paleos_table_sheet.py <table folder> [<table folder> ...]``
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

from aragog.eos.entropy import EntropyEOS

GPA = 1.0e9
MK_I = 'Miyazaki & Korenaga 2019, JGR 124, part 1'
MK_II = 'Miyazaki & Korenaga 2019, JGR 124, part 2'
MK_A = 'Miyazaki & Korenaga 2022, Astrobiology 22'
UPPER_MANTLE_GPA = (1.0, 5.0, 10.0, 20.0)
DELTA_S_LOW_P = (150.0, 600.0)  # J/kg/K, a factor 2 around the published 300
GAP_BAND = (500.0, 900.0)  # K: experimental lower bound and model upper value at 50-100 GPa
PHI0_BAND_GPA = (3.0, 8.0)  # around the published 5 GPa


class Row(NamedTuple):
    """One line of the sheet: check, measured value, published band, source, status."""

    check: str
    value: str
    band: str
    source: str
    status: str


def _reaches(eos: EntropyEOS, p: float) -> bool:
    return bool(eos.P_min <= p <= eos.P_max)


def _boundary(eos: EntropyEOS, prop: str, p: float, phase: str) -> float:
    return float(eos._lookup_at_phase_boundary(prop, np.array([p]), phase)[0])


def _alpha(eos: EntropyEOS, p: float, phase: str) -> float:
    """Expansivity [1/K] of a pure phase at its phase boundary, rho cp |dT/dP|_S / T."""
    rho, cp, grad, t = (
        _boundary(eos, name, p, phase)
        for name in ('density', 'heat_capacity', 'dTdPs', 'temperature')
    )
    return rho * cp * abs(grad) / t


def _delta_s(eos: EntropyEOS, p: float) -> float:
    return float(eos.liquidus_entropy(p) - eos.solidus_entropy(p))


def _mix_gradient(eos: EntropyEOS, p: float) -> float:
    """dT/dP [K/Pa] at fixed entropy halfway between solidus and liquidus, central difference.

    The wide step of 25 % of the pressure averages over the steps of the entropy grid.
    """
    s = 0.5 * float(eos.solidus_entropy(p) + eos.liquidus_entropy(p))
    dp = 0.25 * p
    t_hi, t_lo = (
        float(eos.temperature(np.array([x]), np.array([s]))[0]) for x in (p + dp, p - dp)
    )
    return (t_hi - t_lo) / (2.0 * dp)


def _phi0_pressure(eos: EntropyEOS) -> float | None:
    """Pressure [Pa] where the isentrope with melt fraction 0.4 at the lowest pressure reaches 0."""
    p0 = float(eos.P_min)
    s = float(eos.solidus_entropy(p0)) + 0.4 * _delta_s(eos, p0)
    above_solidus = lambda p: float(eos.solidus_entropy(p)) - s  # noqa: E731
    if above_solidus(eos.P_max) < 0.0:
        return None
    return float(brentq(above_solidus, p0, float(eos.P_max)))


def _padded_span(table: dict, column: int) -> float:
    """Entropy span [J/kg/K] above the table minimum whose temperature copies the first row."""
    t = np.asarray(table['values'])[column]
    flat = np.flatnonzero(t != t[0])
    return (
        float(table['S'][flat[0] - 1] - table['S'][0])
        if flat.size
        else float(np.ptp(table['S']))
    )


def _melt_copied_above_liquidus(eos: EntropyEOS, p: float) -> float:
    """Entropy span [J/kg/K] above the liquidus where the melt temperature stays at its value."""
    table = eos._tables['temperature_melt']
    column = np.asarray(table['values'])[int(np.argmin(np.abs(table['P'] - p)))]
    s_grid, s_liq = np.asarray(table['S']), float(eos.liquidus_entropy(p))
    at_liq = int(np.searchsorted(s_grid, s_liq))
    changed = np.flatnonzero(
        (s_grid > s_liq) & (column != column[min(at_liq, s_grid.size - 1)])
    )
    return (
        float(s_grid[changed[0] - 1] - s_liq)
        if changed.size and changed[0] - 1 > at_liq
        else 0.0
    )


def sheet(eos: EntropyEOS) -> list[Row]:
    """Measure every check on one loaded table set.

    Parameters
    ----------
    eos : EntropyEOS
        The loaded tables.

    Returns
    -------
    list of Row
        The rows in sheet order; a row's status is ``pass``, ``outside`` or ``n/a``.
    """
    rows: list[Row] = []

    def add(check, value, band, source, ok):
        rows.append(Row(check, value, band, source, 'pass' if ok else 'outside'))

    def absent(check, band, source):
        rows.append(Row(check, '-', band, source, 'n/a'))

    for p_gpa in UPPER_MANTLE_GPA:
        p = p_gpa * GPA
        check = f'rho_melt - rho_solid at {p_gpa:g} GPa [kg/m3]'
        band, source = '< 0', f'{MK_I} p13; {MK_II} p9'
        if not _reaches(eos, p):
            absent(check, band, source)
            continue
        d = _boundary(eos, 'density', p, 'melt') - _boundary(eos, 'density', p, 'solid')
        add(check, f'{d:.1f}', band, source, d < 0.0)

    for p_gpa in UPPER_MANTLE_GPA:
        p = p_gpa * GPA
        check = f'alpha_melt / alpha_solid at {p_gpa:g} GPa'
        band, source = '> 1 (4.6e-5 / 3.0e-5 1/K)', f'{MK_A} p23'
        if not _reaches(eos, p):
            absent(check, band, source)
            continue
        a_m, a_s = _alpha(eos, p, 'melt'), _alpha(eos, p, 'solid')
        add(
            check,
            f'{a_m / a_s if a_s else np.inf:.3f} ({a_m:.3g} / {a_s:.3g})',
            band,
            source,
            a_m > a_s,
        )

    for p_gpa in UPPER_MANTLE_GPA:
        p = p_gpa * GPA
        check = f'(dT/dP)_S mixed vs pure phases at {p_gpa:g} GPa [K/GPa]'
        band, source = 'mixed above both', f'{MK_II} p5'
        if not _reaches(eos, p):
            absent(check, band, source)
            continue
        mix, melt, solid = (
            GPA * x
            for x in (
                _mix_gradient(eos, p),
                _boundary(eos, 'dTdPs', p, 'melt'),
                _boundary(eos, 'dTdPs', p, 'solid'),
            )
        )
        add(
            check,
            f'{mix:.2f} (melt {melt:.2f}, solid {solid:.2f})',
            band,
            source,
            mix > max(melt, solid),
        )

    p_low = float(eos.P_min)
    d_low = _delta_s(eos, p_low)
    add(
        f'Delta S at the lowest pressure {p_low / GPA:.3g} GPa [J/kg/K]',
        f'{d_low:.1f}',
        f'300; band {DELTA_S_LOW_P[0]:g}-{DELTA_S_LOW_P[1]:g}',
        f'{MK_A} p23',
        DELTA_S_LOW_P[0] <= d_low <= DELTA_S_LOW_P[1],
    )
    check, band, source = 'Delta S at 100 GPa / lowest pressure', '< 1', f'{MK_I} p3 Eq 3, p15'
    if _reaches(eos, 100.0 * GPA):
        ratio = _delta_s(eos, 100.0 * GPA) / d_low
        add(check, f'{ratio:.3f}', band, source, ratio < 1.0)
    else:
        absent(check, band, source)

    for p_gpa in (50.0, 100.0):
        p = p_gpa * GPA
        check = f'T_liquidus - T_solidus at {p_gpa:g} GPa [K]'
        band, source = f'{GAP_BAND[0]:g}-{GAP_BAND[1]:g}', f'{MK_I} p12-13'
        if not _reaches(eos, p):
            absent(check, band, source)
            continue
        gap = _boundary(eos, 'temperature', p, 'melt') - _boundary(
            eos, 'temperature', p, 'solid'
        )
        add(check, f'{gap:.0f}', band, source, GAP_BAND[0] <= gap <= GAP_BAND[1])

    check = 'pressure where the isentrope with phi 0.4 at the surface reaches phi 0 [GPa]'
    band, source = f'about 5; band {PHI0_BAND_GPA[0]:g}-{PHI0_BAND_GPA[1]:g}', f'{MK_A} p5'
    p_phi0 = _phi0_pressure(eos)
    if p_phi0 is None:
        absent(check, band, source)
    else:
        add(
            check,
            f'{p_phi0 / GPA:.2f}',
            band,
            source,
            PHI0_BAND_GPA[0] * GPA <= p_phi0 <= PHI0_BAND_GPA[1] * GPA,
        )

    table = eos._tables['temperature_solid']
    for name, column in (('lowest', 0), ('highest', len(table['P']) - 1)):
        span = _padded_span(table, column)
        add(
            f'solid temperature rows copied above S_min, {name} pressure [J/kg/K]',
            f'{span:.1f}',
            '0',
            'property of the table',
            span == 0.0,
        )
    for p_gpa in (1.0, 20.0):
        p = p_gpa * GPA
        check = f'melt temperature rows copied above the liquidus at {p_gpa:g} GPa [J/kg/K]'
        if not _reaches(eos, p):
            absent(check, '0', 'property of the table')
            continue
        span = _melt_copied_above_liquidus(eos, p)
        add(check, f'{span:.1f}', '0', 'property of the table', span == 0.0)
    return rows


def describe(folder: Path) -> str:
    """Sheet of one table folder as Markdown, headed by its range and the md5 of its tables."""
    eos = EntropyEOS(folder)
    lines = [
        f'## {folder}',
        f'S {eos.S_min:.3f} to {eos.S_max:.3f} J/kg/K, P {eos.P_min:.3g} to {eos.P_max:.3g} Pa',
    ]
    for name in ('temperature_solid.dat', 'temperature_melt.dat'):
        lines.append(f'{name} md5 {hashlib.md5((folder / name).read_bytes()).hexdigest()}')
    lines += ['', '| check | value | band | source | status |', '|---|---|---|---|---|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in sheet(eos)]
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the sheet of each table folder; an outside row does not change the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folders', nargs='+', type=Path, help='P-S table folders')
    for folder in parser.parse_args(argv).folders:
        print(describe(folder))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
