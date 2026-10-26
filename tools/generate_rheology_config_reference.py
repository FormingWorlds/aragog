"""Generate the configuration reference table for [phase_solid.rheology].

Reads SolidRheologyParams from aragog.rheology to generate a Markdown
reference table of all configurable rheology parameters, their types, units,
default values, valid ranges, and physical effects.

Usage
-----
    python tools/generate_rheology_config_reference.py
    python tools/generate_rheology_config_reference.py --check
    python tools/generate_rheology_config_reference.py --write
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from aragog.rheology import SolidRheologyParams  # noqa: E402

UNITS: dict[str, str] = {
    'enabled': '--',
    'g_homologous': '--',
    't_sol_0': 'K',
    'dt_sol_dp': 'K/Pa',
    'arrhenius_t_ref': 'K',
    'viscosity_max_log10': 'log10(Pa·s)',
    'water_prefactor': '--',
    'yield_stress_c': 'Pa',
    'yield_stress_mu': '--',
    'yield_stress_max': 'Pa',
    'stress_closure_mode': '--',
    'interior_flux_fraction': '--',
    'lid_base_mode': '--',
    'lid_base_temperature': 'K',
    'lid_contrast_coeff': '--',
    'lid_mask_width_cells': 'cells',
    'phi_visc_single': '--',
    'mlt_bottom_slope': '--',
    's_top_inf': '--',
    'c_theta': '--',
    'activation_energy': 'J/mol',
    'activation_volume': 'm³/mol',
    'activation_volume_decay_pressure': 'Pa',
}

RANGES: dict[str, str] = {
    'enabled': '`true` \\| `false`',
    'g_homologous': '[0, inf)',
    't_sol_0': '(0, inf)',
    'dt_sol_dp': '[0, inf)',
    'arrhenius_t_ref': '(0, inf)',
    'viscosity_max_log10': '(20, 300]',
    'water_prefactor': '(0, inf)',
    'yield_stress_c': '[0, inf)',
    'yield_stress_mu': '[0, inf)',
    'yield_stress_max': '(0, inf)',
    'stress_closure_mode': '`"lid"` \\| `"local"`',
    'interior_flux_fraction': '[0.05, 0.5]',
    'lid_base_mode': '`"rheological"` \\| `"fixed"`',
    'lid_base_temperature': '(0, inf)',
    'lid_contrast_coeff': '(0, inf)',
    'lid_mask_width_cells': '(0, inf)',
    'phi_visc_single': '(0, 1)',
    'mlt_bottom_slope': '(0, 1]',
    's_top_inf': '(0, 1]',
    'c_theta': '[0, inf)',
    'activation_energy': '(0, inf) \\| `null`',
    'activation_volume': '[0, inf)',
    'activation_volume_decay_pressure': '(0, inf]',
}

EFFECTS: dict[str, str] = {
    'enabled': 'Enable solid-state mantle rheology and variable viscosity',
    'g_homologous': 'Homologous temperature scaling factor in activation enthalpy H(P)',
    't_sol_0': 'Zero-pressure solidus temperature for linear solidus and enthalpy normalization',
    'dt_sol_dp': 'Linear solidus Clapeyron slope when table solidus is absent',
    'arrhenius_t_ref': 'Reference temperature where mantle viscosity equals reference value at P=0',
    'viscosity_max_log10': 'Logarithmic viscosity ceiling preventing numerical overflow in cold lithosphere',
    'water_prefactor': 'Viscosity multiplier accounting for mantle hydration and volatile content',
    'yield_stress_c': 'Cohesion intercept in Byerlee brittle frictional yielding criterion',
    'yield_stress_mu': 'Friction coefficient in Byerlee brittle frictional yielding criterion',
    'yield_stress_max': 'Maximum ductile yield stress ceiling capping frictional strength at depth',
    'stress_closure_mode': 'Boundary-layer stress closure mode: "lid" for stagnant lid, "local" for local stress',
    'interior_flux_fraction': 'Convective heat flux fraction carried across interior boundary layer under lid',
    'lid_base_mode': 'Stagnant-lid base criterion: "rheological" for viscosity contrast, "fixed" for isotherm',
    'lid_base_temperature': 'Prescribed temperature at stagnant-lid base when lid_base_mode is "fixed"',
    'lid_contrast_coeff': 'Viscosity contrast exponent in rheological stagnant-lid thickness scaling',
    'lid_mask_width_cells': 'Smoothing transition width at stagnant-lid base in cell units',
    'phi_visc_single': 'Critical melt fraction transition width for solid-liquid viscosity reduction',
    'mlt_bottom_slope': 'Mixing-length boundary slope factor at core-mantle boundary (bottom boundary)',
    's_top_inf': 'Asymptotic top mixing-length slope at infinite viscosity contrast',
    'c_theta': 'Sensitivity of top mixing-length slope to rheological temperature contrast scale theta',
    'activation_energy': 'Activation energy Ea in saturating integral enthalpy law; homologous scaling when unset',
    'activation_volume': 'Zero-pressure activation volume V0 in saturating integral enthalpy law',
    'activation_volume_decay_pressure': 'Pressure decay scale P_decay for activation volume saturation; linear when inf',
}

_START_MARKER = '<!-- rheology-config-reference:start -->'
_END_MARKER = '<!-- rheology-config-reference:end -->'
_SECTION_PATTERN = re.compile(
    rf'{re.escape(_START_MARKER)}.*?{re.escape(_END_MARKER)}', re.DOTALL
)


def generate_table() -> str:
    """Generate Markdown table rows for all fields in SolidRheologyParams."""
    lines = [
        '| Key | Type | Unit | Default | Range | Effect |',
        '|-----|------|------|---------|-------|--------|',
    ]
    for f in dataclasses.fields(SolidRheologyParams):
        if f.type is bool or f.type == 'bool':
            tname = 'bool'
        elif f.type is str or f.type == 'str':
            tname = 'str'
        elif 'None' in str(f.type):
            tname = 'float \\| None'
        else:
            tname = 'float'

        val = f.default
        if val is False:
            dstr = '`false`'
        elif val is True:
            dstr = '`true`'
        elif isinstance(val, str):
            dstr = f'`"{val}"`'
        elif val is None:
            dstr = '`null`'
        else:
            dstr = f'`{val}`'

        unit = UNITS.get(f.name, '--')
        rng = RANGES.get(f.name, '--')
        eff = EFFECTS.get(f.name, '')
        lines.append(f'| `{f.name}` | `{tname}` | {unit} | {dstr} | {rng} | {eff} |')

    return '\n'.join(lines)


def check_table_fields(content: str) -> bool:
    """Verify that every field in SolidRheologyParams is present in the table."""
    match = _SECTION_PATTERN.search(content)
    if not match:
        print('Error: Rheology reference markers not found.', file=sys.stderr)
        return False
    table_text = match.group(0)
    found_keys = set(re.findall(r'^\|\s*`([a-z0-9_]+)`\s*\|', table_text, re.MULTILINE))
    expected_keys = {f.name for f in dataclasses.fields(SolidRheologyParams)}
    missing = expected_keys - found_keys
    extra = found_keys - expected_keys
    if missing:
        print(f'Error: Missing config row for fields: {sorted(missing)}', file=sys.stderr)
        return False
    if extra:
        print(f'Error: Unexpected keys in table: {sorted(extra)}', file=sys.stderr)
        return False
    return True


def update_doc(path: Path, write: bool = False, check: bool = False) -> int:
    """Update or verify the rheology reference table in the target document."""
    content = path.read_text(encoding='utf-8')
    table = generate_table()
    block = f'{_START_MARKER}\n{table}\n{_END_MARKER}'

    if _START_MARKER not in content or _END_MARKER not in content:
        print(f'Error: Markers not found in {path}', file=sys.stderr)
        return 1

    new_content = _SECTION_PATTERN.sub(block, content)

    if check:
        if not check_table_fields(content):
            return 1
        if new_content != content:
            print(f'Error: {path} is not up to date with SolidRheologyParams.', file=sys.stderr)
            return 1
        print(
            f'Check passed: {path} is up to date with all '
            f'{len(dataclasses.fields(SolidRheologyParams))} fields.'
        )
        return 0

    if write:
        path.write_text(new_content, encoding='utf-8')
        print(f'Updated {path}.')
        return 0

    print(table)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--doc',
        type=Path,
        default=Path('docs/How-to/configuration.md'),
        help='Path to configuration.md',
    )
    parser.add_argument('--write', action='store_true', help='Update configuration.md in place')
    parser.add_argument(
        '--check', action='store_true', help='Verify configuration.md is current'
    )
    args = parser.parse_args()

    doc_path = args.doc
    if not doc_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        doc_path = repo_root / doc_path

    if args.write or args.check:
        return update_doc(doc_path, write=args.write, check=args.check)

    print(generate_table())
    return 0


if __name__ == '__main__':
    sys.exit(main())
