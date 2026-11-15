"""The Gaussian core path gives the stored values of every budget and profile method.

The values in ``tests/reference/core_gaussian_golden.json`` come from the Gaussian path before
the tabulated profile was added; the tabulated path must leave this path unchanged. Write the
file again with ``python tests/test_core_gaussian_default_invariance.py --write``.
"""

from __future__ import annotations

import json
import pathlib
import sys

import jax
import numpy as np
import pytest

from aragog.core.module import build_core_module_budget

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

GOLDEN = pathlib.Path(__file__).parent / 'reference' / 'core_gaussian_golden.json'
# Factory budget of an Earth core and a fitted exoplanet core with light elements.
CASES = {
    'earth_factory': (
        {'p_cmb': 136e9},
        {'r_cmb': 3480e3, 'p_cmb_fallback': 136e9},
    ),
    'exoplanet_fit_alloy': (
        {
            'm_core': 9.8365400909e23,
            'p_cen': 184117858870.0,
            'light_element_fraction': 0.1,
            'depression': 1.2,
            'alpha_c': 0.8,
            'c_light': 0.05,
        },
        {'r_cmb': 2867012.4963, 'p_cmb_fallback': 54848888186.0},
    ),
}
T_CMB = [3000.0, 3300.0, 3600.0, 4100.0, 4400.0, 4600.0, 5200.0, 6000.0]
RADII = [0.0, 0.25, 0.5, 0.75, 1.0]  # fractions of r_cmb


def _values(budget) -> dict:
    """Every budget and profile output on the fixed T_cmb and radius grids, as floats."""
    p = budget.profiles
    r = [x * p.r_cmb for x in RADII]
    t = np.asarray(T_CMB)
    out = {
        'density': [float(p.density(x)) for x in r],
        'enclosed_mass': [float(p.enclosed_mass(x)) for x in r],
        'gravity': [float(p.gravity(x)) for x in r],
        'pressure': [float(p.pressure(x)) for x in r],
        'potential': [float(p.potential(x)) for x in r],
        'adiabat': [float(p.adiabat(x, 4000.0)) for x in r],
        'adiabat_gradient': [float(p.adiabat_gradient(x, 4000.0)) for x in r],
        'r_peak': [float(p.r_peak)],
        'secular_capacity': [float(budget.secular_capacity())],
        'effective_tfac': [float(budget.effective_tfac())],
        't_onset': [budget.t_onset],
        't_freeze': [budget.t_freeze],
        'superheat_centre': [float(budget.superheat(0.0, x)) for x in t],
        'r_icb': [float(budget.r_icb(x)) for x in t],
        'nucleation_factor': [float(budget.nucleation_factor(x)) for x in t],
        'latent_capacity': [float(budget.latent_capacity(x)) for x in t],
        'gravitational_capacity': [float(budget.gravitational_capacity(x)) for x in t],
        'effective_capacity': [float(budget.effective_capacity(x)) for x in t],
        'd_capacity_dt': [float(jax.grad(budget.effective_capacity)(x)) for x in t],
        'dtcmb_dt': [float(budget.dtcmb_dt(x, 1e13, 1e12)) for x in t],
        'heat_content': [budget.heat_content(x) for x in t],
        'heat_change': [budget.heat_change(x, -37.5) for x in t],
    }
    return out


def _all_values() -> dict:
    """Values of every case, keyed by case name."""
    return {
        name: _values(build_core_module_budget(params, **geometry))
        for name, (params, geometry) in CASES.items()
    }


def _assert_matches(key, got, ref):
    """Assert `got` equals the stored `ref` of output `key` to 1e-12 relative.

    The nucleation factor f = sigmoid(-x) takes 1e-12 * max(1, |ln f|), which is 1e-12 on x.
    """
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    rtol = np.full(ref.shape, 1e-12)
    if key == 'nucleation_factor':
        rtol *= np.maximum(1.0, np.abs(np.log(np.where(ref > 0.0, ref, 1.0))))
    assert got.shape == ref.shape, key
    bad = ~(np.abs(got - ref) <= rtol * np.abs(ref))
    assert not bad.any(), (key, got[bad], ref[bad])


def test_the_comparison_takes_the_round_off_of_another_cpu():
    """The nucleation factor of a Linux x86 runner passes; a 1e-9 change of its exponent fails.

    The factor is sigmoid(-x) with x near 21.8 at 3600 K, so a round-off of x reaches the
    factor multiplied by x; the Linux value differs from the stored one by 1.002e-12.
    """
    stored = 3.398694876168228e-10
    _assert_matches('nucleation_factor', [3.3986948761716334e-10], [stored])
    x = -np.log(stored)
    with pytest.raises(AssertionError):
        _assert_matches('nucleation_factor', [float(np.exp(-x * (1 + 1e-9)))], [stored])
    with pytest.raises(AssertionError):
        _assert_matches('density', [1.0 + 2e-12], [1.0])


@pytest.mark.physics_invariant
@pytest.mark.parametrize('case', sorted(CASES))
def test_gaussian_path_is_unchanged(case):
    """Each method of the Gaussian budget gives its stored value to 1e-12 relative.

    1e-12 passes the round-off of another CPU and refuses any change of a formula, a quadrature
    order or a default; the stored grid crosses the inner-core onset and freeze-out of both
    cases, so the latent and gravitational branches are held as well as the secular one.
    """
    stored = json.loads(GOLDEN.read_text())[case]
    params, geometry = CASES[case]
    got = _values(build_core_module_budget(params, **geometry))
    assert sorted(got) == sorted(stored)
    for key, ref in stored.items():
        _assert_matches(key, got[key], ref)
    # The grid must reach the inner core, or the boundary terms are held only at zero.
    assert max(stored['latent_capacity']) > 0.0
    assert max(stored['r_icb']) > 0.0


if __name__ == '__main__' and '--write' in sys.argv:
    vals = _all_values()
    GOLDEN.write_text(
        '{\n'
        + ',\n'.join(f'{json.dumps(k)}: {json.dumps(v)}' for k, v in vals.items())
        + '\n}\n'
    )
