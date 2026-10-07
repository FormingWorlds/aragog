"""The core verification page: every tagged number matches the values file, every test it
names exists, and items 4, 6, 10 and 11 of the script reproduce their recorded values."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'docs' / 'Explanations' / 'core_verification.md'
VALUES = json.loads(
    (ROOT / 'docs' / 'figures' / 'vv' / 'core_verification_values.json').read_text()
)
TAG = re.compile(r'(-?[\d.]+(?:e[-+]?\d+)?)\s?%?<!--k:(\d+)\.([^:>]+)(?::(-?[\d.]+))?-->')


@pytest.mark.unit
def test_page_numbers_match_the_values():
    """A tagged number equals its value times the optional factor to the last digit shown."""
    tags = TAG.findall(PAGE.read_text())
    assert len(tags) > 100
    for shown, item, key, factor in tags:
        value = VALUES[item][key] * float(factor or 1)
        half_digit = 0.5 * 10.0 ** Decimal(shown).as_tuple().exponent
        assert abs(float(shown) - value) <= half_digit, (item, key, shown, value)


@pytest.mark.unit
def test_tests_named_on_the_page_exist():
    """`tests/x.py::name` and the `::name` shorthand after it point at real test functions."""
    path = None
    for file, name in re.findall(r'`(tests/\w+\.py)?(?:::(\w+))?`', PAGE.read_text()):
        path = ROOT / file if file else path
        if file or name:
            assert path.exists(), path
            assert not name or f'def {name}(' in path.read_text(), (path.name, name)


@pytest.fixture
def script(tmp_path, monkeypatch):
    """The verification script with its figures sent to a temporary folder."""
    sys.path.insert(0, str(ROOT / 'tools' / 'verification'))
    import run_core_verification as script

    monkeypatch.setattr(script, 'OUT', tmp_path)
    monkeypatch.setattr(script, 'VALUES', {})
    yield script
    sys.path.remove(str(ROOT / 'tools' / 'verification'))


def _reproduces(script, item):
    """Run one item and compare what it records with the values file: an error measure
    (a key with 'rel' or 'diff') may not double, every other number holds to 1e-4."""
    script.ITEMS[item]()
    got, want = script.VALUES[str(item)], VALUES[str(item)]
    assert got.keys() == want.keys()
    for key, value in got.items():
        if 'rel' in key or 'diff' in key:
            assert abs(value) <= 2.0 * abs(want[key]) + 1e-15, key
        else:
            assert value == pytest.approx(want[key], rel=1e-4, abs=1e-12), key
    return got


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_inner_core_radius_grows_as_the_square_root_of_undercooling(script):
    got = _reproduces(script, 4)
    assert abs(got['sqrt_slope'] - 0.5) < 1e-3 and got['jvp_vs_fd_max_rel'] < 1e-5


@pytest.mark.slow
@pytest.mark.reference_pinned
def test_budget_terms_match_the_thermal_history_table(script):
    got = _reproduces(script, 6)
    assert max(got[f'{k}_max_rel'] for k in ('secular', 'latent', 'latent_entropy')) < 1e-3
    assert got['gravitational_enrichment_corrected_max_rel'] < 1e-3


@pytest.mark.slow
@pytest.mark.reference_pinned
def test_core_history_matches_the_thermal_history_table(script):
    got = _reproduces(script, 10)
    assert got['t_cmb_max_abs_diff'] < 0.1 and got['onset_myr_aragog'] == got['onset_myr_leeds']


@pytest.mark.slow
def test_coupled_tables_reproduce_the_page(script):
    got = _reproduces(script, 11)
    assert got['wrong_sign_rows_core_module'] == 0
    assert got['wrong_sign_rows_energy_balance'] == got['rows_energy_balance']
    assert abs(got['core_residual_frac_end']) <= 1e-4
