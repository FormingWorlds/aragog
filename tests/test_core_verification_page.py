"""The core verification page: every tagged number matches the values file, every test it
names exists, and the fast items of the script reproduce their recorded values."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'docs' / 'Explanations' / 'core_verification.md'
VALUES = json.loads(
    (ROOT / 'docs' / 'figures' / 'vv' / 'core_verification_values.json').read_text()
)
TAG = re.compile(r'(-?[\d.]+(?:e[-+]?\d+)?)\s?%?<!--k:(\d+)\.([^:>]+)(?::(-?[\d.]+))?-->')

pytestmark = pytest.mark.unit


def test_page_numbers_match_the_values():
    """A tagged number equals its value times the optional factor to the last digit shown."""
    tags = TAG.findall(PAGE.read_text())
    assert len(tags) > 100
    for shown, item, key, factor in tags:
        value = VALUES[item][key] * float(factor or 1)
        half_digit = 0.5 * 10.0 ** Decimal(shown).as_tuple().exponent
        assert abs(float(shown) - value) <= max(half_digit, 5e-13), (item, key, shown, value)


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
    """Run one item and compare every number it records with the values file."""
    script.ITEMS[item]()
    got = script.VALUES[str(item)]
    assert got.keys() == VALUES[str(item)].keys()
    for key, value in got.items():
        assert value == pytest.approx(VALUES[str(item)][key], rel=1e-6, abs=1e-12), key
    return got


@pytest.mark.physics_invariant
def test_inner_core_radius_grows_as_the_square_root_of_undercooling(script):
    got = _reproduces(script, 4)
    assert abs(got['sqrt_slope'] - 0.5) < 1e-3 and got['jvp_vs_fd_max_rel'] < 1e-5


@pytest.mark.reference_pinned
def test_budget_terms_match_the_thermal_history_table(script):
    got = _reproduces(script, 6)
    assert max(got[f'{k}_max_rel'] for k in ('secular', 'latent', 'latent_entropy')) < 1e-3
    assert got['gravitational_enrichment_corrected_max_rel'] < 1e-3


@pytest.mark.reference_pinned
def test_core_history_matches_the_thermal_history_table(script):
    got = _reproduces(script, 10)
    assert got['t_cmb_max_abs_diff'] < 0.1 and got['onset_myr_aragog'] == got['onset_myr_leeds']


@pytest.mark.physics_invariant
def test_coupled_tables_reproduce_the_page(script):
    got = _reproduces(script, 11)
    assert got['wrong_sign_rows_core_module'] == 0
    assert got['wrong_sign_rows_energy_balance'] == got['rows_energy_balance']
    assert np.isclose(got['core_residual_frac_end'], 0.0, atol=1e-4)
