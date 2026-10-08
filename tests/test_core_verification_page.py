"""The core verification page: every tagged number matches the values file, every test it
names exists, and items 4, 6, 10, 11 and 13 of the script reproduce their recorded values."""

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
ROUNDING = {
    'jvp_vs_fd_max_rel',
    'secular_max_rel',
    'secular_entropy_max_rel',
    'conduction_sink_rel',
    't_cmb_max_abs_diff_before_onset',
}
# Item 13 values that are a rounding-level residual or a spread between two integrations
RUN_SPREAD = {f'identity_{t}' for t in ('8TW', '12TW', '-2TW', '0TW')} | {
    'erosion_identity',
    'tcen_max_rtol_noise_K',
    'tcen_max_abs_diff_K_rtol1e-8',
    'onset_max_abs_diff_myr_rtol1e-8',
}
TAG = re.compile(r'(-?[\d.]+(?:e[-+]?\d+)?)\s?%?<!--k:(\d+)\.([^:>]+)(?::(-?[\d.]+))?-->')


@pytest.mark.unit
def test_page_numbers_match_the_values():
    """A tagged number equals its value times the optional factor to the last digit shown."""
    text = PAGE.read_text()
    tags = TAG.findall(text)
    assert len(tags) == text.count('<!--k:') > 100
    for shown, item, key, factor in tags:
        value = VALUES[item][key] * float(factor or 1)
        half_digit = 0.5 * 10.0 ** Decimal(shown).as_tuple().exponent
        assert abs(float(shown) - value) <= half_digit * (1 + 1e-9), (item, key, shown, value)


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
    """Run one item and compare what it records with the values file: a rounding-level
    error in ROUNDING or RUN_SPREAD may not double, every other number holds to
    1e-4."""
    script.ITEMS[item]()
    got, want = script.VALUES[str(item)], VALUES[str(item)]
    assert got.keys() == want.keys()
    for key, value in got.items():
        if key in ROUNDING or key in RUN_SPREAD:
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
    inputs = script._thermal_history()[0]['inputs']  # sections 5 and 7 use the same core
    assert {k: inputs[k] for k in script.NIMMO} == script.NIMMO
    assert max(got[f'{k}_max_rel'] for k in ('secular', 'latent', 'latent_entropy')) < 1e-3
    assert got['gravitational_enrichment_corrected_max_rel'] < 1e-3


@pytest.mark.slow
@pytest.mark.reference_pinned
def test_core_history_matches_the_thermal_history_table(script):
    got = _reproduces(script, 10)
    assert got['t_cmb_max_abs_diff'] < 0.2 and got['onset_myr_aragog'] == got['onset_myr_leeds']


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_coupled_tables_reproduce_the_page(script):
    got = _reproduces(script, 11)
    assert got['wrong_sign_rows_core_module'] == 0
    assert got['wrong_sign_rows_energy_balance'] == got['rows_energy_balance']
    assert got['core_residual_frac_max_after_1kyr'] < 1e-5


@pytest.mark.slow
@pytest.mark.reference_pinned
def test_the_stable_layer_meets_the_thermal_history_bounds(script):
    """Layers under fixed CMB flows stay within the bounds against thermal_history: T_cmb 5 K,
    T_cen 10 K, theta-0.1 depth within a factor 1.5, the inner core within 2 % at the end with
    its onset within 5 %. Under the eroding flow the centre temperature stays within 10 K, the
    stored heat drains in the time the shell takes to mix, and the layer re-forms to a depth
    within a factor 1.5; the CMB temperature there is a model difference without a bound. The
    heat rates of core and shell equal the CMB flow at every sample."""
    got = _reproduces(script, 13)
    for t in ('8TW', '12TW', '-2TW', '0TW'):
        assert got[f'identity_{t}'] < 1e-12
        assert got[f'tcmb_max_abs_diff_K_{t}'] < 5.0
        assert got[f'tcen_max_abs_diff_K_{t}'] < 10.0
        assert 1 / 1.5 < got[f'layer_ratio_min_{t}'] <= got[f'layer_ratio_max_{t}'] < 1.5
        assert got[f'ricb_end_rel_diff_{t}'] < 0.02
        assert got[f'onset_myr_aragog_{t}'] == pytest.approx(
            got[f'onset_myr_leeds_{t}'], rel=0.05
        )
    assert got['erosion_identity'] < 1e-12
    assert got['erosion_tcen_max_abs_diff_K'] < 10.0
    eroding = got['erosion_removed_myr_aragog'] - got['erosion_start_myr']
    assert got['erosion_drain_myr'] == pytest.approx(eroding, rel=0.05)
    reformed = got['erosion_depth_km_end_aragog'] / got['erosion_depth_km_end_leeds']
    assert 1 / 1.5 < reformed < 1.5
