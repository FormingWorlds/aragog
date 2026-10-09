"""The core verification page: every tagged number matches the values file, every test it
names exists, and items 4, 5, 6, 10, 11 and 13 of the script reproduce their recorded values."""

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
# Item 13 values that are a rounding-level residual or the spread between two tolerances
RUN_SPREAD = {f'identity_{t}' for t in ('8TW', '12TW', '-2TW', '0TW')} | {
    'erosion_identity',
    'tcen_max_rtol_noise_K',
}
# Item 13 differences of an unconverged run from thermal_history: within a factor 2 either way
UNCONVERGED = {'tcen_max_abs_diff_K_rtol1e-8', 'onset_max_abs_diff_myr_rtol1e-8'}
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
    error in ROUNDING or RUN_SPREAD may not double, a value in UNCONVERGED stays within a
    factor 2 either way, every other number holds to 1e-4."""
    script.ITEMS[item]()
    got, want = script.VALUES[str(item)], VALUES[str(item)]
    assert got.keys() == want.keys()
    for key, value in got.items():
        if key in ROUNDING or key in RUN_SPREAD:
            assert abs(value) <= 2.0 * abs(want[key]) + 1e-15, key
        elif key in UNCONVERGED:
            assert 0.5 * want[key] <= value <= 2.0 * want[key], key
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
def test_the_earth_budget_meets_nimmo_tables_4_and_5(script):
    """Every energy and entropy term of Nimmo (2015, Table 4) and the cooling rate hold to 2 %
    at both heat flows; four quantities are documented exceptions, held at their measured
    values: the growth rate (aragog's C_r against the printed one), the inner-core age (Table 4
    divides the drop by the present cooling rate), W_s and W_tot (the chapter's own Table 4
    capacity times its 60 K drop is 4 % below its Table 5 W_s)."""
    got = _reproduces(script, 5)
    for flow in ('15.2TW', '12TW'):
        for term in ('Qs', 'QL', 'Qg', 'Qk', 'Es', 'EL', 'Eg', 'Ek', 'cooling'):
            assert abs(got[f'{term}_ratio_{flow}'] - 1.0) < 0.02, (term, flow)
        assert got[f'growth_ratio_{flow}'] == pytest.approx(1.024, abs=1e-3)
    assert got['age_ratio_15.2TW'] == pytest.approx(0.786, abs=1e-3)
    assert got['age_ratio_12TW'] == pytest.approx(0.783, abs=1e-3)
    assert got['Ws_ratio'] == pytest.approx(0.947, abs=1e-3)
    assert got['Wtot_ratio'] == pytest.approx(0.967, abs=1e-3)
    assert abs(got['WL_ratio'] - 1.0) < 0.02 and abs(got['Wg_ratio'] - 1.0) < 0.02
    assert got['Ws_from_table4_1e28J'] / 11.6 == pytest.approx(0.957, abs=1e-3)


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
    assert got['t_cen_max_abs_diff'] < 1.0 and got['r_icb_end_rel_diff'] < 0.01


@pytest.mark.slow
@pytest.mark.physics_invariant
def test_coupled_tables_reproduce_the_page(script):
    got = _reproduces(script, 11)
    assert got['wrong_sign_rows_energy_balance'] == got['rows_energy_balance']
    for case, bound in (('1me', 1e-6), ('1me_rtol1e-10', 1e-6), ('3me', 5e-6), ('5me', 5e-6)):
        assert got[f'closure_max_after_1kyr_{case}'] < bound
        assert got[f'closure_end_{case}'] < bound
        assert got[f'closure_first_call_{case}'] < 3e-4  # 1.5e-4 at 1 Earth mass
    for case in ('core_module', '3me', '5me', 'wb_1me'):
        assert got[f'wrong_sign_rows_{case}'] == 0 < got[f'rows_{case}']
        assert got[f'zero_flux_rows_{case}'] <= 1
    for mass in ('1me', '3me', '5me'):  # no inner core in any run
        assert got[f't_cmb_end_K_{mass}'] > max(
            got[f't_onset_K_{mass}'], got[f't_freeze_K_{mass}']
        )
    assert got['t_freeze_K_1me'] < got['t_onset_K_1me']  # bottom-up at 1 Earth mass
    assert (
        got['t_freeze_K_3me'] > got['t_onset_K_3me']
        and got['t_freeze_K_5me'] > got['t_onset_K_5me']
    )
    # E1, bounds set before the runs: 5 % on times and fluxes, 10 K on temperatures
    for q in ('t_bf', 'F10', 'F100', 'F1000', 'F_bf'):
        assert got[f'e1_{q}_d_160_320'] < 0.05
        assert got[f'e1_{q}_d_80_rtol'] < 1e-3
    assert got['e1_T_core_bf_d_160_320'] < 10.0 and got['e1_T_core_end_d_160_320'] < 10.0
    assert got['e1_T_core_bf_d_80_rtol'] < 0.01
    for pair in ('40_80', '80_160', '160_320'):  # a half-cell conduction flux would double
        for q in ('F_2bf', 'F_4bf', 'F_10kyr', 'F_100kyr', 'F_500kyr'):
            assert got[f'e1_{q}_d_{pair}'] < 0.1


@pytest.mark.slow
@pytest.mark.reference_pinned
def test_the_stable_layer_meets_the_thermal_history_bounds(script):
    """Layers under fixed CMB flows stay within the bounds against thermal_history: T_cmb 5 K,
    T_cen 10 K, theta-0.1 depth within a factor 1.5, the inner core within 2 % at the end with
    its onset within 5 %. Under the eroding flow the centre temperature, once the shell has mixed,
    stays within 0.05 K of aragog without a layer, the stored heat drains in the time the shell
    takes to mix, and the layer re-forms to a depth within a factor 1.5. The heat rates of core
    and shell equal the CMB flow at every sample."""
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
    assert got['tcmb_max_diff_sign_max'] < 0  # aragog colder at the largest difference
    assert got['erosion_identity'] < 1e-12
    assert got['erosion_tcen_self_max_K'] < 0.05
    eroding = got['erosion_removed_myr_aragog'] - got['erosion_start_myr']
    assert got['erosion_drain_myr'] == pytest.approx(eroding, rel=0.05)
    reformed = got['erosion_depth_km_end_aragog'] / got['erosion_depth_km_end_leeds']
    assert 1 / 1.5 < reformed < 1.5
