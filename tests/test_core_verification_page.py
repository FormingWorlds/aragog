"""The core verification page: every tagged number matches the values file, every test it
names exists, and items 4, 5, 6, 9, 10, 11 and 13 of the script reproduce their recorded values."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from tests.conftest import needs_eos

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
FLOWS = ('8TW', '12TW', '-2TW', '0TW')
# Values set by the integration: a rounding-level residual, a closure error, the spread between
# two tolerances, or a count of BDF stalls (items 9 and 13)
RUN_SPREAD = {f'identity_{t}' for t in FLOWS} | {
    'erosion_identity',
    'tcen_max_rtol_noise_K',
    'bdf_restarts',
    'core_vs_cmb_rel',
    'core_vs_cmb_rel_rtol1e-8',
    'reconstructed_cmb_max_rel',
}
# Item 13 differences of an unconverged run from thermal_history: within a factor 2 either way
UNCONVERGED = {'tcen_max_abs_diff_K_rtol1e-8', 'onset_max_abs_diff_myr_rtol1e-8'}
# Item 13 T_cen changes under the mixing constants, of the size of the integration noise: sign only
SIGN_ONLY = {'mixing_tcen_max_K', 'mixing_tcen_min_K'}
# Differences of two solutions that the integration moves by more than 1e-4 of themselves: held
# to 2 to 3 times the largest change under a tenfold tighter tolerance or a 1-ulp start temperature
BANDS = {
    # item 10, LSODA at rtol 1e-10: these two T_cmb differences move by up to 1.0e-6 K
    **dict.fromkeys(('t_cmb_abs_diff_at_onset', 't_cmb_diff_max_after_onset_K'), 3e-6),
    # item 13: T_cen moves by 0.25 K at -2 TW, 0.024 K or less in the others (Linux 8 TW 0.072 K)
    **{f'tcen_max_abs_diff_K_{t}': 0.6 if t == '-2TW' else 0.2 for t in FLOWS},
    # item 13: the end inner-core radius moves by up to 8.3e-5 of itself
    **dict.fromkeys([f'ricb_end_rel_diff_{t}' for t in FLOWS], 2e-4),
    # item 13: the late layer depth, its change under the mixing constants and the end depth
    'depth_late_max_rel': 2e-3,
    'mixing_depth_max_rel': 5e-3,
    'erosion_depth_km_end_aragog': 0.2,
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


def _holds(key, value, want):
    """A value in ROUNDING or RUN_SPREAD may not double, one in UNCONVERGED stays within a
    factor 2 either way, one in SIGN_ONLY stays positive, one in BANDS within its band, every
    other number holds to 1e-4."""
    if key in SIGN_ONLY:
        return value > 0
    if key in ROUNDING or key in RUN_SPREAD:
        return abs(value) <= 2.0 * abs(want) + 1e-15
    if key in UNCONVERGED:
        return 0.5 * want <= value <= 2.0 * want
    if key in BANDS:
        return abs(value - want) <= BANDS[key]
    return value == pytest.approx(want, rel=1e-4, abs=1e-12)


def _reproduces(script, item):
    """Run one item and compare every value it records with the values file."""
    script.ITEMS[item]()
    got, want = script.VALUES[str(item)], VALUES[str(item)]
    assert got.keys() == want.keys()
    bad = [(k, v, want[k]) for k, v in got.items() if not _holds(k, v, want[k])]
    assert not bad, bad
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


@needs_eos
@pytest.mark.slow
@pytest.mark.physics_invariant
def test_a_cvode_solve_across_the_onset_closes_its_heat(script):
    got = _reproduces(script, 9)
    assert 0 < got['core_vs_cmb_rel'] < 1e-6 < got['core_vs_cmb_rel_rtol1e-8']


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
    assert got['mixing_depth_max_rel'] > 0  # with mixing_tcen_min_K > 0: both constants act
    for t in FLOWS:
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
