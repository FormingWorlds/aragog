"""Solve-level checks of accepted states, end states and the temperature tables."""

from __future__ import annotations

import numpy as np
import pytest

from aragog.eos import entropy as eos_mod
from aragog.solver.entropy_solver import AcceptedStateError

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos
from .test_eos_floor import _FLUX, _drained

pytestmark = [pytest.mark.smoke, needs_eos]


@pytest.fixture(scope='module')
def shared_eos():
    return entropy_eos_copy(EOS_DIR)


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_drain_without_the_floor_raises_instead_of_clamping(shared_eos, method):
    S_min = float(shared_eos.S_min)
    with pytest.raises(AcceptedStateError) as e:
        _drained(shared_eos, S_min + 400.0, margin=None, method=method)
    assert e.value.check == 'entropy_table' and e.value.bound == S_min
    assert e.value.value < S_min and 0.0 < e.value.time <= 10.0


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
@pytest.mark.parametrize('flux', [_FLUX, 0.0])
def test_a_call_inside_the_tables_does_not_raise(shared_eos, method, flux):
    """The floor keeps a drained call inside; a call without the drain stays inside."""
    s, f = _drained(shared_eos, float(shared_eos.S_min) + 400.0, flux=flux, method=method)
    assert f.min() >= float(shared_eos.S_min)


@pytest.mark.parametrize('bad', [0.0, -5.0, np.nan])
def test_a_temperature_table_with_a_non_positive_or_non_finite_value_is_refused(
    monkeypatch, bad
):
    load = eos_mod._load_spider_ps_table

    def spoiled(path, *a, **k):
        tbl = load(path, *a, **k)
        if path.name == 'temperature_melt.dat':
            tbl['values'] = np.array(tbl['values'], dtype=float)
            tbl['values'][3, 4] = bad
        return tbl

    monkeypatch.setattr(eos_mod, '_load_spider_ps_table', spoiled)
    with pytest.raises(ValueError, match='temperature_melt'):
        eos_mod.EntropyEOS(EOS_DIR)


def test_get_state_checks_the_end_state(shared_eos, monkeypatch):
    s, _ = _drained(shared_eos, float(shared_eos.S_min) + 400.0, flux=0.0)
    s.get_state()
    monkeypatch.setattr(
        s.entropy_eos, 'melt_fraction', lambda P, S: np.full(np.size(S), np.nan)
    )
    with pytest.raises(AcceptedStateError) as e:
        s.get_state()
    assert e.value.check == 'end_melt_fraction'
