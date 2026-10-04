"""The cached ``EntropyEOS`` copies that the tests share are independent."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.conftest import entropy_eos_copy, needs_eos


@pytest.mark.smoke
@needs_eos
def test_mutating_one_copy_leaves_the_next_copy_pristine(caplog):
    P = np.array([5.0e10])
    first = entropy_eos_copy()
    S_solid = first.solidus_entropy(P) - 500.0
    T_pristine = first.temperature(P, S_solid)
    S_out = np.array([first.S_max + 1.0e5])

    first._check_entropy_range(S_out, first.S_min, first.S_max, 'probe')
    first._tables['temperature_solid']['interp'].values[:] = np.nan
    assert first._range_warning_counts == {'probe': 1}
    assert np.isnan(first.temperature(P, S_solid)).all()

    second = entropy_eos_copy()
    assert second._range_warning_counts == {}
    np.testing.assert_array_equal(second.temperature(P, S_solid), T_pristine)
    with caplog.at_level(logging.WARNING):
        second._check_entropy_range(S_out, second.S_min, second.S_max, 'probe')
    assert any('probe' in r.message for r in caplog.records)
