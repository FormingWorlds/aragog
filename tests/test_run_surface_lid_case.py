"""The JAX right-hand side and the options of ``tools/verification/run_surface_lid_case.py``."""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import entropy_eos_copy
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, _build, needs_eos

jax = pytest.importorskip('jax')

pytestmark = [pytest.mark.timeout(600), needs_eos]

DRIVER = Path(__file__).parents[1] / 'tools' / 'verification' / 'run_surface_lid_case.py'


@functools.cache
def _driver():
    spec = importlib.util.spec_from_file_location('run_surface_lid_case', DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rates(eos, heating, jax_rhs):
    """dS/dt [J/(kg K yr)] of the 10-node lid column at its start, NumPy or through the driver's
    JAX factory, with internal heating [W/kg] or without."""
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=10, end_time=1.0)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.normalize()
    if heating:
        p.energy.tidal, p.energy.tidal_array = True, np.array([heating])
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(3050.0)
    y = np.asarray(s._S0, float).copy()
    if not jax_rhs:
        return np.asarray(s.dSdt(0.0, y), float)[: s._n_stag]
    _driver()._install_factory(s, EOS_DIR)
    sc = s._build_nondim_scales()
    rhs, _ = s._jax_cvode_factory(sc, s._core_bc)
    off = 0.0 if sc.state_offset is None else np.asarray(sc.state_offset)
    y_nd = (y - off) / np.asarray(sc.state_scale)
    yd = np.zeros_like(y_nd)
    rhs(0.0, y_nd, yd)
    return (yd * np.asarray(sc.state_scale) / float(sc.t_ref))[: s._n_stag]


def test_driver_jax_rhs_carries_the_internal_heating():
    """The heating set in the config reaches the JAX right-hand side of the driver: its heated
    minus unheated rate equals that of the NumPy right-hand side in every cell."""
    eos = entropy_eos_copy(EOS_DIR)
    H = 3.0e-7
    d_np = _rates(eos, H, False) - _rates(eos, None, False)
    d_jax = _rates(eos, H, True) - _rates(eos, None, True)
    assert np.all(d_np > 0.0)
    np.testing.assert_allclose(d_jax, d_np, rtol=1e-9, atol=0.0)


def _options(**kw):
    import argparse

    return argparse.Namespace(**{'eta': None, 'heating': None, **kw})


@given(st.floats(allow_nan=True, allow_infinity=True))
@settings(max_examples=200, deadline=None)
def test_eta_option_accepts_only_finite_positive_values(eta):
    """--eta passes when finite and positive and stops the run otherwise."""
    check = _driver()._check_options
    if np.isfinite(eta) and eta > 0.0:
        check(_options(eta=eta))
    else:
        with pytest.raises(SystemExit, match='--eta must be finite and positive'):
            check(_options(eta=eta))


@given(st.floats(allow_nan=True, allow_infinity=True))
@settings(max_examples=200, deadline=None)
def test_heating_option_accepts_only_finite_non_negative_values(heating):
    """--heating passes when finite and non-negative and stops the run otherwise."""
    check = _driver()._check_options
    if np.isfinite(heating) and heating >= 0.0:
        check(_options(heating=heating))
    else:
        with pytest.raises(SystemExit, match='--heating must be finite and non-negative'):
            check(_options(heating=heating))


@pytest.mark.parametrize('flag', ['--eta', '--heating'])
def test_options_refuse_a_value_that_is_not_a_number(flag):
    """A value that is not a number (as a TOML true) stops the parser; no value keeps the
    default None."""
    d = _driver()
    with pytest.raises(SystemExit):
        d.parse(['out', flag, 'true'])
    assert getattr(d.parse(['out']), flag[2:]) is None
