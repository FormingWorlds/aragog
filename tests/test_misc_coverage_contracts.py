"""Tests for miscellaneous edge branches across boundary, EOS, parser, and rheology.

Exercises array return in ``cmb_flux``, const_log10visc and scalar-P rheology in
``EntropyPhaseEvaluator``, legacy config rejection in ``Parameters.from_file``,
and alias/dir handling in ``aragog.rheology``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import aragog.rheology as rheo
from aragog.cmb_boundary_layer import cmb_flux
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.parser import Parameters

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)


@pytest.mark.unit
def test_cmb_flux_returns_array_for_array_inputs():
    """cmb_flux returns ndarray when temperature inputs are arrays."""
    T_c = np.array([4000.0, 4100.0])
    T_s = np.array([1800.0, 1850.0])
    T_m = np.array([3200.0, 3250.0])
    q = cmb_flux(
        T_c=T_c,
        T_s=T_s,
        T_m=T_m,
        depth=2.89e6,
        rho=4000.0,
        g=9.81,
        alpha=3e-5,
        kappa=1e-6,
        eta=1e21,
        k=4.0,
    )
    assert isinstance(q, np.ndarray)
    assert q.shape == (2,)
    assert np.all(np.isfinite(q))
    assert np.all(q > 0.0)


@pytest.mark.unit
def test_entropy_phase_const_log10visc():
    """EntropyPhaseEvaluator exposes const_log10visc property."""
    ev = EntropyPhaseEvaluator(
        entropy_eos=None,
        gravitational_acceleration=9.81,
        const_properties=True,
        const_rho=4000.0,
        const_Cp=1200.0,
        const_cond=4.0,
        const_log10visc=21.0,
    )
    assert ev.const_log10visc == pytest.approx(21.0, rel=1e-12)


@needs_eos
@pytest.mark.unit
def test_entropy_phase_scalar_p_and_s_update():
    """_update_eos handles 0D scalar pressure and entropy inputs with rheology."""
    from aragog.eos.entropy import EntropyEOS

    eos = EntropyEOS(EOS_DIR)
    ev = EntropyPhaseEvaluator(
        entropy_eos=eos,
        gravitational_acceleration=9.81,
        enabled=True,
        stress_closure_mode='lid',
        lid_base_mode='fixed',
    )
    ev.pressure = 3.0e9
    ev.entropy = 3000.0
    ev.update()
    assert ev.viscosity() is not None
    assert np.isfinite(ev.viscosity()).all()
    assert ev._eta_diff is not None
    assert ev._tau_y is not None
    assert ev._eta_diff.ndim == 1
    assert len(ev._eta_diff) == 1


@pytest.mark.unit
def test_parser_rejects_legacy_yield_switch_width():
    """Parameters.from_file rejects legacy yield_switch_width setting in .cfg."""
    cfg_content = """[rheology]
yield_switch_width = 1.0e6
"""
    with tempfile.NamedTemporaryFile('w', suffix='.cfg', delete=False) as f:
        f.write(cfg_content)
        f_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Rheology field 'yield_switch_width'"):
            Parameters.from_file(f_path)
    finally:
        f_path.unlink(missing_ok=True)


@pytest.mark.unit
def test_rheology_t_ref_alias_and_module_dir():
    """compute_arrhenius_viscosity accepts t_ref alias and dir(aragog.rheology) includes exports."""
    eta1 = rheo.compute_arrhenius_viscosity(
        temperature=1600.0, pressure=0.0, t_ref=1600.0, viscosity_solid=1.0e20
    )
    assert eta1 == pytest.approx(1.0e20, rel=1e-12)

    d = dir(rheo)
    assert 'compute_stagnant_lid_state' in d
    assert 'compute_effective_viscosity' in d
    assert 'stress_closure' in d
