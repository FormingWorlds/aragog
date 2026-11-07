"""The schematic of docs/Explanations/core_alloy_liquidus.md: it builds in both themes with no
text on a curve, on another text or on the frame, and its curves are the output of
``AlloyLiquidus``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from aragog.core import AlloyLiquidus, IronMeltingCurve, eutectic

pytestmark = pytest.mark.unit
pytest.importorskip('proteus_mpl')

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools' / 'verification' / 'figures' / 'sketch_alloy_liquidus.py'


@pytest.fixture(scope='module')
def sketch():
    spec = importlib.util.spec_from_file_location('sketch_alloy_liquidus', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_the_sketch_has_no_text_on_a_curve(sketch, theme):
    """No callout, legend entry or note touches a curve, another text or the frame; a text put
    on the pure-iron curve is found."""
    import matplotlib.pyplot as plt

    fig, ax = sketch.figure(theme)
    try:
        assert sketch.collisions(fig, ax) == []
        ax.text(200.0, float(IronMeltingCurve.t_melt_pure(200e9)), 'on the curve')
        assert "'on the curve' touches a curve" in sketch.collisions(fig, ax)
    finally:
        plt.close(fig)


def test_the_sketch_curves_are_the_code_output(sketch):
    """The plotted curves are AlloyLiquidus, the pure-iron curve and the Fe-S eutectic on the
    grid of the sketch; the example adiabat meets pure iron at the inner-core boundary and stays
    above the liquidus of 2 wt% S."""
    c, pa = sketch.curves(), sketch.P * 1e9
    np.testing.assert_array_equal(c['S2'], np.asarray(AlloyLiquidus({'S': 0.02}).t_melt(pa)))
    np.testing.assert_array_equal(c['S48'], np.asarray(AlloyLiquidus({'S': 0.048}).t_melt(pa)))
    np.testing.assert_array_equal(c['TeS'], np.asarray(eutectic('S', pa)[0]))
    np.testing.assert_array_equal(c['Fe'], np.asarray(IronMeltingCurve.t_melt_pure(pa)))
    p = sketch.icb_pressure()
    assert float(sketch.adiabat(p)) == pytest.approx(
        float(IronMeltingCurve.t_melt_pure(p * 1e9))
    )
    up = sketch.P >= sketch.P_CMB
    assert np.all(c['adiabat'][up] > c['S2'][up])


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_the_sketch_files_are_small(theme):
    path = ROOT / 'docs' / 'figures' / f'alloy_liquidus_sketch_{theme}.svg'
    assert path.stat().st_size <= 40_000
