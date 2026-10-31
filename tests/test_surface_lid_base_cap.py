"""The lid base at the melt temperature: the R1 column with the solid rheology and the 'lid'
closure, on 100, 200 and 400 nodes and on 100 nodes with the jump floor at 1e2, 1e3 and
1e4 J/kg, from the runs that ``tools/verification/run_surface_lid_case.py --collect`` stored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import assert_legend_clear

pytestmark = [pytest.mark.unit]

REF = Path(__file__).resolve().parent / 'reference' / 'surface_lid_base_cap.npz'
MESHES = ('n100', 'n200', 'n400')
FLOORS = (('n100_dh1e2', 1e2), ('n100', 1e3), ('n100_dh1e4', 1e4))
PAIR = 1e-3  # round-off pair of the 100-node run
STEP_3E5 = 1.2  # mesh step factor of delta at 3e5 yr: 200 to 400 below, 100 to 200 above
STEP_4E5 = 1.4  # at 4e5 yr both steps stay above: the late lid depends on the mesh
FLOOR_TOL = 4e-2  # floor runs against the 1e3 J/kg run up to 3e5 yr


def _step(a, b):
    """The factor between two thicknesses, at least 1."""
    r = a / b
    return np.maximum(r, 1.0 / r)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_lid_base_cap_on_the_mesh_and_the_floor(validation_plot):
    """Up to 3e5 yr the lid thickness converges with the mesh: the step from 200 to 400 nodes
    changes it by less than STEP_3E5 and the step from 100 to 200 nodes by more. At 4e5 yr
    both steps exceed STEP_4E5 and the 200-node lid is the thinnest, so the order is not
    monotone. Up to 3e5 yr the floor runs agree within FLOOR_TOL; at 4e5 yr the floor of
    1e4 J/kg gives a thicker lid by more than a factor 2."""
    with np.load(REF) as f:
        d = {k: f[k] for k in f.files}
    t = d['times_yr']
    i3, i4 = (int(np.flatnonzero(t == x)[0]) for x in (3e5, 4e5))
    np.testing.assert_allclose(d['n100_p_delta'], d['n100_delta'], rtol=PAIR)
    f12 = _step(d['n100_delta'], d['n200_delta'])
    f24 = _step(d['n400_delta'], d['n200_delta'])
    assert f24[i3] < STEP_3E5 < f12[i3]
    assert min(f12[i4], f24[i4]) > STEP_4E5
    assert d['n200_delta'][i4] < min(d['n100_delta'][i4], d['n400_delta'][i4])
    for tag, _ in FLOORS:
        np.testing.assert_allclose(
            d[f'{tag}_delta'][: i3 + 1], d['n100_delta'][: i3 + 1], rtol=FLOOR_TOL
        )
    assert d['n100_dh1e4_delta'][i4] > 2 * d['n100_delta'][i4]

    def figure():
        import matplotlib.pyplot as plt

        ink = plt.rcParams['text.color']
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
        panels = (
            (axes[0], MESHES, '(a) nodes\nfloor 1000 J/kg'),
            (axes[1], [m for m, _ in FLOORS], '(b) floor (J/kg)\n100 nodes'),
        )
        for ax, tags, head in panels:
            for i, tag in enumerate(tags):
                label = tag[1:] if ax is axes[0] else f'{dict(FLOORS)[tag]:.0f}'
                ax.semilogy(
                    t / 1e3,
                    d[f'{tag}_delta'],
                    ls=('-', '--', '-.')[i],
                    color=f'C{i + 1}',
                    label=label,
                )
            ax.vlines(
                [300, 400], 0, 1, transform=ax.get_xaxis_transform(), colors=ink, ls=':', lw=0.8
            )
            ax.minorticks_off()
            ax.grid(False, axis='x')
            ax.yaxis.set_major_formatter('{x:g}')
            ax.set_xticks([100, 200, 300, 400])
            ax.legend(
                title=head,
                fontsize=8,
                title_fontsize=8,
                loc='lower center',
                bbox_to_anchor=(0.5, 1.0),
                ncol=3,
                frameon=False,
            )
        axes[0].set_ylabel('lid thickness (m)')
        axes[1].tick_params(axis='y', left=False)
        fig.supxlabel('time (kyr)', fontsize=plt.rcParams['axes.labelsize'])
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_base_cap_mesh_floor')
