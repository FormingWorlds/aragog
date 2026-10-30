"""Validation of the refined mesh on abe_solid: the end cells hold their set thickness.

Module under test: ``aragog.mesh.Mesh`` with ``mesh.surface_cell_thickness`` and
``mesh.cmb_cell_thickness``. Invariants: on the 1000 km, 100-node mantle of abe_solid the
radii increase strictly, the end radii are those of the uniform mesh, each set end cell
equals its target to 1e-6 relative in radius mode and 2e-3 m in mass-coordinate mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_mesh_refinement import _mesh, _radii
from tests.test_mesh_refinement_study import FIGSIZE, _check_layout

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

CASES = {
    'uniform': (0.0, 0.0),
    'top 200 m': (200.0, 0.0),
    'top 50 m': (50.0, 0.0),
    'top 200 m + CMB 2 km': (200.0, 2000.0),
}


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_refined_mesh_has_the_set_end_cells(validation_plot):
    """A 200 m and a 50 m top cell, and a 200 m top cell with a 2 km CMB cell, on the
    abe_solid grid (uniform cell 10.1 km in radius mode): in both mesh modes the radii
    increase, the end radii stay those of the uniform mesh and each set end cell equals its
    target, to 1e-6 relative in radius mode and 2e-3 m in mass-coordinate mode, twice the
    1e-3 m root tolerance of the one solved node of an end cell."""
    cells = {}
    for mass_coordinates, rel, tol in ((False, 1e-6, 0.0), (True, 0.0, 2e-3)):
        base = _radii(_mesh(mass_coordinates=mass_coordinates))
        for label, (top, bottom) in CASES.items():
            r = _radii(_mesh(top, bottom, mass_coordinates))
            d = np.diff(r)
            assert np.all(d > 0.0), label
            assert (r[0], r[-1]) == (base[0], base[-1]), label
            for target, cell in ((top, d[-1]), (bottom, d[0])):
                if target:
                    assert cell == pytest.approx(target, rel=rel, abs=tol), (label, target)
            if mass_coordinates and label != 'top 200 m':
                cells[label] = (base[-1] - 0.5 * (r[1:] + r[:-1]), d)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
        fig.get_layout_engine().set(h_pad=0.05)
        for label, (depth, d) in cells.items():
            (line,) = ax.plot(depth / 1e3, d, lw=1.2, label=label)
            ax.plot(depth[[0, -1]] / 1e3, d[[0, -1]], 'o', ms=3, color=line.get_color())
        ax.set(xlabel='depth [km]', ylabel='cell [m]', yscale='log')
        ax.minorticks_off()
        ax.set_yticks([1e1, 1e2, 1e3, 1e4])
        ax.set_xticks([0, 250, 500, 750, 1000])
        fig.legend(loc='outside lower center', ncols=1)
        return fig

    _check_layout(figure)
    validation_plot(figure, 'mesh_refinement_end_cells')


def _cells(top=0.0, bottom=0.0, mass_coordinates=True, n=None):
    return np.diff(_radii(_mesh(top, bottom, mass_coordinates, n)))


def _growth(d):
    """Largest ratio of the larger to the smaller of two neighbouring cells, minus 1."""
    return float(np.max(np.maximum(d[1:] / d[:-1], d[:-1] / d[1:]))) - 1


def test_the_mesh_numbers_the_docs_quote():
    """The cell sizes of abe_solid that the Explanation, How-to and Validation pages quote, and
    the largest cell of each mesh of the convergence study and of the CMB case."""
    km = lambda x: f'{x / 1e3:.1f}'  # noqa: E731
    assert km(_cells(mass_coordinates=False)[0]) == '10.1'
    u = _cells()
    assert [km(u[0]), km(u[-1]), km(u.min()), km(u.max())] == ['9.3', '10.9', '9.3', '10.9']
    for mass, cmb in ((False, ['23.4', '33.0', '41.1']), (True, ['22.1', '31.0', '38.5'])):
        assert [km(_cells(t, 0.0, mass)[0]) for t in (1e3, 200.0, 50.0)] == cmb
        two = _cells(1e3, 2e3, mass)
        assert km(two.max()) == '21.5' and f'{_growth(two):.1%}' == '8.4%'
    assert f'{_growth(_cells(1e3, 0.0, False)):.1%}' == '4.6%'
    top50 = _cells(50.0)
    assert km(top50.max()) == '38.6' and int(top50.argmax()) == 2
    both = _cells(200.0, 2e3)
    assert f'{both.max():.0f}' == '26584' and 30 < int(both.argmax()) < 70
    for n, cmb, largest in (
        (100, 9327.0, ['23.6', '25.8', '28.0']),
        (200, 4638.0, ['10.6', '11.7', '12.8']),
    ):
        assert [km(_cells(t, cmb, True, n).max()) for t in (200.0, 100.0, 50.0)] == largest
