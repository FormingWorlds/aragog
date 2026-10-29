"""Validation of the refined mesh on abe_solid: the end cells hold their set thickness.

Module under test: ``aragog.mesh.Mesh`` with ``mesh.surface_cell_thickness`` and
``mesh.cmb_cell_thickness``. Invariants: on the 1000 km, 100-node mantle of abe_solid the
radii increase strictly, the end radii are those of the uniform mesh, each set end cell
equals its target to 1e-6 relative in radius mode and 2e-3 m in mass-coordinate mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import aragog
from aragog.mesh import Mesh
from aragog.parser import Parameters

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

CFG = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'
CASES = {
    'uniform': (0.0, 0.0),
    'top 200 m': (200.0, 0.0),
    'top 50 m': (50.0, 0.0),
    'top 200 m + CMB 2 km': (200.0, 2000.0),
}


def _radii(top, bottom, mass_coordinates):
    p = Parameters.from_file(CFG)
    p.mesh.mass_coordinates = mass_coordinates
    p.mesh.surface_cell_thickness, p.mesh.cmb_cell_thickness = top, bottom
    return np.ravel(Mesh(p).basic.radii)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_refined_mesh_has_the_set_end_cells(validation_plot):
    """A 200 m and a 50 m top cell, and a 200 m top cell with a 2 km CMB cell, on the
    uniform 10.1 km grid of abe_solid: in both mesh modes the radii increase, the end radii
    stay those of the uniform mesh and each set end cell equals its target, to 1e-6
    relative in radius mode and 2e-3 m in mass-coordinate mode (the 1e-3 m root tolerance
    of the 2 nodes)."""
    cells = {}
    for mass_coordinates, rel, tol in ((False, 1e-6, 0.0), (True, 0.0, 2e-3)):
        base = _radii(0.0, 0.0, mass_coordinates)
        for label, (top, bottom) in CASES.items():
            r = _radii(top, bottom, mass_coordinates)
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

        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for label, (depth, d) in cells.items():
            (line,) = ax.plot(depth / 1e3, d, lw=1.2, label=label)
            ax.plot(depth[[0, -1]] / 1e3, d[[0, -1]], 'o', ms=3, color=line.get_color())
        ax.set(xlabel='depth [km]', ylabel='cell [m]', yscale='log')
        ax.minorticks_off()
        ax.set_yticks([1e1, 1e2, 1e3, 1e4])
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'mesh_refinement_end_cells')
