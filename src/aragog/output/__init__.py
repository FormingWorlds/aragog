"""Output subpackage: diagnostics for model results.

Hosts standalone diagnostic functions that operate on a
``SolverOutput`` or its constituent arrays. The primary output
contract for callers is the ``SolverOutput`` dataclass returned by
``EntropySolver.get_state()``; everything in this subpackage is
auxiliary.
"""

from __future__ import annotations

from aragog.output.diagnostics import (
    melt_fraction_global,
    rheological_front,
    total_enthalpy,
    volume_average,
)

__all__ = [
    'melt_fraction_global',
    'rheological_front',
    'total_enthalpy',
    'volume_average',
]
