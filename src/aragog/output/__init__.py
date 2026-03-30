"""Output subpackage: diagnostics for model results.

The T-based Output class has been removed. Diagnostic functions
(melt_fraction_global, rheological_front) remain available for
use by the entropy solver and PROTEUS wrapper.
"""

from __future__ import annotations

from aragog.output.diagnostics import (
    melt_fraction_global,
    rheological_front,
)

__all__ = [
    "melt_fraction_global",
    "rheological_front",
]
