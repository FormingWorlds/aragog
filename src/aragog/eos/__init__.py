"""EOS subpackage: equation of state evaluators.

Provides entropy-formulation EOS (PALEOS P-S tables) and phase evaluator.
"""

from __future__ import annotations

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator

__all__ = [
    'EntropyEOS',
    'EntropyPhaseEvaluator',
]
