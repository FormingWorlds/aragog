"""Staged core-evolution module.

Replaces the isothermal-reservoir core boundary condition with a core that
carries its own state: energy budget with inner-core nucleation (B1), entropy
budget and dynamo diagnostics (B2), and stable stratification with
crystallization-regime flags (B3). The module receives the CMB heat flow and
returns the CMB temperature; with every feature disabled it reproduces the
isothermal-reservoir closure, which is the regression anchor.

All radial structure uses the closed-form Gaussian profile family
(Labrosse et al. 2001; Nimmo 2015, Treatise on Geophysics 9.08), so every
budget term is an analytic integral and the whole module stays at ODE cost.
"""

from __future__ import annotations

from aragog.core.profiles import GaussianCoreProfiles

__all__ = ['GaussianCoreProfiles']
