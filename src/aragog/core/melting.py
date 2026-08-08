"""Iron melting curve with light-element depression.

The pure-iron curve is the PALEOS prescription (``paleos/iron_eos.py``,
``T_melt_Fe`` at 66ac273 / v1.0.0), the two-branch Simon-Glatzel fit of
Anzellini et al. (2013, Science 340, 464): anchored at (5.2 GPa, 1991 K),
switching branches at the gamma-epsilon-liquid triple point (98.5 GPa,
3712 K). Sharing this prescription with the structure side keeps one iron
thermodynamics across the stack. The piecewise fit carries a ~0.7 K jump at
the branch switch, reproduced verbatim here so the module and PALEOS agree
bitwise in each branch.

Light elements depress the melting point multiplicatively,
``T_m(P, x) = T_m_Fe(P) * (1 - depression * x)``, with the mole fraction
``x`` and the depression coefficient held by the instance; ``x = 0``
recovers pure iron exactly, which is the features-off limit the module's
regression anchor relies on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

# Anzellini et al. (2013) fit constants as adopted by PALEOS T_melt_Fe.
_P0 = 5.2e9  # reference point [Pa]
_T0 = 1991.0  # reference temperature [K]
_PT = 98.5e9  # gamma-epsilon-liquid triple point [Pa]
_TT = 3712.0  # triple-point temperature [K]
_DP_LOW = 27.39  # low-branch pressure scale [GPa]
_DP_HIGH = 161.2  # high-branch pressure scale [GPa]
_EXP_LOW = 1.0 / 2.38
_EXP_HIGH = 1.0 / 1.72


class IronMeltingCurve:
    """Melting temperature of the core alloy.

    Parameters
    ----------
    light_element_fraction : float
        Mole fraction ``x`` of light elements in the core alloy, in
        ``[0, 1)``. Zero is pure iron.
    depression : float
        Melting-point depression per unit mole fraction (the ``alpha_c`` of
        the linear alloy correction), non-negative. The product
        ``depression * x`` must stay below one so the curve stays positive.

    Raises
    ------
    ValueError
        If ``x`` is outside ``[0, 1)``, ``depression`` is negative, or the
        combined depression factor would reach zero or below.
    """

    def __init__(self, *, light_element_fraction: float = 0.0, depression: float = 0.0):
        x = float(light_element_fraction)
        dep = float(depression)
        if not 0.0 <= x < 1.0:
            raise ValueError(f'light_element_fraction must be in [0, 1), got {x}')
        if dep < 0.0:
            raise ValueError(f'depression must be non-negative, got {dep}')
        if dep * x >= 1.0:
            raise ValueError(
                f'depression * light_element_fraction = {dep * x} reaches 1; '
                'the depressed melting curve would not stay positive'
            )
        self.light_element_fraction = x
        self.depression = dep

    @staticmethod
    def t_melt_pure(pressure):
        """Pure-iron melting temperature [K] at ``pressure`` [Pa]."""
        p_gpa = jnp.asarray(pressure) / 1e9
        low = _T0 * ((p_gpa - _P0 / 1e9) / _DP_LOW + 1.0) ** _EXP_LOW
        high = _TT * ((p_gpa - _PT / 1e9) / _DP_HIGH + 1.0) ** _EXP_HIGH
        return jnp.where(jnp.asarray(pressure) < _PT, low, high)

    def t_melt(self, pressure, light_element_fraction=None):
        """Alloy melting temperature [K] at ``pressure`` [Pa].

        ``light_element_fraction`` overrides the instance value when given,
        which is the hook the volatile-dissolution stage uses to evolve the
        alloy without rebuilding the curve.
        """
        x = (
            self.light_element_fraction
            if light_element_fraction is None
            else light_element_fraction
        )
        return self.t_melt_pure(pressure) * (1.0 - self.depression * x)
