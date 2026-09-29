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
        alloy without rebuilding the curve. A concrete override is held to
        the constructor's contract (the depressed curve must stay
        positive); a traced override is the caller's responsibility, since
        a trace cannot raise on data.

        Raises
        ------
        ValueError
            If a concrete override is outside ``[0, 1)`` or drives the
            depression factor to zero or below.
        """
        if light_element_fraction is None:
            x = self.light_element_fraction
        else:
            x = light_element_fraction
            if isinstance(x, (int, float)):
                if not 0.0 <= float(x) < 1.0:
                    raise ValueError(f'light_element_fraction must be in [0, 1), got {x}')
                if self.depression * float(x) >= 1.0:
                    raise ValueError(
                        f'depression * light_element_fraction = {self.depression * float(x)} '
                        'reaches 1; the depressed melting curve would not stay positive'
                    )
        return self.t_melt_pure(pressure) * (1.0 - self.depression * x)


class QuadraticMeltingCurve:
    """Quadratic-in-pressure melting curve, ``T_m0 (1 + T_m1 P + T_m2 P^2)``.

    The parameterisation of Nimmo (2015, Treatise on Geophysics 9.08,
    Eq. 6), where ``T_m0`` already incorporates the light-element
    depression; it is the form the Table 2 benchmark models are defined in,
    and a configurable alternative to the PALEOS prescription wherever a
    fitted curve is preferred. Valid over core pressures only; ``T_m0`` is
    not a zero-pressure melting temperature.

    Parameters
    ----------
    t_m0 : float
        Prefactor [K], positive.
    t_m1 : float
        Linear coefficient [Pa-1]; either sign occurs in published fits.
    t_m2 : float
        Quadratic coefficient [Pa-2]; either sign occurs in published fits.

    Notes
    -----
    The fit is only meaningful over the pressure range it was constructed
    for; outside it a negative-coefficient curve can fall through zero.
    Callers own the range check, and the budget's freeze-out and
    nucleation factors bound the physical consequences of an out-of-range
    evaluation.
    """

    def __init__(self, *, t_m0: float, t_m1: float, t_m2: float):
        if not float(t_m0) > 0.0:
            raise ValueError(f't_m0 must be positive, got {t_m0}')
        self.t_m0 = float(t_m0)
        self.t_m1 = float(t_m1)
        self.t_m2 = float(t_m2)

    def t_melt(self, pressure, light_element_fraction=None):
        """Melting temperature [K] at ``pressure`` [Pa].

        The unused ``light_element_fraction`` keeps the call signature
        interchangeable with :class:`IronMeltingCurve`; the depression is
        already folded into ``t_m0`` in this parameterisation.
        """
        p = jnp.asarray(pressure)
        return self.t_m0 * (1.0 + self.t_m1 * p + self.t_m2 * p**2)

    def gradient(self, pressure):
        """Melting-curve slope dT_m/dP [K/Pa] at ``pressure`` [Pa]."""
        p = jnp.asarray(pressure)
        return self.t_m0 * (self.t_m1 + 2.0 * self.t_m2 * p)
