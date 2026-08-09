"""Core entropy budget, dynamo criterion, and field-strength scaling.

The entropy balance (Nimmo 2015, Treatise 9.08, Eq. 10) mirrors the energy
budget term for term: cooling, latent heat, and gravitational energy supply
entropy in proportion to the cooling rate, radiogenic heating supplies it
directly, and thermal conduction along the adiabat is the sink. The margin
``dE = Es + EL + Eg + ER - Ek`` is the entropy production rate available to
ohmic dissipation; a dynamo requires it positive. Term structures follow
the Leeds ``thermal_history`` implementation (Gubbins et al. 2003
formalism) evaluated on the Gaussian profiles, where the conduction sink
has the closed form ``Ek = 16 pi k r_cmb^5 / (5 D^4)`` because
``|dT_a/dr| / T_a = 2 r / D^2`` exactly.

Field strength uses the energy-flux scaling of Christensen, Holzwarth &
Reiners (2009, Nature 457, 167, Eq. 2): ``<B>^2 / (2 mu0) = c f_ohm
<rho>^{1/3} (F q_o)^{2/3}`` with ``c = 0.63``, and their Earth-core
efficiency factors ``F = 0.88 alpha g_cmb r_cmb / c_p`` (constant total
convected flux) or ``0.45 ...`` (flux vanishing at the outer boundary).
The reference flux ``q_o`` here is the superadiabatic part of the CMB heat
flow spread over the CMB area; compositional enhancement of the effective
buoyancy flux is not folded in at this stage, so subadiabatic
compositionally-driven dynamos get a conservative (low) field estimate
while the criterion itself still comes from the full entropy margin.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as _np
from scipy import constants as sp_constants

from aragog.core.budget import CoreEnergyBudget

jax.config.update('jax_enable_x64', True)

MU0 = sp_constants.mu_0
_CHR09_C = 0.63  # proportionality constant of CHR09 Eq. 2
_CHR09_F_GEOMETRY = {'const_flux': 0.88, 'zero_outer': 0.45}

_GL_X, _GL_W = _np.polynomial.legendre.leggauss(48)
_GL_X = jnp.asarray(_GL_X)
_GL_W = jnp.asarray(_GL_W)


class CoreEntropyBudget:
    """Entropy balance and dynamo diagnostics on top of the energy budget.

    Parameters
    ----------
    budget : CoreEnergyBudget
        The energy budget whose profiles, melting curve, and capacity terms
        this balance reuses; entropy efficiencies attach to its terms.
    k_core : float
        Thermal conductivity of the core [W m-1 K-1], treated as constant.
    f_ohm : float
        Fraction of dissipation that is ohmic, in (0, 1]; CHR09 adopt 1
        for planets.
    flux_geometry : str
        ``'const_flux'`` or ``'zero_outer'``: which of the two printed
        Earth-core efficiency factors converts thermodynamic properties
        into the CHR09 ``F``.

    Raises
    ------
    ValueError
        On non-positive ``k_core``, ``f_ohm`` outside (0, 1], or an
        unknown ``flux_geometry``.
    """

    def __init__(
        self,
        budget: CoreEnergyBudget,
        *,
        k_core: float,
        f_ohm: float = 1.0,
        flux_geometry: str = 'const_flux',
    ) -> None:
        if not float(k_core) > 0.0:
            raise ValueError(f'k_core must be positive, got {k_core}')
        if not 0.0 < float(f_ohm) <= 1.0:
            raise ValueError(f'f_ohm must be in (0, 1], got {f_ohm}')
        if flux_geometry not in _CHR09_F_GEOMETRY:
            raise ValueError(f'unknown flux_geometry {flux_geometry!r}')
        if (
            getattr(budget, 'stratification', False)
            and budget.k_core is not None
            and float(budget.k_core) != float(k_core)
        ):
            raise ValueError(
                'one core, one conductivity: the stratified budget carries '
                f'k_core={budget.k_core} but the entropy budget was given '
                f'{k_core}; the layer depth and the conduction sink would '
                'disagree about the same physical quantity'
            )
        self.budget = budget
        self.k_core = float(k_core)
        self.f_ohm = float(f_ohm)
        self.flux_geometry = flux_geometry

    # -- quadrature helper ---------------------------------------------------

    def _quad_0_rcmb(self, integrand):
        return self._quad_0_upper(self.budget.profiles.r_cmb, integrand)

    def _quad_0_upper(self, upper, integrand):
        half = upper / 2.0
        r = half + half * _GL_X
        return half * jnp.sum(_GL_W * integrand(r))

    def _upper(self, t_cmb, q_cmb=None):
        """Convecting-volume top for the entropy integrals.

        Defers to the energy budget, so one conductive-matching depth
        reduces both budgets (the constructor enforces one shared
        conductivity when stratification is on); the CMB radius when
        stratification is off.
        """
        return self.budget.convecting_radius(t_cmb, q_cmb)

    # -- entropy sink and sources --------------------------------------------

    def conduction_sink(self, upper=None):
        """Entropy sink of conduction along the adiabat, ``Ek`` [W/K].

        ``4 pi k int (dTa/dr / Ta)^2 r^2 dr`` with the Gaussian adiabat's
        exact ratio ``2 r / D^2``: ``Ek = 16 pi k upper^5 / (5 D^4)``,
        over the convecting volume (``upper`` defaults to the CMB
        radius). Independent of ``T_cmb``.
        """
        p = self.budget.profiles
        top = p.r_cmb if upper is None else upper
        return 16.0 * jnp.pi * self.k_core * top**5 / (5.0 * p.d_scale**4)

    def secular_entropy_capacity(self, t_cmb, upper=None):
        """Entropy per unit cooling from secular cooling [J/K^2].

        ``int rho c_p (Ta/T_cmb - 1) dV / T_cmb``: heat extracted at
        temperature ``Ta`` and delivered at ``T_cmb`` produces entropy in
        proportion to the temperature drop. Integrated over the
        convecting volume (``upper`` defaults to the CMB radius).
        """
        p = self.budget.profiles

        def integrand(r):
            shape = p.adiabat(r, 1.0)
            return p.density(r) * p.c_p * (shape - 1.0) * 4.0 * jnp.pi * r**2

        top = p.r_cmb if upper is None else upper
        return self._quad_0_upper(top, integrand) / t_cmb

    def latent_entropy_capacity(self, t_cmb):
        """Entropy per unit cooling from latent heat [J/K^2].

        The latent capacity released at ``T_icb`` and delivered at
        ``T_cmb``: ``C_lat (T_icb - T_cmb) / (T_icb T_cmb)``.
        """
        b = self.budget
        radius = b.r_icb(t_cmb)
        t_icb = b.profiles.adiabat(radius, t_cmb)
        return b.latent_capacity(t_cmb) * (t_icb - t_cmb) / (t_icb * t_cmb)

    def gravitational_entropy_capacity(self, t_cmb, upper=None):
        """Entropy per unit cooling from gravitational energy [J/K^2].

        Gravitational energy dissipates in full within the convecting
        core: ``C_grav / T_cmb``.
        """
        return self.budget.gravitational_capacity(t_cmb, upper=upper) / t_cmb

    def radiogenic_entropy(self, t_cmb, q_radio, upper=None):
        """Entropy rate from internal heating [W/K].

        ``int h rho (1/T_cmb - 1/Ta) dV`` with the heating rate per unit
        mass ``h = q_radio / M_core`` uniform, integrated over the
        convecting volume (``upper`` defaults to the CMB radius).
        """
        p = self.budget.profiles
        mass = p.enclosed_mass(p.r_cmb)

        def integrand(r):
            inv_gap = 1.0 / t_cmb - 1.0 / p.adiabat(r, t_cmb)
            return p.density(r) * inv_gap * 4.0 * jnp.pi * r**2

        top = p.r_cmb if upper is None else upper
        return (q_radio / mass) * self._quad_0_upper(top, integrand)

    # -- dynamo criterion ----------------------------------------------------

    def entropy_margin(self, t_cmb, q_cmb, q_radio=0.0):
        """Entropy production available to the dynamo, ``dE`` [W/K].

        The cooling rate follows from the energy budget for the given heat
        flow, the three cooling-proportional sources scale with it, and
        conduction subtracts. Positive margin sustains a dynamo. With
        stratification enabled on the energy budget, every volume term
        runs over the convecting region for this heat flow, so the layer
        shrinks the sources and the sink together.
        """
        cooling = -self.budget.dtcmb_dt(t_cmb, q_cmb, q_sources=q_radio)
        upper = self._upper(t_cmb, q_cmb)
        capacity = (
            self.secular_entropy_capacity(t_cmb, upper=upper)
            + self.latent_entropy_capacity(t_cmb)
            + self.gravitational_entropy_capacity(t_cmb, upper=upper)
        )
        return (
            capacity * cooling
            + self.radiogenic_entropy(t_cmb, q_radio, upper=upper)
            - self.conduction_sink(upper=upper)
        )

    # -- field strength ------------------------------------------------------

    def adiabatic_heat_flow(self, t_cmb):
        """Heat conducted down the adiabat at the CMB, ``Qk`` [W]."""
        p = self.budget.profiles
        grad = 2.0 * p.r_cmb * t_cmb / p.d_scale**2
        return 4.0 * jnp.pi * p.r_cmb**2 * self.k_core * grad

    def chr09_efficiency_factor(self):
        """The printed Earth-core ``F``: ``0.88 (or 0.45) alpha g r / c_p``."""
        p = self.budget.profiles
        g_cmb = p.gravity(p.r_cmb)
        return _CHR09_F_GEOMETRY[self.flux_geometry] * p.alpha * g_cmb * p.r_cmb / p.c_p

    def b_rms_core(self, t_cmb, q_cmb):
        """Volume-averaged core field strength [T] from CHR09 Eq. 2.

        The reference flux is the superadiabatic part of the CMB heat flow
        over the CMB area; with nothing superadiabatic the field estimate
        is zero even where compositional convection keeps the entropy
        margin positive (a conservative floor, see the module docstring).
        """
        p = self.budget.profiles
        area = 4.0 * jnp.pi * p.r_cmb**2
        q_o = jnp.maximum(q_cmb - self.adiabatic_heat_flow(t_cmb), 0.0) / area
        mean_rho = p.enclosed_mass(p.r_cmb) / (4.0 / 3.0 * jnp.pi * p.r_cmb**3)
        energy_density = (
            _CHR09_C
            * self.f_ohm
            * mean_rho ** (1.0 / 3.0)
            * (self.chr09_efficiency_factor() * q_o) ** (2.0 / 3.0)
        )
        return jnp.sqrt(2.0 * MU0 * energy_density)

    def b_dipole_cmb(self, t_cmb, q_cmb, dipolarity: float = 1.0):
        """Dipole field at the CMB [T]: the core rms field times a
        dipolarity fraction; a potential-field continuation to the planet
        surface is the caller's ``(r_cmb / r_planet)^3`` factor."""
        return dipolarity * self.b_rms_core(t_cmb, q_cmb)
