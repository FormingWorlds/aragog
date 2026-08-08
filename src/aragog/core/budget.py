"""Core energy balance: secular cooling and inner-core latent heat.

Given the CMB heat flow, the budget returns the CMB cooling rate through an
effective heat capacity: ``Q_cmb = -C_eff(T_cmb) dT_cmb/dt + Q_sources``.
``C_eff`` carries the secular term (the mass-weighted adiabat integral over
the Gaussian profiles) plus, once the centre adiabat reaches the melting
curve, the latent heat and light-element gravitational energy of inner-core
growth. Nucleation activates through a sigmoid of the centre superheat and
winds down through a matching sigmoid of the CMB superheat at freeze-out
completion, so the budget is differentiable through both ends of inner-core
growth; zero width recovers the hard switches. Radiogenic or tidal powers
enter as per-call source terms supplied by the caller.

The ``legacy`` capacity mode reproduces the isothermal-reservoir closure
(Bower et al. 2018, Eq. 37 constants: uniform core density and a fixed
core-temperature factor), which is the regression anchor the module is
cross-checked against with every feature off.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as _np

from aragog.core.melting import IronMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles

jax.config.update('jax_enable_x64', True)

_GL_X, _GL_W = _np.polynomial.legendre.leggauss(48)
_GL_X = jnp.asarray(_GL_X)
_GL_W = jnp.asarray(_GL_W)

_BISECT_ITERS = 80  # halves the bracket to ~1e-24 of r_cmb: machine precision


class CoreEnergyBudget:
    """Energy budget of a well-mixed core with smoothed inner-core nucleation.

    Parameters
    ----------
    profiles : GaussianCoreProfiles
        Radial structure the budget integrates over.
    melting_curve : IronMeltingCurve
        Alloy melting curve; its light-element fraction is the alloy state.
    ds_fusion : float
        Entropy of fusion at the inner-core boundary [J kg-1 K-1]; sets the
        latent heat per unit mass as ``T_icb * ds_fusion``.
    icn_width : float
        Temperature width [K] of the nucleation sigmoid. Positive; the
        hard-switch limit is recovered as the width goes to zero.
    latent_heat : float, optional
        Constant latent heat of fusion [J/kg]. When given it replaces the
        ``T_icb * ds_fusion`` form; parameterised models in the literature
        (Nimmo 2015 Table 2 uses 750 kJ/kg) prescribe the constant.
    alpha_c : float, optional
        Compositional expansivity of the outer-core alloy (fractional
        density decrease per unit light-element mass fraction), for the
        gravitational-energy term. Zero disables the term.
    c_light : float, optional
        Light-element mass fraction of the outer core, rejected entirely
        by the solid on freezing (the complete-rejection limit; the
        volatile-dissolution stage refines the partitioning). Zero
        disables the gravitational term.
    capacity_mode : str
        ``'profile'`` integrates the secular capacity over the Gaussian
        profiles; ``'legacy'`` uses the isothermal-reservoir constants
        below and disables nucleation, reproducing the boundary-condition
        closure this module replaces.
    legacy_rho_core : float, optional
        Uniform core density [kg m-3] of the legacy closure (required in
        legacy mode).
    legacy_tfac : float, optional
        Core-temperature factor of the legacy closure (required in legacy
        mode; 1.147 is the Earth-like default used across the ecosystem).

    Raises
    ------
    ValueError
        On a non-positive ``ds_fusion`` or ``icn_width``, an unknown
        ``capacity_mode``, or missing legacy constants in legacy mode.
    """

    def __init__(
        self,
        profiles: GaussianCoreProfiles,
        melting_curve: IronMeltingCurve,
        *,
        ds_fusion: float,
        icn_width: float,
        latent_heat: float | None = None,
        alpha_c: float = 0.0,
        c_light: float = 0.0,
        capacity_mode: str = 'profile',
        legacy_rho_core: float | None = None,
        legacy_tfac: float | None = None,
    ) -> None:
        if not float(ds_fusion) > 0.0:
            raise ValueError(f'ds_fusion must be positive, got {ds_fusion}')
        if not float(icn_width) > 0.0:
            raise ValueError(f'icn_width must be positive, got {icn_width}')
        if latent_heat is not None and not float(latent_heat) > 0.0:
            raise ValueError(f'latent_heat must be positive, got {latent_heat}')
        if float(alpha_c) < 0.0 or float(c_light) < 0.0:
            raise ValueError('alpha_c and c_light must be non-negative')
        if capacity_mode not in ('profile', 'legacy'):
            raise ValueError(f'unknown capacity_mode {capacity_mode!r}')
        if capacity_mode == 'legacy' and (legacy_rho_core is None or legacy_tfac is None):
            raise ValueError('legacy mode needs legacy_rho_core and legacy_tfac')
        self.profiles = profiles
        self.melting_curve = melting_curve
        self.ds_fusion = float(ds_fusion)
        self.icn_width = float(icn_width)
        self.latent_heat = None if latent_heat is None else float(latent_heat)
        self.alpha_c = float(alpha_c)
        self.c_light = float(c_light)
        self.capacity_mode = capacity_mode
        self.legacy_rho_core = None if legacy_rho_core is None else float(legacy_rho_core)
        self.legacy_tfac = None if legacy_tfac is None else float(legacy_tfac)

    # -- static integrals ----------------------------------------------------

    def _quad_0_rcmb(self, integrand):
        """Fixed 48-point Gauss-Legendre integral of ``integrand(r)`` on [0, r_cmb]."""
        half = self.profiles.r_cmb / 2.0
        r = half + half * _GL_X
        return half * jnp.sum(_GL_W * integrand(r))

    def secular_capacity(self):
        """Secular heat capacity dQ_s / d(dT_cmb/dt) [J/K].

        Profile mode: ``c_p int rho(r) f(r) 4 pi r^2 dr`` with ``f`` the
        adiabat shape ``T_a(r)/T_cmb``, i.e. ``c_p M tfac_eff``. Legacy
        mode: ``(4/3) pi r_cmb^3 rho c_p tfac``, the reservoir constant.
        """
        p = self.profiles
        if self.capacity_mode == 'legacy':
            volume = 4.0 / 3.0 * jnp.pi * p.r_cmb**3
            return volume * self.legacy_rho_core * p.c_p * self.legacy_tfac

        def integrand(r):
            shape = p.adiabat(r, 1.0)  # T_a / T_cmb: anchor-independent
            return p.density(r) * shape * 4.0 * jnp.pi * r**2

        return p.c_p * self._quad_0_rcmb(integrand)

    def effective_tfac(self):
        """Mass-weighted mean of ``T_a/T_cmb`` over the core (profile mode)."""
        p = self.profiles

        def mass_integrand(r):
            return p.density(r) * 4.0 * jnp.pi * r**2

        mass = self._quad_0_rcmb(mass_integrand)
        return self.secular_capacity() / (p.c_p * mass)

    # -- inner core ----------------------------------------------------------

    def _superheat(self, r, t_cmb):
        """Adiabat minus melting curve [K] at radius ``r``; positive = liquid."""
        p = self.profiles
        return p.adiabat(r, t_cmb) - self.melting_curve.t_melt(p.pressure(r))

    def _r_icb_bisect(self, t_cmb):
        """Primal bisection for the boundary radius; see :meth:`r_icb`."""
        p = self.profiles

        def body(_, bracket):
            lo, hi = bracket
            mid = (lo + hi) / 2.0
            frozen = self._superheat(mid, t_cmb) < 0.0
            return jnp.where(frozen, mid, lo), jnp.where(frozen, hi, mid)

        lo, hi = jax.lax.fori_loop(
            0,
            _BISECT_ITERS,
            body,
            (jnp.zeros_like(t_cmb * 1.0), jnp.full_like(t_cmb * 1.0, p.r_cmb)),
        )
        root = (lo + hi) / 2.0
        # All-liquid guard: with positive centre superheat there is no root.
        return jnp.where(self._superheat(0.0, t_cmb) > 0.0, 0.0, root)

    def r_icb(self, t_cmb):
        """Inner-core boundary radius [m] at ``t_cmb``.

        Fixed-iteration bisection of the superheat on [0, r_cmb]; with no
        crossing it converges to 0 (fully liquid) or r_cmb (fully frozen),
        so the value is always defined and trace-safe. A liquid centre
        returns zero regardless of frozen shells above it (the top-down
        and snow topologies have no inner core in the bottom-up sense);
        ``crystallization_regime`` is the diagnostic for those states, and
        the smoothed activation factors zero the boundary terms there.

        The derivative is the implicit-function sensitivity attached as a
        custom JVP: a comparison-driven bisection carries no gradient of
        its own (autodiff sees only the converged constant trace), so
        without this rule anything differentiating through the boundary
        would silently read zero.
        """
        return self._r_icb_root(t_cmb)

    @property
    def _r_icb_root(self):
        """The custom-JVP-wrapped boundary solve, built once per instance."""
        cached = getattr(self, '_r_icb_root_cached', None)
        if cached is not None:
            return cached

        @jax.custom_jvp
        def root(t_cmb):
            return self._r_icb_bisect(t_cmb)

        @root.defjvp
        def root_jvp(primals, tangents):
            (t_cmb,) = primals
            (t_dot,) = tangents
            radius = self._r_icb_bisect(t_cmb)
            d_dr = jax.grad(self._superheat, argnums=0)(radius, t_cmb)
            d_dt = jax.grad(self._superheat, argnums=1)(radius, t_cmb)
            safe = jnp.where(jnp.abs(d_dr) > 0.0, d_dr, 1.0)
            # Interior boundary: dr/dT from the implicit function theorem;
            # pinned at the domain ends where the root does not move.
            interior = (radius > 0.0) & (radius < self.profiles.r_cmb)
            drdt = jnp.where(interior, -d_dt / safe, 0.0)
            return radius, drdt * t_dot

        self._r_icb_root_cached = root
        return root

    def nucleation_factor(self, t_cmb):
        """Smoothed activation in [0, 1]: sigmoid of centre subcooling."""
        return jax.nn.sigmoid(-self._superheat(0.0, t_cmb) / self.icn_width)

    def _boundary_sensitivity(self, t_cmb):
        """|dr_icb/dT_cmb| [m/K] from the implicit-function theorem.

        Written as the explicit ratio of superheat partials at the boundary
        rather than ``|grad(r_icb)|``: the same first-order value, but a
        smooth composition that higher-order autodiff (the capacity
        gradient the solver Jacobian needs) differentiates correctly,
        where nesting through the custom-JVP rule would not.
        """
        radius = self.r_icb(t_cmb)
        d_dr = jax.grad(self._superheat, argnums=0)(radius, t_cmb)
        d_dt = jax.grad(self._superheat, argnums=1)(radius, t_cmb)
        safe = jnp.where(jnp.abs(d_dr) > 0.0, d_dr, 1.0)
        return jnp.abs(-d_dt / safe)

    def freeze_out_factor(self, t_cmb):
        """Smoothed survival of the liquid outer core, in [0, 1].

        Sigmoid of the CMB superheat over the nucleation width: one while
        liquid remains at the CMB, falling to zero as the last liquid
        freezes, so the boundary terms wind down smoothly instead of
        stepping. The same width parameter governs onset and completion.
        """
        return jax.nn.sigmoid(self._superheat(self.profiles.r_cmb, t_cmb) / self.icn_width)

    def _liquid_remains(self, t_cmb):
        """False once even the CMB sits below the melting curve."""
        return self._superheat(self.profiles.r_cmb, t_cmb) > 0.0

    def latent_capacity(self, t_cmb):
        """Latent contribution to the effective heat capacity [J/K].

        ``L rho(r_icb) 4 pi r_icb^2 |dr_icb/dT_cmb|`` with the latent heat
        per unit mass ``L`` either the prescribed constant or
        ``T_icb * ds_fusion``, scaled by the nucleation factor and shut off
        at freeze-out completion when no liquid remains.
        """
        p = self.profiles
        radius = self.r_icb(t_cmb)
        if self.latent_heat is not None:
            heat = self.latent_heat
        else:
            heat = p.adiabat(radius, t_cmb) * self.ds_fusion
        area_mass = p.density(radius) * 4.0 * jnp.pi * radius**2
        capacity = heat * area_mass * self._boundary_sensitivity(t_cmb)
        return self.freeze_out_factor(t_cmb) * self.nucleation_factor(t_cmb) * capacity

    def gravitational_capacity(self, t_cmb):
        """Gravitational contribution to the effective heat capacity [J/K].

        Light elements rejected by the growing inner core mix through the
        outer core and release gravitational energy. Per unit cooling:
        ``[int_oc rho psi dV - M_oc psi(r_icb)] * alpha_c * Cc *
        |dr_icb/dT_cmb|`` with ``Cc = 4 pi r_icb^2 rho(r_icb) c_light /
        M_oc`` the enrichment per unit boundary advance in the
        complete-rejection limit; the structure of the Leeds
        ``thermal_history`` core model (Greenwood et al. 2021 lineage,
        Gubbins et al. 2003 formalism). Zero when either compositional
        parameter is zero, before onset, and after freeze-out.
        """
        p = self.profiles
        radius = self.r_icb(t_cmb)

        def integrand(r):
            return p.density(r) * p.potential(r) * 4.0 * jnp.pi * r**2

        # Outer-core integrals on [r_icb, r_cmb] with the shared GL panel.
        half = (p.r_cmb - radius) / 2.0
        centre = (p.r_cmb + radius) / 2.0
        s = centre[..., None] + half[..., None] * _GL_X
        rho_psi = half * jnp.sum(_GL_W * integrand(s), axis=-1)
        mass_oc = half * jnp.sum(_GL_W * p.density(s) * 4.0 * jnp.pi * s**2, axis=-1)
        potential_moment = rho_psi - mass_oc * p.potential(radius)

        enrichment = 4.0 * jnp.pi * radius**2 * p.density(radius) * self.c_light
        safe_mass = jnp.where(mass_oc > 0.0, mass_oc, 1.0)
        capacity = (
            potential_moment
            * self.alpha_c
            * (enrichment / safe_mass)
            * self._boundary_sensitivity(t_cmb)
        )
        return self.freeze_out_factor(t_cmb) * self.nucleation_factor(t_cmb) * capacity

    # -- assembled budget ----------------------------------------------------

    def effective_capacity(self, t_cmb):
        """Total dQ/d(dT_cmb/dt) [J/K]: secular plus latent plus
        gravitational (profile mode)."""
        secular = self.secular_capacity()
        if self.capacity_mode == 'legacy':
            return secular
        return secular + self.latent_capacity(t_cmb) + self.gravitational_capacity(t_cmb)

    def dtcmb_dt(self, t_cmb, q_cmb, q_sources=0.0):
        """CMB cooling rate [K/s] for heat flow ``q_cmb`` [W] out of the core.

        ``dT_cmb/dt = (q_sources - q_cmb) / C_eff(T_cmb)``; positive
        ``q_cmb`` cools the core, and internal sources (radiogenic, tidal)
        offset it.
        """
        return (q_sources - q_cmb) / self.effective_capacity(t_cmb)
