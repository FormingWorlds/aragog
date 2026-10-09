"""Closed-form radial structure of the core.

The Gaussian profile family (Labrosse et al. 2001; Nimmo 2015, Treatise on
Geophysics 9.08): density and the adiabat are Gaussians in radius, so mass,
gravity, and every energy-budget integral downstream have closed forms or
cheap fixed-order quadratures, keeping the core module at ODE cost.

Density is ``rho(r) = rho_cen * exp(-r^2 / L^2)`` with the length scale ``L``
taken as a direct parameter. Gravity follows exactly from the enclosed mass
of that density (an erf expression, no series truncation), and pressure
integrates hydrostatic balance inward from the CMB anchor with fixed-order
Gauss-Legendre panels. The adiabat solves ``d ln T / dr = -alpha g / c_p``:
with the exact gravity it is ``T(r) = T_cmb * exp(-alpha psi(r) / c_p)``, ``psi``
the potential on the same panels (``adiabat_mode = 'exact'``); with gravity
linear in ``r`` it is the closed form ``T_cmb * exp((r_cmb^2 - r^2) / D^2)``,
``D^2 = 3 c_p / (2 pi alpha rho_cen G)`` (``'small_radius'``), the form of
Labrosse et al. (2001) and Nimmo (2015).

Everything evaluates through ``jax.numpy`` and is jit- and grad-safe; the
constructor validates its scalar parameters eagerly, outside any trace.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as _np
from jax.scipy.special import erf
from scipy import constants as sp_constants
from scipy.optimize import minimize_scalar, root_scalar

jax.config.update('jax_enable_x64', True)

G = sp_constants.G
_SQRT_PI = float(jnp.sqrt(jnp.pi))

# Nodes and weights of 32-point Gauss-Legendre on [-1, 1], generated once at
# import; the fixed order keeps the hydrostatic-pressure quadrature jit-safe
# and its error far below solver tolerances for the smooth Gaussian integrand.
_GL_X, _GL_W = _np.polynomial.legendre.leggauss(32)
_GL_X = jnp.asarray(_GL_X)
_GL_W = jnp.asarray(_GL_W)


class GaussianCoreProfiles:
    """Radial density, mass, gravity, pressure, and adiabat of the core.

    Parameters
    ----------
    rho_cen : float
        Density at the planet centre [kg m-3].
    length_scale : float
        Gaussian density length scale ``L`` [m]; sets the compressibility
        of the profile via ``rho(r_cmb) = rho_cen * exp(-r_cmb^2/L^2)``.
    r_cmb : float
        Core-mantle boundary radius [m].
    p_cmb : float
        Pressure at the CMB [Pa]; anchors the hydrostatic integration.
    alpha : float
        Thermal expansion coefficient [K-1], treated as constant.
    c_p : float
        Isobaric specific heat capacity [J kg-1 K-1], treated as constant.
    pressure_mode : str
        ``'quadrature'`` integrates hydrostatic balance against the exact
        erf gravity; ``'labrosse'`` evaluates the printed closed form
        (Labrosse et al. 2001; Nimmo 2015, ch. 9.08, Eq. 3), whose derivation uses
        the small-radius gravity expansion. The two differ by a few
        tenths of a percent in the deep core (0.3% at the Earth centre),
        enough to move a tangent melting-curve crossing by hundreds of
        kilometres when reproducing models built on the printed form.
    adiabat_mode : str
        ``'exact'`` integrates ``d ln T / dr = -alpha g / c_p`` with the exact
        erf gravity; ``'small_radius'`` takes the closed form of linear gravity,
        which raises ``T_cen / T_cmb`` by 2 % for the Earth's core and by more
        for larger cores.

    Raises
    ------
    ValueError
        If any parameter is non-positive, or ``length_scale`` does not
        exceed zero compression at the CMB (``r_cmb >= 3 L`` would put the
        whole core in the far Gaussian tail, outside the family's regime).
    """

    def __init__(
        self,
        *,
        rho_cen: float,
        length_scale: float,
        r_cmb: float,
        p_cmb: float,
        alpha: float,
        c_p: float,
        pressure_mode: str = 'quadrature',
        adiabat_mode: str = 'exact',
    ) -> None:
        if pressure_mode not in ('quadrature', 'labrosse'):
            raise ValueError(f'unknown pressure_mode {pressure_mode!r}')
        if adiabat_mode not in ('exact', 'small_radius'):
            raise ValueError(f'unknown adiabat_mode {adiabat_mode!r}')
        self.pressure_mode = pressure_mode
        self.adiabat_mode = adiabat_mode
        params = {
            'rho_cen': rho_cen,
            'length_scale': length_scale,
            'r_cmb': r_cmb,
            'p_cmb': p_cmb,
            'alpha': alpha,
            'c_p': c_p,
        }
        for name, value in params.items():
            if not float(value) > 0.0:
                raise ValueError(f'{name} must be positive, got {value}')
        if r_cmb >= 3.0 * length_scale:
            raise ValueError(
                f'r_cmb={r_cmb} lies beyond 3 length scales ({length_scale}); '
                'the Gaussian family is not a credible core profile there'
            )
        self.rho_cen = float(rho_cen)
        self.length_scale = float(length_scale)
        self.r_cmb = float(r_cmb)
        self.p_cmb = float(p_cmb)
        self.alpha = float(alpha)
        self.c_p = float(c_p)
        # Adiabatic length scale from the small-r identity
        # d ln T / dr = -alpha g / c_p with g = (4 pi / 3) G rho_cen r.
        self.d_scale = float(jnp.sqrt(3.0 * c_p / (2.0 * jnp.pi * alpha * self.rho_cen * G)))

    # -- density and mass ---------------------------------------------------

    def density(self, r):
        """Density [kg m-3] at radius ``r`` [m]."""
        r = jnp.asarray(r)
        return self.rho_cen * jnp.exp(-((r / self.length_scale) ** 2))

    def enclosed_mass(self, r):
        """Mass [kg] inside radius ``r``: the closed-form Gaussian integral.

        ``4 pi rho_cen [ (sqrt(pi) L^3 / 4) erf(r/L) - (L^2 r / 2)
        exp(-r^2/L^2) ]``, the exact antiderivative of
        ``4 pi rho(s) s^2``.
        """
        r = jnp.asarray(r)
        length = self.length_scale
        x = r / length
        integral = (_SQRT_PI * length**3 / 4.0) * erf(x) - (length**2 * r / 2.0) * jnp.exp(
            -(x**2)
        )
        return 4.0 * jnp.pi * self.rho_cen * integral

    def gravity(self, r):
        """Gravitational acceleration magnitude [m s-2] at radius ``r``.

        Exact for the Gaussian density: ``G M(r) / r^2``, with the removable
        singularity at the centre replaced by its analytic limit
        ``(4 pi / 3) G rho_cen r``.
        """
        r = jnp.asarray(r)
        small = self.length_scale * 1e-6
        safe_r = jnp.where(r > small, r, small)
        exact = G * self.enclosed_mass(safe_r) / safe_r**2
        centre = 4.0 * jnp.pi / 3.0 * G * self.rho_cen * r
        return jnp.where(r > small, exact, centre)

    # -- pressure -----------------------------------------------------------

    def _pressure_labrosse(self, r):
        """Printed closed form (Nimmo 2015, ch. 9.08, Eq. 3), anchored at the CMB.

        ``P(r) = p_cmb + (4 pi G rho_cen^2 / 3) [f(r_cmb) - f(r)]`` with
        ``f(x) = (3 x^2 / 10 - L^2 / 5) exp(-x^2/L^2)``; the exact
        antiderivative of density times the small-radius gravity expansion.
        """
        r = jnp.asarray(r)
        length2 = self.length_scale**2

        def f(x):
            return (0.3 * x**2 - 0.2 * length2) * jnp.exp(-(x**2) / length2)

        prefactor = 4.0 * jnp.pi * G * self.rho_cen**2 / 3.0
        return self.p_cmb + prefactor * (f(self.r_cmb) - f(r))

    def pressure(self, r):
        """Pressure [Pa] at radius ``r``, from hydrostatic balance.

        Quadrature mode: ``P(r) = p_cmb + int_r^{r_cmb} rho(s) g(s) ds`` on
        a fixed 32-point Gauss-Legendre panel, jit-safe with no adaptive
        control flow. Labrosse mode: the printed closed form.
        """
        if self.pressure_mode == 'labrosse':
            return self._pressure_labrosse(r)
        r = jnp.asarray(r)
        half_span = (self.r_cmb - r) / 2.0
        centre = (self.r_cmb + r) / 2.0
        # Broadcast quadrature nodes over any leading shape of r.
        s = centre[..., None] + half_span[..., None] * _GL_X
        integrand = self.density(s) * self.gravity(s)
        integral = half_span * jnp.sum(_GL_W * integrand, axis=-1)
        return self.p_cmb + integral

    def potential(self, r):
        """Gravitational potential [J/kg] at radius ``r``, zero at the CMB.

        ``psi(r) = -int_r^{r_cmb} g(s) ds`` on the same fixed 32-point
        Gauss-Legendre panel as the pressure; negative inside the core,
        which is the reference the gravitational-energy budget term uses.
        """
        r = jnp.asarray(r)
        half_span = (self.r_cmb - r) / 2.0
        centre = (self.r_cmb + r) / 2.0
        s = centre[..., None] + half_span[..., None] * _GL_X
        integral = half_span * jnp.sum(_GL_W * self.gravity(s), axis=-1)
        return -integral

    # -- adiabat ------------------------------------------------------------

    def adiabat(self, r, t_cmb):
        """Adiabatic temperature [K] at radius ``r`` anchored at ``t_cmb``; see ``adiabat_mode``.

        Hotter inward, equal to ``t_cmb`` at the CMB by construction.
        """
        r = jnp.asarray(r)
        if self.adiabat_mode == 'exact':
            return t_cmb * jnp.exp(-self.alpha * self.potential(r) / self.c_p)
        return t_cmb * jnp.exp((self.r_cmb**2 - r**2) / self.d_scale**2)

    def adiabat_gradient(self, r, t_cmb):
        """``dT/dr`` [K/m] of the adiabat, ``-alpha g T / c_p``, with the gravity of the mode."""
        r = jnp.asarray(r)
        g = (
            self.gravity(r)
            if self.adiabat_mode == 'exact'
            else 4.0 * jnp.pi / 3.0 * G * self.rho_cen * r
        )
        return -self.alpha * g * self.adiabat(r, t_cmb) / self.c_p

    @functools.cached_property
    def r_peak(self) -> float:
        """Radius [m] where the heat conducted down the adiabat, ``r^2 g T``, peaks; ``r_cmb``
        when it rises through the whole core. ``D sqrt(3/2)`` for linear gravity."""
        if self.adiabat_mode == 'small_radius':
            return float(self.d_scale * _np.sqrt(1.5))
        with jax.ensure_compile_time_eval():  # concrete even when first read inside a trace
            return self._r_peak_exact()

    def _r_peak_exact(self) -> float:
        r = _np.linspace(0.0, self.r_cmb, 513)
        i = int(_np.argmax(-(r**2) * _np.asarray(self.adiabat_gradient(r, 1.0))))
        if i == r.size - 1:
            return self.r_cmb
        res = minimize_scalar(
            lambda x: float(x**2 * self.adiabat_gradient(x, 1.0)),
            bounds=(r[i - 1], r[i + 1]),
            method='bounded',
            options={'xatol': 1.0},
        )
        return float(res.x)

    def t_cen(self, t_cmb):
        """Centre temperature [K] on the adiabat anchored at ``t_cmb``."""
        return self.adiabat(0.0, t_cmb)

    @classmethod
    def from_structure(
        cls,
        *,
        m_core: float,
        p_cen: float,
        r_cmb: float,
        p_cmb: float,
        alpha: float,
        c_p: float,
        pressure_mode: str = 'quadrature',
        adiabat_mode: str = 'exact',
    ) -> GaussianCoreProfiles:
        """Fit central density and length scale to core mass and central pressure.

        Parameters
        ----------
        m_core : float
            Total core mass [kg], positive.
        p_cen : float
            Central pressure [Pa], must exceed ``p_cmb``.
        r_cmb : float
            Core-mantle boundary radius [m], positive.
        p_cmb : float
            Pressure at the core-mantle boundary [Pa], positive.
        alpha : float
            Thermal expansion coefficient [K-1], positive.
        c_p : float
            Isobaric specific heat capacity [J kg-1 K-1], positive.
        pressure_mode : str, optional
            Pressure mode: ``'quadrature'`` (default) or ``'labrosse'``.
        adiabat_mode : str, optional
            Adiabat mode: ``'exact'`` (default) or ``'small_radius'``.

        Returns
        -------
        GaussianCoreProfiles
            Profile instance with fitted ``rho_cen`` and ``length_scale``.

        Raises
        ------
        ValueError
            If parameters are non-positive, if ``p_cen <= p_cmb``, if the
            central pressure is below the incompressible sphere limit, or if the
            root solve does not converge within the Gaussian family regime.
        """
        params = {
            'm_core': m_core,
            'p_cen': p_cen,
            'r_cmb': r_cmb,
            'p_cmb': p_cmb,
            'alpha': alpha,
            'c_p': c_p,
        }
        for name, value in params.items():
            if not float(value) > 0.0:
                raise ValueError(f'{name} must be positive, got {value}')

        m_core = float(m_core)
        p_cen = float(p_cen)
        r_cmb = float(r_cmb)
        p_cmb = float(p_cmb)
        alpha = float(alpha)
        c_p = float(c_p)

        if p_cen <= p_cmb:
            raise ValueError(f'p_cen ({p_cen:.4e} Pa) must exceed p_cmb ({p_cmb:.4e} Pa)')

        # Uniform density sphere has lowest central pressure for given mass.
        vol = 4.0 / 3.0 * _np.pi * r_cmb**3
        rho_avg = m_core / vol
        p_incomp = p_cmb + (2.0 / 3.0) * _np.pi * G * rho_avg**2 * r_cmb**2
        if p_cen < p_incomp:
            raise ValueError(
                f'p_cen ({p_cen:.4e} Pa) is below the incompressible central pressure '
                f'({p_incomp:.4e} Pa)'
            )

        def build(rho_c: float, length: float) -> GaussianCoreProfiles:
            return cls(
                rho_cen=rho_c,
                length_scale=length,
                r_cmb=r_cmb,
                p_cmb=p_cmb,
                alpha=alpha,
                c_p=c_p,
                pressure_mode=pressure_mode,
                adiabat_mode=adiabat_mode,
            )

        def rho_for(length: float) -> float:
            return float(m_core / float(build(1.0, length).enclosed_mass(r_cmb)))

        def residual(length: float) -> float:
            return float(build(rho_for(length), length).pressure(0.0)) - p_cen

        # Enforce the Gaussian family validity regime (r_cmb < 3 * length_scale).
        l_min = r_cmb / 2.999
        l_max = 100.0 * r_cmb

        res_min = residual(l_min)
        res_max = residual(l_max)
        if res_min < 0.0:
            p_max = float(p_cen + res_min)
            raise ValueError(
                f'p_cen ({p_cen:.4e} Pa) exceeds maximum central pressure ({p_max:.4e} Pa) '
                'achievable within valid Gaussian regime (r_cmb < 3 L)'
            )
        if res_min * res_max > 0.0:
            raise ValueError(
                f'Gaussian core profile fit did not bracket a root for m_core={m_core:.4e} kg, '
                f'p_cen={p_cen:.4e} Pa (residual span [{res_min:.4e}, {res_max:.4e}])'
            )

        sol = root_scalar(
            residual,
            bracket=[l_min, l_max],
            method='brentq',
            xtol=1e-8,
            rtol=1e-10,
        )
        if not sol.converged:
            raise ValueError(
                'Gaussian core profile fit to M_core and P_cen did not converge: ' + str(sol)
            )

        l_fit = float(sol.root)
        return build(rho_for(l_fit), l_fit)


fit_gaussian_core_profiles = GaussianCoreProfiles.from_structure
