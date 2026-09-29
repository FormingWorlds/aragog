"""Closed-form radial structure of the core.

The Gaussian profile family (Labrosse et al. 2001; Nimmo 2015, Treatise on
Geophysics 9.08): density and the adiabat are Gaussians in radius, so mass,
gravity, and every energy-budget integral downstream have closed forms or
cheap fixed-order quadratures, keeping the core module at ODE cost.

Density is ``rho(r) = rho_cen * exp(-r^2 / L^2)`` with the length scale ``L``
taken as a direct parameter. Gravity follows exactly from the enclosed mass
of that density (an erf expression, no series truncation), and pressure
integrates hydrostatic balance inward from the CMB anchor with fixed-order
Gauss-Legendre panels. The adiabat is ``T(r) = T_cmb * exp((r_cmb^2 - r^2)
/ D^2)`` with ``D^2 = 3 c_p / (2 pi alpha rho_cen G)``, the scale for which
``d ln T / dr = -alpha g / c_p`` holds exactly in the small-radius limit
where gravity is linear in ``r``.

Everything evaluates through ``jax.numpy`` and is jit- and grad-safe; the
constructor validates its scalar parameters eagerly, outside any trace.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as _np
from jax.scipy.special import erf
from scipy import constants as sp_constants

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
        (Labrosse et al. 2001; Nimmo 2015, Eq. 3), whose derivation uses
        the small-radius gravity expansion. The two differ by a few
        tenths of a percent in the deep core (0.3% at the Earth centre),
        enough to move a tangent melting-curve crossing by hundreds of
        kilometres when reproducing models built on the printed form.

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
    ) -> None:
        if pressure_mode not in ('quadrature', 'labrosse'):
            raise ValueError(f'unknown pressure_mode {pressure_mode!r}')
        self.pressure_mode = pressure_mode
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
        """Gravitational acceleration [m s-2] at radius ``r``, positive outward-pull.

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
        """Printed closed form (Nimmo 2015, Eq. 3), anchored at the CMB.

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
        """Adiabatic temperature [K] at radius ``r`` anchored at ``t_cmb``.

        ``T(r) = t_cmb * exp((r_cmb^2 - r^2) / D^2)``; hotter inward,
        equal to ``t_cmb`` at the CMB by construction.
        """
        r = jnp.asarray(r)
        d2 = self.d_scale**2
        return t_cmb * jnp.exp((self.r_cmb**2 - r**2) / d2)

    def t_cen(self, t_cmb):
        """Centre temperature [K] on the adiabat anchored at ``t_cmb``."""
        return self.adiabat(0.0, t_cmb)
