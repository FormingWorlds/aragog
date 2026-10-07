"""Resolved thermal stratification at the top of the core.

The outer core above a fixed radius ``r_sh`` is a shell of finite volumes whose temperatures
are part of the solver state. Heat moves through the shell by conduction, and where the
temperature gradient is steeper than the adiabat (superadiabatic) also by convective mixing,
whose flux grows as the superadiabatic gradient to the power 3/2, as in mixing-length theory,
so it vanishes smoothly at the adiabat. A subadiabatic part conducts only: that part is the
stable layer, and its base rises and falls with the profile, so onset,
growth, erosion and re-formation need no events. The heat equation and the CMB condition are
those of Greenwood et al. (2021, eqs. 20 and 21); where their scheme mixes an unstable part
instantly and moves the layer base by a stability check (eq. 25), the mixing here acts over
the time ``L^2 / k_mix``.

Below ``r_sh`` the convecting core keeps its adiabat ``T_a(r; T_c)``; the shell base sits on
it, and the heat crossing that face is what the convecting core loses.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as _np

from aragog.core.profiles import GaussianCoreProfiles

jax.config.update('jax_enable_x64', True)


def _geometric_faces(r_base: float, r_top: float, n: int, d_top: float) -> _np.ndarray:
    """Cell faces from ``r_base`` to ``r_top`` growing geometrically downward from ``d_top``."""
    length = r_top - r_base
    if n * d_top >= length:
        return _np.linspace(r_base, r_top, n + 1)
    lo, hi = 1.0, 2.0
    for _ in range(200):  # the growth factor that fills the shell with n cells
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if d_top * (mid**n - 1.0) / (mid - 1.0) < length else (lo, mid)
    widths = d_top * lo ** _np.arange(n)
    widths *= length / widths.sum()
    return r_top - _np.concatenate([[0.0], _np.cumsum(widths)])[::-1]


class CoreShell:
    """Finite-volume shell ``r_sh <= r <= r_cmb`` over the convecting core.

    Parameters
    ----------
    profiles : GaussianCoreProfiles
        Core structure; the shell uses its density, specific heat and adiabat.
    k_core : float
        Thermal conductivity [W/m/K].
    base_fraction : float
        Shell base ``r_sh`` as a fraction of the CMB radius.
    n_cells : int
        Number of finite volumes.
    top_cell : float
        Width of the cell at the CMB [m]; the cells widen geometrically downward.
    k_mix : float
        Eddy diffusivity of convective mixing [m^2/s] at the reference superadiabatic gradient.
    g_mix : float
        Reference superadiabatic gradient, as a fraction of the adiabatic gradient at the CMB.
    """

    def __init__(
        self,
        profiles: GaussianCoreProfiles,
        k_core: float,
        *,
        base_fraction: float = 0.4,
        n_cells: int = 64,
        top_cell: float = 2.0e3,
        k_mix: float = 100.0,
        g_mix: float = 1.0e-3,
    ) -> None:
        if not 0.0 < base_fraction < 1.0:
            raise ValueError(f'base_fraction must lie in (0, 1), got {base_fraction}')
        if n_cells < 2 or not k_core > 0.0 or not k_mix > 0.0 or not g_mix > 0.0:
            raise ValueError('CoreShell needs n_cells >= 2 and positive k_core, k_mix, g_mix')
        p = profiles
        self.profiles, self.k_core, self.k_mix, self.g_mix = p, float(k_core), k_mix, g_mix
        faces = _geometric_faces(base_fraction * p.r_cmb, p.r_cmb, int(n_cells), top_cell)
        self.r_faces = jnp.asarray(faces)
        self.r_cells = jnp.asarray(0.5 * (faces[1:] + faces[:-1]))
        mass = _np.asarray(p.enclosed_mass(jnp.asarray(faces)))
        self.mass = jnp.asarray(_np.diff(mass))
        self.area = 4.0 * jnp.pi * self.r_faces**2
        self.rho_cp_faces = p.density(self.r_faces) * p.c_p
        self.r_base = float(faces[0])

    @property
    def n_cells(self) -> int:
        return int(self.r_cells.shape[0])

    def adiabatic_profile(self, t_c):
        """Cell temperatures [K] on the adiabat of the convecting core: a shell without a layer."""
        return self.profiles.adiabat(self.r_cells, t_c)

    def _adiabat_gradient(self, r, t_c):
        return -2.0 * r * self.profiles.adiabat(r, t_c) / self.profiles.d_scale**2

    def _anomaly_gradient(self, t_shell, t_c):
        """Gradient [K/m] of the departure from the adiabat at faces 0 .. n-1 (the base face sits
        on the adiabat), and the reference superadiabatic gradient."""
        theta = jnp.concatenate([jnp.zeros(1), t_shell - self.adiabatic_profile(t_c)])
        r_inner = jnp.concatenate([self.r_faces[:1], self.r_cells])
        g_ref = self.g_mix * jnp.abs(self._adiabat_gradient(self.profiles.r_cmb, t_c))
        return jnp.diff(theta) / jnp.diff(r_inner), g_ref

    def face_fluxes(self, t_shell, t_c, q_cmb):
        """Outward heat flow [W] through every face; face 0 is the shell base (the flow out of
        the convecting core), the last face the CMB (``q_cmb``)."""
        anomaly, g_ref = self._anomaly_gradient(t_shell, t_c)
        faces = self.r_faces[:-1]
        unstable = jnp.maximum(-anomaly, 0.0)
        mixed = self.rho_cp_faces[:-1] * self.k_mix * unstable**1.5 / jnp.sqrt(g_ref)
        grad = self._adiabat_gradient(faces, t_c) + anomaly
        inner = self.area[:-1] * (mixed - self.k_core * grad)
        return jnp.concatenate([inner, jnp.atleast_1d(q_cmb)])

    def rates(self, t_shell, t_c, q_cmb, heating=0.0):
        """Cell temperature rates [K/s] and the flow [W] out of the convecting core, for an
        internal heating ``heating`` [W/kg]."""
        flux = self.face_fluxes(t_shell, t_c, q_cmb)
        net = flux[:-1] - flux[1:] + heating * self.mass
        return net / (self.mass * self.profiles.c_p), flux[0]

    def top_temperature(self, t_shell, q_cmb):
        """Temperature [K] at the CMB: the top cell conducted over its upper half."""
        half = self.profiles.r_cmb - self.r_cells[-1]
        return t_shell[-1] - q_cmb * half / (self.area[-1] * self.k_core)

    def heat_content(self, t_shell):
        """Sensible heat [J] of the shell, ``c_p sum m_j T_j``."""
        return self.profiles.c_p * jnp.sum(self.mass * t_shell)

    def layer_base(self, t_shell, t_c):
        """Base radius [m] of the stable layer: the shell base plus the width of every cell whose
        lower face is not stably stratified (anomaly gradient below the reference gradient),
        weighted smoothly; the CMB without a layer."""
        anomaly, g_ref = self._anomaly_gradient(t_shell, t_c)
        mixed = jax.nn.sigmoid(10.0 * (g_ref - anomaly) / g_ref)
        return self.r_base + jnp.sum(mixed * jnp.diff(self.r_faces))
