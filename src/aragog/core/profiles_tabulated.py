"""Core structure from tabulated profiles on a PALEOS liquid-iron isentrope.

Zalmoxis solves the core on its own nodes and returns ``r, m, P, rho, g`` per node (its
``result['core']``). Here the structure stays fixed between structure solves, and the outer
core is one isentrope of the PALEOS liquid iron table (:class:`aragog.core.iron_table.
LiquidIronTable`): the well-mixed core has a uniform specific entropy ``s_c = s(P_cmb,
T_cmb)``, and the temperature at each node solves ``s(P_k, T_a,k) = s_c``. The adiabat is
tabulated once, at install, on the nodes and on a grid of CMB temperatures, as ``f = ln(T_a /
T_cmb)`` with its slopes from the table derivatives, ``df/dr = (dT/dP)_s (dP/dr) / T_a`` with
``(dT/dP)_s = -s_P / s_T`` and ``dT_a/dT_cmb = s_T(P_cmb, T_cmb) / s_T(P_k, T_a,k)`` (the cross
slope from a spline in ``T_cmb``); the solver reads it through bicubic Hermite patches, jit- and
grad-safe.

The secular heat capacity is ``C_s(T_cmb) = (c_p / T_cmb) int T_a dm`` (from ``dh = T ds`` at
fixed pressure and the uniform ``s_c``: ``dH/dT_cmb = (ds_c/dT_cmb) int T_a dm``), with ``c_p``
the CMB value of the table or a constant override (840 J/kg/K by default, because the PALEOS
liquid sets ``C_V = 3nR`` with no electronic term). It is tabulated with its derivative and
read as a cubic Hermite in ``T_cmb``; the heat change of a call is the exact integral of that
same cubic, so the energy the solver books and the energy it integrates are one function.
"""

from __future__ import annotations

import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np
from scipy import constants as sp_constants
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq

from aragog.core.iron_table import LiquidIronTable

jax.config.update('jax_enable_x64', True)

G = sp_constants.G
REQUIRED_KEYS = ('r', 'm', 'P', 'rho', 'g')
ISENTROPE_RTOL = 1e-6  # relative entropy miss of the isentrope, at nodes and midpoints
MASS_RTOL = 1e-10  # arrays against the structure mass
MASS_DENSITY_RTOL = 1e-2  # int 4 pi r^2 rho dr against m; 1.1e-3 measured at an ICB jump
MAX_CELLS = 2_000_000  # nodes x CMB temperatures of the isentrope grid
# Calibration window of the Luo et al. (2024) liquid (Attia et al. 2026, p. 5).
LUO_P_PA = (50e9, 1300e9)
LUO_T_K = (8000.0, 14000.0)
_GL3_X, _GL3_W = np.polynomial.legendre.leggauss(3)  # exact for the cubic pieces


def is_positive_number(x) -> bool:
    """True for a finite positive real number that is not a bool."""
    return (
        isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x) and x > 0
    )


def _clamp(x, lo, hi):
    """``x`` limited to ``[lo, hi]``, with gradient 1 at the bounds (``jnp.clip`` gives 0.5)."""
    return jnp.where(x < lo, lo, jnp.where(x > hi, hi, x))


def _cell(x, nodes):
    """Interval index, width and local coordinate of ``x`` on sorted ``nodes`` (clamped)."""
    i = jnp.clip(jnp.searchsorted(nodes, x, side='right') - 1, 0, nodes.size - 2)
    h = nodes[i + 1] - nodes[i]
    return i, h, (x - nodes[i]) / h


def _basis(u):
    """Cubic Hermite basis ``(h00, h10, h01, h11)`` at local coordinate ``u``."""
    u2 = u * u
    u3 = u2 * u
    return 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2


def _hermite(x, nodes, values, slopes):
    """Cubic Hermite interpolant through ``values`` with ``slopes`` at ``nodes``."""
    i, h, u = _cell(x, nodes)
    b0, b1, b2, b3 = _basis(u)
    return b0 * values[i] + b1 * h * slopes[i] + b2 * values[i + 1] + b3 * h * slopes[i + 1]


class TabulatedCoreProfiles:
    """Density, mass, gravity, pressure, potential and isentrope of a tabulated core.

    The interface is that of :class:`aragog.core.profiles.GaussianCoreProfiles` that
    :class:`aragog.core.budget.CoreEnergyBudget` reads, plus :meth:`secular_capacity` and
    :meth:`secular_heat_change`, which depend on ``T_cmb`` here.

    Parameters
    ----------
    core : dict
        Core arrays from the centre to the CMB, the names of the Zalmoxis ``result['core']``:
        ``r`` [m] (starting at 0, strictly increasing), ``m`` [kg] (enclosed mass, starting
        at 0), ``P`` [Pa] (strictly decreasing), ``rho`` [kg m-3] and ``g`` [m s-2]. Other
        keys (``T``, ``phase``, ``w_<el>``, the scalars) are accepted and not used.
    table : LiquidIronTable
        The PALEOS liquid iron table the isentrope is read from.
    c_p : float or None
        Constant isobaric heat capacity [J kg-1 K-1] of the secular term, or ``None`` for the
        table value at the CMB.
    t_cmb_range : tuple of float, optional
        Lowest and highest CMB temperature [K] of the tabulated isentropes; every node must
        stay inside the table box over the range. By default the whole range the table
        holds: from the CMB temperature whose isentrope reaches the lowest table temperature
        at the node of highest entropy there (the CMB where ``(ds/dP)_T = -alpha/rho < 0``) to
        the one whose isentrope reaches the highest table temperature at the node of lowest
        entropy there (the centre where ``(ds/dP)_T < 0``).
    t_cmb_step : float
        Spacing [K] of the CMB temperature grid.
    m_core : float, optional
        Core mass [kg] of the structure; when given, the last node of ``m`` must equal it to
        1e-10 relative.

    Raises
    ------
    ValueError
        On a missing or malformed array, core pressures outside the pressure range of the
        table, a density whose integral misses the node mass by more than 1e-2 (Zalmoxis
        cores miss by at most 1.1e-3, at a density jump of the inner-core boundary), an
        isentrope that leaves the table between nodes, a node outside the table box on the
        range, an isentrope that misses ``s_c`` by more than 1e-6 relative, a mass that
        differs from ``m_core``, or a heat capacity that is not positive.
    """

    adiabat_mode = 'tabulated'

    def __init__(
        self,
        core: dict,
        table: LiquidIronTable,
        *,
        c_p: float | None = 840.0,
        t_cmb_range: tuple[float, float] | None = None,
        t_cmb_step: float = 10.0,
        m_core: float | None = None,
    ) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in core]
        if missing:
            raise ValueError(f'core arrays are missing {missing}')
        arr = {k: np.asarray(core[k], dtype=float) for k in REQUIRED_KEYS}
        n = arr['r'].size
        for k, v in arr.items():
            if v.shape != (n,):
                raise ValueError(f'core array {k} has shape {v.shape}, expected ({n},)')
            if not np.all(np.isfinite(v)):
                raise ValueError(f'core array {k} holds a non-finite value')
        if n < 4:
            raise ValueError(f'core arrays need at least 4 nodes, got {n}')
        r, m, p, rho, g = (arr[k] for k in REQUIRED_KEYS)
        if r[0] != 0.0 or m[0] != 0.0:
            raise ValueError('core arrays must start at the centre (r = 0, m = 0)')
        if not np.all(np.diff(r) > 0.0):
            raise ValueError('core radius r must be strictly increasing')
        if not np.all(np.diff(m) > 0.0):
            raise ValueError('core enclosed mass m must be strictly increasing')
        if not np.all(np.diff(p) < 0.0):
            raise ValueError('core pressure P must be strictly decreasing outward')
        if not (np.all(rho > 0.0) and np.all(g >= 0.0)):
            raise ValueError('core density must be positive and gravity non-negative')
        if not (table.pressure[0] <= p[-1] and p[0] <= table.pressure[-1]):
            raise ValueError(
                f'core pressures {p[-1]:.4e} to {p[0]:.4e} Pa leave the pressure range of the '
                f'liquid iron table ({table.pressure[0]:.4e} to {table.pressure[-1]:.4e} Pa); '
                'give a table that covers the core'
            )
        if m_core is not None and not abs(m[-1] / float(m_core) - 1.0) <= MASS_RTOL:
            raise ValueError(
                f'core arrays hold {m[-1]:.10e} kg, the structure {m_core:.10e} kg'
            )
        m_rho = PchipInterpolator(r, 4.0 * np.pi * r**2 * rho).integrate(0.0, r[-1])
        if not abs(m_rho / m[-1] - 1.0) <= MASS_DENSITY_RTOL:
            raise ValueError(
                f'core density integrates to {m_rho:.6e} kg, the node mass m is {m[-1]:.6e} kg '
                f'(more than {MASS_DENSITY_RTOL:g} apart); check rho and m of the core arrays'
            )
        if c_p is not None and not is_positive_number(c_p):
            raise ValueError(f'c_p must be a positive finite number or None, got {c_p!r}')
        t_lo, t_hi = self._full_range(table, p) if t_cmb_range is None else t_cmb_range
        if not (is_positive_number(t_lo) and is_positive_number(t_hi) and t_lo < t_hi) or not (
            is_positive_number(t_cmb_step)
        ):
            raise ValueError(f'bad CMB temperature grid {t_cmb_range}, step {t_cmb_step}')
        t_lo, t_hi = float(t_lo), float(t_hi)
        n_grid = int(np.ceil((t_hi - t_lo) / t_cmb_step)) + 1
        if n * n_grid > MAX_CELLS:
            raise ValueError(
                f'the isentropes need {n} nodes x {n_grid} temperatures, at most {MAX_CELLS} '
                'cells; raise core_t_cmb_step'
            )

        self.table, self.c_p_override = table, None if c_p is None else float(c_p)
        self.r_cmb, self.p_cmb, self.m_core = float(r[-1]), float(p[-1]), float(m[-1])
        self.n_nodes, self.rho_cen = n, float(rho[0])
        self._r = self.r_nodes = jnp.asarray(r)
        dm = 4.0 * np.pi * r**2 * rho
        dg = np.where(r > 0.0, 4.0 * np.pi * G * rho - 2.0 * g / np.where(r > 0, r, 1.0), 0.0)
        dg[0] = 4.0 * np.pi * G * rho[0] / 3.0
        psi = PchipInterpolator(r, g).antiderivative()(r)
        psi -= psi[-1]  # -int_r^r_cmb g ds, zero at the CMB
        fields = {
            'rho': (rho, PchipInterpolator(r, rho).derivative()(r)),
            'm': (m, dm),
            'P': (p, -rho * g),
            'g': (g, dg),
            'psi': (psi, g),
        }
        self._fields = {k: (jnp.asarray(a), jnp.asarray(b)) for k, (a, b) in fields.items()}
        self._install_isentropes(r, p, rho, g, m, np.linspace(t_lo, t_hi, n_grid))

    # -- install -------------------------------------------------------------

    @staticmethod
    def _full_range(table, p):
        """CMB temperatures [K] whose isentropes stay in the table box at every node.

        The hottest isentrope meets the highest table temperature at the node of lowest entropy
        there, the coldest the lowest table temperature at the node of highest entropy; these
        are the centre and the CMB only where ``(ds/dP)_T < 0`` along the whole core.
        """
        t_min, t_max = float(table.temperature[0]), float(table.temperature[-1])
        s_top = float(np.min(table.value('s', p, t_max)))
        s_bot = float(np.max(table.value('s', p, t_min)))

        def gap(t, s):
            return float(table.value('s', float(p[-1]), t)) - s

        if not (gap(t_min, s_top) < 0.0 < gap(t_max, s_top) and gap(t_max, s_bot) > 0.0):
            raise ValueError(
                'no CMB temperature keeps the isentrope inside the liquid iron table '
                f'({t_min:.1f} to {t_max:.1f} K) from the CMB to the centre'
            )
        t_hi = brentq(gap, t_min, t_max, args=(s_top,)) * (1.0 - 1e-9)
        t_lo = t_min
        if gap(t_min, s_bot) < 0.0:
            t_lo = brentq(gap, t_min, t_max, args=(s_bot,)) * (1.0 + 1e-9)
        if not t_lo < t_hi:
            raise ValueError(
                'no CMB temperature keeps the isentrope inside the liquid iron table '
                f'({t_min:.1f} to {t_max:.1f} K) at every node'
            )
        return float(t_lo), float(t_hi)

    def _install_isentropes(self, r, p, rho, g, m, t_grid):
        """Invert the table entropy on every node and CMB temperature of ``t_grid``; check the
        interpolated isentrope between them; tabulate C_s."""
        tab = self.table
        p2 = np.broadcast_to(p[:, None], (p.size, t_grid.size))
        s_c = tab.value('s', self.p_cmb, t_grid)
        t_a = self._invert_entropy(p2, s_c, np.broadcast_to(t_grid, p2.shape))
        miss = np.abs(tab.value('s', p2, t_a) / s_c - 1.0)
        if not np.max(miss) <= ISENTROPE_RTOL:
            raise ValueError(f'isentrope misses s_c by {np.max(miss):.2e} relative')
        s_t = tab.ds_dT(p2, t_a)
        dtdp_s = -tab.ds_dP(p2, t_a) / s_t
        f = np.log(t_a / t_grid)
        f_r = dtdp_s * (-rho * g)[:, None] / t_a
        f_t = tab.ds_dT(self.p_cmb, t_grid)[None, :] / (s_t * t_a) - 1.0 / t_grid
        f_rt = CubicSpline(t_grid, f_r, axis=1)(t_grid, 1)
        self.t_grid = jnp.asarray(t_grid)
        self._f = tuple(jnp.asarray(x) for x in (f, f_r, f_t, f_rt))

        def check(r_x, t_x, where):
            p_x = np.asarray(self.pressure(r_x))[:, None]
            t_a_x = self.adiabat(r_x[:, None], t_x)
            try:
                s_x = tab.value('s', p_x, t_a_x)
            except ValueError as err:
                raise ValueError(
                    f'the isentrope leaves the liquid iron table {where.split(";")[0]} ({err}); '
                    'check the core arrays (r, m, P, rho, g) for a consistent structure'
                ) from err
            miss = np.max(np.abs(s_x / tab.value('s', self.p_cmb, t_x) - 1))
            if not miss <= ISENTROPE_RTOL:
                raise ValueError(f'isentrope misses s_c by {miss:.2e} relative {where}')

        t_mid, r_mid = 0.5 * (t_grid[1:] + t_grid[:-1]), 0.5 * (r[1:] + r[:-1])
        check(r, t_mid, 'between grid temperatures; lower core_t_cmb_step')
        check(r_mid, t_grid, 'between radial nodes; give the core arrays more nodes')
        check(r_mid, t_mid, 'inside the grid cells; lower core_t_cmb_step or add nodes')

        mass_t = PchipInterpolator(m, t_a, axis=0).integrate(0.0, m[-1])  # int T_a dm
        dta_dt = t_a * (f_t + 1.0 / t_grid)
        mass_dt = PchipInterpolator(m, dta_dt, axis=0).integrate(0.0, m[-1])
        if self.c_p_override is None:
            cp = tab.value('cp', self.p_cmb, t_grid)
            dcp = CubicSpline(t_grid, cp)(t_grid, 1)
        else:
            cp, dcp = np.full(t_grid.size, self.c_p_override), np.zeros(t_grid.size)
        c_s = cp * mass_t / t_grid
        dc_s = dcp * mass_t / t_grid - cp * mass_t / t_grid**2 + cp * mass_dt / t_grid
        if not np.all(c_s > 0.0):
            raise ValueError('secular heat capacity must be positive on the whole grid')
        self._c_s = (jnp.asarray(c_s), jnp.asarray(dc_s))
        self._luo_mass = self._outside_luo(p2, t_a, m)

    def _invert_entropy(self, p, s_target, t_start):
        """Temperature [K] with ``s(p, T) = s_target``: Newton on ``ds/dT`` inside the box.

        Raises
        ------
        ValueError
            Where ``s_target`` lies outside the entropy range of the table at ``p``, so that
            the isentrope leaves the box.
        """
        tab = self.table
        t_min, t_max = float(tab.temperature[0]), float(tab.temperature[-1])
        outside = (s_target > tab.value('s', p, t_max)) | (s_target < tab.value('s', p, t_min))
        if np.any(outside):
            k = np.flatnonzero(outside.ravel())[0]
            raise ValueError(
                f'core isentrope leaves the liquid iron table at P = {p.ravel()[k]:.4e} Pa '
                f'(T between {t_min:.1f} and {t_max:.1f} K); narrow t_cmb_range'
            )
        t = np.array(t_start, dtype=float)
        for _ in range(60):
            step = (tab.value('s', p, t) - s_target) / tab.ds_dT(p, t)
            t = np.clip(t - step, t_min, t_max)
            if np.max(np.abs(step / t)) < 1e-14:
                break
        return t

    def _outside_luo(self, p2, t_a, m):
        """Mass [kg] per CMB temperature outside the Luo et al. (2024) calibration window."""
        dmass = np.diff(m)
        mid = lambda x: 0.5 * (x[1:] + x[:-1])  # noqa: E731
        p_mid, t_mid = mid(p2), mid(t_a)
        out = (
            (p_mid < LUO_P_PA[0])
            | (p_mid > LUO_P_PA[1])
            | (t_mid < LUO_T_K[0])
            | (t_mid > LUO_T_K[1])
        )
        return np.sum(out * dmass[:, None], axis=0)

    # -- radial structure ----------------------------------------------------

    def _radial(self, name, r):
        values, slopes = self._fields[name]
        return _hermite(_clamp(jnp.asarray(r), 0.0, self.r_cmb), self._r, values, slopes)

    def density(self, r):
        """Density [kg m-3] at radius ``r`` [m]."""
        return self._radial('rho', r)

    def enclosed_mass(self, r):
        """Mass [kg] inside radius ``r`` [m]; the Zalmoxis mass at the nodes."""
        return self._radial('m', r)

    def gravity(self, r):
        """Gravitational acceleration [m s-2] at radius ``r`` [m]."""
        return self._radial('g', r)

    def pressure(self, r):
        """Pressure [Pa] at radius ``r`` [m]; slopes from hydrostatic balance."""
        return self._radial('P', r)

    def potential(self, r):
        """Gravitational potential [J/kg] at radius ``r`` [m], zero at the CMB."""
        return self._radial('psi', r)

    # -- isentrope -----------------------------------------------------------

    def _log_ratio(self, r, t_cmb):
        """``ln(T_a / T_cmb)`` on the bicubic Hermite patches in ``(r, T_cmb)``."""
        r, t = jnp.broadcast_arrays(_clamp(jnp.asarray(r), 0.0, self.r_cmb), jnp.asarray(t_cmb))
        t = _clamp(t, self.t_grid[0], self.t_grid[-1])
        i, hr, u = _cell(r, self._r)
        j, ht, v = _cell(t, self.t_grid)
        bu, bv = _basis(u), _basis(v)
        f, f_r, f_t, f_rt = self._f
        total = 0.0
        for a, di in ((0, 0), (2, 1)):
            for b, dj in ((0, 0), (2, 1)):
                k, l = i + di, j + dj
                total = total + bu[a] * bv[b] * f[k, l]
                total = total + bu[a + 1] * hr * bv[b] * f_r[k, l]
                total = total + bu[a] * bv[b + 1] * ht * f_t[k, l]
                total = total + bu[a + 1] * hr * bv[b + 1] * ht * f_rt[k, l]
        return total

    def adiabat(self, r, t_cmb):
        """Temperature [K] at radius ``r`` on the isentrope through ``t_cmb`` at the CMB."""
        return jnp.asarray(t_cmb) * jnp.exp(self._log_ratio(r, t_cmb))

    def adiabat_gradient(self, r, t_cmb):
        """``dT/dr`` [K/m] of the isentrope through ``t_cmb``."""
        grad = jnp.vectorize(jax.grad(self.adiabat))
        return grad(jnp.asarray(r, dtype=float), jnp.asarray(t_cmb, dtype=float))

    def t_cen(self, t_cmb):
        """Centre temperature [K] on the isentrope through ``t_cmb``."""
        return self.adiabat(0.0, t_cmb)

    def t_cmb_at_centre(self, t_centre: float) -> float:
        """CMB temperature [K] of the isentrope with centre temperature ``t_centre`` [K].

        Raises
        ------
        ValueError
            If no tabulated isentrope reaches ``t_centre``.
        """
        lo, hi = float(self.t_grid[0]), float(self.t_grid[-1])

        def gap(t):
            return float(self.t_cen(t)) - float(t_centre)

        if gap(lo) * gap(hi) > 0.0:
            raise ValueError(f'no tabulated isentrope has a centre at {t_centre:.1f} K')
        return float(brentq(gap, lo, hi, xtol=1e-9, rtol=1e-14))

    # -- secular term ----------------------------------------------------------

    def secular_capacity(self, t_cmb):
        """Secular heat capacity ``(c_p / T_cmb) int T_a dm`` [J/K] at ``t_cmb`` [K]."""
        t = _clamp(jnp.asarray(t_cmb), self.t_grid[0], self.t_grid[-1])
        return _hermite(t, self.t_grid, *self._c_s)

    def secular_heat_change(self, t_cmb: float, d_t: float) -> float:
        """Exact integral [J] of :meth:`secular_capacity` from ``t_cmb`` to ``t_cmb + d_t``.

        Each piece of the cubic between grid temperatures is integrated by 3-point
        Gauss-Legendre, exact for a cubic, so the result is the change of the antiderivative
        without the difference of 2 large contents.
        """
        a, d = float(t_cmb), float(d_t)
        grid = np.asarray(self.t_grid)
        inner = grid[(grid > min(a, a + d)) & (grid < max(a, a + d))]
        edges = np.concatenate(([0.0], np.sort(np.abs(inner - a)), [abs(d)]))  # exact widths
        half = 0.5 * np.diff(edges)
        sign = 1.0 if d >= 0.0 else -1.0
        nodes = a + sign * (edges[:-1, None] + half[:, None] * (_GL3_X + 1.0))
        return sign * float(np.sum(half * (np.asarray(self.secular_capacity(nodes)) @ _GL3_W)))

    def check_t_cmb(self, t_cmb) -> None:
        """Refuse CMB temperatures [K] outside the tabulated isentropes (NaN included).

        Raises
        ------
        ValueError
            At the first temperature outside ``[t_grid[0], t_grid[-1]]``, naming it.
        """
        t = np.atleast_1d(np.asarray(t_cmb, dtype=float))
        lo, hi = float(self.t_grid[0]), float(self.t_grid[-1])
        off = ~((t >= lo) & (t <= hi))
        if off.any():
            raise ValueError(
                f'T_core = {t[np.argmax(off)]:.10g} K lies outside the tabulated isentropes '
                f'({lo:.10g} to {hi:.10g} K)'
            )

    def outside_calibration_fraction(self, t_cmb) -> float:
        """Mass fraction of the core outside the Luo et al. (2024) window at ``t_cmb`` [K]."""
        frac = np.interp(float(t_cmb), np.asarray(self.t_grid), self._luo_mass) / self.m_core
        return float(frac)
