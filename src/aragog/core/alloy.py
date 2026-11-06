"""Liquidus of the core alloy: ideal-solution Fe-X binaries joined by a harmonic mean of their A.

Each Fe-X binary follows the ideal-solution liquidus of Mori et al. (2017, Eq. 3) and Mashino
et al. (2019): ``x_Fe = exp[A (1 - T_Fe / T)]``, with ``x_Fe`` the mole fraction of free Fe among
Fe and the Fe_k X_m units of the alloy, ``T_Fe`` the pure-iron curve of
:class:`aragog.core.melting.IronMeltingCurve`, and ``A`` fixed at each pressure by the binary
eutectic point (``w_e``, ``T_e``). The units are FeO for O; for S they are FeS below 18 GPa and
Fe3S above 26 GPa, and for C Fe3C below 255 GPa and Fe7C3 above, each pair joined by a C2 blend of
the two liquidus temperatures (over 18 to 26 GPa and 230 to 280 GPa). With several elements,
``x_Fe = 1 - sum x_i`` and ``1/A = sum (x_i/X) / A_i``, which keeps the dilute limit of each binary, ``dT ~ T_Fe sum x_i / A_i``. Si is treated as
lowering no melting point: its atoms count with the Fe solvent, so an Fe-Si alloy melts like Fe.
H and N have no liquidus data at core pressures and leave the mole count: an alloy melts like the
same alloy without them.

The eutectic compositions and the Fe-O eutectic temperature are tables read from the papers'
figures (``aragog/core/eutectics/*.csv``, sources in each header); the Fe-S and Fe-C eutectic
temperatures are the Simon fits of Mori et al. (2017) and Mashino et al. (2019). Below them the
Fe-S eutectic is linear between the nodes of Fei et al. (1997) at 1 bar, 10 and 14 GPa, towards
1348 K at 21 GPa and the composition table at 20 GPa, joined into the fit and the table by a
smootherstep over 18 to 21 GPa; the Fe-C eutectic is the cubic from 1 bar (1416.15 K, 4.3 wt%)
with zero curvature there and the value and slope of the fit and the table at 5 GPa. Above the
data, the tables are read at an effective pressure ``q`` whose slope fades from 1 to 0 by a
smootherstep over the 40 GPa above the end, so ``q`` stops 20 GPa above the end: the eutectic temperature is its value
at ``q`` times ``T_Fe(P) / T_Fe(q)``, homologous to the pure-iron curve at ``q``, and the
composition and Fe-O temperature tables continue along their last segment up to ``q``. A
composition past the Fe-rich side of the eutectic (``sum w_i / w_e,i > 1``) is evaluated at its
boundary.
"""

from __future__ import annotations

import functools
import importlib.resources
import numbers

import jax
import jax.numpy as jnp
import numpy as np

from aragog.core.melting import IronMeltingCurve

jax.config.update('jax_enable_x64', True)

DEPRESSING = ('S', 'C', 'O')
INERT = ('Si', 'H', 'N')
ELEMENTS = DEPRESSING + INERT
SOLVENT = ('Si',)
# Standard atomic weights [g/mol]; only ratios enter.
_MOLAR = {
    'Fe': 55.845,
    'S': 32.06,
    'C': 12.011,
    'O': 15.999,
    'Si': 28.085,
    'H': 1.008,
    'N': 14.007,
}
# Fe_k X_m units (k, m): FeO; FeS and Fe3S; Fe3C and Fe7C3.
_UNIT_O = (1.0, 1.0)
_UNITS_S = ((1.0, 1.0), (3.0, 1.0))
_UNITS_C = ((3.0, 1.0), (7.0, 3.0))
S_UNIT_BLEND_GPA = (18.0, 26.0)
C_UNIT_SWITCH_PA = 255.0e9
C_UNIT_BLEND_HALF_WIDTH_PA = 25.0e9
HOMOLOGOUS_FADE_GPA = 40.0
# Simon fits of the eutectic temperature, T_ref [K], P_ref [GPa], a [GPa], c.
_SIMON = {'S': (1348.0, 21.0, 36.5, 2.07), 'C': (1513.0, 5.0, 8.4, 4.1)}
S_JOIN_GPA = (18.0, _SIMON['S'][1])


def _table(name: str) -> np.ndarray:
    """Table of ``aragog/core/eutectics/<name>.csv``, all columns, pressure in GPa first."""
    path = importlib.resources.files('aragog.core').joinpath('eutectics', f'{name}.csv')
    rows = [
        ln for ln in path.read_text(encoding='utf-8').splitlines() if not ln.startswith('#')
    ]
    return np.array([[float(v) for v in ln.split(',')] for ln in rows[1:]])


@functools.cache
def _data() -> dict:
    """Eutectic tables, the low-pressure nodes (P, T_e, w_e) and the pressure range [GPa] of each
    binary's data."""
    comp = {el: _table(f'fe_{el.lower()}_composition') for el in DEPRESSING}
    low = {el: _table(f'fe_{el.lower()}_low_pressure') for el in ('S', 'C')}
    t_o = _table('fe_o_temperature')
    rng = {
        'S': (low['S'][0, 0], comp['S'][-1, 0]),
        'C': (low['C'][0, 0], comp['C'][-1, 0]),
        'O': (max(comp['O'][0, 0], t_o[0, 0]), min(comp['O'][-1, 0], t_o[-1, 0])),
    }
    return {'composition': comp, 'low': low, 't_o': t_o, 'range': rng}


def _interp(p_gpa, table):
    """Linear interpolation of a 2-column table, its end segments continued beyond its ends."""
    x, y = table[:, 0], table[:, 1]
    first = (y[1] - y[0]) / (x[1] - x[0])
    last = (y[-1] - y[-2]) / (x[-1] - x[-2])
    beyond = jnp.where(p_gpa < x[0], (p_gpa - x[0]) * first, 0.0)
    return jnp.interp(p_gpa, x, y) + jnp.where(p_gpa > x[-1], (p_gpa - x[-1]) * last, beyond)


def _smootherstep(z):
    """C2 step from 0 at ``z <= 0`` to 1 at ``z >= 1``: 6z^5 - 15z^4 + 10z^3."""
    z = jnp.clip(z, 0.0, 1.0)
    return z * z * z * (z * (z * 6.0 - 15.0) + 10.0)


def _natural_cubic(x, span, y0, y1, s1):
    """Cubic through ``y0`` at ``x = 0`` with zero curvature there, and ``y1`` with slope ``s1``
    at ``x = span``."""
    d = (s1 * span - (y1 - y0)) / (2.0 * span**3)
    return y0 + (s1 - 3.0 * d * span**2) * x + d * x**3


def _eutectic_in_data(el: str, q):
    """Eutectic temperature [K] and composition of the Fe-``el`` data at ``q`` [GPa] inside the
    data, each table continued along its end segments beyond them."""
    comp = _data()['composition'][el]
    if el == 'O':
        return _interp(q, _data()['t_o']), _interp(q, comp)
    low = _data()['low'][el]
    t_ref, p_ref, a, c = _SIMON[el]
    t_fit, w_fit = t_ref * ((q - p_ref) / a + 1.0) ** (1.0 / c), _interp(q, comp)
    if el == 'S':
        p0, p1 = S_JOIN_GPA
        t_low = _interp(q, np.vstack([low[:, :2], [p_ref, t_ref]]))
        w_low = _interp(q, np.vstack([low[:, [0, 2]], comp[:1]]))
        b = _smootherstep((q - p0) / (p1 - p0))
        return (1.0 - b) * t_low + b * t_fit, (1.0 - b) * w_low + b * w_fit
    span, x = p_ref - low[0, 0], q - low[0, 0]
    w_slope = (comp[1, 1] - comp[0, 1]) / (comp[1, 0] - comp[0, 0])
    t_cubic = _natural_cubic(x, span, low[0, 1], t_ref, t_ref / (a * c))
    w_cubic = _natural_cubic(x, span, low[0, 2], comp[0, 1], w_slope)
    return jnp.where(q < p_ref, t_cubic, t_fit), jnp.where(q < p_ref, w_cubic, w_fit)


def eutectic(el: str, pressure):
    """Eutectic temperature [K] and composition (mass fraction of ``el``) of the Fe-``el`` binary.

    The data are taken at an effective pressure ``q`` [GPa]: ``q = max(P, lo)`` up to the
    end of the data ``hi``, ``q = hi + h`` above ``hi + 2h`` and, in between,
    ``q = hi + 2h (z - z^4 (5/2 - 3z + z^2))`` with ``z = (P - hi) / 2h``, so that
    ``dq/dP = 1 - b(z)`` with the smootherstep ``b = 6z^5 - 15z^4 + 10z^3`` (``2h`` =
    :data:`HOMOLOGOUS_FADE_GPA`). The temperature keeps its ratio to the pure-iron curve at ``q``
    (homologous outside the data); inside, both follow the data, joined at the low-pressure end as
    the module docstring states, each table continued along its last segment above ``hi``.
    """
    lo, hi = _data()['range'][el]
    fade = HOMOLOGOUS_FADE_GPA
    p_gpa = jnp.asarray(pressure) / 1e9
    z = (p_gpa - hi) / fade
    faded = hi + fade * (z - z**4 * (2.5 - 3.0 * z + z * z))
    p_in = jnp.where(p_gpa < lo, lo, jnp.where(p_gpa > hi, hi, p_gpa))
    q = jnp.where(p_gpa <= hi, p_in, jnp.where(p_gpa >= hi + fade, hi + 0.5 * fade, faded))
    t_e, w_e = _eutectic_in_data(el, q)
    t_fe = IronMeltingCurve.t_melt_pure(p_gpa * 1e9)
    return t_e * t_fe / IronMeltingCurve.t_melt_pure(q * 1e9), w_e


def _mole_numbers(w: dict, units: dict):
    """Solvent moles (free Fe plus the Si atoms) and the mole numbers of the Fe_k X_m units, per
    unit mass of alloy, for a composition without H and N (see :func:`_without_h_n`)."""
    free = (1.0 - sum(w.values())) / _MOLAR['Fe'] + sum(w[el] / _MOLAR[el] for el in SOLVENT)
    u = {}
    for el in DEPRESSING:
        k, m = units[el]
        u[el] = w[el] / _MOLAR[el] / m
        free = free - k * u[el]
    return free, u


def _binary_a(el: str, t_fe, t_e, w_e, unit):
    """A of the Fe-``el`` ideal liquidus through (0, T_Fe) and the eutectic (w_e, T_e)."""
    k, m = unit
    u = w_e / _MOLAR[el] / m
    free = (1.0 - w_e) / _MOLAR['Fe'] - k * u
    return jnp.log(free / (free + u)) / (1.0 - t_fe / t_e)


def binary_liquidus(el: str, w, t_fe, t_e, w_e, unit):
    """Liquidus [K] of the Fe-``el`` binary at mass fraction ``w``, on the ideal form through
    (0, ``t_fe``) and the eutectic (``w_e``, ``t_e``) with Fe_k X_m units ``unit`` = (k, m)."""
    k, m = unit
    a = _binary_a(el, t_fe, t_e, w_e, unit)
    u = jnp.asarray(w) / _MOLAR[el] / m
    free = (1.0 - jnp.asarray(w)) / _MOLAR['Fe'] - k * u
    return t_fe / (1.0 - jnp.log(free / (free + u)) / a)


def _without_h_n(w: dict) -> dict:
    """The composition without H and N, the rest renormalised: the alloy the liquidus sees."""
    keep = 1.0 - w['H'] - w['N']
    return {el: (0.0 if el in ('H', 'N') else w[el] / keep) for el in ELEMENTS}


def _liquidus(t_fe, eut: dict, w: dict, c_unit, s_unit):
    """Alloy liquidus [K] for one C and one S unit choice, from the pure-iron curve, the eutectic
    points and the composition without H and N; the composition is scaled back to the Fe-rich
    boundary where ``sum w_i / w_e,i`` exceeds one."""
    excess = sum(w[el] / eut[el][1] for el in DEPRESSING)
    scale = jnp.where(excess > 1.0, 1.0 / jnp.maximum(excess, 1.0), 1.0)
    ws = {el: (w[el] * scale if el in DEPRESSING else w[el]) for el in ELEMENTS}
    units = {'S': s_unit, 'C': c_unit, 'O': _UNIT_O}
    free, u = _mole_numbers(ws, units)
    total = free + sum(u.values())
    x = {el: u[el] / total for el in DEPRESSING}
    big_x = sum(x.values())
    a_i = {el: _binary_a(el, t_fe, eut[el][0], eut[el][1], units[el]) for el in DEPRESSING}
    # log1p(-X) 1/A = (log1p(-X)/X) sum x_i/A_i; the ratio is -1 at X = 0, so dT/dw stays finite
    safe_x = jnp.where(big_x > 0.0, big_x, 0.5)
    ratio = jnp.where(big_x > 0.0, jnp.log1p(-safe_x) / safe_x, -1.0 - 0.5 * big_x)
    return t_fe / (1.0 - ratio * sum(x[el] / a_i[el] for el in DEPRESSING)), a_i


def _checked(composition) -> dict:
    """The composition as a dict of mass fractions over :data:`ELEMENTS`, missing keys zero.

    The values of a traced composition are taken as they are; concrete values become floats.

    Raises
    ------
    ValueError
        On a key outside :data:`ELEMENTS`, a value that is not a finite real scalar (a bool,
        string or array of more than 0 dimensions included), a negative value, or a total of one
        or more.
    """
    if not isinstance(composition, dict):
        raise ValueError(f'composition must be a dict of mass fractions, got {composition!r}')
    unknown = sorted(set(composition) - set(ELEMENTS))
    if unknown:
        raise ValueError(f'unknown light elements {unknown}; known: {list(ELEMENTS)}')
    if any(isinstance(v, jax.core.Tracer) for v in composition.values()):
        return {el: composition.get(el, 0.0) for el in ELEMENTS}
    out = {}
    for el in ELEMENTS:
        v = composition.get(el, 0.0)
        try:
            if isinstance(v, numbers.Real) and not isinstance(v, bool):
                v = float(v)
            a = np.asarray(v)
        except (TypeError, ValueError, OverflowError):
            a = np.asarray(None)
        if a.ndim != 0 or a.dtype.kind not in 'fiu' or not np.isfinite(a):
            raise ValueError(f'mass fraction of {el} must be a finite real number, got {v!r}')
        if a < 0.0:
            raise ValueError(f'mass fraction of {el} must not be negative, got {v}')
        out[el] = float(a)
    if sum(out.values()) >= 1.0:
        raise ValueError(
            f'light-element mass fractions sum to {sum(out.values())}, not below 1'
        )
    return out


class AlloyLiquidus:
    """Liquidus of an Fe alloy with S, C, O (melting depression) and Si, H, N (none).

    Parameters
    ----------
    composition : dict
        Mass fractions of the light elements, keys from :data:`ELEMENTS`; missing keys are zero.

    Raises
    ------
    ValueError
        On an invalid composition (see :func:`_checked`).
    """

    def __init__(self, composition: dict):
        self.composition = _checked(composition)

    def t_melt(self, pressure, composition: dict | None = None):
        """Liquidus temperature [K] at ``pressure`` [Pa]; ``composition`` overrides the
        instance's (values checked unless traced). Concrete results are checked against the
        pure-iron curve and each binary's A.

        The S blend (FeS to Fe3S, 18 to 26 GPa) ends below the C blend (230 to 280 GPa), so the
        two act one at a time. Below 18 GPa, where Fe3S units cannot hold the eutectic liquid, FeS
        units stand in for them at zero weight.
        """
        w = _without_h_n(_checked(self.composition if composition is None else composition))
        p = jnp.asarray(pressure, dtype=float)
        s0, s1 = S_UNIT_BLEND_GPA
        c_s = _smootherstep((p / 1e9 - s0) / (s1 - s0))
        c_c = _smootherstep(0.5 * ((p - C_UNIT_SWITCH_PA) / C_UNIT_BLEND_HALF_WIDTH_PA + 1.0))
        # S units of the upper blend end: Fe3S, or FeS at zero weight below 18 GPa (finite there)
        s_hi = (jnp.where(c_s > 0.0, _UNITS_S[1][0], _UNITS_S[0][0]), _UNITS_S[1][1])
        t_fe = IronMeltingCurve.t_melt_pure(p)
        eut = {el: eutectic(el, p) for el in DEPRESSING}
        t_s, a_s = _liquidus(t_fe, eut, w, _UNITS_C[0], _UNITS_S[0])
        t3, a3 = _liquidus(t_fe, eut, w, _UNITS_C[0], s_hi)
        t7, a7 = _liquidus(t_fe, eut, w, _UNITS_C[1], s_hi)
        t_liq = (1.0 - c_s) * t_s + c_s * ((1.0 - c_c) * t3 + c_c * t7)
        if not isinstance(t_liq, jax.core.Tracer):
            _assert_invariants(p, t_liq, a_s, a3, a7)
        return t_liq

    def gradient(self, pressure):
        """dT_liq/dP [K/Pa] at ``pressure`` [Pa], by automatic differentiation."""
        p = jnp.atleast_1d(jnp.asarray(pressure, dtype=float))
        return jax.vmap(jax.grad(lambda q: self.t_melt(q)))(p).reshape(jnp.shape(pressure))

    def status(self, pressure) -> dict:
        """Per-pressure flags (NumPy bool arrays): ``past_eutectic`` where the composition is
        past the Fe-rich side and is held at the boundary; ``extrapolated`` where a depressing
        element present in the alloy is outside its binary's data range."""
        p_gpa = np.asarray(pressure, dtype=float) / 1e9
        w = _without_h_n(self.composition)
        excess = sum(w[el] / np.asarray(eutectic(el, p_gpa * 1e9)[1]) for el in DEPRESSING)
        rng = _data()['range']
        outside = np.zeros_like(p_gpa, dtype=bool)
        for el in DEPRESSING:
            if w[el] > 0.0:
                outside |= (p_gpa < rng[el][0]) | (p_gpa > rng[el][1])
        return {'past_eutectic': excess > 1.0, 'extrapolated': outside}


def _assert_invariants(p, t_liq, *a_sets):
    """Raise on a concrete liquidus above the pure-iron curve, non-finite, or a binary A <= 0."""
    t_fe = np.asarray(IronMeltingCurve.t_melt_pure(p))
    t = np.asarray(t_liq)
    if not np.all(np.isfinite(t)) or np.any(t > t_fe * (1.0 + 1e-12)) or np.any(t <= 0.0):
        raise ValueError(f'alloy liquidus out of (0, T_Fe]: {t} against T_Fe {t_fe}')
    for a_set in a_sets:
        for el, a in a_set.items():
            if not np.all(np.asarray(a) > 0.0):
                raise ValueError(f'A of the Fe-{el} binary is not positive: {np.asarray(a)}')
