"""Liquidus of the core alloy: ideal-solution Fe-X binaries joined by a harmonic mean of their A.

Each Fe-X binary follows the ideal-solution liquidus of Mori et al. (2017, Eq. 3) and Mashino
et al. (2019): ``x_Fe = exp[A (1 - T_Fe / T)]``, with ``x_Fe`` the mole fraction of free Fe among
Fe and the Fe_k X_m units of the alloy, ``T_Fe`` the pure-iron curve of
:class:`aragog.core.melting.IronMeltingCurve`, and ``A`` fixed at each pressure by the binary
eutectic point (``w_e``, ``T_e``). The units are Fe3S for S and FeO for O; for C they are Fe3C
below 255 GPa and Fe7C3 above, joined by a C2 blend of the two liquidus temperatures over
230 to 280 GPa. With several elements, ``x_Fe = 1 - sum x_i`` and ``1/A = sum (x_i/X) / A_i``,
which keeps the dilute limit of each binary, ``dT ~ T_Fe sum x_i / A_i``. Si, H and N lower no
melting point: their atoms count with the Fe solvent, so an Fe-Si alloy melts like Fe.

The eutectic compositions and the Fe-O eutectic temperature are tables read from the papers'
figures (``aragog/core/eutectics/*.csv``, sources in each header); the Fe-S and Fe-C eutectic
temperatures are the Simon fits of Mori et al. (2017) and Mashino et al. (2019). Outside the
pressures of each binary's data, the eutectic temperature follows the pure-iron curve at a fixed
ratio (homologous) and the eutectic composition holds its end value. A composition past the
Fe-rich side of the eutectic (``sum w_i / w_e,i > 1``) is evaluated at its boundary.
"""

from __future__ import annotations

import functools
import importlib.resources
import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np

from aragog.core.melting import IronMeltingCurve

jax.config.update('jax_enable_x64', True)

DEPRESSING = ('S', 'C', 'O')
INERT = ('Si', 'H', 'N')
ELEMENTS = DEPRESSING + INERT
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
# Fe_k X_m unit of each binary: (k, m).
_UNITS = {'S': (3.0, 1.0), 'O': (1.0, 1.0)}
_UNITS_C = ((3.0, 1.0), (7.0, 3.0))
C_UNIT_SWITCH_PA = 255.0e9
C_UNIT_BLEND_HALF_WIDTH_PA = 25.0e9
# Simon fits of the eutectic temperature, T_ref [K], P_ref [GPa], a [GPa], c.
_SIMON = {'S': (1348.0, 21.0, 36.5, 2.07), 'C': (1513.0, 5.0, 8.4, 4.1)}


def _table(name: str) -> np.ndarray:
    """Two-column table of ``aragog/core/eutectics/<name>.csv``, pressure in GPa first."""
    path = importlib.resources.files('aragog.core').joinpath('eutectics', f'{name}.csv')
    rows = [
        ln for ln in path.read_text(encoding='utf-8').splitlines() if not ln.startswith('#')
    ]
    return np.array([[float(v) for v in ln.split(',')[:2]] for ln in rows[1:]])


@functools.cache
def _data() -> dict:
    """Eutectic tables and the pressure range [GPa] of each binary's data."""
    comp = {el: _table(f'fe_{el.lower()}_composition') for el in DEPRESSING}
    t_o = _table('fe_o_temperature')
    rng = {
        'S': (_SIMON['S'][1], comp['S'][-1, 0]),
        'C': (_SIMON['C'][1], comp['C'][-1, 0]),
        'O': (max(comp['O'][0, 0], t_o[0, 0]), t_o[-1, 0]),
    }
    return {'composition': comp, 't_o': t_o, 'range': rng}


def _eutectic_at(el: str, p_gpa):
    """Eutectic temperature [K] and composition (mass fraction) of the Fe-``el`` binary inside
    its data range, at ``p_gpa`` [GPa] already clipped to that range."""
    d = _data()
    table = d['composition'][el]
    w_e = jnp.interp(p_gpa, table[:, 0], table[:, 1])
    if el == 'O':
        t_e = jnp.interp(p_gpa, d['t_o'][:, 0], d['t_o'][:, 1])
    else:
        t_ref, p_ref, a, c = _SIMON[el]
        t_e = t_ref * ((p_gpa - p_ref) / a + 1.0) ** (1.0 / c)
    return t_e, w_e


def eutectic(el: str, pressure):
    """Eutectic temperature [K] and composition (mass fraction of ``el``) of the Fe-``el`` binary.

    Outside the binary's data range the temperature keeps its ratio to the pure-iron curve at the
    range end and the composition its end value.
    """
    lo, hi = _data()['range'][el]
    p_gpa = jnp.asarray(pressure) / 1e9
    clipped = jnp.clip(p_gpa, lo, hi)
    t_e, w_e = _eutectic_at(el, clipped)
    t_fe = IronMeltingCurve.t_melt_pure(p_gpa * 1e9)
    t_fe_end = IronMeltingCurve.t_melt_pure(clipped * 1e9)
    return t_e * t_fe / t_fe_end, w_e


def _units(w: dict, units: dict):
    """Solvent moles (free Fe plus the Si, H and N atoms) and the mole numbers of the Fe_k X_m
    units, per unit mass of alloy."""
    free = (1.0 - sum(w.values())) / _MOLAR['Fe'] + sum(w[el] / _MOLAR[el] for el in INERT)
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


def _liquidus(pressure, w: dict, c_unit):
    """Alloy liquidus [K] for one C unit choice; the composition is scaled back to the Fe-rich
    boundary where ``sum w_i / w_e,i`` exceeds one."""
    t_fe = IronMeltingCurve.t_melt_pure(pressure)
    eut = {el: eutectic(el, pressure) for el in DEPRESSING}
    excess = sum(w[el] / eut[el][1] for el in DEPRESSING)
    scale = jnp.where(excess > 1.0, 1.0 / jnp.maximum(excess, 1.0), 1.0)
    ws = {el: (w[el] * scale if el in DEPRESSING else w[el]) for el in ELEMENTS}
    units = {**_UNITS, 'C': c_unit}
    free, u = _units(ws, units)
    total = free + sum(u.values())
    x = {el: u[el] / total for el in DEPRESSING}
    big_x = sum(x.values())
    a_i = {el: _binary_a(el, t_fe, eut[el][0], eut[el][1], units[el]) for el in DEPRESSING}
    safe_x = jnp.where(big_x > 0.0, big_x, 1.0)
    inv_a = sum((x[el] / safe_x) / a_i[el] for el in DEPRESSING)
    t_liq = t_fe / (1.0 - jnp.log1p(-big_x) * inv_a)
    return jnp.where(big_x > 0.0, t_liq, t_fe), a_i


def _checked(composition) -> dict:
    """The composition as a dict of float mass fractions over :data:`ELEMENTS`.

    Raises
    ------
    ValueError
        On a key outside :data:`ELEMENTS`, a value that is not a finite real number (a bool or
        string included), a negative value, or a total of one or more.
    """
    if not isinstance(composition, dict):
        raise ValueError(f'composition must be a dict of mass fractions, got {composition!r}')
    unknown = sorted(set(composition) - set(ELEMENTS))
    if unknown:
        raise ValueError(f'unknown light elements {unknown}; known: {list(ELEMENTS)}')
    out = {}
    for el in ELEMENTS:
        v = composition.get(el, 0.0)
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
            raise ValueError(f'mass fraction of {el} must be a finite real number, got {v!r}')
        if v < 0.0:
            raise ValueError(f'mass fraction of {el} must not be negative, got {v}')
        out[el] = float(v)
    if sum(out.values()) >= 1.0:
        raise ValueError(
            f'light-element mass fractions sum to {sum(out.values())}, not below 1'
        )
    return out


class AlloyLiquidus:
    """Liquidus of an Fe alloy with S, C, O (melting depression) and Si, H, N (no depression).

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
        instance's (checked unless traced). Concrete results are checked against the pure-iron
        curve and each binary's A."""
        w = self.composition if composition is None else composition
        if not any(isinstance(v, jax.core.Tracer) for v in w.values()):
            w = _checked(w)
        p = jnp.asarray(pressure, dtype=float)
        lo = jnp.clip(
            0.5 * ((p - C_UNIT_SWITCH_PA) / C_UNIT_BLEND_HALF_WIDTH_PA + 1.0), 0.0, 1.0
        )
        blend = lo * lo * lo * (lo * (lo * 6.0 - 15.0) + 10.0)
        t3, a3 = _liquidus(p, w, _UNITS_C[0])
        t7, a7 = _liquidus(p, w, _UNITS_C[1])
        t_liq = (1.0 - blend) * t3 + blend * t7
        if not isinstance(t_liq, jax.core.Tracer):
            _assert_invariants(p, t_liq, a3, a7)
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
        w = self.composition
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
